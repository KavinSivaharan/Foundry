from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from foundry.training import grpo_gpu as gpu
from foundry.training.config import canonical_sha256
from foundry.training.grpo_config import load_grpo_config
from foundry.training.grpo_reward import (
    reward_configuration_sha256,
    reward_implementation_sha256,
)


class _Cuda:
    def __init__(self) -> None:
        self.available = True
        self.count = 1
        self.current = 0
        self.name = gpu.EXPECTED_GPU_NAME
        self.total = gpu.EXPECTED_GPU_TOTAL_MEMORY_BYTES
        self.free = self.total - 1024
        self.capability = gpu.EXPECTED_GPU_COMPUTE_CAPABILITY
        self.multiprocessors = gpu.EXPECTED_GPU_MULTIPROCESSOR_COUNT
        self.events: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def current_device(self) -> int:
        return self.current

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self.name

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(
            total_memory=self.total,
            multi_processor_count=self.multiprocessors,
        )

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return self.capability

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        return self.free, self.total

    def reset_peak_memory_stats(self, index: int) -> None:
        assert index == 0
        self.events.append("reset_peak")

    def synchronize(self, index: int) -> None:
        assert index == 0
        self.events.append("synchronize")

    def empty_cache(self) -> None:
        self.events.append("empty_cache")

    def memory_allocated(self, index: int) -> int:
        assert index == 0
        return 11

    def memory_reserved(self, index: int) -> int:
        assert index == 0
        return 22

    def max_memory_allocated(self, index: int) -> int:
        assert index == 0
        return 33

    def max_memory_reserved(self, index: int) -> int:
        assert index == 0
        return 44


class _Torch:
    def __init__(self) -> None:
        self.cuda = _Cuda()
        self.version = SimpleNamespace(cuda=gpu.EXPECTED_TORCH_CUDA_RUNTIME)
        self.__version__ = gpu.EXPECTED_TORCH_VERSION

    @staticmethod
    def are_deterministic_algorithms_enabled() -> bool:
        return True

    @staticmethod
    def is_deterministic_algorithms_warn_only_enabled() -> bool:
        return False


def _runtime_paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        python_executable=Path(sys.executable),
        source_root=tmp_path / "Foundry-grpo-frozen-v4",
        artifact_root=tmp_path / "runtime",
        model_cache_root=tmp_path / "cache",
    )


def _run(result: str = "a" * 64, devices: tuple[str, ...] = ("cuda:0",)) -> object:
    return gpu._CudaProbeRun(
        result_sha256=result,
        tensor_devices=devices,
        allocation_succeeded=True,
        arithmetic_succeeded=True,
        matrix_multiplication_succeeded=True,
        synchronization_succeeded=True,
    )


def _collect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    torch: _Torch | None = None,
    probe_runner: Callable[[object], object] | None = None,
) -> gpu.ChildCudaComputeEvidence:
    monkeypatch.delenv("PYTORCH_NVML_BASED_CUDA_CHECK", raising=False)
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    monkeypatch.delitem(sys.modules, "nvidia_ml_py", raising=False)
    monkeypatch.setattr(
        gpu,
        "validate_deterministic_process_environment",
        lambda *args, **kwargs: {"environment_sha256": "e" * 64},
    )
    import_path = tmp_path / "Foundry-grpo-frozen-v4" / "src" / "foundry" / "__init__.py"
    monkeypatch.setattr(
        gpu,
        "validate_foundry_import",
        lambda paths: {"foundry_import_path": str(import_path)},
    )
    monkeypatch.setattr(
        gpu,
        "_run_cuda_probe_once",
        probe_runner or (lambda module: _run()),
    )
    return gpu.collect_child_cuda_compute_evidence(
        torch or _Torch(),  # type: ignore[arg-type]
        _runtime_paths(tmp_path),  # type: ignore[arg-type]
        stage="fixture",
    )


def test_parent_nvidia_smi_evidence_is_monitoring_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "nvidia-smi.exe"
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(gpu.shutil, "which", lambda name, path: str(executable))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3080, 610.47, GPU-fixture, 00000000:01:00.0\n",
            stderr="",
        )

    monkeypatch.setattr(gpu.subprocess, "run", run)
    evidence = gpu.collect_host_gpu_evidence({"PATH": str(tmp_path)})
    assert evidence.succeeded is True
    assert evidence.gpu_name == gpu.EXPECTED_GPU_NAME
    assert evidence.driver_version == "610.47"
    assert evidence.payload()["child_cuda_gate_authority"] is False
    assert "env" not in calls[0][1]


def test_failing_parent_monitor_does_not_override_successful_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    child = _collect(monkeypatch, tmp_path)
    host = gpu.HostGpuEvidence(
        executable_path=None,
        command=("nvidia-smi",),
        exit_code=255,
        path_sha256="a" * 64,
        stdout_sha256="b" * 64,
        stderr_sha256="c" * 64,
        parse_succeeded=False,
        gpu_name=None,
        driver_version=None,
        gpu_uuid=None,
        pci_bus_id=None,
    )
    decision = gpu.adjudicate_gpu_evidence(host, child)
    assert decision["child_cuda_gate_passed"] is True
    assert decision["host_nvidia_smi_succeeded"] is False
    assert decision["parent_measurement_can_invalidate_child"] is False
    with pytest.raises(RuntimeError, match="completely unavailable"):
        gpu.adjudicate_gpu_evidence(host, replace(child, gpu_name="", cuda_device_count=0))


def test_child_cuda_validation_never_invokes_nvml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("child invoked nvidia-smi"),
    )
    evidence = _collect(monkeypatch, tmp_path)
    assert evidence.nvml_invoked is False
    assert evidence.pynvml_imported is False


def test_child_cuda_unavailable_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch = _Torch()
    torch.cuda.available = False
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _collect(monkeypatch, tmp_path, torch)


def test_child_wrong_gpu_identity_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch = _Torch()
    torch.cuda.name = "Different GPU"
    with pytest.raises(RuntimeError, match="GPU identity differs"):
        _collect(monkeypatch, tmp_path, torch)


def test_child_cpu_tensor_placement_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CPU fallback"):
        _collect(monkeypatch, tmp_path, probe_runner=lambda module: _run(devices=("cpu",)))


def test_child_cuda_allocation_failure_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(module: object) -> object:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _collect(monkeypatch, tmp_path, probe_runner=fail)


def test_child_cuda_synchronization_failure_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = _Torch()

    def fail(index: int) -> None:
        raise RuntimeError("synchronization failed")

    torch.cuda.synchronize = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="synchronization failed"):
        _collect(monkeypatch, tmp_path, torch)


def test_child_nonidentical_result_hashes_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = iter((_run("a" * 64), _run("b" * 64), _run("a" * 64)))
    with pytest.raises(RuntimeError, match="result hashes differ"):
        _collect(monkeypatch, tmp_path, probe_runner=lambda module: next(values))


def test_child_records_pytorch_memory_and_import_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = _collect(monkeypatch, tmp_path)
    assert evidence.free_memory_bytes_before == gpu.EXPECTED_GPU_TOTAL_MEMORY_BYTES - 1024
    assert evidence.gpu_total_memory_bytes == gpu.EXPECTED_GPU_TOTAL_MEMORY_BYTES
    assert evidence.allocated_memory_bytes == 11
    assert evidence.peak_reserved_memory_bytes == 44
    assert "Foundry-grpo-frozen-v4" in evidence.foundry_import_path
    assert evidence.repeat_result_sha256s == ("a" * 64,) * 3


def test_child_environment_mutation_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        gpu,
        "validate_deterministic_process_environment",
        lambda *args, **kwargs: {"environment_sha256": next(values)},
    )
    monkeypatch.setattr(
        gpu,
        "validate_foundry_import",
        lambda paths: {"foundry_import_path": str(tmp_path / "immutable" / "foundry.py")},
    )
    monkeypatch.setattr(gpu, "_run_cuda_probe_once", lambda module: _run())
    with pytest.raises(RuntimeError, match="environment changed"):
        gpu.collect_child_cuda_compute_evidence(
            _Torch(),  # type: ignore[arg-type]
            _runtime_paths(tmp_path),  # type: ignore[arg-type]
            stage="fixture",
        )


def test_child_nvml_based_cuda_check_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    with pytest.raises(RuntimeError, match="NVML-based"):
        gpu.collect_child_cuda_compute_evidence(
            _Torch(),  # type: ignore[arg-type]
            _runtime_paths(tmp_path),  # type: ignore[arg-type]
            stage="fixture",
        )


def test_current_memory_evidence_uses_torch_cuda_only() -> None:
    value = gpu.current_cuda_memory_evidence(_Torch())
    assert value["measurement_source"] == "torch.cuda"
    assert value["nvml_used"] is False
    assert value["total_memory_bytes"] == gpu.EXPECTED_GPU_TOTAL_MEMORY_BYTES


def test_probe_and_scientific_hashes_remain_frozen() -> None:
    repository = Path(__file__).resolve().parents[3]
    config = load_grpo_config(repository / "configs" / "training" / "verifier_grpo_v1.json")
    assert gpu.CHILD_CUDA_PROBE_CONFIGURATION_SHA256 == canonical_sha256(
        gpu.CHILD_CUDA_PROBE_CONFIGURATION
    )
    assert gpu.HOST_GPU_EVIDENCE_CONTRACT_SHA256 == canonical_sha256(gpu.HOST_GPU_EVIDENCE_CONTRACT)
    assert gpu.CHILD_CUDA_COMPUTE_EVIDENCE_CONTRACT_SHA256 == canonical_sha256(
        gpu.CHILD_CUDA_COMPUTE_EVIDENCE_CONTRACT
    )
    assert config.config_sha256 == (
        "01515d186f2485662ea20ef0b444902bdf368a2b4a8cde335f34bfe1b9222bda"
    )
    assert reward_implementation_sha256() == (
        "089650105e29ead3c4ad62f1e0e41263e6c2af5fb8a12cb2851644aca3599616"
    )
    assert reward_configuration_sha256() == (
        "4a47359fa3129b1bfd79dd158ecb609177e9b1642a95368c106e016a1554a965"
    )


def test_host_evidence_hashes_raw_output_without_storing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "nvidia-smi.exe"
    executable.write_bytes(b"fixture")
    stdout = "NVIDIA GeForce RTX 3080, 610.47, GPU-id, bus-id\n"
    monkeypatch.setattr(gpu.shutil, "which", lambda name, path: str(executable))
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )
    evidence = gpu.collect_host_gpu_evidence({"PATH": str(tmp_path)})
    assert evidence.stdout_sha256 == hashlib.sha256(stdout.encode("utf-8")).hexdigest()
    assert stdout not in json_text(evidence.evidence())


def json_text(value: object) -> str:
    return str(value)
