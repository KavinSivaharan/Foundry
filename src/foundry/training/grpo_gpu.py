"""Host-monitoring and child CUDA-compute contracts for verifier-GRPO.

The deterministic child contract deliberately has no NVML dependency.  Host
``nvidia-smi`` evidence is observational and is collected only from the normal
parent environment.  Counted children prove their actual compute path through
the installed PyTorch CUDA runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundry.training.config import canonical_sha256
from foundry.training.grpo_environment import (
    transformers_determinism_source_evidence,
    validate_deterministic_process_environment,
)
from foundry.training.grpo_paths import (
    GrpoRuntimePaths,
    assert_artifact_path,
    load_runtime_paths,
    validate_foundry_import,
    validate_runtime_paths,
)

HOST_GPU_EVIDENCE_ID = "foundry-verifier-grpo-host-gpu-evidence-v1"
CHILD_CUDA_COMPUTE_EVIDENCE_ID = "foundry-verifier-grpo-child-cuda-compute-evidence-v1"
CHILD_CUDA_PROBE_ID = "foundry-verifier-grpo-child-cuda-probe-v1"
GPU_EVIDENCE_SCHEMA_VERSION = 1

EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3080"
EXPECTED_GPU_NAME_FRAGMENT = "RTX 3080"
EXPECTED_GPU_TOTAL_MEMORY_BYTES = 10_736_893_952
EXPECTED_GPU_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_GPU_MULTIPROCESSOR_COUNT = 68
EXPECTED_TORCH_VERSION = "2.5.1+cu121"
EXPECTED_TORCH_CUDA_RUNTIME = "12.1"
CHILD_CUDA_PROBE_SEED = 20_260_720
CHILD_CUDA_PROBE_REPEATS = 3
CHILD_CUDA_PROBE_SHAPE = (32, 32)
CHILD_CUDA_PROBE_CONFIGURATION = {
    "probe_id": CHILD_CUDA_PROBE_ID,
    "seed": CHILD_CUDA_PROBE_SEED,
    "repeats": CHILD_CUDA_PROBE_REPEATS,
    "shape": list(CHILD_CUDA_PROBE_SHAPE),
    "dtype": "float32",
    "device": "cuda:0",
    "operations": ["elementwise_scale_add", "matrix_multiply", "elementwise_add"],
    "isolated_cuda_generator_per_repeat": True,
}
CHILD_CUDA_PROBE_CONFIGURATION_SHA256 = canonical_sha256(CHILD_CUDA_PROBE_CONFIGURATION)
HOST_GPU_EVIDENCE_CONTRACT = {
    "schema_version": GPU_EVIDENCE_SCHEMA_VERSION,
    "evidence_id": HOST_GPU_EVIDENCE_ID,
    "collection_environment": "normal_parent_process",
    "provider": "nvidia-smi",
    "gate_authority": False,
    "raw_output_persisted": False,
    "fields": [
        "executable_path",
        "command",
        "exit_code",
        "path_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "parse_succeeded",
        "succeeded",
        "gpu_name",
        "driver_version",
        "gpu_uuid",
        "pci_bus_id",
    ],
}
HOST_GPU_EVIDENCE_CONTRACT_SHA256 = canonical_sha256(HOST_GPU_EVIDENCE_CONTRACT)
CHILD_CUDA_COMPUTE_EVIDENCE_CONTRACT = {
    "schema_version": GPU_EVIDENCE_SCHEMA_VERSION,
    "evidence_id": CHILD_CUDA_COMPUTE_EVIDENCE_ID,
    "collection_environment": "minimized_deterministic_child",
    "provider": "torch.cuda",
    "gate_authority": True,
    "nvml_permitted": False,
    "model_loading_permitted": False,
    "stable_fields": [
        "probe_configuration_sha256",
        "python_executable_sha256",
        "foundry_import_path",
        "torch_version",
        "torch_cuda_runtime",
        "cuda_available",
        "cuda_device_count",
        "cuda_current_device",
        "gpu_name",
        "gpu_compute_capability",
        "gpu_multiprocessor_count",
        "gpu_total_memory_bytes",
        "result_sha256",
        "repeat_result_sha256s",
        "tensor_devices",
        "allocation_succeeded",
        "arithmetic_succeeded",
        "matrix_multiplication_succeeded",
        "synchronization_succeeded",
        "no_cpu_fallback",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "environment_sha256",
        "environment_unchanged",
        "nvml_invoked",
        "pynvml_imported",
    ],
    "resource_fields": [
        "free_memory_bytes_before",
        "free_memory_bytes_after",
        "allocated_memory_bytes",
        "reserved_memory_bytes",
        "peak_allocated_memory_bytes",
        "peak_reserved_memory_bytes",
        "post_cleanup_allocated_memory_bytes",
        "post_cleanup_reserved_memory_bytes",
    ],
}
CHILD_CUDA_COMPUTE_EVIDENCE_CONTRACT_SHA256 = canonical_sha256(CHILD_CUDA_COMPUTE_EVIDENCE_CONTRACT)
_NVML_MODULE_NAMES = frozenset({"nvidia_ml_py", "pynvml"})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite GPU evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class HostGpuEvidence:
    """Monitoring-only GPU/driver evidence from the normal parent environment."""

    executable_path: str | None
    command: tuple[str, ...]
    exit_code: int
    path_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    parse_succeeded: bool
    gpu_name: str | None
    driver_version: str | None
    gpu_uuid: str | None
    pci_bus_id: str | None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and self.parse_succeeded

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": GPU_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": HOST_GPU_EVIDENCE_ID,
            "contract_sha256": HOST_GPU_EVIDENCE_CONTRACT_SHA256,
            "executable_path": self.executable_path,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "path_sha256": self.path_sha256,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "parse_succeeded": self.parse_succeeded,
            "succeeded": self.succeeded,
            "gpu_name": self.gpu_name,
            "driver_version": self.driver_version,
            "gpu_uuid": self.gpu_uuid,
            "pci_bus_id": self.pci_bus_id,
            "monitoring_only": True,
            "child_cuda_gate_authority": False,
            "raw_output_included": False,
        }

    @property
    def evidence_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def evidence(self) -> dict[str, object]:
        return {**self.payload(), "evidence_sha256": self.evidence_sha256}


def collect_host_gpu_evidence(
    base_environment: Mapping[str, str] | None = None,
) -> HostGpuEvidence:
    """Collect non-gating ``nvidia-smi`` evidence without minimizing the environment."""

    environment = os.environ if base_environment is None else base_environment
    path_value = environment.get("PATH", "")
    path_sha256 = hashlib.sha256(path_value.encode("utf-8")).hexdigest()
    executable = shutil.which("nvidia-smi", path=path_value)
    if executable is None:
        return HostGpuEvidence(
            executable_path=None,
            command=("nvidia-smi",),
            exit_code=127,
            path_sha256=path_sha256,
            stdout_sha256=hashlib.sha256(b"").hexdigest(),
            stderr_sha256=hashlib.sha256(b"").hexdigest(),
            parse_succeeded=False,
            gpu_name=None,
            driver_version=None,
            gpu_uuid=None,
            pci_bus_id=None,
        )
    command = (
        executable,
        "--query-gpu=name,driver_version,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    )
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    stdout = completed.stdout
    stderr = completed.stderr
    rows = list(csv.reader(stdout.splitlines(), skipinitialspace=True))
    parsed = completed.returncode == 0 and len(rows) == 1 and len(rows[0]) == 4
    fields: tuple[str | None, str | None, str | None, str | None]
    if parsed:
        fields = tuple(item.strip() or None for item in rows[0])  # type: ignore[assignment]
    else:
        fields = (None, None, None, None)
    return HostGpuEvidence(
        executable_path=str(Path(executable).resolve(strict=True)),
        command=command,
        exit_code=int(completed.returncode),
        path_sha256=path_sha256,
        stdout_sha256=hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        parse_succeeded=parsed,
        gpu_name=fields[0],
        driver_version=fields[1],
        gpu_uuid=fields[2],
        pci_bus_id=fields[3],
    )


@dataclass(frozen=True)
class _CudaProbeRun:
    result_sha256: str
    tensor_devices: tuple[str, ...]
    allocation_succeeded: bool
    arithmetic_succeeded: bool
    matrix_multiplication_succeeded: bool
    synchronization_succeeded: bool


@dataclass(frozen=True)
class ChildCudaComputeEvidence:
    """Direct PyTorch CUDA compute and resource evidence from one child process."""

    python_executable: str
    python_executable_sha256: str
    foundry_import_path: str
    torch_version: str
    torch_cuda_runtime: str
    cuda_available: bool
    cuda_device_count: int
    cuda_current_device: int
    gpu_name: str
    gpu_compute_capability: tuple[int, int]
    gpu_multiprocessor_count: int
    gpu_total_memory_bytes: int
    free_memory_bytes_before: int
    free_memory_bytes_after: int
    result_sha256: str
    repeat_result_sha256s: tuple[str, ...]
    tensor_devices: tuple[str, ...]
    allocated_memory_bytes: int
    reserved_memory_bytes: int
    peak_allocated_memory_bytes: int
    peak_reserved_memory_bytes: int
    post_cleanup_allocated_memory_bytes: int
    post_cleanup_reserved_memory_bytes: int
    allocation_succeeded: bool
    arithmetic_succeeded: bool
    matrix_multiplication_succeeded: bool
    synchronization_succeeded: bool
    no_cpu_fallback: bool
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    environment_sha256: str
    environment_unchanged: bool
    nvml_invoked: bool
    pynvml_imported: bool

    def stable_payload(self) -> dict[str, object]:
        """Return exact replay fields, excluding dynamic free/allocator measurements."""

        return {
            "schema_version": GPU_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": CHILD_CUDA_COMPUTE_EVIDENCE_ID,
            "contract_sha256": CHILD_CUDA_COMPUTE_EVIDENCE_CONTRACT_SHA256,
            "probe_configuration_sha256": CHILD_CUDA_PROBE_CONFIGURATION_SHA256,
            "python_executable_sha256": self.python_executable_sha256,
            "foundry_import_path": self.foundry_import_path,
            "torch_version": self.torch_version,
            "torch_cuda_runtime": self.torch_cuda_runtime,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_current_device": self.cuda_current_device,
            "gpu_name": self.gpu_name,
            "gpu_compute_capability": list(self.gpu_compute_capability),
            "gpu_multiprocessor_count": self.gpu_multiprocessor_count,
            "gpu_total_memory_bytes": self.gpu_total_memory_bytes,
            "result_sha256": self.result_sha256,
            "repeat_result_sha256s": list(self.repeat_result_sha256s),
            "tensor_devices": list(self.tensor_devices),
            "allocation_succeeded": self.allocation_succeeded,
            "arithmetic_succeeded": self.arithmetic_succeeded,
            "matrix_multiplication_succeeded": self.matrix_multiplication_succeeded,
            "synchronization_succeeded": self.synchronization_succeeded,
            "no_cpu_fallback": self.no_cpu_fallback,
            "deterministic_algorithms": self.deterministic_algorithms,
            "deterministic_warn_only": self.deterministic_warn_only,
            "environment_sha256": self.environment_sha256,
            "environment_unchanged": self.environment_unchanged,
            "nvml_invoked": self.nvml_invoked,
            "pynvml_imported": self.pynvml_imported,
        }

    @property
    def stable_sha256(self) -> str:
        return canonical_sha256(self.stable_payload())

    def stable_evidence(self) -> dict[str, object]:
        return {**self.stable_payload(), "stable_sha256": self.stable_sha256}

    def resource_payload(self) -> dict[str, object]:
        return {
            "free_memory_bytes_before": self.free_memory_bytes_before,
            "free_memory_bytes_after": self.free_memory_bytes_after,
            "allocated_memory_bytes": self.allocated_memory_bytes,
            "reserved_memory_bytes": self.reserved_memory_bytes,
            "peak_allocated_memory_bytes": self.peak_allocated_memory_bytes,
            "peak_reserved_memory_bytes": self.peak_reserved_memory_bytes,
            "post_cleanup_allocated_memory_bytes": self.post_cleanup_allocated_memory_bytes,
            "post_cleanup_reserved_memory_bytes": self.post_cleanup_reserved_memory_bytes,
        }

    def payload(self) -> dict[str, object]:
        return {
            **self.stable_payload(),
            "python_executable": self.python_executable,
            **self.resource_payload(),
        }

    @property
    def evidence_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def evidence(self) -> dict[str, object]:
        return {
            **self.payload(),
            "stable_sha256": self.stable_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


def _run_cuda_probe_once(torch: Any) -> _CudaProbeRun:
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(CHILD_CUDA_PROBE_SEED)
    left = torch.rand(
        CHILD_CUDA_PROBE_SHAPE,
        generator=generator,
        device="cuda:0",
        dtype=torch.float32,
    )
    right = torch.rand(
        CHILD_CUDA_PROBE_SHAPE,
        generator=generator,
        device="cuda:0",
        dtype=torch.float32,
    )
    scaled = left.mul(1.25).add(0.5)
    product = torch.matmul(scaled, right.transpose(0, 1))
    result = product.add(left)
    tensor_devices = tuple(str(item.device) for item in (left, right, scaled, product, result))
    if set(tensor_devices) != {"cuda:0"}:
        raise RuntimeError(f"child CUDA probe used an unexpected tensor device: {tensor_devices}")
    torch.cuda.synchronize(0)
    result_bytes = result.detach().cpu().contiguous().numpy().tobytes(order="C")
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    del result, product, scaled, right, left, generator
    return _CudaProbeRun(
        result_sha256=result_sha256,
        tensor_devices=tensor_devices,
        allocation_succeeded=True,
        arithmetic_succeeded=True,
        matrix_multiplication_succeeded=True,
        synchronization_succeeded=True,
    )


def collect_child_cuda_compute_evidence(
    torch: Any,
    runtime_paths: GrpoRuntimePaths,
    *,
    stage: str,
) -> ChildCudaComputeEvidence:
    """Run three exact CUDA computations without invoking NVML or changing global RNG state."""

    if not stage:
        raise ValueError("child CUDA evidence stage must be named")
    if os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") is not None:
        raise RuntimeError("NVML-based PyTorch CUDA checks are prohibited in the child")
    if any(name in sys.modules for name in _NVML_MODULE_NAMES):
        raise RuntimeError("an NVML Python module was imported before the child CUDA probe")
    before = validate_deterministic_process_environment(
        runtime_paths,
        f"{stage}_before_child_cuda_compute",
        torch_module=torch,
        require_strict=True,
    )
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable in the deterministic child")
    device_count = int(torch.cuda.device_count())
    current_device = int(torch.cuda.current_device())
    if device_count != 1 or current_device != 0:
        raise RuntimeError(
            f"child CUDA device selection differs: count={device_count}, current={current_device}"
        )
    gpu_name = str(torch.cuda.get_device_name(0))
    if gpu_name != EXPECTED_GPU_NAME or EXPECTED_GPU_NAME_FRAGMENT not in gpu_name:
        raise RuntimeError(f"child CUDA GPU identity differs: {gpu_name}")
    properties = torch.cuda.get_device_properties(0)
    capability_raw = torch.cuda.get_device_capability(0)
    capability = (int(capability_raw[0]), int(capability_raw[1]))
    total_memory = int(properties.total_memory)
    multiprocessors = int(properties.multi_processor_count)
    torch_version = str(getattr(torch, "__version__", ""))
    torch_cuda_runtime = str(getattr(torch.version, "cuda", ""))
    expected_identity = (
        EXPECTED_GPU_TOTAL_MEMORY_BYTES,
        EXPECTED_GPU_COMPUTE_CAPABILITY,
        EXPECTED_GPU_MULTIPROCESSOR_COUNT,
        EXPECTED_TORCH_VERSION,
        EXPECTED_TORCH_CUDA_RUNTIME,
    )
    actual_identity = (
        total_memory,
        capability,
        multiprocessors,
        torch_version,
        torch_cuda_runtime,
    )
    if actual_identity != expected_identity:
        raise RuntimeError(f"child CUDA hardware or runtime differs: {actual_identity}")
    free_before, mem_total_before = (int(item) for item in torch.cuda.mem_get_info(0))
    if mem_total_before != total_memory or not 0 < free_before <= total_memory:
        raise RuntimeError("child CUDA free/total memory evidence differs")
    torch.cuda.reset_peak_memory_stats(0)
    runs: list[_CudaProbeRun] = []
    for index in range(CHILD_CUDA_PROBE_REPEATS):
        try:
            runs.append(_run_cuda_probe_once(torch))
        except Exception as error:
            raise RuntimeError(f"child CUDA probe run {index + 1} failed: {error}") from error
    result_hashes = tuple(run.result_sha256 for run in runs)
    if len(set(result_hashes)) != 1:
        raise RuntimeError(f"child CUDA deterministic result hashes differ: {result_hashes}")
    tensor_devices = tuple(sorted({device for run in runs for device in run.tensor_devices}))
    if tensor_devices != ("cuda:0",):
        raise RuntimeError(f"child CUDA probe observed CPU fallback: {tensor_devices}")
    torch.cuda.synchronize(0)
    allocated = int(torch.cuda.memory_allocated(0))
    reserved = int(torch.cuda.memory_reserved(0))
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    free_after, mem_total_after = (int(item) for item in torch.cuda.mem_get_info(0))
    if mem_total_after != total_memory or not 0 < free_after <= total_memory:
        raise RuntimeError("child CUDA post-probe free/total memory evidence differs")
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    post_cleanup_allocated = int(torch.cuda.memory_allocated(0))
    post_cleanup_reserved = int(torch.cuda.memory_reserved(0))
    after = validate_deterministic_process_environment(
        runtime_paths,
        f"{stage}_after_child_cuda_compute",
        torch_module=torch,
        require_strict=True,
    )
    if before["environment_sha256"] != after["environment_sha256"]:
        raise RuntimeError("deterministic environment changed during child CUDA computation")
    if any(name in sys.modules for name in _NVML_MODULE_NAMES):
        raise RuntimeError("an NVML Python module was imported by the child CUDA probe")
    executable = Path(sys.executable).resolve(strict=True)
    import_evidence = validate_foundry_import(runtime_paths)
    return ChildCudaComputeEvidence(
        python_executable=str(executable),
        python_executable_sha256=_file_sha256(executable),
        foundry_import_path=import_evidence["foundry_import_path"],
        torch_version=torch_version,
        torch_cuda_runtime=torch_cuda_runtime,
        cuda_available=True,
        cuda_device_count=device_count,
        cuda_current_device=current_device,
        gpu_name=gpu_name,
        gpu_compute_capability=capability,
        gpu_multiprocessor_count=multiprocessors,
        gpu_total_memory_bytes=total_memory,
        free_memory_bytes_before=free_before,
        free_memory_bytes_after=free_after,
        result_sha256=result_hashes[0],
        repeat_result_sha256s=result_hashes,
        tensor_devices=tensor_devices,
        allocated_memory_bytes=allocated,
        reserved_memory_bytes=reserved,
        peak_allocated_memory_bytes=peak_allocated,
        peak_reserved_memory_bytes=peak_reserved,
        post_cleanup_allocated_memory_bytes=post_cleanup_allocated,
        post_cleanup_reserved_memory_bytes=post_cleanup_reserved,
        allocation_succeeded=all(run.allocation_succeeded for run in runs),
        arithmetic_succeeded=all(run.arithmetic_succeeded for run in runs),
        matrix_multiplication_succeeded=all(run.matrix_multiplication_succeeded for run in runs),
        synchronization_succeeded=all(run.synchronization_succeeded for run in runs),
        no_cpu_fallback=True,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        environment_sha256=str(after["environment_sha256"]),
        environment_unchanged=True,
        nvml_invoked=False,
        pynvml_imported=False,
    )


def current_cuda_memory_evidence(torch: Any) -> dict[str, object]:
    """Record current child memory solely through public PyTorch CUDA APIs."""

    torch.cuda.synchronize(0)
    free_memory, total_memory = (int(item) for item in torch.cuda.mem_get_info(0))
    if total_memory != EXPECTED_GPU_TOTAL_MEMORY_BYTES:
        raise RuntimeError("current CUDA total memory differs from the frozen RTX 3080")
    value: dict[str, object] = {
        "device_index": int(torch.cuda.current_device()),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "free_memory_bytes": free_memory,
        "total_memory_bytes": total_memory,
        "allocated_memory_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_memory_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(0)),
        "measurement_source": "torch.cuda",
        "nvml_used": False,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def adjudicate_gpu_evidence(
    host: HostGpuEvidence,
    child: ChildCudaComputeEvidence,
) -> dict[str, object]:
    """Make the child compute result authoritative and keep host evidence observational."""

    child_identity_available = bool(child.gpu_name) and child.cuda_device_count > 0
    host_identity_available = bool(host.gpu_name)
    if not child_identity_available and not host_identity_available:
        raise RuntimeError("GPU identity is completely unavailable")
    if not (
        child.cuda_available
        and child.no_cpu_fallback
        and child.allocation_succeeded
        and child.arithmetic_succeeded
        and child.matrix_multiplication_succeeded
        and child.synchronization_succeeded
        and not child.nvml_invoked
    ):
        raise RuntimeError("direct child CUDA compute evidence did not pass")
    value: dict[str, object] = {
        "child_cuda_gate_passed": True,
        "child_cuda_evidence_sha256": child.evidence_sha256,
        "host_gpu_evidence_sha256": host.evidence_sha256,
        "host_nvidia_smi_succeeded": host.succeeded,
        "host_gpu_identity_available": host_identity_available,
        "child_gpu_identity_available": child_identity_available,
        "host_evidence_monitoring_only": True,
        "parent_measurement_can_invalidate_child": False,
    }
    value["decision_sha256"] = canonical_sha256(value)
    return value


def _host_payload(output_path: Path) -> dict[str, object]:
    evidence = collect_host_gpu_evidence()
    value: dict[str, object] = {
        "schema_version": GPU_EVIDENCE_SCHEMA_VERSION,
        "artifact_id": HOST_GPU_EVIDENCE_ID,
        "host_gpu_evidence_contract_sha256": HOST_GPU_EVIDENCE_CONTRACT_SHA256,
        "output_path": str(output_path.resolve(strict=False)),
        "host_gpu_evidence": evidence.evidence(),
        "host_monitoring_only": True,
        "child_cuda_gate_authority": False,
    }
    value["summary_sha256"] = canonical_sha256(value)
    return value


def _child_payload(runtime_paths: GrpoRuntimePaths, output_path: Path) -> dict[str, object]:
    before_import = validate_deterministic_process_environment(
        runtime_paths, "child_cuda_probe_before_torch_import"
    )
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    after_import = validate_deterministic_process_environment(
        runtime_paths,
        "child_cuda_probe_after_torch_import",
        torch_module=torch,
    )
    source_evidence = transformers_determinism_source_evidence(transformers)
    helper = getattr(getattr(transformers, "trainer_utils", None), "enable_full_determinism", None)
    if not callable(helper):
        raise RuntimeError("Transformers full-determinism helper is unavailable")
    helper(CHILD_CUDA_PROBE_SEED, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    after_determinism = validate_deterministic_process_environment(
        runtime_paths,
        "child_cuda_probe_after_full_determinism",
        torch_module=torch,
        require_strict=True,
    )
    evidence = collect_child_cuda_compute_evidence(
        torch,
        runtime_paths,
        stage="standalone_child_cuda_probe",
    )
    final_environment = validate_deterministic_process_environment(
        runtime_paths,
        "child_cuda_probe_publication",
        torch_module=torch,
        require_strict=True,
    )
    value: dict[str, object] = {
        "schema_version": GPU_EVIDENCE_SCHEMA_VERSION,
        "artifact_id": CHILD_CUDA_PROBE_ID,
        "child_cuda_compute_evidence_contract_sha256": (
            CHILD_CUDA_COMPUTE_EVIDENCE_CONTRACT_SHA256
        ),
        "output_path": str(output_path.resolve(strict=False)),
        "probe_configuration": CHILD_CUDA_PROBE_CONFIGURATION,
        "probe_configuration_sha256": CHILD_CUDA_PROBE_CONFIGURATION_SHA256,
        "transformers_determinism_source": source_evidence,
        "deterministic_environment_stages": [
            before_import,
            after_import,
            after_determinism,
            final_environment,
        ],
        "child_cuda_compute_evidence": evidence.evidence(),
        "child_cuda_gate_passed": True,
        "model_loaded": False,
        "nvml_invoked": False,
        "prompts_completions_or_answers_in_evidence": False,
    }
    value["summary_sha256"] = canonical_sha256(value)
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    host = subparsers.add_parser("host-evidence")
    host.add_argument("--output", type=Path, required=True)
    child = subparsers.add_parser("child-probe")
    child.add_argument("--runtime-paths", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.output.resolve(strict=False)
    if args.command == "host-evidence":
        if not output.is_absolute():
            raise ValueError("host GPU evidence path must be absolute")
        value = _host_payload(output)
    else:
        runtime_paths = load_runtime_paths(args.runtime_paths)
        assert_artifact_path(runtime_paths, output, "child CUDA probe evidence")
        value = _child_payload(runtime_paths, output)
        validate_runtime_paths(runtime_paths)
    _write_json_new(output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
