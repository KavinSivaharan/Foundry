from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from foundry.training import grpo_environment as process_environment
from foundry.training import grpo_gpu as gpu
from foundry.training import grpo_replay_runtime as runtime
from foundry.training.config import canonical_sha256


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _runtime_paths(tmp_path: Path, *, python_executable: Path | None = None) -> SimpleNamespace:
    contract = _hash("runtime-path-contract")
    environment = _hash("process-environment")
    command_template = _hash("process-command-template")
    evidence = {
        "runtime_path_contract_sha256": contract,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "source_tracked_manifest_sha256": _hash("source-manifest"),
        "python_executable_sha256": _hash("python"),
        "model_artifact_manifest_sha256": _hash("model"),
        "process_environment_sha256": environment,
        "process_command_template_sha256": command_template,
    }
    return SimpleNamespace(
        source_root=tmp_path / "source",
        primary_repository_root=tmp_path / "primary",
        python_executable=(python_executable or Path(runtime.sys.executable)),
        artifact_root=tmp_path,
        model_cache_root=tmp_path / "cache",
        model_snapshot_root=tmp_path / "model",
        contract_sha256=contract,
        process_environment_sha256=environment,
        process_command_template_sha256=command_template,
        evidence=lambda: evidence,
    )


class _FakeCuda:
    def __init__(self) -> None:
        self.manual_seeds: list[int] = []

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seeds.append(seed)

    @staticmethod
    def is_initialized() -> bool:
        return False


class _FakeTorch:
    def __init__(self, *, enable_succeeds: bool = True) -> None:
        self.enabled = False
        self.warn_only = True
        self.enable_succeeds = enable_succeeds
        self.manual_seeds: list[int] = []
        self.cuda = _FakeCuda()
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(benchmark=True, deterministic=False),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        if self.enable_succeeds:
            self.enabled = enabled
            self.warn_only = warn_only

    def manual_seed(self, seed: int) -> None:
        self.manual_seeds.append(seed)


class _SeedRecorder:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)


def _fake_transformers_with_full_determinism(
    fake_torch: _FakeTorch,
    numpy_random: _SeedRecorder,
    calls: list[tuple[int, bool]],
    *,
    transition_cublas: bool = True,
    cublas_value: str | None = None,
) -> SimpleNamespace:
    def enable_full_determinism(seed: int, warn_only: bool = False) -> None:
        calls.append((seed, warn_only))
        random.seed(seed)
        numpy_random.seed(seed)
        fake_torch.manual_seed(seed)
        fake_torch.cuda.manual_seed_all(seed)
        runtime.os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        if transition_cublas:
            runtime.os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG
                if cublas_value is None
                else cublas_value
            )
        runtime.os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
        runtime.os.environ["HCCL_DETERMINISTIC"] = "1"
        runtime.os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        fake_torch.use_deterministic_algorithms(True, warn_only=warn_only)
        fake_torch.backends.cudnn.deterministic = True
        fake_torch.backends.cudnn.benchmark = False

    return SimpleNamespace(
        trainer_utils=SimpleNamespace(enable_full_determinism=enable_full_determinism)
    )


def test_strict_determinism_requires_enabled_non_warning_mode() -> None:
    fake = _FakeTorch()
    assert runtime._strict_determinism(fake) is False
    fake.enabled = True
    assert runtime._strict_determinism(fake) is False
    fake.warn_only = False
    assert runtime._strict_determinism(fake) is True


def test_filtered_environment_hash_excludes_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    environment = {
        "PATH": "stable-path",
        "AUTHORIZATION": "secret-one",
        "service_API_TOKEN": "secret-two",
        "ARBITRARY_PARENT_FIELD": "alpha",
    }
    monkeypatch.setattr(runtime.os, "environ", environment)
    first_hash, retained, excluded = runtime._filtered_environment_sha256(  # type: ignore[arg-type]
        runtime_paths
    )
    environment["AUTHORIZATION"] = "changed-secret"
    environment["service_API_TOKEN"] = "another-secret"
    second_hash, second_retained, second_excluded = runtime._filtered_environment_sha256(
        runtime_paths
    )  # type: ignore[arg-type]
    environment["ARBITRARY_PARENT_FIELD"] = "beta"
    arbitrary_hash, _, _ = runtime._filtered_environment_sha256(  # type: ignore[arg-type]
        runtime_paths
    )
    environment["PATH"] = "changed-allowed-path"
    changed_hash, _, _ = runtime._filtered_environment_sha256(  # type: ignore[arg-type]
        runtime_paths
    )

    assert first_hash == second_hash
    assert first_hash == arbitrary_hash
    assert first_hash != changed_hash
    assert retained == second_retained
    assert (excluded, second_excluded) == (2, 2)


def test_external_process_evidence_proves_hash_seed_and_exact_cublas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    environment = process_environment.deterministic_environment_values(runtime_paths)
    monkeypatch.setattr(runtime.os, "environ", environment)
    monkeypatch.setattr(
        process_environment,
        "_PROCESS_START_CORE_ENVIRONMENT",
        tuple(
            sorted(
                (key, environment[key])
                for key in process_environment.FROZEN_CORE_PROCESS_ENVIRONMENT
            )
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_PROCESS_START_CUBLAS_WORKSPACE_CONFIG",
        runtime.FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG,
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"{hash(runtime._PYTHON_HASH_PROBE)}\n")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    first = runtime._external_process_evidence(
        runtime.FROZEN_REPLAY_SEED,
        runtime_paths,  # type: ignore[arg-type]
        expected_entry_cublas_workspace_config=(
            runtime.FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
        ),
    )
    assert first["python_hash_seed"] == "20260720"
    assert first["process_start_cublas_workspace_config"] == ":16:8"
    assert first["active_cublas_workspace_config"] == ":16:8"
    assert first["environment_transition_occurred"] is False
    assert len(str(first["python_hash_probe_sha256"])) == 64
    assert calls[0][0][:3] == [runtime.sys.executable, "-S", "-c"]
    assert calls[0][1]["env"] == environment

    subsequent = runtime._external_process_evidence(
        runtime.FROZEN_REPLAY_SEED,
        runtime_paths,  # type: ignore[arg-type]
        expected_entry_cublas_workspace_config=(runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG),
    )
    assert subsequent == first

    environment["CUBLAS_WORKSPACE_CONFIG"] = ":32:8"
    with pytest.raises(RuntimeError, match="deterministic process environment differs"):
        runtime._external_process_evidence(
            runtime.FROZEN_REPLAY_SEED,
            runtime_paths,  # type: ignore[arg-type]
            expected_entry_cublas_workspace_config=(runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG),
        )
    with pytest.raises(ValueError, match="not frozen"):
        environment["CUBLAS_WORKSPACE_CONFIG"] = runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG
        runtime._external_process_evidence(
            runtime.FROZEN_REPLAY_SEED,
            runtime_paths,  # type: ignore[arg-type]
            expected_entry_cublas_workspace_config=":32:8",
        )
    environment["PYTHONHASHSEED"] = "7"
    with pytest.raises(RuntimeError, match="deterministic process environment differs"):
        runtime._external_process_evidence(
            runtime.FROZEN_REPLAY_SEED,
            runtime_paths,  # type: ignore[arg-type]
            expected_entry_cublas_workspace_config=(runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG),
        )
    environment["PYTHONHASHSEED"] = "20260720"
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="different\n"),
    )
    with pytest.raises(RuntimeError, match="running interpreter hash seed"):
        runtime._external_process_evidence(
            runtime.FROZEN_REPLAY_SEED,
            runtime_paths,  # type: ignore[arg-type]
            expected_entry_cublas_workspace_config=(runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG),
        )

    monkeypatch.setattr(runtime, "_PROCESS_START_CUBLAS_WORKSPACE_CONFIG", ":32:8")
    with pytest.raises(RuntimeError, match="module must be imported"):
        runtime._external_process_evidence(
            runtime.FROZEN_REPLAY_SEED,
            runtime_paths,  # type: ignore[arg-type]
            expected_entry_cublas_workspace_config=(runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG),
        )


def test_replay_source_contains_no_child_nvml_probe() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "nvidia-smi" not in source
    assert "pynvml" not in source


def test_runtime_environment_evidence_binds_interpreter_packages_and_os(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / ".venv-training" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python-fixture")
    monkeypatch.setattr(
        runtime,
        "_FROZEN_PYTHON_EXECUTABLE_SHA256",
        hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "_FROZEN_PYTHON_EXECUTABLE_SIZE_BYTES",
        executable.stat().st_size,
    )
    monkeypatch.setattr(runtime.sys, "executable", str(executable))
    monkeypatch.setattr(runtime.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.10")
    monkeypatch.setattr(runtime.platform, "architecture", lambda: ("64bit", "WindowsPE"))
    monkeypatch.setattr(runtime.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(runtime.platform, "release", lambda: "11")
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.platform, "version", lambda: "10.0.26200")
    versions = dict(runtime._FROZEN_SOFTWARE_VERSIONS)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__=versions["tokenizers"]),
    )

    runtime_paths = _runtime_paths(tmp_path, python_executable=executable)
    value = runtime._runtime_environment_evidence(  # type: ignore[arg-type]
        runtime_paths,
        SimpleNamespace(__version__="2.5.1"),
    )
    assert value["sys_executable"] == str(executable.resolve())
    assert (
        value["sys_executable_file_sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    )
    assert value["python"] == runtime._FROZEN_PYTHON
    assert value["operating_system"] == runtime._FROZEN_OS
    assert value["software_versions"] == runtime._FROZEN_SOFTWARE_VERSIONS

    versions["tokenizers"] = "0.0.0"
    with pytest.raises(RuntimeError, match="software versions differ"):
        runtime._runtime_environment_evidence(
            runtime_paths,  # type: ignore[arg-type]
            SimpleNamespace(__version__="2.5.1"),
        )


def test_frozen_cuda_validation_and_synchronized_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Cuda:
        def get_device_properties(self, index: int) -> SimpleNamespace:
            assert index == 0
            return SimpleNamespace(multi_processor_count=68)

        def get_device_capability(self, index: int) -> tuple[int, int]:
            assert index == 0
            return (8, 6)

        def device_count(self) -> int:
            return 1

        def current_device(self) -> int:
            return 0

        def synchronize(self, index: int) -> None:
            assert index == 0
            calls.append("synchronize")

        def empty_cache(self) -> None:
            calls.append("empty_cache")

        def ipc_collect(self) -> None:
            calls.append("ipc_collect")

        def reset_peak_memory_stats(self, index: int) -> None:
            assert index == 0
            calls.append("reset_peak")

        def memory_allocated(self, index: int) -> int:
            assert index == 0
            return 11

        def memory_reserved(self, index: int) -> int:
            assert index == 0
            return 22

    fake_torch = SimpleNamespace(cuda=Cuda())
    sentinel = cast(gpu.ChildCudaComputeEvidence, object())
    runtime_paths = _runtime_paths(Path.cwd())
    calls_to_validate: list[tuple[object, object, str]] = []

    def validate(torch: object, paths: object, *, stage: str) -> gpu.ChildCudaComputeEvidence:
        calls_to_validate.append((torch, paths, stage))
        return sentinel

    monkeypatch.setattr(runtime, "_validate_cuda", validate)
    monkeypatch.setattr(runtime.gc, "collect", lambda: calls.append("gc") or 0)

    assert (
        runtime._validate_frozen_cuda(  # type: ignore[arg-type]
            fake_torch, runtime_paths
        )
        is sentinel
    )
    assert calls_to_validate == [(fake_torch, runtime_paths, "generation_replay")]
    runtime._prepare_cuda_replay(fake_torch)
    cleanup = runtime._cleanup_cuda_replay(fake_torch)
    assert calls == [
        "synchronize",
        "gc",
        "empty_cache",
        "ipc_collect",
        "synchronize",
        "reset_peak",
        "synchronize",
        "gc",
        "empty_cache",
        "ipc_collect",
        "synchronize",
    ]
    assert cleanup == {
        "post_cleanup_allocated_vram_bytes": 11,
        "post_cleanup_reserved_vram_bytes": 22,
    }


def test_single_replay_cleanup_runs_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cleanup_calls: list[object] = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(runtime.sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        runtime,
        "_cleanup_cuda_replay",
        lambda torch: cleanup_calls.append(torch)
        or {
            "post_cleanup_allocated_vram_bytes": 0,
            "post_cleanup_reserved_vram_bytes": 0,
        },
    )
    monkeypatch.setattr(
        runtime,
        "_single_generation_replay_impl",
        lambda **kwargs: ({"packet_sha256": _hash("packet")}, {"runtime_seconds": 1.0}),
    )
    monkeypatch.setattr(runtime, "validate_runtime_paths", lambda paths: {})
    runtime_paths = _runtime_paths(tmp_path)
    _, resource = runtime._single_generation_replay(
        runtime_paths=runtime_paths,  # type: ignore[arg-type]
        config_path=tmp_path / "config.json",
        packet_path=tmp_path / "packet.json",
        manifest_path=tmp_path / "manifest.json",
        arm="generic_control",
        trainer_output_dir=tmp_path / "trainer",
        expected_entry_cublas_workspace_config=(
            runtime.FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
        ),
    )
    assert resource["post_cleanup_allocated_vram_bytes"] == 0
    assert cleanup_calls == [fake_torch]

    def fail(**kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(runtime, "_single_generation_replay_impl", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        runtime._single_generation_replay(
            runtime_paths=runtime_paths,  # type: ignore[arg-type]
            config_path=tmp_path / "config.json",
            packet_path=tmp_path / "packet.json",
            manifest_path=tmp_path / "manifest.json",
            arm="generic_control",
            trainer_output_dir=tmp_path / "trainer",
            expected_entry_cublas_workspace_config=(
                runtime.FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
            ),
        )
    assert cleanup_calls == [fake_torch, fake_torch]


def test_frozen_schedule_and_reward_contracts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = SimpleNamespace(
        arm=runtime.FROZEN_REPLAY_ARM,
        packet_sha256=runtime.FROZEN_GENERIC_PACKET_SHA256,
        manifest_sha256=runtime.FROZEN_GENERIC_MANIFEST_SHA256,
    )
    runtime._assert_frozen_schedule_binding(
        schedule,
        arm="generic_control",
        group_ids=runtime.FROZEN_REPLAY_GROUP_IDS,
    )
    with pytest.raises(ValueError, match="group IDs differ"):
        runtime._assert_frozen_schedule_binding(
            schedule,
            arm="generic_control",
            group_ids=(*runtime.FROZEN_REPLAY_GROUP_IDS[:2], "different"),
        )

    monkeypatch.setattr(
        runtime,
        "reward_configuration_sha256",
        lambda: runtime.FROZEN_REWARD_CONFIGURATION_SHA256,
    )
    monkeypatch.setattr(
        runtime,
        "reward_implementation_sha256",
        lambda: runtime.FROZEN_REWARD_IMPLEMENTATION_SHA256,
    )
    assert runtime._assert_frozen_reward_contract() == {
        "reward_configuration_sha256": runtime.FROZEN_REWARD_CONFIGURATION_SHA256,
        "reward_implementation_sha256": runtime.FROZEN_REWARD_IMPLEMENTATION_SHA256,
    }
    monkeypatch.setattr(runtime, "reward_implementation_sha256", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="reward contract differs"):
        runtime._assert_frozen_reward_contract()


def test_reward_variance_decisions_are_explicit_for_every_frozen_group() -> None:
    groups = [SimpleNamespace(group_id=group_id) for group_id in runtime.FROZEN_REPLAY_GROUP_IDS]
    records: list[SimpleNamespace] = []
    for group_index, group in enumerate(groups):
        for completion in range(4):
            total = 1.0 if group_index != 1 else float(completion)
            records.append(
                SimpleNamespace(
                    group_id=group.group_id,
                    reward=SimpleNamespace(total=total),
                )
            )
    value = runtime._reward_variance_decisions(records, groups)
    assert value["optimizer_updates"] == 0
    assert value["zero_variance_groups"] == 2
    decisions = value["decisions"]
    assert isinstance(decisions, list)
    assert [item["group_id"] for item in decisions] == list(runtime.FROZEN_REPLAY_GROUP_IDS)
    assert [item["zero_variance"] for item in decisions] == [True, False, True]
    with pytest.raises(RuntimeError, match="exactly four"):
        runtime._reward_variance_decisions(records[:-1], groups)


def test_seed_everything_configures_every_rng_and_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    numpy_random = _SeedRecorder()
    calls: list[tuple[int, bool]] = []
    transformers = _fake_transformers_with_full_determinism(fake_torch, numpy_random, calls)
    numpy_module = SimpleNamespace(random=numpy_random)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: numpy_module)
    source_evidence = {
        "function_source_sha256": runtime.FROZEN_ENABLE_FULL_DETERMINISM_SHA256,
        "source_file_sha256": _hash("trainer-utils"),
    }
    monkeypatch.setattr(
        runtime, "transformers_determinism_source_evidence", lambda value: source_evidence
    )
    environment = dict(process_environment.FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT)
    monkeypatch.setattr(runtime.os, "environ", environment)
    python_state = random.getstate()
    try:
        result = runtime._seed_everything(
            {"torch": fake_torch, "transformers": transformers}, 5172026
        )
        evidence = runtime._full_determinism_evidence(
            {"torch": fake_torch, "transformers": transformers}, 5172026
        )
    finally:
        random.setstate(python_state)

    assert result is numpy_module
    assert calls == [(5172026, False)]
    assert numpy_random.seeds == [5172026]
    assert fake_torch.manual_seeds == [5172026]
    assert fake_torch.cuda.manual_seeds == [5172026]
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert runtime._strict_determinism(fake_torch)
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert evidence["helper_source_sha256"] == (runtime.FROZEN_ENABLE_FULL_DETERMINISM_SHA256)
    assert evidence["process_start_cublas_workspace_config"] == ":16:8"
    assert evidence["active_cublas_workspace_config"] == ":16:8"
    assert evidence["effective_environment_changed"] is False
    assert evidence["python_random_seeded"] is True
    evidence_payload = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    assert evidence["evidence_sha256"] == canonical_sha256(evidence_payload)


def test_seed_everything_fails_if_strict_mode_cannot_be_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(enable_succeeds=False)
    numpy_random = _SeedRecorder()
    numpy_module = SimpleNamespace(random=numpy_random)
    calls: list[tuple[int, bool]] = []
    transformers = _fake_transformers_with_full_determinism(fake_torch, numpy_random, calls)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: numpy_module)
    monkeypatch.setattr(
        runtime,
        "transformers_determinism_source_evidence",
        lambda value: {"function_source_sha256": runtime.FROZEN_ENABLE_FULL_DETERMINISM_SHA256},
    )
    monkeypatch.setattr(
        runtime.os,
        "environ",
        dict(process_environment.FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT),
    )
    python_state = random.getstate()
    try:
        with pytest.raises(RuntimeError, match="strict deterministic mode"):
            runtime._seed_everything({"torch": fake_torch, "transformers": transformers}, 7)
    finally:
        random.setstate(python_state)


def test_seed_everything_rejects_helper_source_drift_and_environment_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    numpy_random = _SeedRecorder()
    calls: list[tuple[int, bool]] = []
    transformers = _fake_transformers_with_full_determinism(fake_torch, numpy_random, calls)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: SimpleNamespace(random=numpy_random),
    )
    environment = dict(process_environment.FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT)
    monkeypatch.setattr(runtime.os, "environ", environment)
    monkeypatch.setattr(
        runtime,
        "transformers_determinism_source_evidence",
        lambda value: (_ for _ in ()).throw(RuntimeError("helper source differs")),
    )
    with pytest.raises(RuntimeError, match="helper source differs"):
        runtime._seed_everything({"torch": fake_torch, "transformers": transformers}, 7)
    assert calls == []

    mutating_helper = _fake_transformers_with_full_determinism(
        fake_torch,
        numpy_random,
        calls,
        cublas_value=":4096:8",
    )
    monkeypatch.setattr(
        runtime,
        "transformers_determinism_source_evidence",
        lambda value: {"function_source_sha256": runtime.FROZEN_ENABLE_FULL_DETERMINISM_SHA256},
    )
    python_state = random.getstate()
    try:
        with pytest.raises(RuntimeError, match="mutated the environment"):
            runtime._seed_everything({"torch": fake_torch, "transformers": mutating_helper}, 7)
    finally:
        random.setstate(python_state)


def test_token_lengths_stop_at_eos_and_reject_empty_rows() -> None:
    completion_ids = torch.tensor(
        [
            [10, 2, 99, 99],
            [10, 11, 12, 13],
            [2, 99, 99, 99],
        ]
    )
    assert runtime._token_lengths(completion_ids, 2) == [2, 4, 1]
    with pytest.raises(RuntimeError, match="lengths are invalid"):
        runtime._token_lengths(torch.empty((0, 3), dtype=torch.int64), 2)
    with pytest.raises(RuntimeError, match="lengths are invalid"):
        runtime._token_lengths(torch.empty((1, 0), dtype=torch.int64), 2)


def test_policy_and_kl_captures_exact_finite_tensors() -> None:
    prior_enabled = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    reference = torch.tensor([[-0.8, -1.2]])
    policy = torch.tensor([[-1.0, -1.1]])
    prompt_ids = torch.tensor([[1, 2]])
    completion_ids = torch.tensor([[3, 4]])
    prompt_mask = torch.tensor([[1, 1]])
    completion_mask = torch.tensor([[1, 1]])

    class Trainer:
        model = object()
        args = SimpleNamespace(per_device_train_batch_size=4)

        def _get_per_token_logps(
            self,
            model: object,
            ids: torch.Tensor,
            attention_mask: torch.Tensor,
            logits_to_keep: int,
            batch_size: int,
        ) -> torch.Tensor:
            assert model is self.model
            assert torch.equal(ids, torch.tensor([[1, 2, 3, 4]]))
            assert torch.equal(attention_mask, torch.tensor([[1, 1, 1, 1]]))
            assert logits_to_keep == 2
            assert batch_size == 4
            return policy

    result = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "ref_per_token_logps": reference,
    }
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        captured_reference, captured_policy, per_token_kl = runtime._policy_and_kl(
            Trainer(), result, torch
        )
        expected = torch.exp(reference - policy) - (reference - policy) - 1
        assert captured_reference is reference
        assert captured_policy is policy
        assert torch.equal(per_token_kl, expected)

        with pytest.raises(RuntimeError, match="no reference log probabilities"):
            runtime._policy_and_kl(Trainer(), {**result, "ref_per_token_logps": None}, torch)
        with pytest.raises(RuntimeError, match="not finite"):
            runtime._policy_and_kl(
                Trainer(), {**result, "ref_per_token_logps": torch.tensor([[float("nan")]])}, torch
            )
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(RuntimeError, match="strict determinism"):
            runtime._policy_and_kl(Trainer(), result, torch)
    finally:
        torch.use_deterministic_algorithms(prior_enabled, warn_only=prior_warn_only)


def test_write_json_new_round_trips_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    runtime._write_json_new(path, {"z": 1, "a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2, "z": 1}
    assert path.read_text(encoding="utf-8").startswith('{\n  "a": 2,')
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runtime._write_json_new(path, {})


def test_same_process_replay_runs_exactly_three_fresh_trainers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated: list[tuple[Path, str]] = []
    trainer_paths: list[Path] = []
    cublas_entries: list[str] = []
    written_packets: list[tuple[Path, str, str]] = []

    def fake_artifact(paths: object, path: Path, label: str) -> Path:
        validated.append((path, label))
        return path

    def fake_single(**kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        trainer_path = kwargs["trainer_output_dir"]
        assert isinstance(trainer_path, Path)
        trainer_paths.append(trainer_path)
        cublas_entry = kwargs["expected_entry_cublas_workspace_config"]
        assert isinstance(cublas_entry, str)
        cublas_entries.append(cublas_entry)
        index = len(trainer_paths)
        return {"packet_sha256": _hash(f"packet-{index}")}, {"run": index}

    def fake_write(path: Path, packet: dict[str, object], *, kind: str) -> str:
        packet_hash = str(packet["packet_sha256"])
        written_packets.append((path, packet_hash, kind))
        return packet_hash

    def fake_exact(packets: list[dict[str, object]], *, expected_kind: str) -> str:
        assert len(packets) == 3
        assert expected_kind == "generation_only"
        return _hash("common")

    monkeypatch.setattr(runtime, "assert_artifact_path", fake_artifact)
    monkeypatch.setattr(runtime, "validate_runtime_paths", lambda paths: {})
    monkeypatch.setattr(runtime, "_single_generation_replay", fake_single)
    monkeypatch.setattr(runtime, "write_replay_packet_new", fake_write)
    monkeypatch.setattr(runtime, "assert_exact_replay", fake_exact)
    raw_directory = tmp_path / "raw"
    summary_path = tmp_path / "summary.json"
    runtime_paths = _runtime_paths(tmp_path)
    summary = runtime.run_same_process_replay(
        runtime_paths=runtime_paths,  # type: ignore[arg-type]
        config_path=tmp_path / "config.yaml",
        packet_path=tmp_path / "packet.json",
        manifest_path=tmp_path / "manifest.json",
        arm="generic_control",
        raw_directory=raw_directory,
        summary_path=summary_path,
    )

    assert validated == [
        (raw_directory, "same-process replay evidence"),
        (summary_path, "same-process replay summary"),
    ]
    assert trainer_paths == [
        raw_directory / "run_1" / "trainer",
        raw_directory / "run_2" / "trainer",
        raw_directory / "run_3" / "trainer",
    ]
    assert cublas_entries == [
        runtime.FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG,
        runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG,
        runtime.FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG,
    ]
    assert [path for path, _, _ in written_packets] == [
        raw_directory / "run_1.json",
        raw_directory / "run_2.json",
        raw_directory / "run_3.json",
    ]
    assert {kind for _, _, kind in written_packets} == {"generation_only"}
    assert summary["runs"] == 3
    assert summary["completions_per_run"] == 12
    assert summary["common_packet_sha256"] == _hash("common")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    assert summary["summary_sha256"] == canonical_sha256(payload)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    with pytest.raises(FileExistsError, match="must start unused"):
        runtime.run_same_process_replay(
            runtime_paths=runtime_paths,  # type: ignore[arg-type]
            config_path=tmp_path / "config.yaml",
            packet_path=tmp_path / "packet.json",
            manifest_path=tmp_path / "manifest.json",
            arm="generic_control",
            raw_directory=raw_directory,
            summary_path=summary_path,
        )


def test_one_fresh_process_packet_writes_self_hashed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated_paths: list[Path] = []
    trainer_output = tmp_path / "trainer"
    packet_hash = _hash("packet")

    monkeypatch.setattr(
        runtime,
        "assert_artifact_path",
        lambda paths, path, label: validated_paths.append(path) or path,
    )
    monkeypatch.setattr(runtime, "validate_runtime_paths", lambda paths: {})

    def fake_single(**kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        assert kwargs["trainer_output_dir"] == trainer_output
        assert kwargs["expected_entry_cublas_workspace_config"] == (
            runtime.FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
        )
        return {"packet_sha256": packet_hash}, {"runtime_seconds": 1.25}

    monkeypatch.setattr(runtime, "_single_generation_replay", fake_single)
    monkeypatch.setattr(
        runtime,
        "write_replay_packet_new",
        lambda path, packet, *, kind: packet_hash,
    )
    monkeypatch.setattr(runtime.sys, "argv", ["python", "-m", "foundry.replay", "one-process"])
    metadata_path = tmp_path / "metadata.json"
    raw_packet_path = tmp_path / "packet.json"
    runtime_paths = _runtime_paths(tmp_path)
    metadata = runtime.run_one_fresh_process_packet(
        runtime_paths=runtime_paths,  # type: ignore[arg-type]
        config_path=tmp_path / "config.yaml",
        packet_path=tmp_path / "schedule.json",
        manifest_path=tmp_path / "manifest.json",
        arm="generic_control",
        raw_packet_path=raw_packet_path,
        trainer_output_dir=trainer_output,
        metadata_path=metadata_path,
    )

    assert validated_paths == [raw_packet_path, trainer_output, metadata_path]
    assert metadata["packet_sha256"] == packet_hash
    process_contract = runtime.deterministic_process_contract(runtime_paths)  # type: ignore[arg-type]
    assert metadata["process_command_sha256"] == process_contract.process_command_sha256
    assert metadata["deterministic_process_contract_sha256"] == (process_contract.contract_sha256)
    assert metadata["process_id"] == runtime.os.getpid()
    assert metadata["process_instance_sha256"] == runtime._PROCESS_INSTANCE_SHA256
    assert metadata["raw_packet_path_sha256"] == runtime._resolved_path_sha256(raw_packet_path)
    assert metadata["trainer_output_dir_sha256"] == runtime._resolved_path_sha256(trainer_output)
    assert metadata["metadata_path_sha256"] == runtime._resolved_path_sha256(metadata_path)
    payload = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    assert metadata["metadata_sha256"] == canonical_sha256(payload)
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata


def _write_metadata(
    path: Path,
    *,
    packet_hash: str,
    packet_path: Path,
    trainer_path: Path,
    process: int,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "runtime_id": runtime.REPLAY_RUNTIME_ID,
        "packet_sha256": packet_hash,
        "process_command_sha256": _hash(f"command-{process}"),
        "deterministic_process_contract_sha256": _hash(f"process-contract-{process}"),
        "process_id": 1000 + process,
        "parent_process_id": 900,
        "process_instance_sha256": _hash(f"process-{process}"),
        "runtime_path_contract_sha256": _hash("runtime-path-contract"),
        "process_environment_sha256": _hash("process-environment"),
        "process_command_template_sha256": _hash("process-command-template"),
        "raw_packet_path_sha256": runtime._resolved_path_sha256(packet_path),
        "trainer_output_dir_sha256": runtime._resolved_path_sha256(trainer_path),
        "metadata_path_sha256": runtime._resolved_path_sha256(path),
        "resource_measurement": {"process": process},
    }
    value["metadata_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_combine_fresh_process_replay_validates_metadata_and_self_hashes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common_hash = _hash("common-packet")
    packet_paths = [tmp_path / f"packet-{index}.json" for index in range(3)]
    metadata_paths = [tmp_path / f"metadata-{index}.json" for index in range(3)]
    for index, path in enumerate(packet_paths):
        path.write_text(f"packet-{index}\n", encoding="utf-8")
        _write_metadata(
            metadata_paths[index],
            packet_hash=common_hash,
            packet_path=path,
            trainer_path=tmp_path / f"trainer-{index}",
            process=index,
        )
    monkeypatch.setattr(
        runtime,
        "compare_fresh_process_packets",
        lambda paths, *, expected_kind: common_hash,
    )
    monkeypatch.setattr(runtime, "validate_runtime_paths", lambda paths: {})
    monkeypatch.setattr(runtime, "assert_artifact_path", lambda paths, path, label: path)
    summary_path = tmp_path / "summary.json"
    runtime_paths = _runtime_paths(tmp_path)
    summary = runtime.combine_fresh_process_replay(
        runtime_paths=runtime_paths,  # type: ignore[arg-type]
        packet_paths=packet_paths,
        metadata_paths=metadata_paths,
        summary_path=summary_path,
    )

    assert summary["processes"] == 3
    assert summary["common_packet_sha256"] == common_hash
    assert summary["process_ids"] == [1000, 1001, 1002]
    assert summary["distinct_processes_verified"] is True
    assert summary["distinct_output_paths_verified"] is True
    assert summary["packet_file_sha256s"] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in packet_paths
    ]
    assert summary["process_command_sha256s"] == [
        _hash("command-0"),
        _hash("command-1"),
        _hash("command-2"),
    ]
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    assert summary["summary_sha256"] == canonical_sha256(payload)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary

    with pytest.raises(ValueError, match="exactly three"):
        runtime.combine_fresh_process_replay(
            runtime_paths=runtime_paths,  # type: ignore[arg-type]
            packet_paths=packet_paths[:2],
            metadata_paths=metadata_paths[:2],
            summary_path=tmp_path / "short.json",
        )
    with pytest.raises(ValueError, match="paths must be distinct"):
        runtime.combine_fresh_process_replay(
            runtime_paths=runtime_paths,  # type: ignore[arg-type]
            packet_paths=[packet_paths[0], packet_paths[0], packet_paths[2]],
            metadata_paths=metadata_paths,
            summary_path=tmp_path / "duplicate-path.json",
        )
    duplicate_process = json.loads(metadata_paths[2].read_text(encoding="utf-8"))
    duplicate_process["process_id"] = 1001
    duplicate_process.pop("metadata_sha256")
    duplicate_process["metadata_sha256"] = canonical_sha256(duplicate_process)
    metadata_paths[2].write_text(json.dumps(duplicate_process), encoding="utf-8")
    with pytest.raises(RuntimeError, match="three distinct processes"):
        runtime.combine_fresh_process_replay(
            runtime_paths=runtime_paths,  # type: ignore[arg-type]
            packet_paths=packet_paths,
            metadata_paths=metadata_paths,
            summary_path=tmp_path / "duplicate-process.json",
        )
    _write_metadata(
        metadata_paths[2],
        packet_hash=common_hash,
        packet_path=packet_paths[2],
        trainer_path=tmp_path / "trainer-2",
        process=2,
    )
    tampered = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    tampered["resource_measurement"] = {"process": 99}
    metadata_paths[0].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash differs"):
        runtime.combine_fresh_process_replay(
            runtime_paths=runtime_paths,  # type: ignore[arg-type]
            packet_paths=packet_paths,
            metadata_paths=metadata_paths,
            summary_path=tmp_path / "tampered.json",
        )


def test_main_dispatches_all_replay_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def same_process(**kwargs: object) -> dict[str, object]:
        calls.append(("same", kwargs))
        return {"command": "same"}

    def one_process(**kwargs: object) -> dict[str, object]:
        calls.append(("one", kwargs))
        return {"command": "one"}

    def combine(**kwargs: object) -> dict[str, object]:
        calls.append(("combine", kwargs))
        return {"command": "combine"}

    monkeypatch.setattr(runtime, "run_same_process_replay", same_process)
    monkeypatch.setattr(runtime, "run_one_fresh_process_packet", one_process)
    monkeypatch.setattr(runtime, "combine_fresh_process_replay", combine)
    runtime_paths = _runtime_paths(tmp_path)
    monkeypatch.setattr(runtime, "load_runtime_paths", lambda path: runtime_paths)
    common = [
        "--runtime-paths",
        str(tmp_path / "runtime-paths.json"),
        "--config",
        str(tmp_path / "config.yaml"),
        "--packet",
        str(tmp_path / "packet.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
    ]
    assert (
        runtime.main(
            [
                "same-process",
                *common,
                "--arm",
                "generic_control",
                "--raw-directory",
                str(tmp_path / "raw"),
                "--summary",
                str(tmp_path / "same-summary.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"command": "same"}
    assert (
        runtime.main(
            [
                "one-process",
                *common,
                "--arm",
                "generic_control",
                "--raw-packet",
                str(tmp_path / "raw-packet.json"),
                "--trainer-output",
                str(tmp_path / "trainer"),
                "--metadata",
                str(tmp_path / "metadata.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"command": "one"}
    packets = [tmp_path / f"packet-{index}.json" for index in range(3)]
    metadata = [tmp_path / f"metadata-{index}.json" for index in range(3)]
    assert (
        runtime.main(
            [
                "combine",
                "--runtime-paths",
                str(tmp_path / "runtime-paths.json"),
                "--packets",
                *(str(path) for path in packets),
                "--metadata",
                *(str(path) for path in metadata),
                "--summary",
                str(tmp_path / "combine-summary.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"command": "combine"}

    assert [name for name, _ in calls] == ["same", "one", "combine"]
    assert calls[0][1]["arm"] == "generic_control"
    assert calls[0][1]["raw_directory"] == tmp_path / "raw"
    assert calls[1][1]["arm"] == "generic_control"
    assert calls[1][1]["raw_packet_path"] == tmp_path / "raw-packet.json"
    assert calls[2][1]["packet_paths"] == packets
    assert calls[2][1]["metadata_paths"] == metadata


def test_cli_rejects_the_nonfrozen_targeted_arm(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runtime.main(
            [
                "same-process",
                "--runtime-paths",
                str(tmp_path / "runtime-paths.json"),
                "--config",
                str(tmp_path / "config.yaml"),
                "--packet",
                str(tmp_path / "packet.json"),
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--arm",
                "targeted",
                "--raw-directory",
                str(tmp_path / "raw"),
                "--summary",
                str(tmp_path / "summary.json"),
            ]
        )
