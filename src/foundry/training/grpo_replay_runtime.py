"""Exact same-machine generation replay for the frozen verifier-GRPO contract."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.grpo_compatibility import (
    CONTRACT_ID,
    TopPWarningOnlyGenerationContract,
    model_adapter_state,
)
from foundry.training.grpo_config import BASE_REVISION, load_grpo_config
from foundry.training.grpo_environment import (
    FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT,
    allowlisted_environment_evidence,
    assert_idempotent_deterministic_initialization,
    make_environment_guarded_trainer,
    transformers_determinism_source_evidence,
    validate_deterministic_process_environment,
)
from foundry.training.grpo_gpu import (
    ChildCudaComputeEvidence,
    current_cuda_memory_evidence,
)
from foundry.training.grpo_paths import (
    GrpoRuntimePaths,
    assert_artifact_path,
    assert_source_path,
    deterministic_process_contract,
    frozen_process_environment,
    load_runtime_paths,
    same_canonical_path,
    validate_runtime_paths,
)
from foundry.training.grpo_reference import assert_only_lora_trainable
from foundry.training.grpo_replay_evidence import (
    GenerationEvidence,
    assert_exact_replay,
    build_generation_only_packet,
    capture_base_parameter_state,
    capture_generation_evidence,
    capture_lora_state,
    capture_rng_state,
    compare_fresh_process_packets,
    write_replay_packet_new,
)
from foundry.training.grpo_reward import (
    reward_configuration_sha256,
    reward_implementation_sha256,
)
from foundry.training.grpo_runtime import (
    VerifierRewardCallback,
    _assert_offline_model_snapshot,
    _base_reference_hash,
    _completion_token_counter,
    _peak_process_ram,
    _prepare_runtime,
    _repeat_row,
    _runtime_modules,
    _validate_cuda,
    assert_cuda_only_model,
    assert_dropout_disabled,
    assert_frozen_grpo_arguments,
    frozen_grpo_argument_values,
    load_runtime_schedule,
    select_compatibility_groups,
    summarize_reward_records,
)
from foundry.training.grpo_schedule import COMPLETIONS_PER_GROUP, Arm
from foundry.training.grpo_trainer import make_truncation_aware_grpo_trainer
from foundry.training.lora_scaling import base_parameter_signature_sha256

REPLAY_RUNTIME_ID = "foundry-verifier-grpo-generation-replay-v1"
REPLAY_RUNTIME_SCHEMA_VERSION = 1
REPLAY_RUNS = 3
GENERATION_GROUPS = 3
GENERATION_COMPLETIONS = GENERATION_GROUPS * COMPLETIONS_PER_GROUP
FROZEN_REPLAY_ARM: Arm = "generic_control"
FROZEN_REPLAY_SEED = 20_260_720
FROZEN_GENERIC_PACKET_SHA256 = "67f48ebc3a310c0cb0db882b46759e993f3ad99faec9b7309d336d7b97f44400"
FROZEN_GENERIC_MANIFEST_SHA256 = "5848ed6640dda21752ab9692c8e531d9175314a7d5a472616dc19ad834a6351e"
FROZEN_REPLAY_GROUP_IDS = (
    "grpo-generic_control-g001",
    "grpo-generic_control-g002",
    "grpo-generic_control-g005",
)
FROZEN_REWARD_CONFIGURATION_SHA256 = (
    "4a47359fa3129b1bfd79dd158ecb609177e9b1642a95368c106e016a1554a965"
)
FROZEN_REWARD_IMPLEMENTATION_SHA256 = (
    "089650105e29ead3c4ad62f1e0e41263e6c2af5fb8a12cb2851644aca3599616"
)
ZERO_VARIANCE_POLICY_ID = "foundry-generation-only-zero-variance-record-v1"
FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG = ":16:8"
FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG = FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
# Compatibility alias for callers that name the externally supplied setting.
FROZEN_CUBLAS_WORKSPACE_CONFIG = FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
FROZEN_ENABLE_FULL_DETERMINISM_SHA256 = (
    "1893964197a05bfd07d1477815b58e42e883b9e64985f0e795b4562fc9f84834"
)
FULL_DETERMINISM_TRANSITION_ID = "transformers-enable-full-determinism-idempotent-v4.51.3"
_PYTHON_HASH_PROBE = "foundry-verifier-grpo-python-hash-probe-v1"
_PROCESS_START_CUBLAS_WORKSPACE_CONFIG = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
_PROCESS_INSTANCE_STARTED_NS = time.time_ns()
_PROCESS_INSTANCE_SHA256 = canonical_sha256(
    {
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "process_instance_started_ns": _PROCESS_INSTANCE_STARTED_NS,
        "sys_executable": str(Path(sys.executable).resolve()),
    }
)
_FROZEN_SOFTWARE_VERSIONS = {
    "accelerate": "1.7.0",
    "bitsandbytes": "0.49.2",
    "datasets": "5.0.0",
    "numpy": "2.5.1",
    "peft": "0.15.2",
    "psutil": "7.2.2",
    "tokenizers": "0.21.4",
    "torch": "2.5.1+cu121",
    "transformers": "4.51.3",
    "trl": "0.17.0",
}
_FROZEN_PYTHON = {
    "implementation": "CPython",
    "version": "3.12.10",
}
_FROZEN_PYTHON_EXECUTABLE_SHA256 = (
    "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
)
_FROZEN_PYTHON_EXECUTABLE_SIZE_BYTES = 274_424
_FROZEN_OS = {
    "architecture": ["64bit", "WindowsPE"],
    "machine": "AMD64",
    "release": "11",
    "system": "Windows",
    "version": "10.0.26200",
}


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite replay output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _resolved_path_sha256(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def _source_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"replay source file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    characters = frozenset("0123456789abcdef")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in characters for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_distinct_paths(paths: Sequence[Path], *, label: str) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} paths must be distinct")


def _strict_determinism(torch: Any) -> bool:
    return bool(torch.are_deterministic_algorithms_enabled()) and not bool(
        torch.is_deterministic_algorithms_warn_only_enabled()
    )


def _filtered_environment_sha256(
    runtime_paths: GrpoRuntimePaths,
) -> tuple[str, int, int]:
    """Hash the explicit environment allowlist, never the raw parent environment."""

    return allowlisted_environment_evidence(runtime_paths)


def _external_process_evidence(
    seed: int,
    runtime_paths: GrpoRuntimePaths,
    *,
    expected_entry_cublas_workspace_config: str,
) -> dict[str, object]:
    """Prove hash seeding and one unchanged pre-launch deterministic contract."""

    if seed != FROZEN_REPLAY_SEED:
        raise ValueError("generation replay seed differs from the frozen process seed")
    launch_evidence = validate_deterministic_process_environment(
        runtime_paths, "generation_replay_process_entry"
    )
    expected_seed = str(seed)
    if _PROCESS_START_CUBLAS_WORKSPACE_CONFIG != FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "the replay module must be imported by a process launched with "
            f"CUBLAS_WORKSPACE_CONFIG={FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG!r}"
        )
    if expected_entry_cublas_workspace_config != FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG:
        raise ValueError("expected replay-entry cuBLAS configuration is not frozen")
    actual_entry = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_entry != expected_entry_cublas_workspace_config:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG differs from the expected replay-entry state: "
            f"expected {expected_entry_cublas_workspace_config!r}, got {actual_entry!r}"
        )
    if not same_canonical_path(Path(sys.executable), runtime_paths.python_executable):
        raise RuntimeError("replay process did not use the configured Python executable")
    current_hash = hash(_PYTHON_HASH_PROBE)
    completed = subprocess.run(
        [
            str(runtime_paths.python_executable),
            "-S",
            "-c",
            f"print(hash({_PYTHON_HASH_PROBE!r}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=frozen_process_environment(runtime_paths),
    )
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if values != [str(current_hash)]:
        raise RuntimeError(
            "PYTHONHASHSEED environment does not match the running interpreter hash seed"
        )
    return {
        "python_hash_randomization": True,
        "python_hash_seed": expected_seed,
        "python_hash_probe_sha256": canonical_sha256(
            {
                "probe_text_sha256": hashlib.sha256(_PYTHON_HASH_PROBE.encode("utf-8")).hexdigest(),
                "probe_value": str(current_hash),
            }
        ),
        "cublas_transition_id": FULL_DETERMINISM_TRANSITION_ID,
        "process_start_cublas_workspace_config": (FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG),
        "active_cublas_workspace_config": FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG,
        "first_replay_entry_cublas_workspace_config": (
            FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
        ),
        "subsequent_replay_entry_cublas_workspace_config": (
            FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
        ),
        "environment_transition_occurred": False,
        "entry_state_verified": True,
        "deterministic_process_environment": {
            "stage": launch_evidence["stage"],
            "environment": launch_evidence["environment"],
            "environment_sha256": launch_evidence["environment_sha256"],
            "python_hash_randomization": launch_evidence["python_hash_randomization"],
        },
    }


def _runtime_environment_evidence(
    runtime_paths: GrpoRuntimePaths,
    numpy: Any,
) -> dict[str, object]:
    """Require and record the exact frozen interpreter, packages, and OS."""

    executable = Path(sys.executable).resolve()
    expected_executable = runtime_paths.python_executable.resolve(strict=True)
    if not same_canonical_path(executable, expected_executable):
        raise RuntimeError(
            f"replay must use the configured training interpreter: {expected_executable}"
        )
    if not executable.is_file():
        raise FileNotFoundError("frozen training interpreter is missing")
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    executable_size = executable.stat().st_size
    if (
        executable_sha256 != _FROZEN_PYTHON_EXECUTABLE_SHA256
        or executable_size != _FROZEN_PYTHON_EXECUTABLE_SIZE_BYTES
    ):
        raise RuntimeError("training interpreter binary differs from the frozen contract")
    python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    if python != _FROZEN_PYTHON:
        raise RuntimeError(f"Python runtime differs from the frozen contract: {python}")
    operating_system: dict[str, object] = {
        "architecture": list(platform.architecture()),
        "machine": platform.machine(),
        "release": platform.release(),
        "system": platform.system(),
        "version": platform.version(),
    }
    if operating_system != _FROZEN_OS:
        raise RuntimeError(f"operating system differs from the frozen contract: {operating_system}")
    software_versions = {
        name: importlib.metadata.version(name) for name in _FROZEN_SOFTWARE_VERSIONS
    }
    if software_versions != _FROZEN_SOFTWARE_VERSIONS:
        raise RuntimeError(
            f"replay software versions differ from the frozen contract: {software_versions}"
        )
    if str(getattr(numpy, "__version__", "")) != _FROZEN_SOFTWARE_VERSIONS["numpy"]:
        raise RuntimeError("imported NumPy version differs from package metadata")
    tokenizers = importlib.import_module("tokenizers")
    if str(getattr(tokenizers, "__version__", "")) != _FROZEN_SOFTWARE_VERSIONS["tokenizers"]:
        raise RuntimeError("imported tokenizers version differs from package metadata")
    return {
        "sys_executable": str(executable),
        "sys_executable_path_sha256": _resolved_path_sha256(executable),
        "sys_executable_file_sha256": executable_sha256,
        "sys_executable_size_bytes": executable_size,
        "python": python,
        "software_versions": software_versions,
        "operating_system": operating_system,
    }


def _validate_frozen_cuda(torch: Any, runtime_paths: GrpoRuntimePaths) -> ChildCudaComputeEvidence:
    """Require the frozen RTX 3080 through direct PyTorch CUDA computation."""

    return _validate_cuda(torch, runtime_paths, stage="generation_replay")


def _prepare_cuda_replay(torch: Any) -> None:
    torch.cuda.synchronize(0)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)


def _cleanup_cuda_replay(torch: Any) -> dict[str, int]:
    torch.cuda.synchronize(0)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize(0)
    return {
        "post_cleanup_allocated_vram_bytes": int(torch.cuda.memory_allocated(0)),
        "post_cleanup_reserved_vram_bytes": int(torch.cuda.memory_reserved(0)),
    }


def _assert_frozen_reward_contract() -> dict[str, str]:
    value = {
        "reward_configuration_sha256": reward_configuration_sha256(),
        "reward_implementation_sha256": reward_implementation_sha256(),
    }
    expected = {
        "reward_configuration_sha256": FROZEN_REWARD_CONFIGURATION_SHA256,
        "reward_implementation_sha256": FROZEN_REWARD_IMPLEMENTATION_SHA256,
    }
    if value != expected:
        raise RuntimeError(f"reward contract differs from the frozen hashes: {value}")
    return value


def _assert_frozen_schedule_binding(
    schedule: Any,
    *,
    arm: Arm,
    group_ids: Sequence[str],
) -> None:
    if arm != FROZEN_REPLAY_ARM or getattr(schedule, "arm", None) != FROZEN_REPLAY_ARM:
        raise ValueError("generation replay is bound to the frozen generic-control arm")
    if getattr(schedule, "packet_sha256", None) != FROZEN_GENERIC_PACKET_SHA256:
        raise ValueError("generation replay packet differs from the frozen generic packet")
    if getattr(schedule, "manifest_sha256", None) != FROZEN_GENERIC_MANIFEST_SHA256:
        raise ValueError("generation replay manifest differs from the frozen generic manifest")
    if tuple(group_ids) != FROZEN_REPLAY_GROUP_IDS:
        raise ValueError("generation replay group IDs differ from the frozen three groups")


def _reward_variance_decisions(
    records: Sequence[Any],
    groups: Sequence[Any],
) -> dict[str, object]:
    """Record, but never hide, zero variance in a generation-only replay."""

    decisions: list[dict[str, object]] = []
    for group in groups:
        group_id = str(group.group_id)
        totals = [
            float(record.reward.total).hex()
            for record in records
            if str(record.group_id) == group_id
        ]
        if len(totals) != COMPLETIONS_PER_GROUP:
            raise RuntimeError("reward variance evidence lacks exactly four group completions")
        zero_variance = len(set(totals)) == 1
        decisions.append(
            {
                "group_id": group_id,
                "zero_variance": zero_variance,
                "decision": (
                    "recorded_generation_only_no_optimizer_update"
                    if zero_variance
                    else "recorded_nonzero_variance"
                ),
                "reward_total_hex_sha256": canonical_sha256(totals),
            }
        )
    value: dict[str, object] = {
        "policy_id": ZERO_VARIANCE_POLICY_ID,
        "optimizer_updates": 0,
        "zero_variance_allowed_for_generation_only": True,
        "compatibility_updates_require_nonzero_variance": True,
        "decisions": decisions,
        "zero_variance_groups": sum(bool(item["zero_variance"]) for item in decisions),
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _full_determinism_evidence(modules: Mapping[str, Any], seed: int) -> dict[str, object]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    trainer_utils = getattr(transformers, "trainer_utils", None)
    helper = getattr(trainer_utils, "enable_full_determinism", None)
    if not callable(helper):
        raise RuntimeError("Transformers full-determinism helper is unavailable")
    source_evidence = transformers_determinism_source_evidence(transformers)
    source_sha256 = str(source_evidence["function_source_sha256"])
    active_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if active_cublas != FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Transformers full-determinism helper did not activate the frozen cuBLAS config"
        )
    if not _strict_determinism(torch):
        raise RuntimeError("strict deterministic mode was not enabled before replay")
    if not bool(torch.backends.cudnn.deterministic) or bool(torch.backends.cudnn.benchmark):
        raise RuntimeError("Transformers helper did not enable deterministic cuDNN behavior")
    blocking_environment = {
        name: os.environ.get(name) for name in FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT
    }
    if blocking_environment != dict(FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT):
        raise RuntimeError("Transformers helper did not set its frozen blocking environment")
    if random.getstate() != random.Random(seed).getstate():
        raise RuntimeError("Transformers helper did not seed the Python random generator")
    evidence: dict[str, object] = {
        "transition_id": FULL_DETERMINISM_TRANSITION_ID,
        "helper": "transformers.trainer_utils.enable_full_determinism",
        "helper_source_sha256": source_sha256,
        "seed": seed,
        "warn_only": False,
        "process_start_cublas_workspace_config": (FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG),
        "active_cublas_workspace_config": active_cublas,
        "blocking_environment": blocking_environment,
        "source_evidence": source_evidence,
        "effective_environment_changed": False,
        "torch_strict_determinism": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "python_random_seeded": True,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _seed_everything(modules: Mapping[str, Any], seed: int) -> Any:
    """Enter strict mode through the exact installed Transformers helper."""

    torch = modules["torch"]
    transformers = modules["transformers"]
    numpy = importlib.import_module("numpy")
    trainer_utils = getattr(transformers, "trainer_utils", None)
    helper = getattr(trainer_utils, "enable_full_determinism", None)
    if not callable(helper):
        raise RuntimeError("Transformers full-determinism helper is unavailable")
    transformers_determinism_source_evidence(transformers)
    before = {key: os.environ.get(key) for key in FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT}
    if before != dict(FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT):
        raise RuntimeError("deterministic environment differs before Transformers initialization")
    helper(seed, warn_only=False)
    after = {key: os.environ.get(key) for key in FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT}
    if after != before:
        raise RuntimeError("Transformers deterministic initialization mutated the environment")
    torch.backends.cuda.matmul.allow_tf32 = False
    _full_determinism_evidence(modules, seed)
    return numpy


def _token_lengths(completion_ids: Any, eos_token_id: int) -> list[int]:
    rows = completion_ids.detach().cpu().tolist()
    values: list[int] = []
    for row in rows:
        try:
            values.append(row.index(eos_token_id) + 1)
        except ValueError:
            values.append(len(row))
    if not values or any(value < 1 for value in values):
        raise RuntimeError("generated completion lengths are invalid")
    return values


def _policy_and_kl(trainer: Any, result: Mapping[str, Any], torch: Any) -> tuple[Any, Any, Any]:
    if not _strict_determinism(torch):
        raise RuntimeError("policy/reference evidence must run under strict determinism")
    prompt_ids = result["prompt_ids"]
    prompt_mask = result["prompt_mask"]
    completion_ids = result["completion_ids"]
    completion_mask = result["completion_mask"]
    reference = result["ref_per_token_logps"]
    if reference is None:
        raise RuntimeError("G1 generation replay produced no reference log probabilities")
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    with torch.no_grad():
        policy = trainer._get_per_token_logps(
            trainer.model,
            prompt_completion_ids,
            attention_mask,
            logits_to_keep,
            trainer.args.per_device_train_batch_size,
        )
    delta = reference - policy
    per_token_kl = torch.exp(delta) - delta - 1
    for name, tensor in (
        ("reference log probabilities", reference),
        ("policy log probabilities", policy),
        ("per-token KL", per_token_kl),
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"{name} are not finite")
    return reference, policy, per_token_kl


def _single_generation_replay_impl(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    arm: Arm,
    trainer_output_dir: Path,
    expected_entry_cublas_workspace_config: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if arm != FROZEN_REPLAY_ARM:
        raise ValueError("generation replay is bound to the frozen generic-control arm")
    validate_runtime_paths(runtime_paths)
    assert_source_path(runtime_paths, config_path, "replay configuration")
    assert_source_path(runtime_paths, manifest_path, "replay schedule manifest")
    assert_artifact_path(runtime_paths, packet_path, "replay schedule packet")
    assert_artifact_path(runtime_paths, trainer_output_dir, "generation replay state")
    model_path = runtime_paths.model_snapshot_root
    config = load_grpo_config(config_path)
    if config.base_model.revision != BASE_REVISION:
        raise ValueError("base revision differs from the frozen Qwen checkpoint")
    external_process = _external_process_evidence(
        config.grpo.seed,
        runtime_paths,
        expected_entry_cublas_workspace_config=(expected_entry_cublas_workspace_config),
    )
    reward_contract = _assert_frozen_reward_contract()
    _assert_offline_model_snapshot(model_path, config)
    if trainer_output_dir.exists():
        raise FileExistsError("generation replay trainer path must be unused")
    schedule = load_runtime_schedule(
        packet_path,
        manifest_path,
        expected_arm=FROZEN_REPLAY_ARM,
    )
    update_groups, replay_group = select_compatibility_groups(schedule)
    groups = (*update_groups, replay_group)
    _assert_frozen_schedule_binding(
        schedule,
        arm=arm,
        group_ids=[group.group_id for group in groups],
    )

    deterministic_stages = [
        validate_deterministic_process_environment(runtime_paths, "before_transformers_import")
    ]
    modules = _runtime_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    trl = modules["trl"]
    datasets = modules["datasets"]
    psutil = modules["psutil"]
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_transformers_import",
            torch_module=torch,
        )
    )
    before_full_determinism = deterministic_stages[-1]
    numpy = _seed_everything(modules, config.grpo.seed)
    after_full_determinism = validate_deterministic_process_environment(
        runtime_paths,
        "after_transformers_full_determinism",
        torch_module=torch,
        require_strict=True,
    )
    deterministic_stages.append(after_full_determinism)
    initialization_idempotence = assert_idempotent_deterministic_initialization(
        before_full_determinism, after_full_determinism
    )
    full_determinism = _full_determinism_evidence(modules, config.grpo.seed)
    runtime_environment = _runtime_environment_evidence(runtime_paths, numpy)
    child_cuda = _validate_frozen_cuda(torch, runtime_paths)
    cuda = child_cuda.stable_evidence()
    child_cuda_resource = child_cuda.resource_payload()
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_cuda_initialization",
            torch_module=torch,
            require_strict=True,
        )
    )
    _prepare_cuda_replay(torch)
    process = psutil.Process()
    started = time.perf_counter()
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "before_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    model, tokenizer, lora_config, model_load_seconds = _prepare_runtime(
        config, model_path, modules
    )
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    reward_callback = VerifierRewardCallback(
        groups,
        completion_token_counter=_completion_token_counter(tokenizer),
    )
    argument_values = frozen_grpo_argument_values(
        config,
        variant_id="G1",
        output_dir=trainer_output_dir,
        mode="compatibility",
    )
    arguments = trl.GRPOConfig(**argument_values)
    after_arguments = validate_deterministic_process_environment(
        runtime_paths,
        "after_grpo_config_full_determinism",
        torch_module=torch,
        require_strict=True,
    )
    deterministic_stages.append(after_arguments)
    assert_idempotent_deterministic_initialization(after_full_determinism, after_arguments)
    assert_frozen_grpo_arguments(
        arguments,
        config,
        variant_id="G1",
        output_dir=trainer_output_dir,
        mode="compatibility",
    )
    warning_contract = TopPWarningOnlyGenerationContract(
        torch_module=torch,
        generation_owner=transformers.GenerationMixin,
        top_p_call=transformers.generation.logits_process.TopPLogitsWarper.__call__,
    )
    boundary_counts: dict[str, int] = {}
    boundary_environment_sha256s: set[str] = set()

    def validate_boundary(stage: str) -> None:
        evidence = validate_deterministic_process_environment(
            runtime_paths,
            stage,
            torch_module=torch,
            require_strict=True,
        )
        boundary_counts[stage] = boundary_counts.get(stage, 0) + 1
        boundary_environment_sha256s.add(str(evidence["environment_sha256"]))

    audited_trainer_type = make_truncation_aware_grpo_trainer(
        trl.GRPOTrainer,
        generation_scope_factory=lambda: warning_contract.install("generation"),
    )
    trainer_type = make_environment_guarded_trainer(audited_trainer_type, validate_boundary)
    train_dataset = datasets.Dataset.from_list([group.policy_row() for group in update_groups])
    trainer = trainer_type(
        model=model,
        reward_funcs=reward_callback,
        args=arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    after_trainer = validate_deterministic_process_environment(
        runtime_paths,
        "after_grpo_trainer_construction",
        torch_module=torch,
        require_strict=True,
    )
    deterministic_stages.append(after_trainer)
    assert_idempotent_deterministic_initialization(after_full_determinism, after_trainer)
    if trainer.ref_model is not None:
        raise RuntimeError("generation replay unexpectedly created a second reference model")
    assert_only_lora_trainable(trainer.model)
    assert_cuda_only_model(trainer.model)
    assert_dropout_disabled(trainer.model, torch)
    warning_contract.bind_state_probe(partial(model_adapter_state, trainer.model))
    base_signature_before = base_parameter_signature_sha256(trainer.model)
    base_parameters_before = capture_base_parameter_state(trainer.model)
    base_output_before = _base_reference_hash(trainer.model, tokenizer, groups[0], torch)
    lora_before = capture_lora_state(trainer.model)
    rng_before = capture_rng_state(torch, numpy_random=numpy.random)

    generation_evidence: list[GenerationEvidence] = []
    for group in groups:
        record_start = len(reward_callback.records)
        warning_start = len(warning_contract.call_records())
        result = cast(
            Mapping[str, Any],
            trainer._generate_and_score_completions(_repeat_row(group)),
        )
        if not _strict_determinism(torch):
            raise RuntimeError("warning-only mode leaked after generation and scoring")
        records = reward_callback.records[record_start:]
        if len(records) != COMPLETIONS_PER_GROUP:
            raise RuntimeError("generation replay group did not produce four reward records")
        warning_records = warning_contract.call_records()[warning_start:]
        if len(warning_records) != 1:
            raise RuntimeError("generation replay expected exactly one generate call per group")
        reference, policy, per_token_kl = _policy_and_kl(trainer, result, torch)
        completion_ids = result["completion_ids"]
        generation_evidence.append(
            capture_generation_evidence(
                group_id=group.group_id,
                source_kind=group.source_kind,
                prompt_sha256=group.prompt_sha256,
                generated_token_ids=completion_ids,
                decoded_completions=[record.completion for record in records],
                completion_token_lengths=_token_lengths(
                    completion_ids, int(tokenizer.eos_token_id)
                ),
                truncation_flags=[record.reward.generation_truncated for record in records],
                reward_components=[record.reward.as_dict() for record in records],
                rng_before_sha256=warning_records[0].rng_before_sha256,
                rng_after_sha256=warning_records[0].rng_after_sha256,
                warning_sha256s=warning_records[0].warning_sha256s,
                reference_logprobs=reference,
                policy_logprobs=policy,
                per_token_kl=per_token_kl,
            )
        )

    reward_summary = summarize_reward_records(
        reward_callback.records,
        groups,
        require_nonzero_variance=False,
    )
    reward_variance = _reward_variance_decisions(reward_callback.records, groups)
    rng_after = capture_rng_state(torch, numpy_random=numpy.random)
    lora_after = capture_lora_state(trainer.model)
    if lora_after["lora_state_sha256"] != lora_before["lora_state_sha256"]:
        raise RuntimeError("generation-only replay changed the fresh adapter")
    base_signature_after = base_parameter_signature_sha256(trainer.model)
    base_parameters_after = capture_base_parameter_state(trainer.model)
    base_output_after = _base_reference_hash(trainer.model, tokenizer, groups[0], torch)
    if (
        base_signature_after != base_signature_before
        or base_parameters_after["base_parameter_state_sha256"]
        != base_parameters_before["base_parameter_state_sha256"]
        or base_output_after != base_output_before
    ):
        raise RuntimeError("generation-only replay changed the frozen base")
    warning_evidence = warning_contract.evidence()
    if warning_evidence["generation_calls"] != GENERATION_GROUPS:
        raise RuntimeError("warning contract call count differs from the three replay groups")
    environment_sha256, environment_count, excluded_environment_count = (
        _filtered_environment_sha256(runtime_paths)
    )
    run_contract: dict[str, object] = {
        "runtime_id": REPLAY_RUNTIME_ID,
        "runtime_source_sha256": _source_sha256(Path(__file__)),
        "replay_evidence_source_sha256": _source_sha256(
            Path(__file__).with_name("grpo_replay_evidence.py")
        ),
        "compatibility_source_sha256": _source_sha256(
            Path(__file__).with_name("grpo_compatibility.py")
        ),
        "trainer_hook_source_sha256": _source_sha256(Path(__file__).with_name("grpo_trainer.py")),
        "compatibility_contract_id": CONTRACT_ID,
        "arm": FROZEN_REPLAY_ARM,
        "config_sha256": config.config_sha256,
        "execution_sha256": config.execution_sha256("G1"),
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "group_ids": [group.group_id for group in groups],
        "source_kinds": [group.source_kind for group in groups],
        **reward_contract,
        "reward_summary": reward_summary,
        "reward_variance_decisions": reward_variance,
        "seed": config.grpo.seed,
        "beta": config.variant("G1").beta,
        "num_generations": config.grpo.num_generations,
        "max_completion_length": config.grpo.max_completion_length,
        "temperature": config.grpo.temperature,
        "top_p": config.grpo.top_p,
        "top_k": config.grpo.top_k,
        "base_revision": config.base_model.revision,
        "base_parameter_signature_sha256": base_signature_before,
        "base_parameter_state_sha256": base_parameters_before["base_parameter_state_sha256"],
        "base_parameter_count": base_parameters_before["parameter_count"],
        "base_parameter_numel": base_parameters_before["total_numel"],
        "base_parameter_bytes": base_parameters_before["total_bytes"],
        "base_output_sha256": base_output_before,
        "warning_contract": warning_evidence,
        "external_process_contract": external_process,
        "runtime_paths": runtime_paths.evidence(),
        "full_determinism_transition": full_determinism,
        "deterministic_initialization_idempotence": initialization_idempotence,
        "deterministic_process_environment_sha256": (runtime_paths.process_environment_sha256),
        "runtime_environment": runtime_environment,
        "environment_variable_sha256": environment_sha256,
        "environment_variable_count": environment_count,
        "secret_environment_variables_excluded": excluded_environment_count,
        **cuda,
    }
    packet = build_generation_only_packet(
        run_contract=run_contract,
        generations=generation_evidence,
        rng_before=rng_before,
        rng_after=rng_after,
        lora_state=lora_before,
    )
    torch.cuda.synchronize(0)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    cuda_memory_after_replay = current_cuda_memory_evidence(torch)
    resource: dict[str, object] = {
        "model_load_seconds": model_load_seconds,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_process_ram_bytes": _peak_process_ram(process),
        "child_cuda_probe_resource": child_cuda_resource,
        "cuda_memory_after_replay": cuda_memory_after_replay,
        "entry_cublas_workspace_config": expected_entry_cublas_workspace_config,
        "active_cublas_workspace_config": FROZEN_ACTIVE_CUBLAS_WORKSPACE_CONFIG,
        "deterministic_environment_stages": deterministic_stages,
        "boundary_validation_counts": dict(sorted(boundary_counts.items())),
        "boundary_environment_sha256s": sorted(boundary_environment_sha256s),
        "environment_mutation_observed": False,
    }
    return packet, resource


def _single_generation_replay(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    arm: Arm,
    trainer_output_dir: Path,
    expected_entry_cublas_workspace_config: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run one replay and synchronize/clear CUDA on every exit path."""

    result: tuple[dict[str, object], dict[str, object]] | None = None
    validate_runtime_paths(runtime_paths)
    try:
        result = _single_generation_replay_impl(
            runtime_paths=runtime_paths,
            config_path=config_path,
            packet_path=packet_path,
            manifest_path=manifest_path,
            arm=arm,
            trainer_output_dir=trainer_output_dir,
            expected_entry_cublas_workspace_config=(expected_entry_cublas_workspace_config),
        )
        return result
    finally:
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None)
        available = getattr(cuda, "is_available", None)
        if torch is not None and callable(available) and bool(available()):
            cleanup = _cleanup_cuda_replay(torch)
            if result is not None:
                result[1].update(cleanup)
        validate_runtime_paths(runtime_paths)


def run_same_process_replay(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    arm: Arm,
    raw_directory: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Run three fresh-base/fresh-adapter replays inside one Python process."""

    if arm != FROZEN_REPLAY_ARM:
        raise ValueError("generation replay is bound to the frozen generic-control arm")
    validate_runtime_paths(runtime_paths)
    _require_distinct_paths(
        [raw_directory, summary_path],
        label="same-process replay output",
    )
    assert_artifact_path(runtime_paths, raw_directory, "same-process replay evidence")
    assert_artifact_path(runtime_paths, summary_path, "same-process replay summary")
    if raw_directory.exists() or summary_path.exists():
        raise FileExistsError("same-process replay outputs must start unused")
    raw_directory.mkdir(parents=True)
    packets: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    for index in range(REPLAY_RUNS):
        packet, resource = _single_generation_replay(
            runtime_paths=runtime_paths,
            config_path=config_path,
            packet_path=packet_path,
            manifest_path=manifest_path,
            arm=arm,
            trainer_output_dir=raw_directory / f"run_{index + 1}" / "trainer",
            expected_entry_cublas_workspace_config=(FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG),
        )
        write_replay_packet_new(
            raw_directory / f"run_{index + 1}.json",
            packet,
            kind="generation_only",
        )
        packets.append(packet)
        resources.append(resource)
    common_hash = assert_exact_replay(packets, expected_kind="generation_only")
    summary: dict[str, object] = {
        "schema_version": REPLAY_RUNTIME_SCHEMA_VERSION,
        "runtime_id": REPLAY_RUNTIME_ID,
        "replay_kind": "same_process_generation",
        "runs": REPLAY_RUNS,
        "groups_per_run": GENERATION_GROUPS,
        "completions_per_run": GENERATION_COMPLETIONS,
        "packet_sha256s": [packet["packet_sha256"] for packet in packets],
        "common_packet_sha256": common_hash,
        "exact_replay_passed": True,
        "process_id": os.getpid(),
        "process_instance_sha256": _PROCESS_INSTANCE_SHA256,
        "resource_measurements": resources,
        "external_artifact_root_validated": True,
        "runtime_paths": runtime_paths.evidence(),
        "prompts_or_completions_in_summary": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json_new(summary_path, summary)
    validate_runtime_paths(runtime_paths)
    return summary


def run_one_fresh_process_packet(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    arm: Arm,
    raw_packet_path: Path,
    trainer_output_dir: Path,
    metadata_path: Path,
) -> dict[str, object]:
    """Create one ignored packet for an external fresh-process comparison."""

    if arm != FROZEN_REPLAY_ARM:
        raise ValueError("generation replay is bound to the frozen generic-control arm")
    _require_distinct_paths(
        [raw_packet_path, trainer_output_dir, metadata_path],
        label="fresh-process replay output",
    )
    if raw_packet_path.exists() or trainer_output_dir.exists() or metadata_path.exists():
        raise FileExistsError("fresh-process replay outputs must start unused")
    assert_artifact_path(runtime_paths, raw_packet_path, "fresh-process replay packet")
    assert_artifact_path(runtime_paths, trainer_output_dir, "fresh-process trainer state")
    assert_artifact_path(runtime_paths, metadata_path, "fresh-process replay metadata")
    packet, resource = _single_generation_replay(
        runtime_paths=runtime_paths,
        config_path=config_path,
        packet_path=packet_path,
        manifest_path=manifest_path,
        arm=arm,
        trainer_output_dir=trainer_output_dir,
        expected_entry_cublas_workspace_config=(FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG),
    )
    packet_hash = write_replay_packet_new(raw_packet_path, packet, kind="generation_only")
    process_contract = deterministic_process_contract(runtime_paths)
    metadata: dict[str, object] = {
        "schema_version": REPLAY_RUNTIME_SCHEMA_VERSION,
        "runtime_id": REPLAY_RUNTIME_ID,
        "packet_sha256": packet_hash,
        "process_command_sha256": process_contract.process_command_sha256,
        "deterministic_process_contract_sha256": process_contract.contract_sha256,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "process_instance_sha256": _PROCESS_INSTANCE_SHA256,
        "runtime_path_contract_sha256": runtime_paths.contract_sha256,
        "process_environment_sha256": runtime_paths.process_environment_sha256,
        "process_command_template_sha256": runtime_paths.process_command_template_sha256,
        "raw_packet_path_sha256": _resolved_path_sha256(raw_packet_path),
        "trainer_output_dir_sha256": _resolved_path_sha256(trainer_output_dir),
        "metadata_path_sha256": _resolved_path_sha256(metadata_path),
        "resource_measurement": resource,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    _write_json_new(metadata_path, metadata)
    validate_runtime_paths(runtime_paths)
    return metadata


def combine_fresh_process_replay(
    *,
    runtime_paths: GrpoRuntimePaths,
    packet_paths: Sequence[Path],
    metadata_paths: Sequence[Path],
    summary_path: Path,
) -> dict[str, object]:
    """Require exact replay across three separate process packets."""

    if len(packet_paths) != REPLAY_RUNS or len(metadata_paths) != REPLAY_RUNS:
        raise ValueError("fresh-process replay requires exactly three packets and metadata files")
    validate_runtime_paths(runtime_paths)
    for path in (*packet_paths, *metadata_paths, summary_path):
        assert_artifact_path(runtime_paths, path, "fresh-process replay artifact")
    _require_distinct_paths(
        [*packet_paths, *metadata_paths, summary_path],
        label="fresh-process replay input and output",
    )
    common_hash = compare_fresh_process_packets(packet_paths, expected_kind="generation_only")
    metadata_values: list[dict[str, object]] = []
    expected_metadata_keys = {
        "schema_version",
        "runtime_id",
        "packet_sha256",
        "process_command_sha256",
        "deterministic_process_contract_sha256",
        "process_id",
        "parent_process_id",
        "process_instance_sha256",
        "runtime_path_contract_sha256",
        "process_environment_sha256",
        "process_command_template_sha256",
        "raw_packet_path_sha256",
        "trainer_output_dir_sha256",
        "metadata_path_sha256",
        "resource_measurement",
        "metadata_sha256",
    }
    for index, path in enumerate(metadata_paths):
        value: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("fresh-process metadata must contain an object")
        row = cast(dict[str, object], value)
        if set(row) != expected_metadata_keys:
            raise ValueError("fresh-process metadata fields differ from the frozen schema")
        if row.get("schema_version") != REPLAY_RUNTIME_SCHEMA_VERSION:
            raise ValueError("fresh-process metadata schema version differs")
        declared = _require_sha256(row.get("metadata_sha256"), label="metadata_sha256")
        payload = {key: item for key, item in row.items() if key != "metadata_sha256"}
        if declared != canonical_sha256(payload):
            raise ValueError("fresh-process metadata self-hash differs")
        if row.get("packet_sha256") != common_hash:
            raise ValueError("fresh-process metadata packet hash differs")
        if row.get("runtime_id") != REPLAY_RUNTIME_ID:
            raise ValueError("fresh-process metadata runtime ID differs")
        for field in (
            "packet_sha256",
            "process_command_sha256",
            "deterministic_process_contract_sha256",
            "process_instance_sha256",
            "runtime_path_contract_sha256",
            "process_environment_sha256",
            "process_command_template_sha256",
            "raw_packet_path_sha256",
            "trainer_output_dir_sha256",
            "metadata_path_sha256",
        ):
            _require_sha256(row.get(field), label=field)
        if not isinstance(row.get("resource_measurement"), Mapping):
            raise ValueError("fresh-process resource measurement must contain an object")
        expected_runtime_fields = {
            "runtime_path_contract_sha256": runtime_paths.contract_sha256,
            "process_environment_sha256": runtime_paths.process_environment_sha256,
            "process_command_template_sha256": runtime_paths.process_command_template_sha256,
        }
        if any(row.get(key) != value for key, value in expected_runtime_fields.items()):
            raise RuntimeError(
                "fresh-process runtime path, environment, or command template differs"
            )
        process_id = row.get("process_id")
        parent_process_id = row.get("parent_process_id")
        if (
            isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or process_id <= 0
            or isinstance(parent_process_id, bool)
            or not isinstance(parent_process_id, int)
            or parent_process_id <= 0
        ):
            raise ValueError("fresh-process metadata contains an invalid process identity")
        if row.get("raw_packet_path_sha256") != _resolved_path_sha256(packet_paths[index]):
            raise ValueError("fresh-process packet path binding differs")
        if row.get("metadata_path_sha256") != _resolved_path_sha256(path):
            raise ValueError("fresh-process metadata path binding differs")
        metadata_values.append(row)
    process_ids = [cast(int, value["process_id"]) for value in metadata_values]
    process_instances = [str(value["process_instance_sha256"]) for value in metadata_values]
    trainer_paths = [str(value["trainer_output_dir_sha256"]) for value in metadata_values]
    process_commands = [str(value["process_command_sha256"]) for value in metadata_values]
    if len(set(process_ids)) != REPLAY_RUNS or len(set(process_instances)) != REPLAY_RUNS:
        raise RuntimeError("fresh-process replay did not use three distinct processes")
    if len(set(trainer_paths)) != REPLAY_RUNS:
        raise RuntimeError("fresh-process replay trainer paths are not distinct")
    if len(set(process_commands)) != REPLAY_RUNS:
        raise RuntimeError("fresh-process replay commands do not bind three unique output paths")
    summary: dict[str, object] = {
        "schema_version": REPLAY_RUNTIME_SCHEMA_VERSION,
        "runtime_id": REPLAY_RUNTIME_ID,
        "replay_kind": "fresh_process_generation",
        "processes": REPLAY_RUNS,
        "groups_per_process": GENERATION_GROUPS,
        "completions_per_process": GENERATION_COMPLETIONS,
        "common_packet_sha256": common_hash,
        "process_ids": process_ids,
        "process_instance_sha256s": process_instances,
        "distinct_processes_verified": True,
        "distinct_output_paths_verified": True,
        "packet_file_sha256s": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in packet_paths
        ],
        "process_command_sha256s": [value["process_command_sha256"] for value in metadata_values],
        "runtime_paths": runtime_paths.evidence(),
        "process_metadata_sha256s": [value["metadata_sha256"] for value in metadata_values],
        "resource_measurements": [value["resource_measurement"] for value in metadata_values],
        "exact_replay_passed": True,
        "prompts_or_completions_in_summary": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json_new(summary_path, summary)
    validate_runtime_paths(runtime_paths)
    return summary


def _common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-paths", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=(FROZEN_REPLAY_ARM,), required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    same = subparsers.add_parser("same-process")
    _common_run_arguments(same)
    same.add_argument("--raw-directory", type=Path, required=True)
    same.add_argument("--summary", type=Path, required=True)
    one = subparsers.add_parser("one-process")
    _common_run_arguments(one)
    one.add_argument("--raw-packet", type=Path, required=True)
    one.add_argument("--trainer-output", type=Path, required=True)
    one.add_argument("--metadata", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--runtime-paths", type=Path, required=True)
    combine.add_argument("--packets", type=Path, nargs=REPLAY_RUNS, required=True)
    combine.add_argument("--metadata", type=Path, nargs=REPLAY_RUNS, required=True)
    combine.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime_paths = load_runtime_paths(args.runtime_paths)
    if args.command == "same-process":
        result = run_same_process_replay(
            runtime_paths=runtime_paths,
            config_path=args.config,
            packet_path=args.packet,
            manifest_path=args.manifest,
            arm=cast(Arm, args.arm),
            raw_directory=args.raw_directory,
            summary_path=args.summary,
        )
    elif args.command == "one-process":
        result = run_one_fresh_process_packet(
            runtime_paths=runtime_paths,
            config_path=args.config,
            packet_path=args.packet,
            manifest_path=args.manifest,
            arm=cast(Arm, args.arm),
            raw_packet_path=args.raw_packet,
            trainer_output_dir=args.trainer_output,
            metadata_path=args.metadata,
        )
    else:
        result = combine_fresh_process_replay(
            runtime_paths=runtime_paths,
            packet_paths=args.packets,
            metadata_paths=args.metadata,
            summary_path=args.summary,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
