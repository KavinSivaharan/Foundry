"""Frozen configuration and identity checks for Foundry Cycle 1."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from foundry.phase2.windows_environment import (
    load_frozen_child_environment,
    validate_child_environment,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

CONTROLLER_ID = "foundry-cycle-controller-v1"
CYCLE_ID = "foundry-cycle1-vfbon-sft-v1"
OPTIMIZATION_METHOD = "verifier-filtered-best-of-n-sft-v1"
STAGES = (
    "verify_baseline",
    "diagnose",
    "load_safe_warm_start",
    "generate_candidates",
    "verify_and_select_traces",
    "freeze_training_corpus",
    "compatibility_smoke",
    "train_candidate",
    "select_checkpoint",
    "development_retention",
    "holdout_retention",
    "benchmark",
    "decide",
    "promote_or_reject",
    "publish_trace",
)
REQUIRED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "20260720",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}


class CycleContractError(ValueError):
    """The frozen Cycle 1 contract or one of its identities differs."""


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CycleContractError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CycleContractError(f"{label} must be a list")
    return value


def _require_fields(
    section: dict[str, Any],
    expected: dict[str, object],
    label: str,
) -> None:
    actual = {name: section.get(name) for name in expected}
    if actual != expected:
        raise CycleContractError(f"{label} contract differs")


@dataclass(frozen=True)
class CycleConfig:
    """Validated parsed Cycle 1 configuration."""

    path: Path
    payload: dict[str, Any]
    sha256: str

    @property
    def artifact_root(self) -> Path:
        return Path(str(_mapping(self.payload["roots"], "roots")["artifact_root"]))

    @property
    def source_root(self) -> Path:
        return Path(str(_mapping(self.payload["roots"], "roots")["frozen_source_root"]))

    @property
    def runtime_root(self) -> Path:
        return Path(str(_mapping(self.payload["roots"], "roots")["runtime_root"]))

    @property
    def registry_root(self) -> Path:
        return Path(str(_mapping(self.payload["roots"], "roots")["model_registry_root"]))

    def section(self, name: str) -> dict[str, Any]:
        return _mapping(self.payload[name], name)

    def resolve_artifact(self, relative: str) -> Path:
        path = (self.artifact_root / relative).resolve()
        try:
            path.relative_to(self.artifact_root.resolve())
        except ValueError as error:
            raise CycleContractError("artifact path escapes the primary repository") from error
        return path


def load_cycle_config(path: Path) -> CycleConfig:
    """Load and strictly validate the only authorized Cycle 1 configuration."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = _mapping(raw, "cycle configuration")
    _validate_config(payload)
    return CycleConfig(path.resolve(), payload, canonical_sha256(payload))


def _validate_config(payload: dict[str, Any]) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("controller_id") != CONTROLLER_ID
        or payload.get("cycle_id") != CYCLE_ID
        or payload.get("optimization_method") != OPTIMIZATION_METHOD
    ):
        raise CycleContractError("Cycle 1 controller identity differs")
    roots = _mapping(payload.get("roots"), "roots")
    expected_roots = {
        "artifact_root": r"C:\Users\Admin\Projects\Foundry",
        "frozen_source_root": r"C:\Users\Admin\Projects\Foundry-cycle1-frozen",
        "runtime_root": r"C:\Users\Admin\Projects\Foundry-cycle1-runtime",
        "model_registry_root": (r"C:\Users\Admin\Projects\Foundry-cycle1-runtime\model_registry"),
    }
    if {name: str(roots.get(name)) for name in expected_roots} != expected_roots:
        raise CycleContractError("Cycle 1 root contract differs")
    stages = _list(payload.get("stages"), "stages")
    if tuple(stages) != STAGES:
        raise CycleContractError("Cycle 1 state-machine stages differ")
    _require_fields(
        _mapping(payload.get("starting_state"), "starting_state"),
        {
            "repository_commit": "865143558e581a8561bb96848682f43b3385c0bd",
            "repository_tree": "88f8e92173207e09585ebcb2d31994a86b051563",
            "branch": "main",
        },
        "Cycle 1 starting state",
    )
    _require_fields(
        _mapping(payload.get("model"), "model"),
        {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "snapshot_manifest_sha256": (
                "02bff45c336c3650abe518a94accf4c321f0116678a99c2f56a131cf2eade34d"
            ),
        },
        "Cycle 1 model",
    )
    _require_fields(
        _mapping(payload.get("dataset"), "dataset"),
        {
            "identity_sha256": ("ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"),
            "targeted_training_sha256": (
                "3322a9de2e56d24e170129112a8973dd45e5c6575969da52e5cac44f23cea215"
            ),
            "targeted_validation_sha256": (
                "4329bc2e4a09f0001c128f24ba99f32f60c3031ceae1fbbd639ac9c83309c74a"
            ),
            "training_records": 180,
            "validation_records": 20,
            "selected_failure_families": [
                "constraint_distribution_or_discrete_reasoning",
                "multi_step_bookkeeping_or_omission",
                "rate_ratio_percentage_or_average",
            ],
        },
        "Cycle 1 dataset",
    )
    _require_fields(
        _mapping(payload.get("warm_start"), "warm_start"),
        {
            "arm": "targeted",
            "scope": "L3",
            "checkpoint": 64,
            "adapter_sha256": ("4e195ff2cb32c4faa6858915b95507862c911bb2eb853b060717416d825df91d"),
            "generic_adapter_sha256": (
                "67c6f1dd34c0fa1ddebb354dfe14c43e61c48fdd90c687ba1a9290d2401479cd"
            ),
            "generic_gsm1k_correct": 517,
            "targeted_gsm1k_correct": 519,
            "generic_holdout_preserved": 315,
            "targeted_holdout_preserved": 315,
            "holdout_total": 317,
        },
        "Cycle 1 warm start",
    )
    generation = _mapping(payload.get("generation"), "generation")
    if (
        generation.get("completions_per_prompt") != 8
        or generation.get("max_new_tokens") != 256
        or generation.get("temperature") != 0.8
        or generation.get("top_p") != 0.95
        or generation.get("top_k") != 50
        or generation.get("seed") != 20260720
        or generation.get("vllm") is not False
        or generation.get("cpu_offload") is not False
    ):
        raise CycleContractError("Cycle 1 generation settings differ")
    _require_fields(
        _mapping(payload.get("selection"), "selection"),
        {
            "maximum_selected_per_prompt": 1,
            "hierarchy": [
                "eligible",
                "shortest_completion_token_count",
                "lowest_normalized_completion_sha256",
                "lowest_raw_completion_sha256",
            ],
            "required_final_answer_prefix": "Final answer:",
            "token_count_minimum": 1,
            "token_count_maximum": 256,
            "prohibit_prompt_echo": True,
            "prohibit_question_generation": True,
            "prohibit_malformed_output": True,
            "prohibit_truncation": True,
            "prohibit_backend_error": True,
        },
        "Cycle 1 selection",
    )
    _require_fields(
        _mapping(payload.get("coverage_gate"), "coverage_gate"),
        {
            "all_prompts_processed_once": True,
            "attempts_per_prompt": 8,
            "backend_failures": 0,
            "verifier_disagreements": 0,
            "minimum_overall_fraction": 0.5,
            "minimum_each_family_fraction": 0.5,
            "minimum_changed_selected_traces": 60,
            "selected_prompt_echo": 0,
            "selected_question_generation": 0,
            "selected_malformed": 0,
            "selected_answer_disagreements": 0,
            "gsm1k_exact_overlap": 0,
            "gsm1k_normalized_exact_overlap": 0,
            "failure_classification": "verifier_filtered_optimizer_signal_insufficient",
        },
        "Cycle 1 verifier coverage",
    )
    training = _mapping(payload.get("training"), "training")
    if (
        training.get("rank") != 8
        or training.get("alpha") != 16
        or training.get("dropout") != 0.05
        or training.get("bias") != "none"
        or training.get("target_modules") != ["q_proj", "k_proj", "v_proj", "o_proj"]
        or training.get("adapted_layers") != list(range(14, 28))
        or training.get("optimizer") != "PagedAdamW8bit"
        or training.get("learning_rate") != 5e-6
        or training.get("optimizer_steps") != 32
        or training.get("checkpoints") != [8, 16, 32]
        or training.get("scheduler") != "cosine"
        or training.get("warmup_ratio") != 0.05
        or training.get("warmup_steps") != 2
        or training.get("seed") != 20260720
        or training.get("gradient_checkpointing") is not False
        or training.get("cpu_offload") is not False
        or training.get("weight_decay") != 0.0
        or training.get("max_sequence_length") != 512
        or training.get("gradient_clip_norm") != 1.0
        or training.get("offline_required") is not True
    ):
        raise CycleContractError("Cycle 1 continuation-training recipe differs")
    corpus = _mapping(payload.get("corpus"), "corpus")
    if (
        corpus.get("total_assistant_tokens") != 32_000
        or corpus.get("task_assistant_tokens") != 24_000
        or corpus.get("replay_assistant_tokens") != 8_000
        or corpus.get("selected_prompt_original_fraction") != 0.5
        or corpus.get("selected_prompt_trace_fraction") != 0.5
        or corpus.get("fallback_prompt_original_fraction") != 1.0
        or corpus.get("task_fraction") != 0.75
        or corpus.get("replay_fraction") != 0.25
        or corpus.get("replay_file_sha256")
        != "a9f25258d23f05a785dfea9f8ae0e05a246b52c9798a0d10e683fdc4e01a87f6"
        or corpus.get("source_schedule_sha256")
        != "88c5378cac7efe927b29d3f421d97777cd6d917187c71c8388b60bbe7b57e259"
        or corpus.get("source_schedule_32_step_prefix_sha256")
        != "40214539356fb1f219e6d7bce57dc4a7db064dca030515eb4aab7008f12b07a7"
    ):
        raise CycleContractError("Cycle 1 corpus contract differs")
    retention = _mapping(payload.get("retention"), "retention")
    _require_fields(
        _mapping(retention.get("gates"), "retention.gates"),
        {
            "minimum_overall_preservation": 0.9,
            "minimum_arithmetic_preservation": 0.9,
            "minimum_format_preservation": 0.9,
            "minimum_instruction_preservation": 0.9,
            "minimum_wilson_lower_bound": 0.85,
            "maximum_prompt_echo_rate": 0.02,
            "maximum_question_generation": 0,
            "maximum_backend_failures": 0,
            "maximum_failure_family_count": 3,
        },
        "Cycle 1 retention gate",
    )
    holdout = _mapping(retention.get("holdout_v2"), "retention.holdout_v2")
    _require_fields(
        holdout,
        {
            "subset_sha256": ("a23b1014d92e9f98b74da3b29913a430bdaebf8e07a16b31b4c3dcc831f1f420"),
            "starting_preserved": 315,
            "total": 317,
        },
        "Cycle 1 holdout",
    )
    benchmark = _mapping(payload.get("benchmark"), "benchmark")
    _require_fields(
        benchmark,
        {
            "frozen_manifest_sha256": (
                "5e810d3ab644bef1d43c598a14a6164ba6464b27fde50e92a2f241816ce87897"
            ),
            "untouched_base_correct": 521,
            "generic_l3_correct": 517,
            "targeted_l3_correct": 519,
            "total": 814,
            "minimum_candidate_correct": 522,
            "minimum_extractability": 0.9138,
            "maximum_backend_failures": 0,
            "untargeted_population": (
                "all_manifest_ids_excluding_published_base_failure_ids_in_"
                "selected_reasoning_categories"
            ),
            "maximum_untargeted_aggregate_decline_percentage_points": 2.0,
        },
        "Cycle 1 promotion benchmark",
    )
    _require_fields(
        _mapping(payload.get("smoke"), "smoke"),
        {
            "prompt_count": 4,
            "completions_per_prompt": 8,
            "training_records": 4,
            "optimizer_steps": 2,
            "scheduler_horizon_steps": 32,
            "warmup_steps": 2,
            "trials": 2,
            "require_exact_duplicate": True,
        },
        "Cycle 1 compatibility smoke",
    )
    environment = _mapping(payload.get("environment"), "environment")
    _require_fields(
        environment,
        {
            "general_interpreter": r"C:\Users\Admin\Projects\Foundry\.venv\Scripts\python.exe",
            "training_interpreter": (
                r"C:\Users\Admin\Projects\Foundry\.venv-training\Scripts\python.exe"
            ),
            "interpreter_sha256": (
                "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
            ),
            "contract_id": "foundry-vetted-qlora-windows-operational-env-v2",
            "operational_environment_file_sha256": (
                "7d5094347c4446a8ce925c01adfe45813a451105aa229ae4acde1399a4ab6089"
            ),
            "tracked_environment_file_sha256": (
                "9366db7c33448a3d82a84fb1bde5c83d354b629520a8ca7bef1690d64a749012"
            ),
            "environment_evidence_sha256": (
                "9244dd7aa9d4d5138ef01f1b4fb20b911fc390e034e5704ded4ba8fcd967244b"
            ),
            "combined_child_environment_sha256": (
                "1d402ec0cb661adeb50a3d3bd9510895f3f9068cbb393fb381565a5670de995b"
            ),
            "deterministic_projection_sha256": (
                "241e3e69f58d48a36c9411528aa129796319d23e61188b56cbcb4695e7fdf189"
            ),
        },
        "Cycle 1 operational environment",
    )
    required = _mapping(environment.get("required"), "environment.required")
    if {str(key): str(value) for key, value in required.items()} != REQUIRED_ENVIRONMENT:
        raise CycleContractError("Cycle 1 deterministic environment differs")


def normalized_completion(value: str) -> str:
    """Return the exact frozen normalization used only for trace tie-breaking."""

    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(normalized.split()).strip()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_subseed(cycle_id: str, source_id: str, completion_index: int) -> int:
    """Derive the frozen zero-based prompt/completion seed."""

    if cycle_id != CYCLE_ID or completion_index < 0:
        raise CycleContractError("prompt sub-seed inputs are outside Cycle 1")
    digest = hashlib.sha256(f"{cycle_id}{source_id}{completion_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def frozen_child_environment(config: CycleConfig) -> dict[str, str]:
    """Reconstruct the exact audited 37-field Windows child environment."""

    contract = config.section("environment")
    raw_path = validate_file_identity(
        config,
        str(contract["operational_environment_relative_path"]),
        str(contract["operational_environment_file_sha256"]),
    )
    tracked_path = validate_file_identity(
        config,
        str(contract["tracked_environment_relative_path"]),
        str(contract["tracked_environment_file_sha256"]),
    )
    child = load_frozen_child_environment(
        raw_environment_path=raw_path,
        tracked_evidence_path=tracked_path,
    )
    evidence = read_json_object(tracked_path)
    if (
        evidence["contract_id"] != contract["contract_id"]
        or evidence["environment_evidence_sha256"] != contract["environment_evidence_sha256"]
        or evidence["combined_child_environment_sha256"]
        != contract["combined_child_environment_sha256"]
        or evidence["deterministic_projection_sha256"]
        != contract["deterministic_projection_sha256"]
    ):
        raise CycleContractError("frozen Windows environment evidence differs")
    return child


def validate_process_environment(
    environment: Mapping[str, str] | None = None,
    *,
    config: CycleConfig | None = None,
) -> None:
    """Fail unless the process inherited the exact audited Windows environment."""

    actual = os.environ if environment is None else environment
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in REQUIRED_ENVIRONMENT.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise CycleContractError(f"deterministic process environment differs: {mismatches}")
    if sys.flags.hash_randomization != 1:
        raise CycleContractError("Python hash randomization is disabled")
    if config is not None:
        expected = frozen_child_environment(config)
        normalized_actual = {name.upper(): value for name, value in actual.items()}
        if len(normalized_actual) != len(actual):
            raise CycleContractError("duplicate case-insensitive environment variable")
        if normalized_actual != expected:
            raise CycleContractError("process environment differs from frozen 37-field contract")
        evidence_path = config.resolve_artifact(
            str(config.section("environment")["tracked_environment_relative_path"])
        )
        validate_child_environment(normalized_actual, read_json_object(evidence_path))


def git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.stdout.strip()


def validate_frozen_source(config: CycleConfig) -> dict[str, str]:
    """Validate detached, clean, synchronized controller source."""

    root = config.source_root.resolve()
    if root != Path(r"C:\Users\Admin\Projects\Foundry-cycle1-frozen").resolve():
        raise CycleContractError("controller source root differs")
    if git_output(root, "branch", "--show-current"):
        raise CycleContractError("controller source worktree is not detached")
    status = git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise CycleContractError("controller source worktree is not clean")
    commit = git_output(root, "rev-parse", "HEAD")
    tree = git_output(root, "rev-parse", "HEAD^{tree}")
    return {"commit": commit, "tree": tree, "status": "clean", "import_root": str(root / "src")}


def validate_file_identity(config: CycleConfig, relative: str, expected: str) -> Path:
    path = config.resolve_artifact(relative)
    if not path.is_file() or file_sha256(path) != expected:
        raise CycleContractError(f"frozen file identity differs: {relative}")
    return path


def content_free_projection(value: object) -> object:
    """Remove prompt/model-output fields before state or publication persistence."""

    forbidden = {
        "question",
        "prompt",
        "reference",
        "canonical_answer",
        "assistant_completion",
        "response",
        "completion",
        "generated_text",
        "token_ids",
        "input_ids",
        "labels",
    }
    if isinstance(value, dict):
        return {
            str(key): content_free_projection(item)
            for key, item in value.items()
            if str(key) not in forbidden
        }
    if isinstance(value, list):
        return [content_free_projection(item) for item in value]
    return value


def read_json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def verified_payload(path: Path, hash_field: str) -> dict[str, Any]:
    """Load one self-hashed object and require exact reconstruction."""

    value = read_json_object(path)
    supplied = value.get(hash_field)
    payload = {key: item for key, item in value.items() if key != hash_field}
    if supplied != canonical_sha256(payload):
        raise CycleContractError(f"{path.name} does not reconstruct")
    return value
