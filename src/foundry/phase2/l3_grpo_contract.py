"""Milestone 14A starting-state, recipe, and publication contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_reference import reference_mechanism_contract
from foundry.phase2.l3_grpo_reward import (
    calibrate_reward_contract,
    reward_configuration_sha256,
    reward_contract_sha256,
    reward_fixture_sha256,
    reward_implementation_sha256,
)
from foundry.phase2.l3_grpo_schedule import (
    CHECKPOINT_STEPS,
    COMPLETIONS_PER_ARM,
    COMPLETIONS_PER_GROUP,
    DATASET_SHA256,
    GROUPS_PER_ARM,
    OPTIMIZER_STEPS,
    REPLAY_GROUPS_PER_ARM,
    SCHEDULE_ID,
    TASK_GROUPS_PER_ARM,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

MILESTONE_ID = "foundry-milestone14a-l3-verifier-grpo-v1"
STARTING_COMMIT = "97f8cdb8fb6bba4260365f514a8ecc83a8f069b5"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_MANIFEST_SHA256 = "02bff45c336c3650abe518a94accf4c321f0116678a99c2f56a131cf2eade34d"
ENVIRONMENT_EVIDENCE_SHA256 = "9244dd7aa9d4d5138ef01f1b4fb20b911fc390e034e5704ded4ba8fcd967244b"
COMBINED_CHILD_ENVIRONMENT_SHA256 = (
    "1d402ec0cb661adeb50a3d3bd9510895f3f9068cbb393fb381565a5670de995b"
)
INTERPRETER_SHA256 = "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
PACKAGE_INVENTORY_SHA256 = "2d4dbf699b73b53206d96687f1381ec22dac8a2d1575b0a43791627b9b43b2c8"
STARTING_ADAPTER_SHA256 = {
    "generic": "67c6f1dd34c0fa1ddebb354dfe14c43e61c48fdd90c687ba1a9290d2401479cd",
    "targeted": "4e195ff2cb32c4faa6858915b95507862c911bb2eb853b060717416d825df91d",
}
STARTING_ADAPTER_WEIGHT_SHA256 = {
    "generic": "6d7e5cf18386504cb660d20b82a1a8427ca47ff167c255ae57d437315ff4969c",
    "targeted": "f86e3b82d06abc2d800331a9d35732e1d42f2fc29cbf3a2314eafa53fddb2d02",
}
ADAPTER_CONFIG_FILE_SHA256 = "e116598015bb55f28b622dca11325946108156d327ae3c694ff950a5c7c1b964"
TRAINABLE_INVENTORY_SHA256 = "98ecbfd20ca9b483ff8eeeecfaa1f18cd7c7503babfcc3e2080693808a15461f"
TRAINABLE_TENSORS = 112
TRAINABLE_PARAMETERS = 1_089_536
LAYERS = tuple(range(14, 28))
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "20260720",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}

GRPO_RECIPE: dict[str, object] = {
    "schema_version": 1,
    "recipe_id": "foundry-l3-verifier-grpo-recipe-v1",
    "beta": 0.04,
    "optimizer_steps": OPTIMIZER_STEPS,
    "groups_per_arm": GROUPS_PER_ARM,
    "task_groups_per_arm": TASK_GROUPS_PER_ARM,
    "replay_groups_per_arm": REPLAY_GROUPS_PER_ARM,
    "generations_per_group": COMPLETIONS_PER_GROUP,
    "completions_per_arm": COMPLETIONS_PER_ARM,
    "checkpoints": list(CHECKPOINT_STEPS),
    "max_prompt_length": 512,
    "max_completion_length": 256,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.000001,
    "optimizer": "paged_adamw_8bit",
    "scheduler": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.0,
    "max_gradient_norm": 1.0,
    "epsilon": 0.2,
    "policy_iterations": 1,
    "reward_scaling": False,
    "loss_form": "dr_grpo",
    "mask_truncated_completions": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "seed": 20260720,
    "use_vllm": False,
    "cpu_offload": False,
    "gradient_checkpointing": False,
    "disable_dropout_during_grpo": True,
    "strict_determinism_outside_generation": True,
    "generation_contract": "foundry-warning-only-top-p-replay-v1",
    "reference_policy": "frozen_corresponding_l3_checkpoint64_adapter",
    "reference_full_base_count": 1,
}
GRPO_RECIPE_SHA256 = canonical_sha256(GRPO_RECIPE)

FIXED_LIBRARY_NOTICE_CLASSES: tuple[dict[str, str], ...] = (
    {
        "class_id": "transformers-qwen2-sliding-window-sdpa-unimplemented-v1",
        "required_substring": "Sliding Window Attention is enabled but not implemented for `sdpa`",
    },
    {
        "class_id": "transformers-peft-empty-label-names-informational-v1",
        "required_substring": "No label_names provided for model class",
    },
    {
        "class_id": "transformers-dynamic-cache-torch-export-version-uncertainty-v1",
        "required_substring": "DynamicCache",
    },
)
FIXED_LIBRARY_NOTICE_CONTRACT_SHA256 = canonical_sha256(
    {
        "classes": list(FIXED_LIBRARY_NOTICE_CLASSES),
        "broad_warning_suppression": False,
        "generation_warning_separate": True,
        "classes_may_be_absent_when_their_library_path_is_not_exercised": True,
    }
)


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify(value: dict[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _model_path(root: Path) -> Path:
    return (
        root
        / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
        / f"snapshots/{MODEL_REVISION}"
    )


def model_manifest(root: Path) -> dict[str, object]:
    """Reconstruct the exact local offline snapshot identity."""

    model = _model_path(root)
    rows = [
        {
            "name": path.relative_to(model).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(model.rglob("*"))
        if path.is_file()
    ]
    value = canonical_sha256(rows)
    if value != MODEL_MANIFEST_SHA256:
        raise ValueError("offline model snapshot manifest differs")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "file_count": len(rows),
        "bytes": sum(cast(int, row["bytes"]) for row in rows),
        "manifest_sha256": value,
    }


def adapter_path(root: Path, arm: str) -> Path:
    if arm not in STARTING_ADAPTER_SHA256:
        raise ValueError("arm is outside the frozen L3 pair")
    return (
        root
        / f"results/raw/phase2_vetted_corpus/milestone13e/full/{arm}"
        / "training/checkpoint-64/adapter"
    )


def _starting_adapter(root: Path, arm: str) -> dict[str, object]:
    path = adapter_path(root, arm)
    actual_directory = directory_sha256(path)
    if actual_directory != STARTING_ADAPTER_SHA256[arm]:
        raise ValueError(f"{arm} starting adapter directory hash differs")
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if (
        file_sha256(config_path) != ADAPTER_CONFIG_FILE_SHA256
        or file_sha256(weights_path) != STARTING_ADAPTER_WEIGHT_SHA256[arm]
    ):
        raise ValueError(f"{arm} starting adapter file identity differs")
    config = _read(config_path)
    if (
        config.get("r") != 8
        or config.get("lora_alpha") != 16
        or config.get("lora_dropout") != 0.05
        or config.get("bias") != "none"
        or config.get("layers_to_transform") != list(LAYERS)
        or set(cast(list[str], config.get("target_modules"))) != set(PROJECTIONS)
        or config.get("layers_pattern") != "layers"
    ):
        raise ValueError(f"{arm} starting adapter configuration differs")
    training = _read(path.parents[2] / "training_summary.json")
    _verify(training, "result_sha256")
    inventory = cast(dict[str, object], training["trainable_inventory"])
    if (
        inventory.get("tensor_inventory_sha256") != TRAINABLE_INVENTORY_SHA256
        or inventory.get("trainable_tensor_count") != TRAINABLE_TENSORS
        or inventory.get("trainable_parameter_count") != TRAINABLE_PARAMETERS
        or inventory.get("layer_indices") != list(LAYERS)
    ):
        raise ValueError(f"{arm} starting tensor inventory differs")
    checkpoint = cast(dict[str, Any], training["checkpoints"])["64"]
    if checkpoint["adapter_sha256"] != STARTING_ADAPTER_SHA256[arm]:
        raise ValueError(f"{arm} checkpoint summary adapter identity differs")
    return {
        "arm": arm,
        "directory_sha256": actual_directory,
        "adapter_config_file_sha256": file_sha256(config_path),
        "adapter_weights_file_sha256": file_sha256(weights_path),
        "adapter_configuration": {
            "rank": config["r"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
            "bias": config["bias"],
            "layers": config["layers_to_transform"],
            "target_modules": sorted(cast(list[str], config["target_modules"])),
            "layers_pattern": config["layers_pattern"],
        },
        "trainable_inventory": inventory,
        "checkpoint_summary": checkpoint,
        "training_result_sha256": training["result_sha256"],
    }


def build_starting_state(
    root: Path,
    *,
    allow_milestone_implementation_changes: bool = False,
) -> dict[str, object]:
    """Fail closed unless every published Milestone 13E input agrees."""

    repository = root.resolve()
    if repository != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("Milestone 14A is attached to the wrong repository")
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    main = _git(root, "rev-parse", "main")
    origin = _git(root, "rev-parse", "origin/main")
    divergence = _git(root, "rev-list", "--left-right", "--count", "main...origin/main")
    dirty = _git(root, "status", "--porcelain")
    if allow_milestone_implementation_changes:
        allowed_prefixes = (
            "src/foundry/phase2/l3_grpo_",
            "tests/unit/phase2/test_l3_grpo_",
        )
        dirty_paths = [line[3:].replace("\\", "/") for line in dirty.splitlines() if len(line) >= 4]
        if any(not path.startswith(allowed_prefixes) for path in dirty_paths):
            raise RuntimeError("preparation found changes outside Milestone 14A implementation")
    if (
        branch != "main"
        or head != STARTING_COMMIT
        or main != STARTING_COMMIT
        or origin != STARTING_COMMIT
        or divergence.split() != ["0", "0"]
        or (dirty and not allow_milestone_implementation_changes)
    ):
        raise RuntimeError("repository starting state differs from Milestone 14A authorization")
    environment = _read(root / "results/phase2_vetted_corpus/windows_operational_environment.json")
    if (
        environment.get("environment_evidence_sha256") != ENVIRONMENT_EVIDENCE_SHA256
        or environment.get("combined_child_environment_sha256") != COMBINED_CHILD_ENVIRONMENT_SHA256
        or environment.get("deterministic_projection_sha256")
        != canonical_sha256(DETERMINISTIC_ENVIRONMENT)
    ):
        raise ValueError("published Windows operational environment differs")
    for name in (".venv", ".venv-training"):
        executable = root / name / "Scripts/python.exe"
        if file_sha256(executable) != INTERPRETER_SHA256:
            raise ValueError(f"{name} interpreter hash differs")
    dataset = _read(root / "results/phase2_vetted_corpus/dataset_summary.json")
    if dataset.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("published vetted dataset identity differs")
    selection = _read(root / "results/phase2_vetted_corpus/milestone13e_full_selection.json")
    holdout = _read(root / "results/phase2_vetted_corpus/milestone13e_holdout_v2_decision.json")
    gsm1k = _read(root / "results/phase2_vetted_corpus/milestone13e_gsm1k_decision.json")
    _verify(selection, "full_selection_sha256")
    _verify(holdout, "holdout_decision_sha256")
    _verify(gsm1k, "gsm1k_decision_sha256")
    if (
        selection.get("selected_scope") != "L3"
        or selection.get("selected_checkpoint") != 64
        or selection.get("selected_adapter_sha256_by_arm") != STARTING_ADAPTER_SHA256
        or holdout.get("both_arms_pass") is not True
        or holdout.get("adapter_evaluations") != 2
        or gsm1k.get("adapter_evaluations") != 2
        or any(
            packet.get("sealed_final_accessed") is not False
            for packet in (dataset, selection, holdout, gsm1k)
        )
    ):
        raise ValueError("published L3 selection or sealed-boundary evidence differs")
    development = cast(list[dict[str, Any]], selection["checkpoint_results"])[0]["by_arm"]
    starting_state: dict[str, object] = {
        "schema_version": 1,
        "milestone_id": MILESTONE_ID,
        "repository": str(repository),
        "branch": branch,
        "starting_commit": head,
        "origin_main": origin,
        "ahead_behind": [0, 0],
        "worktree_clean": True,
        "preflight_clean_observed_before_milestone_implementation": True,
        "preparation_changes_limited_to_milestone14a": (allow_milestone_implementation_changes),
        "model": model_manifest(root),
        "environment_evidence_sha256": ENVIRONMENT_EVIDENCE_SHA256,
        "combined_child_environment_sha256": COMBINED_CHILD_ENVIRONMENT_SHA256,
        "deterministic_environment_sha256": canonical_sha256(DETERMINISTIC_ENVIRONMENT),
        "interpreter_sha256": INTERPRETER_SHA256,
        "package_inventory_sha256": PACKAGE_INVENTORY_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "starting_adapters": {arm: _starting_adapter(root, arm) for arm in ("generic", "targeted")},
        "development_retention": {
            arm: cast(dict[str, Any], development[arm])["development_retention"]
            for arm in ("generic", "targeted")
        },
        "holdout_v2": holdout["arms"],
        "frozen_gsm1k": {
            "base": gsm1k["base"],
            "generic": gsm1k["generic"],
            "targeted": gsm1k["targeted"],
        },
        "selected_scope": "L3",
        "selected_checkpoint": 64,
        "sealed_final_accessed": False,
        "no_recorded_sealed_access_after_milestone13e": True,
        "model_processes_at_preflight": 0,
        "milestone14_artifacts_at_preflight": 0,
    }
    starting_state["starting_state_sha256"] = canonical_sha256(starting_state)
    return starting_state


def build_experiment_contract(
    starting_state: dict[str, object],
    schedule_summary: dict[str, object],
    implementation_sha256: str,
) -> dict[str, object]:
    """Bind starting evidence, schedules, reward, reference, and recipe."""

    if starting_state.get("starting_state_sha256") != canonical_sha256(
        {key: value for key, value in starting_state.items() if key != "starting_state_sha256"}
    ):
        raise ValueError("starting-state evidence does not reconstruct")
    if schedule_summary.get("schedule_id") != SCHEDULE_ID:
        raise ValueError("paired schedule identity differs")
    if not isinstance(implementation_sha256, str) or len(implementation_sha256) != 64:
        raise ValueError("implementation SHA-256 is required")
    calibration = calibrate_reward_contract()
    reference = reference_mechanism_contract()
    payload: dict[str, object] = {
        "schema_version": 1,
        "milestone_id": MILESTONE_ID,
        "starting_state_sha256": starting_state["starting_state_sha256"],
        "schedule_id": SCHEDULE_ID,
        "paired_schedule_sha256": schedule_summary["paired_schedule_sha256"],
        "implementation_sha256": implementation_sha256,
        "reward": {
            "implementation_sha256": reward_implementation_sha256(),
            "configuration_sha256": reward_configuration_sha256(),
            "fixture_sha256": reward_fixture_sha256(),
            "calibration_sha256": calibration["calibration_sha256"],
            "contract_sha256": reward_contract_sha256(),
        },
        "reference": reference,
        "recipe": GRPO_RECIPE,
        "recipe_sha256": GRPO_RECIPE_SHA256,
        "fixed_library_notice_contract_sha256": (FIXED_LIBRARY_NOTICE_CONTRACT_SHA256),
        "scientific_settings_frozen_before_generation": True,
        "sealed_content_use": 0,
        "gsm1k_training_prompt_use": 0,
        "holdout_v2_training_prompt_use": 0,
    }
    payload["experiment_contract_sha256"] = canonical_sha256(payload)
    return payload


def write_json_new_or_identical(path: Path, value: object) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"existing contract artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
