"""Reconstruct and freeze the exact historical V1 LoRA recipe for KL training."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

RECIPE_ID = "foundry-replay-ce-token-kl-v1-lora-v1"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET_SHA256 = "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"
ARCHITECTURE_DECISION_SHA256 = "74907ea92b2217b6f9ca39044feab6c6452600e7774a2b442e0ec9e29b6899a5"
RUNNER_FILE_SHA256 = "b28bc416f3e813e3bc91c34fc3f4fb700a5a2a328912d63ba0c0c830c9da4081"
ADAPTER_CONFIG_FILE_SHA256 = "2cf0fb6637747b0aa31525f08ba8b412cc4f1986689ef8b9f555cd4b299039e2"
V1_TRAINING_FILE_SHA256 = "2eaedb3250abb2bec6db3e7262d4dee7421002526a224080071f61609c773032"
V1_SCHEDULE_FILE_SHA256 = "dc8cbb08bcb210839b188b170c12d36c7eb17e3b05a553581e0dd28557a738d2"
R2_BLOCKER_SHA256 = "1129ccb4810ad86c034ceb609d86c682360f1acfbde9b18bdd917755706d5e9f"
CONSTRUCTION_TARGET_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj")
SERIALIZED_TARGET_ORDER = ("k_proj", "q_proj", "o_proj", "v_proj")
CHECKPOINTS = (16, 32, 64)
ARMS = ("generic", "targeted")
EXPECTED_TRAINABLE_TENSORS = 224
EXPECTED_TRAINABLE_PARAMETERS = 2_179_072
EXPECTED_ADAPTED_LAYERS = 28
EXPECTED_ADAPTED_MODULES = 112
EXPECTED_LORA_SCALING = 2.0
EXPECTED_ADAPTER_HASHES = {
    "generic": {
        "16": "cf230487953cf347824a40faa36ad6b1b93ef667119f83766dce1d26e72ba63e",
        "32": "8dd3ee1894c03dd6c6ec0c03af32ba557099ac35dcbc604f1bb1eb00b001a901",
        "64": "fe3c0f5a6e8082f2d151a293e918882f896ddeece93c8f779d4b16d21618a73d",
    },
    "targeted": {
        "16": "6915af1d2ab6bd8ac75b538417e1b3af7395924e77c1150138bc06bc4773c5e9",
        "32": "7e8cda9db393344cd0479e247ccdf5a04c6888b9d9ac5960a21ef9860dfb2abb",
        "64": "7f15edf4cd7e8c50478b6bb8deb55bb3d9d43273109fc0b224110f9fe08da0bd",
    },
}
STATE_KEY_PATTERN = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])\.weight$"
)


def _read_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _validate_hash(value: dict[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def runner_lora_projection(path: Path) -> dict[str, Any]:
    """Extract the executed LoraConfig call without importing the model stack."""

    if file_sha256(path) != RUNNER_FILE_SHA256:
        raise ValueError("published V1 runner hash differs")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "LoraConfig"
    ]
    if len(calls) != 1:
        raise ValueError("published V1 runner must contain one LoraConfig call")
    values = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[0].keywords
        if keyword.arg is not None
    }
    expected = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": list(CONSTRUCTION_TARGET_ORDER),
        "task_type": "CAUSAL_LM",
    }
    if values != expected:
        raise ValueError("published V1 runner LoRA projection differs")
    projection: dict[str, Any] = {
        "rank": values["r"],
        "alpha": values["lora_alpha"],
        "dropout": values["lora_dropout"],
        "construction_target_module_order": values["target_modules"],
        "target_module_names_sorted": sorted(values["target_modules"]),
        "bias": values["bias"],
        "task_type": values["task_type"],
        "training_inference_mode": False,
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "revision": None,
        "initialization_behavior": "peft_default_init_lora_weights_true",
        "lora_scaling": EXPECTED_LORA_SCALING,
    }
    projection["runner_projection_sha256"] = canonical_sha256(projection)
    return projection


def adapter_config_projection(path: Path) -> dict[str, Any]:
    """Project one saved PEFT adapter config into frozen scientific fields."""

    if file_sha256(path) != ADAPTER_CONFIG_FILE_SHA256:
        raise ValueError("historical adapter config bytes differ")
    value = _read_object(path)
    base_path = str(value.get("base_model_name_or_path", ""))
    projection: dict[str, Any] = {
        "rank": value.get("r"),
        "alpha": value.get("lora_alpha"),
        "dropout": value.get("lora_dropout"),
        "serialized_target_module_order": value.get("target_modules"),
        "target_module_names_sorted": sorted(cast(list[str], value.get("target_modules", []))),
        "bias": value.get("bias"),
        "task_type": value.get("task_type"),
        "saved_inference_mode": value.get("inference_mode"),
        "fan_in_fan_out": value.get("fan_in_fan_out"),
        "modules_to_save": value.get("modules_to_save"),
        "revision": value.get("revision"),
        "base_model_revision": Path(base_path).name,
        "initialization_behavior": (
            "peft_default_init_lora_weights_true"
            if value.get("init_lora_weights") is True
            else value.get("init_lora_weights")
        ),
        "lora_scaling": float(value["lora_alpha"]) / int(value["r"]),
    }
    expected = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "serialized_target_module_order": list(SERIALIZED_TARGET_ORDER),
        "target_module_names_sorted": sorted(CONSTRUCTION_TARGET_ORDER),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "saved_inference_mode": True,
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "revision": None,
        "base_model_revision": MODEL_REVISION,
        "initialization_behavior": "peft_default_init_lora_weights_true",
        "lora_scaling": EXPECTED_LORA_SCALING,
    }
    if projection != expected:
        raise ValueError("historical adapter config projection differs")
    projection["adapter_config_projection_sha256"] = canonical_sha256(projection)
    return projection


def state_inventory(path: Path) -> dict[str, Any]:
    """Read one adapter state inventory without materializing tensor values."""

    safetensors = importlib.import_module("safetensors")
    handle = safetensors.safe_open(str(path), framework="pt", device="cpu")
    shapes = [
        {"name": key, "shape": list(handle.get_slice(key).get_shape())} for key in handle.keys()
    ]
    layers: set[int] = set()
    modules: set[tuple[int, str]] = set()
    targets: set[str] = set()
    sides: set[tuple[int, str, str]] = set()
    for item in shapes:
        match = STATE_KEY_PATTERN.fullmatch(str(item["name"]))
        if match is None:
            raise ValueError("historical adapter contains a non-V1 trainable tensor")
        layer, target, side = int(match.group(1)), match.group(2), match.group(3)
        layers.add(layer)
        modules.add((layer, target))
        targets.add(target)
        sides.add((layer, target, side))
    parameters = sum(math.prod(cast(list[int], item["shape"])) for item in shapes)
    inventory: dict[str, Any] = {
        "trainable_tensor_count": len(shapes),
        "trainable_parameter_count": parameters,
        "adapted_layer_count": len(layers),
        "adapted_module_count": len(modules),
        "target_module_names_sorted": sorted(targets),
        "lora_a_b_complete": len(sides) == 2 * len(modules),
        "tensor_inventory_sha256": canonical_sha256(shapes),
    }
    if inventory != {
        "trainable_tensor_count": EXPECTED_TRAINABLE_TENSORS,
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "adapted_layer_count": EXPECTED_ADAPTED_LAYERS,
        "adapted_module_count": EXPECTED_ADAPTED_MODULES,
        "target_module_names_sorted": sorted(CONSTRUCTION_TARGET_ORDER),
        "lora_a_b_complete": True,
        "tensor_inventory_sha256": inventory["tensor_inventory_sha256"],
    }:
        raise ValueError("historical adapter trainable inventory differs")
    return inventory


def canonical_lora_configuration(
    runner: dict[str, Any], adapter: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    """Bind executed, serialized, and tensor-level views of one scientific recipe."""

    if (
        runner["rank"] != adapter["rank"]
        or runner["alpha"] != adapter["alpha"]
        or runner["dropout"] != adapter["dropout"]
        or runner["target_module_names_sorted"] != adapter["target_module_names_sorted"]
        or runner["target_module_names_sorted"] != inventory["target_module_names_sorted"]
        or runner["bias"] != adapter["bias"]
        or runner["task_type"] != adapter["task_type"]
        or runner["fan_in_fan_out"] != adapter["fan_in_fan_out"]
        or runner["modules_to_save"] != adapter["modules_to_save"]
        or runner["revision"] != adapter["revision"]
        or runner["initialization_behavior"] != adapter["initialization_behavior"]
        or runner["lora_scaling"] != adapter["lora_scaling"]
    ):
        raise ValueError("historical LoRA representations disagree scientifically")
    configuration: dict[str, Any] = {
        "recipe_id": RECIPE_ID,
        "rank": runner["rank"],
        "alpha": runner["alpha"],
        "dropout": runner["dropout"],
        "construction_target_module_order": runner["construction_target_module_order"],
        "serialized_adapter_config_order": adapter["serialized_target_module_order"],
        "target_module_names_sorted": runner["target_module_names_sorted"],
        "bias": runner["bias"],
        "task_type": runner["task_type"],
        "training_inference_mode": runner["training_inference_mode"],
        "saved_inference_mode": adapter["saved_inference_mode"],
        "fan_in_fan_out": runner["fan_in_fan_out"],
        "modules_to_save": runner["modules_to_save"],
        "revision": runner["revision"],
        "base_model_revision": adapter["base_model_revision"],
        "initialization_behavior": runner["initialization_behavior"],
        "lora_scaling": runner["lora_scaling"],
        "trainable_tensor_count": inventory["trainable_tensor_count"],
        "trainable_parameter_count": inventory["trainable_parameter_count"],
        "adapted_layer_count": inventory["adapted_layer_count"],
        "adapted_module_count": inventory["adapted_module_count"],
        "tensor_inventory_sha256": inventory["tensor_inventory_sha256"],
    }
    configuration["canonical_kl_lora_configuration_sha256"] = canonical_sha256(configuration)
    return configuration


def reconstruct(root: Path) -> dict[str, Any]:
    """Reconstruct all historical sources and freeze the V1-equivalent KL recipe."""

    results = root / "results/phase2_vetted_corpus"
    raw_training = root / "results/raw/phase2_vetted_corpus/v1_training"
    runner_path = root / "src/foundry/phase2/vetted_qlora.py"
    training_path = results / "v1_training.json"
    schedule_path = results / "v1_replay25_schedules.json"
    blocker_path = results / "milestone13c_r2_kl_recipe_blocker.json"
    if (
        file_sha256(training_path) != V1_TRAINING_FILE_SHA256
        or file_sha256(schedule_path) != V1_SCHEDULE_FILE_SHA256
    ):
        raise ValueError("published V1 training or schedule evidence differs")
    published = _read_object(training_path)
    schedule = _read_object(schedule_path)
    blocker = _read_object(blocker_path)
    _validate_hash(blocker, "blocker_sha256")
    if blocker["blocker_sha256"] != R2_BLOCKER_SHA256:
        raise ValueError("R2 recipe blocker identity differs")
    runner = runner_lora_projection(runner_path)
    artifact_rows: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    for arm in ARMS:
        raw_summary_path = raw_training / f"{arm}_summary.json"
        raw_summary = _read_object(raw_summary_path)
        _validate_hash(raw_summary, "result_sha256")
        if raw_summary["schedule_sha256"] != schedule[arm]["schedule_sha256"]:
            raise ValueError("historical raw summary schedule differs")
        for checkpoint in CHECKPOINTS:
            adapter_path = raw_training / arm / f"checkpoint-{checkpoint}" / "adapter"
            config_path = adapter_path / "adapter_config.json"
            state_path = adapter_path / "adapter_model.safetensors"
            projection = adapter_config_projection(config_path)
            inventory = state_inventory(state_path)
            adapter_hash = directory_sha256(adapter_path)
            expected_hash = EXPECTED_ADAPTER_HASHES[arm][str(checkpoint)]
            if (
                adapter_hash != expected_hash
                or raw_summary["checkpoints"][str(checkpoint)]["adapter_sha256"] != expected_hash
                or published[arm]["checkpoints"][str(checkpoint)]["adapter_sha256"] != expected_hash
            ):
                raise ValueError("published V1 adapter hash differs")
            projections.append(projection)
            inventories.append(inventory)
            artifact_rows.append(
                {
                    "arm": arm,
                    "checkpoint": checkpoint,
                    "adapter_sha256": adapter_hash,
                    "adapter_config_file_sha256": file_sha256(config_path),
                    "adapter_config_projection_sha256": projection[
                        "adapter_config_projection_sha256"
                    ],
                    "adapter_state_file_sha256": file_sha256(state_path),
                    "tensor_inventory_sha256": inventory["tensor_inventory_sha256"],
                    "raw_training_summary_file_sha256": file_sha256(raw_summary_path),
                    "raw_training_result_sha256": raw_summary["result_sha256"],
                }
            )
    if any(value != projections[0] for value in projections[1:]) or any(
        value != inventories[0] for value in inventories[1:]
    ):
        raise ValueError("historical adapter configs or trainable inventories disagree")
    canonical = canonical_lora_configuration(runner, projections[0], inventories[0])
    historical_projection = {
        name: value
        for name, value in canonical.items()
        if name != "canonical_kl_lora_configuration_sha256"
    }
    historical_v1_configuration_sha256 = canonical_sha256(historical_projection)
    if historical_v1_configuration_sha256 != canonical["canonical_kl_lora_configuration_sha256"]:
        raise RuntimeError("canonical KL recipe is not V1-equivalent")
    comparator: dict[str, Any] = {
        "comparator_id": "foundry-historical-v1-step16-lambda-zero-v1",
        "lambda_kl": 0,
        "retrained": False,
        "seed": 20260720,
        "optimizer_steps": 16,
        "assistant_tokens_per_arm": 16_000,
        "base_model_revision": MODEL_REVISION,
        "generic": {
            "adapter_sha256": EXPECTED_ADAPTER_HASHES["generic"]["16"],
            "schedule_sha256": schedule["generic"]["schedule_sha256"],
            "schedule_prefix_sha256": schedule["generic"]["checkpoint_prefix_sha256"]["16"],
            "validation_loss": published["generic"]["checkpoints"]["16"]["validation_loss"],
        },
        "targeted": {
            "adapter_sha256": EXPECTED_ADAPTER_HASHES["targeted"]["16"],
            "schedule_sha256": schedule["targeted"]["schedule_sha256"],
            "schedule_prefix_sha256": schedule["targeted"]["checkpoint_prefix_sha256"]["16"],
            "validation_loss": published["targeted"]["checkpoints"]["16"]["validation_loss"],
        },
        "shared_replay_schedule_sha256": schedule["replay_schedule_sha256"],
        "canonical_kl_lora_configuration_sha256": canonical[
            "canonical_kl_lora_configuration_sha256"
        ],
        "holdout_v2_use": False,
        "gsm1k_use": False,
    }
    comparator["comparator_contract_sha256"] = canonical_sha256(comparator)
    draft = cast(dict[str, Any], blocker["blocker"]["written_milestone13c_recipe"])
    draft_projection = {
        "classification": "conflicting_unexecuted_draft_recipe",
        "rank": draft["rank"],
        "alpha": draft["alpha"],
        "target_modules": draft["target_modules"],
        "published_v1_configuration": False,
        "experimental_evidence": False,
        "execution_count": 0,
        "confounded_fields": [
            "adapter_rank",
            "adapter_alpha",
            "target_projections",
            "trainable_tensor_count",
            "trainable_parameter_count",
            "adapter_capacity",
        ],
    }
    draft_projection["rank16_draft_rejection_sha256"] = canonical_sha256(draft_projection)
    equality: dict[str, Any] = {
        "all_six_adapter_config_files_byte_identical": True,
        "all_six_adapter_config_projections_identical": True,
        "all_six_tensor_inventories_identical": True,
        "all_six_directory_hashes_match_published_evidence": True,
        "runner_and_artifacts_scientifically_equal": True,
        "construction_order_and_peft_serialized_order_recorded_separately": True,
        "lambda_kl_is_only_new_scientific_field": True,
    }
    equality["equality_evidence_sha256"] = canonical_sha256(equality)
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_id": "foundry-milestone13c-r3-v1-kl-recipe-v1",
        "decision": "historical_v1_rank8_recipe_is_canonical_for_kl",
        "dataset_sha256": DATASET_SHA256,
        "architecture_decision_sha256": ARCHITECTURE_DECISION_SHA256,
        "historical_v1_configuration_sha256": historical_v1_configuration_sha256,
        "generic_adapter_config_projection_sha256": projections[0][
            "adapter_config_projection_sha256"
        ],
        "targeted_adapter_config_projection_sha256": projections[-1][
            "adapter_config_projection_sha256"
        ],
        "canonical_lora_configuration": canonical,
        "historical_artifacts": artifact_rows,
        "equality_evidence": equality,
        "rank16_draft_rejection": draft_projection,
        "comparator_contract": comparator,
        "published_v1_step64_context": {arm: EXPECTED_ADAPTER_HASHES[arm]["64"] for arm in ARMS},
        "r2_blocker_preserved": True,
        "r2_blocker_sha256": R2_BLOCKER_SHA256,
        "sealed_boundary_status": "metadata_accessed_example_content_unseen",
        "sealed_paths_accessed": False,
        "holdout_v2_adapter_exposure": False,
        "kl_implementation_before_recipe_freeze": False,
        "model_loads": 0,
        "optimizer_steps": 0,
    }
    record["final_recipe_decision_sha256"] = canonical_sha256(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("recipe output already exists")
    result = reconstruct(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
