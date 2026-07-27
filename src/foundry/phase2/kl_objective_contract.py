"""Freeze and validate the replay-ce-token-kl-v1 objective contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_recipe
from foundry.phase2.launch_contract import ALLOWLISTED_ENVIRONMENT
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

OBJECTIVE_ID = "foundry-replay-ce-token-kl-v1"
RECIPE_SHA256 = "3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"
RECIPE_DECISION_SHA256 = "b03dfc9d6f66843f2a83e9b4ff5b82e133fb9c6a4a27ab1783d64991a0f7d118"
ENVIRONMENT_V2_SHA256 = "c9faa8afafafb20b84fcd0cb5e7de1b57749e822adfa27c8b401bbaf8f0153dc"
COEFFICIENTS = [0.01, 0.03, 0.10, 0.30]
SCHEDULE_SHA256 = {
    "generic": "4bc00d29d5cf308c12c77111d7943567521cc533b13440dc06c3d8b39c74e9df",
    "targeted": "88c5378cac7efe927b29d3f421d97777cd6d917187c71c8388b60bbe7b57e259",
}


def _read_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _validate_self_hash(value: dict[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _source_contract(root: Path) -> dict[str, Any]:
    files = {
        "objective_helpers": file_sha256(root / "src/foundry/phase2/kl_objective.py"),
        "training_runner": file_sha256(root / "src/foundry/phase2/vetted_qlora_kl.py"),
    }
    return {
        "files": files,
        "objective_source_sha256": canonical_sha256(files),
    }


def _configuration_contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "objective_id": OBJECTIVE_ID,
        "vetted_loss": "assistant_only_cross_entropy",
        "replay_loss": (
            "assistant_only_replay_target_cross_entropy+lambda_kl*mean_forward_token_kl"
        ),
        "kl_direction": "adapter_disabled_frozen_base||active_adapter_policy",
        "kl_formula": "sum(p_base*(log_p_base-log_p_adapter))",
        "kl_reduction": "mean_over_valid_replay_assistant_tokens",
        "kl_probability_dtype": "float32",
        "model_compute_dtype": "float16",
        "candidate_coefficients_in_order": COEFFICIENTS,
        "reference_model_count": 1,
        "cpu_offload": False,
        "vetted_kl": False,
        "optimizer": "PagedAdamW8bit",
        "learning_rate": 1e-5,
        "scheduler": "cosine",
        "warmup_steps": 4,
        "scheduler_horizon_steps": 64,
        "seed": 20260720,
        "schedule_sha256": SCHEDULE_SHA256,
        "recipe_sha256": RECIPE_SHA256,
        "recipe_decision_sha256": RECIPE_DECISION_SHA256,
    }
    value["objective_configuration_sha256"] = canonical_sha256(value)
    return value


def _masking_contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "alignment": "logits[:, :-1, :] predicts labels[:, 1:]",
        "selection": "labels[:, 1:] != -100",
        "included": ["assistant_content", "final_assistant_eos"],
        "excluded": [
            "system",
            "user",
            "assistant_header",
            "padding",
            "post_eos",
        ],
        "maximum_sequence_tokens": 512,
        "ce_mask_unchanged": True,
    }
    value["masking_sha256"] = canonical_sha256(value)
    return value


def _reference_contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "mechanism": "same_model_disable_adapter_context",
        "reference_context": "torch.no_grad",
        "reference_logits_detached": True,
        "second_full_model_instantiated": False,
        "base_parameters_optimizer_owned": False,
        "direction": "KL(base||adapter)",
        "replay_only": True,
    }
    value["reference_mechanism_sha256"] = canonical_sha256(value)
    return value


def _fixture_contract(root: Path, runtime_path: Path) -> dict[str, Any]:
    runtime = _read_object(runtime_path)
    _validate_self_hash(runtime, "fixture_sha256")
    required = {
        "fixture_id": "foundry-replay-ce-token-kl-v1-runtime-fixture-v1",
        "two_updates": True,
        "finite_losses": True,
        "finite_gradients": True,
        "reference_no_grad": True,
        "optimizer_owned_only_lora": True,
        "base_parameters_unchanged": True,
        "base_restoration": True,
        "adapter_saved": False,
        "holdout_v2_use": False,
        "gsm1k_use": False,
    }
    if any(runtime.get(name) != expected for name, expected in required.items()):
        raise ValueError("runtime objective fixture gate failed")
    if runtime.get("identical_logits_kl") != 0.0:
        raise ValueError("runtime identical-logit KL is not zero")
    if float(runtime.get("post_update_replay_kl", 0.0)) <= 0.0:
        raise ValueError("runtime post-update KL is not positive")
    if runtime["launch_evidence"]["preimport"]["environment"] != ALLOWLISTED_ENVIRONMENT:
        raise ValueError("runtime fixture deterministic environment differs")
    inventory = runtime.get("trainable_inventory")
    if not isinstance(inventory, dict) or inventory.get("tensor_inventory_sha256") != (
        "d3edea65d6d09226eb743182474ea51b2af1c0f94b163812ce67913ffc865e78"
    ):
        raise ValueError("runtime fixture LoRA inventory differs")
    projection: dict[str, Any] = {
        **required,
        "identical_logits_kl": runtime["identical_logits_kl"],
        "post_update_replay_kl": runtime["post_update_replay_kl"],
        "changed_lora_tensor_count": runtime["changed_lora_tensor_count"],
        "base_parameter_fingerprint": runtime["base_parameter_fingerprint_before"],
        "trainable_inventory": inventory,
        "runtime_fixture_record_sha256": runtime["fixture_sha256"],
        "unit_fixture_source_sha256": file_sha256(root / "tests/unit/phase2/test_kl_objective.py"),
    }
    projection["fixture_sha256"] = canonical_sha256(projection)
    return projection


def build(root: Path, runtime_path: Path) -> dict[str, Any]:
    """Build the content-free objective contract from sources and the runtime fixture."""

    contract: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": "foundry-milestone13c-r3-kl-objective-v1",
        "objective_id": OBJECTIVE_ID,
        "dataset_sha256": kl_recipe.DATASET_SHA256,
        "environment_v2_contract_sha256": ENVIRONMENT_V2_SHA256,
        "historical_v1_recipe_sha256": RECIPE_SHA256,
        "source": _source_contract(root),
        "configuration": _configuration_contract(),
        "masking": _masking_contract(),
        "reference": _reference_contract(),
        "fixture": _fixture_contract(root, runtime_path),
        "holdout_v2_adapter_exposure": False,
        "gsm1k_use": False,
    }
    contract["objective_contract_sha256"] = canonical_sha256(contract)
    return contract


def validate(root: Path, contract: dict[str, Any]) -> None:
    """Validate a published contract without requiring ignored runtime artifacts."""

    _validate_self_hash(contract, "objective_contract_sha256")
    if contract.get("source") != _source_contract(root):
        raise ValueError("published objective source hash differs")
    if contract.get("configuration") != _configuration_contract():
        raise ValueError("published objective configuration differs")
    if contract.get("masking") != _masking_contract():
        raise ValueError("published masking contract differs")
    if contract.get("reference") != _reference_contract():
        raise ValueError("published reference contract differs")
    fixture = contract.get("fixture")
    if not isinstance(fixture, dict):
        raise ValueError("published fixture contract is absent")
    supplied = fixture.get("fixture_sha256")
    payload = {name: item for name, item in fixture.items() if name != "fixture_sha256"}
    if supplied != canonical_sha256(payload):
        raise ValueError("published fixture hash differs")
    if fixture.get("unit_fixture_source_sha256") != file_sha256(
        root / "tests/unit/phase2/test_kl_objective.py"
    ):
        raise ValueError("published unit-fixture source hash differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("objective contract output already exists")
    result = build(args.root.resolve(), args.runtime_fixture.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
