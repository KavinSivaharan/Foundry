"""Fail-closed parity checks for token-matched QLoRA v2 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256

PARITY_LIMIT = 0.005
EQUAL_FIELDS = (
    "recipe_id",
    "recipe_sha256",
    "selected_method",
    "base_recipe_sha256",
    "base_model_id",
    "base_revision",
    "requirements_lock_sha256",
    "sft_format_sha256",
    "software_versions",
    "cuda_runtime",
    "gpu_name",
    "optimizer_steps",
    "scheduler_steps",
    "training_source_records",
    "training_truncated_records",
    "validation_source_records",
    "validation_truncated_records",
    "trainable_parameters",
    "total_parameters",
    "only_lora_trainable",
    "base_loaded_in_4bit",
)


def _load(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("training summary must be an object")
    return cast(dict[str, Any], value)


def compare_token_matched_runs(
    generic: dict[str, Any], targeted: dict[str, Any]
) -> dict[str, object]:
    """Compare immutable runtime controls and actual loss-bearing exposure."""

    equal_fields = {field: generic.get(field) == targeted.get(field) for field in EQUAL_FIELDS}
    generic_tokens = int(generic["actual_loss_bearing_tokens"])
    targeted_tokens = int(targeted["actual_loss_bearing_tokens"])
    absolute = abs(targeted_tokens - generic_tokens)
    relative = absolute / max(generic_tokens, targeted_tokens)
    gate_checks = {
        "metadata_parity": all(equal_fields.values()),
        "actual_tokens_match_schedules": generic.get("token_count_matches_schedule") is True
        and targeted.get("token_count_matches_schedule") is True,
        "actual_token_parity_within_0_5_percent": relative <= PARITY_LIMIT,
        "losses_finite": generic.get("losses_all_finite") is True
        and targeted.get("losses_all_finite") is True,
        "gradients_finite": generic.get("gradients_all_finite") is True
        and targeted.get("gradients_all_finite") is True,
        "offline_reload_passed": generic.get("adapter_offline_reload_ok") is True
        and targeted.get("adapter_offline_reload_ok") is True,
        "no_development_exposure": generic.get("development_benchmark_exposure_during_training")
        is False
        and targeted.get("development_benchmark_exposure_during_training") is False,
    }
    return {
        "equal_fields": equal_fields,
        "generic_actual_loss_bearing_tokens": generic_tokens,
        "targeted_actual_loss_bearing_tokens": targeted_tokens,
        "absolute_token_difference": absolute,
        "relative_token_difference": relative,
        "parity_limit": PARITY_LIMIT,
        "gate_checks": gate_checks,
        "parity_gate_passed": all(gate_checks.values()),
    }


def run_parity(
    *,
    generic_summary_path: Path,
    targeted_summary_path: Path,
    generic_adapter_path: Path,
    targeted_adapter_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Verify adapter hashes and persist a content-free parity result."""

    generic = _load(generic_summary_path)
    targeted = _load(targeted_summary_path)
    comparison = compare_token_matched_runs(generic, targeted)
    adapter_hashes = {
        "generic_control": directory_sha256(generic_adapter_path),
        "targeted": directory_sha256(targeted_adapter_path),
    }
    adapter_hash_checks = {
        "generic_control": adapter_hashes["generic_control"] == generic.get("adapter_sha256"),
        "targeted": adapter_hashes["targeted"] == targeted.get("adapter_sha256"),
    }
    passed = bool(comparison["parity_gate_passed"]) and all(adapter_hash_checks.values())
    result: dict[str, object] = {
        "schema_version": 1,
        "run_kind": generic.get("run_kind"),
        "comparison": comparison,
        "adapter_sha256": adapter_hashes,
        "adapter_hash_checks": adapter_hash_checks,
        "parity_gate_passed": passed,
        "benchmark_evaluation_authorized": passed and generic.get("run_kind") == "final_adapter",
    }
    result["summary_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-summary", required=True, type=Path)
    parser.add_argument("--targeted-summary", required=True, type=Path)
    parser.add_argument("--generic-adapter", required=True, type=Path)
    parser.add_argument("--targeted-adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_parity(
        generic_summary_path=args.generic_summary,
        targeted_summary_path=args.targeted_summary,
        generic_adapter_path=args.generic_adapter,
        targeted_adapter_path=args.targeted_adapter,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
