"""Evaluate and select the frozen retention-safe adaptation ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256

CHECKPOINTS = (8, 16, 24, 32)
VARIANTS = {
    "a": {"format_id": "v3", "learning_rate": 5e-5},
    "b": {"format_id": "v4", "learning_rate": 5e-5},
    "c": {"format_id": "v4", "learning_rate": 2e-5},
    "d": {"format_id": "v4", "learning_rate": 1e-5},
}


def _load(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return cast(dict[str, Any], value)


def _accuracy(summary: dict[str, Any], section: str) -> float:
    metrics = summary.get("section_metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get(section), dict):
        raise ValueError("retention summary lacks section metrics")
    return float(cast(dict[str, Any], metrics[section])["accuracy"])


def evaluate_checkpoint(
    *,
    base: dict[str, Any],
    training: dict[str, Any],
    retention: dict[str, Any],
    checkpoint: int,
) -> dict[str, Any]:
    """Apply the calibration gate to one arm and one saved checkpoint."""

    checkpoint_values = training.get("checkpoints")
    if not isinstance(checkpoint_values, dict) or not isinstance(
        checkpoint_values.get(str(checkpoint)), dict
    ):
        raise ValueError("training summary lacks checkpoint evidence")
    trained = cast(dict[str, Any], checkpoint_values[str(checkpoint)])
    base_arithmetic = _accuracy(base, "arithmetic")
    arithmetic = _accuracy(retention, "arithmetic")
    format_accuracy = _accuracy(retention, "format")
    instruction = _accuracy(retention, "instruction")
    extractability = float(retention["extractability"])
    prompt_echo_rate = float(retention["prompt_echo_rate"])
    question_generation = int(retention["question_generation"])
    backend_failures = int(retention["backend_failures"])
    validation_loss = float(trained["synthetic_validation_loss"])
    checks = {
        "arithmetic_retention": arithmetic + 1e-12 >= base_arithmetic - 0.10,
        "format_following": format_accuracy + 1e-12 >= 0.90,
        "instruction_following": instruction + 1e-12 >= 0.90,
        "extractability": extractability + 1e-12 >= 0.90,
        "prompt_echo": prompt_echo_rate <= 0.05 + 1e-12,
        "question_generation": question_generation == 0,
        "backend_failures": backend_failures == 0,
        "finite_training_losses": bool(training["losses_all_finite"]),
        "finite_validation_loss": math.isfinite(validation_loss),
        "token_count_matches_schedule": bool(training["token_count_matches_schedule"]),
    }
    margins = {
        "arithmetic": arithmetic - (base_arithmetic - 0.10),
        "format": format_accuracy - 0.90,
        "instruction": instruction - 0.90,
        "extractability": extractability - 0.90,
        "prompt_echo": 0.05 - prompt_echo_rate,
    }
    return {
        "adapter_sha256": retention["adapter_sha256"],
        "cumulative_actual_loss_bearing_tokens": int(
            trained["cumulative_actual_loss_bearing_tokens"]
        ),
        "metrics": {
            "arithmetic_accuracy": arithmetic,
            "arithmetic_absolute_drop": base_arithmetic - arithmetic,
            "format_accuracy": format_accuracy,
            "instruction_accuracy": instruction,
            "extractability": extractability,
            "prompt_echo_rate": prompt_echo_rate,
            "question_generation": question_generation,
            "backend_failures": backend_failures,
            "synthetic_validation_loss": validation_loss,
        },
        "margins": margins,
        "minimum_retention_margin": min(margins.values()),
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }


def evaluate_variant(*, variant: str, base: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    """Evaluate all four common checkpoints for one predeclared variant."""

    arms: dict[str, Any] = {}
    for arm in ("generic_control", "targeted"):
        training = _load(raw_root / f"variant_{variant}" / f"{arm}_training_summary.json")
        arms[arm] = {
            str(step): evaluate_checkpoint(
                base=base,
                training=training,
                retention=_load(
                    raw_root
                    / f"variant_{variant}"
                    / arm
                    / "retention"
                    / f"checkpoint_{step}_summary.json"
                ),
                checkpoint=step,
            )
            for step in CHECKPOINTS
        }
    common: dict[str, Any] = {}
    for step in CHECKPOINTS:
        generic = arms["generic_control"][str(step)]
        targeted = arms["targeted"][str(step)]
        generic_tokens = int(generic["cumulative_actual_loss_bearing_tokens"])
        targeted_tokens = int(targeted["cumulative_actual_loss_bearing_tokens"])
        parity = abs(generic_tokens - targeted_tokens) / max(generic_tokens, targeted_tokens)
        common[str(step)] = {
            "generic_control_passed": generic["passed"],
            "targeted_passed": targeted["passed"],
            "token_parity_relative_difference": parity,
            "token_parity_passed": parity <= 0.005 + 1e-12,
            "minimum_retention_margin": min(
                float(generic["minimum_retention_margin"]),
                float(targeted["minimum_retention_margin"]),
            ),
            "passed": bool(generic["passed"] and targeted["passed"] and parity <= 0.005 + 1e-12),
        }
    passing = [step for step in CHECKPOINTS if common[str(step)]["passed"]]
    latest = max(passing) if passing else None
    config = VARIANTS[variant]
    return {
        "variant": variant,
        **config,
        "arms": arms,
        "common_checkpoints": common,
        "latest_common_passing_checkpoint": latest,
        "selection_margin": None
        if latest is None
        else common[str(latest)]["minimum_retention_margin"],
    }


def build_ladder_result(*, base: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    """Build the complete result and apply the deterministic hierarchy."""

    variants = {
        variant: evaluate_variant(variant=variant, base=base, raw_root=raw_root)
        for variant in VARIANTS
    }
    passing = [value for value in variants.values() if value["latest_common_passing_checkpoint"]]
    selected = max(
        passing,
        key=lambda item: (
            int(item["latest_common_passing_checkpoint"]),
            float(item["selection_margin"]),
            item["format_id"] == "v4",
            float(item["learning_rate"]),
        ),
        default=None,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "result_id": "foundry-retention-safe-adaptation-ladder-result-v1",
        "base_retention_summary_sha256": base["summary_sha256"],
        "selection_hierarchy": [
            "latest_common_passing_checkpoint",
            "highest_minimum_retention_margin",
            "prefer_concise_v4",
            "prefer_higher_learning_rate",
        ],
        "variants": variants,
        "selected_variant": None if selected is None else selected["variant"],
        "selected_checkpoint": None
        if selected is None
        else selected["latest_common_passing_checkpoint"],
        "selected_format_id": None if selected is None else selected["format_id"],
        "selected_learning_rate": None if selected is None else selected["learning_rate"],
        "retention_safe_variant_found": selected is not None,
        "gsm1k_consulted_for_selection": False,
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_ladder_result(base=_load(args.base), raw_root=args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
