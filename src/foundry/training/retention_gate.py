"""Deterministic retention-smoke gate for assistant-only SFT v3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256


def _load_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _section_accuracy(summary: dict[str, Any], section: str) -> float:
    sections = summary.get("section_metrics")
    if not isinstance(sections, dict):
        raise ValueError("retention summary lacks section_metrics")
    metric = sections.get(section)
    if not isinstance(metric, dict):
        raise ValueError(f"retention summary lacks {section} metrics")
    return float(metric["accuracy"])


def evaluate_arm(
    *,
    base: dict[str, Any],
    training: dict[str, Any],
    retention: dict[str, Any],
) -> dict[str, Any]:
    """Apply every frozen retention-smoke requirement to one trained arm."""

    base_arithmetic = _section_accuracy(base, "arithmetic")
    arithmetic = _section_accuracy(retention, "arithmetic")
    format_accuracy = _section_accuracy(retention, "format")
    instruction = _section_accuracy(retention, "instruction")
    extractability = float(retention["extractability"])
    prompt_echo_rate = float(retention["prompt_echo_rate"])
    question_generation = int(retention["question_generation"])
    backend_failures = int(retention["backend_failures"])
    final_validation_loss = float(training["final_validation_loss"])
    checks = {
        "arithmetic_retention": arithmetic + 1e-12 >= base_arithmetic - 0.10,
        "format_following": format_accuracy + 1e-12 >= 0.90,
        "instruction_following": instruction + 1e-12 >= 0.90,
        "extractability": extractability + 1e-12 >= 0.90,
        "prompt_echo": prompt_echo_rate <= 0.05 + 1e-12,
        "question_generation": question_generation == 0,
        "backend_failures": backend_failures == 0,
        "finite_training_losses": bool(training["losses_all_finite"]),
        "finite_validation_loss": math.isfinite(final_validation_loss),
        "token_count_matches_schedule": bool(training["token_count_matches_schedule"]),
    }
    return {
        "adapter_sha256": retention["adapter_sha256"],
        "actual_loss_bearing_tokens": int(training["actual_loss_bearing_tokens"]),
        "metrics": {
            "arithmetic_accuracy": arithmetic,
            "arithmetic_absolute_drop": base_arithmetic - arithmetic,
            "format_accuracy": format_accuracy,
            "instruction_accuracy": instruction,
            "extractability": extractability,
            "prompt_echo_rate": prompt_echo_rate,
            "question_generation": question_generation,
            "backend_failures": backend_failures,
            "final_validation_loss": final_validation_loss,
        },
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }


def evaluate_recipe(
    *,
    recipe_id: str,
    learning_rate: float,
    base: dict[str, Any],
    generic_training: dict[str, Any],
    generic_retention: dict[str, Any],
    targeted_training: dict[str, Any],
    targeted_retention: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen paired-arm recipe gate, including token parity."""

    generic = evaluate_arm(base=base, training=generic_training, retention=generic_retention)
    targeted = evaluate_arm(base=base, training=targeted_training, retention=targeted_retention)
    generic_tokens = int(generic["actual_loss_bearing_tokens"])
    targeted_tokens = int(targeted["actual_loss_bearing_tokens"])
    difference = abs(generic_tokens - targeted_tokens)
    difference_rate = difference / max(generic_tokens, targeted_tokens)
    parity_passed = difference_rate <= 0.005 + 1e-12
    return {
        "recipe_id": recipe_id,
        "learning_rate": learning_rate,
        "generic_control": generic,
        "targeted": targeted,
        "token_parity": {
            "generic_tokens": generic_tokens,
            "targeted_tokens": targeted_tokens,
            "difference_tokens": difference,
            "difference_rate": difference_rate,
            "maximum_difference_rate": 0.005,
            "passed": parity_passed,
        },
        "passed": bool(generic["passed"] and targeted["passed"] and parity_passed),
    }


def build_gate_summary(*, base: dict[str, Any], recipes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the immutable milestone decision from evaluated recipes."""

    passing = [recipe for recipe in recipes if recipe["passed"]]
    selected = passing[0]["recipe_id"] if passing else None
    payload: dict[str, Any] = {
        "schema_version": 1,
        "gate_id": "foundry-assistant-only-sft-v3-retention-smoke-gate-v1",
        "base_retention_summary_sha256": base["summary_sha256"],
        "thresholds": {
            "maximum_arithmetic_absolute_drop": 0.10,
            "minimum_format_accuracy": 0.90,
            "minimum_instruction_accuracy": 0.90,
            "minimum_extractability": 0.90,
            "maximum_prompt_echo_rate": 0.05,
            "maximum_token_difference_rate": 0.005,
            "question_generation": 0,
            "backend_failures": 0,
            "finite_training_and_validation_losses": True,
        },
        "recipes": recipes,
        "selected_recipe_id": selected,
        "retention_safe_recipe_found": selected is not None,
        "full_retraining_authorized": selected is not None,
        "frozen_development_evaluation_authorized": False,
        "stop_reason": None
        if selected is not None
        else "both predeclared learning-rate recipes failed the frozen retention gate",
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--recipe", action="append", nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = _load_object(args.base)
    recipes: list[dict[str, Any]] = []
    for (
        recipe_id,
        learning_rate,
        generic_train,
        generic_eval,
        target_train,
        target_eval,
    ) in args.recipe:
        recipes.append(
            evaluate_recipe(
                recipe_id=recipe_id,
                learning_rate=float(learning_rate),
                base=base,
                generic_training=_load_object(Path(generic_train)),
                generic_retention=_load_object(Path(generic_eval)),
                targeted_training=_load_object(Path(target_train)),
                targeted_retention=_load_object(Path(target_eval)),
            )
        )
    summary = build_gate_summary(base=base, recipes=recipes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
