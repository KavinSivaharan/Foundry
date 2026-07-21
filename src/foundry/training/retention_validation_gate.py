"""Disjoint retention-validation gate for the selected ladder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256


def _load(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return cast(dict[str, Any], value)


def _accuracy(summary: dict[str, Any], section: str) -> float:
    values = summary.get("section_metrics")
    if not isinstance(values, dict) or not isinstance(values.get(section), dict):
        raise ValueError("retention summary lacks section metrics")
    return float(cast(dict[str, Any], values[section])["accuracy"])


def evaluate_validation_arm(*, base: dict[str, Any], retention: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen disjoint-suite requirements to one selected arm."""

    arithmetic = _accuracy(retention, "arithmetic")
    format_accuracy = _accuracy(retention, "format")
    instruction = _accuracy(retention, "instruction")
    base_arithmetic = _accuracy(base, "arithmetic")
    extractability = float(retention["extractability"])
    echo = float(retention["prompt_echo_rate"])
    question_generation = int(retention["question_generation"])
    backend = int(retention["backend_failures"])
    checks = {
        "arithmetic_retention": arithmetic + 1e-12 >= base_arithmetic - 0.10,
        "format_following": format_accuracy + 1e-12 >= 0.90,
        "instruction_following": instruction + 1e-12 >= 0.90,
        "extractability": extractability + 1e-12 >= 0.90,
        "prompt_echo": echo <= 0.05 + 1e-12,
        "question_generation": question_generation == 0,
        "backend_failures": backend == 0,
    }
    return {
        "adapter_sha256": retention["adapter_sha256"],
        "metrics": {
            "arithmetic_accuracy": arithmetic,
            "arithmetic_absolute_drop": base_arithmetic - arithmetic,
            "format_accuracy": format_accuracy,
            "instruction_accuracy": instruction,
            "extractability": extractability,
            "prompt_echo_rate": echo,
            "question_generation": question_generation,
            "backend_failures": backend,
        },
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }


def build_validation_result(
    *,
    base: dict[str, Any],
    ladder: dict[str, Any],
    generic: dict[str, Any],
    targeted: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the mandatory proceed-or-stop decision."""

    if ladder.get("selected_variant") != "a" or ladder.get("selected_checkpoint") != 32:
        raise ValueError("ladder selection differs from the frozen input")
    generic_result = evaluate_validation_arm(base=base, retention=generic)
    targeted_result = evaluate_validation_arm(base=base, retention=targeted)
    passed = bool(generic_result["passed"] and targeted_result["passed"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "gate_id": "foundry-retention-safe-validation-gate-v1",
        "ladder_result_sha256": ladder["summary_sha256"],
        "validation_suite_sha256": base["suite_sha256"],
        "validation_base_summary_sha256": base["summary_sha256"],
        "selected_variant": "a",
        "selected_checkpoint": 32,
        "generic_control": generic_result,
        "targeted": targeted_result,
        "passed": passed,
        "protocol_commit_authorized": passed,
        "full_retraining_authorized": passed,
        "gsm1k_evaluation_authorized": False,
        "final_holdout_opened": False,
        "stop_reason": None
        if passed
        else "both selected arms failed the 90% disjoint instruction-retention threshold",
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_validation_result(
        base=_load(args.base),
        ladder=_load(args.ladder),
        generic=_load(args.generic),
        targeted=_load(args.targeted),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
