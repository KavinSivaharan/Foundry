"""Deterministic paired analysis for the token-matched development evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260720
EXPECTED_EXAMPLES = 814
TARGETED_CATEGORIES = frozenset(
    {
        "multi_step_bookkeeping_or_omission",
        "rate_ratio_percentage_or_average",
        "constraint_distribution_or_discrete_reasoning",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _load_predictions(path: Path) -> dict[str, bool]:
    records: dict[str, bool] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"prediction row {line_number} is not an object")
        stable_id = value.get("stable_id")
        correct = value.get("correct")
        if not isinstance(stable_id, str) or len(stable_id) != 64:
            raise ValueError(f"prediction row {line_number} has an invalid stable ID")
        if not isinstance(correct, bool):
            raise ValueError(f"prediction row {line_number} has non-boolean correctness")
        if stable_id in records:
            raise ValueError(f"duplicate prediction stable ID: {stable_id[:12]}")
        records[stable_id] = correct
    return records


def _load_taxonomy(path: Path, valid_ids: set[str]) -> dict[str, str]:
    categories: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"taxonomy row {line_number} is not an object")
        stable_id = value.get("stable_id")
        category = value.get("primary_category")
        if not isinstance(stable_id, str) or stable_id not in valid_ids:
            raise ValueError(f"taxonomy row {line_number} has an unknown stable ID")
        if not isinstance(category, str) or not category:
            raise ValueError(f"taxonomy row {line_number} has an invalid category")
        if stable_id in categories:
            raise ValueError(f"duplicate taxonomy stable ID: {stable_id[:12]}")
        categories[stable_id] = category
    return categories


def _percentile(sorted_values: list[float], probability: float) -> float:
    """Return a deterministic linearly interpolated empirical percentile."""

    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_interval(
    differences: tuple[int, ...],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Bootstrap a paired mean difference by resampling stable example positions."""

    if not differences or replicates <= 0:
        raise ValueError("paired bootstrap requires differences and positive replicates")
    generator = random.Random(seed)
    count = len(differences)
    means = [
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    ]
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _transition_counts(base: bool, generic: bool, targeted: bool) -> str:
    return f"base_{int(base)}_generic_{int(generic)}_targeted_{int(targeted)}"


def _category_results(
    *,
    base: dict[str, bool],
    generic: dict[str, bool],
    targeted: dict[str, bool],
    taxonomy: dict[str, str],
) -> tuple[dict[str, dict[str, int | float]], dict[str, int | float | bool]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for stable_id, category in taxonomy.items():
        grouped[category].append(stable_id)

    results: dict[str, dict[str, int | float]] = {}
    for category in sorted(grouped):
        ids = grouped[category]
        base_correct = sum(base[item] for item in ids)
        generic_correct = sum(generic[item] for item in ids)
        targeted_correct = sum(targeted[item] for item in ids)
        results[category] = {
            "examples": len(ids),
            "base_correct": base_correct,
            "base_accuracy": base_correct / len(ids),
            "generic_correct": generic_correct,
            "generic_accuracy": generic_correct / len(ids),
            "targeted_correct": targeted_correct,
            "targeted_accuracy": targeted_correct / len(ids),
            "targeted_delta_vs_base": (targeted_correct - base_correct) / len(ids),
            "targeted_delta_vs_generic": (targeted_correct - generic_correct) / len(ids),
        }

    untargeted_ids = [
        stable_id for stable_id, category in taxonomy.items() if category not in TARGETED_CATEGORIES
    ]
    if not untargeted_ids:
        raise ValueError("frozen taxonomy has no aggregate untargeted category set")
    base_correct = sum(base[item] for item in untargeted_ids)
    generic_correct = sum(generic[item] for item in untargeted_ids)
    targeted_correct = sum(targeted[item] for item in untargeted_ids)
    delta = (targeted_correct - base_correct) / len(untargeted_ids)
    untargeted: dict[str, int | float | bool] = {
        "examples": len(untargeted_ids),
        "base_correct": base_correct,
        "base_accuracy": base_correct / len(untargeted_ids),
        "generic_correct": generic_correct,
        "generic_accuracy": generic_correct / len(untargeted_ids),
        "targeted_correct": targeted_correct,
        "targeted_accuracy": targeted_correct / len(untargeted_ids),
        "targeted_delta_vs_base": delta,
        "maximum_allowed_absolute_decline": 0.02,
        "no_greater_than_two_point_decline": delta >= -0.02,
    }
    return results, untargeted


def analyze_paired_results(
    *,
    base_predictions_path: Path,
    generic_predictions_path: Path,
    targeted_predictions_path: Path,
    generic_summary_path: Path,
    targeted_summary_path: Path,
    taxonomy_path: Path,
    final_parity_path: Path,
    retention_decision_path: Path,
    output_path: Path,
    expected_examples: int = EXPECTED_EXAMPLES,
) -> dict[str, Any]:
    """Validate alignment, calculate paired effects, and apply the frozen signal gate."""

    base = _load_predictions(base_predictions_path)
    generic = _load_predictions(generic_predictions_path)
    targeted = _load_predictions(targeted_predictions_path)
    if len(base) != expected_examples:
        raise ValueError(f"expected {expected_examples} base predictions, found {len(base)}")
    if set(base) != set(generic) or set(base) != set(targeted):
        raise ValueError("base, generic, and targeted stable-ID sets differ")

    generic_summary = _load_object(generic_summary_path)
    targeted_summary = _load_object(targeted_summary_path)
    final_parity = _load_object(final_parity_path)
    retention_decision = _load_object(retention_decision_path)
    if int(generic_summary["correct_examples"]) != sum(generic.values()):
        raise ValueError("generic summary and predictions disagree")
    if int(targeted_summary["correct_examples"]) != sum(targeted.values()):
        raise ValueError("targeted summary and predictions disagree")

    taxonomy = _load_taxonomy(taxonomy_path, set(base))
    ordered_ids = sorted(base)
    transitions = Counter(
        _transition_counts(base[item], generic[item], targeted[item]) for item in ordered_ids
    )
    differences = tuple(int(targeted[item]) - int(generic[item]) for item in ordered_ids)
    bootstrap_low, bootstrap_high = paired_bootstrap_interval(differences)

    generic_fixed = sum(not base[item] and generic[item] for item in ordered_ids)
    targeted_fixed = sum(not base[item] and targeted[item] for item in ordered_ids)
    generic_broken = sum(base[item] and not generic[item] for item in ordered_ids)
    targeted_broken = sum(base[item] and not targeted[item] for item in ordered_ids)
    targeted_only_fixed = sum(
        not base[item] and targeted[item] and not generic[item] for item in ordered_ids
    )
    generic_only_fixed = sum(
        not base[item] and generic[item] and not targeted[item] for item in ordered_ids
    )
    targeted_only_broken = sum(
        base[item] and not targeted[item] and generic[item] for item in ordered_ids
    )
    generic_only_broken = sum(
        base[item] and not generic[item] and targeted[item] for item in ordered_ids
    )
    targeted_wins = sum(targeted[item] and not generic[item] for item in ordered_ids)
    generic_wins = sum(generic[item] and not targeted[item] for item in ordered_ids)

    category_results, untargeted = _category_results(
        base=base, generic=generic, targeted=targeted, taxonomy=taxonomy
    )
    generic_correct = sum(generic.values())
    targeted_correct = sum(targeted.values())
    extractability = float(targeted_summary["extractable_answer_rate"])
    signal_checks = {
        "targeted_at_least_529_correct": targeted_correct >= 529,
        "targeted_at_least_four_over_generic": targeted_correct >= generic_correct + 4,
        "targeted_extractability_at_least_91_38_percent": extractability >= 0.9138,
        "targeted_zero_backend_failures": int(targeted_summary["generation_failures"]) == 0,
        "untargeted_aggregate_decline_within_two_points": bool(
            untargeted["no_greater_than_two_point_decline"]
        ),
        "final_training_parity_passed": bool(final_parity["benchmark_evaluation_authorized"]),
        "common_scale_retention_passed_on_all_three_subsets": bool(
            retention_decision["retention_passed"] and retention_decision["gsm1k_authorized"]
        ),
    }

    result: dict[str, Any] = {
        "schema_version": 1,
        "status_label": (
            "Provisional one-seed result pending stratified human language review and "
            "second-seed confirmation."
        ),
        "input_sha256": {
            "base_predictions": _sha256(base_predictions_path),
            "generic_predictions": _sha256(generic_predictions_path),
            "targeted_predictions": _sha256(targeted_predictions_path),
            "frozen_taxonomy": _sha256(taxonomy_path),
            "generic_summary": _sha256(generic_summary_path),
            "targeted_summary": _sha256(targeted_summary_path),
            "final_parity": _sha256(final_parity_path),
            "retention_decision": _sha256(retention_decision_path),
        },
        "examples": expected_examples,
        "correct": {
            "base": sum(base.values()),
            "generic_control": generic_correct,
            "targeted": targeted_correct,
        },
        "accuracy": {
            "base": sum(base.values()) / expected_examples,
            "generic_control": generic_correct / expected_examples,
            "targeted": targeted_correct / expected_examples,
        },
        "correct_deltas": {
            "generic_vs_base": generic_correct - sum(base.values()),
            "targeted_vs_base": targeted_correct - sum(base.values()),
            "targeted_vs_generic": targeted_correct - generic_correct,
        },
        "paired_changes": {
            "generic_fixed_base_failures": generic_fixed,
            "targeted_fixed_base_failures": targeted_fixed,
            "generic_broke_base_successes": generic_broken,
            "targeted_broke_base_successes": targeted_broken,
            "generic_only_fixed": generic_only_fixed,
            "targeted_only_fixed": targeted_only_fixed,
            "generic_only_broken": generic_only_broken,
            "targeted_only_broken": targeted_only_broken,
            "targeted_wins_over_generic": targeted_wins,
            "generic_wins_over_targeted": generic_wins,
            "targeted_net_wins": targeted_wins - generic_wins,
            "transition_counts": dict(sorted(transitions.items())),
        },
        "paired_bootstrap": {
            "estimand": "targeted_accuracy_minus_generic_accuracy",
            "point_estimate": sum(differences) / expected_examples,
            "confidence_level": 0.95,
            "interval": [bootstrap_low, bootstrap_high],
            "method": "paired_nonparametric_percentile_bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "stable_order": "lexicographic_stable_id",
        },
        "frozen_taxonomy_scope": {
            "classified_examples": len(taxonomy),
            "scope": "development_failures_only",
            "targeted_categories": sorted(TARGETED_CATEGORIES),
        },
        "category_results": category_results,
        "aggregate_untargeted_category_set": untargeted,
        "signal_gate_checks": signal_checks,
        "one_seed_signal_gate_passed": all(signal_checks.values()),
        "sealed_final_accessed": False,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["analysis_sha256"] = hashlib.sha256(canonical).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--generic-predictions", required=True, type=Path)
    parser.add_argument("--targeted-predictions", required=True, type=Path)
    parser.add_argument("--generic-summary", required=True, type=Path)
    parser.add_argument("--targeted-summary", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--final-parity", required=True, type=Path)
    parser.add_argument("--retention-decision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = analyze_paired_results(
        base_predictions_path=args.base_predictions,
        generic_predictions_path=args.generic_predictions,
        targeted_predictions_path=args.targeted_predictions,
        generic_summary_path=args.generic_summary,
        targeted_summary_path=args.targeted_summary,
        taxonomy_path=args.taxonomy,
        final_parity_path=args.final_parity,
        retention_decision_path=args.retention_decision,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
