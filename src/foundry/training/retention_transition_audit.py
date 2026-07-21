"""Pairwise audit of a frozen retention instruction slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.retention import load_suite

ALLOWED_CATEGORIES = {
    "genuine_instruction_noncompliance",
    "exact_match_formatting_defect",
    "ambiguous_or_underspecified_prompt",
    "reference_or_scorer_defect",
    "partial_compliance",
    "repetition_or_unrelated_output",
    "another_directly_evidenced_category",
}


def _load_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return cast(dict[str, Any], value)


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a row list")
    rows: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise ValueError(f"{path} contains an invalid row")
        row = cast(dict[str, Any], raw)
        item_id = cast(str, row["id"])
        if item_id in rows:
            raise ValueError(f"{path} contains a duplicate ID")
        rows[item_id] = row
    return rows


def _correct(row: dict[str, Any]) -> bool:
    score = row.get("score")
    if not isinstance(score, dict) or not isinstance(score.get("correct"), bool):
        raise ValueError("audit row lacks a Boolean correctness decision")
    return cast(bool, score["correct"])


def build_instruction_transition_audit(
    *,
    suite_path: Path,
    base_path: Path,
    generic_path: Path,
    targeted_path: Path,
    classifications_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return content-free aggregate evidence plus a content-bearing local packet."""

    suite = _load_object(suite_path)
    raw_items = suite.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("suite lacks items")
    items = [
        cast(dict[str, Any], item)
        for item in raw_items
        if isinstance(item, dict) and item.get("section") == "instruction"
    ]
    if len(items) != 25:
        raise ValueError("powered audit requires the frozen 25-item instruction slice")
    classifications_raw = _load_object(classifications_path)
    classifications = classifications_raw.get("classifications")
    if not isinstance(classifications, dict):
        raise ValueError("classification packet lacks classifications")
    base = _load_rows(base_path)
    generic = _load_rows(generic_path)
    targeted = _load_rows(targeted_path)
    item_ids = [cast(str, item["id"]) for item in items]
    if any(set(rows) & set(item_ids) != set(item_ids) for rows in (base, generic, targeted)):
        raise ValueError("evaluation rows do not cover the instruction slice")

    detailed: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    failure_categories: Counter[str] = Counter()
    adapter_failure_ids: set[str] = set()
    shared_adapter_failures = 0
    generic_unique_failures = 0
    targeted_unique_failures = 0
    genuine_regressions: set[str] = set()
    for item in items:
        item_id = cast(str, item["id"])
        decisions = {
            "base": _correct(base[item_id]),
            "generic_control": _correct(generic[item_id]),
            "targeted": _correct(targeted[item_id]),
        }
        transition = "/".join("pass" if decisions[name] else "fail" for name in decisions)
        transitions[transition] += 1
        category: str | None = None
        if not decisions["generic_control"] or not decisions["targeted"]:
            adapter_failure_ids.add(item_id)
            raw_category = classifications.get(item_id)
            if not isinstance(raw_category, str) or raw_category not in ALLOWED_CATEGORIES:
                raise ValueError(f"missing approved classification for {item_id}")
            category = raw_category
            failure_categories[category] += 1
        if not decisions["generic_control"] and not decisions["targeted"]:
            shared_adapter_failures += 1
        elif not decisions["generic_control"]:
            generic_unique_failures += 1
        elif not decisions["targeted"]:
            targeted_unique_failures += 1
        if decisions["base"] and (not decisions["generic_control"] or not decisions["targeted"]):
            genuine_regressions.add(item_id)
        detailed.append(
            {
                "candidate_id": item_id,
                "skill": item.get("skill"),
                "prompt": item.get("prompt"),
                "expected": item.get("expected"),
                "expected_kind": item.get("kind"),
                "decisions": decisions,
                "classification": category,
                "outputs": {
                    name: {
                        "response": rows[item_id].get("response"),
                        "response_sha256": rows[item_id].get("response_sha256"),
                        "score": rows[item_id].get("score"),
                    }
                    for name, rows in (
                        ("base", base),
                        ("generic_control", generic),
                        ("targeted", targeted),
                    )
                },
            }
        )
    base_success = {item_id for item_id in item_ids if _correct(base[item_id])}
    generic_success = {item_id for item_id in item_ids if _correct(generic[item_id])}
    targeted_success = {item_id for item_id in item_ids if _correct(targeted[item_id])}
    raw_sha256 = hashlib.sha256(
        (json.dumps(detailed, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-retention-validation-instruction-transition-audit-v1",
        "suite_sha256": load_suite(suite_path).suite_sha256,
        "instruction_items": len(item_ids),
        "correct": {
            "base": len(base_success),
            "generic_control": len(generic_success),
            "targeted": len(targeted_success),
        },
        "transitions": dict(sorted(transitions.items())),
        "base_only_successes": len(base_success - generic_success - targeted_success),
        "generic_only_successes": len(generic_success - base_success - targeted_success),
        "targeted_only_successes": len(targeted_success - base_success - generic_success),
        "shared_adapter_failures": shared_adapter_failures,
        "generic_unique_failures": generic_unique_failures,
        "targeted_unique_failures": targeted_unique_failures,
        "adapter_failure_item_ids": sorted(adapter_failure_ids),
        "failure_categories": dict(sorted(failure_categories.items())),
        "confirmed_prompt_or_scorer_defects": sum(
            failure_categories[name]
            for name in (
                "ambiguous_or_underspecified_prompt",
                "reference_or_scorer_defect",
            )
        ),
        "genuine_behavior_regressions": len(genuine_regressions),
        "genuine_behavior_regression_item_ids": sorted(genuine_regressions),
        "raw_packet_sha256": raw_sha256,
        "old_validation_gate_preserved_failed": True,
        "gsm1k_consulted": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary, detailed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--generic", required=True, type=Path)
    parser.add_argument("--targeted", required=True, type=Path)
    parser.add_argument("--classifications", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()
    summary, detailed = build_instruction_transition_audit(
        suite_path=args.suite,
        base_path=args.base,
        generic_path=args.generic,
        targeted_path=args.targeted,
        classifications_path=args.classifications,
    )
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(
        json.dumps(detailed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
