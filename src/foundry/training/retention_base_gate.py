"""Audit the untouched-base usability gate for a powered retention suite."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256
from foundry.training.retention import load_suite

ALLOWED_FAILURE_CATEGORIES = {
    "genuine_terminal_contract_noncompliance",
    "genuine_format_noncompliance",
    "genuine_instruction_noncompliance",
    "ambiguous_or_underspecified_prompt",
    "reference_or_scorer_defect",
}
DEFECT_CATEGORIES = {
    "ambiguous_or_underspecified_prompt",
    "reference_or_scorer_defect",
}
SECTION_MINIMUM = 90
EXTRACTABLE_MINIMUM = 285


def _load_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return cast(dict[str, Any], value)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain row objects")
    return cast(list[dict[str, Any]], value)


def assess_base_suite(
    *,
    suite_path: Path,
    summary_path: Path,
    raw_path: Path,
    classifications_path: Path,
) -> dict[str, Any]:
    """Return content-free base-gate evidence after a blind failure classification."""

    suite = load_suite(suite_path)
    summary = _load_object(summary_path)
    rows = _load_rows(raw_path)
    classifications_root = _load_object(classifications_path)
    classifications = classifications_root.get("classifications")
    if not isinstance(classifications, dict):
        raise ValueError("base-gate classifications are required")
    if summary.get("adapter_sha256") is not None:
        raise ValueError("base gate cannot use an adapter")
    if summary.get("suite_sha256") != suite.suite_sha256:
        raise ValueError("base summary suite identity differs")
    if len(rows) != 300 or len(suite.items) != 300:
        raise ValueError("powered base gate requires exactly 300 items")
    row_ids = [row.get("id") for row in rows]
    item_ids = [item.item_id for item in suite.items]
    if row_ids != item_ids or len(set(cast(list[str], row_ids))) != 300:
        raise ValueError("base rows do not match suite order and identity")

    failed_ids: list[str] = []
    failures_by_skill: Counter[str] = Counter()
    section_correct: Counter[str] = Counter()
    failure_categories: Counter[str] = Counter()
    for item, row in zip(suite.items, rows, strict=True):
        score = row.get("score")
        if not isinstance(score, dict) or not isinstance(score.get("correct"), bool):
            raise ValueError("base row lacks an objective decision")
        if score["correct"]:
            section_correct[item.section] += 1
            continue
        failed_ids.append(item.item_id)
        failures_by_skill[f"{item.section}/{item.skill}"] += 1
        category = classifications.get(item.item_id)
        if not isinstance(category, str) or category not in ALLOWED_FAILURE_CATEGORIES:
            raise ValueError(f"missing approved classification for {item.item_id}")
        failure_categories[category] += 1
    if set(classifications) != set(failed_ids):
        raise ValueError("classification IDs differ from objective failures")

    section_gate = {
        section: section_correct[section] >= SECTION_MINIMUM
        for section in ("arithmetic", "format", "instruction")
    }
    extractable = int(summary.get("extractable", -1))
    gate_checks = {
        "arithmetic_at_least_90": section_gate["arithmetic"],
        "format_at_least_90": section_gate["format"],
        "instruction_at_least_90": section_gate["instruction"],
        "extractable_at_least_285": extractable >= EXTRACTABLE_MINIMUM,
        "zero_backend_failures": summary.get("backend_failures") == 0,
        "zero_prompt_echo": summary.get("prompt_echo") == 0,
        "zero_ambiguous_reference_answers": failure_categories["ambiguous_or_underspecified_prompt"]
        == 0,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-retention-adjudication-v2-base-usability-gate-v1",
        "suite_sha256": suite.suite_sha256,
        "suite_file_sha256": file_sha256(suite_path),
        "evaluation_summary_sha256": summary.get("summary_sha256"),
        "raw_packet_sha256": summary.get("raw_packet_sha256"),
        "classification_packet_sha256": file_sha256(classifications_path),
        "items": 300,
        "correct": sum(section_correct.values()),
        "section_correct": {
            section: section_correct[section] for section in ("arithmetic", "format", "instruction")
        },
        "extractable": extractable,
        "malformed_outputs": summary.get("malformed_outputs"),
        "backend_failures": summary.get("backend_failures"),
        "prompt_echo": summary.get("prompt_echo"),
        "question_generation": summary.get("question_generation"),
        "failure_categories": dict(sorted(failure_categories.items())),
        "failures_by_skill": dict(sorted(failures_by_skill.items())),
        "confirmed_prompt_reference_or_scorer_defects": sum(
            failure_categories[category] for category in DEFECT_CATEGORIES
        ),
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "decision": "STOP_BEFORE_ADAPTER_ADJUDICATION",
        "adapter_outputs_inspected": False,
        "holdout_evaluated": False,
        "gsm1k_evaluated": False,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--classifications", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = assess_base_suite(
        suite_path=args.suite,
        summary_path=args.summary,
        raw_path=args.raw,
        classifications_path=args.classifications,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
