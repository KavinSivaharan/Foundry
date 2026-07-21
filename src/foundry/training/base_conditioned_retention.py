"""Freeze and assess base-conditioned retention preservation subsets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256
from foundry.training.retention import (
    BaseConditionedSubset,
    RetentionSuite,
    load_base_conditioned_subset,
    load_suite,
    score_response,
)

INSTRUMENT_ID = "foundry-base-conditioned-retention-v1"
SECTION_ORDER = ("arithmetic", "format", "instruction")
OVERALL_MINIMUM = 0.90
CATEGORY_MINIMUM = 0.90
WILSON_MINIMUM = 0.85
PROMPT_ECHO_MAXIMUM = 0.02
INSTRUCTION_FAMILY_FAILURE_MAXIMUM = 3


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


def _validated_summary(path: Path) -> dict[str, Any]:
    summary = _load_object(path)
    expected_hash = summary.get("summary_sha256")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if not isinstance(expected_hash, str) or expected_hash != canonical_sha256(payload):
        raise ValueError("retention evaluation summary hash differs")
    return summary


def _correct(row: dict[str, Any]) -> bool:
    score = row.get("score")
    if not isinstance(score, dict) or not isinstance(score.get("correct"), bool):
        raise ValueError("retention row lacks a Boolean correctness decision")
    return cast(bool, score["correct"])


def freeze_base_correct_subset(
    *,
    suite_path: Path,
    base_summary_path: Path,
    base_raw_path: Path,
    subset_id: str,
) -> dict[str, Any]:
    """Freeze ordered content-free IDs the untouched base answered correctly."""

    suite = load_suite(suite_path)
    summary = _validated_summary(base_summary_path)
    rows = _load_rows(base_raw_path)
    if summary.get("adapter_sha256") is not None:
        raise ValueError("base-conditioned subset cannot be selected from an adapter")
    if summary.get("suite_sha256") != suite.suite_sha256:
        raise ValueError("base result suite identity differs")
    if summary.get("raw_packet_sha256") != file_sha256(base_raw_path):
        raise ValueError("base raw packet hash differs")
    if summary.get("total") != len(suite.items) or len(rows) != len(suite.items):
        raise ValueError("base result must cover the complete frozen suite")
    if [row.get("id") for row in rows] != [item.item_id for item in suite.items]:
        raise ValueError("base result order or IDs differ from the frozen suite")

    selected: list[dict[str, str]] = []
    section_counts: Counter[str] = Counter()
    for item, row in zip(suite.items, rows, strict=True):
        if row.get("section") != item.section or row.get("skill") != item.skill:
            raise ValueError("base row category label differs from the frozen suite")
        if _correct(row):
            selected.append({"id": item.item_id, "section": item.section, "skill": item.skill})
            section_counts[item.section] += 1
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "instrument_id": INSTRUMENT_ID,
        "subset_id": subset_id,
        "definition": "frozen_scorer_correct_on_untouched_base",
        "suite_id": suite.suite_id,
        "suite_sha256": suite.suite_sha256,
        "suite_file_sha256": file_sha256(suite_path),
        "base_evaluation_id": summary.get("evaluation_id"),
        "base_summary_sha256": summary["summary_sha256"],
        "base_summary_file_sha256": file_sha256(base_summary_path),
        "base_raw_packet_sha256": summary["raw_packet_sha256"],
        "section_counts": {section: section_counts[section] for section in SECTION_ORDER},
        "total": len(selected),
        "items": selected,
        "adapter_outputs_read": False,
        "prompt_reference_or_scorer_modified": False,
        "prompts_or_references_in_manifest": False,
    }
    manifest["subset_sha256"] = canonical_sha256(manifest)
    return manifest


def assess_holdout_instrument_usability(
    *,
    suite_path: Path,
    base_summary_path: Path,
    artifact_evidence_path: Path,
) -> dict[str, Any]:
    """Assess the predeclared sample-size and objective-integrity holdout gate."""

    suite = load_suite(suite_path)
    summary = _validated_summary(base_summary_path)
    evidence = _load_object(artifact_evidence_path)
    if summary.get("adapter_sha256") is not None:
        raise ValueError("holdout usability must be assessed on the untouched base")
    if summary.get("suite_sha256") != suite.suite_sha256:
        raise ValueError("holdout base result suite identity differs")
    if len(suite.items) != 300 or summary.get("total") != 300:
        raise ValueError("holdout usability requires exactly 300 items")
    suites = evidence.get("suites")
    cross_artifact = evidence.get("cross_artifact")
    if not isinstance(suites, dict) or not isinstance(cross_artifact, dict):
        raise ValueError("powered-artifact evidence is incomplete")
    holdout = suites.get("anchor_holdout")
    if not isinstance(holdout, dict) or holdout.get("suite_sha256") != suite.suite_sha256:
        raise ValueError("powered-artifact evidence holdout identity differs")
    self_score_failures = sum(
        not bool(score_response(item, item.expected)["correct"]) for item in suite.items
    )
    section_metrics = summary.get("section_metrics")
    if not isinstance(section_metrics, dict):
        raise ValueError("holdout base result lacks section metrics")
    section_correct = {
        section: int(cast(dict[str, Any], section_metrics[section])["correct"])
        for section in SECTION_ORDER
    }
    ambiguous_references = int(cross_artifact.get("ambiguous_reference_answers", -1))
    gate_checks = {
        "arithmetic_at_least_40": section_correct["arithmetic"] >= 40,
        "format_at_least_40": section_correct["format"] >= 40,
        "instruction_at_least_40": section_correct["instruction"] >= 40,
        "overall_at_least_150": sum(section_correct.values()) >= 150,
        "zero_backend_failures": summary.get("backend_failures") == 0,
        "zero_reference_or_scorer_defects": self_score_failures == 0 and ambiguous_references == 0,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "gate_id": "foundry-anchor-holdout-instrument-usability-gate-v1",
        "suite_id": suite.suite_id,
        "suite_sha256": suite.suite_sha256,
        "base_summary_sha256": summary["summary_sha256"],
        "artifact_evidence_sha256": evidence.get("summary_sha256"),
        "section_correct": section_correct,
        "overall_correct": sum(section_correct.values()),
        "extractable": summary.get("extractable"),
        "prompt_echo": summary.get("prompt_echo"),
        "question_generation": summary.get("question_generation"),
        "malformed_outputs": summary.get("malformed_outputs"),
        "backend_failures": summary.get("backend_failures"),
        "reference_self_score_failures": self_score_failures,
        "ambiguous_reference_answers": ambiguous_references,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "adapter_outputs_read": False,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def wilson_lower_bound(successes: int, total: int) -> float:
    """Return the standard two-sided 95% Wilson interval lower bound."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson inputs are outside their valid range")
    z = 1.959963984540054
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    centre = proportion + z_squared / (2.0 * total)
    spread = z * math.sqrt((proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total)
    return (centre - spread) / denominator


def _validate_adapter_result(
    *,
    suite: RetentionSuite,
    subset: BaseConditionedSubset,
    summary_path: Path,
    raw_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _validated_summary(summary_path)
    rows = _load_rows(raw_path)
    if not isinstance(summary.get("adapter_sha256"), str):
        raise ValueError("preservation assessment requires an adapter result")
    if summary.get("suite_sha256") != suite.suite_sha256:
        raise ValueError("adapter result suite identity differs")
    if summary.get("base_conditioned_subset_sha256") != subset.subset_sha256:
        raise ValueError("adapter result base-conditioned subset differs")
    if summary.get("raw_packet_sha256") != file_sha256(raw_path):
        raise ValueError("adapter raw packet hash differs")
    expected_ids = [item[0] for item in subset.items]
    if summary.get("total") != len(expected_ids) or len(rows) != len(expected_ids):
        raise ValueError("adapter result does not cover the complete subset")
    if [row.get("id") for row in rows] != expected_ids:
        raise ValueError("adapter result order or IDs differ from the subset")
    return summary, rows


def assess_preservation(
    *,
    suite_path: Path,
    subset_manifest_path: Path,
    summary_path: Path,
    raw_path: Path,
) -> dict[str, Any]:
    """Assess one adapter against one immutable base-correct subset."""

    suite = load_suite(suite_path)
    subset = load_base_conditioned_subset(subset_manifest_path, suite)
    summary, rows = _validate_adapter_result(
        suite=suite,
        subset=subset,
        summary_path=summary_path,
        raw_path=raw_path,
    )
    item_index = {item[0]: item for item in subset.items}
    correct_by_section: Counter[str] = Counter()
    total_by_section: Counter[str] = Counter()
    broken_by_instruction_family: Counter[str] = Counter()
    broken_ids: list[str] = []
    for row in rows:
        item_id = cast(str, row["id"])
        _, section, skill = item_index[item_id]
        total_by_section[section] += 1
        if _correct(row):
            correct_by_section[section] += 1
        else:
            broken_ids.append(item_id)
            if section == "instruction":
                broken_by_instruction_family[skill] += 1
    preserved = sum(correct_by_section.values())
    total = len(rows)
    section_preservation = {
        section: {
            "preserved": correct_by_section[section],
            "total": total_by_section[section],
            "rate": correct_by_section[section] / total_by_section[section],
        }
        for section in SECTION_ORDER
    }
    prompt_echo = int(summary.get("prompt_echo", -1))
    gate_checks = {
        "overall_preservation_at_least_90_percent": preserved / total >= OVERALL_MINIMUM,
        "arithmetic_preservation_at_least_90_percent": section_preservation["arithmetic"]["rate"]
        >= CATEGORY_MINIMUM,
        "format_preservation_at_least_90_percent": section_preservation["format"]["rate"]
        >= CATEGORY_MINIMUM,
        "instruction_preservation_at_least_90_percent": section_preservation["instruction"]["rate"]
        >= CATEGORY_MINIMUM,
        "overall_wilson_lower_bound_at_least_85_percent": wilson_lower_bound(preserved, total)
        >= WILSON_MINIMUM,
        "prompt_echo_at_most_2_percent": prompt_echo / total <= PROMPT_ECHO_MAXIMUM,
        "zero_question_generation": summary.get("question_generation") == 0,
        "zero_backend_failures": summary.get("backend_failures") == 0,
        "instruction_family_adapter_only_failures_at_most_3": max(
            broken_by_instruction_family.values(), default=0
        )
        <= INSTRUCTION_FAMILY_FAILURE_MAXIMUM,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "assessment_id": "foundry-base-conditioned-retention-assessment-v1",
        "instrument_id": INSTRUMENT_ID,
        "suite_id": suite.suite_id,
        "suite_sha256": suite.suite_sha256,
        "subset_id": subset.subset_id,
        "subset_sha256": subset.subset_sha256,
        "adapter_sha256": summary["adapter_sha256"],
        "evaluation_summary_sha256": summary["summary_sha256"],
        "raw_packet_sha256": summary["raw_packet_sha256"],
        "total": total,
        "preserved": preserved,
        "broken": total - preserved,
        "overall_preservation": preserved / total,
        "overall_wilson_95_lower_bound": wilson_lower_bound(preserved, total),
        "section_preservation": section_preservation,
        "extractable": summary.get("extractable"),
        "extractability": summary.get("extractability"),
        "prompt_echo": prompt_echo,
        "prompt_echo_rate": prompt_echo / total,
        "question_generation": summary.get("question_generation"),
        "malformed_outputs": summary.get("malformed_outputs"),
        "backend_failures": summary.get("backend_failures"),
        "paired_transitions": {
            "base_pass_adapter_pass": preserved,
            "base_pass_adapter_fail": total - preserved,
        },
        "broken_item_ids": broken_ids,
        "instruction_family_adapter_only_failures": dict(
            sorted(broken_by_instruction_family.items())
        ),
        "maximum_instruction_family_adapter_only_failures": max(
            broken_by_instruction_family.values(), default=0
        ),
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def build_pair_decision(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the final two-adapter by two-subset retention decision."""

    if len(assessments) != 4:
        raise ValueError("retention decision requires exactly four assessments")
    keys = {(str(item.get("adapter_sha256")), str(item.get("suite_id"))) for item in assessments}
    if len({key[0] for key in keys}) != 2 or len({key[1] for key in keys}) != 2:
        raise ValueError("retention decision requires two adapters on two suites")
    if len(keys) != 4:
        raise ValueError("retention assessment matrix is incomplete or duplicated")
    gate_passed = all(item.get("gate_passed") is True for item in assessments)
    result: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": "foundry-base-conditioned-retention-pair-decision-v1",
        "instrument_id": INSTRUMENT_ID,
        "assessment_summary_sha256s": [str(item["summary_sha256"]) for item in assessments],
        "adapter_sha256s": sorted({key[0] for key in keys}),
        "suite_ids": sorted({key[1] for key in keys}),
        "all_four_assessments_passed": gate_passed,
        "decision": (
            "base_conditioned_retention_approved_short_run_adapters"
            if gate_passed
            else "failed_base_conditioned_retention"
        ),
        "gsm1k_authorized": gate_passed,
        "sft_adaptation_line_stopped": not gate_passed,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--suite", required=True, type=Path)
    freeze.add_argument("--base-summary", required=True, type=Path)
    freeze.add_argument("--base-raw", required=True, type=Path)
    freeze.add_argument("--subset-id", required=True)
    freeze.add_argument("--output", required=True, type=Path)
    assess = subparsers.add_parser("assess")
    assess.add_argument("--suite", required=True, type=Path)
    assess.add_argument("--subset", required=True, type=Path)
    assess.add_argument("--summary", required=True, type=Path)
    assess.add_argument("--raw", required=True, type=Path)
    assess.add_argument("--output", required=True, type=Path)
    decide = subparsers.add_parser("decide")
    decide.add_argument("--assessment", required=True, action="append", type=Path)
    decide.add_argument("--output", required=True, type=Path)
    base_gate = subparsers.add_parser("base-gate")
    base_gate.add_argument("--suite", required=True, type=Path)
    base_gate.add_argument("--base-summary", required=True, type=Path)
    base_gate.add_argument("--artifact-evidence", required=True, type=Path)
    base_gate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_base_correct_subset(
            suite_path=args.suite,
            base_summary_path=args.base_summary,
            base_raw_path=args.base_raw,
            subset_id=args.subset_id,
        )
    elif args.command == "assess":
        result = assess_preservation(
            suite_path=args.suite,
            subset_manifest_path=args.subset,
            summary_path=args.summary,
            raw_path=args.raw,
        )
    elif args.command == "decide":
        result = build_pair_decision([_validated_summary(path) for path in args.assessment])
    else:
        result = assess_holdout_instrument_usability(
            suite_path=args.suite,
            base_summary_path=args.base_summary,
            artifact_evidence_path=args.artifact_evidence,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
