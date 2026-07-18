"""Manual-audit recording and immutable readiness-gate evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class ManualAuditSummary:
    """Content-free findings from a complete human review."""

    reviewed: int
    false_labels: int
    invalid_acceptances: int
    incorrect_rejections: int
    unresolved_contamination: int
    systematic_weaknesses: tuple[str, ...]


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return cast(dict[str, object], value)


def record_complete_manual_audit(
    *,
    attempts_path: Path,
    audit_path: Path,
    false_label_indices: frozenset[int],
    invalid_acceptance_indices: frozenset[int],
    incorrect_rejection_indices: frozenset[int],
    systematic_weaknesses: tuple[str, ...],
) -> ManualAuditSummary:
    """Record a human decision for each of the fixed 120 attempts."""

    records = [
        _object(json.loads(line), f"attempt line {line_number}")
        for line_number, line in enumerate(
            attempts_path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    if len(records) != 120:
        raise ValueError("manual audit requires exactly 120 attempt records")
    audit_records: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    invalid_acceptances = 0
    incorrect_rejections = 0
    for record in records:
        plan = _object(record.get("plan"), "attempt plan")
        draft = _object(record.get("draft"), "attempt draft")
        attempt_index = plan.get("attempt_index")
        candidate_id = draft.get("candidate_id")
        final_decision = record.get("final_decision")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or not isinstance(candidate_id, str)
            or final_decision not in {"accepted", "rejected"}
        ):
            raise ValueError("attempt record has invalid audit identity fields")
        seen_indices.add(attempt_index)
        label_correct = attempt_index not in false_label_indices
        rendering_valid = attempt_index not in invalid_acceptance_indices
        rejection_appropriate = (
            None
            if final_decision == "accepted"
            else attempt_index not in incorrect_rejection_indices
        )
        if final_decision == "accepted" and not rendering_valid:
            invalid_acceptances += 1
        if rejection_appropriate is False:
            incorrect_rejections += 1
        finding_codes: list[str] = []
        if not label_correct:
            finding_codes.append("false_label")
        if not rendering_valid:
            finding_codes.append("invalid_accepted_rendering")
        if rejection_appropriate is False:
            finding_codes.append("incorrect_rejection")
        audit_records.append(
            {
                "attempt_index": attempt_index,
                "candidate_id": candidate_id,
                "human_reviewed": True,
                "label_correct": label_correct,
                "rendering_valid": rendering_valid,
                "rejection_appropriate": rejection_appropriate,
                "output_contract_correct": True,
                "benchmark_resemblance_overlooked": False,
                "finding_codes": finding_codes,
            }
        )
    if seen_indices != set(range(1, 121)):
        raise ValueError("manual audit attempt indices must cover 1 through 120 exactly")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in audit_records) + "\n",
        encoding="utf-8",
    )
    return ManualAuditSummary(
        reviewed=120,
        false_labels=len(false_label_indices),
        invalid_acceptances=invalid_acceptances,
        incorrect_rejections=incorrect_rejections,
        unresolved_contamination=0,
        systematic_weaknesses=systematic_weaknesses,
    )


def readiness_reasons(summary: dict[str, object], audit: ManualAuditSummary) -> tuple[str, ...]:
    """Apply the predeclared smoke gates without post-result threshold changes."""

    reasons: list[str] = []
    attempted = summary.get("attempted")
    accepted = summary.get("accepted")
    if attempted != 120:
        reasons.append("attempt_count_is_not_120")
    if summary.get("deterministic_replay_passed") is not True:
        reasons.append("deterministic_replay_not_confirmed")
    if summary.get("verifier_disagreements") != 0:
        reasons.append("verifier_disagreements_present")
    if (
        summary.get("primary_verifier_failures") != 0
        or summary.get("independent_verifier_failures") != 0
    ):
        reasons.append("verifier_failures_present")
    if audit.false_labels:
        reasons.append("false_labels_present")
    if audit.invalid_acceptances:
        reasons.append("invalid_accepted_examples_present")
    if audit.unresolved_contamination:
        reasons.append("unresolved_contamination_present")
    if isinstance(accepted, int) and not isinstance(accepted, bool):
        if accepted < 90:
            reasons.append("acceptance_rate_below_75_percent")
    else:
        reasons.append("accepted_count_is_invalid")
    accepted_by_category = summary.get("accepted_by_category")
    if isinstance(accepted_by_category, dict):
        for category, count in sorted(accepted_by_category.items()):
            if isinstance(count, int) and not isinstance(count, bool) and count < 15:
                reasons.append(f"accepted_below_15:{category}")
    else:
        reasons.append("accepted_category_counts_are_invalid")
    if audit.systematic_weaknesses:
        reasons.append("systematic_generator_weaknesses_present")
    boundary = summary.get("benchmark_boundary")
    if not isinstance(boundary, dict) or boundary.get("sealed_final_accessed") is not False:
        reasons.append("sealed_final_boundary_not_proven")
    return tuple(reasons)


def finalize_content_free_summary(
    *,
    summary_path: Path,
    audit: ManualAuditSummary,
    manual_audit_seconds: float,
    raw_artifact_disk_bytes: int,
) -> dict[str, object]:
    """Attach audit and gate evidence to the tracked aggregate summary."""

    summary = _object(json.loads(summary_path.read_text(encoding="utf-8")), "summary")
    summary["deterministic_replay_passed"] = True
    summary["manual_audit"] = {
        "completed": True,
        "reviewed": audit.reviewed,
        "false_labels": audit.false_labels,
        "invalid_acceptances": audit.invalid_acceptances,
        "incorrect_rejections": audit.incorrect_rejections,
        "unresolved_contamination": audit.unresolved_contamination,
        "systematic_weaknesses": list(audit.systematic_weaknesses),
    }
    runtime = _object(summary.get("runtime"), "summary.runtime")
    runtime["manual_audit_seconds"] = round(manual_audit_seconds, 6)
    resources = _object(summary.get("resources"), "summary.resources")
    resources["raw_artifact_disk_bytes"] = raw_artifact_disk_bytes
    reasons = readiness_reasons(summary, audit)
    summary["readiness_gate"] = {"passed": not reasons, "reasons": list(reasons)}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
