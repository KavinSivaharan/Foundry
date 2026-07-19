"""Validate and aggregate genuine template-bank human review without content leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

EXPECTED_REVIEW_SHA256 = "564a8ca584984ee7a0b997eec4a6a6f377308c869b62cf65ebeef5375cef0791"
EXPECTED_REVIEW_COUNT = 120
EXPECTED_DECISIONS = {"approve": 60, "reject": 60, "unsure": 0}


@dataclass(frozen=True)
class PlanDisposition:
    """Content-free interpretation of one reviewed sentence-plan family."""

    plan_id: str
    classification: str
    reason_category: str
    replacement_plan_id: str | None


PLAN_DISPOSITIONS = (
    PlanDisposition(
        "chronological_active",
        "systematically_defective",
        "update_log_wording",
        "direct_changes",
    ),
    PlanDisposition(
        "ledger_passive",
        "compatibility_limited",
        "transfer_record_and_inanimate_target_wording",
        "plain_location_changes",
    ),
    PlanDisposition(
        "movement_first",
        "systematically_defective",
        "scheduled_movement_and_filler_wording",
        "direct_sequence",
    ),
    PlanDisposition(
        "register_summary",
        "removable_phrase_defect",
        "register_filler_and_group_source_ambiguity",
        "compact_sequence",
    ),
    PlanDisposition("direct_relation", "approved_only", "none", None),
    PlanDisposition(
        "operator_record",
        "compatibility_limited",
        "weighted_average_wording",
        "actor_context",
    ),
    PlanDisposition(
        "condition_first",
        "compatibility_limited",
        "ratio_percentage_and_weighted_average_wording",
        "relation_rephrased",
    ),
    PlanDisposition(
        "paired_summary",
        "compatibility_limited",
        "ratio_side_wording",
        "balanced_facts",
    ),
    PlanDisposition("constraint_sequence", "approved_only", "none", None),
    PlanDisposition(
        "planner_brief",
        "compatibility_limited",
        "two_type_noun_elision",
        "actor_setup",
    ),
    PlanDisposition(
        "condition_fronted",
        "systematically_defective",
        "task_can_proceed_wording",
        "direct_conditions",
    ),
    PlanDisposition(
        "operations_note",
        "compatibility_limited",
        "production_and_complete_group_wording",
        "alternate_direct",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _load_attempts(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("attempt records must be JSON objects")
        rows.append(cast(dict[str, Any], value))
    return tuple(rows)


def _breakdown(rows: tuple[dict[str, Any], ...], key: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result: dict[str, dict[str, object]] = {}
    for label, subset in sorted(grouped.items()):
        approved = sum(row["decision"] == "approve" for row in subset)
        rejected = sum(row["decision"] == "reject" for row in subset)
        result[label] = {
            "attempted": len(subset),
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / len(subset),
        }
    return result


def import_human_review(
    *, review_path: Path, attempts_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the genuine export and return content-free summary plus manifest."""

    review_hash = _sha256(review_path)
    if review_hash != EXPECTED_REVIEW_SHA256:
        raise ValueError("human-review SHA-256 differs from the approved export")
    review = _load_json_object(review_path)
    if review.get("schema_version") != 1:
        raise ValueError("human-review schema version differs")
    if review.get("review_kind") != "genuine_user_human_review":
        raise ValueError("human-review kind differs")
    candidates = review.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_REVIEW_COUNT:
        raise ValueError("human-review record count differs")
    decisions: dict[str, str] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("human-review candidates must be objects")
        candidate_id = candidate.get("candidate_id")
        decision = candidate.get("decision")
        if not isinstance(candidate_id, str) or candidate_id in decisions:
            raise ValueError("human-review candidate IDs must be unique strings")
        if decision not in EXPECTED_DECISIONS:
            raise ValueError("human-review decision is invalid")
        decisions[candidate_id] = cast(str, decision)
    counts = Counter(decisions.values())
    if {key: counts.get(key, 0) for key in EXPECTED_DECISIONS} != EXPECTED_DECISIONS:
        raise ValueError("human-review decision counts differ")

    attempts = _load_attempts(attempts_path)
    attempt_ids = [str(row.get("candidate_id")) for row in attempts]
    if len(attempts) != EXPECTED_REVIEW_COUNT or len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("Milestone 6B attempt identity is invalid")
    if set(attempt_ids) != set(decisions):
        raise ValueError("human-review IDs differ from the Milestone 6B packet")
    joined = tuple({**row, "decision": decisions[str(row["candidate_id"])]} for row in attempts)

    plan_counts = _breakdown(joined, "sentence_plan_id")
    only_approved = sorted(key for key, value in plan_counts.items() if value["rejected"] == 0)
    only_rejected = sorted(key for key, value in plan_counts.items() if value["approved"] == 0)
    mixed = sorted(
        key for key, value in plan_counts.items() if value["approved"] and value["rejected"]
    )
    approved_rows = tuple(row for row in joined if row["decision"] == "approve")
    rejected_rows = tuple(row for row in joined if row["decision"] == "reject")
    approved_keys = sorted(
        f"{row['template_id']}/{row['sentence_plan_id']}" for row in approved_rows
    )
    quarantined_keys = sorted(
        f"{row['template_id']}/{row['sentence_plan_id']}" for row in rejected_rows
    )

    summary: dict[str, object] = {
        "schema_version": 1,
        "review_kind": "genuine_user_human_review",
        "review_export_sha256": review_hash,
        "packet_identity_match": True,
        "attempted": len(joined),
        "approved": counts["approve"],
        "rejected": counts["reject"],
        "unsure": counts.get("unsure", 0),
        "approval_rate": counts["approve"] / len(joined),
        "rejection_rate": counts["reject"] / len(joined),
        "by_group": _breakdown(joined, "group"),
        "by_reasoning_family": _breakdown(joined, "category"),
        "by_difficulty": _breakdown(joined, "difficulty"),
        "by_output_contract": _breakdown(joined, "output_contract_enabled"),
        "by_template_id": _breakdown(joined, "template_id"),
        "by_sentence_plan_id": plan_counts,
        "plans_producing_only_approvals": only_approved,
        "plans_producing_only_rejections": only_rejected,
        "plans_producing_mixed_decisions": mixed,
        "root_causes_are_inferred_not_user_labels": True,
        "full_generation_gate_passed": False,
        "full_generation_gate_status": "FAILED_GENUINE_HUMAN_LANGUAGE_REVIEW",
    }
    summary["summary_sha256"] = _canonical_sha256(summary)

    dispositions = [
        {
            **disposition.__dict__,
            **plan_counts[disposition.plan_id],
        }
        for disposition in PLAN_DISPOSITIONS
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": "foundry-template-bank-human-review-v2",
        "review_export_sha256": review_hash,
        "approved_template_ids": sorted({str(row["template_id"]) for row in approved_rows}),
        "approved_sentence_plan_ids": sorted(
            {str(row["sentence_plan_id"]) for row in approved_rows}
        ),
        "approved_reviewed_plan_keys": approved_keys,
        "quarantined_template_ids": sorted({str(row["template_id"]) for row in rejected_rows}),
        "quarantined_sentence_plan_ids": sorted(
            {str(row["sentence_plan_id"]) for row in rejected_rows}
        ),
        "quarantined_reviewed_plan_keys": quarantined_keys,
        "plan_dispositions": dispositions,
        "unreviewed_plan_policy": "human_review_pending",
        "historical_definitions_retained_in_git": True,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return summary, manifest


def write_review_evidence(
    *, review_path: Path, attempts_path: Path, summary_path: Path, manifest_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Write only content-free, review-derived evidence."""

    summary, manifest = import_human_review(review_path=review_path, attempts_path=attempts_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary, manifest


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    summary, manifest = write_review_evidence(
        review_path=args.review,
        attempts_path=args.attempts,
        summary_path=args.summary,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "approved": summary["approved"],
                "rejected": summary["rejected"],
                "summary_sha256": summary["summary_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
