"""Validate and aggregate the required 90-beam blinded Milestone 5D audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import cast


def _rows(path: Path) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def record_uniform_rejection_decisions(
    *, raw_path: Path, decisions_path: Path, defect_category: str
) -> None:
    """Persist the explicit all-unnatural/all-drifted human audit decision."""

    if not defect_category.strip():
        raise ValueError("uniform audit requires a content-free defect category")
    decisions: list[dict[str, object]] = []
    for record in _rows(raw_path):
        plan = cast(dict[str, object], record["plan"])
        for beam in cast(list[dict[str, object]], record["beams"]):
            decisions.append(
                {
                    "ir_index": plan["attempt_index"],
                    "beam_index": beam["beam_index"],
                    "raw_sha256": beam["raw_sha256"],
                    "naturalness": "unnatural",
                    "semantic_preservation": "drifted",
                    "false_label": False,
                    "defect_category": defect_category,
                }
            )
    if len(decisions) != 90:
        raise ValueError("uniform audit must record exactly 90 human decisions")
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in decisions) + "\n",
        encoding="utf-8",
    )


def apply_stronger_model_audit(
    *,
    raw_path: Path,
    decisions_path: Path,
    output_path: Path,
    summary_path: Path,
    systematic_wording_defect: bool,
) -> dict[str, object]:
    """Join explicit blinded decisions to all 90 returned beams and freeze counts."""

    records = _rows(raw_path)
    decisions = _rows(decisions_path)
    if len(records) != 30 or len(decisions) != 90:
        raise ValueError("stronger-model audit requires 30 IR records and 90 decisions")
    decision_index: dict[tuple[int, int, str], dict[str, object]] = {}
    for decision in decisions:
        naturalness = decision.get("naturalness")
        preservation = decision.get("semantic_preservation")
        if naturalness not in {"natural", "unnatural", "uncertain"}:
            raise ValueError("invalid naturalness decision")
        if preservation not in {"preserved", "drifted", "uncertain"}:
            raise ValueError("invalid semantic-preservation decision")
        if not isinstance(decision.get("defect_category"), str):
            raise ValueError("audit decision requires a content-free defect category")
        if not isinstance(decision.get("false_label"), bool):
            raise ValueError("audit decision requires an explicit false-label Boolean")
        key = (
            cast(int, decision["ir_index"]),
            cast(int, decision["beam_index"]),
            cast(str, decision["raw_sha256"]),
        )
        if key in decision_index:
            raise ValueError("duplicate stronger-model audit decision")
        decision_index[key] = decision
    audit_rows: list[dict[str, object]] = []
    clean_irs: list[dict[str, object]] = []
    seen: set[tuple[int, int, str]] = set()
    for record in records:
        plan = cast(dict[str, object], record["plan"])
        ir_index = cast(int, plan["attempt_index"])
        beams = cast(list[dict[str, object]], record["beams"])
        if len(beams) != 3:
            raise ValueError("every stronger-model IR must have exactly three beams")
        for beam in beams:
            key = (ir_index, cast(int, beam["beam_index"]), cast(str, beam["raw_sha256"]))
            if key not in decision_index:
                raise ValueError("audit decision is missing or has the wrong beam hash")
            seen.add(key)
            decision = decision_index[key]
            natural = decision["naturalness"] == "natural"
            preserved = decision["semantic_preservation"] == "preserved"
            false_label = cast(bool, decision["false_label"])
            acceptable = natural and preserved and not false_label
            automatic = cast(bool, beam["automatic_pass"])
            if automatic and acceptable:
                pipeline = "correct_acceptance"
            elif automatic:
                pipeline = "invalid_acceptance"
            elif acceptable:
                pipeline = "incorrect_rejection"
            else:
                pipeline = "correct_rejection"
            row = {
                **decision,
                "pipeline_decision": pipeline,
                "automatic_pass": automatic,
                "selected": beam["selected"],
            }
            audit_rows.append(row)
            if cast(bool, beam["selected"]) and acceptable:
                clean_irs.append(plan)
    if seen != set(decision_index) or len(audit_rows) != 90:
        raise ValueError("stronger-model audit decision set does not equal returned beams")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + "\n",
        encoding="utf-8",
    )
    naturalness_counts = Counter(cast(str, row["naturalness"]) for row in audit_rows)
    preservation_counts = Counter(cast(str, row["semantic_preservation"]) for row in audit_rows)
    pipeline_counts = Counter(cast(str, row["pipeline_decision"]) for row in audit_rows)
    defect_counts = Counter(cast(str, row["defect_category"]) for row in audit_rows)
    by_group = Counter(cast(str, plan["group"]) for plan in clean_irs)
    by_category = Counter(cast(str, plan["category"]) for plan in clean_irs)
    by_difficulty = Counter(cast(str, plan["difficulty"]) for plan in clean_irs)
    by_output = Counter(
        "enabled" if cast(bool, plan["output_contract_enabled"]) else "disabled"
        for plan in clean_irs
    )
    aggregate: dict[str, object] = {
        "status": "complete",
        "audited_beams": 90,
        "audit_method": "all beams reviewed with canonical answers and verifier evidence hidden",
        "label_and_verifier_visibility_first_pass": "hidden",
        "naturalness": dict(sorted(naturalness_counts.items())),
        "semantic_preservation": dict(sorted(preservation_counts.items())),
        "pipeline_decisions": dict(sorted(pipeline_counts.items())),
        "defect_categories": dict(sorted(defect_counts.items())),
        "false_labels": sum(cast(bool, row["false_label"]) for row in audit_rows),
        "semantic_drift_outputs": preservation_counts["drifted"],
        "invalid_acceptances": pipeline_counts["invalid_acceptance"],
        "incorrect_rejections": pipeline_counts["incorrect_rejection"],
        "clean_accepted_irs": len(clean_irs),
        "clean_acceptance_by_group": dict(sorted(by_group.items())),
        "clean_acceptance_by_category": dict(sorted(by_category.items())),
        "clean_acceptance_by_difficulty": dict(sorted(by_difficulty.items())),
        "clean_acceptance_by_output_track": dict(sorted(by_output.items())),
        "systematic_wording_defect": systematic_wording_defect,
        "uncertainty_is_rejection": True,
    }
    summary: dict[str, object] = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manual_audit"] = aggregate
    summary["readiness_gate"] = {"status": "pending_exact_replay"}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--systematic-wording-defect", action="store_true")
    parser.add_argument("--record-uniform-rejections", action="store_true")
    parser.add_argument(
        "--uniform-defect-category", default="systematic_compact_protocol_nonrealization"
    )
    args = parser.parse_args()
    if args.record_uniform_rejections:
        record_uniform_rejection_decisions(
            raw_path=args.raw,
            decisions_path=args.decisions,
            defect_category=args.uniform_defect_category,
        )
    aggregate = apply_stronger_model_audit(
        raw_path=args.raw,
        decisions_path=args.decisions,
        output_path=args.output,
        summary_path=args.summary,
        systematic_wording_defect=args.systematic_wording_defect,
    )
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["apply_stronger_model_audit", "record_uniform_rejection_decisions"]
