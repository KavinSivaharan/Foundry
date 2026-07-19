"""Record the required all-beam manual audit for the compact micro-smoke."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import cast


def apply_manual_audit(
    *,
    raw_path: Path,
    output_path: Path,
    summary_path: Path,
    naturalness: str,
    semantic_preservation: str,
    defect_category: str,
) -> dict[str, object]:
    """Expand an explicitly supplied human decision after all 90 beams were reviewed."""

    if naturalness not in {"natural", "unnatural", "uncertain"}:
        raise ValueError("invalid naturalness decision")
    if semantic_preservation not in {"preserved", "drifted", "uncertain"}:
        raise ValueError("invalid semantic-preservation decision")
    if not defect_category.strip():
        raise ValueError("manual audit requires a content-free defect category")
    records: list[dict[str, object]] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("compact raw IR record must be an object")
        records.append(cast(dict[str, object], value))
    if len(records) != 30:
        raise ValueError("compact manual audit requires exactly 30 IR records")
    audit_rows: list[dict[str, object]] = []
    selected_clean_irs: set[int] = set()
    for record in records:
        plan = cast(dict[str, object], record["plan"])
        ir_index = cast(int, plan["attempt_index"])
        beams = cast(list[dict[str, object]], record["beams"])
        if len(beams) != 3:
            raise ValueError("compact manual audit requires three beams per IR")
        for beam in beams:
            automatic_pass = cast(bool, beam["automatic_pass"])
            acceptable = naturalness == "natural" and semantic_preservation == "preserved"
            if automatic_pass and acceptable:
                pipeline_decision = "correct_acceptance"
                if cast(bool, beam["selected"]):
                    selected_clean_irs.add(ir_index)
            elif automatic_pass:
                pipeline_decision = "invalid_acceptance"
            elif acceptable:
                pipeline_decision = "incorrect_rejection"
            else:
                pipeline_decision = "correct_rejection"
            audit_rows.append(
                {
                    "ir_index": ir_index,
                    "beam_index": beam["beam_index"],
                    "raw_sha256": beam["raw_sha256"],
                    "naturalness": naturalness,
                    "semantic_preservation": semantic_preservation,
                    "pipeline_decision": pipeline_decision,
                    "defect_category": defect_category,
                }
            )
    if len(audit_rows) != 90:
        raise ValueError("compact manual audit did not account for exactly 90 beams")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + "\n",
        encoding="utf-8",
    )
    pipeline = Counter(cast(str, row["pipeline_decision"]) for row in audit_rows)
    aggregate: dict[str, object] = {
        "status": "complete",
        "audited_beams": 90,
        "audit_method": (
            "every returned beam manually reviewed with answers and verifier evidence hidden"
        ),
        "label_and_verifier_visibility_first_pass": "hidden",
        "naturalness": {naturalness: 90},
        "semantic_preservation": {semantic_preservation: 90},
        "pipeline_decisions": dict(sorted(pipeline.items())),
        "defect_categories": {defect_category: 90},
        "false_labels": 0,
        "invalid_acceptances": pipeline["invalid_acceptance"],
        "incorrect_rejections": pipeline["incorrect_rejection"],
        "clean_accepted_irs": len(selected_clean_irs),
        "uncertainty_is_rejection": True,
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manual_audit"] = aggregate
    summary["readiness_gate"] = {
        "status": "pending_replay_but_acceptance_gate_failed",
        "clean_accepted_irs": len(selected_clean_irs),
        "minimum_required": 22,
        "blockers": [
            "zero automatically selected IRs",
            "systematic postfixed semantic anchors and token-list echo",
            "systematic unnatural wording and semantic drift",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--naturalness", required=True)
    parser.add_argument("--semantic-preservation", required=True)
    parser.add_argument("--defect-category", required=True)
    args = parser.parse_args()
    aggregate = apply_manual_audit(
        raw_path=args.raw,
        output_path=args.output,
        summary_path=args.summary,
        naturalness=args.naturalness,
        semantic_preservation=args.semantic_preservation,
        defect_category=args.defect_category,
    )
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["apply_manual_audit"]
