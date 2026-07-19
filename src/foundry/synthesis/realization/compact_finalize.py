"""Finalize the content-free replay and readiness evidence for Milestone 5C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def finalize_compact_summary(*, summary_path: Path, replay_path: Path) -> dict[str, object]:
    """Validate immutable counts and freeze the failed gate plus one approved pivot."""

    summary: dict[str, object] = json.loads(summary_path.read_text(encoding="utf-8"))
    replay: dict[str, object] = json.loads(replay_path.read_text(encoding="utf-8"))
    if summary.get("attempted_irs") != 30 or summary.get("generated_beams") != 90:
        raise ValueError("compact summary does not account for exactly 30 IRs and 90 beams")
    if replay.get("status") != "passed" or replay.get("irs") != 30 or replay.get("beams") != 90:
        raise ValueError("compact replay did not pass exact count and identity checks")
    expected = summary.get("deterministic_run_sha256")
    if replay.get("actual_sha256") != expected or replay.get("expected_sha256") != expected:
        raise ValueError("compact replay hash differs from the counted run")
    audit = cast(dict[str, object], summary.get("manual_audit"))
    if audit.get("audited_beams") != 90:
        raise ValueError("compact manual audit is incomplete")
    if any(
        audit.get(key) != 0
        for key in ("false_labels", "invalid_acceptances", "incorrect_rejections")
    ):
        raise ValueError("compact audit contains a safety defect")
    categories = cast(dict[str, int], summary.get("automatic_acceptance_by_category"))
    summary["deterministic_replay"] = replay
    summary["readiness_gate"] = {
        "status": "failed",
        "criteria": {
            "exactly_30_irs": True,
            "at_most_90_beams": True,
            "at_least_22_clean_irs": False,
            "bookkeeping_at_least_8": categories["multi_step_bookkeeping_or_omission"] >= 8,
            "rates_at_least_6": categories["rate_ratio_percentage_or_average"] >= 6,
            "discrete_at_least_5": categories["constraint_distribution_or_discrete_reasoning"] >= 5,
            "zero_false_labels": True,
            "zero_accepted_semantic_drift": True,
            "zero_invalid_acceptances": True,
            "zero_verifier_disagreements": True,
            "zero_unresolved_contamination": True,
            "deterministic_replay_matches": True,
            "no_systematic_tagged_or_wording_defect": False,
        },
        "clean_accepted_irs": audit["clean_accepted_irs"],
        "minimum_required": 22,
        "blockers": [
            "zero clean IRs across all groups and mathematical families",
            "systematic postfixed semantic anchors and token-list echo",
            "all 90 returned beams were unnatural and semantically drifted",
        ],
        "final_qwen3_stop_rule": "active",
        "recommended_pivot": (
            "test a stronger local realization model using the same frozen compact protocol"
        ),
    }
    summary["final_ignored_raw_artifact_bytes"] = _directory_bytes(replay_path.parent)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    summary = finalize_compact_summary(summary_path=args.summary, replay_path=args.replay)
    print(json.dumps(summary["readiness_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["finalize_compact_summary"]
