"""Apply content-free Codex inspection evidence to the automatic smoke summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast


def finalize_summary(summary_path: Path, inspection_path: Path) -> dict[str, object]:
    """Freeze the final technical gate without exposing rendered questions."""

    summary_raw: object = json.loads(summary_path.read_text(encoding="utf-8"))
    inspection_raw: object = json.loads(inspection_path.read_text(encoding="utf-8"))
    if not isinstance(summary_raw, dict) or not isinstance(inspection_raw, dict):
        raise ValueError("summary and inspection must be JSON objects")
    summary = cast(dict[str, object], summary_raw)
    inspection = cast(dict[str, object], inspection_raw)
    findings = inspection.get("findings")
    if summary.get("attempted") != 120 or inspection.get("attempts_inspected") != 120:
        raise ValueError("template-bank inspection must cover exactly 120 attempts")
    if not isinstance(findings, list) or len(findings) != inspection.get(
        "invalid_or_unnatural_count"
    ):
        raise ValueError("inspection count differs from its content-free findings")
    indices = [item.get("attempt_index") for item in findings if isinstance(item, dict)]
    if len(indices) != len(findings) or len(set(indices)) != len(indices):
        raise ValueError("inspection attempt references must be complete and unique")
    summary["codex_inspection"] = {
        "status": inspection.get("inspection_kind"),
        "attempts_inspected": 120,
        "invalid_or_unnatural_count": inspection.get("invalid_or_unnatural_count"),
        "systematic_template_defect": inspection.get("systematic_template_defect"),
        "human_review_status": inspection.get("human_review_status"),
    }
    if inspection.get("systematic_template_defect") is True:
        summary["technical_gate_passed"] = False
        summary["technical_status"] = "TECHNICAL GATE FAILED"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    args = parser.parse_args()
    finalized = finalize_summary(args.summary, args.inspection)
    print(json.dumps(finalized["codex_inspection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
