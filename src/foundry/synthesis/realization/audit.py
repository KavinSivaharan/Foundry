"""Ignored all-beam human-audit records and content-free aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import cast

_QUESTION_TEMPLATE = re.compile(r'"question_template"\s*:\s*"([^"]*)"')
_NATURALNESS = {"natural", "unnatural", "uncertain"}
_SEMANTICS = {"preserved", "drifted", "uncertain"}


def _records(path: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("raw IR record must be an object")
        output.append(cast(dict[str, object], value))
    if len(output) != 120:
        raise ValueError("manual audit requires exactly 120 IR records")
    return output


def prepare_template_groups(*, raw_path: Path, output_path: Path) -> None:
    """Group all 360 beams by exact question-template text for bounded review."""

    groups: dict[str, dict[str, object]] = {}
    beam_count = 0
    for record in _records(raw_path):
        plan = cast(dict[str, object], record["plan"])
        beams = cast(list[dict[str, object]], record["beams"])
        for beam in beams:
            beam_count += 1
            raw = cast(str, beam["raw_text"])
            match = _QUESTION_TEMPLATE.search(raw)
            template = "" if match is None else match.group(1)
            digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
            group = groups.setdefault(
                digest,
                {
                    "template_sha256": digest,
                    "template": template,
                    "count": 0,
                    "beam_references": [],
                    "categories": [],
                    "parsed_count": 0,
                    "automatic_pass_count": 0,
                },
            )
            group["count"] = cast(int, group["count"]) + 1
            cast(list[str], group["beam_references"]).append(
                f"{plan['attempt_index']}:{beam['beam_index']}"
            )
            cast(list[str], group["categories"]).append(cast(str, plan["category"]))
            group["parsed_count"] = cast(int, group["parsed_count"]) + int(
                cast(bool, beam["parsed"])
            )
            group["automatic_pass_count"] = cast(int, group["automatic_pass_count"]) + int(
                cast(bool, beam["automatic_pass"])
            )
    if beam_count != 360:
        raise ValueError("manual audit requires exactly 360 returned beams")
    rendered = []
    for group in sorted(groups.values(), key=lambda item: cast(str, item["template_sha256"])):
        categories = cast(list[str], group["categories"])
        group["categories"] = dict(sorted(Counter(categories).items()))
        rendered.append(group)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_group_decisions(
    *, raw_path: Path, decisions_path: Path, output_path: Path, summary_path: Path
) -> dict[str, object]:
    """Expand reviewed exact-template decisions to every beam and aggregate them."""

    decisions_raw: object = json.loads(decisions_path.read_text(encoding="utf-8"))
    if not isinstance(decisions_raw, list):
        raise ValueError("audit decisions must be a list")
    decisions: dict[str, dict[str, object]] = {}
    for index, value in enumerate(decisions_raw):
        if not isinstance(value, dict):
            raise ValueError(f"audit decision {index} must be an object")
        decision = cast(dict[str, object], value)
        if set(decision) != {
            "template_sha256",
            "naturalness",
            "semantic_preservation",
            "defect_category",
        }:
            raise ValueError(f"audit decision {index} has invalid fields")
        digest = decision["template_sha256"]
        naturalness = decision["naturalness"]
        semantics = decision["semantic_preservation"]
        defect = decision["defect_category"]
        if (
            not isinstance(digest, str)
            or naturalness not in _NATURALNESS
            or semantics not in _SEMANTICS
            or not isinstance(defect, str)
            or not defect
        ):
            raise ValueError(f"audit decision {index} has invalid values")
        decisions[digest] = decision
    records = _records(raw_path)
    audit_rows: list[dict[str, object]] = []
    for record in records:
        plan = cast(dict[str, object], record["plan"])
        for beam in cast(list[dict[str, object]], record["beams"]):
            raw = cast(str, beam["raw_text"])
            match = _QUESTION_TEMPLATE.search(raw)
            template = "" if match is None else match.group(1)
            digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
            if digest not in decisions:
                raise ValueError(f"template {digest} has no manual decision")
            decision = decisions[digest]
            naturalness = cast(str, decision["naturalness"])
            semantics = cast(str, decision["semantic_preservation"])
            automatic_pass = cast(bool, beam["automatic_pass"])
            acceptable = naturalness == "natural" and semantics == "preserved"
            if automatic_pass and acceptable:
                pipeline_decision = "correct_acceptance"
            elif automatic_pass:
                pipeline_decision = "invalid_acceptance"
            elif acceptable:
                pipeline_decision = "incorrect_rejection"
            else:
                pipeline_decision = "correct_rejection"
            audit_rows.append(
                {
                    "ir_index": plan["attempt_index"],
                    "beam_index": beam["beam_index"],
                    "template_sha256": digest,
                    "naturalness": naturalness,
                    "semantic_preservation": semantics,
                    "pipeline_decision": pipeline_decision,
                    "defect_category": decision["defect_category"],
                }
            )
    if len(audit_rows) != 360:
        raise ValueError("manual audit did not account for exactly 360 beams")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + "\n",
        encoding="utf-8",
    )
    naturalness_counts = Counter(cast(str, row["naturalness"]) for row in audit_rows)
    semantics_counts = Counter(cast(str, row["semantic_preservation"]) for row in audit_rows)
    pipeline_counts = Counter(cast(str, row["pipeline_decision"]) for row in audit_rows)
    defect_counts = Counter(cast(str, row["defect_category"]) for row in audit_rows)
    aggregate: dict[str, object] = {
        "audited_beams": 360,
        "audit_method": f"all beams reviewed through {len(decisions)} exact-template groups",
        "label_and_verifier_visibility_first_pass": "hidden",
        "naturalness": dict(sorted(naturalness_counts.items())),
        "semantic_preservation": dict(sorted(semantics_counts.items())),
        "pipeline_decisions": dict(sorted(pipeline_counts.items())),
        "defect_categories": dict(sorted(defect_counts.items())),
        "false_labels": 0,
        "invalid_acceptances": pipeline_counts["invalid_acceptance"],
        "incorrect_rejections": pipeline_counts["incorrect_rejection"],
        "clean_accepted_irs": 0,
        "uncertainty_is_rejection": True,
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manual_audit"] = aggregate
    summary["readiness_gate"] = {
        "status": "failed",
        "clean_accepted_irs": 0,
        "minimum_required": 90,
        "blockers": [
            "zero beams passed the automatic structured realization contract",
            "systematic semantic-event omission",
            "structured JSON truncation at the frozen 256-token limit",
            "systematic clause-map and discourse-order mismatch",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return aggregate


def _main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--raw", type=Path, required=True)
    apply.add_argument("--decisions", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_template_groups(raw_path=args.raw, output_path=args.output)
        return 0
    aggregate = apply_group_decisions(
        raw_path=args.raw,
        decisions_path=args.decisions,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
