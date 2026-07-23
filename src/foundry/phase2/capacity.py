from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

from foundry.phase2.asdiv import canonical_sha256, file_sha256
from foundry.synthesis.contamination import normalize_text

FAMILY_BOOKKEEPING: Final = "multi_step_bookkeeping_or_omission"
FAMILY_RATE: Final = "rate_ratio_percentage_or_average"
FAMILY_DISCRETE: Final = "constraint_distribution_or_discrete_reasoning"

COMBINED_FAMILY_REQUIREMENTS: Final = {
    300: {FAMILY_BOOKKEEPING: 265, FAMILY_RATE: 170, FAMILY_DISCRETE: 165},
    250: {FAMILY_BOOKKEEPING: 222, FAMILY_RATE: 141, FAMILY_DISCRETE: 137},
    200: {FAMILY_BOOKKEEPING: 177, FAMILY_RATE: 114, FAMILY_DISCRETE: 109},
}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: object = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(cast(dict[str, object], raw))
    return rows


def _string(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def _integer(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"field {key!r} must be an integer")
    return value


def operation_count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3_plus"


def depth_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3_plus"


def answer_magnitude_bucket(value: Fraction) -> str:
    magnitude = abs(value)
    if magnitude == 0:
        return "zero"
    if magnitude < 1:
        return "below_1"
    if magnitude < 10:
        return "1_to_9"
    if magnitude < 100:
        return "10_to_99"
    if magnitude < 1000:
        return "100_to_999"
    return "1000_plus"


def question_token_length(text: str) -> int:
    return len(re.findall(r"[a-z0-9]+|<num>", normalize_text(text, replace_numbers=False)))


def question_token_bucket(value: int) -> str:
    if value <= 32:
        return "1_to_32"
    if value <= 64:
        return "33_to_64"
    if value <= 96:
        return "65_to_96"
    return "97_plus"


def build_capacity_summary(clean_path: Path, output_path: Path) -> dict[str, object]:
    rows = _load_jsonl(clean_path)
    family_counts: Counter[str] = Counter()
    grade_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    magnitude_counts: Counter[str] = Counter()
    token_length_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_ids: set[str] = set()
    question_hashes: set[str] = set()
    program_hashes: set[str] = set()

    for row in rows:
        source_id = _string(row, "source_id")
        if source_id in source_ids:
            raise ValueError("clean candidates contain a duplicate source ID")
        source_ids.add(source_id)
        family_counts[_string(row, "family")] += 1
        grade_counts[_string(row, "grade")] += 1
        operation_counts[operation_count_bucket(_integer(row, "operation_count"))] += 1
        depth_counts[depth_bucket(_integer(row, "formula_depth"))] += 1
        answer_type_counts[_string(row, "answer_type")] += 1
        magnitude_counts[answer_magnitude_bucket(Fraction(_string(row, "canonical_answer")))] += 1
        token_length_counts[
            question_token_bucket(question_token_length(_string(row, "combined_question")))
        ] += 1
        source_counts["ASDiv"] += 1
        question_hashes.add(_string(row, "question_sha256"))
        program_hashes.add(_string(row, "program_sha256"))

    size_checks: dict[str, object] = {}
    eligible_sizes: list[int] = []
    for size in (300, 250, 200):
        required = COMBINED_FAMILY_REQUIREMENTS[size]
        deficits = {
            family: required_count - family_counts[family]
            for family, required_count in required.items()
            if family_counts[family] < required_count
        }
        eligible = not deficits
        if eligible:
            eligible_sizes.append(size)
        size_checks[str(size)] = {
            "per_arm_size": size,
            "combined_examples": size * 2,
            "required_family_counts": required,
            "available_family_counts": {
                family: family_counts[family] for family in sorted(required)
            },
            "deficits": deficits,
            "structurally_eligible": eligible,
        }

    summary: dict[str, object] = {
        "schema_version": 1,
        "clean_input_sha256": file_sha256(clean_path),
        "clean_count": len(rows),
        "unique_source_ids": len(source_ids),
        "unique_question_hashes": len(question_hashes),
        "unique_program_hashes": len(program_hashes),
        "counts": {
            "family": dict(sorted(family_counts.items())),
            "grade": dict(sorted(grade_counts.items())),
            "operation_count_bucket": dict(sorted(operation_counts.items())),
            "formula_depth_bucket": dict(sorted(depth_counts.items())),
            "answer_type": dict(sorted(answer_type_counts.items())),
            "answer_magnitude_bucket": dict(sorted(magnitude_counts.items())),
            "question_token_length_bucket": dict(sorted(token_length_counts.items())),
            "source_corpus": dict(sorted(source_counts.items())),
        },
        "size_checks": size_checks,
        "asdiv_only_structurally_eligible_sizes": eligible_sizes,
        "largest_structurally_eligible_size": eligible_sizes[0] if eligible_sizes else None,
        "model_evaluation_used": False,
        "mathqa_activated": False,
        "decision": "evaluate_asdiv_before_fallback_decision",
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the ASDiv-only structural capacity census")
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_capacity_summary(args.clean, args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
