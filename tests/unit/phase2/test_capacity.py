import json
from pathlib import Path

from foundry.phase2.capacity import (
    FAMILY_BOOKKEEPING,
    FAMILY_DISCRETE,
    FAMILY_RATE,
    answer_magnitude_bucket,
    build_capacity_summary,
    depth_bucket,
    operation_count_bucket,
    question_token_bucket,
)


def test_covariate_buckets_are_frozen() -> None:
    from fractions import Fraction

    assert operation_count_bucket(0) == "0"
    assert operation_count_bucket(3) == "3_plus"
    assert depth_bucket(2) == "2"
    assert answer_magnitude_bucket(Fraction(1, 2)) == "below_1"
    assert answer_magnitude_bucket(Fraction(1000)) == "1000_plus"
    assert question_token_bucket(32) == "1_to_32"
    assert question_token_bucket(33) == "33_to_64"


def test_capacity_requires_combined_disjoint_arm_quotas(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    family_counts = {FAMILY_BOOKKEEPING: 177, FAMILY_RATE: 113, FAMILY_DISCRETE: 109}
    index = 0
    for family, count in family_counts.items():
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "source_id": f"id-{index:04d}",
                    "family": family,
                    "grade": "3",
                    "operation_count": 1,
                    "formula_depth": 1,
                    "answer_type": "integer",
                    "canonical_answer": "4",
                    "combined_question": "A short original arithmetic prompt.",
                    "question_sha256": f"question-{index:04d}",
                    "program_sha256": f"program-{index:04d}",
                }
            )
    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    summary = build_capacity_summary(clean, tmp_path / "summary.json")

    check = summary["size_checks"]
    assert isinstance(check, dict)
    smallest = check["200"]
    assert isinstance(smallest, dict)
    assert smallest["structurally_eligible"] is False
    assert smallest["deficits"] == {FAMILY_RATE: 1}
    assert summary["asdiv_only_structurally_eligible_sizes"] == []
    assert summary["decision"] == "evaluate_asdiv_before_fallback_decision"
