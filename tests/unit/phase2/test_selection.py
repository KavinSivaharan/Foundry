import json
from pathlib import Path

import pytest

from foundry.phase2.capacity import FAMILY_BOOKKEEPING, FAMILY_DISCRETE, FAMILY_RATE
from foundry.phase2.selection import (
    Candidate,
    _hungarian,
    balance_report,
    matching_cost,
    output_token_bucket,
)


def _candidate(
    identifier: str, *, source: str = "MathQA", family: str = FAMILY_BOOKKEEPING
) -> Candidate:
    return Candidate(
        source_id=identifier,
        source_corpus=source,
        family=family,
        difficulty="not_available",
        solution_type="general",
        question_sha256=f"question-{identifier}",
        program_sha256=f"program-{identifier}",
        program_structure_sha256="structure",
        question_token_count=24,
        question_token_bucket="1_to_32",
        formula_depth=1,
        operation_count=2,
        answer_type="integer",
        answer_magnitude_bucket="10_to_99",
        base_extractable=True,
        base_output_tokens=40,
        base_output_token_bucket="33_to_64",
        stable_rank=identifier,
    )


def test_output_token_buckets_are_frozen() -> None:
    assert output_token_bucket(32) == "1_to_32"
    assert output_token_bucket(33) == "33_to_64"
    assert output_token_bucket(129) == "129_to_256"
    assert output_token_bucket(300) == "257_plus"


def test_hungarian_returns_exact_minimum_assignment() -> None:
    assert _hungarian([[4, 1, 3], [2, 0, 5], [3, 2, 2]]) == [1, 0, 2]


def test_matching_forbids_cross_source_assignment() -> None:
    with pytest.raises(ValueError, match="source corpora"):
        matching_cost(_candidate("a", source="ASDiv"), _candidate("b"))


def test_balance_report_accepts_identical_noncurriculum_covariates() -> None:
    targeted = [
        _candidate("t-book", family=FAMILY_BOOKKEEPING),
        _candidate("t-rate", family=FAMILY_RATE),
        _candidate("t-disc", family=FAMILY_DISCRETE),
    ]
    generic = [
        _candidate("g-book", family=FAMILY_BOOKKEEPING),
        _candidate("g-rate", family=FAMILY_RATE),
        _candidate("g-disc", family=FAMILY_DISCRETE),
    ]
    report = balance_report(targeted, generic)
    assert report["matching_quality_gate_passed"] is True
    assert report["numerical_smd"] == {
        "base_output_tokens": 0.0,
        "formula_depth": 0.0,
        "operation_count": 0.0,
        "question_token_count": 0.0,
    }


def test_manifests_do_not_need_question_text(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"source_id": "id", "question_sha256": "hash"}) + "\n")
    assert "question" not in json.loads(path.read_text())
