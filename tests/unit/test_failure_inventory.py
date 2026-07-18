from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from foundry.evaluation.failure_inventory import (
    FAILURE_SAMPLE_SEED,
    FailureInventoryError,
    FailureRecord,
    count_failures,
    deterministic_mathematical_failure_sample,
    load_failure_records,
    sample_sha256,
)


def _record(index: int, *, correct: bool = False) -> FailureRecord:
    return FailureRecord(
        stable_id=f"{index:064x}",
        row_index=index,
        correct=correct,
        predicted_answer=None if index % 5 == 0 else index,
        exact_format_compliant=index % 3 == 0,
        extraction_failure_category=("ambiguous_terminal_answer" if index % 5 == 0 else None),
        generation_truncated=index == 5,
        generation_failed=index == 10,
    )


def test_failure_counts_keep_measured_groups_separate() -> None:
    records = tuple(_record(index, correct=index in {1, 2}) for index in range(12))

    counts = count_failures(records)

    assert counts.attempted == 12
    assert counts.correct == 2
    assert counts.incorrect == 10
    assert counts.extractable_incorrect == 7
    assert counts.unextractable == 3
    assert counts.truncated == 1
    assert counts.generation_failures == 1
    assert counts.exact_format_noncompliant == 8
    assert counts.extraction_failure_categories == {"ambiguous_terminal_answer": 3}


def test_sample_is_bounded_deterministic_and_content_independent() -> None:
    records = tuple(_record(index) for index in range(150))

    first = deterministic_mathematical_failure_sample(records)
    second = deterministic_mathematical_failure_sample(tuple(reversed(records)))

    assert len(first) == 100
    assert first == second
    assert all(record.predicted_answer is not None for record in first)
    assert all(not record.correct and not record.generation_failed for record in first)
    assert sample_sha256(first) == sample_sha256(second)
    assert FAILURE_SAMPLE_SEED == "foundry-milestone-2-mathematical-failures-v1"


def test_sample_rejects_sizes_outside_approved_bound() -> None:
    records = (_record(1),)

    with pytest.raises(FailureInventoryError, match="from 1 to 100"):
        deterministic_mathematical_failure_sample(records, sample_size=101)


def test_loader_rejects_duplicate_identities(tmp_path: Path) -> None:
    record = _record(1)
    payload = {
        **asdict(record),
        "error": None,
        "response": "ignored content",
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(payload) + "\n" + json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(FailureInventoryError, match="duplicate prediction identity"):
        load_failure_records(path)


def test_loader_reads_only_required_content_free_fields(tmp_path: Path) -> None:
    record = replace(_record(7), generation_failed=False)
    payload = {
        **asdict(record),
        "error": None,
        "response": "benchmark content deliberately ignored by the loader",
        "reference_answer": 99,
    }
    payload.pop("generation_failed")
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert load_failure_records(path) == (record,)


def test_loader_rejects_boolean_as_a_predicted_number(tmp_path: Path) -> None:
    payload = {
        **asdict(_record(8)),
        "error": None,
        "predicted_answer": True,
    }
    payload.pop("generation_failed")
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(FailureInventoryError, match="predicted_answer has an invalid type"):
        load_failure_records(path)
