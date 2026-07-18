"""Deterministic, content-free sampling for development-baseline failures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

FAILURE_SAMPLE_SEED = "foundry-milestone-2-mathematical-failures-v1"


class FailureInventoryError(ValueError):
    """Raised when raw baseline records cannot support a trustworthy inventory."""


@dataclass(frozen=True)
class FailureRecord:
    """Content-free fields needed to count and sample one prediction record."""

    stable_id: str
    row_index: int
    correct: bool
    predicted_answer: int | str | None
    exact_format_compliant: bool
    extraction_failure_category: str | None
    generation_truncated: bool
    generation_failed: bool


@dataclass(frozen=True)
class FailureCounts:
    """Measured, mutually understandable population counts for a baseline run."""

    attempted: int
    correct: int
    incorrect: int
    extractable_incorrect: int
    unextractable: int
    truncated: int
    generation_failures: int
    exact_format_noncompliant: int
    extraction_failure_categories: dict[str, int]


def _require_bool(item: dict[str, object], field: str, line_number: int) -> bool:
    value = item.get(field)
    if not isinstance(value, bool):
        raise FailureInventoryError(f"line {line_number} field {field} must be a boolean")
    return value


def _parse_record(raw: Any, line_number: int) -> FailureRecord:
    if not isinstance(raw, dict):
        raise FailureInventoryError(f"line {line_number} must contain a JSON object")
    item = cast(dict[str, object], raw)
    stable_id = item.get("stable_id")
    row_index = item.get("row_index")
    predicted_answer = item.get("predicted_answer")
    failure_category = item.get("extraction_failure_category")
    error = item.get("error")
    if not isinstance(stable_id, str) or len(stable_id) != 64:
        raise FailureInventoryError(f"line {line_number} has an invalid stable identifier")
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise FailureInventoryError(f"line {line_number} field row_index must be an integer")
    if predicted_answer is not None and (
        isinstance(predicted_answer, bool) or not isinstance(predicted_answer, int | str)
    ):
        raise FailureInventoryError(
            f"line {line_number} field predicted_answer has an invalid type"
        )
    if failure_category is not None and not isinstance(failure_category, str):
        raise FailureInventoryError(
            f"line {line_number} field extraction_failure_category has an invalid type"
        )
    if error is not None and not isinstance(error, str):
        raise FailureInventoryError(f"line {line_number} field error has an invalid type")
    return FailureRecord(
        stable_id=stable_id,
        row_index=row_index,
        correct=_require_bool(item, "correct", line_number),
        predicted_answer=predicted_answer,
        exact_format_compliant=_require_bool(item, "exact_format_compliant", line_number),
        extraction_failure_category=failure_category,
        generation_truncated=_require_bool(item, "generation_truncated", line_number),
        generation_failed=error is not None and error.startswith("generation failure:"),
    )


def load_failure_records(path: Path) -> tuple[FailureRecord, ...]:
    """Load only content-free fields from ignored JSONL prediction records."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FailureInventoryError(f"could not read prediction records: {error}") from error
    if not lines:
        raise FailureInventoryError("prediction record file is empty")
    records: list[FailureRecord] = []
    identities: set[tuple[str, int]] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise FailureInventoryError(f"line {line_number} is not valid JSON: {error}") from error
        record = _parse_record(raw, line_number)
        identity = (record.stable_id, record.row_index)
        if identity in identities:
            raise FailureInventoryError(f"duplicate prediction identity at line {line_number}")
        identities.add(identity)
        records.append(record)
    return tuple(records)


def count_failures(records: tuple[FailureRecord, ...]) -> FailureCounts:
    """Count measured outcome groups without interpreting mathematical causes."""

    if not records:
        raise FailureInventoryError("failure counts require at least one record")
    categories: dict[str, int] = {}
    for record in records:
        if record.extraction_failure_category is not None:
            category = record.extraction_failure_category
            categories[category] = categories.get(category, 0) + 1
    attempted = len(records)
    correct = sum(record.correct for record in records)
    extractable_incorrect = sum(
        not record.correct and record.predicted_answer is not None for record in records
    )
    return FailureCounts(
        attempted=attempted,
        correct=correct,
        incorrect=attempted - correct,
        extractable_incorrect=extractable_incorrect,
        unextractable=sum(record.predicted_answer is None for record in records),
        truncated=sum(record.generation_truncated for record in records),
        generation_failures=sum(record.generation_failed for record in records),
        exact_format_noncompliant=sum(not record.exact_format_compliant for record in records),
        extraction_failure_categories=dict(sorted(categories.items())),
    )


def _sample_rank(record: FailureRecord, seed: str) -> str:
    material = f"{seed}:{record.stable_id}:{record.row_index}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def deterministic_mathematical_failure_sample(
    records: tuple[FailureRecord, ...],
    *,
    sample_size: int = 100,
    seed: str = FAILURE_SAMPLE_SEED,
) -> tuple[FailureRecord, ...]:
    """Select extractable-but-wrong records by a stable content-independent rank."""

    if not 1 <= sample_size <= 100:
        raise FailureInventoryError("mathematical failure sample size must be from 1 to 100")
    eligible = tuple(
        record
        for record in records
        if not record.correct
        and record.predicted_answer is not None
        and not record.generation_failed
    )
    ranked = sorted(eligible, key=lambda record: (_sample_rank(record, seed), record.row_index))
    return tuple(ranked[:sample_size])


def sample_sha256(records: tuple[FailureRecord, ...]) -> str:
    """Hash the ordered, identifier-only sample for reproducibility checks."""

    material = "".join(f"{record.stable_id}:{record.row_index}\n" for record in records)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
