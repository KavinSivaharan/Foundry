"""Deterministic aggregate re-scoring of existing ignored predictions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from foundry.evaluation.answer_extraction import (
    CANONICAL_EXTRACTOR_ID,
    canonical_extractor_sha256,
    score_canonical_answer,
)
from foundry.evaluation.scoring import AnswerExtractionError, score_response


class RescoringError(ValueError):
    """Raised when an existing prediction artifact cannot be safely re-scored."""


@dataclass(frozen=True)
class RescoreSummary:
    """Content-free aggregate metrics from one existing prediction file."""

    schema_version: int
    source_predictions_sha256: str
    canonical_extractor_id: str
    canonical_extractor_sha256: str
    max_new_tokens: int
    processed_examples: int
    exact_format_compliant_examples: int
    exact_format_compliance_rate: float
    extractable_examples: int
    extractable_answer_rate: float
    correct_examples: int
    benchmark_accuracy: float
    ambiguous_or_rejected_examples: int
    generation_failures: int
    extraction_failure_categories: dict[str, int]


def _load_records(path: Path) -> tuple[list[dict[str, object]], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RescoringError(f"could not read predictions {path}: {error}") from error
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RescoringError(f"prediction line {line_number} is invalid JSON") from error
        if not isinstance(record, dict):
            raise RescoringError(f"prediction line {line_number} must be an object")
        records.append(record)
    if not records:
        raise RescoringError("prediction artifact is empty")
    return records, hashlib.sha256(payload).hexdigest()


def rescore_predictions(
    predictions_path: Path,
    output_path: Path,
    *,
    max_new_tokens: int,
) -> RescoreSummary:
    """Re-score existing raw records without generating or changing predictions."""

    if max_new_tokens <= 0:
        raise RescoringError("max_new_tokens must be positive")
    records, source_sha256 = _load_records(predictions_path)
    exact_compliant = 0
    extractable = 0
    correct = 0
    rejected = 0
    generation_failures = 0
    failure_categories: dict[str, int] = {}

    for position, record in enumerate(records):
        response = record.get("response")
        reference = record.get("reference_answer")
        output_tokens = record.get("output_tokens")
        error = record.get("error")
        is_generation_failure = response is None or (
            isinstance(error, str) and error.startswith("generation failure:")
        )
        if is_generation_failure:
            generation_failures += 1
            failure_categories["generation_failure"] = (
                failure_categories.get("generation_failure", 0) + 1
            )
            continue
        if not isinstance(response, str):
            raise RescoringError(f"prediction {position} response must be a string or null")
        if isinstance(reference, bool) or not isinstance(reference, int):
            raise RescoringError(f"prediction {position} reference_answer must be an integer")
        if output_tokens is not None and (
            isinstance(output_tokens, bool) or not isinstance(output_tokens, int)
        ):
            raise RescoringError(f"prediction {position} output_tokens must be an integer or null")

        reference_text = str(reference)
        try:
            exact_score = score_response(response, reference_text)
            canonical_score = score_canonical_answer(
                response,
                reference_text,
                generation_truncated=(
                    output_tokens is not None and output_tokens >= max_new_tokens
                ),
            )
        except AnswerExtractionError as extraction_error:
            raise RescoringError(
                f"prediction {position} has an invalid benchmark reference"
            ) from extraction_error
        if exact_score.error is None:
            exact_compliant += 1
        if canonical_score.predicted is not None:
            extractable += 1
            correct += int(canonical_score.correct)
        else:
            rejected += 1
            if canonical_score.failure_category is None:
                raise RescoringError(f"prediction {position} rejection lacks a category")
            category = canonical_score.failure_category
            failure_categories[category] = failure_categories.get(category, 0) + 1

    processed = len(records)
    summary = RescoreSummary(
        schema_version=2,
        source_predictions_sha256=source_sha256,
        canonical_extractor_id=CANONICAL_EXTRACTOR_ID,
        canonical_extractor_sha256=canonical_extractor_sha256(),
        max_new_tokens=max_new_tokens,
        processed_examples=processed,
        exact_format_compliant_examples=exact_compliant,
        exact_format_compliance_rate=exact_compliant / processed,
        extractable_examples=extractable,
        extractable_answer_rate=extractable / processed,
        correct_examples=correct,
        benchmark_accuracy=correct / processed,
        ambiguous_or_rejected_examples=rejected,
        generation_failures=generation_failures,
        extraction_failure_categories=failure_categories,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
