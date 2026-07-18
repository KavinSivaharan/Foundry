"""Auditable evaluation runner and machine-readable result writer."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from foundry.config import EvaluationConfig
from foundry.evaluation.answer_extraction import (
    CANONICAL_EXTRACTOR_ID,
    canonical_extractor_sha256,
    score_canonical_answer,
    serialize_canonical_number,
)
from foundry.evaluation.backends import MetricValue, ModelBackend
from foundry.evaluation.benchmark import BenchmarkExample
from foundry.evaluation.manifests import BenchmarkManifest
from foundry.evaluation.prompting import prompt_sha256, render_messages
from foundry.evaluation.scoring import AnswerExtractionError, score_response


@dataclass(frozen=True)
class EvaluationRecord:
    """One raw, ignored prediction record for later auditing."""

    stable_id: str
    row_index: int
    response: str | None
    predicted_answer: int | str | None
    exact_format_compliant: bool
    exact_format_predicted_answer: int | None
    exact_format_error: str | None
    reference_answer: int | None
    correct: bool
    error: str | None
    extraction_failure_category: str | None
    generation_truncated: bool
    input_tokens: int | None
    output_tokens: int | None
    generation_seconds: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate measurements that do not contain benchmark examples or labels."""

    schema_version: int
    backend: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    manifest_partition: str
    manifest_sha256: str
    config_sha256: str
    prompt_sha256: str
    canonical_extractor_id: str
    canonical_extractor_sha256: str
    processed_examples: int
    exact_format_compliant_examples: int
    exact_format_compliance_rate: float
    extractable_examples: int
    extractable_answer_rate: float
    correct_examples: int
    invalid_examples: int
    ambiguous_or_rejected_examples: int
    generation_failures: int
    accuracy: float
    accuracy_among_extractable_answers: float
    extractable_incorrect_examples: int
    unextractable_examples: int
    truncated_examples: int
    extraction_failure_categories: dict[str, int]
    evaluation_seconds: float
    total_runtime_seconds: float
    examples_per_second: float
    total_input_tokens: int | None
    total_output_tokens: int | None
    average_output_tokens: float | None
    generated_tokens_per_second: float | None
    backend_metrics: dict[str, MetricValue]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_evaluation(
    *,
    config: EvaluationConfig,
    manifest: BenchmarkManifest,
    examples: tuple[BenchmarkExample, ...],
    backend: ModelBackend,
    output_dir: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvaluationSummary:
    """Evaluate examples, store ignored raw records, and return an auditable summary."""

    if not examples:
        raise ValueError("evaluation requires at least one example")

    records: list[EvaluationRecord] = []
    evaluation_started = time.perf_counter()
    total_examples = len(examples)
    for position, example in enumerate(examples, start=1):
        messages = render_messages(config.prompt, example.question)
        generation_started = time.perf_counter()
        try:
            result = backend.generate(example.stable_id, messages, config.generation)
            generation_seconds = time.perf_counter() - generation_started
            exact_score = score_response(result.text, example.reference_answer)
            generation_truncated = (
                result.output_tokens is not None
                and result.output_tokens >= config.generation.max_new_tokens
            )
            canonical_score = score_canonical_answer(
                result.text,
                example.reference_answer,
                generation_truncated=generation_truncated,
            )
            records.append(
                EvaluationRecord(
                    stable_id=example.stable_id,
                    row_index=example.row_index,
                    response=result.text,
                    predicted_answer=(
                        serialize_canonical_number(canonical_score.predicted)
                        if canonical_score.predicted is not None
                        else None
                    ),
                    exact_format_compliant=exact_score.error is None,
                    exact_format_predicted_answer=exact_score.predicted,
                    exact_format_error=exact_score.error,
                    reference_answer=canonical_score.expected,
                    correct=canonical_score.correct,
                    error=canonical_score.error,
                    extraction_failure_category=canonical_score.failure_category,
                    generation_truncated=generation_truncated,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    generation_seconds=generation_seconds,
                )
            )
        except AnswerExtractionError as error:
            raise RuntimeError(
                f"pinned benchmark answer is invalid at row {example.row_index}: {error}"
            ) from error
        except Exception as error:  # noqa: BLE001 - failures must become auditable records
            records.append(
                EvaluationRecord(
                    stable_id=example.stable_id,
                    row_index=example.row_index,
                    response=None,
                    predicted_answer=None,
                    exact_format_compliant=False,
                    exact_format_predicted_answer=None,
                    exact_format_error="generation failure",
                    reference_answer=None,
                    correct=False,
                    error=f"generation failure: {type(error).__name__}: {error}",
                    extraction_failure_category="generation_failure",
                    generation_truncated=False,
                    input_tokens=None,
                    output_tokens=None,
                    generation_seconds=time.perf_counter() - generation_started,
                )
            )
        if progress_callback is not None:
            progress_callback(position, total_examples)
    evaluation_seconds = time.perf_counter() - evaluation_started

    backend_metrics = backend.metrics()
    load_seconds_raw = backend_metrics.get("backend_load_seconds")
    load_seconds = float(load_seconds_raw) if isinstance(load_seconds_raw, int | float) else 0.0
    processed = len(records)
    exact_format_compliant = sum(record.exact_format_compliant for record in records)
    extractable = sum(record.predicted_answer is not None for record in records)
    correct = sum(record.correct for record in records)
    truncated = sum(record.generation_truncated for record in records)
    generation_failures = sum(
        record.error is not None and record.error.startswith("generation failure:")
        for record in records
    )
    invalid = sum(
        record.error is not None and not record.error.startswith("generation failure:")
        for record in records
    )
    failure_categories: dict[str, int] = {}
    for record in records:
        if record.extraction_failure_category is not None:
            category = record.extraction_failure_category
            failure_categories[category] = failure_categories.get(category, 0) + 1
    input_token_values = [
        record.input_tokens for record in records if record.input_tokens is not None
    ]
    output_token_values = [
        record.output_tokens for record in records if record.output_tokens is not None
    ]
    total_runtime = evaluation_seconds + load_seconds
    total_input_tokens = sum(input_token_values) if input_token_values else None
    total_output_tokens = sum(output_token_values) if output_token_values else None
    summary = EvaluationSummary(
        schema_version=3,
        backend=backend.name,
        model_id=config.model.repo_id,
        model_revision=config.model.revision,
        dataset_id=config.dataset.repo_id,
        dataset_revision=config.dataset.revision,
        manifest_partition=manifest.partition,
        manifest_sha256=manifest.manifest_sha256,
        config_sha256=config.sha256,
        prompt_sha256=prompt_sha256(config.prompt),
        canonical_extractor_id=CANONICAL_EXTRACTOR_ID,
        canonical_extractor_sha256=canonical_extractor_sha256(),
        processed_examples=processed,
        exact_format_compliant_examples=exact_format_compliant,
        exact_format_compliance_rate=exact_format_compliant / processed,
        extractable_examples=extractable,
        extractable_answer_rate=extractable / processed,
        correct_examples=correct,
        invalid_examples=invalid,
        ambiguous_or_rejected_examples=invalid,
        generation_failures=generation_failures,
        accuracy=correct / processed,
        accuracy_among_extractable_answers=correct / extractable if extractable else 0.0,
        extractable_incorrect_examples=extractable - correct,
        unextractable_examples=processed - extractable,
        truncated_examples=truncated,
        extraction_failure_categories=failure_categories,
        evaluation_seconds=evaluation_seconds,
        total_runtime_seconds=total_runtime,
        examples_per_second=processed / evaluation_seconds,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        average_output_tokens=(
            total_output_tokens / processed if total_output_tokens is not None else None
        ),
        generated_tokens_per_second=(
            total_output_tokens / evaluation_seconds if total_output_tokens is not None else None
        ),
        backend_metrics=backend_metrics,
    )

    raw_path = output_dir / "raw" / "predictions.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_json(output_dir / "summary.json", asdict(summary))
    return summary
