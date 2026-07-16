"""Auditable evaluation runner and machine-readable result writer."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from foundry.config import EvaluationConfig
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
    predicted_answer: int | None
    reference_answer: int | None
    correct: bool
    error: str | None
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
    processed_examples: int
    correct_examples: int
    invalid_examples: int
    generation_failures: int
    accuracy: float
    evaluation_seconds: float
    total_runtime_seconds: float
    examples_per_second: float
    total_input_tokens: int | None
    total_output_tokens: int | None
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
) -> EvaluationSummary:
    """Evaluate examples, store ignored raw records, and return an auditable summary."""

    if not examples:
        raise ValueError("evaluation requires at least one example")

    records: list[EvaluationRecord] = []
    evaluation_started = time.perf_counter()
    for example in examples:
        messages = render_messages(config.prompt, example.question)
        generation_started = time.perf_counter()
        try:
            result = backend.generate(example.stable_id, messages, config.generation)
            generation_seconds = time.perf_counter() - generation_started
            score = score_response(result.text, example.reference_answer)
            records.append(
                EvaluationRecord(
                    stable_id=example.stable_id,
                    row_index=example.row_index,
                    response=result.text,
                    predicted_answer=score.predicted,
                    reference_answer=score.expected,
                    correct=score.correct,
                    error=score.error,
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
                    reference_answer=None,
                    correct=False,
                    error=f"generation failure: {type(error).__name__}: {error}",
                    input_tokens=None,
                    output_tokens=None,
                    generation_seconds=time.perf_counter() - generation_started,
                )
            )
    evaluation_seconds = time.perf_counter() - evaluation_started

    backend_metrics = backend.metrics()
    load_seconds_raw = backend_metrics.get("backend_load_seconds")
    load_seconds = float(load_seconds_raw) if isinstance(load_seconds_raw, int | float) else 0.0
    processed = len(records)
    correct = sum(record.correct for record in records)
    generation_failures = sum(
        record.error is not None and record.error.startswith("generation failure:")
        for record in records
    )
    invalid = sum(
        record.error is not None and not record.error.startswith("generation failure:")
        for record in records
    )
    input_token_values = [
        record.input_tokens for record in records if record.input_tokens is not None
    ]
    output_token_values = [
        record.output_tokens for record in records if record.output_tokens is not None
    ]
    total_runtime = evaluation_seconds + load_seconds
    summary = EvaluationSummary(
        schema_version=1,
        backend=backend.name,
        model_id=config.model.repo_id,
        model_revision=config.model.revision,
        dataset_id=config.dataset.repo_id,
        dataset_revision=config.dataset.revision,
        manifest_partition=manifest.partition,
        manifest_sha256=manifest.manifest_sha256,
        config_sha256=config.sha256,
        prompt_sha256=prompt_sha256(config.prompt),
        processed_examples=processed,
        correct_examples=correct,
        invalid_examples=invalid,
        generation_failures=generation_failures,
        accuracy=correct / processed,
        evaluation_seconds=evaluation_seconds,
        total_runtime_seconds=total_runtime,
        examples_per_second=processed / evaluation_seconds,
        total_input_tokens=sum(input_token_values) if input_token_values else None,
        total_output_tokens=sum(output_token_values) if output_token_values else None,
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
