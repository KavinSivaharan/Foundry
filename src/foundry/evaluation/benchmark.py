"""Benchmark loading behind pinned, label-free manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from foundry.config import EvaluationConfig
from foundry.evaluation.manifests import (
    BenchmarkManifest,
    ManifestError,
    require_partition_access,
)


class BenchmarkError(RuntimeError):
    """Raised when benchmark data differs from its pinned manifest."""


@dataclass(frozen=True)
class BenchmarkExample:
    """One benchmark example held only in memory during evaluation."""

    stable_id: str
    row_index: int
    question: str
    reference_answer: str


@dataclass(frozen=True)
class FixtureExample:
    """Synthetic integration-test example and its deterministic fake response."""

    example: BenchmarkExample
    fake_response: str


def _selected_entries(
    manifest: BenchmarkManifest,
    *,
    limit: int | None,
) -> tuple[tuple[str, int], ...]:
    if limit is not None and limit < 1:
        raise BenchmarkError("limit must be positive")
    entries = manifest.entries if limit is None else manifest.entries[:limit]
    return tuple((entry.stable_id, entry.row_index) for entry in entries)


def load_fixture_examples(
    path: Path,
    manifest: BenchmarkManifest,
    *,
    allow_sealed_final: bool = False,
    limit: int | None = None,
) -> tuple[FixtureExample, ...]:
    """Load synthetic test rows while enforcing the same manifest boundary."""

    require_partition_access(manifest, allow_sealed_final=allow_sealed_final)
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"could not load fixture {path}: {error}") from error
    if not isinstance(raw, list):
        raise BenchmarkError("fixture root must be a list")

    by_identity: dict[tuple[str, int], FixtureExample] = {}
    for position, item_raw in enumerate(raw):
        if not isinstance(item_raw, dict):
            raise BenchmarkError(f"fixture item {position} must be an object")
        item = cast(dict[str, object], item_raw)
        required = {"stable_id", "row_index", "question", "answer", "response"}
        if item.keys() != required:
            raise BenchmarkError(f"fixture item {position} has an invalid schema")
        stable_id = item["stable_id"]
        row_index = item["row_index"]
        question = item["question"]
        answer = item["answer"]
        response = item["response"]
        if not isinstance(stable_id, str):
            raise BenchmarkError(f"fixture item {position} stable_id must be a string")
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise BenchmarkError(f"fixture item {position} row_index must be an integer")
        if not isinstance(question, str) or not question.strip():
            raise BenchmarkError(f"fixture item {position} question must be non-empty")
        if not isinstance(answer, str) or not answer.strip():
            raise BenchmarkError(f"fixture item {position} answer must be non-empty")
        if not isinstance(response, str):
            raise BenchmarkError(f"fixture item {position} response must be a string")
        identity = (stable_id, row_index)
        if identity in by_identity:
            raise BenchmarkError(f"fixture contains duplicate identity {identity}")
        by_identity[identity] = FixtureExample(
            example=BenchmarkExample(
                stable_id=stable_id,
                row_index=row_index,
                question=question,
                reference_answer=answer,
            ),
            fake_response=response,
        )

    selected: list[FixtureExample] = []
    for identity in _selected_entries(manifest, limit=limit):
        if identity not in by_identity:
            raise BenchmarkError(f"fixture is missing manifest identity {identity}")
        selected.append(by_identity[identity])
    return tuple(selected)


def load_huggingface_examples(
    config: EvaluationConfig,
    manifest: BenchmarkManifest,
    *,
    allow_sealed_final: bool = False,
    limit: int | None = None,
) -> tuple[BenchmarkExample, ...]:
    """Download a pinned dataset revision and materialize only manifest-selected rows."""

    require_partition_access(manifest, allow_sealed_final=allow_sealed_final)
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as error:
        raise BenchmarkError(
            "real benchmark loading requires the pinned 'smoke' optional dependencies"
        ) from error

    dataset: Any = load_dataset(
        config.dataset.repo_id,
        config.dataset.config_name,
        split=config.dataset.source_split,
        revision=config.dataset.revision,
    )
    actual_examples = len(dataset)
    if actual_examples != config.dataset.expected_examples:
        raise BenchmarkError(
            f"pinned dataset length changed: expected {config.dataset.expected_examples}, "
            f"received {actual_examples}"
        )

    examples: list[BenchmarkExample] = []
    for stable_id, row_index in _selected_entries(manifest, limit=limit):
        row: Any = dataset[row_index]
        if not isinstance(row, dict):
            raise BenchmarkError(f"dataset row {row_index} is not a mapping")
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise BenchmarkError(f"dataset row {row_index} has an invalid question")
        if not isinstance(answer, str) or not answer.strip():
            raise BenchmarkError(f"dataset row {row_index} has an invalid answer")
        examples.append(
            BenchmarkExample(
                stable_id=stable_id,
                row_index=row_index,
                question=question,
                reference_answer=answer,
            )
        )
    return tuple(examples)


def assert_manifest_matches_config(
    manifest: BenchmarkManifest,
    config: EvaluationConfig,
) -> None:
    """Make an explicit public assertion for callers composing their own loaders."""

    if manifest.config_sha256 != config.sha256:
        raise ManifestError("manifest does not belong to this evaluation configuration")
