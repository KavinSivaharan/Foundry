"""Deterministic fresh answer-validation and main-baseline manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

from foundry.config import EvaluationConfig
from foundry.evaluation.calibration import DevelopmentSubsetManifest
from foundry.evaluation.manifests import BenchmarkManifest, ManifestEntry

ValidationPurpose = Literal["answer_extraction_validation", "main_development_baseline"]
ANSWER_EXTRACTION_VALIDATION: ValidationPurpose = "answer_extraction_validation"
MAIN_DEVELOPMENT_BASELINE: ValidationPurpose = "main_development_baseline"
FinalEvaluatorPurpose = Literal["final_evaluator_validation", "main_development_baseline"]
FINAL_EVALUATOR_VALIDATION: FinalEvaluatorPurpose = "final_evaluator_validation"
FINAL_MAIN_DEVELOPMENT_BASELINE: FinalEvaluatorPurpose = "main_development_baseline"


class ValidationManifestError(ValueError):
    """Raised when a fresh answer-validation partition is invalid."""


def assert_final_evaluator_config(
    base: EvaluationConfig,
    evaluator: EvaluationConfig,
) -> None:
    """Allow only the verified 512-to-768 generation-limit adjustment."""

    if base.generation.max_new_tokens != 512:
        raise ValidationManifestError("base evaluator config must use 512 maximum tokens")
    expected = replace(
        base,
        generation=replace(base.generation, max_new_tokens=768),
    )
    if evaluator != expected:
        raise ValidationManifestError(
            "final evaluator config may change only max_new_tokens from 512 to 768"
        )


@dataclass(frozen=True)
class AnswerValidationManifest:
    """Identifier-only subset derived from the reserved future-baseline pool."""

    schema_version: int
    purpose: ValidationPurpose
    dataset_id: str
    dataset_revision: str
    config_name: str
    source_split: str
    canonical_development_manifest_sha256: str
    source_pool_manifest_sha256: str
    selection_seed: str
    entries: tuple[ManifestEntry, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class FinalEvaluatorValidationManifest:
    """Identifier-only final gate and remaining baseline partition."""

    schema_version: int
    purpose: FinalEvaluatorPurpose
    dataset_id: str
    dataset_revision: str
    config_name: str
    source_split: str
    canonical_development_manifest_sha256: str
    source_baseline_manifest_sha256: str
    selection_seed: str
    entries: tuple[ManifestEntry, ...]
    manifest_sha256: str


def _unsigned_payload(manifest: AnswerValidationManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload.pop("manifest_sha256")
    return cast(dict[str, object], payload)


def _digest_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _selection_rank(selection_seed: str, entry: ManifestEntry) -> str:
    material = f"{selection_seed}:{entry.stable_id}:{entry.row_index}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _create_manifest(
    source_pool: DevelopmentSubsetManifest,
    canonical_source: BenchmarkManifest,
    *,
    purpose: ValidationPurpose,
    selection_seed: str,
    entries: tuple[ManifestEntry, ...],
) -> AnswerValidationManifest:
    unsigned = AnswerValidationManifest(
        schema_version=1,
        purpose=purpose,
        dataset_id=canonical_source.dataset_id,
        dataset_revision=canonical_source.dataset_revision,
        config_name=canonical_source.config_name,
        source_split=canonical_source.source_split,
        canonical_development_manifest_sha256=canonical_source.manifest_sha256,
        source_pool_manifest_sha256=source_pool.manifest_sha256,
        selection_seed=selection_seed,
        entries=entries,
        manifest_sha256="",
    )
    return replace(unsigned, manifest_sha256=_digest_payload(_unsigned_payload(unsigned)))


def _validate_manifest(
    manifest: AnswerValidationManifest,
    source_pool: DevelopmentSubsetManifest,
    canonical_source: BenchmarkManifest,
) -> None:
    if manifest.schema_version != 1:
        raise ValidationManifestError("answer-validation schema_version must be 1")
    if manifest.purpose not in {ANSWER_EXTRACTION_VALIDATION, MAIN_DEVELOPMENT_BASELINE}:
        raise ValidationManifestError("answer-validation purpose is invalid")
    if source_pool.purpose != MAIN_DEVELOPMENT_BASELINE:
        raise ValidationManifestError("answer validation must derive from the baseline pool")
    expected_identity = (
        canonical_source.dataset_id,
        canonical_source.dataset_revision,
        canonical_source.config_name,
        canonical_source.source_split,
        canonical_source.manifest_sha256,
        source_pool.manifest_sha256,
    )
    actual_identity = (
        manifest.dataset_id,
        manifest.dataset_revision,
        manifest.config_name,
        manifest.source_split,
        manifest.canonical_development_manifest_sha256,
        manifest.source_pool_manifest_sha256,
    )
    if actual_identity != expected_identity:
        raise ValidationManifestError("answer-validation manifest has the wrong source identity")
    if not manifest.selection_seed.strip():
        raise ValidationManifestError("answer-validation selection_seed must be non-empty")
    if manifest.manifest_sha256 != _digest_payload(_unsigned_payload(manifest)):
        raise ValidationManifestError("answer-validation manifest digest is invalid")
    identities = [(entry.stable_id, entry.row_index) for entry in manifest.entries]
    if len(identities) != len(set(identities)):
        raise ValidationManifestError("answer-validation manifest contains duplicate IDs")
    pool_ids = {(entry.stable_id, entry.row_index) for entry in source_pool.entries}
    if not set(identities).issubset(pool_ids):
        raise ValidationManifestError("answer-validation manifest contains an ID outside its pool")


def build_answer_validation_manifests(
    source_pool: DevelopmentSubsetManifest,
    canonical_source: BenchmarkManifest,
    *,
    validation_size: int,
    selection_seed: str,
) -> tuple[AnswerValidationManifest, AnswerValidationManifest]:
    """Reserve fresh validation IDs and leave a disjoint main-baseline pool."""

    if not selection_seed.strip():
        raise ValidationManifestError("selection_seed must be non-empty")
    if not 0 < validation_size < len(source_pool.entries):
        raise ValidationManifestError("validation_size must leave IDs in both subsets")
    ranked = sorted(
        source_pool.entries,
        key=lambda entry: (_selection_rank(selection_seed, entry), entry.row_index),
    )
    validation_ids = {(entry.stable_id, entry.row_index) for entry in ranked[:validation_size]}
    validation_entries = tuple(
        entry
        for entry in source_pool.entries
        if (entry.stable_id, entry.row_index) in validation_ids
    )
    baseline_entries = tuple(
        entry
        for entry in source_pool.entries
        if (entry.stable_id, entry.row_index) not in validation_ids
    )
    validation = _create_manifest(
        source_pool,
        canonical_source,
        purpose=ANSWER_EXTRACTION_VALIDATION,
        selection_seed=selection_seed,
        entries=validation_entries,
    )
    baseline = _create_manifest(
        source_pool,
        canonical_source,
        purpose=MAIN_DEVELOPMENT_BASELINE,
        selection_seed=selection_seed,
        entries=baseline_entries,
    )
    validate_answer_validation_pair(validation, baseline, source_pool, canonical_source)
    return validation, baseline


def validate_answer_validation_pair(
    validation: AnswerValidationManifest,
    baseline: AnswerValidationManifest,
    source_pool: DevelopmentSubsetManifest,
    canonical_source: BenchmarkManifest,
) -> None:
    """Validate purpose, disjointness, source binding, and complete pool coverage."""

    _validate_manifest(validation, source_pool, canonical_source)
    _validate_manifest(baseline, source_pool, canonical_source)
    if validation.purpose != ANSWER_EXTRACTION_VALIDATION:
        raise ValidationManifestError("fresh validation manifest has the wrong purpose")
    if baseline.purpose != MAIN_DEVELOPMENT_BASELINE:
        raise ValidationManifestError("main-baseline manifest has the wrong purpose")
    if validation.selection_seed != baseline.selection_seed:
        raise ValidationManifestError("fresh validation selection seeds differ")
    validation_ids = {(entry.stable_id, entry.row_index) for entry in validation.entries}
    baseline_ids = {(entry.stable_id, entry.row_index) for entry in baseline.entries}
    pool_ids = {(entry.stable_id, entry.row_index) for entry in source_pool.entries}
    if validation_ids & baseline_ids:
        raise ValidationManifestError("fresh validation and main-baseline IDs overlap")
    if validation_ids | baseline_ids != pool_ids:
        raise ValidationManifestError("fresh validation partitions do not cover the source pool")


def save_answer_validation_manifest(manifest: AnswerValidationManifest, path: Path) -> None:
    """Write an answer-validation manifest in a stable identifier-only form."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_answer_validation_manifest(
    path: Path,
    source_pool: DevelopmentSubsetManifest,
    canonical_source: BenchmarkManifest,
) -> AnswerValidationManifest:
    """Load and validate an answer-validation or main-baseline manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationManifestError(
            f"could not load answer-validation manifest: {error}"
        ) from error
    required = {
        "schema_version",
        "purpose",
        "dataset_id",
        "dataset_revision",
        "config_name",
        "source_split",
        "canonical_development_manifest_sha256",
        "source_pool_manifest_sha256",
        "selection_seed",
        "entries",
        "manifest_sha256",
    }
    if not isinstance(raw, dict) or raw.keys() != required:
        raise ValidationManifestError("answer-validation manifest keys differ from schema")
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        raise ValidationManifestError("answer-validation entries must be a list")
    entries: list[ManifestEntry] = []
    for position, entry_raw in enumerate(entries_raw):
        if not isinstance(entry_raw, dict) or entry_raw.keys() != {"stable_id", "row_index"}:
            raise ValidationManifestError(f"answer-validation entry {position} is invalid")
        stable_id = entry_raw["stable_id"]
        row_index = entry_raw["row_index"]
        if (
            not isinstance(stable_id, str)
            or isinstance(row_index, bool)
            or not isinstance(row_index, int)
        ):
            raise ValidationManifestError(f"answer-validation entry {position} is invalid")
        entries.append(ManifestEntry(stable_id=stable_id, row_index=row_index))
    purpose = raw["purpose"]
    if purpose not in {ANSWER_EXTRACTION_VALIDATION, MAIN_DEVELOPMENT_BASELINE}:
        raise ValidationManifestError("answer-validation purpose is invalid")
    manifest = AnswerValidationManifest(
        schema_version=int(raw["schema_version"]),
        purpose=cast(ValidationPurpose, purpose),
        dataset_id=str(raw["dataset_id"]),
        dataset_revision=str(raw["dataset_revision"]),
        config_name=str(raw["config_name"]),
        source_split=str(raw["source_split"]),
        canonical_development_manifest_sha256=str(raw["canonical_development_manifest_sha256"]),
        source_pool_manifest_sha256=str(raw["source_pool_manifest_sha256"]),
        selection_seed=str(raw["selection_seed"]),
        entries=tuple(entries),
        manifest_sha256=str(raw["manifest_sha256"]),
    )
    _validate_manifest(manifest, source_pool, canonical_source)
    return manifest


def as_benchmark_manifest(
    manifest: AnswerValidationManifest,
    source_pool: DevelopmentSubsetManifest,
    canonical_source: BenchmarkManifest,
    config: EvaluationConfig,
) -> BenchmarkManifest:
    """Adapt only the fresh validation manifest for the existing runner."""

    _validate_manifest(manifest, source_pool, canonical_source)
    if manifest.purpose != ANSWER_EXTRACTION_VALIDATION:
        raise ValidationManifestError("only fresh validation IDs may use answer-validate")
    return BenchmarkManifest(
        schema_version=1,
        dataset_id=config.dataset.repo_id,
        dataset_revision=config.dataset.revision,
        config_name=config.dataset.config_name,
        source_split=config.dataset.source_split,
        partition="development",
        partition_seed=config.partition.seed,
        expected_dataset_examples=config.dataset.expected_examples,
        config_sha256=config.sha256,
        entries=manifest.entries,
        manifest_sha256=manifest.manifest_sha256,
    )


def _final_unsigned_payload(
    manifest: FinalEvaluatorValidationManifest,
) -> dict[str, object]:
    payload = asdict(manifest)
    payload.pop("manifest_sha256")
    return cast(dict[str, object], payload)


def _create_final_manifest(
    source_baseline: AnswerValidationManifest,
    canonical_source: BenchmarkManifest,
    *,
    purpose: FinalEvaluatorPurpose,
    selection_seed: str,
    entries: tuple[ManifestEntry, ...],
) -> FinalEvaluatorValidationManifest:
    unsigned = FinalEvaluatorValidationManifest(
        schema_version=1,
        purpose=purpose,
        dataset_id=canonical_source.dataset_id,
        dataset_revision=canonical_source.dataset_revision,
        config_name=canonical_source.config_name,
        source_split=canonical_source.source_split,
        canonical_development_manifest_sha256=canonical_source.manifest_sha256,
        source_baseline_manifest_sha256=source_baseline.manifest_sha256,
        selection_seed=selection_seed,
        entries=entries,
        manifest_sha256="",
    )
    return replace(
        unsigned,
        manifest_sha256=_digest_payload(_final_unsigned_payload(unsigned)),
    )


def _validate_final_manifest(
    manifest: FinalEvaluatorValidationManifest,
    source_baseline: AnswerValidationManifest,
    canonical_source: BenchmarkManifest,
) -> None:
    if manifest.schema_version != 1:
        raise ValidationManifestError("final-evaluator schema_version must be 1")
    if manifest.purpose not in {
        FINAL_EVALUATOR_VALIDATION,
        FINAL_MAIN_DEVELOPMENT_BASELINE,
    }:
        raise ValidationManifestError("final-evaluator purpose is invalid")
    if source_baseline.purpose != MAIN_DEVELOPMENT_BASELINE:
        raise ValidationManifestError("final evaluator must derive from the 844-ID baseline")
    expected_identity = (
        canonical_source.dataset_id,
        canonical_source.dataset_revision,
        canonical_source.config_name,
        canonical_source.source_split,
        canonical_source.manifest_sha256,
        source_baseline.manifest_sha256,
    )
    actual_identity = (
        manifest.dataset_id,
        manifest.dataset_revision,
        manifest.config_name,
        manifest.source_split,
        manifest.canonical_development_manifest_sha256,
        manifest.source_baseline_manifest_sha256,
    )
    if actual_identity != expected_identity:
        raise ValidationManifestError("final-evaluator manifest has the wrong source identity")
    if not manifest.selection_seed.strip():
        raise ValidationManifestError("final-evaluator selection_seed must be non-empty")
    if manifest.manifest_sha256 != _digest_payload(_final_unsigned_payload(manifest)):
        raise ValidationManifestError("final-evaluator manifest digest is invalid")
    identities = [(entry.stable_id, entry.row_index) for entry in manifest.entries]
    if len(identities) != len(set(identities)):
        raise ValidationManifestError("final-evaluator manifest contains duplicate IDs")
    baseline_ids = {(entry.stable_id, entry.row_index) for entry in source_baseline.entries}
    if not set(identities).issubset(baseline_ids):
        raise ValidationManifestError("final-evaluator manifest contains an ID outside its pool")


def build_final_evaluator_manifests(
    source_baseline: AnswerValidationManifest,
    canonical_source: BenchmarkManifest,
    *,
    validation_size: int,
    selection_seed: str,
) -> tuple[FinalEvaluatorValidationManifest, FinalEvaluatorValidationManifest]:
    """Reserve the last fresh evaluator gate and leave the 814-ID baseline."""

    if not selection_seed.strip():
        raise ValidationManifestError("selection_seed must be non-empty")
    if not 0 < validation_size < len(source_baseline.entries):
        raise ValidationManifestError("validation_size must leave IDs in both final subsets")
    ranked = sorted(
        source_baseline.entries,
        key=lambda entry: (_selection_rank(selection_seed, entry), entry.row_index),
    )
    validation_ids = {(entry.stable_id, entry.row_index) for entry in ranked[:validation_size]}
    validation_entries = tuple(
        entry
        for entry in source_baseline.entries
        if (entry.stable_id, entry.row_index) in validation_ids
    )
    baseline_entries = tuple(
        entry
        for entry in source_baseline.entries
        if (entry.stable_id, entry.row_index) not in validation_ids
    )
    validation = _create_final_manifest(
        source_baseline,
        canonical_source,
        purpose=FINAL_EVALUATOR_VALIDATION,
        selection_seed=selection_seed,
        entries=validation_entries,
    )
    baseline = _create_final_manifest(
        source_baseline,
        canonical_source,
        purpose=FINAL_MAIN_DEVELOPMENT_BASELINE,
        selection_seed=selection_seed,
        entries=baseline_entries,
    )
    validate_final_evaluator_pair(validation, baseline, source_baseline, canonical_source)
    return validation, baseline


def validate_final_evaluator_pair(
    validation: FinalEvaluatorValidationManifest,
    baseline: FinalEvaluatorValidationManifest,
    source_baseline: AnswerValidationManifest,
    canonical_source: BenchmarkManifest,
) -> None:
    """Validate the final 30/814 split, parent binding, and complete coverage."""

    _validate_final_manifest(validation, source_baseline, canonical_source)
    _validate_final_manifest(baseline, source_baseline, canonical_source)
    if validation.purpose != FINAL_EVALUATOR_VALIDATION:
        raise ValidationManifestError("final evaluator validation has the wrong purpose")
    if baseline.purpose != FINAL_MAIN_DEVELOPMENT_BASELINE:
        raise ValidationManifestError("final evaluator baseline has the wrong purpose")
    if validation.selection_seed != baseline.selection_seed:
        raise ValidationManifestError("final evaluator selection seeds differ")
    validation_ids = {(entry.stable_id, entry.row_index) for entry in validation.entries}
    baseline_ids = {(entry.stable_id, entry.row_index) for entry in baseline.entries}
    source_ids = {(entry.stable_id, entry.row_index) for entry in source_baseline.entries}
    if validation_ids & baseline_ids:
        raise ValidationManifestError("final evaluator validation and baseline IDs overlap")
    if validation_ids | baseline_ids != source_ids:
        raise ValidationManifestError("final evaluator partitions do not cover their source")


def save_final_evaluator_manifest(
    manifest: FinalEvaluatorValidationManifest,
    path: Path,
) -> None:
    """Write a final-evaluator manifest in stable identifier-only form."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_final_evaluator_manifest(
    path: Path,
    source_baseline: AnswerValidationManifest,
    canonical_source: BenchmarkManifest,
) -> FinalEvaluatorValidationManifest:
    """Load and validate a final-evaluator validation or baseline manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationManifestError(
            f"could not load final-evaluator manifest: {error}"
        ) from error
    required = {
        "schema_version",
        "purpose",
        "dataset_id",
        "dataset_revision",
        "config_name",
        "source_split",
        "canonical_development_manifest_sha256",
        "source_baseline_manifest_sha256",
        "selection_seed",
        "entries",
        "manifest_sha256",
    }
    if not isinstance(raw, dict) or raw.keys() != required:
        raise ValidationManifestError("final-evaluator manifest keys differ from schema")
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        raise ValidationManifestError("final-evaluator entries must be a list")
    entries: list[ManifestEntry] = []
    for position, entry_raw in enumerate(entries_raw):
        if not isinstance(entry_raw, dict) or entry_raw.keys() != {"stable_id", "row_index"}:
            raise ValidationManifestError(f"final-evaluator entry {position} is invalid")
        stable_id = entry_raw["stable_id"]
        row_index = entry_raw["row_index"]
        if (
            not isinstance(stable_id, str)
            or isinstance(row_index, bool)
            or not isinstance(row_index, int)
        ):
            raise ValidationManifestError(f"final-evaluator entry {position} is invalid")
        entries.append(ManifestEntry(stable_id=stable_id, row_index=row_index))
    purpose = raw["purpose"]
    if purpose not in {FINAL_EVALUATOR_VALIDATION, FINAL_MAIN_DEVELOPMENT_BASELINE}:
        raise ValidationManifestError("final-evaluator purpose is invalid")
    manifest = FinalEvaluatorValidationManifest(
        schema_version=int(raw["schema_version"]),
        purpose=cast(FinalEvaluatorPurpose, purpose),
        dataset_id=str(raw["dataset_id"]),
        dataset_revision=str(raw["dataset_revision"]),
        config_name=str(raw["config_name"]),
        source_split=str(raw["source_split"]),
        canonical_development_manifest_sha256=str(raw["canonical_development_manifest_sha256"]),
        source_baseline_manifest_sha256=str(raw["source_baseline_manifest_sha256"]),
        selection_seed=str(raw["selection_seed"]),
        entries=tuple(entries),
        manifest_sha256=str(raw["manifest_sha256"]),
    )
    _validate_final_manifest(manifest, source_baseline, canonical_source)
    return manifest


def as_final_benchmark_manifest(
    manifest: FinalEvaluatorValidationManifest,
    source_baseline: AnswerValidationManifest,
    canonical_source: BenchmarkManifest,
    config: EvaluationConfig,
) -> BenchmarkManifest:
    """Adapt only the untouched final evaluator IDs for one bounded run."""

    _validate_final_manifest(manifest, source_baseline, canonical_source)
    if manifest.purpose != FINAL_EVALUATOR_VALIDATION:
        raise ValidationManifestError("only final evaluator validation IDs may be evaluated")
    return BenchmarkManifest(
        schema_version=1,
        dataset_id=config.dataset.repo_id,
        dataset_revision=config.dataset.revision,
        config_name=config.dataset.config_name,
        source_split=config.dataset.source_split,
        partition="development",
        partition_seed=config.partition.seed,
        expected_dataset_examples=config.dataset.expected_examples,
        config_sha256=config.sha256,
        entries=manifest.entries,
        manifest_sha256=manifest.manifest_sha256,
    )


def as_frozen_baseline_manifest(
    manifest: FinalEvaluatorValidationManifest,
    source_baseline: AnswerValidationManifest,
    canonical_source: BenchmarkManifest,
    config: EvaluationConfig,
) -> BenchmarkManifest:
    """Adapt only the frozen 814-ID main-development baseline for evaluation."""

    _validate_final_manifest(manifest, source_baseline, canonical_source)
    if manifest.purpose != FINAL_MAIN_DEVELOPMENT_BASELINE:
        raise ValidationManifestError("only the final main-development baseline may be evaluated")
    if len(manifest.entries) != 814:
        raise ValidationManifestError("frozen main-development baseline requires exactly 814 IDs")
    return BenchmarkManifest(
        schema_version=1,
        dataset_id=config.dataset.repo_id,
        dataset_revision=config.dataset.revision,
        config_name=config.dataset.config_name,
        source_split=config.dataset.source_split,
        partition="development",
        partition_seed=config.partition.seed,
        expected_dataset_examples=config.dataset.expected_examples,
        config_sha256=config.sha256,
        entries=manifest.entries,
        manifest_sha256=manifest.manifest_sha256,
    )
