"""Deterministic development-only manifests for prompt-format calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

from foundry.config import EvaluationConfig
from foundry.evaluation.manifests import BenchmarkManifest, ManifestEntry

SubsetPurpose = Literal["prompt_format_calibration", "main_development_baseline"]
PROMPT_FORMAT_CALIBRATION: SubsetPurpose = "prompt_format_calibration"
MAIN_DEVELOPMENT_BASELINE: SubsetPurpose = "main_development_baseline"


class CalibrationError(ValueError):
    """Raised when a development-only calibration artifact is invalid."""


@dataclass(frozen=True)
class DevelopmentSubsetManifest:
    """Identifier-only subset derived from the canonical development manifest."""

    schema_version: int
    purpose: SubsetPurpose
    dataset_id: str
    dataset_revision: str
    config_name: str
    source_split: str
    source_development_manifest_sha256: str
    selection_seed: str
    entries: tuple[ManifestEntry, ...]
    manifest_sha256: str


def _unsigned_payload(manifest: DevelopmentSubsetManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload.pop("manifest_sha256")
    return cast(dict[str, object], payload)


def _digest_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _create_subset(
    source: BenchmarkManifest,
    *,
    purpose: SubsetPurpose,
    selection_seed: str,
    entries: tuple[ManifestEntry, ...],
) -> DevelopmentSubsetManifest:
    unsigned = DevelopmentSubsetManifest(
        schema_version=1,
        purpose=purpose,
        dataset_id=source.dataset_id,
        dataset_revision=source.dataset_revision,
        config_name=source.config_name,
        source_split=source.source_split,
        source_development_manifest_sha256=source.manifest_sha256,
        selection_seed=selection_seed,
        entries=entries,
        manifest_sha256="",
    )
    return replace(unsigned, manifest_sha256=_digest_payload(_unsigned_payload(unsigned)))


def _selection_rank(selection_seed: str, entry: ManifestEntry) -> str:
    material = f"{selection_seed}:{entry.stable_id}:{entry.row_index}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_format_calibration_manifests(
    source: BenchmarkManifest,
    *,
    calibration_size: int,
    selection_seed: str,
) -> tuple[DevelopmentSubsetManifest, DevelopmentSubsetManifest]:
    """Split canonical development IDs into calibration and future-baseline sets."""

    if source.partition != "development":
        raise CalibrationError("format calibration must derive from a development manifest")
    if not selection_seed.strip():
        raise CalibrationError("selection_seed must be non-empty")
    if not 0 < calibration_size < len(source.entries):
        raise CalibrationError("calibration_size must leave identifiers in both subsets")

    ranked = sorted(
        source.entries,
        key=lambda entry: (_selection_rank(selection_seed, entry), entry.row_index),
    )
    calibration_identities = {
        (entry.stable_id, entry.row_index) for entry in ranked[:calibration_size]
    }
    calibration_entries = tuple(
        entry
        for entry in source.entries
        if (entry.stable_id, entry.row_index) in calibration_identities
    )
    baseline_entries = tuple(
        entry
        for entry in source.entries
        if (entry.stable_id, entry.row_index) not in calibration_identities
    )
    calibration = _create_subset(
        source,
        purpose=PROMPT_FORMAT_CALIBRATION,
        selection_seed=selection_seed,
        entries=calibration_entries,
    )
    baseline = _create_subset(
        source,
        purpose=MAIN_DEVELOPMENT_BASELINE,
        selection_seed=selection_seed,
        entries=baseline_entries,
    )
    validate_format_calibration_pair(calibration, baseline, source)
    return calibration, baseline


def _validate_subset(
    manifest: DevelopmentSubsetManifest,
    source: BenchmarkManifest,
) -> None:
    if manifest.schema_version != 1:
        raise CalibrationError("development subset schema_version must be 1")
    if manifest.purpose not in {PROMPT_FORMAT_CALIBRATION, MAIN_DEVELOPMENT_BASELINE}:
        raise CalibrationError("development subset purpose is invalid")
    if source.partition != "development":
        raise CalibrationError("development subsets require a development source manifest")
    expected_identity = (
        source.dataset_id,
        source.dataset_revision,
        source.config_name,
        source.source_split,
        source.manifest_sha256,
    )
    actual_identity = (
        manifest.dataset_id,
        manifest.dataset_revision,
        manifest.config_name,
        manifest.source_split,
        manifest.source_development_manifest_sha256,
    )
    if actual_identity != expected_identity:
        raise CalibrationError("development subset does not match its source manifest")
    if not manifest.selection_seed.strip():
        raise CalibrationError("development subset selection_seed must be non-empty")
    if manifest.manifest_sha256 != _digest_payload(_unsigned_payload(manifest)):
        raise CalibrationError("development subset digest is invalid")

    identities = [(entry.stable_id, entry.row_index) for entry in manifest.entries]
    if len(identities) != len(set(identities)):
        raise CalibrationError("development subset contains duplicate identifiers")
    source_identities = {(entry.stable_id, entry.row_index) for entry in source.entries}
    if not set(identities).issubset(source_identities):
        raise CalibrationError("development subset contains an identifier outside development")


def validate_format_calibration_pair(
    calibration: DevelopmentSubsetManifest,
    baseline: DevelopmentSubsetManifest,
    source: BenchmarkManifest,
) -> None:
    """Validate purpose, disjointness, and complete development coverage."""

    _validate_subset(calibration, source)
    _validate_subset(baseline, source)
    if calibration.purpose != PROMPT_FORMAT_CALIBRATION:
        raise CalibrationError("calibration manifest has the wrong purpose")
    if baseline.purpose != MAIN_DEVELOPMENT_BASELINE:
        raise CalibrationError("baseline manifest has the wrong purpose")
    if calibration.selection_seed != baseline.selection_seed:
        raise CalibrationError("development subset selection seeds differ")

    calibration_ids = {(entry.stable_id, entry.row_index) for entry in calibration.entries}
    baseline_ids = {(entry.stable_id, entry.row_index) for entry in baseline.entries}
    source_ids = {(entry.stable_id, entry.row_index) for entry in source.entries}
    if calibration_ids & baseline_ids:
        raise CalibrationError("calibration and future-baseline identifiers overlap")
    if calibration_ids | baseline_ids != source_ids:
        raise CalibrationError("development subsets do not cover the source manifest")


def save_development_subset(manifest: DevelopmentSubsetManifest, path: Path) -> None:
    """Write an identifier-only development subset in a stable representation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_development_subset(
    path: Path,
    source: BenchmarkManifest,
) -> DevelopmentSubsetManifest:
    """Load and validate an identifier-only development subset."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"could not load development subset {path}: {error}") from error
    if not isinstance(raw, dict):
        raise CalibrationError("development subset root must be an object")
    required = {
        "schema_version",
        "purpose",
        "dataset_id",
        "dataset_revision",
        "config_name",
        "source_split",
        "source_development_manifest_sha256",
        "selection_seed",
        "entries",
        "manifest_sha256",
    }
    if raw.keys() != required:
        raise CalibrationError("development subset keys differ from schema")
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        raise CalibrationError("development subset entries must be a list")
    entries: list[ManifestEntry] = []
    for position, entry_raw in enumerate(entries_raw):
        if not isinstance(entry_raw, dict) or entry_raw.keys() != {"stable_id", "row_index"}:
            raise CalibrationError(f"development subset entry {position} has an invalid schema")
        stable_id = entry_raw["stable_id"]
        row_index = entry_raw["row_index"]
        if not isinstance(stable_id, str):
            raise CalibrationError(f"development subset entry {position} stable_id is invalid")
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise CalibrationError(f"development subset entry {position} row_index is invalid")
        entries.append(ManifestEntry(stable_id=stable_id, row_index=row_index))

    purpose = raw["purpose"]
    if purpose not in {PROMPT_FORMAT_CALIBRATION, MAIN_DEVELOPMENT_BASELINE}:
        raise CalibrationError("development subset purpose is invalid")
    try:
        manifest = DevelopmentSubsetManifest(
            schema_version=int(raw["schema_version"]),
            purpose=cast(SubsetPurpose, purpose),
            dataset_id=str(raw["dataset_id"]),
            dataset_revision=str(raw["dataset_revision"]),
            config_name=str(raw["config_name"]),
            source_split=str(raw["source_split"]),
            source_development_manifest_sha256=str(raw["source_development_manifest_sha256"]),
            selection_seed=str(raw["selection_seed"]),
            entries=tuple(entries),
            manifest_sha256=str(raw["manifest_sha256"]),
        )
    except (TypeError, ValueError) as error:
        raise CalibrationError("development subset scalar fields have invalid types") from error
    _validate_subset(manifest, source)
    return manifest


def assert_prompt_only_variant(base: EvaluationConfig, variant: EvaluationConfig) -> None:
    """Reject calibration variants that change anything other than prompt text."""

    controls_match = (
        variant.schema_version == base.schema_version
        and variant.model == base.model
        and variant.dataset == base.dataset
        and variant.partition == base.partition
        and variant.generation == base.generation
    )
    if not controls_match:
        raise CalibrationError("format calibration variants may change only prompt text")


def as_development_benchmark_manifest(
    subset: DevelopmentSubsetManifest,
    source: BenchmarkManifest,
    config: EvaluationConfig,
) -> BenchmarkManifest:
    """Adapt a validated calibration subset for the existing evaluation runner."""

    _validate_subset(subset, source)
    if subset.purpose != PROMPT_FORMAT_CALIBRATION:
        raise CalibrationError("only prompt-format calibration IDs may use calibration evaluation")
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
        entries=subset.entries,
        manifest_sha256=subset.manifest_sha256,
    )
