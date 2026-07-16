"""Deterministic benchmark partition manifests containing identifiers only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

from foundry.config import EvaluationConfig

PartitionName = Literal["development", "sealed_final"]


class ManifestError(ValueError):
    """Raised when a benchmark manifest is invalid or has been modified."""


@dataclass(frozen=True)
class ManifestEntry:
    """Stable identity and pinned row position for one benchmark example."""

    stable_id: str
    row_index: int


@dataclass(frozen=True)
class BenchmarkManifest:
    """A label-free partition of an immutable benchmark revision."""

    schema_version: int
    dataset_id: str
    dataset_revision: str
    config_name: str
    source_split: str
    partition: PartitionName
    partition_seed: str
    expected_dataset_examples: int
    config_sha256: str
    entries: tuple[ManifestEntry, ...]
    manifest_sha256: str


def _entry_identity(config: EvaluationConfig, row_index: int) -> str:
    dataset = config.dataset
    return (
        f"{dataset.repo_id}@{dataset.revision}:"
        f"{dataset.config_name}:{dataset.source_split}:{row_index}"
    )


def _stable_id(config: EvaluationConfig, row_index: int) -> str:
    identity = _entry_identity(config, row_index)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _partition_rank(config: EvaluationConfig, row_index: int) -> str:
    identity = _entry_identity(config, row_index)
    material = f"{config.partition.seed}:{identity}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _unsigned_payload(manifest: BenchmarkManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload.pop("manifest_sha256")
    return cast(dict[str, object], payload)


def _digest_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _create_manifest(
    config: EvaluationConfig,
    partition: PartitionName,
    entries: tuple[ManifestEntry, ...],
) -> BenchmarkManifest:
    unsigned = BenchmarkManifest(
        schema_version=1,
        dataset_id=config.dataset.repo_id,
        dataset_revision=config.dataset.revision,
        config_name=config.dataset.config_name,
        source_split=config.dataset.source_split,
        partition=partition,
        partition_seed=config.partition.seed,
        expected_dataset_examples=config.dataset.expected_examples,
        config_sha256=config.sha256,
        entries=entries,
        manifest_sha256="",
    )
    return replace(
        unsigned,
        manifest_sha256=_digest_payload(_unsigned_payload(unsigned)),
    )


def build_manifests(
    config: EvaluationConfig,
) -> tuple[BenchmarkManifest, BenchmarkManifest]:
    """Create deterministic, exhaustive, non-overlapping development/final manifests."""

    ranked_indices = sorted(
        range(config.dataset.expected_examples),
        key=lambda index: (_partition_rank(config, index), index),
    )
    final_indices = set(ranked_indices[: config.partition.sealed_final_size])

    development_entries = tuple(
        ManifestEntry(stable_id=_stable_id(config, index), row_index=index)
        for index in range(config.dataset.expected_examples)
        if index not in final_indices
    )
    sealed_final_entries = tuple(
        ManifestEntry(stable_id=_stable_id(config, index), row_index=index)
        for index in range(config.dataset.expected_examples)
        if index in final_indices
    )

    development = _create_manifest(config, "development", development_entries)
    sealed_final = _create_manifest(config, "sealed_final", sealed_final_entries)
    validate_manifest_pair(development, sealed_final, config)
    return development, sealed_final


def _validate_manifest(manifest: BenchmarkManifest, config: EvaluationConfig) -> None:
    if manifest.schema_version != 1:
        raise ManifestError("manifest schema_version must be 1")
    if manifest.dataset_id != config.dataset.repo_id:
        raise ManifestError("manifest dataset_id does not match the evaluation config")
    if manifest.dataset_revision != config.dataset.revision:
        raise ManifestError("manifest dataset_revision does not match the evaluation config")
    if manifest.config_name != config.dataset.config_name:
        raise ManifestError("manifest config_name does not match the evaluation config")
    if manifest.source_split != config.dataset.source_split:
        raise ManifestError("manifest source_split does not match the evaluation config")
    if manifest.partition_seed != config.partition.seed:
        raise ManifestError("manifest partition_seed does not match the evaluation config")
    if manifest.expected_dataset_examples != config.dataset.expected_examples:
        raise ManifestError("manifest expected example count does not match the evaluation config")
    if manifest.config_sha256 != config.sha256:
        raise ManifestError("manifest config hash does not match the evaluation config")
    expected_digest = _digest_payload(_unsigned_payload(manifest))
    if manifest.manifest_sha256 != expected_digest:
        raise ManifestError("manifest digest is invalid; the file may have been modified")

    row_indices = [entry.row_index for entry in manifest.entries]
    stable_ids = [entry.stable_id for entry in manifest.entries]
    if len(row_indices) != len(set(row_indices)):
        raise ManifestError("manifest contains duplicate row indices")
    if len(stable_ids) != len(set(stable_ids)):
        raise ManifestError("manifest contains duplicate stable identifiers")
    for entry in manifest.entries:
        if not 0 <= entry.row_index < config.dataset.expected_examples:
            raise ManifestError(f"row index {entry.row_index} is outside the pinned dataset")
        if entry.stable_id != _stable_id(config, entry.row_index):
            raise ManifestError(f"stable identifier mismatch at row {entry.row_index}")


def validate_manifest_pair(
    development: BenchmarkManifest,
    sealed_final: BenchmarkManifest,
    config: EvaluationConfig,
) -> None:
    """Prove that manifests are valid, disjoint, and cover the pinned dataset."""

    _validate_manifest(development, config)
    _validate_manifest(sealed_final, config)
    if development.partition != "development":
        raise ManifestError("development manifest has the wrong partition label")
    if sealed_final.partition != "sealed_final":
        raise ManifestError("sealed-final manifest has the wrong partition label")

    development_rows = {entry.row_index for entry in development.entries}
    final_rows = {entry.row_index for entry in sealed_final.entries}
    overlap = development_rows & final_rows
    if overlap:
        raise ManifestError(f"development and sealed-final manifests overlap: {sorted(overlap)}")

    expected_rows = set(range(config.dataset.expected_examples))
    if development_rows | final_rows != expected_rows:
        raise ManifestError("development and sealed-final manifests do not cover the dataset")
    if len(sealed_final.entries) != config.partition.sealed_final_size:
        raise ManifestError("sealed-final manifest size does not match the evaluation config")


def save_manifest(manifest: BenchmarkManifest, path: Path) -> None:
    """Write a manifest in a stable, human-reviewable JSON representation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")


def load_manifest(path: Path, config: EvaluationConfig) -> BenchmarkManifest:
    """Load, type-check, and cryptographically verify a manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"could not load manifest {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ManifestError("manifest root must be an object")

    required = {
        "schema_version",
        "dataset_id",
        "dataset_revision",
        "config_name",
        "source_split",
        "partition",
        "partition_seed",
        "expected_dataset_examples",
        "config_sha256",
        "entries",
        "manifest_sha256",
    }
    if raw.keys() != required:
        raise ManifestError(
            f"manifest keys differ from schema; missing={sorted(required - raw.keys())}, "
            f"unknown={sorted(raw.keys() - required)}"
        )
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        raise ManifestError("manifest entries must be a list")

    entries: list[ManifestEntry] = []
    for position, entry_raw in enumerate(entries_raw):
        if not isinstance(entry_raw, dict) or entry_raw.keys() != {"stable_id", "row_index"}:
            raise ManifestError(f"manifest entry {position} has an invalid schema")
        stable_id = entry_raw["stable_id"]
        row_index = entry_raw["row_index"]
        if not isinstance(stable_id, str):
            raise ManifestError(f"manifest entry {position} stable_id must be a string")
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise ManifestError(f"manifest entry {position} row_index must be an integer")
        entries.append(ManifestEntry(stable_id=stable_id, row_index=row_index))

    partition = raw["partition"]
    if partition not in {"development", "sealed_final"}:
        raise ManifestError("manifest partition must be development or sealed_final")

    try:
        manifest = BenchmarkManifest(
            schema_version=int(raw["schema_version"]),
            dataset_id=str(raw["dataset_id"]),
            dataset_revision=str(raw["dataset_revision"]),
            config_name=str(raw["config_name"]),
            source_split=str(raw["source_split"]),
            partition=cast(PartitionName, partition),
            partition_seed=str(raw["partition_seed"]),
            expected_dataset_examples=int(raw["expected_dataset_examples"]),
            config_sha256=str(raw["config_sha256"]),
            entries=tuple(entries),
            manifest_sha256=str(raw["manifest_sha256"]),
        )
    except (TypeError, ValueError) as error:
        raise ManifestError("manifest scalar fields have invalid types") from error
    _validate_manifest(manifest, config)
    return manifest


def require_partition_access(
    manifest: BenchmarkManifest,
    *,
    allow_sealed_final: bool,
) -> None:
    """Prevent accidental final-holdout evaluation without an explicit flag."""

    if manifest.partition == "sealed_final" and not allow_sealed_final:
        raise ManifestError(
            "sealed-final access denied; pass an explicit allow_sealed_final=True only "
            "after the candidate and evaluation settings are frozen"
        )
