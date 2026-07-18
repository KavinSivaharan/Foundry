"""Pinned local sentence embeddings for contamination screening."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as functional
import yaml
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

from foundry.synthesis.contamination import ContaminationOutcome


class SemanticArtifactError(RuntimeError):
    """Raised when the pinned semantic artifact or inference result is invalid."""


@dataclass(frozen=True)
class SemanticThresholds:
    """Immutable similarity bands from the Milestone 3 design."""

    automatic_pass_below: float
    manual_review_at_or_above: float
    automatic_reject_at_or_above: float

    def __post_init__(self) -> None:
        if (
            self.automatic_pass_below != 0.75
            or self.manual_review_at_or_above != 0.75
            or self.automatic_reject_at_or_above != 0.82
        ):
            raise ValueError("semantic thresholds differ from the frozen 0.75/0.82 policy")

    def classify(self, similarity: float) -> ContaminationOutcome:
        """Classify one cosine similarity under the frozen thresholds."""

        if similarity >= self.automatic_reject_at_or_above:
            return ContaminationOutcome.REJECT
        if similarity >= self.manual_review_at_or_above:
            return ContaminationOutcome.MANUAL_REVIEW
        return ContaminationOutcome.PASS


@dataclass(frozen=True)
class SemanticArtifactConfig:
    """Operational pin for a standard local Transformers sentence encoder."""

    artifact_id: str
    model_id: str
    revision: str
    license_id: str
    expected_download_bytes: int
    embedding_dimension: int
    pooling: str
    normalization: str
    max_length: int
    device: str
    dtype: str
    batch_size: int
    trust_remote_code: bool
    local_files_only_after_download: bool
    required_files: tuple[str, ...]
    cache_root: Path
    thresholds: SemanticThresholds

    def __post_init__(self) -> None:
        if self.model_id != "sentence-transformers/all-MiniLM-L6-v2":
            raise ValueError("unexpected semantic model ID")
        if self.revision != "1110a243fdf4706b3f48f1d95db1a4f5529b4d41":
            raise ValueError("semantic model revision is not pinned")
        if self.license_id != "apache-2.0":
            raise ValueError("semantic artifact license differs from the approved pin")
        if self.pooling != "attention_mask_weighted_mean" or self.normalization != "l2":
            raise ValueError("semantic pooling or normalization differs from the approved pin")
        if self.embedding_dimension != 384 or self.max_length != 256:
            raise ValueError("semantic dimensions differ from the approved pin")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("the bounded smoke requires deterministic CPU float32 inference")
        if self.trust_remote_code or not self.local_files_only_after_download:
            raise ValueError("remote code and network loading must remain disabled")
        if self.expected_download_bytes <= 0 or self.expected_download_bytes > 500_000_000:
            raise ValueError("semantic artifact exceeds the approved size boundary")
        if self.batch_size < 1:
            raise ValueError("semantic batch size must be positive")

    def snapshot_path(self, repository_root: Path) -> Path:
        """Return the exact ignored local snapshot directory."""

        return repository_root / self.cache_root / self.model_id.rsplit("/", 1)[-1] / self.revision


@dataclass(frozen=True)
class ArtifactFileEvidence:
    """Hash and size for one approved local artifact file."""

    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SemanticFixturePair:
    """One original pair used to validate frozen similarity behavior."""

    fixture_id: str
    category: str
    left: str
    right: str
    expected_outcome: ContaminationOutcome


@dataclass(frozen=True)
class SemanticFixtureResult:
    """Content-free result for one fixture pair."""

    fixture_id: str
    category: str
    similarity: float
    outcome: ContaminationOutcome
    expected_outcome: ContaminationOutcome


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SemanticArtifactError(f"{name} must be a mapping")
    return cast(dict[str, object], value)


def _required(mapping: dict[str, object], name: str, expected_type: type[Any]) -> Any:
    if name not in mapping or not isinstance(mapping[name], expected_type):
        raise SemanticArtifactError(f"semantic config field {name!r} has an invalid type")
    return mapping[name]


def load_semantic_artifact_config(path: Path) -> SemanticArtifactConfig:
    """Load and validate the exact selected semantic artifact configuration."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SemanticArtifactError(f"could not load semantic config: {error}") from error
    root = _mapping(raw, "semantic config")
    thresholds_raw = _mapping(root.get("semantic_thresholds"), "semantic_thresholds")
    files_raw = root.get("required_files")
    if not isinstance(files_raw, list) or not all(isinstance(item, str) for item in files_raw):
        raise SemanticArtifactError("required_files must be a string list")
    return SemanticArtifactConfig(
        artifact_id=cast(str, _required(root, "artifact_id", str)),
        model_id=cast(str, _required(root, "model_id", str)),
        revision=cast(str, _required(root, "revision", str)),
        license_id=cast(str, _required(root, "license", str)),
        expected_download_bytes=cast(int, _required(root, "expected_download_bytes", int)),
        embedding_dimension=cast(int, _required(root, "embedding_dimension", int)),
        pooling=cast(str, _required(root, "pooling", str)),
        normalization=cast(str, _required(root, "normalization", str)),
        max_length=cast(int, _required(root, "max_length", int)),
        device=cast(str, _required(root, "device", str)),
        dtype=cast(str, _required(root, "dtype", str)),
        batch_size=cast(int, _required(root, "batch_size", int)),
        trust_remote_code=cast(bool, _required(root, "trust_remote_code", bool)),
        local_files_only_after_download=cast(
            bool, _required(root, "local_files_only_after_download", bool)
        ),
        required_files=tuple(cast(list[str], files_raw)),
        cache_root=Path(cast(str, _required(root, "cache_root", str))),
        thresholds=SemanticThresholds(
            automatic_pass_below=float(_required(thresholds_raw, "automatic_pass_below", float)),
            manual_review_at_or_above=float(
                _required(thresholds_raw, "manual_review_at_or_above", float)
            ),
            automatic_reject_at_or_above=float(
                _required(thresholds_raw, "automatic_reject_at_or_above", float)
            ),
        ),
    )


def verify_local_artifact(
    config: SemanticArtifactConfig,
    repository_root: Path,
) -> tuple[ArtifactFileEvidence, ...]:
    """Verify required local-only files, total size, and stable content hashes."""

    snapshot = config.snapshot_path(repository_root)
    evidence: list[ArtifactFileEvidence] = []
    for relative in config.required_files:
        path = snapshot / relative
        if not path.is_file():
            raise SemanticArtifactError(f"semantic artifact is missing {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence.append(ArtifactFileEvidence(relative, path.stat().st_size, digest))
    if sum(item.size_bytes for item in evidence) != config.expected_download_bytes:
        raise SemanticArtifactError("semantic artifact size differs from the approved metadata")
    return tuple(evidence)


class PinnedSentenceEncoder:
    """Deterministic CPU encoder loaded only from the approved local snapshot."""

    def __init__(self, config: SemanticArtifactConfig, repository_root: Path) -> None:
        verify_local_artifact(config, repository_root)
        snapshot = config.snapshot_path(repository_root)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.config = config
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model: Any = AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        self.model.eval()
        self.model.to("cpu")

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return L2-normalized CPU float32 embeddings in deterministic batches."""

        if not texts or any(not text.strip() for text in texts):
            raise SemanticArtifactError("semantic inputs must be non-empty strings")
        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch_texts = list(texts[start : start + self.config.batch_size])
            encoded: Any = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output: Any = self.model(**encoded)
            token_embeddings = cast(torch.Tensor, output.last_hidden_state).to(torch.float32)
            attention = cast(torch.Tensor, encoded["attention_mask"])
            expanded = attention.unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
            batches.append(functional.normalize(pooled, p=2, dim=1).cpu())
        embeddings = torch.cat(batches, dim=0)
        if embeddings.shape != (len(texts), self.config.embedding_dimension):
            raise SemanticArtifactError("semantic embedding shape differs from the approved pin")
        if not torch.isfinite(embeddings).all():
            raise SemanticArtifactError("semantic inference produced a non-finite value")
        return embeddings

    @staticmethod
    def cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return cosine scores for already normalized embeddings."""

        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise SemanticArtifactError("semantic matrices have incompatible shapes")
        return left @ right.transpose(0, 1)


def evaluate_semantic_fixtures(
    encoder: PinnedSentenceEncoder,
    fixtures: Sequence[SemanticFixturePair],
) -> tuple[tuple[SemanticFixtureResult, ...], str]:
    """Evaluate original fixtures and hash their deterministic embeddings and results."""

    texts: list[str] = []
    for fixture in fixtures:
        texts.extend((fixture.left, fixture.right))
    embeddings = encoder.encode(texts)
    results: list[SemanticFixtureResult] = []
    for index, fixture in enumerate(fixtures):
        similarity = float(torch.dot(embeddings[index * 2], embeddings[index * 2 + 1]).item())
        results.append(
            SemanticFixtureResult(
                fixture_id=fixture.fixture_id,
                category=fixture.category,
                similarity=similarity,
                outcome=encoder.config.thresholds.classify(similarity),
                expected_outcome=fixture.expected_outcome,
            )
        )
    payload = {
        "artifact_id": encoder.config.artifact_id,
        "embedding_sha256": hashlib.sha256(embeddings.contiguous().numpy().tobytes()).hexdigest(),
        "results": [
            {
                "fixture_id": item.fixture_id,
                "category": item.category,
                "similarity": format(item.similarity, ".8f"),
                "outcome": item.outcome,
                "expected_outcome": item.expected_outcome,
            }
            for item in results
        ],
    }
    output_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return tuple(results), output_hash
