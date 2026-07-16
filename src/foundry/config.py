"""Typed loading and validation for immutable evaluation configurations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import yaml

_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ConfigError(ValueError):
    """Raised when an evaluation configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class ModelConfig:
    """Pinned model identity and local execution settings."""

    repo_id: str
    revision: str
    dtype: str
    device: str


@dataclass(frozen=True)
class DatasetConfig:
    """Pinned benchmark identity and expected immutable shape."""

    repo_id: str
    revision: str
    config_name: str
    source_split: str
    expected_examples: int


@dataclass(frozen=True)
class PartitionConfig:
    """Rules and destinations for development and sealed-final manifests."""

    seed: str
    sealed_final_size: int
    development_manifest: str
    sealed_final_manifest: str


@dataclass(frozen=True)
class PromptConfig:
    """Stable prompt text used for every benchmark example."""

    system: str
    user_template: str


@dataclass(frozen=True)
class GenerationConfig:
    """Deterministic decoding settings for a controlled comparison."""

    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Complete, validated configuration for one evaluation protocol."""

    schema_version: int
    model: ModelConfig
    dataset: DatasetConfig
    partition: PartitionConfig
    prompt: PromptConfig
    generation: GenerationConfig

    @property
    def sha256(self) -> str:
        """Return a stable hash of the validated semantic configuration."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{location} keys must be strings")
    return cast(dict[str, object], value)


def _check_keys(
    values: dict[str, object],
    *,
    location: str,
    required: set[str],
) -> None:
    missing = required - values.keys()
    unknown = values.keys() - required
    if missing:
        raise ConfigError(f"{location} is missing keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{location} contains unknown keys: {sorted(unknown)}")


def _string(values: dict[str, object], key: str, location: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location}.{key} must be a non-empty string")
    return value


def _integer(values: dict[str, object], key: str, location: str) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{location}.{key} must be an integer")
    return value


def _number(values: dict[str, object], key: str, location: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{location}.{key} must be a number")
    return float(value)


def _boolean(values: dict[str, object], key: str, location: str) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{location}.{key} must be a boolean")
    return value


def _require_revision(revision: str, location: str) -> None:
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ConfigError(f"{location} must be an immutable 40-character lowercase commit hash")


def _parse_model(root: dict[str, object]) -> ModelConfig:
    values = _mapping(root["model"], "model")
    _check_keys(
        values,
        location="model",
        required={"repo_id", "revision", "dtype", "device"},
    )
    model = ModelConfig(
        repo_id=_string(values, "repo_id", "model"),
        revision=_string(values, "revision", "model"),
        dtype=_string(values, "dtype", "model"),
        device=_string(values, "device", "model"),
    )
    _require_revision(model.revision, "model.revision")
    if model.dtype != "float16":
        raise ConfigError("model.dtype must be float16 for the approved RTX 3080 smoke protocol")
    if model.device != "cuda":
        raise ConfigError("model.device must be cuda for the approved real-model smoke protocol")
    return model


def _parse_dataset(root: dict[str, object]) -> DatasetConfig:
    values = _mapping(root["dataset"], "dataset")
    _check_keys(
        values,
        location="dataset",
        required={
            "repo_id",
            "revision",
            "config_name",
            "source_split",
            "expected_examples",
        },
    )
    dataset = DatasetConfig(
        repo_id=_string(values, "repo_id", "dataset"),
        revision=_string(values, "revision", "dataset"),
        config_name=_string(values, "config_name", "dataset"),
        source_split=_string(values, "source_split", "dataset"),
        expected_examples=_integer(values, "expected_examples", "dataset"),
    )
    _require_revision(dataset.revision, "dataset.revision")
    if dataset.expected_examples < 2:
        raise ConfigError("dataset.expected_examples must be at least 2")
    return dataset


def _parse_partition(root: dict[str, object], dataset: DatasetConfig) -> PartitionConfig:
    values = _mapping(root["partition"], "partition")
    _check_keys(
        values,
        location="partition",
        required={
            "seed",
            "sealed_final_size",
            "development_manifest",
            "sealed_final_manifest",
        },
    )
    partition = PartitionConfig(
        seed=_string(values, "seed", "partition"),
        sealed_final_size=_integer(values, "sealed_final_size", "partition"),
        development_manifest=_string(values, "development_manifest", "partition"),
        sealed_final_manifest=_string(values, "sealed_final_manifest", "partition"),
    )
    if not 0 < partition.sealed_final_size < dataset.expected_examples:
        raise ConfigError(
            "partition.sealed_final_size must leave at least one example in each partition"
        )
    if partition.development_manifest == partition.sealed_final_manifest:
        raise ConfigError("development and sealed-final manifest paths must differ")
    return partition


def _parse_prompt(root: dict[str, object]) -> PromptConfig:
    values = _mapping(root["prompt"], "prompt")
    _check_keys(values, location="prompt", required={"system", "user_template"})
    prompt = PromptConfig(
        system=_string(values, "system", "prompt"),
        user_template=_string(values, "user_template", "prompt"),
    )
    if prompt.user_template.count("{question}") != 1:
        raise ConfigError("prompt.user_template must contain exactly one {question} placeholder")
    try:
        prompt.user_template.format(question="validation")
    except (KeyError, ValueError) as error:
        raise ConfigError("prompt.user_template contains an unsupported placeholder") from error
    if "Final answer: <integer>" not in prompt.user_template:
        raise ConfigError("prompt.user_template must state the exact final-answer format")
    return prompt


def _parse_generation(root: dict[str, object]) -> GenerationConfig:
    values = _mapping(root["generation"], "generation")
    _check_keys(
        values,
        location="generation",
        required={"do_sample", "temperature", "top_p", "max_new_tokens"},
    )
    generation = GenerationConfig(
        do_sample=_boolean(values, "do_sample", "generation"),
        temperature=_number(values, "temperature", "generation"),
        top_p=_number(values, "top_p", "generation"),
        max_new_tokens=_integer(values, "max_new_tokens", "generation"),
    )
    if generation.do_sample:
        raise ConfigError("generation.do_sample must be false for deterministic evaluation")
    if generation.temperature != 0.0:
        raise ConfigError("generation.temperature must be 0.0 for deterministic evaluation")
    if generation.top_p != 1.0:
        raise ConfigError("generation.top_p must be 1.0 for deterministic evaluation")
    if not 1 <= generation.max_new_tokens <= 2048:
        raise ConfigError("generation.max_new_tokens must be between 1 and 2048")
    return generation


def load_config(path: Path) -> EvaluationConfig:
    """Load a YAML evaluation config and reject unpinned or unknown settings."""

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"could not load configuration {path}: {error}") from error

    root = _mapping(raw, "root")
    _check_keys(
        root,
        location="root",
        required={"schema_version", "model", "dataset", "partition", "prompt", "generation"},
    )
    schema_version = _integer(root, "schema_version", "root")
    if schema_version != 1:
        raise ConfigError("schema_version must be 1")

    model = _parse_model(root)
    dataset = _parse_dataset(root)
    return EvaluationConfig(
        schema_version=schema_version,
        model=model,
        dataset=dataset,
        partition=_parse_partition(root, dataset),
        prompt=_parse_prompt(root),
        generation=_parse_generation(root),
    )
