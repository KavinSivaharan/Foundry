"""Frozen design policies for a future bounded local-model realization smoke."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.realization.model_contracts import SYSTEM_PROMPT_SHA256

_REVISION = re.compile(r"^[0-9a-f]{40}$")


class GenerationStrategy(StrEnum):
    """Audited candidate-generation strategies considered in Milestone 5A."""

    GREEDY = "greedy"
    DETERMINISTIC_BEAM = "deterministic_beam"
    SEEDED_SAMPLING = "seeded_sampling"


class InternalDiversityMode(StrEnum):
    """Relationship between semantic screening and internal diversity."""

    CALIBRATE_SEPARATELY = "calibrate_separately_before_smoke"


@dataclass(frozen=True)
class PinnedRealizationModel:
    """Immutable model identity and runtime boundary; no loading behavior."""

    repo_id: str
    revision: str
    license_id: str
    minimum_transformers: str
    trust_remote_code: bool
    thinking_enabled: bool
    chat_template_sha256: str


@dataclass(frozen=True)
class DeterministicGenerationPolicy:
    """Fixed future decoding behavior without retry-until-success."""

    strategy: GenerationStrategy
    seed: int
    do_sample: bool
    num_beams: int
    num_return_sequences: int
    candidates_per_ir: int
    max_new_tokens: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        if self.strategy is not GenerationStrategy.DETERMINISTIC_BEAM:
            raise ValueError("Milestone 5A freezes deterministic beam search")
        if self.do_sample:
            raise ValueError("deterministic realization cannot sample")
        if not (self.num_beams == self.num_return_sequences == self.candidates_per_ir == 3):
            raise ValueError("future smoke requires exactly three fixed beam candidates per IR")
        if self.max_new_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("generation limits must be positive")


@dataclass(frozen=True)
class SemanticScreeningPolicy:
    """Separate benchmark-contamination and internal-diversity roles."""

    artifact_id: str
    revision: str
    benchmark_manual_review_threshold: float
    benchmark_rejection_threshold: float
    internal_diversity_mode: InternalDiversityMode
    internal_semantic_threshold: float | None

    def __post_init__(self) -> None:
        if not (
            0 <= self.benchmark_manual_review_threshold < self.benchmark_rejection_threshold <= 1
        ):
            raise ValueError("benchmark semantic thresholds are invalid")
        if self.internal_semantic_threshold is not None:
            raise ValueError("internal semantic threshold must await independent calibration")


@dataclass(frozen=True)
class FutureSmokePolicy:
    """Fixed budget, allocations, and unchanged quality gates."""

    ir_attempts: int
    targeted_allocations: dict[str, int]
    generic_allocations: dict[str, int]
    output_contract_per_group: int
    minimum_clean_accepts: int
    minimum_accepts_per_family: int
    maximum_false_labels: int
    maximum_semantic_drift_accepts: int
    maximum_invalid_accepts: int
    maximum_unresolved_contamination: int

    def __post_init__(self) -> None:
        if self.ir_attempts != 120:
            raise ValueError("future realization smoke is bounded to 120 IR attempts")
        if sum(self.targeted_allocations.values()) != 60:
            raise ValueError("targeted allocations must total 60")
        if sum(self.generic_allocations.values()) != 60:
            raise ValueError("generic allocations must total 60")
        if self.output_contract_per_group != 12:
            raise ValueError("output-contract track must remain 20% per group")
        if self.minimum_clean_accepts != 90 or self.minimum_accepts_per_family != 15:
            raise ValueError("readiness acceptance gates cannot be lowered")
        if any(
            value != 0
            for value in (
                self.maximum_false_labels,
                self.maximum_semantic_drift_accepts,
                self.maximum_invalid_accepts,
                self.maximum_unresolved_contamination,
            )
        ):
            raise ValueError("all safety defect tolerances must remain zero")


@dataclass(frozen=True)
class LocalRealizationDesign:
    """Typed design configuration for the separately approvable pivot smoke."""

    schema_version: int
    design_only: bool
    system_prompt_sha256: str
    response_schema_version: int
    primary_model: PinnedRealizationModel
    fallback_model: PinnedRealizationModel
    generation: DeterministicGenerationPolicy
    semantic_screening: SemanticScreeningPolicy
    future_smoke: FutureSmokePolicy


class DesignConfigError(ValueError):
    """Raised when the design YAML weakens a frozen boundary."""


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DesignConfigError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _exact_keys(values: dict[str, object], location: str, expected: set[str]) -> None:
    if set(values) != expected:
        raise DesignConfigError(f"{location} keys must exactly match {sorted(expected)}")


def _model(values: object, location: str) -> PinnedRealizationModel:
    data = _mapping(values, location)
    expected = {
        "repo_id",
        "revision",
        "license_id",
        "minimum_transformers",
        "trust_remote_code",
        "thinking_enabled",
        "chat_template_sha256",
    }
    _exact_keys(data, location, expected)
    string_keys = expected - {"trust_remote_code", "thinking_enabled"}
    if not all(isinstance(data[key], str) for key in string_keys):
        raise DesignConfigError(f"{location} string fields are invalid")
    if not isinstance(data["trust_remote_code"], bool) or not isinstance(
        data["thinking_enabled"], bool
    ):
        raise DesignConfigError(f"{location} boolean fields are invalid")
    model = PinnedRealizationModel(
        repo_id=cast(str, data["repo_id"]),
        revision=cast(str, data["revision"]),
        license_id=cast(str, data["license_id"]),
        minimum_transformers=cast(str, data["minimum_transformers"]),
        trust_remote_code=data["trust_remote_code"],
        thinking_enabled=data["thinking_enabled"],
        chat_template_sha256=cast(str, data["chat_template_sha256"]),
    )
    if not _REVISION.fullmatch(model.revision):
        raise DesignConfigError(f"{location}.revision must be an immutable commit")
    if model.trust_remote_code or model.thinking_enabled:
        raise DesignConfigError(f"{location} must disable remote code and thinking")
    return model


def _int(data: dict[str, object], key: str, location: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise DesignConfigError(f"{location}.{key} must be an integer")
    return value


def _bool(data: dict[str, object], key: str, location: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise DesignConfigError(f"{location}.{key} must be a boolean")
    return value


def _float(data: dict[str, object], key: str, location: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DesignConfigError(f"{location}.{key} must be numeric")
    return float(value)


def _allocation(value: object, location: str) -> dict[str, int]:
    data = _mapping(value, location)
    expected = {"bookkeeping", "rates", "discrete"}
    _exact_keys(data, location, expected)
    return {key: _int(data, key, location) for key in sorted(expected)}


def load_local_realization_design(path: Path) -> LocalRealizationDesign:
    """Load the design-only YAML and enforce all frozen smoke boundaries."""

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(loaded, "root")
    _exact_keys(
        root,
        "root",
        {
            "schema_version",
            "design_only",
            "system_prompt_sha256",
            "response_schema_version",
            "primary_model",
            "fallback_model",
            "generation",
            "semantic_screening",
            "future_smoke",
        },
    )
    if root["design_only"] is not True:
        raise DesignConfigError("configuration must remain design-only")
    if root["system_prompt_sha256"] != SYSTEM_PROMPT_SHA256:
        raise DesignConfigError("system prompt hash differs from the frozen contract")

    generation_data = _mapping(root["generation"], "generation")
    _exact_keys(
        generation_data,
        "generation",
        {
            "strategy",
            "seed",
            "do_sample",
            "num_beams",
            "num_return_sequences",
            "candidates_per_ir",
            "max_new_tokens",
            "timeout_seconds",
        },
    )
    semantic_data = _mapping(root["semantic_screening"], "semantic_screening")
    _exact_keys(
        semantic_data,
        "semantic_screening",
        {
            "artifact_id",
            "revision",
            "benchmark_manual_review_threshold",
            "benchmark_rejection_threshold",
            "internal_diversity_mode",
            "internal_semantic_threshold",
        },
    )
    smoke_data = _mapping(root["future_smoke"], "future_smoke")
    _exact_keys(
        smoke_data,
        "future_smoke",
        {
            "ir_attempts",
            "targeted_allocations",
            "generic_allocations",
            "output_contract_per_group",
            "minimum_clean_accepts",
            "minimum_accepts_per_family",
            "maximum_false_labels",
            "maximum_semantic_drift_accepts",
            "maximum_invalid_accepts",
            "maximum_unresolved_contamination",
        },
    )
    try:
        generation = DeterministicGenerationPolicy(
            strategy=GenerationStrategy(str(generation_data["strategy"])),
            seed=_int(generation_data, "seed", "generation"),
            do_sample=_bool(generation_data, "do_sample", "generation"),
            num_beams=_int(generation_data, "num_beams", "generation"),
            num_return_sequences=_int(generation_data, "num_return_sequences", "generation"),
            candidates_per_ir=_int(generation_data, "candidates_per_ir", "generation"),
            max_new_tokens=_int(generation_data, "max_new_tokens", "generation"),
            timeout_seconds=_int(generation_data, "timeout_seconds", "generation"),
        )
        semantic = SemanticScreeningPolicy(
            artifact_id=str(semantic_data["artifact_id"]),
            revision=str(semantic_data["revision"]),
            benchmark_manual_review_threshold=_float(
                semantic_data, "benchmark_manual_review_threshold", "semantic_screening"
            ),
            benchmark_rejection_threshold=_float(
                semantic_data, "benchmark_rejection_threshold", "semantic_screening"
            ),
            internal_diversity_mode=InternalDiversityMode(
                str(semantic_data["internal_diversity_mode"])
            ),
            internal_semantic_threshold=None
            if semantic_data["internal_semantic_threshold"] is None
            else _float(semantic_data, "internal_semantic_threshold", "semantic_screening"),
        )
        smoke = FutureSmokePolicy(
            ir_attempts=_int(smoke_data, "ir_attempts", "future_smoke"),
            targeted_allocations=_allocation(
                smoke_data["targeted_allocations"], "future_smoke.targeted_allocations"
            ),
            generic_allocations=_allocation(
                smoke_data["generic_allocations"], "future_smoke.generic_allocations"
            ),
            output_contract_per_group=_int(smoke_data, "output_contract_per_group", "future_smoke"),
            minimum_clean_accepts=_int(smoke_data, "minimum_clean_accepts", "future_smoke"),
            minimum_accepts_per_family=_int(
                smoke_data, "minimum_accepts_per_family", "future_smoke"
            ),
            maximum_false_labels=_int(smoke_data, "maximum_false_labels", "future_smoke"),
            maximum_semantic_drift_accepts=_int(
                smoke_data, "maximum_semantic_drift_accepts", "future_smoke"
            ),
            maximum_invalid_accepts=_int(smoke_data, "maximum_invalid_accepts", "future_smoke"),
            maximum_unresolved_contamination=_int(
                smoke_data, "maximum_unresolved_contamination", "future_smoke"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise DesignConfigError(str(exc)) from exc
    return LocalRealizationDesign(
        schema_version=_int(root, "schema_version", "root"),
        design_only=True,
        system_prompt_sha256=str(root["system_prompt_sha256"]),
        response_schema_version=_int(root, "response_schema_version", "root"),
        primary_model=_model(root["primary_model"], "primary_model"),
        fallback_model=_model(root["fallback_model"], "fallback_model"),
        generation=generation,
        semantic_screening=semantic,
        future_smoke=smoke,
    )
