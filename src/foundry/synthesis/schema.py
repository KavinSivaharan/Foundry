"""Typed, design-only schema for future procedurally generated examples."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DifficultyLevel(StrEnum):
    """Predeclared generator difficulty bands."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ValidityStatus(StrEnum):
    """Lifecycle status for one generated candidate."""

    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ReviewStatus(StrEnum):
    """Status for deduplication and contamination gates."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class ExactValue:
    """An exact rational label with no floating-point representation."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("exact-value denominator must be positive")

    @property
    def fraction(self) -> Fraction:
        """Return the normalized exact value."""

        return Fraction(self.numerator, self.denominator)

    def render(self) -> str:
        """Render a stable canonical integer or fraction."""

        value = self.fraction
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class ProgramParameter:
    """One named exact input sampled before rendering."""

    name: str
    value: ExactValue
    unit: str | None = None


@dataclass(frozen=True)
class ProgramStep:
    """One auditable operation in a latent arithmetic program."""

    output_symbol: str
    operation: str
    input_symbols: tuple[str, ...]
    exact_result: ExactValue


@dataclass(frozen=True)
class LatentProgramSpec:
    """Structured source of truth from which question and label are derived."""

    program_family: str
    parameters: tuple[ProgramParameter, ...]
    steps: tuple[ProgramStep, ...]
    constraints: tuple[str, ...]
    answer_symbol: str

    def __post_init__(self) -> None:
        symbols = {parameter.name for parameter in self.parameters}
        for step in self.steps:
            if step.output_symbol in symbols:
                raise ValueError("latent-program symbols must be unique")
            if not set(step.input_symbols) <= symbols:
                raise ValueError("latent-program step refers to an unknown symbol")
            symbols.add(step.output_symbol)
        if self.answer_symbol not in symbols:
            raise ValueError("latent-program answer symbol is undefined")


@dataclass(frozen=True)
class VerificationEvidence:
    """Content-free evidence emitted by one exact verifier."""

    verifier_id: str
    verifier_version: str
    method_family: str
    computed_answer: ExactValue
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.evidence_sha256):
            raise ValueError("verification evidence requires a SHA-256 digest")


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Provenance that forbids benchmark-derived generation."""

    source_kind: str
    generator_config_sha256: str
    taxonomy_sha256: str
    benchmark_content_used_as_generator_input: bool

    def __post_init__(self) -> None:
        if self.source_kind != "independent_procedural":
            raise ValueError("the pilot accepts only independent procedural provenance")
        if self.benchmark_content_used_as_generator_input:
            raise ValueError("benchmark content cannot be generator input")
        for digest in (self.generator_config_sha256, self.taxonomy_sha256):
            if not _SHA256.fullmatch(digest):
                raise ValueError("provenance hashes must be SHA-256 digests")


@dataclass(frozen=True)
class SyntheticExample:
    """Complete schema for a future generated training example."""

    synthetic_example_id: str
    generator_version: str
    random_seed: int
    target_failure_category: str
    secondary_skill_tags: tuple[str, ...]
    difficulty_level: DifficultyLevel
    latent_program: LatentProgramSpec
    rendered_question: str
    deterministic_solution_trace: tuple[str, ...]
    canonical_final_answer: ExactValue
    required_final_answer_format: str
    primary_verification_evidence: VerificationEvidence
    independent_verification_evidence: VerificationEvidence
    validity_status: ValidityStatus
    rejection_reason: str | None
    normalized_text_hash: str
    latent_program_hash: str
    provenance: ProvenanceMetadata
    contamination_check_status: ReviewStatus
    deduplication_status: ReviewStatus

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random seed must be non-negative")
        if not self.rendered_question.strip() or not self.deterministic_solution_trace:
            raise ValueError("question and deterministic solution trace are required")
        if self.required_final_answer_format != "Final answer: <canonical-number>":
            raise ValueError("the pilot final-answer contract is immutable")
        if self.primary_verification_evidence.verifier_id == (
            self.independent_verification_evidence.verifier_id
        ):
            raise ValueError("primary and independent verifiers must be distinct")
        if self.primary_verification_evidence.method_family == (
            self.independent_verification_evidence.method_family
        ):
            raise ValueError("independent verification must use a different method family")
        expected = self.canonical_final_answer.fraction
        if self.primary_verification_evidence.computed_answer.fraction != expected:
            raise ValueError("primary verifier disagrees with the canonical answer")
        if self.independent_verification_evidence.computed_answer.fraction != expected:
            raise ValueError("independent verifier disagrees with the canonical answer")
        if self.validity_status is ValidityStatus.ACCEPTED and self.rejection_reason is not None:
            raise ValueError("accepted examples cannot have a rejection reason")
        if self.validity_status is ValidityStatus.REJECTED and not self.rejection_reason:
            raise ValueError("rejected examples require a rejection reason")
        for digest in (self.normalized_text_hash, self.latent_program_hash):
            if not _SHA256.fullmatch(digest):
                raise ValueError("example hashes must be SHA-256 digests")
        if (
            math.gcd(
                self.canonical_final_answer.numerator,
                self.canonical_final_answer.denominator,
            )
            != 1
        ):
            raise ValueError("canonical final answers must be normalized")
