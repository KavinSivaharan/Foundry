"""Deterministic procedural generator contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction

from foundry.synthesis.schema import DifficultyLevel, ExactValue, LatentProgramSpec


@dataclass(frozen=True)
class CandidateDraft:
    """Content-bearing candidate before the ordered acceptance pipeline."""

    candidate_id: str
    generator_id: str
    generator_version: str
    random_seed: int
    target_failure_category: str
    secondary_skill_tags: tuple[str, ...]
    difficulty_level: DifficultyLevel
    output_contract_enabled: bool
    latent_program: LatentProgramSpec
    rendered_question: str
    deterministic_solution_trace: tuple[str, ...]
    canonical_final_answer: ExactValue
    training_completion: str
    structure_signature: dict[str, object]
    verifier_payload: dict[str, object]
    ambiguity_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.random_seed < 0 or not self.rendered_question.strip():
            raise ValueError("candidate seed and rendered question are required")
        if not self.deterministic_solution_trace or not self.training_completion.strip():
            raise ValueError("candidate solution trace and completion are required")
        expected_line = f"Final answer: {self.canonical_final_answer.render()}"
        if self.output_contract_enabled:
            if self.training_completion.strip().splitlines()[-1] != expected_line:
                raise ValueError("output-contract completion lacks its canonical terminal line")
        elif any(
            line.startswith("Final answer:") for line in self.training_completion.splitlines()
        ):
            raise ValueError("non-output-track completion cannot contain the terminal contract")

    @property
    def structure_sha256(self) -> str:
        """Hash the value-neutral latent structure and rendering family."""

        rendered = json.dumps(self.structure_signature, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GeneratorVerification:
    """Exact result from one generator-specific verifier."""

    verifier_id: str
    verifier_version: str
    method_family: str
    success: bool
    answer: ExactValue | None
    failure_reason: str | None
    evidence_sha256: str


def exact_value(value: Fraction | int) -> ExactValue:
    """Convert exact arithmetic into the normalized schema representation."""

    normalized = Fraction(value)
    return ExactValue(normalized.numerator, normalized.denominator)


def candidate_id(generator_id: str, seed: int, variant: int) -> str:
    """Create a stable ID without benchmark-derived material."""

    digest = hashlib.sha256(f"{generator_id}:{seed}:{variant}".encode()).hexdigest()
    return f"syn-{digest[:24]}"


def training_completion(
    trace: tuple[str, ...],
    answer: ExactValue,
    *,
    output_contract_enabled: bool,
) -> str:
    """Render deterministic training output, with the contract only on its shared track."""

    lines = list(trace)
    if output_contract_enabled:
        lines.append(f"Final answer: {answer.render()}")
    return "\n".join(lines)


def verification_result(
    *,
    verifier_id: str,
    method_family: str,
    answer: Fraction | None,
    failure_reason: str | None,
    evidence_payload: dict[str, object],
) -> GeneratorVerification:
    """Build stable content-free evidence for a verifier result."""

    success = answer is not None and failure_reason is None
    payload = {
        "verifier_id": verifier_id,
        "verifier_version": "1",
        "method_family": method_family,
        "success": success,
        "answer": None if answer is None else [answer.numerator, answer.denominator],
        "failure_reason": failure_reason,
        "evidence": evidence_payload,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return GeneratorVerification(
        verifier_id=verifier_id,
        verifier_version="1",
        method_family=method_family,
        success=success,
        answer=None if answer is None else exact_value(answer),
        failure_reason=failure_reason,
        evidence_sha256=digest,
    )


__all__ = ["CandidateDraft", "GeneratorVerification"]
