"""Independent-verifier contracts; no synthetic question generator is implemented here."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from foundry.synthesis.schema import ExactValue, VerificationEvidence

_FINAL_LINE = re.compile(r"^Final answer: (?P<answer>[+-]?(?:0|[1-9]\d*)(?:/[1-9]\d*)?)$")


class VerificationStatus(StrEnum):
    """Combined dual-verification outcome."""

    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class VerificationDecision:
    """Result of comparing two logically independent verifiers."""

    status: VerificationStatus
    canonical_answer: ExactValue | None
    rejection_reason: str | None


@dataclass(frozen=True)
class CategoryVerifierContract:
    """Frozen primary/independent methods for one curriculum category."""

    category: str
    primary_method: str
    independent_method: str
    disagreement_behavior: str
    timeout_behavior: str
    numeric_normalization: str
    forced_rejection_conditions: tuple[str, ...]


VERIFIER_CONTRACTS: tuple[CategoryVerifierContract, ...] = (
    CategoryVerifierContract(
        "multi_step_bookkeeping_or_omission",
        "Topological Fraction execution of the latent arithmetic DAG.",
        "Independent step-ledger replay with state conservation and inverse checks.",
        "Reject; never select one answer.",
        "Reject on either timeout.",
        "Normalize integers, Decimal strings, and Fraction values to reduced Fraction.",
        ("undefined symbol", "cycle", "division by zero", "unused required branch"),
    ),
    CategoryVerifierContract(
        "rate_ratio_percentage_or_average",
        "Exact Fraction equation evaluation with explicit numerator and denominator.",
        "Cross-multiplication or inverse-substitution dimensional check.",
        "Reject; never average disagreeing results.",
        "Reject on either timeout.",
        "Use reduced Fraction; render Decimal only from an exact terminating rational.",
        ("unit mismatch", "zero denominator", "implicit percent base", "rounding required"),
    ),
    CategoryVerifierContract(
        "constraint_distribution_or_discrete_reasoning",
        "Constructive exact solver for the sampled bounded constraints.",
        "Independent brute-force enumeration over the frozen finite domain.",
        "Reject; never use an LLM judge as a tie-breaker.",
        "Reject if enumeration exceeds the configured state or time bound.",
        "Require integral normalized results and explicit floor/ceiling semantics.",
        ("multiple solutions", "no solution", "out-of-domain value", "ambiguous rounding"),
    ),
    CategoryVerifierContract(
        "terminal-final-answer-contract-v1",
        "Strict final-line parser over the rendered deterministic solution.",
        "Exact string round-trip from the canonical rational answer.",
        "Reject the training example.",
        "Reject on either timeout.",
        "Use `Final answer: <canonical-number>` as the last and only answer line.",
        ("missing line", "multiple answer lines", "trailing text", "answer disagreement"),
    ),
)


def combine_independent_evidence(
    primary: VerificationEvidence,
    independent: VerificationEvidence,
) -> VerificationDecision:
    """Accept only distinct methods that compute the same exact value."""

    if primary.verifier_id == independent.verifier_id:
        return VerificationDecision(VerificationStatus.REJECT, None, "same_verifier_id")
    if primary.method_family == independent.method_family:
        return VerificationDecision(VerificationStatus.REJECT, None, "same_method_family")
    if primary.computed_answer.fraction != independent.computed_answer.fraction:
        return VerificationDecision(VerificationStatus.REJECT, None, "verifier_disagreement")
    return VerificationDecision(VerificationStatus.ACCEPT, primary.computed_answer, None)


def validate_final_answer_contract(response: str, expected: ExactValue) -> bool:
    """Validate the synthesis output contract independently of benchmark scoring."""

    lines = response.strip().splitlines()
    matching = [line for line in lines if line.startswith("Final answer:")]
    if len(matching) != 1 or matching[0] != lines[-1]:
        return False
    match = _FINAL_LINE.fullmatch(lines[-1])
    return match is not None and match.group("answer") == expected.render()
