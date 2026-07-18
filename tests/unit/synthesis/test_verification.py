from foundry.synthesis.schema import ExactValue, VerificationEvidence
from foundry.synthesis.verification import (
    VerificationStatus,
    combine_independent_evidence,
    validate_final_answer_contract,
)

_HASH = "c" * 64


def _evidence(verifier: str, method: str, answer: ExactValue) -> VerificationEvidence:
    return VerificationEvidence(verifier, "1", method, answer, _HASH)


def test_independent_exact_methods_accept_only_when_they_agree() -> None:
    decision = combine_independent_evidence(
        _evidence("symbolic", "dag", ExactValue(3, 2)),
        _evidence("inverse", "inverse", ExactValue(6, 4)),
    )

    assert decision.status is VerificationStatus.ACCEPT
    assert decision.canonical_answer == ExactValue(3, 2)


def test_disagreement_rejects_without_selecting_an_answer() -> None:
    decision = combine_independent_evidence(
        _evidence("symbolic", "dag", ExactValue(3, 2)),
        _evidence("enumerator", "brute-force", ExactValue(2)),
    )

    assert decision.status is VerificationStatus.REJECT
    assert decision.canonical_answer is None
    assert decision.rejection_reason == "verifier_disagreement"


def test_same_method_family_is_not_independent() -> None:
    decision = combine_independent_evidence(
        _evidence("symbolic-a", "dag", ExactValue(2)),
        _evidence("symbolic-b", "dag", ExactValue(2)),
    )

    assert decision.status is VerificationStatus.REJECT
    assert decision.rejection_reason == "same_method_family"


def test_final_answer_contract_requires_one_exact_terminal_line() -> None:
    assert validate_final_answer_contract("Reasoning.\nFinal answer: -3/2", ExactValue(-3, 2))
    assert not validate_final_answer_contract(
        "Final answer: -3/2\nAdditional text", ExactValue(-3, 2)
    )
    assert not validate_final_answer_contract(
        "Final answer: -3/2\nFinal answer: -3/2", ExactValue(-3, 2)
    )
