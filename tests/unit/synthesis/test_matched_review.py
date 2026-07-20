"""Focused blind-language and packet-identity checks."""

from __future__ import annotations

from foundry.synthesis.template_bank.matched_review import _blind_audit


def test_blind_audit_approves_clear_original_question() -> None:
    audit = _blind_audit(
        "original-1",
        "A workshop has 12 sealed cases, and each case holds 4 filters. "
        "How many filters are stored altogether?",
    )

    assert audit.recommendation == "approve"
    assert audit.confidence == "high"
    assert not audit.defect_labels


def test_blind_audit_rejects_repetition_internal_terms_and_missing_question() -> None:
    audit = _blind_audit(
        "original-2",
        "The crate crate follows frame_id and contains 1th sample.",
    )

    assert audit.recommendation == "reject"
    assert not audit.naturalness_pass
    assert not audit.grammar_pass
    assert not audit.target_clarity_pass
    assert not audit.self_contained_pass
    assert set(audit.defect_labels) == {
        "repeated wording",
        "grammar",
        "missing information",
        "internal terminology",
    }
