import pytest

from foundry.phase2.retention_failure_interpretation import (
    failure_rule,
    freeze,
    normalize_text,
    select_architecture,
    structural_signature,
)


def test_text_normalization_and_structure_are_deterministic() -> None:
    assert normalize_text("  Final   ANSWER: 42 ") == "final answer: 42"
    assert structural_signature("Add 17 to 25.") == structural_signature("Add 9 to 33.")
    assert structural_signature("Add 17 to 25.") != structural_signature("Divide 25 by 5.")


def test_failure_rule_distinguishes_numeric_and_format_failures() -> None:
    assert failure_rule("arithmetic", {"extractable": True}) == (
        "numeric_terminal_expected_answer_equality",
        "wrong_extractable_numeric_answer",
    )
    assert failure_rule("arithmetic", {"extractable": False}) == (
        "numeric_terminal_extraction_required",
        "malformed_nonextractable_numeric_answer",
    )
    assert failure_rule("format", {"extractable": True}) == (
        "exact_text_equality",
        "exact_format_mismatch",
    )


def test_freeze_is_tamper_sensitive_and_rejects_existing_hash() -> None:
    first = freeze({"value": 1}, "evidence_sha256")
    second = freeze({"value": 2}, "evidence_sha256")
    assert first["evidence_sha256"] != second["evidence_sha256"]
    with pytest.raises(ValueError, match="already contains"):
        freeze(first, "evidence_sha256")


def test_predeclared_hierarchy_selects_kl_only_when_item_one_holds() -> None:
    assert (
        select_architecture(
            shared_failure_fraction=1.0,
            replay_ratio_trajectories_aligned=True,
            explicit_logit_constraint=False,
            capacity_implicated=False,
            gradient_conflict_measured=False,
        )
        == "replay-ce-token-kl-v1"
    )
    assert (
        select_architecture(
            shared_failure_fraction=0.5,
            replay_ratio_trajectories_aligned=True,
            explicit_logit_constraint=False,
            capacity_implicated=True,
            gradient_conflict_measured=False,
        )
        == "layer-restricted-lora-v1"
    )
    assert (
        select_architecture(
            shared_failure_fraction=0.5,
            replay_ratio_trajectories_aligned=False,
            explicit_logit_constraint=False,
            capacity_implicated=False,
            gradient_conflict_measured=True,
        )
        == "multiobjective-gradient-balanced-sft-v1"
    )
    assert (
        select_architecture(
            shared_failure_fraction=0.5,
            replay_ratio_trajectories_aligned=False,
            explicit_logit_constraint=False,
            capacity_implicated=False,
            gradient_conflict_measured=False,
        )
        is None
    )
