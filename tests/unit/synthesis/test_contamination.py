import pytest

from foundry.synthesis.contamination import (
    ContaminationOutcome,
    assess_pair,
    canonical_number_neutral_identity,
    number_neutral_identity_contract_sha256,
    numeric_template_sha256,
    require_number_neutral_identity,
    token_ngram_jaccard,
)


def _assess(
    left: str,
    right: str,
    *,
    generated_structure: str = "a" * 64,
    comparison_structure: str | None = "b" * 64,
    semantic_similarity: float | None = 0.1,
):
    return assess_pair(
        left,
        right,
        generated_structure_sha256=generated_structure,
        comparison_structure_sha256=comparison_structure,
        semantic_similarity=semantic_similarity,
    )


def test_obvious_surface_copy_is_rejected() -> None:
    decision = _assess(
        "A lunar depot stores 14 cobalt tiles.",
        "A LUNAR depot stores 14 cobalt tiles!",
    )

    assert decision.outcome is ContaminationOutcome.REJECT
    assert decision.reason == "exact_normalized_text"


def test_number_swap_is_rejected_as_same_template() -> None:
    left = "A lunar depot stores 14 cobalt tiles."
    right = "A lunar depot stores 83 cobalt tiles."

    assert numeric_template_sha256(left) == numeric_template_sha256(right)
    assert _assess(left, right).reason == "numeric_template_copy"


def test_structural_copy_is_rejected_despite_different_wording() -> None:
    decision = _assess(
        "Robots place parts into balanced trays.",
        "Crates receive an equal allocation of components.",
        generated_structure="c" * 64,
        comparison_structure="c" * 64,
    )

    assert decision.outcome is ContaminationOutcome.REJECT
    assert decision.reason == "latent_structure_copy"


def test_high_semantic_similarity_is_rejected() -> None:
    decision = _assess(
        "Orbital beacons are checked after several cycles.",
        "Engineers inspect remote markers following repeated intervals.",
        semantic_similarity=0.9,
    )

    assert decision.outcome is ContaminationOutcome.REJECT
    assert decision.reason == "semantic_similarity"


def test_missing_semantic_screen_cannot_auto_accept() -> None:
    decision = _assess(
        "A ceramic studio tracks kiln batches.",
        "A weather station averages pressure readings.",
        semantic_similarity=None,
    )

    assert decision.outcome is ContaminationOutcome.MANUAL_REVIEW
    assert decision.reason == "semantic_check_not_run"


def test_dissimilar_pair_passes_all_frozen_pairwise_checks() -> None:
    decision = _assess(
        "A ceramic studio tracks kiln batches.",
        "A weather station averages pressure readings.",
        semantic_similarity=0.1,
    )

    assert decision.outcome is ContaminationOutcome.PASS
    assert token_ngram_jaccard("one two three", "seven eight nine", size=2) == 0.0


def test_canonical_number_neutral_identity_preserves_frozen_semantics() -> None:
    identity = canonical_number_neutral_identity("A shelf holds 12 boxes!")

    assert identity.normalized_text == "a shelf holds <num> boxes"
    assert identity.sha256 == numeric_template_sha256("A shelf holds 12 boxes!")
    assert len(number_neutral_identity_contract_sha256()) == 64


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ("A drone plan permits 4 kits.", "A drone plan permits 19 kits!"),
        ("The weighted mean uses 3 groups.", "The weighted mean uses 8 groups."),
        ("A textile ledger records 6 spools.", "A textile ledger records 42 spools."),
        ("Count 2 valid reef allocations.", "Count 11 valid reef allocations."),
        ("A robot schedule has 5 arrangements.", "A robot schedule has 23 arrangements."),
    ),
)
def test_metadata_cannot_make_identical_runtime_surfaces_unique(left: str, right: str) -> None:
    assert canonical_number_neutral_identity(left).sha256 == (
        canonical_number_neutral_identity(right).sha256
    )


def test_distinct_normalized_surfaces_remain_distinct() -> None:
    assert canonical_number_neutral_identity("Count 4 valid allocations.").sha256 != (
        canonical_number_neutral_identity("Find the average of 4 readings.").sha256
    )


def test_schedule_runtime_identity_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="schedule/runtime"):
        require_number_neutral_identity("Count 4 valid allocations.", "0" * 64)
