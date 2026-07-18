from foundry.synthesis.contracts import (
    ARCHITECTURE_OPTIONS,
    PILOT_SIZE,
    SUCCESS_GATES,
    synthesis_contract_sha256,
)
from foundry.synthesis.taxonomy import (
    CATEGORY_DEFINITIONS,
    OUTPUT_CONTRACT_TRACK_ID,
    SELECTED_REASONING_CATEGORIES,
    FailureCategory,
    taxonomy_contract_sha256,
)


def test_complete_taxonomy_covers_all_development_failures() -> None:
    assert sum(item.observed_count for item in CATEGORY_DEFINITIONS) == 293
    assert {item.category for item in CATEGORY_DEFINITIONS} == set(FailureCategory)


def test_exactly_three_reasoning_categories_and_separate_output_track_are_frozen() -> None:
    assert SELECTED_REASONING_CATEGORIES == (
        FailureCategory.MULTI_STEP_BOOKKEEPING,
        FailureCategory.RATE_RATIO_PERCENTAGE,
        FailureCategory.CONSTRAINT_DISCRETE,
    )
    assert OUTPUT_CONTRACT_TRACK_ID == "terminal-final-answer-contract-v1"


def test_first_pilot_selects_procedural_architecture() -> None:
    selected = [
        option.option_id for option in ARCHITECTURE_OPTIONS if option.pilot_decision == "selected"
    ]

    assert selected == ["A"]


def test_targeted_and_generic_controls_have_matched_budgets() -> None:
    assert PILOT_SIZE.targeted == PILOT_SIZE.generic_control
    assert PILOT_SIZE.targeted.accepted_examples == 4000
    assert PILOT_SIZE.generator_smoke_examples == 120


def test_success_gates_cover_generation_and_later_training() -> None:
    gate_ids = {gate.gate_id for gate in SUCCESS_GATES}

    assert {"dual-verification", "contamination", "deduplication"} <= gate_ids
    assert {"one-seed-control", "two-seed-overall", "category-gain"} <= gate_ids


def test_content_free_contract_hashes_are_frozen() -> None:
    assert (
        taxonomy_contract_sha256()
        == "021837a1f1a3bb5a189b1f39c808bb907e415e28d8fa722a8a03c3114717cf28"
    )
    assert (
        synthesis_contract_sha256()
        == "910bf21dba7cef833fd9f7bd83842034e9e7261cf93979d7cdddc0479094d347"
    )
