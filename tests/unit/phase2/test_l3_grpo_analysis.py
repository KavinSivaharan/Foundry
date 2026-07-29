from __future__ import annotations

from foundry.phase2.l3_grpo_analysis import (
    MAXIMUM_UNTARGETED_DECLINE,
    MINIMUM_TARGETED_EXTRACTABILITY,
    _category_effects,
    _paired,
)


def test_paired_transition_orientation_reports_net_right_advantage() -> None:
    left = {"a": True, "b": False, "c": True, "d": False}
    right = {"a": True, "b": True, "c": False, "d": False}
    result = _paired(left, right)
    assert result == {
        "left_wins": 1,
        "right_wins": 1,
        "net_right_advantage": 0,
        "both_correct": 1,
        "both_wrong": 1,
    }


def test_untargeted_gate_is_relative_to_starting_targeted() -> None:
    taxonomy = {
        "a": "multi_step_bookkeeping_or_omission",
        "b": "arithmetic_execution",
        "c": "output_format_or_answer_extraction",
        "d": "rate_ratio_percentage_or_average",
    }
    models = {
        "base": {key: False for key in taxonomy},
        "starting_generic": {key: False for key in taxonomy},
        "starting_targeted": {"a": True, "b": True, "c": False, "d": True},
        "grpo_generic": {key: False for key in taxonomy},
        "grpo_targeted": {"a": True, "b": True, "c": False, "d": True},
    }
    categories, untargeted = _category_effects(taxonomy, models)
    assert set(categories) == set(taxonomy.values())
    assert untargeted["examples"] == 2
    assert untargeted["accuracy_change_vs_starting_targeted"] == 0.0
    assert untargeted["gate_passed"] is True
    assert MAXIMUM_UNTARGETED_DECLINE == 0.02
    assert MINIMUM_TARGETED_EXTRACTABILITY == 0.9138
