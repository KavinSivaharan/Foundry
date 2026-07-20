from typing import Any

from foundry.training.retention_gate import (
    build_gate_summary,
    evaluate_arm,
    evaluate_recipe,
)


def _base() -> dict[str, Any]:
    return {
        "summary_sha256": "a" * 64,
        "section_metrics": {
            "arithmetic": {"accuracy": 1.0},
            "format": {"accuracy": 14 / 15},
            "instruction": {"accuracy": 14 / 15},
        },
    }


def _training(tokens: int = 14_404) -> dict[str, Any]:
    return {
        "actual_loss_bearing_tokens": tokens,
        "final_validation_loss": 0.25,
        "losses_all_finite": True,
        "token_count_matches_schedule": True,
    }


def _retention(*, arithmetic: float = 0.9, instruction: float = 0.9) -> dict[str, Any]:
    return {
        "adapter_sha256": "b" * 64,
        "section_metrics": {
            "arithmetic": {"accuracy": arithmetic},
            "format": {"accuracy": 0.9},
            "instruction": {"accuracy": instruction},
        },
        "extractability": 0.9,
        "prompt_echo_rate": 0.05,
        "question_generation": 0,
        "backend_failures": 0,
    }


def test_arm_passes_exact_gate_boundaries() -> None:
    result = evaluate_arm(base=_base(), training=_training(), retention=_retention())
    assert result["passed"] is True
    assert result["failed_checks"] == []


def test_arm_reports_instruction_failure() -> None:
    result = evaluate_arm(
        base=_base(), training=_training(), retention=_retention(instruction=13 / 15)
    )
    assert result["passed"] is False
    assert result["failed_checks"] == ["instruction_following"]


def test_recipe_requires_both_arms_and_token_parity() -> None:
    result = evaluate_recipe(
        recipe_id="fixture",
        learning_rate=5e-5,
        base=_base(),
        generic_training=_training(14_404),
        generic_retention=_retention(),
        targeted_training=_training(14_404),
        targeted_retention=_retention(arithmetic=25 / 30),
    )
    assert result["passed"] is False
    assert result["token_parity"]["passed"] is True
    assert result["targeted"]["failed_checks"] == ["arithmetic_retention"]


def test_gate_summary_stops_when_all_recipes_fail() -> None:
    summary = build_gate_summary(base=_base(), recipes=[{"recipe_id": "x", "passed": False}])
    assert summary["selected_recipe_id"] is None
    assert summary["full_retraining_authorized"] is False
    assert summary["frozen_development_evaluation_authorized"] is False
    assert len(summary["summary_sha256"]) == 64
