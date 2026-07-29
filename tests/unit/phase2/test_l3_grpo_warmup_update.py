from __future__ import annotations

import pytest

from foundry.phase2.l3_grpo_warmup_update import (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    EXPECTED_ZERO_LR_WARMUP_NOOP,
    INVALID_OR_AMBIGUOUS,
    NONZERO_POLICY_UPDATE,
    UNEXPECTED_POSITIVE_LR_NO_UPDATE,
    UNEXPECTED_ZERO_GRADIENT,
    changed_state_tensor_count,
    classify_update,
    complete_warmup_smoke_gate,
    counted_update_gate,
    run_deterministic_fixtures,
    update_contract,
)


def _evidence(
    *,
    variance: float = 0.25,
    advantages: list[float] | None = None,
    learning_rate: float = 0.0,
    gradient_norm: float = 1.0,
    nonzero_gradients: int = 1,
    delta_norm: float = 0.0,
    changed_policy: int = 0,
) -> dict[str, object]:
    return {
        "reward_variance": variance,
        "advantages": advantages or [-0.5, 0.5, -0.5, 0.5],
        "effective_learning_rates": [learning_rate],
        "policy_gradient_norm": gradient_norm,
        "nonzero_policy_gradient_tensor_count": nonzero_gradients,
        "policy_gradient_finite": True,
        "policy_delta_norm": delta_norm,
        "changed_policy_tensor_count": changed_policy,
        "policy_delta_finite": True,
        "changed_optimizer_state_tensor_count": 1,
        "reference_parameter_changed": False,
        "base_parameter_changed": False,
        "evidence_complete": True,
    }


def test_six_way_update_classification() -> None:
    warmup = _evidence()
    assert classify_update(warmup) == EXPECTED_ZERO_LR_WARMUP_NOOP

    update = _evidence(learning_rate=1.0e-6, delta_norm=1.0e-7, changed_policy=4)
    assert classify_update(update) == NONZERO_POLICY_UPDATE

    positive_no_update = _evidence(learning_rate=1.0e-6)
    assert classify_update(positive_no_update) == UNEXPECTED_POSITIVE_LR_NO_UPDATE

    zero_gradient = _evidence(gradient_norm=0.0, nonzero_gradients=0)
    assert classify_update(zero_gradient) == UNEXPECTED_ZERO_GRADIENT

    zero_advantage = _evidence(
        variance=0.0,
        advantages=[0.0, 0.0, 0.0, 0.0],
        learning_rate=1.0e-6,
        gradient_norm=0.0,
        nonzero_gradients=0,
    )
    assert classify_update(zero_advantage) == EXPECTED_ZERO_ADVANTAGE_NOOP

    invalid = dict(warmup)
    invalid["reference_parameter_changed"] = True
    assert classify_update(invalid) == INVALID_OR_AMBIGUOUS


def test_zero_lr_does_not_hide_zero_gradient() -> None:
    evidence = _evidence(gradient_norm=0.0, nonzero_gradients=0)
    assert classify_update(evidence) == UNEXPECTED_ZERO_GRADIENT


def test_complete_smoke_requires_positive_lr_policy_update() -> None:
    warmup = {
        **_evidence(),
        "classification": EXPECTED_ZERO_LR_WARMUP_NOOP,
        "completion_count": 4,
        "optimizer_call_completed": True,
        "scheduler_step_completed": True,
    }
    update = {
        **_evidence(learning_rate=1.0e-6, delta_norm=1.0e-7, changed_policy=4),
        "classification": NONZERO_POLICY_UPDATE,
        "completion_count": 4,
        "optimizer_call_completed": True,
        "scheduler_step_completed": True,
    }
    passed = complete_warmup_smoke_gate((warmup, update))
    assert passed["passed"] is True
    failed = complete_warmup_smoke_gate((warmup, {**update, "classification": update_contract()}))
    assert failed["passed"] is False


def test_counted_gate_accepts_noops_but_requires_an_update() -> None:
    warmup = {
        **_evidence(),
        "classification": EXPECTED_ZERO_LR_WARMUP_NOOP,
        "optimizer_call_completed": True,
        "scheduler_step_completed": True,
    }
    update = {
        **_evidence(learning_rate=1.0e-6, delta_norm=1.0e-7, changed_policy=1),
        "classification": NONZERO_POLICY_UPDATE,
        "optimizer_call_completed": True,
        "scheduler_step_completed": True,
    }
    zero = {
        **_evidence(
            variance=0.0,
            advantages=[0.0, 0.0, 0.0, 0.0],
            learning_rate=1.0e-6,
            gradient_norm=0.0,
            nonzero_gradients=0,
        ),
        "classification": EXPECTED_ZERO_ADVANTAGE_NOOP,
        "optimizer_call_completed": True,
        "scheduler_step_completed": True,
    }
    result = counted_update_gate(
        (warmup, update, zero),
        expected_steps=3,
        expected_learning_rates=(0.0, 1.0e-6, 1.0e-6),
    )
    assert result["passed"] is True
    no_update = counted_update_gate(
        (warmup, zero),
        expected_steps=2,
        expected_learning_rates=(0.0, 1.0e-6),
    )
    assert no_update["passed"] is False


def test_changed_optimizer_state_tensor_count() -> None:
    before = {
        "tensor_evidence": [
            {"path": "$/state/0/step", "sha256": "a" * 64},
            {"path": "$/state/0/exp_avg", "sha256": "b" * 64},
        ]
    }
    after = {
        "tensor_evidence": [
            {"path": "$/state/0/step", "sha256": "c" * 64},
            {"path": "$/state/0/exp_avg", "sha256": "b" * 64},
            {"path": "$/state/0/exp_avg_sq", "sha256": "d" * 64},
        ]
    }
    assert changed_state_tensor_count(before, after) == 2


def test_all_fourteen_model_free_fixtures_pass() -> None:
    torch = pytest.importorskip("torch")
    result = run_deterministic_fixtures(torch)
    assert result["fixture_count"] == 14
    assert all(row["passed"] for row in result["fixtures"])


def test_update_contract_freezes_exact_six_classes() -> None:
    result = update_contract()
    assert result["contract_id"] == "foundry-grpo-warmup-aware-update-v1"
    assert result["classifications"] == [
        EXPECTED_ZERO_ADVANTAGE_NOOP,
        EXPECTED_ZERO_LR_WARMUP_NOOP,
        NONZERO_POLICY_UPDATE,
        UNEXPECTED_ZERO_GRADIENT,
        UNEXPECTED_POSITIVE_LR_NO_UPDATE,
        INVALID_OR_AMBIGUOUS,
    ]
    assert result["scientific_recipe_changed"] is False
