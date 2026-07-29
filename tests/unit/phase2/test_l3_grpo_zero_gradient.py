from __future__ import annotations

from typing import Any

import pytest

from foundry.phase2.l3_grpo_zero_gradient import (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    INVALID_OR_AMBIGUOUS,
    NONZERO_GRADIENT_UPDATE,
    UNEXPECTED_ZERO_GRADIENT,
    classification_contract,
    classify_group,
    complete_smoke_gate,
    objective_components,
    reward_projection,
    run_deterministic_fixtures,
)
from foundry.training.config import canonical_sha256

torch = pytest.importorskip("torch")


def _projection(*, zero: bool, connected: bool = True) -> dict[str, object]:
    return {
        "finite": True,
        "exactly_zero": zero,
        "graph_connected": connected,
        "nonzero_gradient_count": 0 if zero else 1,
    }


def _observation(
    *,
    rewards: list[float],
    variance: float,
    advantages: list[float],
    policy_zero: bool,
    kl_zero: bool,
    combined_zero: bool,
) -> dict[str, object]:
    return {
        "rewards": rewards,
        "reward_variance": variance,
        "advantages": advantages,
        "valid_completion_token_counts": [2, 2, 2, 2],
        "policy_logprobs_finite": True,
        "reference_logprobs_finite": True,
        "kl_finite": True,
        "adapters_identical_at_step_start": True,
        "controlled_live_policy_fixture_passed": True,
        "requires_grad_policy_tensor_count": 1,
        "optimizer_owned_tensor_count": 1,
        "base_gradient_count": 0,
        "reference_gradient_count": 0,
        "policy_gradient": _projection(zero=policy_zero),
        "kl_gradient": _projection(zero=kl_zero),
        "combined_gradient": _projection(zero=combined_zero),
    }


def test_all_fifteen_deterministic_fixtures_pass_and_replay_exactly() -> None:
    first = run_deterministic_fixtures(torch)
    second = run_deterministic_fixtures(torch)
    assert first == second
    assert first["fixture_count"] == 15
    assert len(first["fixtures"]) == 15
    assert all(row["passed"] is True for row in first["fixtures"])
    fixture_sha256 = first["fixture_sha256"]
    projected = dict(first)
    projected.pop("fixture_sha256")
    assert fixture_sha256 == canonical_sha256(projected)


def test_classification_contract_reconstructs() -> None:
    contract = classification_contract()
    contract_sha256 = contract["classification_contract_sha256"]
    projected = dict(contract)
    projected.pop("classification_contract_sha256")
    assert contract_sha256 == canonical_sha256(projected)
    assert contract["classifications"] == [
        EXPECTED_ZERO_ADVANTAGE_NOOP,
        UNEXPECTED_ZERO_GRADIENT,
        NONZERO_GRADIENT_UPDATE,
        INVALID_OR_AMBIGUOUS,
    ]
    assert contract["classification_uses_total_norm_only"] is False


def test_equal_rewards_are_an_expected_connected_noop() -> None:
    rewards = reward_projection(torch, [1.15, 1.15, 1.15, 1.15])
    evidence = _observation(
        rewards=rewards["rewards"],
        variance=rewards["reward_variance"],
        advantages=rewards["advantages"],
        policy_zero=True,
        kl_zero=True,
        combined_zero=True,
    )
    assert classify_group(evidence) == EXPECTED_ZERO_ADVANTAGE_NOOP


def test_nonzero_advantages_with_zero_connected_gradient_are_unexpected() -> None:
    evidence = _observation(
        rewards=[0.0, 1.0, 0.0, 1.0],
        variance=0.25,
        advantages=[-0.5, 0.5, -0.5, 0.5],
        policy_zero=True,
        kl_zero=True,
        combined_zero=True,
    )
    assert classify_group(evidence) == UNEXPECTED_ZERO_GRADIENT


def test_nonzero_advantages_with_live_gradient_are_an_update() -> None:
    evidence = _observation(
        rewards=[0.0, 1.0, 0.0, 1.0],
        variance=0.25,
        advantages=[-0.5, 0.5, -0.5, 0.5],
        policy_zero=False,
        kl_zero=True,
        combined_zero=False,
    )
    assert classify_group(evidence) == NONZERO_GRADIENT_UPDATE


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("rewards", [0.0, float("nan"), 0.0, 0.0]),
        ("advantages", [0.0, float("inf"), 0.0, 0.0]),
        ("valid_completion_token_counts", [2, 0, 2, 2]),
        ("policy_logprobs_finite", False),
        ("reference_logprobs_finite", False),
        ("kl_finite", False),
        ("reference_gradient_count", 1),
        ("base_gradient_count", 1),
        ("optimizer_owned_tensor_count", 2),
    ),
)
def test_invalid_or_ambiguous_inputs_are_rejected(field: str, value: Any) -> None:
    evidence = _observation(
        rewards=[1.0] * 4,
        variance=0.0,
        advantages=[0.0] * 4,
        policy_zero=True,
        kl_zero=True,
        combined_zero=True,
    )
    evidence[field] = value
    assert classify_group(evidence) == INVALID_OR_AMBIGUOUS


def test_detached_policy_logprobs_are_rejected() -> None:
    with pytest.raises(ValueError, match="detached"):
        objective_components(
            torch,
            policy_logprobs=torch.zeros((4, 2)),
            reference_logprobs=torch.zeros((4, 2)),
            advantages=torch.zeros(4),
            completion_mask=torch.ones((4, 2)),
        )


def test_complete_smoke_requires_both_accounting_and_a_real_update() -> None:
    noop = {
        "classification": EXPECTED_ZERO_ADVANTAGE_NOOP,
        "completion_count": 4,
        "reward_variance": 0.0,
        "policy_parameter_changed": False,
        "optimizer_step_completed": True,
        "scheduler_step_completed": True,
        "reference_parameter_changed": False,
        "base_parameter_changed": False,
    }
    assert complete_smoke_gate((noop,))["passed"] is False
    update = {
        "classification": NONZERO_GRADIENT_UPDATE,
        "completion_count": 4,
        "reward_variance": 0.25,
        "policy_parameter_changed": True,
        "optimizer_step_completed": True,
        "scheduler_step_completed": True,
        "reference_parameter_changed": False,
        "base_parameter_changed": False,
    }
    assert complete_smoke_gate((noop, update))["passed"] is True
