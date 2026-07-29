from __future__ import annotations

import math
from typing import Any

import pytest

from foundry.phase2.l3_grpo_advantage_equivalence import (
    ABSOLUTE_TOLERANCE,
    EXPECTED_EXACT_MISMATCH_VECTOR_COUNT,
    EXPECTED_FIXTURE_VECTOR_COUNT,
    EXPECTED_MAXIMUM_FIXTURE_DIFFERENCE,
    RELATIVE_TOLERANCE,
    advantage_equivalence_contract,
    evaluate_advantage_equivalence,
)
from foundry.training.config import canonical_sha256


def _evaluate(
    *,
    rewards: list[float] | None = None,
    canonical: list[float] | None = None,
    duplicate: list[float] | None = None,
    diagnostic: list[float] | None = None,
    reward_hashes_equal: bool = True,
    stock_match: bool = True,
) -> dict[str, object]:
    canonical_values = canonical if canonical is not None else [-0.5, 0.5, -0.5, 0.5]
    return evaluate_advantage_equivalence(
        reward_vector=rewards if rewards is not None else [0.0, 1.0, 0.0, 1.0],
        canonical_cuda_advantages=canonical_values,
        duplicate_cuda_advantages=duplicate if duplicate is not None else canonical_values,
        cpu_diagnostic_advantages=diagnostic if diagnostic is not None else canonical_values,
        canonical_reward_vector_sha256="a" * 64,
        duplicate_reward_vector_sha256=("a" if reward_hashes_equal else "b") * 64,
        canonical_projection_matches_stock=stock_match,
    )


def test_one_float32_epsilon_difference_passes() -> None:
    result = _evaluate(
        canonical=[-0.5, 0.5, -0.5, 0.5],
        diagnostic=[
            -0.5 + EXPECTED_MAXIMUM_FIXTURE_DIFFERENCE,
            0.5,
            -0.5 + EXPECTED_MAXIMUM_FIXTURE_DIFFERENCE,
            0.5,
        ],
    )
    assert result["passed"] is True
    assert result["maximum_absolute_difference"] == EXPECTED_MAXIMUM_FIXTURE_DIFFERENCE


def test_difference_above_frozen_tolerance_fails() -> None:
    result = _evaluate(
        canonical=[-0.5, 0.5, -0.5, 0.5],
        diagnostic=[-0.5, 0.5, -0.5, 0.5 + ABSOLUTE_TOLERANCE * 4],
    )
    assert result["passed"] is False
    assert result["failure_classification"] == "advantage_projection_mismatch"


def test_equal_rewards_require_exact_zero_advantages() -> None:
    passed = _evaluate(
        rewards=[1.0] * 4,
        canonical=[0.0] * 4,
        duplicate=[0.0] * 4,
        diagnostic=[0.0] * 4,
    )
    assert passed["passed"] is True
    failed = _evaluate(
        rewards=[1.0] * 4,
        canonical=[0.0] * 4,
        duplicate=[0.0] * 4,
        diagnostic=[1e-8, 0.0, 0.0, 0.0],
    )
    assert failed["passed"] is False


def test_nonzero_variance_requires_canonical_nonzero_advantage() -> None:
    result = _evaluate(
        rewards=[0.0, 1.0, 0.0, 1.0],
        canonical=[0.0] * 4,
        duplicate=[0.0] * 4,
        diagnostic=[0.0] * 4,
    )
    assert result["passed"] is False
    assert (
        result["conditions"]["nonzero_variance_rewards_produce_canonical_nonzero_advantage"]
        is False
    )


def test_reward_and_advantage_ordering_are_required() -> None:
    result = _evaluate(
        rewards=[0.0, 1.0, 2.0, 3.0],
        canonical=[-1.5, -0.5, 0.5, 1.5],
        diagnostic=[-1.5, 0.5, -0.5, 1.5],
    )
    assert result["passed"] is False
    assert result["conditions"]["advantage_ordering_preserved_for_unequal_rewards"] is False


def test_material_sign_change_fails() -> None:
    result = _evaluate(diagnostic=[0.5, -0.5, -0.5, 0.5])
    assert result["passed"] is False
    assert result["conditions"]["material_sign_classification_preserved"] is False


def test_group_zero_nonzero_classification_change_fails() -> None:
    result = _evaluate(
        canonical=[-1e-8, 0.0, 0.0, 0.0],
        duplicate=[-1e-8, 0.0, 0.0, 0.0],
        diagnostic=[0.0] * 4,
    )
    assert result["passed"] is False
    assert result["conditions"]["group_zero_nonzero_classification_identical"] is False


@pytest.mark.parametrize(
    ("canonical", "duplicate", "diagnostic"),
    (
        ([math.nan, 0.0, 0.0, 0.0], [math.nan, 0.0, 0.0, 0.0], [math.nan, 0.0, 0.0, 0.0]),
        ([math.inf, 0.0, 0.0, 0.0], [math.inf, 0.0, 0.0, 0.0], [math.inf, 0.0, 0.0, 0.0]),
    ),
)
def test_nonfinite_values_fail(
    canonical: list[float],
    duplicate: list[float],
    diagnostic: list[float],
) -> None:
    result = _evaluate(canonical=canonical, duplicate=duplicate, diagnostic=diagnostic)
    assert result["passed"] is False
    assert result["conditions"]["all_values_finite"] is False


def test_shape_mismatch_fails() -> None:
    result = _evaluate(diagnostic=[-0.5, 0.5, -0.5])
    assert result["passed"] is False
    assert result["conditions"]["shape_identical"] is False


def test_reward_bytes_and_stock_recomputation_are_required() -> None:
    assert _evaluate(reward_hashes_equal=False)["passed"] is False
    assert _evaluate(stock_match=False)["passed"] is False


def test_contract_freezes_full_fixture_and_tolerances() -> None:
    fixture: dict[str, object] = {
        "fixture_vector_count": EXPECTED_FIXTURE_VECTOR_COUNT,
        "exact_mismatch_vector_count": EXPECTED_EXACT_MISMATCH_VECTOR_COUNT,
        "maximum_absolute_difference": EXPECTED_MAXIMUM_FIXTURE_DIFFERENCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "all_values_within_tolerance": True,
    }
    fixture["fixture_sha256"] = canonical_sha256(fixture)
    contract = advantage_equivalence_contract(fixture)
    assert contract["absolute_tolerance"] == 2.384185791015625e-07
    assert contract["relative_tolerance"] == 1e-6
    assert contract["exhaustive_fixture"]["fixture_vector_count"] == 65_536
    assert contract["exhaustive_fixture"]["maximum_absolute_difference"] == (1.1920928955078125e-07)
    supplied = contract["advantage_equivalence_contract_sha256"]
    payload: dict[str, Any] = dict(contract)
    payload.pop("advantage_equivalence_contract_sha256")
    assert supplied == canonical_sha256(payload)
