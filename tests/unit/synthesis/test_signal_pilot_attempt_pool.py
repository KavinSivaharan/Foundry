from pathlib import Path

import pytest

from foundry.synthesis.template_bank.attempt_pool import (
    APPROVED_MULTIPLIERS,
    AttemptPoolPreflight,
    build_attempt_pool_allocation,
    build_selection_evidence,
    derive_attempt_counts,
    select_first_feasible,
    write_candidate_config,
)
from foundry.synthesis.template_bank.signal_pilot import load_signal_pilot_config

_BASE = Path("configs/synthesis/signal_pilot.yaml")
_POLICY = Path("configs/synthesis/signal_pilot_submode_policy.yaml")


@pytest.mark.parametrize(
    ("numerator", "denominator", "targeted", "generic", "total"),
    (
        (23, 20, (633, 268, 250), (385, 383, 383), 2_302),
        (9, 8, (619, 263, 245), (376, 375, 375), 2_253),
        (11, 10, (605, 257, 239), (368, 367, 367), 2_203),
    ),
)
def test_approved_multiplier_family_totals(
    numerator: int,
    denominator: int,
    targeted: tuple[int, int, int],
    generic: tuple[int, int, int],
    total: int,
) -> None:
    candidate = derive_attempt_counts(_BASE, numerator, denominator)
    assert tuple(candidate.targeted_attempts.values()) == targeted
    assert tuple(candidate.generic_attempts.values()) == generic
    assert candidate.total_attempts == total


def test_unapproved_multiplier_rejects() -> None:
    with pytest.raises(ValueError, match="three approved"):
        derive_attempt_counts(_BASE, 21, 20)


def test_candidate_config_preserves_accepted_quotas_and_derives_margins(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate.yaml"
    write_candidate_config(_BASE, _POLICY, output, 23, 20)
    config = load_signal_pilot_config(output)
    assert (config.attempt_numerator, config.attempt_denominator) == (23, 20)
    assert (
        sum(
            quota.accepted
            for dataset in config.datasets.values()
            for quota in dataset.families.values()
        )
        == 2_000
    )
    assert (
        sum(
            quota.attempts
            for dataset in config.datasets.values()
            for quota in dataset.families.values()
        )
        == 2_302
    )
    assert (
        sum(
            quota.output_contract_attempts
            for quota in config.datasets["targeted"].families.values()
        )
        == 230
    )


def test_attempt_pool_allocation_reuses_frozen_policies(tmp_path: Path) -> None:
    output = tmp_path / "candidate.yaml"
    write_candidate_config(_BASE, _POLICY, output, 23, 20)
    allocation = build_attempt_pool_allocation(output, _POLICY)
    assert allocation["capacity_gate_passed"] is True
    assert allocation["total_attempts"] == 2_302
    assert allocation["difficulty_required_moves"] == {
        "targeted": 1,
        "generic_control": 3,
    }
    assert allocation["global_attempt_modes"] == {
        "multi_step_bookkeeping_or_omission": {"inventory": 509, "grouping": 509},
        "rate_ratio_percentage_or_average": {
            "rate_total": 96,
            "ratio_scale": 151,
            "percentage": 104,
            "weighted_average": 150,
            "combined_rate": 150,
        },
        "constraint_distribution_or_discrete_reasoning": {
            "two_type_allocation": 181,
            "complete_packages": 181,
            "equal_distribution": 181,
            "dual_capacity": 90,
        },
    }


def _result(index: int, *, feasible: bool) -> AttemptPoolPreflight:
    numerator, denominator = APPROVED_MULTIPLIERS[index]
    candidate = derive_attempt_counts(_BASE, numerator, denominator)
    return AttemptPoolPreflight(
        candidate=candidate,
        feasible=feasible,
        config_sha256=f"config-{index}",
        schedule_sha256=f"schedule-{index}" if feasible else None,
        summary_sha256=f"summary-{index}" if feasible else None,
        blocker=None if feasible else f"blocker-{index}",
    )


def test_selection_stops_at_first_feasible_multiplier() -> None:
    selected = select_first_feasible((_result(0, feasible=False), _result(1, feasible=True)))
    assert selected.candidate.label == "1.125"


def test_selection_rejects_lower_evaluation_after_pass() -> None:
    with pytest.raises(ValueError, match="lower multiplier"):
        select_first_feasible((_result(0, feasible=True), _result(1, feasible=False)))


def test_selection_fails_after_exactly_three_infeasible_results() -> None:
    with pytest.raises(ValueError, match="none of the three"):
        select_first_feasible(tuple(_result(index, feasible=False) for index in range(3)))


def test_stopped_selection_evidence_is_exact_and_reproducible(tmp_path: Path) -> None:
    paths = tuple(tmp_path / f"candidate-{index}.yaml" for index in range(3))
    for path, (numerator, denominator) in zip(paths, APPROVED_MULTIPLIERS, strict=True):
        write_candidate_config(_BASE, _POLICY, path, numerator, denominator)
    blockers = tuple(f"exact-blocker-{index}" for index in range(3))
    first = build_selection_evidence(
        base_config_path=_BASE,
        policy_path=_POLICY,
        candidate_config_paths=paths,
        blockers=blockers,
        selection_config_sha256="selection-config",
    )
    second = build_selection_evidence(
        base_config_path=_BASE,
        policy_path=_POLICY,
        candidate_config_paths=paths,
        blockers=blockers,
        selection_config_sha256="selection-config",
    )
    assert first == second
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert first["selected_multiplier"] is None
    assert first["selection_gate_passed"] is False
    assert [item["total_attempts"] for item in first["results"]] == [2_302, 2_253, 2_203]
