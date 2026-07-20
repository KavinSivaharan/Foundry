from pathlib import Path

import pytest

from foundry.synthesis.template_bank.pilot_size import (
    APPROVED_ACCEPTED_SIZES,
    PilotSizePreflight,
    build_selection_evidence,
    derive_size_candidate,
    select_first_feasible,
    write_size_config,
)
from foundry.synthesis.template_bank.signal_pilot import load_signal_pilot_config

_BASE = Path("configs/synthesis/signal_pilot.yaml")
_POLICY = Path("configs/synthesis/signal_pilot_submode_policy.yaml")


@pytest.mark.parametrize(
    ("size", "targeted", "generic", "attempts"),
    (
        (900, (495, 210, 195), (300, 300, 300), 1_981),
        (800, (440, 186, 174), (267, 267, 266), 1_762),
        (700, (385, 163, 152), (234, 233, 233), 1_544),
        (600, (330, 140, 130), (200, 200, 200), 1_320),
        (500, (275, 117, 108), (167, 167, 166), 1_102),
    ),
)
def test_size_derivation_uses_stable_largest_remainder(
    size: int,
    targeted: tuple[int, int, int],
    generic: tuple[int, int, int],
    attempts: int,
) -> None:
    candidate = derive_size_candidate(size)
    assert tuple(candidate.targeted_accepted.values()) == targeted
    assert tuple(candidate.generic_accepted.values()) == generic
    assert candidate.total_attempts == attempts
    assert candidate.training_per_dataset == size * 9 // 10
    assert candidate.validation_per_dataset == size // 10
    assert candidate.output_contract_per_dataset == size // 5


def test_unapproved_size_rejects() -> None:
    with pytest.raises(ValueError, match="five approved"):
        derive_size_candidate(750)


def test_candidate_config_freezes_exact_reduced_margins(tmp_path: Path) -> None:
    output = tmp_path / "candidate.yaml"
    write_size_config(_BASE, _POLICY, output, 900)
    config = load_signal_pilot_config(output)
    assert {dataset.accepted_total for dataset in config.datasets.values()} == {900}
    assert {dataset.training_accepted for dataset in config.datasets.values()} == {810}
    assert {dataset.validation_accepted for dataset in config.datasets.values()} == {90}
    assert {dataset.output_contract_accepted for dataset in config.datasets.values()} == {180}
    assert (
        sum(
            quota.attempts
            for dataset in config.datasets.values()
            for quota in dataset.families.values()
        )
        == 1_981
    )
    assert all(
        sum(quota.accepted_modes.values()) == quota.accepted
        and sum(quota.attempt_modes.values()) == quota.attempts
        for dataset in config.datasets.values()
        for quota in dataset.families.values()
    )


def _result(index: int, *, feasible: bool) -> PilotSizePreflight:
    candidate = derive_size_candidate(APPROVED_ACCEPTED_SIZES[index])
    return PilotSizePreflight(
        candidate=candidate,
        feasible=feasible,
        config_sha256=f"config-{index}",
        allocation_sha256=f"allocation-{index}",
        schedule_sha256=f"schedule-{index}" if feasible else None,
        summary_sha256=f"summary-{index}" if feasible else None,
        blocker=None if feasible else f"blocker-{index}",
    )


def test_selection_stops_at_largest_feasible_size() -> None:
    selected = select_first_feasible((_result(0, feasible=False), _result(1, feasible=True)))
    assert selected.candidate.accepted_per_dataset == 800


def test_selection_rejects_smaller_evaluation_after_pass() -> None:
    with pytest.raises(ValueError, match="smaller accepted size"):
        select_first_feasible((_result(0, feasible=True), _result(1, feasible=False)))


def test_selection_requires_exact_exhaustion_when_all_fail() -> None:
    with pytest.raises(ValueError, match="none of the five"):
        select_first_feasible(tuple(_result(index, feasible=False) for index in range(5)))


def test_selection_evidence_is_reproducible() -> None:
    results = (_result(0, feasible=False), _result(1, feasible=True))
    allocations = ({"size": 900}, {"size": 800})
    first = build_selection_evidence(
        results,
        selection_config_sha256="selection",
        candidate_allocations=allocations,
    )
    second = build_selection_evidence(
        results,
        selection_config_sha256="selection",
        candidate_allocations=allocations,
    )
    assert first == second
    assert first["selected_accepted_size_per_dataset"] == 800
    assert first["selected_fixed_attempts"] == 1_762
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert first["results"][0]["allocation"] == {"size": 900}


def test_selection_evidence_rejects_misaligned_allocations() -> None:
    with pytest.raises(ValueError, match="align"):
        build_selection_evidence(
            (_result(0, feasible=False),),
            selection_config_sha256="selection",
            candidate_allocations=(),
        )
