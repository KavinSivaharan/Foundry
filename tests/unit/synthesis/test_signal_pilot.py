"""Signal-first pilot quota and finite-capacity tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foundry.synthesis.template_bank.signal_pilot import (
    CATEGORY_ORDER,
    GROUP_ORDER,
    build_signal_capacity_audit,
    load_signal_pilot_config,
)

_CONFIG = Path("configs/synthesis/signal_pilot.yaml")
_PREVIOUS = Path("results/synthesis_smoke/template_bank_bounded_reuse_capacity_audit.json")


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def capacity() -> dict[str, object]:
    """Build the reduced-pilot capacity evidence once."""

    return build_signal_capacity_audit(_CONFIG, previous_capacity_path=_PREVIOUS)


def test_signal_first_contract_freezes_exact_quotas() -> None:
    config = load_signal_pilot_config(_CONFIG)

    assert config.attempt_numerator == 5
    assert config.attempt_denominator == 4
    assert sum(item.accepted_total for item in config.datasets.values()) == 2_000
    assert (
        sum(
            family.attempts
            for dataset in config.datasets.values()
            for family in dataset.families.values()
        )
        == 2_504
    )
    assert {
        group: [config.datasets[group].families[category].accepted for category in CATEGORY_ORDER]
        for group in GROUP_ORDER
    } == {
        "targeted": [550, 233, 217],
        "generic_control": [334, 333, 333],
    }
    assert {
        group: [config.datasets[group].families[category].attempts for category in CATEGORY_ORDER]
        for group in GROUP_ORDER
    } == {
        "targeted": [688, 292, 272],
        "generic_control": [418, 417, 417],
    }
    assert all(dataset.training_accepted == 900 for dataset in config.datasets.values())
    assert all(dataset.validation_accepted == 100 for dataset in config.datasets.values())
    assert all(dataset.output_contract_accepted == 200 for dataset in config.datasets.values())


def test_reduced_capacity_gate_catches_combined_compatibility(
    capacity: dict[str, object],
) -> None:
    assert capacity["capacity_gate_passed"] is False
    assert capacity["full_2504_schedule_feasible"] is False
    assert capacity["required_attempt_total"] == 2_504
    expected_group_gate = {
        ("generic_control", "constraint_distribution_or_discrete_reasoning"): False,
    }
    for group in GROUP_ORDER:
        dataset = _mapping(_mapping(capacity["datasets"])[group])
        for category in CATEGORY_ORDER:
            family = _mapping(dataset[category])
            expected = expected_group_gate.get((group, category), True)
            assert family["gate_passed"] is expected
            strata = [
                _mapping(item)["gate_passed"]
                for stratum_name in (
                    "difficulty_strata",
                    "output_contract_strata",
                    "split_strata",
                )
                for item in _mapping(family[stratum_name]).values()
            ]
            assert all(item is expected for item in strata)


def test_finite_combined_capacity_identifies_exact_shortfalls(
    capacity: dict[str, object],
) -> None:
    families = _mapping(capacity["combined_family_capacity"])
    expected = {
        "multi_step_bookkeeping_or_omission": (1_106, 1_384, True),
        "rate_ratio_percentage_or_average": (709, 695, False),
        "constraint_distribution_or_discrete_reasoning": (689, 598, False),
    }
    for category, (required, available, gate) in expected.items():
        record = _mapping(families[category])
        assert record["required_attempts"] == required
        assert record["available_latent_attempt_capacity"] == available
        assert record["gate_passed"] is gate
    assert capacity["limiting_cross_dataset_families"] == [
        {
            "category": "rate_ratio_percentage_or_average",
            "required_attempts": 709,
            "available_compatible_attempts": 695,
            "shortfall": 14,
        },
        {
            "category": "constraint_distribution_or_discrete_reasoning",
            "required_attempts": 689,
            "available_compatible_attempts": 598,
            "shortfall": 91,
        },
    ]
    assert capacity["allocator_implemented"] is False
    assert capacity["full_schedule_created"] is False
    assert capacity["fresh_smoke_run"] is False
    assert capacity["deterministic_replay_run"] is False
    assert capacity["review_packet_created"] is False


def test_frozen_reuse_and_safety_contracts_are_unchanged(
    capacity: dict[str, object],
) -> None:
    assert capacity["reuse_policy_id"] == "bounded-balanced-template-reuse-v1"
    assert capacity["reuse_policy_sha256"] == (
        "66443bc82db961ab2ee34ef8c051928fb27985a733bc9a31e5c60ba3596c25f0"
    )
    assert capacity["generators_changed"] is False
    assert capacity["verifiers_changed"] is False
    assert capacity["benchmark_contamination_changed"] is False
    assert capacity["sealed_final_accessed"] is False
