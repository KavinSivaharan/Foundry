"""Capacity-aware signal-pilot submode policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry.synthesis.template_bank.submode_policy import (
    SELECTED_POLICY_ID,
    balanced_matrix,
    build_revised_capacity_audit,
    calibrate_policy,
    constrained_attempt_split,
    constrained_difficulty_matrices,
    load_policy_config,
    water_fill,
)

_CONFIG = Path("configs/synthesis/signal_pilot_submode_policy.yaml")


def test_original_fixture_calibration_selects_water_filling() -> None:
    calibration = calibrate_policy(load_policy_config(_CONFIG))
    results = calibration["candidate_results"]
    assert isinstance(results, dict)
    assert calibration["fixture_count"] == 9
    assert calibration["selected_policy_id"] == SELECTED_POLICY_ID
    assert results[SELECTED_POLICY_ID]["exact_matches"] == 9
    assert results[SELECTED_POLICY_ID]["mismatched_fixture_ids"] == []
    assert results["equal-style-uniform-caps-v1"]["exact_matches"] == 5
    assert results["proportional-to-capacity-v1"]["exact_matches"] == 6


def test_real_attempt_water_filling_and_dataset_split_are_exact() -> None:
    config = load_policy_config(_CONFIG)
    totals = {
        "multi_step_bookkeeping_or_omission": (1_106, 688),
        "rate_ratio_percentage_or_average": (709, 292),
        "constraint_distribution_or_discrete_reasoning": (689, 272),
    }
    expected_global = {
        "multi_step_bookkeeping_or_omission": {"inventory": 553, "grouping": 553},
        "rate_ratio_percentage_or_average": {
            "rate_total": 96,
            "ratio_scale": 170,
            "percentage": 104,
            "weighted_average": 170,
            "combined_rate": 169,
        },
        "constraint_distribution_or_discrete_reasoning": {
            "two_type_allocation": 200,
            "complete_packages": 200,
            "equal_distribution": 199,
            "dual_capacity": 90,
        },
    }
    expected_targeted = {
        "multi_step_bookkeeping_or_omission": {"inventory": 344, "grouping": 344},
        "rate_ratio_percentage_or_average": {
            "rate_total": 40,
            "ratio_scale": 70,
            "percentage": 43,
            "weighted_average": 70,
            "combined_rate": 69,
        },
        "constraint_distribution_or_discrete_reasoning": {
            "two_type_allocation": 79,
            "complete_packages": 79,
            "equal_distribution": 78,
            "dual_capacity": 36,
        },
    }
    for family, (total, targeted_total) in totals.items():
        global_modes = water_fill(total, config.capacities[family])
        global_accepted = water_fill(
            {
                "multi_step_bookkeeping_or_omission": 884,
                "rate_ratio_percentage_or_average": 566,
                "constraint_distribution_or_discrete_reasoning": 550,
            }[family],
            config.capacities[family],
        )
        targeted, generic, _, _ = constrained_attempt_split(
            global_modes,
            global_accepted,
            first_attempt_total=targeted_total,
            first_accepted_total={
                "multi_step_bookkeeping_or_omission": 550,
                "rate_ratio_percentage_or_average": 233,
                "constraint_distribution_or_discrete_reasoning": 217,
            }[family],
        )
        assert global_modes == expected_global[family]
        assert targeted == expected_targeted[family]
        assert sum(generic.values()) == total - targeted_total


def test_subordinate_matrix_preserves_exact_margins_deterministically() -> None:
    rows = {"low": 3, "ample": 7, "other": 5}
    columns = {"easy": 5, "medium": 5, "hard": 5}
    first = balanced_matrix(rows, columns)
    second = balanced_matrix(rows, columns)
    assert first == second
    assert {row: sum(values.values()) for row, values in first.items()} == rows
    assert {column: sum(first[row][column] for row in rows) for column in columns} == columns


def test_difficulty_allocation_saturates_low_capacity_cells_without_losing_balance() -> None:
    result = constrained_difficulty_matrices(
        {
            "two_type_allocation": 200,
            "complete_packages": 200,
            "equal_distribution": 199,
            "dual_capacity": 90,
        },
        {
            "two_type_allocation": 79,
            "complete_packages": 79,
            "equal_distribution": 78,
            "dual_capacity": 36,
        },
        targeted_total=272,
        generic_total=417,
        cell_capacities={
            "two_type_allocation": {"easy": 865, "medium": 865, "hard": 865},
            "complete_packages": {"easy": 189, "medium": 315, "hard": 835},
            "equal_distribution": {"easy": 33, "medium": 60, "hard": 160},
            "dual_capacity": {"easy": 30, "medium": 30, "hard": 30},
        },
    )
    assert result["global"]["equal_distribution"] == {
        "easy": 33,
        "medium": 60,
        "hard": 106,
    }
    assert result["global"]["dual_capacity"] == {
        "easy": 30,
        "medium": 30,
        "hard": 30,
    }
    assert {
        difficulty: sum(result["global"][mode][difficulty] for mode in result["global"])
        for difficulty in ("easy", "medium", "hard")
    } == {"easy": 230, "medium": 230, "hard": 229}


def test_water_filling_and_subordinate_allocation_fail_closed() -> None:
    with pytest.raises(ValueError, match="insufficient_total_capacity"):
        water_fill(7, {"alpha": 2, "beta": 2, "gamma": 2})
    with pytest.raises(ValueError, match="impossible_subordinate_compatibility"):
        balanced_matrix(
            {"alpha": 3, "beta": 3},
            {"enabled": 4, "disabled": 2},
            cell_capacities={
                "alpha": {"enabled": 1, "disabled": 2},
                "beta": {"enabled": 1, "disabled": 2},
            },
        )


def test_revised_real_capacity_gate_passes_every_family_and_dataset() -> None:
    audit = build_revised_capacity_audit(_CONFIG)
    assert audit["capacity_gate_passed"] is True
    assert audit["full_2504_schedule_feasible"] is True
    assert audit["required_attempt_total"] == 2_504
    families = audit["families"]
    assert isinstance(families, dict)
    expected = {
        "multi_step_bookkeeping_or_omission": (1_106, 5_524),
        "rate_ratio_percentage_or_average": (709, 1_632),
        "constraint_distribution_or_discrete_reasoning": (689, 2_073),
    }
    for family, (required, available) in expected.items():
        record = families[family]
        assert record["required_attempts"] == required
        assert record["available_unique_capacity"] == available
        assert record["gate_passed"] is True
        assert all(detail["coverage_nonzero"] is True for detail in record["mode_details"].values())
    datasets = audit["datasets"]
    assert isinstance(datasets, dict)
    assert all(
        record["gate_passed"] is True for group in datasets.values() for record in group.values()
    )
    assert audit["allocator_implemented"] is False
    assert audit["full_schedule_created"] is False
    assert audit["fresh_smoke_run"] is False
    assert audit["review_packet_created"] is False
