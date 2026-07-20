"""Minimal weighted-average difficulty-reallocation tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from foundry.synthesis.template_bank.difficulty_reallocation import (
    RATE_FAMILY,
    build_corrected_allocation,
    build_corrected_capacity_audit,
    calibrate_difficulty_reallocation,
    load_difficulty_reallocation_config,
    reallocate_group_difficulties,
    source_rate_matrices,
)
from foundry.synthesis.template_bank.signal_allocator import build_slot_requests
from foundry.synthesis.template_bank.signal_pilot import DIFFICULTY_ORDER, GROUP_ORDER

_CONFIG = Path("configs/synthesis/signal_pilot_difficulty_reallocation.yaml")
_SIGNAL = Path("configs/synthesis/signal_pilot.yaml")
_SUBMODE = Path("configs/synthesis/signal_pilot_submode_policy.yaml")


def _totals(matrix: dict[str, dict[str, int]]) -> tuple[dict[str, int], dict[str, int]]:
    rows = {mode: sum(values.values()) for mode, values in matrix.items()}
    columns = {
        difficulty: sum(values[difficulty] for values in matrix.values())
        for difficulty in DIFFICULTY_ORDER
    }
    return rows, columns


def test_minimal_policy_matches_all_original_fixtures() -> None:
    config = load_difficulty_reallocation_config(_CONFIG)
    result = calibrate_difficulty_reallocation(config)
    selected = cast(
        dict[str, object], cast(dict[str, object], result["candidate_results"])[config.policy_id]
    )
    assert selected["exact_matches"] == 9
    assert selected["mismatched_fixture_ids"] == []
    assert result["selection_frozen_before_schedule_or_smoke"] is True


def test_exact_five_forward_and_five_compensating_moves_are_frozen() -> None:
    config = load_difficulty_reallocation_config(_CONFIG)
    _, shifts = build_corrected_allocation(config)
    weighted = Counter()
    compensating = Counter()
    for group in GROUP_ORDER:
        for shift in shifts[group]:
            count = cast(int, shift["count"])
            target = weighted if shift["mode"] == "weighted_average" else compensating
            target[group] += count
    assert weighted == {"targeted": 3, "generic_control": 2}
    assert compensating == {"targeted": 3, "generic_control": 2}


def test_reallocation_preserves_every_row_and_dataset_difficulty_total() -> None:
    config = load_difficulty_reallocation_config(_CONFIG)
    original, _ = source_rate_matrices(config)
    corrected, _ = build_corrected_allocation(config)
    for group in GROUP_ORDER:
        assert _totals(corrected[group]) == _totals(original[group])


def test_weighted_average_compatibility_shortage_is_exactly_resolved() -> None:
    audit = build_corrected_capacity_audit(_CONFIG)
    assert audit["capacity_gate_passed"] is True
    proof = cast(dict[str, dict[str, object]], audit["weighted_average_compatibility"])
    assert proof["targeted"]["easy_medium_required"] == 44
    assert proof["targeted"]["easy_medium_capacity"] == 44
    assert proof["generic_control"]["easy_medium_required"] == 64
    assert proof["generic_control"]["easy_medium_capacity"] == 64
    assert proof["combined"]["easy_medium_required"] == 108
    assert proof["combined"]["easy_medium_capacity"] == 108


def test_impossible_compensating_shift_fails_closed() -> None:
    config = load_difficulty_reallocation_config(_CONFIG)
    original, accepted = source_rate_matrices(config)
    blocked = {mode: dict(values) for mode, values in accepted["targeted"].items()}
    for mode in config.donor_mode_order:
        blocked[mode]["hard"] = original["targeted"][mode]["hard"]
    with pytest.raises(ValueError, match="no feasible compensating difficulty shift"):
        reallocate_group_difficulties(
            original["targeted"],
            blocked,
            required_moves=1,
            donor_mode_order=config.donor_mode_order,
        )


def test_full_slot_requests_use_the_corrected_rate_matrix() -> None:
    config = load_difficulty_reallocation_config(_CONFIG)
    corrected, _ = build_corrected_allocation(config)
    requests = build_slot_requests(_SIGNAL, _SUBMODE)
    for group in GROUP_ORDER:
        actual = Counter(
            (item.mode, item.difficulty)
            for item in requests
            if item.group == group and item.family == RATE_FAMILY
        )
        assert actual == {
            (mode, difficulty): count
            for mode, row in corrected[group].items()
            for difficulty, count in row.items()
        }
