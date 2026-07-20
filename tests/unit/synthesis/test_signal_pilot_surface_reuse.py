"""Submode-local canonical surface-reuse policy tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from foundry.synthesis.template_bank.signal_pilot import CATEGORY_ORDER, load_signal_pilot_config
from foundry.synthesis.template_bank.surface_reuse import (
    SELECTED_POLICY_ID,
    build_surface_capacity_audit,
    calibrate_surface_reuse,
    derive_surface_caps,
    load_surface_reuse_config,
    load_surface_reuse_fixtures,
)

_CONFIG = Path("configs/synthesis/signal_pilot_surface_reuse.yaml")


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_selected_policy_matches_all_original_fixtures() -> None:
    config = load_surface_reuse_config(_CONFIG)
    result = calibrate_surface_reuse(config, load_surface_reuse_fixtures(config.fixture_path))

    assert result["policy_id"] == SELECTED_POLICY_ID
    selected = _mapping(_mapping(result["candidate_results"])[SELECTED_POLICY_ID])
    assert selected["exact_matches"] == result["fixture_count"] == 10
    assert selected["mismatched_fixture_ids"] == []
    assert (
        _mapping(result["candidate_results"])["family-level-bounded-surface-reuse-v1"] != selected
    )
    assert _mapping(result["candidate_results"])["permissive-exact-latent-only-v1"] != selected


def test_caps_are_derived_for_every_frozen_submode() -> None:
    policy = load_surface_reuse_config(_CONFIG)
    signal = load_signal_pilot_config(policy.signal_pilot_config)
    caps = derive_surface_caps(signal, policy)

    weighted_targeted = caps["targeted"][CATEGORY_ORDER[1]]["weighted_average"]
    weighted_generic = caps["generic_control"][CATEGORY_ORDER[1]]["weighted_average"]
    assert weighted_targeted.max_attempts_per_identity == 11
    assert weighted_generic.max_attempts_per_identity == 16
    assert weighted_targeted.max_accepted_per_identity == 8
    assert weighted_generic.max_accepted_per_identity == 12
    assert weighted_targeted.active_identities == weighted_generic.active_identities == 8
    assert sum(len(modes) for families in caps.values() for modes in families.values()) == 22


def test_aggregate_caps_cover_all_modes_but_difficulty_gate_fails_closed() -> None:
    audit = build_surface_capacity_audit(_CONFIG)

    assert audit["capacity_gate_passed"] is False
    assert audit["full_2504_schedule_authorized"] is False
    assert audit["all_eleven_submodes_audited"] is True
    weighted = _mapping(
        _mapping(_mapping(audit["global_submodes"])[CATEGORY_ORDER[1]])["weighted_average"]
    )
    assert weighted["required_attempts"] == 170
    assert weighted["active_runtime_identities"] == 8
    assert weighted["global_max_attempts_per_identity"] == 27
    assert weighted["global_attempt_capacity"] == 216
    assert weighted["headroom"] == 46
    proof = _mapping(
        _mapping(_mapping(audit["difficulty_compatibility_proof"])[CATEGORY_ORDER[1]])[
            "weighted_average"
        ]
    )
    assert _mapping(proof["targeted"]) == {
        "difficulty_group": ["easy", "medium"],
        "required_attempts": 47,
        "compatible_runtime_identities": 4,
        "max_attempts_per_identity": 11,
        "compatible_capacity": 44,
        "shortfall": 3,
        "gate_passed": False,
    }
    assert _mapping(proof["generic_control"])["shortfall"] == 2
    assert _mapping(proof["combined"])["shortfall"] == 5


def test_policy_preserves_exact_latent_and_benchmark_controls() -> None:
    audit = build_surface_capacity_audit(_CONFIG)

    assert audit["exact_question_uniqueness_changed"] is False
    assert audit["latent_program_uniqueness_changed"] is False
    assert audit["benchmark_contamination_changed"] is False
    assert audit["sealed_final_accessed"] is False
