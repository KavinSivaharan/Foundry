"""Deterministic global signal-pilot allocator tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from foundry.synthesis.template_bank.signal_allocator import (
    LatentCandidate,
    SlotRequest,
    _frame_cap,
    _match_mode_candidates,
    build_full_schedule,
    build_slot_requests,
)
from foundry.synthesis.template_bank.signal_pilot import canonical_sha256, load_signal_pilot_config

_CONFIG = Path("configs/synthesis/signal_pilot.yaml")
_POLICY = Path("configs/synthesis/signal_pilot_submode_policy.yaml")
_RUNTIME_BLOCKER = Path(
    "results/synthesis_smoke/signal_pilot_runtime_identity_capacity_blocker.json"
)


def _candidate(identity: str, *difficulties: str) -> dict[str, LatentCandidate]:
    return {
        difficulty: LatentCandidate(
            seed=int(identity.removeprefix("latent")),
            variant=0,
            latent_sha256=identity,
            semantic_ir_sha256=f"semantic-{identity}-{difficulty}",
            scenario_domain="fixture-domain",
            lexical_family="fixture-lexicon",
            target_type="count",
            primary_evidence_sha256="primary",
            independent_evidence_sha256="independent",
        )
        for difficulty in difficulties
    }


def _request(slot: int, difficulty: str) -> SlotRequest:
    return SlotRequest(
        slot_index=slot,
        group="targeted",
        group_index=slot - 1,
        family="multi_step_bookkeeping_or_omission",
        family_index=slot - 1,
        mode="inventory",
        mode_index=slot - 1,
        difficulty=difficulty,
        output_contract_enabled=False,
        future_split="training",
    )


def test_runtime_exact_weighted_average_capacity_blocker_is_accounted_for() -> None:
    evidence = json.loads(_RUNTIME_BLOCKER.read_text(encoding="utf-8"))

    proof = evidence["fixed_pool_proof"]
    assert proof["unique_runtime_number_neutral_identities"] == 8
    assert proof["strata"] == [
        {
            "group": "targeted",
            "required_attempts": 70,
            "number_neutral_reuse_cap": 5,
            "mathematical_upper_bound": 40,
            "shortfall": 30,
        },
        {
            "group": "generic_control",
            "required_attempts": 100,
            "number_neutral_reuse_cap": 6,
            "mathematical_upper_bound": 48,
            "shortfall": 52,
        },
    ]
    assert evidence["gate"]["complete_2504_slot_schedule"] is False


def test_full_slot_margins_are_exact_and_reproducible() -> None:
    first = build_slot_requests(_CONFIG, _POLICY)
    second = build_slot_requests(_CONFIG, _POLICY)
    assert first == second
    assert len(first) == 2_504
    assert len({item.slot_id for item in first}) == 2_504
    assert canonical_sha256([item.__dict__ for item in first]) == (
        "efea00fff13e6ca0cf9f2f44a66c60135afdfb59f1f0ea3216a18d0db7af2dc6"
    )
    assert Counter((item.group, item.family) for item in first) == {
        ("targeted", "multi_step_bookkeeping_or_omission"): 688,
        ("targeted", "rate_ratio_percentage_or_average"): 292,
        ("targeted", "constraint_distribution_or_discrete_reasoning"): 272,
        ("generic_control", "multi_step_bookkeeping_or_omission"): 418,
        ("generic_control", "rate_ratio_percentage_or_average"): 417,
        ("generic_control", "constraint_distribution_or_discrete_reasoning"): 417,
    }
    assert Counter(item.difficulty for item in first) == {
        "easy": 837,
        "medium": 834,
        "hard": 833,
    }
    assert Counter((item.group, item.output_contract_enabled) for item in first) == {
        ("targeted", True): 250,
        ("targeted", False): 1_002,
        ("generic_control", True): 250,
        ("generic_control", False): 1_002,
    }


def test_latent_matching_is_unique_and_satisfies_difficulty_margins() -> None:
    requests = (
        _request(1, "easy"),
        _request(2, "easy"),
        _request(3, "hard"),
    )
    candidates = {
        "latent1": _candidate("latent1", "easy", "hard"),
        "latent2": _candidate("latent2", "easy"),
        "latent3": _candidate("latent3", "hard"),
    }
    selected = _match_mode_candidates(requests, candidates)
    flattened = [item.latent_sha256 for values in selected.values() for item in values]
    assert len(flattened) == len(set(flattened)) == 3
    assert len(selected["easy"]) == 2
    assert len(selected["hard"]) == 1


def test_latent_matching_fails_closed_when_unique_assignment_is_impossible() -> None:
    requests = (_request(1, "easy"), _request(2, "hard"))
    candidates = {"latent1": _candidate("latent1", "easy", "hard")}
    with pytest.raises(ValueError, match="cannot satisfy"):
        _match_mode_candidates(requests, candidates)


def test_bookkeeping_frame_cap_covers_both_shared_modes() -> None:
    request = _request(1, "easy")
    assert (
        _frame_cap(
            request=request,
            frame_count=18,
            config=load_signal_pilot_config(_CONFIG),
        )
        == 39
    )


def test_complete_schedule_fails_closed_on_joint_surface_compatibility() -> None:
    with pytest.raises(
        ValueError,
        match=("generic_control/constraint_distribution_or_discrete_reasoning/complete_packages"),
    ):
        build_full_schedule(_CONFIG, _POLICY)
