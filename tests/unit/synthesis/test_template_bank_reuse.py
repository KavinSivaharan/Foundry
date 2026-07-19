"""Milestone 6E bounded-reuse policy and finite-capacity tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundry.synthesis.template_bank.reuse import (
    build_capacity_audit,
    calibrate,
    derive_caps,
    load_contract,
    load_fixtures,
)

_CONFIG = Path("configs/synthesis/template_bank_bounded_reuse_candidates.yaml")


@pytest.fixture(scope="module")
def evidence() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build deterministic, content-free evidence once for this module."""

    contract = load_contract(_CONFIG)
    fixtures = load_fixtures(contract.fixture_path)
    calibration = calibrate(contract, fixtures)
    capacity = build_capacity_audit(contract, calibration)
    return calibration, derive_caps(contract), capacity


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        mapping_keys = set(value)
        for nested in value.values():
            mapping_keys.update(_all_keys(nested))
        return mapping_keys
    if isinstance(value, list):
        list_keys: set[str] = set()
        for nested in value:
            list_keys.update(_all_keys(nested))
        return list_keys
    return set()


def test_selected_policy_matches_every_original_fixture(
    evidence: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    calibration, _, _ = evidence
    candidates = _mapping(calibration["candidate_policies"])
    selected = _mapping(candidates["bounded-balanced-template-reuse-v1"])
    legacy = _mapping(candidates["legacy-one-use-number-neutral"])
    permissive = _mapping(candidates["permissive-exact-latent-only"])

    assert calibration["fixture_count"] == 14
    assert calibration["selected_policy_id"] == "bounded-balanced-template-reuse-v1"
    assert selected["exact_matches"] == 14
    assert selected["mismatched_fixture_ids"] == []
    assert int(legacy["exact_matches"]) < 14
    assert int(permissive["exact_matches"]) < 14
    benchmark = _mapping(calibration["benchmark_contamination"])
    assert benchmark == {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "manual_review_at": 0.75,
        "automatic_reject_at": 0.82,
        "trust_remote_code": False,
        "unchanged": True,
    }


def test_quota_caps_are_derived_exactly(
    evidence: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    _, caps, _ = evidence
    targeted = _mapping(caps["targeted"])
    generic = _mapping(caps["generic_control"])

    expected = {
        "multi_step_bookkeeping_or_omission": ((48, 39, 5, 4), (29, 24, 3, 3)),
        "rate_ratio_percentage_or_average": ((19, 15, 17, 14), (27, 21, 24, 19)),
        "constraint_distribution_or_discrete_reasoning": (
            (17, 14, 5, 4),
            (27, 21, 7, 6),
        ),
    }
    for category, (targeted_values, generic_values) in expected.items():
        for values, source in ((targeted_values, targeted), (generic_values, generic)):
            record = _mapping(source[category])
            assert (
                record["max_attempts_per_sentence_plan"],
                record["max_accepted_per_sentence_plan"],
                record["max_attempts_per_number_neutral_signature"],
                record["max_accepted_per_number_neutral_signature"],
            ) == values


def test_capacity_gate_stops_before_allocation(
    evidence: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    _, _, capacity = evidence
    categories = _mapping(capacity["category_capacity"])
    bookkeeping = _mapping(categories["multi_step_bookkeeping_or_omission"])
    rates = _mapping(categories["rate_ratio_percentage_or_average"])
    discrete = _mapping(categories["constraint_distribution_or_discrete_reasoning"])

    assert capacity["required_attempt_total"] == 10003
    assert bookkeeping["required_attempts"] == 4418
    assert bookkeeping["latent_capacity_under_balanced_frame_or_target_caps"] == 5524
    assert bookkeeping["gate_passed"] is True
    assert rates["required_attempts"] == 2834
    assert rates["latent_capacity_under_balanced_frame_or_target_caps"] == 1632
    assert rates["shortfall"] == 1202
    assert discrete["required_attempts"] == 2751
    assert discrete["latent_capacity_under_balanced_frame_or_target_caps"] == 2073
    assert discrete["shortfall"] == 678
    assert capacity["capacity_gate_passed"] is False
    assert capacity["allocator_implemented"] is False
    assert capacity["candidate_schedule_created"] is False
    assert capacity["fresh_smoke_run"] is False
    assert capacity["deterministic_replay_run"] is False
    assert capacity["review_packet_created"] is False


def test_capacity_evidence_is_content_free(
    evidence: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    _, _, capacity = evidence
    forbidden = {
        "question",
        "rendered_question",
        "canonical_answer",
        "answer",
        "solution",
        "model_output",
    }
    assert forbidden.isdisjoint(_all_keys(capacity))
    serialized = json.dumps(capacity, sort_keys=True)
    assert "How many" not in serialized
