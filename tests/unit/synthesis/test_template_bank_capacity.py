"""Milestone 6D runtime-capacity audit tests."""

from __future__ import annotations

import json
from pathlib import Path

from foundry.synthesis.template_bank.capacity import build_capacity_audit


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for nested in value.values():
            result.update(_all_keys(nested))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for nested in value:
            result.update(_all_keys(nested))
        return result
    return set()


def test_full_generation_capacity_gate_fails_before_allocation() -> None:
    audit = build_capacity_audit(Path("results/raw/template_bank_smoke_v3/attempts.jsonl"))
    capacities = audit["capacity_by_category"]
    assert isinstance(capacities, dict)
    assert {
        category: values["number_neutral_signatures"] for category, values in capacities.items()
    } == {
        "multi_step_bookkeeping_or_omission": 768,
        "rate_ratio_percentage_or_average": 88,
        "constraint_distribution_or_discrete_reasoning": 320,
    }
    assert {
        category: values["active_plan_render_signatures"] for category, values in capacities.items()
    } == {
        "multi_step_bookkeeping_or_omission": 72,
        "rate_ratio_percentage_or_average": 80,
        "constraint_distribution_or_discrete_reasoning": 80,
    }
    assert audit["required_125_percent_attempt_pool_total"] == 10003
    assert audit["available_unique_total_under_all_current_controls"] == 232
    assert audit["capacity_gate_passed"] is False
    assert audit["allocator_implemented"] is False
    assert audit["candidate_schedule_created"] is False
    assert audit["fresh_smoke_run"] is False
    assert audit["review_packet_created"] is False


def test_collision_inventory_is_complete_and_content_free() -> None:
    audit = build_capacity_audit(Path("results/raw/template_bank_smoke_v3/attempts.jsonl"))
    inventory = audit["collision_inventory"]
    assert isinstance(inventory, dict)
    diagnoses = inventory["diagnoses"]
    assert isinstance(diagnoses, list)
    assert len(diagnoses) == 16
    assert inventory["number_neutral_collisions"] == 15
    assert inventory["latent_program_collisions"] == 1
    assert inventory["root_cause_counts"] == {
        "latent_seed_collision": 1,
        "same_plan_and_lexical_realization_across_semantic_frames": 15,
    }
    assert {
        "question",
        "rendered_question",
        "canonical_answer",
        "answer",
        "solution",
    }.isdisjoint(_all_keys(audit))
    serialized = json.dumps(audit, sort_keys=True)
    assert "How many" not in serialized
