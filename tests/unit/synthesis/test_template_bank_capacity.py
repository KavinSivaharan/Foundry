"""Milestone 6D runtime-capacity audit tests."""

from __future__ import annotations

import json
from pathlib import Path

from foundry.synthesis.template_bank.bank import build_template_bank
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


def _original_collision_fixture(tmp_path: Path) -> Path:
    bank = build_template_bank()
    first = bank[0]
    shared_plan = first.sentence_plan_variants[0].plan_id
    second = next(
        template
        for template in bank[1:]
        if shared_plan in {plan.plan_id for plan in template.sentence_plan_variants}
    )
    labels = (
        "amber",
        "birch",
        "cobalt",
        "dahlia",
        "elm",
        "fern",
        "granite",
        "hazel",
        "indigo",
        "juniper",
        "kelp",
        "lilac",
        "maple",
        "nickel",
        "opal",
    )
    records: list[dict[str, object]] = []
    for index, label in enumerate(labels, start=1):
        records.append(
            {
                "attempt_index": index,
                "candidate_id": f"fixture-original-{label}",
                "group": "targeted",
                "category": first.reasoning_category,
                "difficulty": "easy",
                "output_contract_enabled": False,
                "template_id": first.template_id,
                "sentence_plan_id": shared_plan,
                "render_signature_sha256": f"original-{label}",
                "rendered_question": f"An original {label} fixture is complete.",
                "latent_program_sha256": f"latent-original-{label}",
                "rejection_reason": None,
            }
        )
    for offset, label in enumerate(labels, start=16):
        records.append(
            {
                "attempt_index": offset,
                "candidate_id": f"fixture-copy-{label}",
                "group": "generic_control",
                "category": second.reasoning_category,
                "difficulty": "medium",
                "output_contract_enabled": True,
                "template_id": second.template_id,
                "sentence_plan_id": shared_plan,
                "render_signature_sha256": f"copy-{label}",
                "rendered_question": f"An original {label} fixture is complete.",
                "latent_program_sha256": f"latent-copy-{label}",
                "rejection_reason": "numeric_template_copy",
            }
        )
    records.append(
        {
            "attempt_index": 31,
            "candidate_id": "fixture-latent-copy",
            "group": "generic_control",
            "category": first.reasoning_category,
            "difficulty": "hard",
            "output_contract_enabled": False,
            "template_id": first.template_id,
            "sentence_plan_id": shared_plan,
            "render_signature_sha256": "latent-copy-signature",
            "rendered_question": "A separate quartz fixture is complete.",
            "latent_program_sha256": "latent-original-amber",
            "rejection_reason": "latent_program_copy",
        }
    )
    path = tmp_path / "original-collision-fixture.jsonl"
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_full_generation_capacity_gate_fails_before_allocation(tmp_path: Path) -> None:
    audit = build_capacity_audit(_original_collision_fixture(tmp_path))
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


def test_collision_inventory_is_complete_and_content_free(tmp_path: Path) -> None:
    audit = build_capacity_audit(_original_collision_fixture(tmp_path))
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
