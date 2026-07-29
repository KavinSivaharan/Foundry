from __future__ import annotations

from typing import Any

from foundry.phase2.l3_grpo_signal_continuity import (
    CLASSIFICATION,
    CONTINUITY_FIELDS,
    compare_fresh_group_to_prior,
)
from foundry.training.config import canonical_sha256


def _manifest(group: dict[str, Any]) -> dict[str, Any]:
    row = {
        "schedule_position": 1,
        "persisted_scientific_field_sha256s": {
            field: canonical_sha256(group[field]) for field in CONTINUITY_FIELDS
        },
    }
    value: dict[str, Any] = {
        "manifest_id": "foundry-l3-grpo-pre-correction-diagnostic-manifest-v1",
        "classification": CLASSIFICATION,
        "gate_eligible": False,
        "generated_group_count": 13,
        "generated_completion_count": 52,
        "durably_recorded_group_count": 12,
        "durably_recorded_completion_count": 48,
        "durable_groups": [row, *[{"schedule_position": index} for index in range(2, 13)]],
        "completions": [
            {
                "classification": CLASSIFICATION,
                "gate_eligible": False,
            }
            for _ in range(52)
        ],
    }
    value["prior_diagnostic_manifest_sha256"] = canonical_sha256(value)
    return value


def _group(position: int = 1) -> dict[str, Any]:
    value: dict[str, Any] = {field: f"value-{field}" for field in CONTINUITY_FIELDS}
    value["schedule_position"] = position
    return value


def test_durable_group_requires_every_field_hash_to_match() -> None:
    group = _group()
    manifest = _manifest(group)
    exact = compare_fresh_group_to_prior(prior_manifest=manifest, fresh_group=group)
    assert exact["passed"] is True
    changed = dict(group)
    changed["reward_vector"] = "changed"
    mismatch = compare_fresh_group_to_prior(prior_manifest=manifest, fresh_group=changed)
    assert mismatch["passed"] is False
    assert mismatch["failure_classification"] == "scientific_replay_drift"


def test_interrupted_group_records_unavailable_fields_without_fabrication() -> None:
    group = _group(position=13)
    result = compare_fresh_group_to_prior(
        prior_manifest=_manifest(_group()),
        fresh_group=group,
    )
    assert result["passed"] is True
    assert result["status"] == "no_prior_persisted_group_record"
    assert result["prior_fields_available"] == []
    assert result["prior_fields_unavailable"] == list(CONTINUITY_FIELDS)
    assert result["field_comparisons"] == {}
