import hashlib

import pytest

from foundry.phase2.sealed_metadata_recovery import (
    VISIBLE_OUTPUT_SHA256,
    canonical_sha256,
    validate_frozen,
    validate_incident,
)


def _incident() -> dict[str, object]:
    visible = '"partition"' + ': "sealed_final"'
    payload: dict[str, object] = {
        "accessed_path_sha256": None,
        "exact_visible_output": visible,
        "gsm1k_evaluations_run_after_access": 0,
        "incident_classification": "metadata_only_protocol_breach",
        "incident_id": "foundry-sealed-manifest-metadata-access-v1",
        "manifest_reopened_for_recovery": False,
        "model_processes_run_after_access": 0,
        "scientific_decisions_using_surfaced_value": 0,
        "sealed_answers_displayed": 0,
        "sealed_final_evaluations_run": 0,
        "sealed_ids_displayed": 0,
        "sealed_labels_displayed": 0,
        "sealed_predictions_displayed": 0,
        "sealed_questions_displayed": 0,
        "sealed_scores_displayed": 0,
        "training_processes_run_after_access": 0,
        "visible_metadata_fields": 1,
        "visible_output_sha256": hashlib.sha256(visible.encode()).hexdigest(),
    }
    payload["incident_evidence_sha256"] = canonical_sha256(payload)
    return payload


def test_incident_accepts_only_the_frozen_metadata_output() -> None:
    incident = _incident()
    assert incident["visible_output_sha256"] == VISIBLE_OUTPUT_SHA256
    validate_incident(incident)


def test_incident_rejects_example_level_or_altered_evidence() -> None:
    incident = _incident()
    incident["sealed_ids_displayed"] = 1
    incident["incident_evidence_sha256"] = canonical_sha256(
        {key: value for key, value in incident.items() if key != "incident_evidence_sha256"}
    )
    with pytest.raises(ValueError, match="metadata-only"):
        validate_incident(incident)


def test_frozen_record_is_tamper_sensitive() -> None:
    record = {"value": 1}
    record["sha256"] = canonical_sha256(record)
    validate_frozen(record, "sha256")
    record["value"] = 2
    with pytest.raises(ValueError, match="canonical replay"):
        validate_frozen(record, "sha256")
