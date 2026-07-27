"""Validate Milestone 13B recovery without reading sealed-final artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from foundry.phase2.retention_failure_interpretation import select_architecture

VISIBLE_OUTPUT_SHA256 = "7a7c0fb852cc29fd141f2ab2e2804d91bb8d8c21987de73e21b8c3a079755348"
INCIDENT_CLASSIFICATION = "metadata_only_protocol_breach"
SEALED_BOUNDARY_STATUS = "metadata_accessed_example_content_unseen"
ARCHITECTURE_ID = "replay-ce-token-kl-v1"


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible data with stable key ordering and separators."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_frozen(record: dict[str, Any], hash_key: str) -> None:
    """Reject a record whose embedded canonical hash does not replay."""

    supplied = record.get(hash_key)
    payload = dict(record)
    payload.pop(hash_key, None)
    if supplied != canonical_sha256(payload):
        raise ValueError(f"{hash_key} differs from canonical replay")


def validate_initial_manifest(record: dict[str, Any]) -> None:
    """Validate the worktree snapshot frozen before recovery edits."""

    validate_frozen(record, "uncommitted_state_manifest_sha256")
    if (
        record.get("manifest_id") != "foundry-milestone13b-uncommitted-state-v1"
        or record.get("published_commit") != "4c4db2b8ec3fddc136d5efac9c47c81017236931"
        or record.get("origin_main") != "4c4db2b8ec3fddc136d5efac9c47c81017236931"
        or record.get("branch") != "main"
        or record.get("file_count") != 12
        or record.get("total_size_bytes") != 1_201_868
        or record.get("model_process_active") is not False
        or record.get("sealed_path_access_after_reported_stop") is not False
    ):
        raise ValueError("uncommitted-state manifest differs from the frozen entry state")


def validate_incident(record: dict[str, Any]) -> None:
    """Require the incident to contain one metadata field and no example content."""

    validate_frozen(record, "incident_evidence_sha256")
    visible = record.get("exact_visible_output")
    if not isinstance(visible, str) or hashlib.sha256(visible.encode()).hexdigest() != (
        VISIBLE_OUTPUT_SHA256
    ):
        raise ValueError("incident visible output differs")
    zero_fields = (
        "sealed_questions_displayed",
        "sealed_answers_displayed",
        "sealed_ids_displayed",
        "sealed_labels_displayed",
        "sealed_predictions_displayed",
        "sealed_scores_displayed",
        "model_processes_run_after_access",
        "training_processes_run_after_access",
        "gsm1k_evaluations_run_after_access",
        "sealed_final_evaluations_run",
        "scientific_decisions_using_surfaced_value",
    )
    if (
        record.get("incident_id") != "foundry-sealed-manifest-metadata-access-v1"
        or record.get("incident_classification") != INCIDENT_CLASSIFICATION
        or record.get("visible_metadata_fields") != 1
        or record.get("visible_output_sha256") != VISIBLE_OUTPUT_SHA256
        or any(record.get(field) != 0 for field in zero_fields)
        or record.get("accessed_path_sha256") is not None
        or record.get("manifest_reopened_for_recovery") is not False
    ):
        raise ValueError("incident exceeds the metadata-only evidence boundary")


def replay_scientific_decision(repository_root: Path) -> dict[str, Any]:
    """Replay selection from tracked content-free 13B evidence only."""

    interpretation = cast(
        dict[str, Any],
        json.loads(
            (
                repository_root / "results/phase2_vetted_corpus/milestone13b_interpretation.json"
            ).read_text(encoding="utf-8")
        ),
    )
    architecture = cast(
        dict[str, Any],
        json.loads(
            (
                repository_root
                / "results/phase2_vetted_corpus/milestone13b_architecture_contract.json"
            ).read_text(encoding="utf-8")
        ),
    )
    validate_frozen(interpretation, "aggregate_sha256")
    validate_frozen(architecture, "architecture_decision_sha256")
    overlap = cast(dict[str, Any], interpretation["overlap"])
    objective = cast(dict[str, Any], interpretation["objective"])
    selection = cast(dict[str, Any], architecture["selection_contract"])
    selected = select_architecture(
        shared_failure_fraction=float(overlap["shared_failures"]) / 10,
        replay_ratio_trajectories_aligned=bool(selection["v1_v2_residual_drift_aligned"]),
        explicit_logit_constraint=bool(objective["explicit_kl_or_logit_constraint"]),
        capacity_implicated=False,
        gradient_conflict_measured=False,
    )
    if (
        selected != ARCHITECTURE_ID
        or interpretation["failure_inventory"]["generic_failures"] != 10
        or interpretation["failure_inventory"]["targeted_failures"] != 10
        or overlap["shared_failures"] != 10
        or overlap["generic_only"] != 0
        or overlap["targeted_only"] != 0
        or architecture["architecture_id"] != ARCHITECTURE_ID
        or architecture["architecture_decision_sha256"]
        != interpretation["architecture"]["architecture_decision_sha256"]
    ):
        raise ValueError("scientific decision replay differs")
    return {
        "architecture_id": selected,
        "architecture_decision_sha256": architecture["architecture_decision_sha256"],
        "failure_inventory_sha256": architecture["failure_inventory_sha256"],
        "generic_failures": 10,
        "generic_only_failures": 0,
        "objective_audit_sha256": architecture["objective_audit_sha256"],
        "options_comparison_sha256": architecture["options_comparison_sha256"],
        "overlap_analysis_sha256": architecture["overlap_analysis_sha256"],
        "replay_coverage_sha256": architecture["replay_coverage_sha256"],
        "sealed_input_dependencies": [],
        "selection_decision_sha256": architecture["selection_decision_sha256"],
        "shared_failures": 10,
        "targeted_failures": 10,
        "targeted_only_failures": 0,
        "trajectory_analysis_sha256": architecture["trajectory_analysis_sha256"],
    }


def validate_recovery_audit(record: dict[str, Any], *, scientific_replay: dict[str, Any]) -> None:
    """Validate content propagation, independence, and boundary decisions."""

    validate_frozen(record, "recovery_audit_sha256")
    content = cast(dict[str, Any], record["content_propagation_audit"])
    independence = cast(dict[str, Any], record["scientific_independence_audit"])
    validate_frozen(content, "content_propagation_sha256")
    validate_frozen(independence, "scientific_independence_sha256")
    dependency_graph = cast(dict[str, Any], independence["dependency_graph"])
    if (
        record.get("incident_classification") != INCIDENT_CLASSIFICATION
        or record.get("sealed_boundary_status") != SEALED_BOUNDARY_STATUS
        or content.get("decision") != "no_example_content_propagation"
        or content.get("prohibited_content_findings") != 0
        or content.get("copied_manifest_objects") != 0
        or content.get("scientific_features_derived_from_sealed_data") != 0
        or independence.get("decision") != "scientifically_independent"
        or independence.get("sealed_input_dependencies") != []
        or independence.get("architecture_id") != ARCHITECTURE_ID
        or independence.get("architecture_decision_sha256")
        != scientific_replay["architecture_decision_sha256"]
        or independence.get("dependency_graph_sha256") != canonical_sha256(dependency_graph)
        or record.get("sealed_examples_unseen") is not True
        or record.get("zero_file_access_claim_withdrawn") is not True
    ):
        raise ValueError("recovery audit does not establish bounded metadata-only access")


def validate_publication_record(
    record: dict[str, Any],
    *,
    incident: dict[str, Any],
    recovery_audit: dict[str, Any],
    scientific_replay: dict[str, Any],
) -> None:
    """Validate the qualified architecture publication record."""

    validate_frozen(record, "publication_record_sha256")
    not_selected = cast(dict[str, Any], record["not_selected_in_recovery"])
    if (
        record.get("architecture_id") != ARCHITECTURE_ID
        or record.get("architecture_decision_sha256")
        != scientific_replay["architecture_decision_sha256"]
        or record.get("incident_evidence_sha256") != incident["incident_evidence_sha256"]
        or record.get("recovery_audit_sha256") != recovery_audit["recovery_audit_sha256"]
        or record.get("sealed_boundary_status") != SEALED_BOUNDARY_STATUS
        or any(value is not None for value in not_selected.values())
    ):
        raise ValueError("publication record changes the recovered scientific result")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return cast(dict[str, Any], value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--initial-manifest", type=Path, required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument("--publication-record", type=Path, required=True)
    args = parser.parse_args()
    initial = _read_json(args.initial_manifest)
    incident = _read_json(args.incident)
    recovery = _read_json(args.recovery_audit)
    publication = _read_json(args.publication_record)
    validate_initial_manifest(initial)
    validate_incident(incident)
    scientific = replay_scientific_decision(args.repository_root.resolve())
    validate_recovery_audit(recovery, scientific_replay=scientific)
    validate_publication_record(
        publication,
        incident=incident,
        recovery_audit=recovery,
        scientific_replay=scientific,
    )
    print(
        json.dumps(
            {
                "architecture_decision_sha256": scientific["architecture_decision_sha256"],
                "architecture_id": scientific["architecture_id"],
                "incident_evidence_sha256": incident["incident_evidence_sha256"],
                "publication_record_sha256": publication["publication_record_sha256"],
                "recovery_audit_sha256": recovery["recovery_audit_sha256"],
                "sealed_boundary_status": recovery["sealed_boundary_status"],
                "uncommitted_state_manifest_sha256": initial["uncommitted_state_manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
