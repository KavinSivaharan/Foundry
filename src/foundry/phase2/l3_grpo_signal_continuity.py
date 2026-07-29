"""Freeze and compare the interrupted Milestone 14B audit without reusing it as a gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

MANIFEST_ID = "foundry-l3-grpo-pre-correction-diagnostic-manifest-v1"
COMPARISON_ID = "foundry-l3-grpo-pre-correction-continuity-v1"
CLASSIFICATION = "pre_correction_non_gate_diagnostic"
PRIOR_RAW_RELATIVE = Path("results/raw/phase2_vetted_corpus/milestone14b/signal_audit")
PRIOR_PARTIAL_RELATIVE = PRIOR_RAW_RELATIVE / "generic/partial_evidence.json"
PRIOR_BLOCKER_RELATIVE = Path("results/phase2_vetted_corpus/milestone14b_signal_audit_blocker.json")
EXPECTED_PRIOR_PARTIAL_SELF_SHA256 = (
    "37313f3f9f378ff994006f2ed185ca50ee5160ef9c34926f7a468342d83be8b5"
)
EXPECTED_PRIOR_PARTIAL_FILE_SHA256 = (
    "35748e1f6448286f7f17b6f7951e4637259fa598e0e942ed3758e9e9643fe9d8"
)
EXPECTED_PRIOR_BLOCKER_SELF_SHA256 = (
    "0597d2a628bf6acfa320cd0f567c4d3a6e296067bad1767974cb446b7c184ae4"
)
EXPECTED_PRIOR_BLOCKER_FILE_SHA256 = (
    "57d0247ed439fe6da8df09bf4f45e5bf636dd5e3bc0f81c83feab3b60d3c9de2"
)
EXPECTED_PRIOR_FILE_COUNT = 5
EXPECTED_PRIOR_DISK_BYTES = 1_045_652
PRIOR_GENERATED_GROUPS = 13
PRIOR_GENERATED_COMPLETIONS = 52
PRIOR_DURABLE_GROUPS = 12
PRIOR_DURABLE_COMPLETIONS = 48

CONTINUITY_FIELDS = (
    "schedule_position",
    "prompt_sha256",
    "generated_token_ids",
    "completion_sha256s",
    "completion_token_counts",
    "decoded_completion_token_counts",
    "reward_vector_unprojected",
    "reward_vector",
    "reward_component_vectors",
    "correctness_vector",
    "correctness_count",
    "extraction_count",
    "compliant_format_count",
    "truncation_count",
    "backend_failure_count",
    "valid_completion_token_counts",
    "valid_completion_token_count",
)
ADVANTAGE_FIELDS_EXCLUDED_FROM_EXACT_CONTINUITY = (
    "advantages",
    "reward_mean",
    "reward_variance",
    "nonzero_advantage_count",
)


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify(value: Mapping[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _prior_files(root: Path) -> list[dict[str, object]]:
    raw_root = root / PRIOR_RAW_RELATIVE
    paths = sorted(path for path in raw_root.rglob("*") if path.is_file())
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    if len(rows) != EXPECTED_PRIOR_FILE_COUNT:
        raise ValueError("prior signal-audit file count differs")
    if sum(cast(int, row["bytes"]) for row in rows) != EXPECTED_PRIOR_DISK_BYTES:
        raise ValueError("prior signal-audit disk accounting differs")
    return rows


def _field_hashes(group: Mapping[str, Any]) -> dict[str, str]:
    missing = [field for field in CONTINUITY_FIELDS if field not in group]
    if missing:
        raise ValueError(f"prior group continuity fields are missing: {missing}")
    return {field: canonical_sha256(group[field]) for field in CONTINUITY_FIELDS}


def build_prior_diagnostic_manifest(root: Path) -> dict[str, object]:
    """Build a tracked, content-free index of all 52 pre-correction completions."""

    root = root.resolve()
    partial_path = root / PRIOR_PARTIAL_RELATIVE
    blocker_path = root / PRIOR_BLOCKER_RELATIVE
    partial = _read(partial_path)
    blocker = _read(blocker_path)
    _verify(partial, "partial_audit_sha256")
    _verify(blocker, "blocker_sha256")
    if (
        partial.get("partial_audit_sha256") != EXPECTED_PRIOR_PARTIAL_SELF_SHA256
        or file_sha256(partial_path) != EXPECTED_PRIOR_PARTIAL_FILE_SHA256
        or blocker.get("blocker_sha256") != EXPECTED_PRIOR_BLOCKER_SELF_SHA256
        or file_sha256(blocker_path) != EXPECTED_PRIOR_BLOCKER_FILE_SHA256
        or partial.get("completed_group_count") != PRIOR_DURABLE_GROUPS
        or partial.get("completed_completion_count") != PRIOR_DURABLE_COMPLETIONS
        or blocker.get("generated_group_count") != PRIOR_GENERATED_GROUPS
        or blocker.get("generated_completion_count") != PRIOR_GENERATED_COMPLETIONS
        or blocker.get("published_group_count") != PRIOR_DURABLE_GROUPS
        or blocker.get("published_completion_count") != PRIOR_DURABLE_COMPLETIONS
        or blocker.get("failed_schedule_position") != 13
        or blocker.get("failed_group_evidence_published") is not False
    ):
        raise ValueError("pre-correction partial audit does not match the published blocker")
    groups_value = partial.get("groups")
    if not isinstance(groups_value, list) or len(groups_value) != PRIOR_DURABLE_GROUPS:
        raise ValueError("pre-correction durable group evidence differs")
    durable_groups: list[dict[str, object]] = []
    completion_index: list[dict[str, object]] = []
    for position, group_value in enumerate(groups_value, start=1):
        if not isinstance(group_value, dict):
            raise ValueError("pre-correction group evidence must be an object")
        group = cast(dict[str, Any], group_value)
        _verify(group, "group_record_sha256")
        completion_hashes = group.get("completion_sha256s")
        if (
            group.get("schedule_position") != position
            or not isinstance(completion_hashes, list)
            or len(completion_hashes) != 4
            or not all(isinstance(value, str) and len(value) == 64 for value in completion_hashes)
        ):
            raise ValueError("pre-correction group identity or completion accounting differs")
        durable_groups.append(
            {
                "schedule_position": position,
                "group_id": group["group_id"],
                "group_record_sha256": group["group_record_sha256"],
                "persisted_scientific_field_sha256s": _field_hashes(group),
                "advantage_fields_excluded_from_exact_continuity": list(
                    ADVANTAGE_FIELDS_EXCLUDED_FROM_EXACT_CONTINUITY
                ),
                "completion_count": 4,
                "classification": CLASSIFICATION,
                "gate_eligible": False,
            }
        )
        completion_index.extend(
            {
                "diagnostic_completion_id": f"generic-position-{position:03d}-completion-{index}",
                "completion_sha256": completion_sha256,
                "identifier_status": "persisted_completion_sha256",
                "classification": CLASSIFICATION,
                "gate_eligible": False,
            }
            for index, completion_sha256 in enumerate(
                cast(list[str], completion_hashes),
                start=1,
            )
        )
    completion_index.extend(
        {
            "diagnostic_completion_id": f"generic-position-013-completion-{index}",
            "completion_sha256": None,
            "identifier_status": "unavailable_from_interrupted_packet",
            "classification": CLASSIFICATION,
            "gate_eligible": False,
        }
        for index in range(1, 5)
    )
    if len(completion_index) != PRIOR_GENERATED_COMPLETIONS:
        raise ValueError("pre-correction completion classification accounting differs")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "classification": CLASSIFICATION,
        "gate_eligible": False,
        "prior_source_commit": partial["source_commit"],
        "original_signal_audit_contract_sha256": partial["signal_audit_contract_sha256"],
        "published_blocker_sha256": blocker["blocker_sha256"],
        "prior_partial_audit_sha256": partial["partial_audit_sha256"],
        "prior_partial_file_sha256": file_sha256(partial_path),
        "prior_blocker_file_sha256": file_sha256(blocker_path),
        "prior_raw_files": _prior_files(root),
        "generated_group_count": PRIOR_GENERATED_GROUPS,
        "generated_completion_count": PRIOR_GENERATED_COMPLETIONS,
        "durably_recorded_group_count": PRIOR_DURABLE_GROUPS,
        "durably_recorded_completion_count": PRIOR_DURABLE_COMPLETIONS,
        "durable_group_position_coverage": list(range(1, PRIOR_DURABLE_GROUPS + 1)),
        "interrupted_group": {
            "schedule_position": 13,
            "generated_completion_count": 4,
            "persisted_fields": [],
            "unavailable_scientific_fields": list(CONTINUITY_FIELDS),
            "completion_identifiers_available": False,
            "classification": CLASSIFICATION,
            "gate_eligible": False,
        },
        "durable_groups": durable_groups,
        "completions": completion_index,
        "excluded_from": [
            "final_signal_density_counts",
            "viability_selection",
            "representative_group_selection",
            "compatibility_selection",
        ],
        "raw_evidence_deleted": False,
        "raw_evidence_overwritten": False,
        "raw_evidence_reinterpreted_as_gate": False,
    }
    manifest["prior_diagnostic_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def verify_prior_diagnostic_manifest(value: Mapping[str, Any]) -> None:
    _verify(value, "prior_diagnostic_manifest_sha256")
    completions = value.get("completions")
    if (
        value.get("manifest_id") != MANIFEST_ID
        or value.get("classification") != CLASSIFICATION
        or value.get("gate_eligible") is not False
        or value.get("generated_group_count") != PRIOR_GENERATED_GROUPS
        or value.get("generated_completion_count") != PRIOR_GENERATED_COMPLETIONS
        or value.get("durably_recorded_group_count") != PRIOR_DURABLE_GROUPS
        or value.get("durably_recorded_completion_count") != PRIOR_DURABLE_COMPLETIONS
        or not isinstance(completions, list)
        or len(completions) != PRIOR_GENERATED_COMPLETIONS
        or any(
            not isinstance(row, dict)
            or row.get("classification") != CLASSIFICATION
            or row.get("gate_eligible") is not False
            for row in completions
        )
    ):
        raise ValueError("pre-correction diagnostic manifest differs")


def compare_fresh_group_to_prior(
    *,
    prior_manifest: Mapping[str, Any],
    fresh_group: Mapping[str, Any],
) -> dict[str, object]:
    """Compare one fresh generic group with all old fields that actually persisted."""

    verify_prior_diagnostic_manifest(prior_manifest)
    position = fresh_group.get("schedule_position")
    if not isinstance(position, int) or isinstance(position, bool):
        raise ValueError("fresh group schedule position differs")
    if position <= PRIOR_DURABLE_GROUPS:
        durable_groups = prior_manifest.get("durable_groups")
        if not isinstance(durable_groups, list) or len(durable_groups) != PRIOR_DURABLE_GROUPS:
            raise ValueError("prior durable group manifest differs")
        prior = durable_groups[position - 1]
        if not isinstance(prior, dict) or prior.get("schedule_position") != position:
            raise ValueError("prior durable group position differs")
        expected_hashes = prior.get("persisted_scientific_field_sha256s")
        if not isinstance(expected_hashes, dict):
            raise ValueError("prior scientific field hashes differ")
        comparisons = {
            field: {
                "prior_sha256": expected_hashes.get(field),
                "fresh_sha256": canonical_sha256(fresh_group.get(field)),
                "exact": expected_hashes.get(field) == canonical_sha256(fresh_group.get(field)),
            }
            for field in CONTINUITY_FIELDS
        }
        passed = all(cast(bool, row["exact"]) for row in comparisons.values())
        status = "exact" if passed else "scientific_replay_drift"
        available_fields = list(CONTINUITY_FIELDS)
        unavailable_fields: list[str] = []
    elif position == PRIOR_GENERATED_GROUPS:
        comparisons = {}
        passed = True
        status = "no_prior_persisted_group_record"
        available_fields = []
        unavailable_fields = list(CONTINUITY_FIELDS)
    else:
        comparisons = {}
        passed = True
        status = "not_part_of_prior_partial_audit"
        available_fields = []
        unavailable_fields = []
    result: dict[str, object] = {
        "schema_version": 1,
        "comparison_id": COMPARISON_ID,
        "schedule_position": position,
        "prior_classification": CLASSIFICATION if position <= PRIOR_GENERATED_GROUPS else None,
        "prior_fields_available": available_fields,
        "prior_fields_unavailable": unavailable_fields,
        "field_comparisons": comparisons,
        "advantage_fields_compared_under_separate_equivalence_contract": list(
            ADVANTAGE_FIELDS_EXCLUDED_FROM_EXACT_CONTINUITY
        ),
        "status": status,
        "passed": passed,
        "failure_classification": None if passed else "scientific_replay_drift",
    }
    result["continuity_comparison_sha256"] = canonical_sha256(result)
    return result
