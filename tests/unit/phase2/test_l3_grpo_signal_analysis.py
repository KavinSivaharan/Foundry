from __future__ import annotations

from typing import Any

from foundry.phase2 import l3_grpo_signal_analysis as analysis
from foundry.training.config import canonical_sha256


def _hashed(value: dict[str, Any], key: str) -> dict[str, Any]:
    value[key] = canonical_sha256(value)
    return value


def test_publication_binds_two_verified_arm_audits(monkeypatch: Any) -> None:
    raw_by_arm: dict[str, dict[str, Any]] = {}
    runtime_by_arm: dict[str, dict[str, Any]] = {}
    file_hashes = {"generic": "a" * 64, "targeted": "b" * 64}
    for arm in ("generic", "targeted"):
        raw_by_arm[arm] = _hashed(
            {
                "arm": arm,
                "source_commit": "c" * 40,
                "signal_audit_contract_sha256": "d" * 64,
                "advantage_equivalence_contract_sha256": "e" * 64,
                "prior_diagnostic_manifest_sha256": "f" * 64,
                "groups": [
                    {
                        "prior_partial_continuity": {
                            "passed": True,
                            "status": (
                                "exact"
                                if arm == "generic" and position <= 12
                                else (
                                    "no_prior_persisted_group_record"
                                    if arm == "generic" and position == 13
                                    else (
                                        "not_part_of_prior_partial_audit"
                                        if arm == "generic"
                                        else "not_applicable_to_targeted_arm"
                                    )
                                )
                            ),
                            "prior_fields_available": [],
                            "prior_fields_unavailable": [],
                        },
                        "advantage_equivalence": {"passed": True},
                        "maximum_cpu_cuda_advantage_difference": 0.0,
                    }
                    for position in range(1, 33)
                ],
            },
            "raw_audit_sha256",
        )
        runtime_by_arm[arm] = _hashed(
            {
                "arm": arm,
                "source_commit": "c" * 40,
                "signal_audit_contract_sha256": "d" * 64,
                "advantage_equivalence_contract_sha256": "e" * 64,
                "prior_diagnostic_manifest_sha256": "f" * 64,
                "raw_evidence_file_sha256": file_hashes[arm],
                "runtime_seconds": 10.0,
                "model_load_seconds": 2.0,
                "generation_seconds": 7.0,
                "peak_allocated_vram_bytes": 100,
                "peak_reserved_vram_bytes": 200,
                "peak_process_rss_bytes": 300,
                "completion_tokens": 400,
                "output_disk_bytes": 500,
            },
            "summary_sha256",
        )
    monkeypatch.setattr(
        analysis,
        "build_signal_summary",
        lambda *_args: {"decision": "reward_signal_insufficient"},
    )
    publication = analysis.build_publication(
        raw_by_arm=raw_by_arm,
        runtime_by_arm=runtime_by_arm,
        raw_file_sha256_by_arm=file_hashes,
    )
    supplied = publication["publication_sha256"]
    payload = dict(publication)
    payload.pop("publication_sha256")
    assert supplied == canonical_sha256(payload)
    assert publication["resource_usage"] == {
        "runtime_seconds": 20.0,
        "model_load_seconds": 4.0,
        "generation_seconds": 14.0,
        "peak_allocated_vram_bytes": 100,
        "peak_reserved_vram_bytes": 200,
        "peak_process_rss_bytes": 300,
        "completion_tokens": 800,
        "raw_output_disk_bytes": 1000,
    }
    assert publication["counted_training_status"] == "not_run"
