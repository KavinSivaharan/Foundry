from __future__ import annotations

from pathlib import Path
from typing import Any

from foundry.phase2 import l3_grpo_signal_correction_prepare as prepare
from foundry.training.config import canonical_sha256


def test_correction_freeze_binds_all_pre_generation_contracts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(prepare, "_require_freeze_boundary", lambda _root: None)
    monkeypatch.setattr(
        prepare,
        "_file_rows",
        lambda _root, paths: [{"path": path, "bytes": 1, "sha256": "a" * 64} for path in paths],
    )
    monkeypatch.setattr(
        prepare,
        "_original_evidence",
        lambda _root: {
            "original_signal_audit_contract_sha256": "b" * 64,
            "original_signal_audit_implementation_sha256": "c" * 64,
            "published_blocker_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        prepare,
        "_frozen_inputs",
        lambda _root: {
            "generic_schedule_sha256": "e" * 64,
            "targeted_schedule_sha256": "f" * 64,
            "shared_replay_sha256": "1" * 64,
            "paired_schedule_sha256": "2" * 64,
        },
    )
    prior = {"prior_diagnostic_manifest_sha256": "3" * 64}
    monkeypatch.setattr(prepare, "build_prior_diagnostic_manifest", lambda _root: prior)
    fixture = {
        "fixture_vector_count": 65_536,
        "exact_mismatch_vector_count": 11_915,
        "maximum_absolute_difference": 1.1920928955078125e-07,
        "absolute_tolerance": 2.384185791015625e-07,
        "relative_tolerance": 1e-6,
        "all_values_within_tolerance": True,
    }
    fixture["fixture_sha256"] = canonical_sha256(fixture)
    monkeypatch.setattr(
        prepare,
        "exhaustive_cross_device_fixture",
        lambda _torch: fixture,
    )
    values = prepare.freeze_correction(tmp_path, torch_module=object())
    frozen_prior, advantage, family, implementation, contract = values
    assert frozen_prior is prior
    assert contract["prior_diagnostic_manifest_sha256"] == "3" * 64
    assert (
        contract["advantage_equivalence_contract_sha256"]
        == advantage["advantage_equivalence_contract_sha256"]
    )
    assert (
        contract["family_aggregation_contract_sha256"]
        == family["family_aggregation_contract_sha256"]
    )
    assert contract["implementation_sha256"] == implementation["implementation_sha256"]
    assert contract["optimizer_creation_authorized"] is False
    assert contract["counted_training_authorized"] is False
    supplied = contract["signal_audit_contract_sha256"]
    payload = dict(contract)
    payload.pop("signal_audit_contract_sha256")
    assert supplied == canonical_sha256(payload)


def test_write_new_or_identical_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    prepare._write_new_or_identical(path, {"value": 1})
    prepare._write_new_or_identical(path, {"value": 1})
    try:
        prepare._write_new_or_identical(path, {"value": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("changed correction contract was overwritten")
