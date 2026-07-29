from __future__ import annotations

from pathlib import Path
from typing import Any

from foundry.phase2 import l3_grpo_signal_prepare as prepare
from foundry.training.config import canonical_sha256


def test_freeze_binds_method_source_inputs_and_prohibitions(
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
        "_frozen_inputs",
        lambda _root: {
            "generic_schedule_sha256": "b" * 64,
            "targeted_schedule_sha256": "c" * 64,
            "shared_replay_sha256": "d" * 64,
            "paired_schedule_sha256": "e" * 64,
            "r1_compatibility_blocker_sha256": "f" * 64,
        },
    )
    implementation, contract = prepare.freeze_signal_audit(tmp_path)
    implementation_hash = implementation["implementation_sha256"]
    implementation_payload = dict(implementation)
    implementation_payload.pop("implementation_sha256")
    assert implementation_hash == canonical_sha256(implementation_payload)
    contract_hash = contract["signal_audit_contract_sha256"]
    contract_payload = dict(contract)
    contract_payload.pop("signal_audit_contract_sha256")
    assert contract_hash == canonical_sha256(contract_payload)
    assert contract["contract_id"] == "foundry-l3-grpo-signal-audit-v1"
    assert contract["implementation_sha256"] == implementation_hash
    assert contract["optimizer_creation_authorized"] is False
    assert contract["backward_authorized"] is False
    assert contract["adapter_mutation_authorized"] is False
    assert contract["counted_training_authorized"] is False
    assert "src/foundry/phase2/l3_grpo_signal_analysis.py" in prepare.IMPLEMENTATION_FILES


def test_write_new_or_identical_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    prepare._write_new_or_identical(path, {"value": 1})
    prepare._write_new_or_identical(path, {"value": 1})
    try:
        prepare._write_new_or_identical(path, {"value": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("changed frozen content was overwritten")
