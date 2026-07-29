from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.phase2 import l3_grpo_zero_gradient_compatibility as compatibility
from foundry.phase2.l3_grpo_zero_gradient import (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    NONZERO_GRADIENT_UPDATE,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256


def _hashed(value: dict[str, object], key: str) -> dict[str, object]:
    result = dict(value)
    result[key] = canonical_sha256(result)
    return result


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path) -> None:
    tracked = root / "results/phase2_vetted_corpus"
    correction = _hashed(
        {
            "classification_contract_sha256": "a" * 64,
        },
        "correction_contract_sha256",
    )
    implementation = _hashed(
        {"files": []},
        "corrected_implementation_sha256",
    )
    decision = _hashed(
        {"classification": EXPECTED_ZERO_ADVANTAGE_NOOP},
        "diagnostic_decision_sha256",
    )
    _write(tracked / "milestone14a_r1_correction_contract.json", correction)
    _write(
        tracked / "milestone14a_r1_corrected_implementation.json",
        implementation,
    )
    _write(tracked / "milestone14a_r1_zero_gradient_decision.json", decision)

    noop = _hashed(
        {"classification": EXPECTED_ZERO_ADVANTAGE_NOOP},
        "classification_evidence_sha256",
    )
    update = _hashed(
        {"classification": NONZERO_GRADIENT_UPDATE},
        "classification_evidence_sha256",
    )
    gate = {
        "passed": True,
        "zero_variance_group_count": 1,
        "nonzero_variance_group_count": 1,
        "expected_noop_group_count": 1,
        "nonzero_gradient_group_count": 1,
        "policy_update_count": 1,
        "optimizer_step_count": 2,
        "scheduler_step_count": 2,
    }
    packet = _hashed(
        {
            "classification_steps": [noop, update],
            "complete_smoke_gate": gate,
            "correction_contract_sha256": correction["correction_contract_sha256"],
            "corrected_implementation_sha256": implementation["corrected_implementation_sha256"],
            "classification_contract_sha256": correction["classification_contract_sha256"],
            "final_policy": {"normalized_tensor_state_sha256": "b" * 64},
            "initial_reference": {"normalized_tensor_state_sha256": "c" * 64},
            "final_reference": {"normalized_tensor_state_sha256": "c" * 64},
            "base_before": {"base_parameter_state_sha256": "d" * 64},
            "base_after": {"base_parameter_state_sha256": "d" * 64},
            "final_adapter": {"directory_sha256": "e" * 64},
        },
        "packet_sha256",
    )
    partial = _hashed(
        {
            "stage": "complete_smoke_gate_persisted",
            "error": None,
        },
        "partial_evidence_sha256",
    )
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a_r1"
    for index in (1, 2):
        run_root = raw / f"compatibility/run-{index}"
        partial_path = run_root / "partial_evidence.json"
        _write(partial_path, partial)
        summary = _hashed(
            {
                "gate_passed": True,
                "optimizer_steps": 2,
                "groups": 2,
                "completions": 8,
                "arm": "generic",
                "complete_smoke_gate": gate,
                "partial_evidence_file_sha256": file_sha256(partial_path),
                "exact_packet": packet,
                "reward": {"completion_tokens": 100},
                "peak_allocated_vram_bytes": 1,
                "peak_reserved_vram_bytes": 2,
                "peak_process_rss_bytes": 3,
                "output_disk_bytes": 4,
                "runtime_seconds": 5.0,
                "training_seconds": 6.0,
            },
            "summary_sha256",
        )
        _write(run_root / "summary.json", summary)
        _write(raw / f"compatibility/logs/run-{index}.stderr.txt", "")


def test_build_compatibility_requires_and_reports_exact_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(
        compatibility,
        "_require_clean_synchronized_main",
        lambda root: "f" * 40,
    )
    result = compatibility.build_compatibility_result(tmp_path)
    assert result["gate_passed"] is True
    assert result["exact_match"] is True
    assert result["official_smoke_runs"] == 2
    assert result["official_smoke_retries"] == 0
    assert result["expected_noop_group_count"] == 1
    assert result["policy_update_count"] == 1
    assert result["corrected_source_commit"] == "f" * 40


def test_build_compatibility_rejects_raw_difference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(
        compatibility,
        "_require_clean_synchronized_main",
        lambda root: "f" * 40,
    )
    path = (
        tmp_path / "results/raw/phase2_vetted_corpus/milestone14a_r1/"
        "compatibility/run-2/partial_evidence.json"
    )
    changed = _hashed(
        {
            "stage": "complete_smoke_gate_persisted",
            "error": None,
            "different": True,
        },
        "partial_evidence_sha256",
    )
    _write(path, changed)
    summary_path = path.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("summary_sha256")
    summary["partial_evidence_file_sha256"] = file_sha256(path)
    _write(summary_path, _hashed(summary, "summary_sha256"))
    with pytest.raises(RuntimeError, match="raw evidence differs"):
        compatibility.build_compatibility_result(tmp_path)
