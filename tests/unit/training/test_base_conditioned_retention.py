import json
from pathlib import Path

import pytest

from foundry.training.base_conditioned_retention import (
    assess_holdout_instrument_usability,
    assess_preservation,
    build_pair_decision,
    freeze_base_correct_subset,
    wilson_lower_bound,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256
from foundry.training.retention import load_base_conditioned_subset, load_suite

SUITE_PATH = Path("configs/training/assistant_only_v3_retention_suite.json")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(*, suite_hash: str, raw_path: Path, adapter_hash: str | None) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "evaluation_id": "fixture-evaluation",
        "suite_sha256": suite_hash,
        "adapter_sha256": adapter_hash,
        "total": 60,
        "extractable": 60,
        "extractability": 1.0,
        "prompt_echo": 0,
        "question_generation": 0,
        "malformed_outputs": 0,
        "backend_failures": 0,
        "raw_packet_sha256": file_sha256(raw_path),
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def _rows(
    correct_ids: set[str] | None = None, suite_path: Path = SUITE_PATH
) -> list[dict[str, object]]:
    suite = load_suite(suite_path)
    allowed = {item.item_id for item in suite.items} if correct_ids is None else correct_ids
    return [
        {
            "id": item.item_id,
            "section": item.section,
            "skill": item.skill,
            "response": "fixture",
            "response_sha256": "0" * 64,
            "score": {
                "correct": item.item_id in allowed,
                "extractable": True,
                "prompt_echo": False,
                "question_generation": False,
                "malformed": False,
            },
        }
        for item in suite.items
    ]


def test_freeze_subset_is_content_free_stable_and_tamper_evident(tmp_path: Path) -> None:
    suite = load_suite(SUITE_PATH)
    rows = _rows()
    raw_path = tmp_path / "base_raw.json"
    summary_path = tmp_path / "base_summary.json"
    _write_json(raw_path, rows)
    _write_json(
        summary_path,
        _summary(suite_hash=suite.suite_sha256, raw_path=raw_path, adapter_hash=None),
    )
    manifest = freeze_base_correct_subset(
        suite_path=SUITE_PATH,
        base_summary_path=summary_path,
        base_raw_path=raw_path,
        subset_id="fixture-base-correct",
    )
    assert manifest["total"] == 60
    assert all(set(item) == {"id", "section", "skill"} for item in manifest["items"])
    assert manifest == freeze_base_correct_subset(
        suite_path=SUITE_PATH,
        base_summary_path=summary_path,
        base_raw_path=raw_path,
        subset_id="fixture-base-correct",
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    load_base_conditioned_subset(manifest_path, suite)
    manifest["total"] = 59
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="identity or hash differs"):
        load_base_conditioned_subset(manifest_path, suite)


def test_preservation_gate_and_instruction_family_concentration(tmp_path: Path) -> None:
    suite_root = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    instruction_items = [item for item in suite_root["items"] if item["section"] == "instruction"]
    for item in instruction_items[:4]:
        item["skill"] = "shared-fixture-family"
    suite_path = tmp_path / "suite.json"
    _write_json(suite_path, suite_root)
    suite = load_suite(suite_path)
    base_raw = tmp_path / "base_raw.json"
    base_summary = tmp_path / "base_summary.json"
    _write_json(base_raw, _rows(suite_path=suite_path))
    _write_json(
        base_summary,
        _summary(suite_hash=suite.suite_sha256, raw_path=base_raw, adapter_hash=None),
    )
    manifest = freeze_base_correct_subset(
        suite_path=suite_path,
        base_summary_path=base_summary,
        base_raw_path=base_raw,
        subset_id="fixture-base-correct",
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    adapter_raw = tmp_path / "adapter_raw.json"
    adapter_summary = tmp_path / "adapter_summary.json"
    rows = _rows(suite_path=suite_path)
    _write_json(adapter_raw, rows)
    summary = _summary(
        suite_hash=suite.suite_sha256,
        raw_path=adapter_raw,
        adapter_hash="a" * 64,
    )
    summary["base_conditioned_subset_sha256"] = manifest["subset_sha256"]
    summary["summary_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    _write_json(adapter_summary, summary)
    result = assess_preservation(
        suite_path=suite_path,
        subset_manifest_path=manifest_path,
        summary_path=adapter_summary,
        raw_path=adapter_raw,
    )
    assert result["gate_passed"] is True
    assert result["preserved"] == 60
    assert result["overall_wilson_95_lower_bound"] > 0.93

    state_hash = "c" * 64
    summary["adapter_scale"] = 0.75
    summary["adapter_scale_evidence"] = {
        "implementation_id": "foundry-common-lora-runtime-scaling-v1",
        "scale": 0.75,
        "adapter_state_sha256_before": state_hash,
        "adapter_state_sha256_after": state_hash,
        "base_parameter_signature_before": state_hash,
        "base_parameter_signature_after": state_hash,
        "original_scaling_restored": True,
        "adapter_state_unchanged": True,
        "base_parameter_signature_unchanged": True,
    }
    summary["summary_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    _write_json(adapter_summary, summary)
    scaled = assess_preservation(
        suite_path=suite_path,
        subset_manifest_path=manifest_path,
        summary_path=adapter_summary,
        raw_path=adapter_raw,
    )
    assert scaled["adapter_scale"] == 0.75
    assert scaled["state_restoration_verified"] is True

    instruction_skill = "shared-fixture-family"
    failures = {
        item.item_id
        for item in suite.items
        if item.section == "instruction" and item.skill == instruction_skill
    }
    assert len(failures) > 3
    failed_rows = _rows({item.item_id for item in suite.items} - failures, suite_path=suite_path)
    _write_json(adapter_raw, failed_rows)
    summary["raw_packet_sha256"] = file_sha256(adapter_raw)
    summary["summary_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    _write_json(adapter_summary, summary)
    failed = assess_preservation(
        suite_path=suite_path,
        subset_manifest_path=manifest_path,
        summary_path=adapter_summary,
        raw_path=adapter_raw,
    )
    assert failed["gate_passed"] is False
    assert failed["gate_checks"]["instruction_family_adapter_only_failures_at_most_3"] is False


def test_wilson_and_pair_decision_contracts() -> None:
    assert 0.89 < wilson_lower_bound(181, 187) < 0.95
    assessments = [
        {
            "adapter_sha256": adapter,
            "suite_id": suite,
            "summary_sha256": (adapter + suite).encode().hex()[:64].ljust(64, "0"),
            "gate_passed": True,
        }
        for adapter in ("a" * 64, "b" * 64)
        for suite in ("suite-a", "suite-b")
    ]
    decision = build_pair_decision(assessments)
    assert decision["gsm1k_authorized"] is True
    assessments[0]["gate_passed"] = False
    assert build_pair_decision(assessments)["sft_adaptation_line_stopped"] is True


def test_holdout_usability_gate_uses_sample_size_and_reference_integrity(
    tmp_path: Path,
) -> None:
    root = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    source_by_section = {
        section: [item for item in root["items"] if item["section"] == section]
        for section in ("arithmetic", "format", "instruction")
    }
    items = []
    for section, source in source_by_section.items():
        for index in range(100):
            item = dict(source[index % len(source)])
            item["id"] = f"{section}-{index:03d}"
            items.append(item)
    root["suite_id"] = "foundry-retention-anchor-holdout-v1"
    root["items"] = items
    suite_path = tmp_path / "holdout.json"
    _write_json(suite_path, root)
    suite = load_suite(suite_path)
    summary: dict[str, object] = {
        "adapter_sha256": None,
        "suite_sha256": suite.suite_sha256,
        "total": 300,
        "section_metrics": {
            "arithmetic": {"correct": 40},
            "format": {"correct": 41},
            "instruction": {"correct": 69},
        },
        "extractable": 290,
        "prompt_echo": 0,
        "question_generation": 0,
        "malformed_outputs": 0,
        "backend_failures": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, summary)
    evidence = {
        "suites": {"anchor_holdout": {"suite_sha256": suite.suite_sha256}},
        "cross_artifact": {"ambiguous_reference_answers": 0},
        "summary_sha256": "e" * 64,
    }
    evidence_path = tmp_path / "evidence.json"
    _write_json(evidence_path, evidence)
    result = assess_holdout_instrument_usability(
        suite_path=suite_path,
        base_summary_path=summary_path,
        artifact_evidence_path=evidence_path,
    )
    assert result["gate_passed"] is True
    assert result["overall_correct"] == 150
    assert result["reference_self_score_failures"] == 0


def test_scale_final_holdout_uses_predeclared_larger_gate(tmp_path: Path) -> None:
    root = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    source_by_section = {
        section: [item for item in root["items"] if item["section"] == section]
        for section in ("arithmetic", "format", "instruction")
    }
    items = []
    for section, source in source_by_section.items():
        for index in range(150):
            item = dict(source[index % len(source)])
            item["id"] = f"scale-{section}-{index:03d}"
            items.append(item)
    root["suite_id"] = "foundry-retention-scale-final-holdout-v1"
    root["items"] = items
    suite_path = tmp_path / "scale_holdout.json"
    _write_json(suite_path, root)
    suite = load_suite(suite_path)
    summary: dict[str, object] = {
        "adapter_sha256": None,
        "suite_sha256": suite.suite_sha256,
        "total": 450,
        "section_metrics": {
            "arithmetic": {"correct": 60},
            "format": {"correct": 70},
            "instruction": {"correct": 120},
        },
        "extractable": 400,
        "prompt_echo": 0,
        "question_generation": 0,
        "malformed_outputs": 0,
        "backend_failures": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, summary)
    evidence = {
        "suite_sha256": suite.suite_sha256,
        "ambiguous_reference_answers": 0,
        "summary_sha256": "f" * 64,
    }
    evidence_path = tmp_path / "evidence.json"
    _write_json(evidence_path, evidence)
    result = assess_holdout_instrument_usability(
        suite_path=suite_path,
        base_summary_path=summary_path,
        artifact_evidence_path=evidence_path,
    )
    assert result["gate_passed"] is True
    assert result["overall_correct"] == 250
    assert result["gate_checks"]["arithmetic_at_least_60"] is True
