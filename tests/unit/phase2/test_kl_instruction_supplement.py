from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import foundry.phase2.kl_instruction_supplement as supplement
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256
from foundry.training.retention import RetentionItem, score_response

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r2_supplement_contract.json"
)
MANIFEST_PATH = (
    ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r2_supplement_manifest.json"
)
COMBINED_SUBSET_PATH = (
    ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r2_combined_retention_subset.json"
)
COMBINED_RECORD_PATH = (
    ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r2_combined_holdout.json"
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _supplement_files(root: Path, *, correct: int) -> dict[str, Path]:
    contract = supplement.load_contract(CONTRACT_PATH)
    items = supplement.construct_items(contract)
    suite: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": supplement.SUPPLEMENT_ID,
        "supplement_contract_sha256": contract["supplement_contract_sha256"],
        "system_prompt": supplement.SYSTEM_PROMPT,
        "generation": supplement.GENERATION,
        "items": items,
    }
    suite["suite_sha256"] = supplement._suite_hash(suite)  # noqa: SLF001
    suite_path = root / "supplement.json"
    _write(suite_path, suite)
    overlap_sources = {
        name: {
            "source_prompt_count": 1,
            "exact_overlap": 0,
            "normalized_exact_overlap": 0,
            "contiguous_12_token_overlap": 0,
        }
        for name in contract["overlap_policy"]["source_groups"]
    }
    integrity: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-kl-instruction-supplement-integrity-v1",
        "supplement_contract_sha256": contract["supplement_contract_sha256"],
        "supplement_suite_sha256": suite["suite_sha256"],
        "candidate_count": 180,
        "family_counts": {family: 30 for family in supplement.FAMILY_IDS},
        "candidate_exact_duplicates": 0,
        "candidate_normalized_duplicates": 0,
        "candidate_prompt_hashes": [],
        "candidate_scorer_hashes": [],
        "reference_self_score_failures": 0,
        "defective_prompts": 0,
        "defective_references": 0,
        "defective_scorers": 0,
        "ambiguous_or_subjective_scorers": 0,
        "llm_judge_used": False,
        "overlap_sources": overlap_sources,
        "source_evidence": {},
        "sealed_paths_accessed": False,
        "adapter_outputs_accessed": False,
        "base_outputs_accessed_during_construction": False,
    }
    integrity["integrity_audit_sha256"] = canonical_sha256(integrity)
    integrity_path = root / "integrity.json"
    _write(integrity_path, integrity)
    rows = []
    for index, raw_item in enumerate(items):
        response = raw_item["expected"] if index < correct else "wrong"
        item = RetentionItem(
            item_id=raw_item["id"],
            section="instruction",
            skill=raw_item["skill"],
            kind="exact_text",
            prompt=raw_item["prompt"],
            expected=raw_item["expected"],
        )
        rows.append(
            {
                "id": raw_item["id"],
                "family": raw_item["family"],
                "response": response,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "score": score_response(item, response),
            }
        )
    raw_path = root / "supplement_raw.json"
    _write(raw_path, rows)
    family_counts = {
        family: sum(row["score"]["correct"] and row["family"] == family for row in rows)
        for family in supplement.FAMILY_IDS
    }
    gate_checks = {
        "base_correct_at_least_24": correct >= 24,
        "zero_backend_failures": True,
        "zero_defective_prompts": True,
        "zero_defective_references": True,
        "zero_defective_scorers": True,
        "prompt_echo_at_most_2_percent": (
            sum(row["score"]["prompt_echo"] for row in rows) / 180 <= 0.02
        ),
        "zero_question_generation": not any(row["score"]["question_generation"] for row in rows),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": "test",
        "supplement_contract_sha256": contract["supplement_contract_sha256"],
        "supplement_suite_sha256": suite["suite_sha256"],
        "integrity_audit_sha256": integrity["integrity_audit_sha256"],
        "base_revision": "test",
        "adapter_sha256": None,
        "total": 180,
        "correct": correct,
        "family_correct_counts": family_counts,
        "extractable": sum(row["score"]["extractable"] for row in rows),
        "exact_format": sum(row["score"]["exact_format"] for row in rows),
        "prompt_echo": sum(row["score"]["prompt_echo"] for row in rows),
        "question_generation": sum(row["score"]["question_generation"] for row in rows),
        "malformed_outputs": sum(row["score"]["malformed"] for row in rows),
        "backend_failures": 0,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "runtime_seconds": 1.0,
        "load_seconds": 1.0,
        "input_tokens": 180,
        "output_tokens": 180,
        "gpu_name": "test",
        "peak_vram_allocated_bytes": 1,
        "peak_vram_reserved_bytes": 1,
        "peak_process_rss_bytes": 1,
        "raw_packet_sha256": file_sha256(raw_path),
        "per_item_decision_sha256": canonical_sha256(
            [
                {
                    "id": row["id"],
                    "response_sha256": row["response_sha256"],
                    "score": row["score"],
                }
                for row in rows
            ]
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = root / "supplement_summary.json"
    _write(summary_path, summary)
    return {
        "suite": suite_path,
        "integrity": integrity_path,
        "raw": raw_path,
        "summary": summary_path,
    }


def _existing_files(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    items = [
        {
            "id": f"existing-{index:03d}",
            "section": (
                "arithmetic" if index < 120 else "format" if index < 240 else "instruction"
            ),
            "skill": "synthetic",
            "kind": "exact_text",
            "prompt": f"Synthetic existing prompt {index}.",
            "expected": "ok",
        }
        for index in range(360)
    ]
    suite: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": "synthetic",
        "system_prompt": supplement.SYSTEM_PROMPT,
        "generation": supplement.GENERATION,
        "items": items,
    }
    suite_hash = supplement._suite_hash(suite)  # noqa: SLF001
    suite_path = root / "existing_suite.json"
    _write(suite_path, suite)
    raw_path = root / "existing_raw.json"
    _write(raw_path, [{"id": item["id"], "score": {"correct": False}} for item in items])
    summary: dict[str, Any] = {"summary_sha256": "pending"}
    summary["summary_sha256"] = canonical_sha256({})
    summary_path = root / "existing_summary.json"
    _write(summary_path, summary)
    selected = (
        [
            {"id": f"existing-{index:03d}", "section": "arithmetic", "skill": "synthetic"}
            for index in range(79)
        ]
        + [
            {"id": f"existing-{120 + index:03d}", "section": "format", "skill": "synthetic"}
            for index in range(89)
        ]
        + [
            {
                "id": f"existing-{240 + index:03d}",
                "section": "instruction",
                "skill": "synthetic",
            }
            for index in range(36)
        ]
    )
    subset: dict[str, Any] = {
        "schema_version": 1,
        "instrument_id": "synthetic",
        "subset_id": "synthetic",
        "suite_sha256": suite_hash,
        "section_counts": {"arithmetic": 79, "format": 89, "instruction": 36},
        "total": 204,
        "items": selected,
    }
    subset["subset_sha256"] = canonical_sha256(subset)
    subset_path = root / "existing_subset.json"
    _write(subset_path, subset)
    monkeypatch.setattr(supplement, "EXPECTED_EXISTING_SUITE_SHA256", suite_hash)
    monkeypatch.setattr(
        supplement, "EXPECTED_EXISTING_BASE_SUMMARY_SHA256", summary["summary_sha256"]
    )
    monkeypatch.setattr(supplement, "EXPECTED_EXISTING_RAW_SHA256", file_sha256(raw_path))
    monkeypatch.setattr(supplement, "EXPECTED_EXISTING_SUBSET_SHA256", subset["subset_sha256"])
    return {
        "suite": suite_path,
        "raw": raw_path,
        "summary": summary_path,
        "subset": subset_path,
    }


def test_contract_reconstructs_and_freezes_six_families() -> None:
    contract = supplement.load_contract(CONTRACT_PATH)
    assert contract["supplement_contract_sha256"] == (
        "d481043e0ea0775aa27144c52000a6f35b965c4005dbf621d5f8a6cc9ba3acf0"
    )
    assert [row["quota"] for row in contract["family_order"]] == [30] * 6


def test_constructs_exactly_180_unique_self_scoring_items() -> None:
    contract = supplement.load_contract(CONTRACT_PATH)
    items = supplement.construct_items(contract)
    assert len(items) == 180
    assert len({item["id"] for item in items}) == 180
    assert len({item["prompt"] for item in items}) == 180
    assert {
        family: sum(item["family"] == family for item in items) for family in supplement.FAMILY_IDS
    } == {family: 30 for family in supplement.FAMILY_IDS}
    for raw in items:
        item = RetentionItem(
            item_id=raw["id"],
            section="instruction",
            skill=raw["skill"],
            kind="exact_text",
            prompt=raw["prompt"],
            expected=raw["expected"],
        )
        assert score_response(item, item.expected)["correct"] is True


def test_overlap_detects_exact_normalized_and_twelve_token_matches() -> None:
    source = ["One two three four five six seven eight nine ten eleven twelve source."]
    assert supplement.overlap(source, source)["exact_overlap"] == 1
    normalized = ["  ONE two three four five six seven eight nine ten eleven twelve source.  "]
    result = supplement.overlap(normalized, source)
    assert result["normalized_exact_overlap"] == 1
    assert result["contiguous_12_token_overlap"] == 1


def test_candidate_text_scan_detects_development_content(tmp_path: Path) -> None:
    reference = "One two three four five six seven eight nine ten eleven twelve source."
    clean = tmp_path / "clean.txt"
    copied = tmp_path / "copied.txt"
    clean.write_text("Independent publication record.", encoding="utf-8")
    copied.write_text(f"prefix {reference} suffix", encoding="utf-8")
    result = supplement.scan_candidate_text([clean, copied], [reference])
    assert result["candidate_count"] == 2
    assert result["exact_reference_hits"] == 1
    assert result["normalized_reference_hits"] == 1
    assert result["contiguous_12_token_reference_hits"] == 1


def test_suite_rejects_family_count_drift(tmp_path: Path) -> None:
    contract = supplement.load_contract(CONTRACT_PATH)
    items = supplement.construct_items(contract)
    items[0]["family"] = supplement.FAMILY_IDS[1]
    suite: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": supplement.SUPPLEMENT_ID,
        "supplement_contract_sha256": contract["supplement_contract_sha256"],
        "system_prompt": supplement.SYSTEM_PROMPT,
        "generation": supplement.GENERATION,
        "items": items,
    }
    suite["suite_sha256"] = supplement._suite_hash(suite)  # noqa: SLF001
    path = tmp_path / "suite.json"
    _write(path, suite)
    with pytest.raises(ValueError, match="family counts"):
        supplement.load_supplement(path, contract)


def test_content_free_manifest_freezes_before_model_exposure(tmp_path: Path) -> None:
    files = _supplement_files(tmp_path, correct=24)
    manifest = supplement.freeze_supplement_manifest(
        contract_path=CONTRACT_PATH,
        suite_path=files["suite"],
        integrity_path=files["integrity"],
    )
    assert manifest["candidate_count"] == 180
    assert manifest["model_loads_before_freeze"] == 0
    assert manifest["adapter_exposure_before_freeze"] is False
    assert "items" not in manifest
    assert "expected" not in json.dumps(manifest)


def test_published_combined_holdout_reconstructs() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    subset = json.loads(COMBINED_SUBSET_PATH.read_text(encoding="utf-8"))
    record = json.loads(COMBINED_RECORD_PATH.read_text(encoding="utf-8"))
    supplement._validate_hash(manifest, "supplement_manifest_sha256")  # noqa: SLF001
    supplement._validate_hash(subset, "subset_sha256")  # noqa: SLF001
    supplement._validate_hash(record, "integrity_decision_sha256")  # noqa: SLF001
    assert manifest["existing_suite_classification"] == (
        "base_calibration_component_for_kl_holdout_v2"
    )
    assert manifest["supplement_suite_sha256"] == (
        "9dd87e3dd91dea1cda7d8d3185f3f08180917653eb26cfc0804c28d2a1607ec0"
    )
    assert record["decision"] == "combined_v2_holdout_frozen_before_adapter_exposure"
    assert record["combined_suite"]["candidate_count"] == 540
    assert record["combined_suite"]["category_counts"] == {
        "arithmetic": 120,
        "format": 120,
        "instruction": 300,
    }
    assert record["base_correct_subset"] == subset
    assert subset["section_counts"] == {
        "arithmetic": 79,
        "format": 89,
        "instruction": 149,
    }
    assert subset["total"] == 317
    assert subset["subset_sha256"] == (
        "a23b1014d92e9f98b74da3b29913a430bdaebf8e07a16b31b4c3dcc831f1f420"
    )
    assert all(record["gate_checks"].values())
    assert record["calibration_or_checkpoint_selection_use"] is False
    assert record["adapter_exposure_before_freeze"] is False
    assert record["sealed_paths_accessed"] is False


def test_untouched_base_result_replays_every_scorer_decision(tmp_path: Path) -> None:
    files = _supplement_files(tmp_path, correct=24)
    replay = supplement.replay_base_result(
        contract_path=CONTRACT_PATH,
        suite_path=files["suite"],
        integrity_path=files["integrity"],
        raw_path=files["raw"],
        summary_path=files["summary"],
    )
    assert replay["decision"] == "supplement_untouched_base_result_reconstructed"
    assert replay["correct"] == 24
    assert replay["gate_passed"] is True


@pytest.mark.parametrize(
    ("correct", "expected_decision"),
    (
        (24, "combined_v2_holdout_frozen_before_adapter_exposure"),
        (23, "combined_base_usability_blocker"),
    ),
)
def test_combined_union_uses_every_base_correct_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    correct: int,
    expected_decision: str,
) -> None:
    existing = _existing_files(tmp_path, monkeypatch)
    new = _supplement_files(tmp_path, correct=correct)
    subset, record = supplement.freeze_combined(
        contract_path=CONTRACT_PATH,
        existing_suite_path=existing["suite"],
        existing_raw_path=existing["raw"],
        existing_summary_path=existing["summary"],
        existing_subset_path=existing["subset"],
        supplement_suite_path=new["suite"],
        supplement_integrity_path=new["integrity"],
        supplement_raw_path=new["raw"],
        supplement_summary_path=new["summary"],
        combined_suite_path=tmp_path / "combined.json",
    )
    assert subset["total"] == 204 + correct
    assert subset["section_counts"]["instruction"] == 36 + correct
    assert subset["all_base_correct_items_included"] is True
    assert len(subset["items"]) == 204 + correct
    assert record["decision"] == expected_decision
