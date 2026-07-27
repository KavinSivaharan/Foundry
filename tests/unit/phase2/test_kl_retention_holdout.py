import hashlib
import json
from pathlib import Path

import pytest

from foundry.phase2.kl_retention_holdout import (
    EXPECTED_COUNTS,
    INSTRUMENT_ID,
    assess_adapter,
    freeze_base_record,
    load_suite,
    validate_integrity,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256
from foundry.training.retention import score_response


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _suite(path: Path) -> None:
    items: list[dict[str, str]] = []
    for section in ("arithmetic", "format", "instruction"):
        for index in range(120):
            expected = "1" if section == "arithmetic" else f"ok-{section}-{index:03d}"
            items.append(
                {
                    "id": f"test-{section}-{index:03d}",
                    "section": section,
                    "skill": f"{section}-skill-{index % 6}",
                    "kind": "numeric_terminal" if section == "arithmetic" else "exact_text",
                    "prompt": f"Test-only {section} prompt number {index:03d}.",
                    "expected": expected,
                }
            )
    _write_json(
        path,
        {
            "schema_version": 1,
            "suite_id": INSTRUMENT_ID,
            "system_prompt": "Test-only system prompt.",
            "generation": {
                "do_sample": False,
                "max_new_tokens": 96,
                "seed": 20260720,
            },
            "items": items,
        },
    )


def _integrity(path: Path, suite_path: Path) -> None:
    suite = load_suite(suite_path)
    overlap = {
        name: {
            "source_prompt_count": 1,
            "exact_overlap": 0,
            "normalized_exact_overlap": 0,
            "contiguous_12_token_overlap": 0,
        }
        for name in (
            "vetted_curriculum_400",
            "replay_prompts_83",
            "gsm1k_development_904",
            "previous_retention_prompts",
            "prior_calibration_prompts",
        )
    }
    audit = {
        "schema_version": 1,
        "audit_id": "foundry-kl-independent-retention-integrity-v1",
        "suite_sha256": suite.suite_sha256,
        "candidate_count": 360,
        "category_counts": EXPECTED_COUNTS,
        "candidate_exact_duplicates": 0,
        "candidate_normalized_duplicates": 0,
        "candidate_prompt_hashes": [
            {
                "id": item.item_id,
                "sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
            }
            for item in suite.items
        ],
        "candidate_scorer_hashes": [
            {
                "id": item.item_id,
                "sha256": canonical_sha256(
                    {
                        "kind": item.kind,
                        "expected": item.expected,
                        "scorer": "foundry.training.retention.score_response",
                    }
                ),
            }
            for item in suite.items
        ],
        "reference_self_score_failures": 0,
        "ambiguous_or_subjective_scorers": 0,
        "llm_judge_used": False,
        "overlap_sources": overlap,
        "sealed_paths_accessed": False,
    }
    audit["integrity_audit_sha256"] = canonical_sha256(audit)
    _write_json(path, audit)


def _result_files(
    raw_path: Path,
    summary_path: Path,
    suite_path: Path,
    *,
    adapter_sha256: str | None,
    subset_sha256: str | None = None,
) -> None:
    suite = load_suite(suite_path)
    rows = [
        {
            "id": item.item_id,
            "section": item.section,
            "skill": item.skill,
            "response": item.expected,
            "response_sha256": hashlib.sha256(item.expected.encode()).hexdigest(),
            "score": score_response(item, item.expected),
        }
        for item in suite.items
    ]
    _write_json(raw_path, rows)
    section_metrics = {
        section: {"total": 120, "correct": 120, "accuracy": 1.0}
        for section in ("arithmetic", "format", "instruction")
    }
    summary = {
        "schema_version": 1,
        "evaluation_id": "test",
        "suite_sha256": suite.suite_sha256,
        "adapter_sha256": adapter_sha256,
        "subset_sha256": subset_sha256,
        "section_metrics": section_metrics,
        "total": 360,
        "correct": 360,
        "extractable": 360,
        "exact_format": 360,
        "prompt_echo": 0,
        "question_generation": 0,
        "malformed_outputs": 0,
        "backend_failures": 0,
        "runtime_seconds": 1.0,
        "load_seconds": 1.0,
        "input_tokens": 360,
        "output_tokens": 360,
        "peak_vram_allocated_bytes": 1,
        "peak_vram_reserved_bytes": 1,
        "peak_process_rss_bytes": 1,
        "raw_packet_sha256": file_sha256(raw_path),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json(summary_path, summary)


def test_freezes_content_free_base_correct_subset(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    integrity_path = tmp_path / "integrity.json"
    raw_path = tmp_path / "raw.json"
    summary_path = tmp_path / "summary.json"
    _suite(suite_path)
    _integrity(integrity_path, suite_path)
    _result_files(raw_path, summary_path, suite_path, adapter_sha256=None)

    subset, record = freeze_base_record(
        suite_path=suite_path,
        integrity_path=integrity_path,
        base_raw_path=raw_path,
        base_summary_path=summary_path,
    )

    assert subset["section_counts"] == EXPECTED_COUNTS
    assert subset["total"] == 360
    assert record["decision"] == "holdout_frozen_before_adapter_exposure"
    assert record["adapter_exposure_before_freeze"] is False


def test_rejects_nonzero_overlap(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    integrity_path = tmp_path / "integrity.json"
    _suite(suite_path)
    _integrity(integrity_path, suite_path)
    audit = json.loads(integrity_path.read_text(encoding="utf-8"))
    audit["overlap_sources"]["replay_prompts_83"]["exact_overlap"] = 1
    audit["integrity_audit_sha256"] = canonical_sha256(
        {key: value for key, value in audit.items() if key != "integrity_audit_sha256"}
    )
    _write_json(integrity_path, audit)

    with pytest.raises(ValueError, match="overlap gate"):
        validate_integrity(integrity_path, load_suite(suite_path))


def test_adapter_assessment_uses_unchanged_gate(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    integrity_path = tmp_path / "integrity.json"
    base_raw = tmp_path / "base_raw.json"
    base_summary = tmp_path / "base_summary.json"
    _suite(suite_path)
    _integrity(integrity_path, suite_path)
    _result_files(base_raw, base_summary, suite_path, adapter_sha256=None)
    subset, _ = freeze_base_record(
        suite_path=suite_path,
        integrity_path=integrity_path,
        base_raw_path=base_raw,
        base_summary_path=base_summary,
    )
    subset_path = tmp_path / "subset.json"
    _write_json(subset_path, subset)
    adapter_raw = tmp_path / "adapter_raw.json"
    adapter_summary = tmp_path / "adapter_summary.json"
    _result_files(
        adapter_raw,
        adapter_summary,
        suite_path,
        adapter_sha256="a" * 64,
        subset_sha256=str(subset["subset_sha256"]),
    )

    result = assess_adapter(
        suite_path=suite_path,
        subset_path=subset_path,
        raw_path=adapter_raw,
        summary_path=adapter_summary,
    )

    assert result["gate_passed"] is True
    assert result["overall_preservation"] == 1.0
