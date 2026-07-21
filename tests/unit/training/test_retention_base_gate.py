from __future__ import annotations

import json
from pathlib import Path

from foundry.training.config import canonical_sha256
from foundry.training.retention import load_suite
from foundry.training.retention_base_gate import assess_base_suite


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    items = []
    rows = []
    classifications = {}
    for section in ("arithmetic", "format", "instruction"):
        for index in range(100):
            item_id = f"{section}-{index:03d}"
            kind = "numeric_terminal" if section == "arithmetic" else "exact_text"
            items.append(
                {
                    "id": item_id,
                    "section": section,
                    "skill": "fixture",
                    "kind": kind,
                    "prompt": f"Original {section} fixture {index}.",
                    "expected": "1",
                }
            )
            correct = index < 89 if section == "instruction" else True
            rows.append(
                {
                    "id": item_id,
                    "section": section,
                    "skill": "fixture",
                    "response": "1",
                    "response_sha256": "a" * 64,
                    "score": {
                        "correct": correct,
                        "extractable": True,
                        "malformed": False,
                        "prompt_echo": False,
                        "question_generation": False,
                        "exact_format": correct,
                        "extracted_hash": "b" * 64,
                    },
                }
            )
            if not correct:
                classifications[item_id] = "genuine_instruction_noncompliance"
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "foundry-retention-adjudication-v2",
                "system_prompt": "Follow the instruction.",
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": 32,
                    "seed": 20260720,
                },
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    suite = load_suite(suite_path)
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(rows), encoding="utf-8")
    summary = {
        "adapter_sha256": None,
        "suite_sha256": suite.suite_sha256,
        "extractable": 300,
        "malformed_outputs": 0,
        "backend_failures": 0,
        "prompt_echo": 0,
        "question_generation": 0,
        "raw_packet_sha256": "c" * 64,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    classifications_path = tmp_path / "classifications.json"
    classifications_path.write_text(
        json.dumps({"classifications": classifications}), encoding="utf-8"
    )
    return suite_path, summary_path, raw_path, classifications_path


def test_base_gate_fails_closed_on_one_section(tmp_path: Path) -> None:
    suite, summary, raw, classifications = _write_fixture(tmp_path)
    result = assess_base_suite(
        suite_path=suite,
        summary_path=summary,
        raw_path=raw,
        classifications_path=classifications,
    )
    assert result["section_correct"] == {
        "arithmetic": 100,
        "format": 100,
        "instruction": 89,
    }
    assert result["gate_checks"]["instruction_at_least_90"] is False
    assert result["gate_passed"] is False
    assert result["decision"] == "STOP_BEFORE_ADAPTER_ADJUDICATION"
