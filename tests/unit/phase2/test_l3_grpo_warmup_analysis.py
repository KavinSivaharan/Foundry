from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.phase2 import l3_grpo_warmup_analysis as analysis
from foundry.training.config import canonical_sha256


def _write_hashed(path: Path, value: dict[str, object], key: str) -> None:
    value[key] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_development_selection_uses_latest_common_passing_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_hashed(
        tmp_path / analysis.TRACKED_ROOT / analysis.TRAINING_OUTPUT,
        {"both_arms_passed": True},
        "training_result_sha256",
    )
    for checkpoint in analysis.CHECKPOINTS:
        for arm in analysis.ARMS:
            for suite in analysis.DEVELOPMENT_SUITES:
                passed = checkpoint in (8, 16)
                assessment = {
                    "backend_failures": 0,
                    "gate_passed": passed,
                    "preserved": 100,
                    "total": 100,
                    "overall_preservation": 1.0,
                    "overall_wilson_95_lower_bound": 0.96,
                    "section_preservation": {},
                    "maximum_instruction_family_adapter_only_failures": 0,
                    "prompt_echo": 0,
                    "question_generation": 0,
                }
                _write_hashed(
                    tmp_path
                    / analysis.RAW_ROOT
                    / "development"
                    / f"checkpoint-{checkpoint}/{arm}/{suite}/assessment.json",
                    assessment,
                    "summary_sha256",
                )
    monkeypatch.setattr(
        analysis,
        "directory_sha256",
        lambda path: f"{path.parts[-3]}-{path.parts[-2]}",
    )
    result = analysis.build_development_selection(tmp_path)
    assert result["latest_common_passing_checkpoint"] == 16
    assert result["development_retention_passed"] is True
    assert result["holdout_v2_authorized"] is True
    assert "holdout_v2" in result["excluded_selection_signals"]
    assert "gsm1k" in result["excluded_selection_signals"]
