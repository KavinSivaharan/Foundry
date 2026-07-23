from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from foundry.phase2 import matching_repair
from foundry.phase2.matching_repair import (
    EXPECTED_BASELINE_SMD,
    RepairConfig,
    load_and_verify_inputs,
    run_repair,
)
from foundry.phase2.selection import _smd

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "results" / "raw" / "phase2_vetted_corpus"


def _live_paths() -> dict[str, Path]:
    paths = {
        "asdiv": RAW_ROOT / "contamination" / "clean_asdiv.jsonl",
        "asdiv_predictions": RAW_ROOT / "base_pool_evaluation" / "predictions.jsonl",
        "mathqa": RAW_ROOT / "mathqa_contamination" / "clean_mathqa.jsonl",
        "mathqa_predictions": RAW_ROOT / "mathqa_base_evaluation" / "predictions.jsonl",
        "baseline_targeted": RAW_ROOT / "selection" / "size_200" / "targeted_full.jsonl",
        "baseline_generic": RAW_ROOT / "selection" / "size_200" / "generic_full.jsonl",
        "baseline_selection": RAW_ROOT / "selection" / "size_200" / "selection_summary.json",
        "input_freeze_path": RAW_ROOT / "matching_repair" / "input_freeze.json",
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("ignored frozen Phase 2 repair inputs are not available")
    return paths


def test_aggregate_smd_matches_project_implementation() -> None:
    left = [float(value) for value in range(200)]
    right = [float(value + 1) for value in range(200)]
    assert matching_repair._aggregate_smd(  # noqa: SLF001
        int(sum(left)),
        int(sum(value * value for value in left)),
        int(sum(right)),
        int(sum(value * value for value in right)),
    ) == pytest.approx(_smd(left, right))


def test_changed_repair_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="frozen contract"):
        RepairConfig(maximum_smd=0.11).validate_frozen()
    with pytest.raises(ValueError, match="frozen contract"):
        RepairConfig(maximum_categorical_difference=0.051).validate_frozen()


def test_live_frozen_single_repair_and_replay(tmp_path: Path) -> None:
    paths = _live_paths()
    first = run_repair(**paths, output_root=tmp_path / "first")
    second = run_repair(**paths, output_root=tmp_path / "second")

    assert first == second
    assert first["selected_replacement"] == {
        "arm": "generic",
        "removed": "mathqa-train-26455",
        "added": "mathqa-train-28853",
    }
    assert first["checked_replacements"] == 155_301
    assert first["legal_replacements"] == 152_226
    assert first["passing_replacements"] == 1_979
    assert first["targeted_size"] == first["generic_size"] == 200
    assert (
        first["targeted_source_counts"]
        == first["generic_source_counts"]
        == {
            "ASDiv": 97,
            "MathQA": 103,
        }
    )
    assert first["targeted_family_counts"] == {
        "constraint_distribution_or_discrete_reasoning": 43,
        "multi_step_bookkeeping_or_omission": 110,
        "rate_ratio_percentage_or_average": 47,
    }
    assert first["generic_family_counts"] == {
        "constraint_distribution_or_discrete_reasoning": 66,
        "multi_step_bookkeeping_or_omission": 67,
        "rate_ratio_percentage_or_average": 67,
    }
    assert first["numerical_smd"] == {
        "question_token_count": 0.002889293374768125,
        "base_output_tokens": 0.00751647648517605,
        "formula_depth": 0.08708987147709694,
        "operation_count": 0.06805619983768806,
    }
    assert max(first["numerical_smd"].values()) <= 0.10  # type: ignore[union-attr]
    assert first["categorical_maximum"] == pytest.approx(0.05)
    assert first["matching_gate_passed"] is True
    assert first["contamination_count"] == 0
    assert first["cross_arm_source_id_overlap"] == 0
    assert first["cross_arm_exact_question_overlap"] == 0
    assert first["cross_arm_normalized_question_overlap"] == 0
    assert first["cross_arm_latent_program_overlap"] == 0
    assert first["cross_arm_near_duplicate_count"] == 0
    assert first["two_replacement_search_run"] is False
    assert first["global_fallback_run"] is False
    for name in (
        "targeted_full.jsonl",
        "generic_full.jsonl",
        "targeted_manifest.jsonl",
        "generic_manifest.jsonl",
        "exact_matches.jsonl",
        "repair_summary.json",
    ):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert "_tmp12e" not in inspect.getsource(matching_repair)

    frozen = load_and_verify_inputs(**paths)
    assert len(frozen.candidates) == 2_716
    assert frozen.input_freeze["candidate_covariates_sha256"] == (
        "1c1ad44d71b9658f8b7c969cd95fc572d8cb847089ec9a76067ad9f1eaa88d6f"
    )

    tampered_freeze = dict(frozen.input_freeze)
    tampered_freeze["candidate_covariates_sha256"] = "0" * 64
    tampered_freeze_path = tmp_path / "tampered-freeze.json"
    tampered_freeze_path.write_text(
        json.dumps(tampered_freeze, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="input-freeze content"):
        load_and_verify_inputs(**{**paths, "input_freeze_path": tampered_freeze_path})

    prediction_lines = paths["asdiv_predictions"].read_text(encoding="utf-8").splitlines()
    changed_prediction = json.loads(prediction_lines[0])
    changed_prediction["question_token_count"] += 1
    prediction_lines[0] = json.dumps(changed_prediction, sort_keys=True)
    changed_predictions_path = tmp_path / "changed-predictions.jsonl"
    changed_predictions_path.write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ASDiv base predictions"):
        load_and_verify_inputs(**{**paths, "asdiv_predictions": changed_predictions_path})

    targeted_lines = paths["baseline_targeted"].read_text(encoding="utf-8").splitlines()
    changed_targeted = json.loads(targeted_lines[0])
    changed_targeted["source_id"] = "tampered-source-id"
    targeted_lines[0] = json.dumps(changed_targeted, sort_keys=True)
    changed_targeted_path = tmp_path / "changed-targeted.jsonl"
    changed_targeted_path.write_text("\n".join(targeted_lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed targeted assignment"):
        load_and_verify_inputs(**{**paths, "baseline_targeted": changed_targeted_path})

    assert EXPECTED_BASELINE_SMD == {
        "question_token_count": 0.0,
        "base_output_tokens": 0.022589325494615863,
        "formula_depth": 0.11389459246177541,
        "operation_count": 0.10876528809635315,
    }
