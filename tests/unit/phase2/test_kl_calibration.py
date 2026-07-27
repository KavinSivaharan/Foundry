from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_calibration
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_PATH = (
    ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r3_historical_comparator.json"
)
CALIBRATION_PATH = ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r3_kl_calibration.json"
SELECTION_PATH = (
    ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r3_kl_coefficient_selection.json"
)
BLOCKER_PATH = (
    ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r3_kl_calibration_blocker.json"
)


def _historical() -> dict[str, Any]:
    value: object = json.loads(HISTORICAL_PATH.read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


def test_published_historical_comparator_self_hashes() -> None:
    value = _historical()
    assert value["objective_contract_sha256"] == (kl_calibration.OBJECTIVE_CONTRACT_SHA256)
    supplied = value.pop("historical_comparator_summary_sha256")
    assert supplied == canonical_sha256(value)


def test_historical_comparator_is_lambda_zero_without_updates() -> None:
    value = _historical()
    for arm in ("generic", "targeted"):
        row = value["arms"][arm]
        assert row["lambda_kl"] == 0
        assert row["optimizer_steps"] == 16
        assert row["loss_bearing_tokens"] == 16_000
        assert row["model_update_performed"] is False
        assert row["base_restoration"]
        assert row["adjudication_retention"]["passed"]
        assert row["anchor_retention"]["passed"]
    assert value["holdout_v2_use"] is False
    assert value["gsm1k_use"] is False


def test_calibration_contains_exact_complete_matrix() -> None:
    value = _read(CALIBRATION_PATH)
    supplied = value.pop("calibration_summary_sha256")
    assert supplied == canonical_sha256(value)
    assert value["coefficient_order"] == [0.01, 0.03, 0.1, 0.3]
    assert value["arm_order"] == ["generic", "targeted"]
    assert value["run_count"] == 8
    assert value["optimizer_step_count"] == 128
    assert value["loss_bearing_token_count"] == 128_000
    assert all(
        row["eligibility_criteria"]["both_development_retention_subsets_pass"]
        for row in value["runs"]
    )
    assert all(
        not row["eligibility_criteria"]["replay_token_kl_at_most_75_percent_historical"]
        for row in value["runs"]
    )


def test_selection_and_blocker_stop_before_full_training() -> None:
    selection = _read(SELECTION_PATH)
    supplied = selection.pop("coefficient_selection_sha256")
    assert supplied == canonical_sha256(selection)
    assert selection["selected_coefficient"] is None
    assert selection["decision"] == "no_common_eligible_coefficient"
    assert selection["stop_before_full_training"]
    assert all(not row["common_eligible"] for row in selection["coefficient_results"])
    blocker = _read(BLOCKER_PATH)
    blocker_hash = blocker.pop("calibration_blocker_sha256")
    assert blocker_hash == canonical_sha256(blocker)
    assert blocker["all_replay_kl_gates_failed"]
    assert blocker["full_training_runs"] == 0
    assert blocker["holdout_v2_adapter_evaluations"] == 0
    assert blocker["gsm1k_adapter_evaluations"] == 0
