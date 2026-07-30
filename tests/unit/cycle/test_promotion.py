from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.cycle.contract import CycleContractError
from foundry.cycle.controller import (
    assert_active_preserved,
    copy_adapter_once,
    evaluate_promotion_gate,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256


def _gate_inputs() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    development = {"passed": True}
    holdout = {"passed": True}
    benchmark = {
        "correct": 522,
        "extractability": 0.92,
        "backend_failures": 0,
        "category_effects": {"untargeted_gate_passed": True},
    }
    training = {"offline_reload": True, "base_parameters_unchanged": True}
    contract = {
        "minimum_candidate_correct": 522,
        "untouched_base_correct": 521,
        "targeted_l3_correct": 519,
        "minimum_extractability": 0.9138,
    }
    return development, holdout, benchmark, training, contract


def test_exact_522_count_passes_complete_promotion_gate() -> None:
    development, holdout, benchmark, training, contract = _gate_inputs()

    conditions = evaluate_promotion_gate(
        development=development,
        holdout=holdout,
        benchmark=benchmark,
        training=training,
        contract=contract,
    )

    assert len(conditions) == 11
    assert all(conditions.values())


def test_521_count_rejects_promotion() -> None:
    development, holdout, benchmark, training, contract = _gate_inputs()
    benchmark["correct"] = 521

    conditions = evaluate_promotion_gate(
        development=development,
        holdout=holdout,
        benchmark=benchmark,
        training=training,
        contract=contract,
    )

    assert conditions["candidate_at_least_522"] is False
    assert conditions["candidate_greater_than_base"] is False


def test_duplicate_promotion_target_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "adapter.safetensors").write_bytes(b"candidate")
    expected = directory_sha256(candidate)
    destination = tmp_path / "registry/model"

    copy_adapter_once(candidate, destination, expected)

    assert directory_sha256(destination) == expected
    with pytest.raises(CycleContractError, match="duplicate"):
        copy_adapter_once(candidate, destination, expected)


def test_active_model_is_preserved_after_rejection(tmp_path: Path) -> None:
    active_path = tmp_path / "active_model.json"
    active = {"logical_model_id": "untouched-base", "adapter_sha256": None}
    active_path.write_text(
        json.dumps(active, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expected = canonical_sha256(active)

    assert assert_active_preserved(active_path, expected) == active
    active_path.write_text(
        json.dumps({"logical_model_id": "changed"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CycleContractError, match="changed"):
        assert_active_preserved(active_path, expected)
