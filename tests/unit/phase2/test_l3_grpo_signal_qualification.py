from __future__ import annotations

from typing import Any

import pytest

from foundry.phase2.l3_grpo_signal_qualification import (
    build_selection_and_gradient_decision,
)
from foundry.training.config import canonical_sha256


def _hashed(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = canonical_sha256(result)
    return result


def _summary(arm: str, run_index: int) -> dict[str, Any]:
    return _hashed(
        {
            "arm": arm,
            "run_index": run_index,
            "source_commit": "a" * 40,
            "task_group_id": f"{arm}-task",
            "gate_passed": True,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "exact_projection_sha256": arm[0] * 64,
            "policy_gradient_global_norm": 1.0,
            "combined_gradient_global_norm": 1.0,
            "policy_nonzero_gradient_tensor_count": 4,
            "reference_gradient_count": 0,
            "base_gradient_count": 0,
            "raw_projection_file_sha256": str(run_index) * 64,
        },
        "summary_sha256",
    )


def _fixtures() -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    selected = {
        arm: {
            "task_group_id": f"{arm}-task",
            "task_group_record_sha256": arm[0] * 64,
            "replay_group_id": f"{arm}-replay",
        }
        for arm in ("generic", "targeted")
    }
    contract = _hashed(
        {
            "selected_candidates": selected,
        },
        "qualification_contract_sha256",
    )
    signal = _hashed(
        {
            "decision": "schedule_viable",
            "viability_passed": True,
            "arms": {
                arm: {
                    "deterministic_replay_status": {
                        "passed": True,
                        "group_id": f"{arm}-task",
                    }
                }
                for arm in ("generic", "targeted")
            },
        },
        "signal_summary_sha256",
    )
    summaries = {arm: [_summary(arm, 1), _summary(arm, 2)] for arm in ("generic", "targeted")}
    return contract, signal, summaries


def test_selection_decision_requires_two_exact_nonzero_projections_per_arm() -> None:
    contract, signal, summaries = _fixtures()
    result = build_selection_and_gradient_decision(
        contract=contract,
        signal_summary=signal,
        summaries=summaries,
    )
    assert result["both_arms_pass"] is True
    assert result["compatibility_authorized"] is True
    assert result["counted_training_authorized"] is False
    assert result["source_commit"] == "a" * 40
    assert result["signal_summary_sha256"] == signal["signal_summary_sha256"]


def test_selection_decision_rejects_zero_policy_gradient() -> None:
    contract, signal, summaries = _fixtures()
    summaries["generic"][1] = _hashed(
        {key: value for key, value in summaries["generic"][1].items() if key != "summary_sha256"}
        | {"policy_gradient_global_norm": 0.0},
        "summary_sha256",
    )
    with pytest.raises(ValueError, match="generic projection qualification failed"):
        build_selection_and_gradient_decision(
            contract=contract,
            signal_summary=signal,
            summaries=summaries,
        )
