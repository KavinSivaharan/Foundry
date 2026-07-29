from __future__ import annotations

from typing import Any

import pytest

from foundry.phase2.l3_grpo_runtime import RuntimeGroup
from foundry.phase2.l3_grpo_signal_qualification_runtime import (
    _gradient_gate,
    _selected_prefix,
)


def _group(position: int, kind: str = "task") -> RuntimeGroup:
    return RuntimeGroup(
        group_id=f"group-{position}",
        arm="generic",
        position=position,
        source_kind=kind,  # type: ignore[arg-type]
        source_id=f"source-{position}",
        category="family",
        messages=(),
        prompt_sha256=str(position) * 64,
        prompt_tokens=1,
        reward_metadata_json="{}",
    )


def test_selected_prefix_preserves_rng_schedule_positions() -> None:
    groups = tuple(_group(position) for position in range(1, 7))
    selected = {
        "task_schedule_position": 5,
        "task_group_id": "group-5",
        "task_prompt_sha256": "5" * 64,
    }
    prefix = _selected_prefix(groups, selected)
    assert [group.position for group in prefix] == [1, 2, 3, 4, 5]


def _projection() -> dict[str, Any]:
    return {
        "finite": True,
        "graph_connected": True,
        "global_norm": 0.5,
        "nonzero_gradient_count": 2,
    }


def test_gradient_gate_accepts_nonzero_policy_and_zero_frozen_gradients() -> None:
    _gradient_gate(
        task_group={
            "reward_variance": 0.25,
            "canonical_cuda_advantages": [-1.0, 0.0, 0.5, 0.5],
            "valid_completion_token_counts": [1, 2, 3, 4],
            "policy_logprobs_finite": True,
            "reference_logprobs_finite": True,
            "backend_failure_count": 0,
        },
        policy_gradient=_projection(),
        combined_gradient=_projection(),
        populated={
            **_projection(),
            "reference_gradient_count": 0,
            "base_gradient_count": 0,
        },
        objective_graph={
            "policy_objective": {"requires_grad": True},
            "combined_objective": {"requires_grad": True},
        },
        stock_loss_equal=True,
    )


def test_gradient_gate_rejects_reference_gradient() -> None:
    with pytest.raises(RuntimeError, match="gradient projection gate failed"):
        _gradient_gate(
            task_group={
                "reward_variance": 0.25,
                "canonical_cuda_advantages": [-1.0, 0.0, 0.5, 0.5],
                "valid_completion_token_counts": [1, 2, 3, 4],
                "policy_logprobs_finite": True,
                "reference_logprobs_finite": True,
                "backend_failure_count": 0,
            },
            policy_gradient=_projection(),
            combined_gradient=_projection(),
            populated={
                **_projection(),
                "reference_gradient_count": 1,
                "base_gradient_count": 0,
            },
            objective_graph={
                "policy_objective": {"requires_grad": True},
                "combined_objective": {"requires_grad": True},
            },
            stock_loss_equal=True,
        )
