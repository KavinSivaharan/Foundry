from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from foundry.phase2.l3_grpo_source_binding import verify_published_warmup_bundle
from foundry.phase2.l3_grpo_warmup_prepare import (
    CONTRACT_OUTPUT,
    _porcelain_paths,
    _trajectory,
    compatibility_order,
)

ROOT = Path(__file__).resolve().parents[3]


def test_porcelain_parser_preserves_trimmed_first_path() -> None:
    assert _porcelain_paths(
        "M src/foundry/phase2/l3_grpo_runtime.py\n"
        "?? tests/unit/phase2/test_l3_grpo_warmup_prepare.py"
    ) == [
        "src/foundry/phase2/l3_grpo_runtime.py",
        "tests/unit/phase2/test_l3_grpo_warmup_prepare.py",
    ]


def test_two_step_scheduler_trajectory_starts_at_zero_then_positive() -> None:
    torch = pytest.importorskip("torch")

    def scheduler_factory(
        optimizer: object,
        *,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> object:
        assert num_warmup_steps == 1
        assert num_training_steps == 2
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                float(step)
                if step < num_warmup_steps
                else max(
                    0.0,
                    0.5
                    * (
                        1.0
                        + math.cos(
                            math.pi
                            * (step - num_warmup_steps)
                            / (num_training_steps - num_warmup_steps)
                        )
                    ),
                )
            ),
        )

    result = _trajectory(
        torch,
        SimpleNamespace(get_cosine_schedule_with_warmup=scheduler_factory),
        total_steps=2,
    )
    assert result["warmup_steps"] == 1
    assert result["first_strictly_positive_optimizer_call_index"] == 2
    assert result["effective_learning_rate_trajectory"] == [0.0, 1.0e-6]


def test_compatibility_order_is_replay_then_selected_task_only() -> None:
    selection = {
        "arms": {
            arm: {
                "replay_group_id": f"{arm}-g004",
                "replay_schedule_position": 4,
                "replay_prompt_sha256": "a" * 64,
                "task_group_id": f"{arm}-task",
                "task_schedule_position": 5,
                "task_prompt_sha256": "b" * 64,
            }
            for arm in ("generic", "targeted")
        }
    }
    result = compatibility_order(
        Path("C:/Foundry"),
        {"selection": selection},
    )
    for arm in ("generic", "targeted"):
        arm_order = result["arms"][arm]
        assert arm_order["optimizer_call_1"]["source_kind"] == "base_replay"
        assert arm_order["optimizer_call_2"]["source_kind"] == "task"
    assert result["compatibility_only"] is True
    assert result["counted_generic_schedule_changed"] is False


def test_published_warmup_contract_reconstructs_when_frozen() -> None:
    path = ROOT / "results/phase2_vetted_corpus" / CONTRACT_OUTPUT
    if not path.exists():
        pytest.skip("warmup-aware source freeze has not been published yet")
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    assert (
        value["warmup_update_contract_sha256"]
        == (verify_published_warmup_bundle(ROOT)["warmup_update_contract_sha256"])
    )
