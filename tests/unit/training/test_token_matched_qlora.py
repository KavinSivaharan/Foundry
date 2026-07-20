from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from foundry.training.token_matched_config import load_token_matched_recipe
from foundry.training.token_matched_qlora import (
    _load_schedule,
    _validate_schedule_records,
    token_weighted_loss,
)


def _schedule_payload() -> list[dict[str, object]]:
    return [
        {
            "step": 1,
            "occurrences": [
                {
                    "synthetic_id": "example-a",
                    "occurrence_index": 1,
                    "loss_bearing_tokens": 2,
                },
                {
                    "synthetic_id": "example-b",
                    "occurrence_index": 1,
                    "loss_bearing_tokens": 1,
                },
            ],
            "loss_bearing_tokens": 3,
        }
    ]


def test_token_matched_recipe_freezes_method_b() -> None:
    recipe = load_token_matched_recipe(
        Path("configs/training/qwen2_5_1_5b_token_matched_qlora_v2.yaml")
    )
    assert (
        recipe.recipe_sha256 == "df7c7b8d7b402683a550fb11ebbe4ceb633ed47597c98ea661affa7876d6fa54"
    )
    assert recipe.optimizer_steps == 200
    assert recipe.micro_batch_size == 1
    assert recipe.whole_examples_only
    assert not recipe.split_examples
    assert not recipe.packed_examples


def test_token_weighted_micro_gradients_match_combined_token_mean() -> None:
    first = torch.tensor([1.0, 2.0], dtype=torch.float64)
    second = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)

    micro_parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    first_loss = ((micro_parameter - first) ** 2).mean()
    second_loss = ((micro_parameter - second) ** 2).mean()
    combined_micro_loss = token_weighted_loss(first_loss, 2, 5) + token_weighted_loss(
        second_loss, 3, 5
    )
    combined_micro_loss.backward()

    reference_parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    reference_loss = ((reference_parameter - torch.cat((first, second))) ** 2).mean()
    reference_loss.backward()

    assert combined_micro_loss.item() == pytest.approx(reference_loss.item(), abs=1e-12)
    assert micro_parameter.grad is not None and reference_parameter.grad is not None
    assert micro_parameter.grad.item() == pytest.approx(reference_parameter.grad.item(), abs=1e-12)


def test_schedule_loader_and_record_validation_preserve_boundaries_and_masks(
    tmp_path: Path,
) -> None:
    payload = _schedule_payload()
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(payload), encoding="utf-8")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    schedule = _load_schedule(
        schedule_path,
        expected_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        expected_steps=1,
    )
    records = [{"synthetic_id": "example-a"}, {"synthetic_id": "example-b"}]
    values = [
        {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0], "labels": [1, 2, -100]},
        {"input_ids": [3, 0, 0], "attention_mask": [1, 0, 0], "labels": [3, -100, -100]},
    ]
    by_id, counts = _validate_schedule_records(schedule, records, values)
    assert set(by_id) == {"example-a", "example-b"}
    assert counts == {"example-a": 2, "example-b": 1}
    assert schedule[0].loss_bearing_tokens == 3


def test_schedule_token_mismatch_fails_closed(tmp_path: Path) -> None:
    payload = _schedule_payload()
    payload[0]["loss_bearing_tokens"] = 4
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with pytest.raises(ValueError, match="step token total differs"):
        _load_schedule(
            path,
            expected_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            expected_steps=1,
        )
