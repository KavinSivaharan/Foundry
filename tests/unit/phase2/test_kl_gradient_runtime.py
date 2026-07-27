from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from foundry.phase2 import kl_gradient_runtime

ROOT = Path(__file__).resolve().parents[3]


def test_replay_projection_is_identical_across_frozen_schedules() -> None:
    root = ROOT / "results/raw/phase2_vetted_corpus/v1_replay25_schedules"
    values = {
        arm: json.loads((root / f"{arm}_schedule.json").read_text(encoding="utf-8"))[:16]
        for arm in ("generic", "targeted")
    }
    generic = kl_gradient_runtime.replay_projection(values["generic"])
    targeted = kl_gradient_runtime.replay_projection(values["targeted"])
    assert generic == targeted
    assert len(generic) == 213
    assert sum(int(row["tokens"]) for row in generic) == 4_000


def test_gradient_summary_captures_direction_and_structure() -> None:
    names = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "base_model.model.model.layers.1.self_attn.v_proj.lora_B.default.weight",
    ]
    ce = {
        names[0]: torch.tensor([3.0, 4.0]),
        names[1]: torch.tensor([0.0, 2.0]),
    }
    kl = {
        names[0]: torch.tensor([4.0, -4.0]),
        names[1]: torch.tensor([0.0, 1.0]),
    }
    value = kl_gradient_runtime.summarize_gradients(ce, kl, torch)
    assert value["ce_global_l2_norm"] == pytest.approx((29.0) ** 0.5)
    assert value["kl_global_l2_norm"] == pytest.approx((33.0) ** 0.5)
    assert value["opposing_gradient_sign_tensor_count"] == 1
    assert value["nonzero_ce_lora_tensor_count"] == 2
    assert value["nonzero_kl_lora_tensor_count"] == 2
    assert set(value["per_layer_ce_l2_norm"]) == {"0", "1"}
    assert set(value["per_projection_kl_l2_norm"]) == {"q_proj", "v_proj"}
    assert value["finite_gradients"]
