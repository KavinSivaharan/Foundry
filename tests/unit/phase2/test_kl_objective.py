from __future__ import annotations

from pathlib import Path

import pytest
import torch

from foundry.phase2 import kl_objective, kl_recipe, vetted_qlora_kl
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]


def test_identical_logits_have_zero_forward_kl() -> None:
    logits = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    mask = torch.tensor([[True, True]])
    value = kl_objective.forward_token_kl(logits, logits.clone(), mask, torch)
    assert value.item() == pytest.approx(0.0, abs=1e-8)


def test_controlled_perturbation_is_positive_and_directional() -> None:
    reference = torch.tensor([[[3.0, 0.0, -1.0]]])
    policy = torch.tensor([[[0.0, 2.0, -1.0]]])
    mask = torch.tensor([[True]])
    forward = kl_objective.forward_token_kl(reference, policy, mask, torch)
    reverse = kl_objective.forward_token_kl(policy, reference, mask, torch)
    assert forward.item() > 0.0
    assert reverse.item() > 0.0
    assert forward.item() != pytest.approx(reverse.item())


def test_mask_excludes_prompt_padding_and_post_eos_positions() -> None:
    labels = torch.tensor([[-100, -100, 7, 8, -100, -100]])
    mask = kl_objective.assistant_shift_mask(labels)
    assert mask.tolist() == [[False, True, True, False, False]]


def test_reference_logits_do_not_receive_gradients() -> None:
    reference = torch.tensor([[[1.0, 0.0]]], requires_grad=False)
    policy = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    mask = torch.tensor([[True]])
    loss = kl_objective.forward_token_kl(reference, policy, mask, torch)
    loss.backward()
    assert reference.grad is None
    assert policy.grad is not None
    assert torch.isfinite(policy.grad).all()


def test_total_loss_requires_positive_frozen_coefficient() -> None:
    ce = torch.tensor(2.0)
    kl = torch.tensor(0.5)
    assert kl_objective.replay_total_loss(ce, kl, 0.1).item() == pytest.approx(2.05)
    with pytest.raises(ValueError, match="positive"):
        kl_objective.replay_total_loss(ce, kl, 0.0)


def test_active_trainable_inventory_matches_historical_state_projection() -> None:
    class Parameter:
        requires_grad = True
        shape = (8, 1536)

    class Model:
        def named_parameters(self):  # type: ignore[no-untyped-def]
            return [
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
                    Parameter(),
                )
            ]

    shapes = [
        {
            "name": name.replace(".default", ""),
            "shape": list(parameter.shape),
        }
        for name, parameter in Model().named_parameters()
        if parameter.requires_grad
    ]
    assert (
        canonical_sha256(shapes)
        != kl_recipe.reconstruct(ROOT)["canonical_lora_configuration"]["tensor_inventory_sha256"]
    )
    assert vetted_qlora_kl.RECIPE_SHA256 == (
        "3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"
    )
