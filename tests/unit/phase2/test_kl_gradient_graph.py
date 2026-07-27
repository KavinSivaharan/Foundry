from __future__ import annotations

import pytest
import torch

from foundry.phase2 import kl_objective


def test_active_and_reference_logits_must_not_share_adapter_state() -> None:
    adapter_delta = torch.tensor([[[0.5, -0.5]]], requires_grad=True)
    base_logits = torch.tensor([[[1.0, 0.0]]])
    reference = base_logits.detach()
    active = base_logits + adapter_delta
    mask = torch.tensor([[True]])
    loss = kl_objective.forward_token_kl(reference, active, mask, torch)
    loss.backward()
    assert loss.item() > 0.0
    assert adapter_delta.grad is not None
    assert torch.linalg.vector_norm(adapter_delta.grad).item() > 0.0
    accidentally_same = kl_objective.forward_token_kl(active, active, mask, torch)
    assert accidentally_same.item() == pytest.approx(0.0, abs=1e-8)


def test_kl_tensor_remains_connected_through_coefficient_multiplication() -> None:
    adapter_delta = torch.tensor([[[0.5, -0.5]]], requires_grad=True)
    reference = torch.tensor([[[1.0, 0.0]]])
    policy = reference + adapter_delta
    mask = torch.tensor([[True]])
    kl = kl_objective.forward_token_kl(reference, policy, mask, torch)
    total = kl_objective.replay_total_loss(torch.tensor(0.25), kl, 3.0)
    total.backward()
    assert total.grad_fn is not None
    assert kl.grad_fn is not None
    assert adapter_delta.grad is not None
    assert torch.linalg.vector_norm(adapter_delta.grad).item() > 0.0
    with pytest.raises(RuntimeError, match="does not require grad"):
        kl.detach().backward()
