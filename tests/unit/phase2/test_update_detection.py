from __future__ import annotations

import pytest
import torch

from foundry.phase2.update_detection import (
    Snapshot,
    detect_updates,
    snapshot_trainable,
    validate_optimizer_ownership,
)


def _parameters() -> list[tuple[str, torch.nn.Parameter]]:
    return [
        ("layer.lora_A.default.weight", torch.nn.Parameter(torch.tensor([1.0]))),
        ("layer.lora_B.default.weight", torch.nn.Parameter(torch.tensor([2.0]))),
    ]


def test_known_nonzero_update_is_detected() -> None:
    values = _parameters()
    snapshots = snapshot_trainable(values)
    values[1][1].grad = torch.tensor([1.0])
    values[1][1].data.add_(-0.1)
    evidence = detect_updates(snapshots, values)
    assert evidence["tensors_changed"] == 1
    assert evidence["global_delta_norm"] > 0


def test_zero_lr_is_expected_noop() -> None:
    values = _parameters()
    snapshots = snapshot_trainable(values)
    optimizer = torch.optim.AdamW([parameter for _, parameter in values], lr=0.0)
    for _, parameter in values:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert detect_updates(snapshots, values)["tensors_changed"] == 0


def test_aliased_snapshot_fails() -> None:
    values = _parameters()
    aliased = [Snapshot(name=name, value=parameter.detach()) for name, parameter in values]
    assert aliased[0].value.data_ptr() == values[0][1].data_ptr()


def test_subset_detector_fails() -> None:
    values = _parameters()
    with pytest.raises(ValueError, match="every"):
        detect_updates(snapshot_trainable(values)[:1], values)


def test_optimizer_and_trainable_sets_must_match() -> None:
    values = _parameters()
    optimizer = torch.optim.AdamW([parameter for _, parameter in values], lr=1e-5)
    validate_optimizer_ownership(values, optimizer)
    partial = torch.optim.AdamW([values[0][1]], lr=1e-5)
    with pytest.raises(ValueError, match="differ"):
        validate_optimizer_ownership(values, partial)
