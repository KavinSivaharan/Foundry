"""Clone-safe LoRA optimizer-update evidence for the Phase 2 QLoRA probe."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from foundry.training.config import canonical_sha256


@dataclass(frozen=True)
class Snapshot:
    name: str
    value: Any


def snapshot_trainable(named_parameters: list[tuple[str, Any]]) -> list[Snapshot]:
    snapshots = [
        Snapshot(name=name, value=parameter.detach().clone())
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if not snapshots:
        raise ValueError("no trainable parameters")
    if any(
        snapshot.value.data_ptr() == parameter.data_ptr()
        for snapshot, (_, parameter) in zip(
            snapshots,
            [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad],
            strict=True,
        )
    ):
        raise ValueError("snapshot aliases live parameter storage")
    return snapshots


def validate_optimizer_ownership(named_parameters: list[tuple[str, Any]], optimizer: Any) -> None:
    trainable = {id(parameter) for _, parameter in named_parameters if parameter.requires_grad}
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if owned != trainable:
        raise ValueError("optimizer-owned and trainable parameter sets differ")
    if any(
        "lora_A" not in name and "lora_B" not in name
        for name, parameter in named_parameters
        if parameter.requires_grad
    ):
        raise ValueError("a trainable parameter is not a LoRA A/B tensor")


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def detect_updates(
    snapshots: list[Snapshot], named_parameters: list[tuple[str, Any]]
) -> dict[str, Any]:
    live = {name: parameter for name, parameter in named_parameters if parameter.requires_grad}
    if set(live) != {snapshot.name for snapshot in snapshots}:
        raise ValueError("update detector does not cover every trainable parameter")
    rows: list[dict[str, Any]] = []
    total_changed = 0
    global_delta_squared = 0.0
    for snapshot in snapshots:
        parameter = live[snapshot.name]
        delta = parameter.detach().float() - snapshot.value.detach().float()
        gradient = parameter.grad
        changed = int((delta != 0).sum().item())
        delta_norm = float(delta.norm().item())
        total_changed += changed
        global_delta_squared += delta_norm * delta_norm
        rows.append(
            {
                "name_sha256": hashlib.sha256(snapshot.name.encode()).hexdigest(),
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "requires_grad": bool(parameter.requires_grad),
                "gradient_present": gradient is not None,
                "gradient_finite": bool(gradient is not None and gradient.isfinite().all().item()),
                "gradient_nonzero_count": int((gradient != 0).sum().item())
                if gradient is not None
                else 0,
                "gradient_max_abs": float(gradient.abs().max().item())
                if gradient is not None
                else 0.0,
                "gradient_frobenius_norm": float(gradient.float().norm().item())
                if gradient is not None
                else 0.0,
                "before_sha256": _tensor_sha256(snapshot.value),
                "after_sha256": _tensor_sha256(parameter),
                "maximum_absolute_delta": float(delta.abs().max().item()),
                "delta_frobenius_norm": delta_norm,
                "changed_element_count": changed,
            }
        )
    evidence = {
        "trainable_tensor_count": len(live),
        "optimizer_tensor_count": len(live),
        "tensors_with_gradients": sum(row["gradient_present"] for row in rows),
        "tensors_with_nonzero_gradients": sum(row["gradient_nonzero_count"] > 0 for row in rows),
        "tensors_changed": sum(row["changed_element_count"] > 0 for row in rows),
        "total_changed_elements": total_changed,
        "global_delta_norm": math.sqrt(global_delta_squared),
        "tensors": rows,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence
