"""Exception-safe runtime scaling for one active PEFT LoRA adapter."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from foundry.training.config import canonical_sha256

SCALING_IMPLEMENTATION_ID = "foundry-common-lora-runtime-scaling-v1"
_ACTIVE_SENTINEL = "_foundry_runtime_lora_scaling_active"


@dataclass
class ScalingEvidence:
    """Mutable evidence populated before, during, and after one scaling context."""

    implementation_id: str
    adapter_name: str
    scale: float
    lora_module_count: int
    original_scaling_sha256: str
    applied_scaling_sha256: str
    adapter_state_sha256_before: str
    base_parameter_signature_before: str
    adapter_state_sha256_after: str | None = None
    base_parameter_signature_after: str | None = None
    original_scaling_restored: bool = False
    adapter_state_unchanged: bool = False
    base_parameter_signature_unchanged: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return stable content-free evidence after the context exits."""

        return {
            "implementation_id": self.implementation_id,
            "adapter_name": self.adapter_name,
            "scale": self.scale,
            "lora_module_count": self.lora_module_count,
            "original_scaling_sha256": self.original_scaling_sha256,
            "applied_scaling_sha256": self.applied_scaling_sha256,
            "adapter_state_sha256_before": self.adapter_state_sha256_before,
            "adapter_state_sha256_after": self.adapter_state_sha256_after,
            "base_parameter_signature_before": self.base_parameter_signature_before,
            "base_parameter_signature_after": self.base_parameter_signature_after,
            "original_scaling_restored": self.original_scaling_restored,
            "adapter_state_unchanged": self.adapter_state_unchanged,
            "base_parameter_signature_unchanged": (self.base_parameter_signature_unchanged),
        }


def _active_adapter(model: Any) -> str:
    active = list(model.active_adapters)
    if len(active) != 1 or not isinstance(active[0], str):
        raise ValueError("runtime scaling requires exactly one active adapter")
    return active[0]


def _lora_modules(model: Any, adapter_name: str) -> list[tuple[str, Any]]:
    modules: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        lora_a = getattr(module, "lora_A", None)
        if not isinstance(scaling, dict) or lora_a is None:
            continue
        if adapter_name not in scaling or adapter_name not in lora_a:
            continue
        if bool(getattr(module, "merged", False)):
            raise ValueError("runtime scaling refuses merged LoRA modules")
        modules.append((name, module))
    if not modules:
        raise ValueError("active adapter has no LoRA modules to scale")
    return modules


def _scaling_payload(modules: list[tuple[str, Any]], adapter_name: str) -> list[dict[str, object]]:
    return [
        {"module": name, "adapter": adapter_name, "scaling": float(module.scaling[adapter_name])}
        for name, module in modules
    ]


def adapter_state_sha256(model: Any, adapter_name: str) -> str:
    """Hash every in-memory adapter tensor without changing its device or value."""

    import torch

    state_provider = getattr(model, "_foundry_adapter_state_dict", None)
    if callable(state_provider):
        state = state_provider(adapter_name)
    else:
        peft = importlib.import_module("peft")
        state = peft.get_peft_model_state_dict(model, adapter_name=adapter_name)
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def base_parameter_signature_sha256(model: Any) -> str:
    """Hash base parameter identity/version metadata to detect in-memory mutation."""

    payload = []
    for name, parameter in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            continue
        payload.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "version": int(parameter._version),
            }
        )
    return canonical_sha256(payload)


@contextmanager
def scaled_lora_adapter(model: Any, scale: float) -> Iterator[ScalingEvidence]:
    """Temporarily multiply every active-adapter LoRA scaling value by ``scale``."""

    if isinstance(scale, bool) or not isinstance(scale, int | float):
        raise ValueError("LoRA scale must be a finite number")
    factor = float(scale)
    if not math.isfinite(factor) or factor < 0.0 or factor > 1.0:
        raise ValueError("LoRA scale must be finite and between 0.0 and 1.0")
    if bool(getattr(model, _ACTIVE_SENTINEL, False)):
        raise RuntimeError("nested runtime LoRA scaling is not allowed")

    adapter_name = _active_adapter(model)
    modules = _lora_modules(model, adapter_name)
    original = [(name, float(module.scaling[adapter_name])) for name, module in modules]
    original_payload = [
        {"module": name, "adapter": adapter_name, "scaling": value} for name, value in original
    ]
    before_state = adapter_state_sha256(model, adapter_name)
    before_base = base_parameter_signature_sha256(model)
    setattr(model, _ACTIVE_SENTINEL, True)
    for (_, module), (_, value) in zip(modules, original, strict=True):
        module.scaling[adapter_name] = value * factor
    applied_payload = _scaling_payload(modules, adapter_name)
    evidence = ScalingEvidence(
        implementation_id=SCALING_IMPLEMENTATION_ID,
        adapter_name=adapter_name,
        scale=factor,
        lora_module_count=len(modules),
        original_scaling_sha256=canonical_sha256(original_payload),
        applied_scaling_sha256=canonical_sha256(applied_payload),
        adapter_state_sha256_before=before_state,
        base_parameter_signature_before=before_base,
    )
    try:
        yield evidence
    finally:
        for (_, module), (_, value) in zip(modules, original, strict=True):
            module.scaling[adapter_name] = value
        delattr(model, _ACTIVE_SENTINEL)
        restored_payload = _scaling_payload(modules, adapter_name)
        evidence.adapter_state_sha256_after = adapter_state_sha256(model, adapter_name)
        evidence.base_parameter_signature_after = base_parameter_signature_sha256(model)
        evidence.original_scaling_restored = restored_payload == original_payload
        evidence.adapter_state_unchanged = evidence.adapter_state_sha256_after == before_state
        evidence.base_parameter_signature_unchanged = (
            evidence.base_parameter_signature_after == before_base
        )
        if not (
            evidence.original_scaling_restored
            and evidence.adapter_state_unchanged
            and evidence.base_parameter_signature_unchanged
        ):
            raise RuntimeError("runtime LoRA scaling state restoration failed")
