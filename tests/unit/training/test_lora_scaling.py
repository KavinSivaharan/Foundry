import math

import pytest
import torch

from foundry.training.lora_scaling import (
    adapter_state_sha256,
    base_parameter_signature_sha256,
    scaled_lora_adapter,
)


class FakeLoraLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})
        self.scaling = {"default": 2.0}
        self.merged = False


class FakePeftModel(torch.nn.Module):
    def __init__(self, active: tuple[str, ...] = ("default",)) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.first = FakeLoraLayer()
        self.second = FakeLoraLayer()
        self.active_adapters = list(active)

    def _foundry_adapter_state_dict(self, adapter_name: str) -> dict[str, torch.Tensor]:
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        }


@pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_all_approved_scales_apply_uniformly_and_restore(factor: float) -> None:
    model = FakePeftModel()
    state_before = adapter_state_sha256(model, "default")
    base_before = base_parameter_signature_sha256(model)
    with scaled_lora_adapter(model, factor) as evidence:
        assert model.first.scaling["default"] == 2.0 * factor
        assert model.second.scaling["default"] == 2.0 * factor
        assert evidence.scale == factor
        assert evidence.lora_module_count == 2
    assert model.first.scaling["default"] == 2.0
    assert model.second.scaling["default"] == 2.0
    assert evidence.original_scaling_restored is True
    assert evidence.adapter_state_unchanged is True
    assert evidence.base_parameter_signature_unchanged is True
    assert adapter_state_sha256(model, "default") == state_before
    assert base_parameter_signature_sha256(model) == base_before


def test_nested_scaling_rejects_and_outer_context_restores() -> None:
    model = FakePeftModel()
    with scaled_lora_adapter(model, 0.75):
        with pytest.raises(RuntimeError, match="nested"):
            with scaled_lora_adapter(model, 0.5):
                pass
        assert model.first.scaling["default"] == 1.5
    assert model.first.scaling["default"] == 2.0


def test_exception_restores_original_state() -> None:
    model = FakePeftModel()
    with pytest.raises(LookupError, match="fixture"):
        with scaled_lora_adapter(model, 0.25):
            assert model.first.scaling["default"] == 0.5
            raise LookupError("fixture")
    assert model.first.scaling["default"] == 2.0
    assert not hasattr(model, "_foundry_runtime_lora_scaling_active")


@pytest.mark.parametrize("factor", [-0.01, 1.01, math.inf, -math.inf, math.nan, True, "0.5"])
def test_invalid_scale_rejects(factor: object) -> None:
    with pytest.raises(ValueError, match="scale"):
        with scaled_lora_adapter(FakePeftModel(), factor):  # type: ignore[arg-type]
            pass


def test_one_active_adapter_only_and_unmerged_enforcement() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        with scaled_lora_adapter(FakePeftModel(("default", "second")), 0.5):
            pass
    model = FakePeftModel()
    model.first.merged = True
    with pytest.raises(ValueError, match="merged"):
        with scaled_lora_adapter(model, 0.5):
            pass
