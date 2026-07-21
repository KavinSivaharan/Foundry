import torch

from foundry.training.lora_scaling import scaled_lora_adapter


class FakeLoraLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})
        self.scaling = {"default": 2.0}
        self.merged = False


class FakePeftModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.first = FakeLoraLayer()
        self.active_adapters = ["default"]

    def _foundry_adapter_state_dict(self, adapter_name: str) -> dict[str, torch.Tensor]:
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        }


def _forward(model: FakePeftModel, value: torch.Tensor) -> torch.Tensor:
    base = model.base(value)
    layer = model.first
    delta = layer.lora_B["default"](layer.lora_A["default"](value))
    return base + layer.scaling["default"] * delta


def test_scale_zero_matches_base_and_scale_one_matches_unscaled() -> None:
    torch.manual_seed(20260720)
    model = FakePeftModel()
    value = torch.tensor([[1.0, -2.0]])
    base = model.base(value)
    unscaled = _forward(model, value)
    with scaled_lora_adapter(model, 0.0):
        zero = _forward(model, value)
    with scaled_lora_adapter(model, 1.0):
        one = _forward(model, value)
    assert torch.equal(zero, base)
    assert torch.equal(one, unscaled)
    assert model.first.scaling["default"] == 2.0
