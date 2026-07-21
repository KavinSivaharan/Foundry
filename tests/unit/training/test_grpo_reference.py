from __future__ import annotations

import hashlib
import math
import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foundry.training import grpo_reference as gr


class _Parameter:
    def __init__(self, count: int, *, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class _FakeTensor:
    def __init__(self, value: int, *, requires_grad: bool = False) -> None:
        self.value = value
        self.requires_grad = requires_grad

    def clone(self) -> _FakeTensor:
        return _FakeTensor(self.value, requires_grad=self.requires_grad)


class _FakeCuda:
    def is_available(self) -> bool:
        return False


class _FakeTorch:
    def __init__(self) -> None:
        self._rng_state = _FakeTensor(7)
        self._grad_enabled = True
        self.cuda = _FakeCuda()

    def get_rng_state(self) -> _FakeTensor:
        return self._rng_state

    def set_rng_state(self, state: _FakeTensor) -> None:
        self._rng_state = state.clone()

    def is_grad_enabled(self) -> bool:
        return self._grad_enabled

    @contextmanager
    def no_grad(self) -> Any:
        previous = self._grad_enabled
        self._grad_enabled = False
        try:
            yield
        finally:
            self._grad_enabled = previous

    @staticmethod
    def equal(left: _FakeTensor, right: _FakeTensor) -> bool:
        return left.value == right.value and left.requires_grad == right.requires_grad


class _FakePeftModel:
    def __init__(self) -> None:
        self.enabled = True
        self.active_adapter = "default"
        self.parameters = {
            "base_model.model.layer.weight": _Parameter(100, requires_grad=False),
            "base_model.model.layer.lora_A.default.weight": _Parameter(8, requires_grad=True),
            "base_model.model.layer.lora_B.default.weight": _Parameter(8, requires_grad=True),
        }

    def named_parameters(self) -> Any:
        return iter(self.parameters.items())

    def get_model_status(self) -> Any:
        return SimpleNamespace(enabled=self.enabled, active_adapters=[self.active_adapter])

    @contextmanager
    def disable_adapter(self) -> Any:
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous


def _write_distribution(site_packages: Path, name: str, version: str) -> None:
    metadata = site_packages / f"{name}-{version}.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )


def _expectation(site_packages: Path, relative_path: str, source: str) -> gr.SourceExpectation:
    path = site_packages / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return gr.SourceExpectation(
        relative_path=relative_path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        required_fragments=(source.splitlines()[0],),
    )


def test_installed_reference_contract_validates_versions_hashes_and_semantics(
    tmp_path: Path,
) -> None:
    _write_distribution(tmp_path, "trl", "0.17.0")
    _write_distribution(tmp_path, "peft", "0.15.2")
    expectations = (
        _expectation(tmp_path, "trl/trainer/grpo_trainer.py", "peft reference marker\n"),
        _expectation(tmp_path, "trl/trainer/grpo_config.py", "config marker\n"),
        _expectation(tmp_path, "peft/peft_model.py", "restoration marker\n"),
    )
    audit = gr.validate_installed_reference_contract(tmp_path, source_contract=expectations)
    assert audit.package_versions == (("peft", "0.15.2"), ("trl", "0.17.0"))
    assert audit.uses_adapter_disabled_reference is True
    assert audit.creates_second_reference_for_peft is False
    assert audit.exception_safe_adapter_restoration is True

    (tmp_path / expectations[0].relative_path).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash differs"):
        gr.validate_installed_reference_contract(tmp_path, source_contract=expectations)


def test_installed_reference_contract_rejects_version_or_api_drift(tmp_path: Path) -> None:
    _write_distribution(tmp_path, "trl", "0.17.1")
    _write_distribution(tmp_path, "peft", "0.15.2")
    with pytest.raises(ValueError, match="trl version differs"):
        gr.validate_installed_reference_contract(tmp_path, source_contract=())

    source = "present marker\n"
    expectation = _expectation(tmp_path, "trl/trainer/grpo_trainer.py", source)
    missing_marker = gr.SourceExpectation(
        expectation.relative_path, expectation.sha256, ("absent marker",)
    )
    with pytest.raises(ValueError, match="API semantics differ"):
        gr.validate_installed_reference_contract(
            tmp_path,
            expected_package_versions={"peft": "0.15.2", "trl": "0.17.1"},
            source_contract=(missing_marker,),
        )


def test_only_lora_parameters_are_trainable() -> None:
    model = _FakePeftModel()
    audit = gr.assert_only_lora_trainable(model)
    assert audit.total_parameters == 116
    assert audit.trainable_parameters == 16
    assert len(audit.trainable_names) == 2

    model.parameters["base_model.model.layer.weight"].requires_grad = True
    with pytest.raises(ValueError, match="non-LoRA parameter"):
        gr.assert_only_lora_trainable(model)


def test_reference_context_disables_adapter_and_restores_rng_and_state() -> None:
    model = _FakePeftModel()
    torch = _FakeTorch()
    python_rng_before = random.getstate()
    with gr.adapter_disabled_no_grad(model, torch):
        assert model.enabled is False
        assert torch.is_grad_enabled() is False
        torch._rng_state = _FakeTensor(99)
        random.random()
    assert model.enabled is True
    assert torch.is_grad_enabled() is True
    assert torch.get_rng_state().value == 7
    assert random.getstate() == python_rng_before


def test_reference_context_restores_state_when_forward_raises() -> None:
    model = _FakePeftModel()
    torch = _FakeTorch()

    def failing_forward() -> None:
        torch._rng_state = _FakeTensor(88)
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        gr.run_adapter_disabled_reference(model, failing_forward, torch)
    assert model.enabled is True
    assert torch.get_rng_state().value == 7
    assert torch.is_grad_enabled() is True


def test_reference_outputs_are_no_grad_deterministic_and_rng_neutral() -> None:
    model = _FakePeftModel()
    torch = _FakeTorch()

    def forward() -> dict[str, _FakeTensor]:
        next_value = torch.get_rng_state().value + 1
        torch._rng_state = _FakeTensor(next_value)
        return {"logits": _FakeTensor(next_value)}

    output = gr.assert_reference_deterministic(model, forward, torch)
    assert output["logits"].value == 8
    assert torch.get_rng_state().value == 7

    with pytest.raises(ValueError, match="requires gradients"):
        gr.assert_no_grad_tensors(_FakeTensor(1, requires_grad=True))


def test_grpo_kl_is_zero_for_equal_logps_and_positive_for_perturbation() -> None:
    assert gr.grpo_per_token_kl((-1.0, -2.0), (-1.0, -2.0)) == (0.0, 0.0)
    assert gr.mean_grpo_kl((-1.0, -2.0), (-1.0, -2.0)) == 0.0
    perturbed = gr.grpo_per_token_kl((-0.5, -2.5), (-1.0, -2.0))
    assert all(value > 0.0 for value in perturbed)
    assert gr.mean_grpo_kl((-0.5, -2.5), (-1.0, -2.0)) > 0.0

    with pytest.raises(ValueError, match="lengths differ"):
        gr.grpo_per_token_kl((-1.0,), (-1.0, -2.0))
    with pytest.raises(ValueError, match="must be finite"):
        gr.grpo_per_token_kl((math.inf,), (-1.0,))


class _CallbackBase:
    pass


def test_exact_checkpoint_callback_saves_only_16_32_and_64() -> None:
    callback = gr.make_exact_checkpoint_callback(_CallbackBase)
    state = SimpleNamespace(max_steps=64, global_step=0)
    control = SimpleNamespace(should_save=True)
    callback.on_train_begin(None, state, control)
    assert control.should_save is False

    observed: list[int] = []
    for step in range(1, 65):
        state.global_step = step
        control.should_save = True
        callback.on_step_end(None, state, control)
        if control.should_save:
            observed.append(step)
    assert observed == [16, 32, 64]

    state.max_steps = 63
    with pytest.raises(ValueError, match="64-step"):
        callback.on_train_begin(None, state, control)
