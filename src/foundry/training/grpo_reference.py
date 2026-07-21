"""Audited helpers for Foundry's PEFT-backed GRPO reference policy."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceExpectation:
    """One exact installed source file and its required semantic markers."""

    relative_path: str
    sha256: str
    required_fragments: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceImplementationAudit:
    """Content-free evidence for the installed official PEFT reference path."""

    package_versions: tuple[tuple[str, str], ...]
    source_hashes: tuple[tuple[str, str], ...]
    uses_adapter_disabled_reference: bool
    creates_second_reference_for_peft: bool
    exception_safe_adapter_restoration: bool


@dataclass(frozen=True)
class TrainabilityAudit:
    """Counts proving that an adapter is the only trainable state."""

    total_parameters: int
    trainable_parameters: int
    trainable_names: tuple[str, ...]


@dataclass(frozen=True)
class AdapterRuntimeState:
    """Minimal PEFT state that must survive a reference pass unchanged."""

    enabled: bool | str
    active_adapters: tuple[str, ...]


EXPECTED_PACKAGE_VERSIONS: Mapping[str, str] = {
    "peft": "0.15.2",
    "trl": "0.17.0",
}

EXPECTED_SOURCE_FILES: tuple[SourceExpectation, ...] = (
    SourceExpectation(
        relative_path="trl/trainer/grpo_trainer.py",
        sha256="425161a6e4f82ee7cc6d4d6ad3fe7e495db970289d28427f45e99368ac5e985a",
        required_fragments=(
            "elif is_peft_model(model):",
            "# If PEFT is used, the reference model is not needed since the adapter can be "
            "disabled",
            "with torch.no_grad():",
            "with self.accelerator.unwrap_model(self.model).disable_adapter():",
            "ref_per_token_logps = self._get_per_token_logps(",
        ),
    ),
    SourceExpectation(
        relative_path="trl/trainer/grpo_config.py",
        sha256="83d53640316958da75c4bb73451f9562f235f886c0cc31a3c825de172c0e17cc",
        required_fragments=(
            "disable_dropout: bool = field(",
            "max_prompt_length: Optional[int] = field(",
            "num_generations: Optional[int] = field(",
            "mask_truncated_completions: bool = field(",
            "sync_ref_model: bool = field(",
        ),
    ),
    SourceExpectation(
        relative_path="peft/peft_model.py",
        sha256="ea36efc37191855bb14fbb1ecd6743148aaa13350fed4ee9a8582c2b7fa29696",
        required_fragments=(
            "def disable_adapter(self):",
            "self.base_model.disable_adapter_layers()",
            "finally:",
            "self.base_model.enable_adapter_layers()",
        ),
    ),
)

FROZEN_CHECKPOINT_STEPS = (16, 32, 64)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_distribution_version(site_packages: Path, distribution: str) -> str:
    metadata_paths = sorted(site_packages.glob(f"{distribution}-*.dist-info/METADATA"))
    if len(metadata_paths) != 1:
        raise ValueError(
            f"expected exactly one {distribution} distribution, found {len(metadata_paths)}"
        )
    for line in metadata_paths[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ").strip()
    raise ValueError(f"{distribution} metadata has no Version field")


def validate_installed_reference_contract(
    site_packages: Path,
    *,
    expected_package_versions: Mapping[str, str] = EXPECTED_PACKAGE_VERSIONS,
    source_contract: Sequence[SourceExpectation] = EXPECTED_SOURCE_FILES,
) -> ReferenceImplementationAudit:
    """Fail closed unless the installed TRL/PEFT reference implementation is exact."""

    if not site_packages.is_dir():
        raise FileNotFoundError(f"site-packages directory does not exist: {site_packages}")

    versions: list[tuple[str, str]] = []
    for package, expected_version in sorted(expected_package_versions.items()):
        actual_version = _installed_distribution_version(site_packages, package)
        if actual_version != expected_version:
            raise ValueError(
                f"{package} version differs: expected {expected_version}, got {actual_version}"
            )
        versions.append((package, actual_version))

    source_hashes: list[tuple[str, str]] = []
    for expectation in source_contract:
        path = site_packages / Path(expectation.relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"required reference source does not exist: {path}")
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expectation.sha256:
            raise ValueError(
                f"installed reference source hash differs: {expectation.relative_path}"
            )
        source = path.read_text(encoding="utf-8")
        missing = [
            fragment for fragment in expectation.required_fragments if fragment not in source
        ]
        if missing:
            raise ValueError(
                f"installed reference API semantics differ: {expectation.relative_path}: {missing}"
            )
        source_hashes.append((expectation.relative_path, actual_sha256))

    return ReferenceImplementationAudit(
        package_versions=tuple(versions),
        source_hashes=tuple(source_hashes),
        uses_adapter_disabled_reference=True,
        creates_second_reference_for_peft=False,
        exception_safe_adapter_restoration=True,
    )


def assert_only_lora_trainable(model: Any) -> TrainabilityAudit:
    """Require a frozen base and a non-empty set of LoRA-only trainable parameters."""

    total_parameters = 0
    trainable_parameters = 0
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        if count < 0:
            raise ValueError(f"parameter has a negative element count: {name}")
        total_parameters += count
        if bool(parameter.requires_grad):
            if "lora_" not in name:
                raise ValueError(f"non-LoRA parameter is trainable: {name}")
            trainable_parameters += count
            trainable_names.append(str(name))
        else:
            frozen_names.append(str(name))

    if not trainable_names:
        raise ValueError("no LoRA parameters are trainable")
    if not frozen_names:
        raise ValueError("no frozen base parameters were found")
    return TrainabilityAudit(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        trainable_names=tuple(trainable_names),
    )


def _normalize_active_adapters(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise TypeError("PEFT active adapter state must be a string or sequence")


def adapter_runtime_state(model: Any) -> AdapterRuntimeState:
    """Read the public PEFT model-status contract without importing PEFT."""

    status = model.get_model_status()
    enabled = status.enabled
    if not isinstance(enabled, bool | str):
        raise TypeError("PEFT enabled state must be bool or str")
    active = getattr(status, "active_adapters", getattr(model, "active_adapter", None))
    return AdapterRuntimeState(enabled=enabled, active_adapters=_normalize_active_adapters(active))


@dataclass(frozen=True)
class _RngSnapshot:
    python_state: tuple[Any, ...]
    torch_cpu_state: Any
    torch_cuda_states: tuple[Any, ...] | None


def _clone_state(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else value


def _capture_rng_state(torch_module: Any) -> _RngSnapshot:
    cpu_state = _clone_state(torch_module.get_rng_state())
    cuda_states: tuple[Any, ...] | None = None
    if bool(torch_module.cuda.is_available()):
        cuda_states = tuple(_clone_state(state) for state in torch_module.cuda.get_rng_state_all())
    return _RngSnapshot(random.getstate(), cpu_state, cuda_states)


def _restore_rng_state(torch_module: Any, snapshot: _RngSnapshot) -> None:
    random.setstate(snapshot.python_state)
    torch_module.set_rng_state(snapshot.torch_cpu_state)
    if snapshot.torch_cuda_states is not None:
        torch_module.cuda.set_rng_state_all(list(snapshot.torch_cuda_states))


@contextmanager
def adapter_disabled_no_grad(model: Any, torch_module: Any) -> Iterator[Any]:
    """Run a reference pass with adapters off and restore adapter/RNG state exactly."""

    before = adapter_runtime_state(model)
    if before.enabled is not True:
        raise ValueError(
            "the active policy adapter must be uniformly enabled before reference scoring"
        )
    rng_before = _capture_rng_state(torch_module)
    try:
        with torch_module.no_grad():
            with model.disable_adapter():
                disabled = adapter_runtime_state(model)
                if disabled.enabled is not False:
                    raise RuntimeError("PEFT did not disable every adapter layer")
                if bool(torch_module.is_grad_enabled()):
                    raise RuntimeError("reference pass did not enter no-grad mode")
                yield model
    finally:
        _restore_rng_state(torch_module, rng_before)
        after = adapter_runtime_state(model)
        if after != before:
            raise RuntimeError("PEFT adapter state was not restored after reference scoring")


def assert_no_grad_tensors(value: object) -> None:
    """Fail if a nested reference output retains an autograd graph."""

    if hasattr(value, "requires_grad"):
        if bool(attrgetter("requires_grad")(value)):
            raise ValueError("reference output unexpectedly requires gradients")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            assert_no_grad_tensors(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            assert_no_grad_tensors(item)


def run_adapter_disabled_reference(
    model: Any,
    forward: Callable[..., Any],
    torch_module: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute one audited adapter-disabled reference forward pass."""

    with adapter_disabled_no_grad(model, torch_module):
        output = forward(*args, **kwargs)
        assert_no_grad_tensors(output)
        return output


def _reference_values_equal(left: object, right: object, torch_module: Any) -> bool:
    if hasattr(left, "requires_grad") or hasattr(right, "requires_grad"):
        try:
            return bool(torch_module.equal(left, right))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if tuple(left) != tuple(right):
            return False
        return all(_reference_values_equal(left[key], right[key], torch_module) for key in left)
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, str | bytes | bytearray)
        and not isinstance(right, str | bytes | bytearray)
    ):
        return len(left) == len(right) and all(
            _reference_values_equal(left_item, right_item, torch_module)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def assert_reference_deterministic(
    model: Any,
    forward: Callable[..., Any],
    torch_module: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run the same frozen-base reference pass twice and require exact equality."""

    first = run_adapter_disabled_reference(model, forward, torch_module, *args, **kwargs)
    second = run_adapter_disabled_reference(model, forward, torch_module, *args, **kwargs)
    if not _reference_values_equal(first, second, torch_module):
        raise ValueError("repeated adapter-disabled reference outputs differ")
    return first


def grpo_per_token_kl(
    policy_log_probabilities: Sequence[float], reference_log_probabilities: Sequence[float]
) -> tuple[float, ...]:
    """Reproduce TRL 0.17's non-negative per-token KL estimator."""

    if len(policy_log_probabilities) != len(reference_log_probabilities):
        raise ValueError("policy and reference log-probability lengths differ")
    if not policy_log_probabilities:
        raise ValueError("at least one log-probability pair is required")
    values: list[float] = []
    for policy_logp, reference_logp in zip(
        policy_log_probabilities, reference_log_probabilities, strict=True
    ):
        if not math.isfinite(policy_logp) or not math.isfinite(reference_logp):
            raise ValueError("log probabilities must be finite")
        delta = reference_logp - policy_logp
        value = math.expm1(delta) - delta
        if value < 0.0 and value > -1e-15:
            value = 0.0
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("computed KL estimate is invalid")
        values.append(value)
    return tuple(values)


def mean_grpo_kl(
    policy_log_probabilities: Sequence[float], reference_log_probabilities: Sequence[float]
) -> float:
    """Return the arithmetic mean of the frozen TRL per-token KL estimator."""

    values = grpo_per_token_kl(policy_log_probabilities, reference_log_probabilities)
    return math.fsum(values) / len(values)


def make_exact_checkpoint_callback(trainer_callback_base: type[Any]) -> Any:
    """Build a Trainer callback that saves exactly steps 16, 32, and 64."""

    class ExactCheckpointCallback(trainer_callback_base):  # type: ignore[misc]
        checkpoint_steps = frozenset(FROZEN_CHECKPOINT_STEPS)

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if int(state.max_steps) != FROZEN_CHECKPOINT_STEPS[-1]:
                raise ValueError("exact checkpoint callback requires a 64-step training run")
            control.should_save = False
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            control.should_save = int(state.global_step) in self.checkpoint_steps
            return control

    ExactCheckpointCallback.__name__ = "ExactCheckpointCallback"
    return ExactCheckpointCallback()
