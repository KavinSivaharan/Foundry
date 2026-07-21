"""Content-free exact-replay evidence for verifier-reward GRPO runs.

The counted runtime owns generation and optimization.  This module only turns
already-observed runtime state into deterministic, self-hashed evidence.  Raw
completion text and tensor values never enter a packet: they are represented
by ordered SHA-256 digests instead.

The helpers deliberately avoid importing PyTorch at module import time so the
evidence contract remains testable in Foundry's lightweight development
environment.  Callers pass the active ``torch`` module when RNG state is
captured.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from foundry.training.config import canonical_sha256

REPLAY_EVIDENCE_ID = "foundry-verifier-grpo-exact-replay-v1"
REPLAY_EVIDENCE_SCHEMA_VERSION = 1

ReplayKind = Literal["generation_only", "two_step_compatibility"]

_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _float_hex(value: object, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number.hex()


def _mapping_key(value: object) -> str:
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    raise TypeError(f"state mapping key has unsupported type: {type(value).__name__}")


def _tensor_bytes(value: object) -> tuple[object, bytes]:
    tensor = value
    for method_name in ("detach", "cpu", "contiguous"):
        method = getattr(tensor, method_name, None)
        if not callable(method):
            raise TypeError(f"tensor value lacks {method_name}()")
        tensor = method()
    numpy_method = getattr(tensor, "numpy", None)
    if not callable(numpy_method):
        raise TypeError("tensor value cannot be converted to exact bytes")
    try:
        array = numpy_method()
    except (TypeError, RuntimeError):
        # NumPy has no native bfloat16 dtype.  A byte view preserves the exact
        # stored representation without a lossy cast.
        import torch  # imported only for the uncommon dtype fallback

        view_method = getattr(tensor, "view", None)
        if not callable(view_method):
            raise TypeError("tensor value cannot be viewed as bytes") from None
        array = view_method(torch.uint8).numpy()
    tobytes = getattr(array, "tobytes", None)
    if not callable(tobytes):
        raise TypeError("tensor array cannot be serialized to bytes")
    return tensor, cast(bytes, tobytes(order="C"))


@dataclass(frozen=True)
class TensorEvidence:
    """Exact digest and shape metadata for one observed tensor."""

    dtype: str
    shape: tuple[int, ...]
    device: str
    numel: int
    requires_grad: bool
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "device": self.device,
            "numel": self.numel,
            "requires_grad": self.requires_grad,
            "sha256": self.sha256,
        }


def tensor_evidence(value: object) -> TensorEvidence:
    """Capture exact tensor storage without retaining tensor values."""

    tensor, raw = _tensor_bytes(value)
    raw_shape = getattr(tensor, "shape", None)
    if raw_shape is None:
        raise TypeError("tensor value lacks shape metadata")
    shape = tuple(int(item) for item in raw_shape)
    numel_method = getattr(tensor, "numel", None)
    if not callable(numel_method):
        raise TypeError("tensor value lacks numel()")
    numel = int(numel_method())
    if numel < 0:
        raise ValueError("tensor numel cannot be negative")
    metadata = {
        "dtype": str(getattr(tensor, "dtype", "")),
        "shape": list(shape),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return TensorEvidence(
        dtype=cast(str, metadata["dtype"]),
        shape=shape,
        device=str(getattr(value, "device", "unknown")),
        numel=numel,
        requires_grad=bool(getattr(value, "requires_grad", False)),
        sha256=canonical_sha256(metadata),
    )


def _is_tensor(value: object) -> bool:
    return all(callable(getattr(value, name, None)) for name in ("detach", "cpu", "numel"))


def _state_manifest(
    value: object,
    *,
    path: str,
    tensors: list[dict[str, object]],
) -> object:
    if _is_tensor(value):
        evidence = tensor_evidence(value)
        record = {"path": path, **evidence.as_dict()}
        tensors.append(record)
        return {"tensor_sha256": evidence.sha256}
    array_tobytes = getattr(value, "tobytes", None)
    array_shape = getattr(value, "shape", None)
    array_dtype = getattr(value, "dtype", None)
    if callable(array_tobytes) and array_shape is not None and array_dtype is not None:
        raw = cast(bytes, array_tobytes(order="C"))
        array_manifest = {
            "dtype": str(array_dtype),
            "shape": [int(item) for item in array_shape],
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
        return {"array_sha256": canonical_sha256(array_manifest), **array_manifest}
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return {"float_hex": _float_hex(value, path)}
    if isinstance(value, bytes | bytearray):
        return {"bytes_sha256": hashlib.sha256(bytes(value)).hexdigest(), "length": len(value)}
    if isinstance(value, Mapping):
        keyed = sorted((_mapping_key(key), item) for key, item in value.items())
        return {
            key: _state_manifest(item, path=f"{path}/{key}", tensors=tensors) for key, item in keyed
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _state_manifest(item, path=f"{path}/{index}", tensors=tensors)
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} has unsupported replay-state type: {type(value).__name__}")


def state_tree_evidence(value: object, *, label: str) -> dict[str, object]:
    """Hash an optimizer, scheduler, RNG, or other nested deterministic state."""

    _require_nonempty(label, "label")
    tensors: list[dict[str, object]] = []
    manifest = _state_manifest(value, path="$", tensors=tensors)
    return {
        "label": label,
        "state_sha256": canonical_sha256(manifest),
        "tensor_count": len(tensors),
        "tensor_evidence": tensors,
        "tensor_evidence_sha256": canonical_sha256(tensors),
    }


def _capture_stateful_object(value: object, *, label: str) -> dict[str, object]:
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"{label} does not expose state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} state_dict() must return a mapping")
    return state_tree_evidence(state, label=label)


def capture_optimizer_state(optimizer: object) -> dict[str, object]:
    """Capture the optimizer scalars, parameter groups, and exact tensor states."""

    return _capture_stateful_object(optimizer, label="optimizer")


def capture_scheduler_state(scheduler: object) -> dict[str, object]:
    """Capture the learning-rate scheduler's complete deterministic state."""

    return _capture_stateful_object(scheduler, label="scheduler")


def capture_rng_state(
    torch_module: object,
    *,
    python_random: object = random,
    numpy_random: object | None = None,
) -> dict[str, object]:
    """Capture Python, CPU Torch, CUDA Torch, and optional NumPy RNG states."""

    get_python_state = getattr(python_random, "getstate", None)
    if not callable(get_python_state):
        raise TypeError("python_random does not expose getstate()")
    get_cpu_state = getattr(torch_module, "get_rng_state", None)
    if not callable(get_cpu_state):
        raise TypeError("torch module does not expose get_rng_state()")
    cuda = getattr(torch_module, "cuda", None)
    cuda_available_method = getattr(cuda, "is_available", None)
    cuda_available = bool(cuda_available_method()) if callable(cuda_available_method) else False
    cuda_states: list[dict[str, object]] = []
    if cuda_available:
        get_cuda_states = getattr(cuda, "get_rng_state_all", None)
        if not callable(get_cuda_states):
            raise TypeError("CUDA RNG state is unavailable")
        cuda_states = [tensor_evidence(item).as_dict() for item in get_cuda_states()]

    numpy_state: dict[str, object] | None = None
    if numpy_random is not None:
        get_numpy_state = getattr(numpy_random, "get_state", None)
        if not callable(get_numpy_state):
            raise TypeError("numpy_random does not expose get_state()")
        numpy_state = state_tree_evidence(get_numpy_state(), label="numpy_rng")

    payload: dict[str, object] = {
        "python": state_tree_evidence(get_python_state(), label="python_rng"),
        "torch_cpu": tensor_evidence(get_cpu_state()).as_dict(),
        "torch_cuda": cuda_states,
        "cuda_available": cuda_available,
        "numpy": numpy_state,
    }
    payload["rng_sha256"] = canonical_sha256(payload)
    return payload


def capture_lora_state(model: object) -> dict[str, object]:
    """Hash every named LoRA parameter in stable name order."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model does not expose named_parameters()")
    rows = [
        {"name": str(name), **tensor_evidence(parameter).as_dict()}
        for name, parameter in sorted(named_parameters(), key=lambda item: str(item[0]))
        if "lora_" in str(name)
    ]
    if not rows:
        raise ValueError("model has no LoRA parameters")
    tensor_identities = [
        {
            "name": row["name"],
            "dtype": row["dtype"],
            "shape": row["shape"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    return {
        "parameter_count": len(rows),
        "parameters": rows,
        "lora_state_sha256": canonical_sha256(rows),
        "lora_tensor_state_sha256": canonical_sha256(tensor_identities),
    }


def capture_base_parameter_state(model: object) -> dict[str, object]:
    """Hash every non-LoRA parameter from its stable name and exact CPU bytes.

    The aggregate identity deliberately excludes the source device and gradient
    state.  It is therefore a byte-exact parameter-value contract rather than a
    runtime-placement or trainability contract.  Those properties remain
    available as content-free diagnostics in each parameter row.
    """

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model does not expose named_parameters()")
    try:
        raw_parameters = list(named_parameters(remove_duplicate=False))
    except TypeError:
        raw_parameters = list(named_parameters())

    parameters: list[tuple[str, object]] = []
    names: set[str] = set()
    for index, item in enumerate(raw_parameters):
        if not isinstance(item, Sequence) or isinstance(item, str | bytes | bytearray):
            raise TypeError(f"named parameter {index} is not a name/value pair")
        if len(item) != 2:
            raise TypeError(f"named parameter {index} is not a name/value pair")
        raw_name, parameter = item
        if not isinstance(raw_name, str):
            raise TypeError(f"named parameter {index} has a non-text name")
        name = _require_nonempty(raw_name, f"named parameter {index} name")
        if name in names:
            raise ValueError(f"model exposes duplicate parameter name: {name}")
        names.add(name)
        if "lora_" not in name:
            parameters.append((name, parameter))

    if not parameters:
        raise ValueError("model has no non-LoRA base parameters")

    rows: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    total_numel = 0
    total_bytes = 0
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        tensor, raw = _tensor_bytes(parameter)
        raw_shape = getattr(tensor, "shape", None)
        if raw_shape is None:
            raise TypeError(f"base parameter lacks shape metadata: {name}")
        shape = tuple(int(item) for item in raw_shape)
        numel_method = getattr(tensor, "numel", None)
        if not callable(numel_method):
            raise TypeError(f"base parameter lacks numel(): {name}")
        numel = int(numel_method())
        if numel < 0:
            raise ValueError(f"base parameter numel cannot be negative: {name}")
        requires_grad = getattr(parameter, "requires_grad", None)
        if not isinstance(requires_grad, bool):
            raise TypeError(f"base parameter requires_grad is not Boolean: {name}")
        dtype = str(getattr(tensor, "dtype", ""))
        if not dtype:
            raise TypeError(f"base parameter lacks dtype metadata: {name}")
        device_value = getattr(parameter, "device", None)
        if device_value is None or not str(device_value):
            raise TypeError(f"base parameter lacks device metadata: {name}")
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        identity: dict[str, object] = {
            "name": name,
            "dtype": dtype,
            "shape": list(shape),
            "raw_sha256": raw_sha256,
        }
        row: dict[str, object] = {
            **identity,
            "numel": numel,
            "byte_count": len(raw),
            "source_device": str(device_value),
            "requires_grad": requires_grad,
            "parameter_sha256": canonical_sha256(identity),
        }
        identities.append(identity)
        rows.append(row)
        total_numel += numel
        total_bytes += len(raw)

    return {
        "parameter_count": len(rows),
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "parameters": rows,
        "base_parameter_state_sha256": canonical_sha256(identities),
    }


def capture_gradient_state(model: object) -> dict[str, object]:
    """Hash LoRA gradients and identify any prohibited frozen-base gradients."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model does not expose named_parameters()")
    lora_rows: list[dict[str, object]] = []
    frozen_gradients: list[str] = []
    for raw_name, parameter in sorted(named_parameters(), key=lambda item: str(item[0])):
        name = str(raw_name)
        gradient = getattr(parameter, "grad", None)
        if "lora_" not in name:
            if gradient is not None:
                frozen_gradients.append(name)
            continue
        row: dict[str, object] = {"name": name, "present": gradient is not None}
        if gradient is not None:
            row["gradient"] = tensor_evidence(gradient).as_dict()
        lora_rows.append(row)
    if not lora_rows:
        raise ValueError("model has no LoRA parameters")
    payload: dict[str, object] = {
        "lora_parameter_count": len(lora_rows),
        "lora_gradient_count": sum(bool(row["present"]) for row in lora_rows),
        "lora_gradients": lora_rows,
        "frozen_gradient_names": frozen_gradients,
        "frozen_gradient_count": len(frozen_gradients),
    }
    payload["gradient_state_sha256"] = canonical_sha256(payload)
    return payload


def _token_rows(value: object) -> list[list[int]]:
    rows = value
    for method_name in ("detach", "cpu"):
        method = getattr(rows, method_name, None)
        if callable(method):
            rows = method()
    tolist = getattr(rows, "tolist", None)
    if callable(tolist):
        rows = tolist()
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes | bytearray):
        raise TypeError("generated token IDs must be a row sequence")
    result: list[list[int]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, str | bytes | bytearray):
            raise TypeError("each generated token-ID row must be a sequence")
        values: list[int] = []
        for token in row:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("generated token IDs must be nonnegative integers")
            values.append(token)
        if not values:
            raise ValueError("generated token-ID rows cannot be empty")
        result.append(values)
    if not result:
        raise ValueError("generated token IDs cannot be empty")
    return result


def _reward_component_rows(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected_keys: tuple[str, ...] | None = None
    for index, value in enumerate(values):
        keys = tuple(sorted(value))
        if not keys or expected_keys not in {None, keys}:
            raise ValueError("reward component keys must be non-empty and identical")
        expected_keys = keys
        row: dict[str, object] = {}
        for key in keys:
            item = value[key]
            if isinstance(item, bool):
                row[key] = item
            elif isinstance(item, int | float):
                row[key] = {"float_hex": _float_hex(item, f"rewards[{index}].{key}")}
            elif item is None or isinstance(item, str):
                row[key] = item
            else:
                raise TypeError(f"reward component {key} has unsupported value type")
        rows.append(row)
    return rows


@dataclass(frozen=True)
class GenerationEvidence:
    """Exact content-free evidence for one ordered completion group."""

    group_id: str
    source_kind: Literal["synthetic", "base_replay"]
    prompt_sha256: str
    completion_count: int
    completion_token_lengths: tuple[int, ...]
    completion_token_lengths_sha256: str
    token_rows_sha256: str
    token_row_sha256s: tuple[str, ...]
    decoded_completion_sha256: str
    decoded_completion_sha256s: tuple[str, ...]
    truncation_flags: tuple[bool, ...]
    truncation_sha256: str
    reward_components: tuple[dict[str, object], ...]
    reward_components_sha256: str
    rng_before_sha256: str
    rng_after_sha256: str
    rng_advanced: bool
    warning_sha256s: tuple[str, ...]
    warning_sequence_sha256: str
    reference_logprobs: dict[str, object] | None
    policy_logprobs: dict[str, object] | None
    per_token_kl: dict[str, object] | None
    evidence_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "source_kind": self.source_kind,
            "prompt_sha256": self.prompt_sha256,
            "completion_count": self.completion_count,
            "completion_token_lengths": list(self.completion_token_lengths),
            "completion_token_lengths_sha256": self.completion_token_lengths_sha256,
            "token_rows_sha256": self.token_rows_sha256,
            "token_row_sha256s": list(self.token_row_sha256s),
            "decoded_completion_sha256": self.decoded_completion_sha256,
            "decoded_completion_sha256s": list(self.decoded_completion_sha256s),
            "truncation_flags": list(self.truncation_flags),
            "truncation_sha256": self.truncation_sha256,
            "reward_components": list(self.reward_components),
            "reward_components_sha256": self.reward_components_sha256,
            "rng_before_sha256": self.rng_before_sha256,
            "rng_after_sha256": self.rng_after_sha256,
            "rng_advanced": self.rng_advanced,
            "warning_count": len(self.warning_sha256s),
            "warning_sha256s": list(self.warning_sha256s),
            "warning_sequence_sha256": self.warning_sequence_sha256,
            "reference_logprobs": self.reference_logprobs,
            "policy_logprobs": self.policy_logprobs,
            "per_token_kl": self.per_token_kl,
            "evidence_sha256": self.evidence_sha256,
        }


def capture_generation_evidence(
    *,
    group_id: str,
    source_kind: Literal["synthetic", "base_replay"],
    prompt_sha256: str,
    generated_token_ids: object,
    decoded_completions: Sequence[str],
    completion_token_lengths: Sequence[int],
    truncation_flags: Sequence[bool],
    reward_components: Sequence[Mapping[str, object]],
    rng_before_sha256: str,
    rng_after_sha256: str,
    warning_messages: Sequence[str] = (),
    warning_sha256s: Sequence[str] = (),
    reference_logprobs: object | None = None,
    policy_logprobs: object | None = None,
    per_token_kl: object | None = None,
) -> GenerationEvidence:
    """Hash all deterministic outputs from one generation-and-scoring group."""

    group_id = _require_nonempty(group_id, "group_id")
    if source_kind not in {"synthetic", "base_replay"}:
        raise ValueError("source_kind must be synthetic or base_replay")
    prompt_sha256 = _require_sha256(prompt_sha256, "prompt_sha256")
    rng_before_sha256 = _require_sha256(rng_before_sha256, "rng_before_sha256")
    rng_after_sha256 = _require_sha256(rng_after_sha256, "rng_after_sha256")
    token_rows = _token_rows(generated_token_ids)
    completions = list(decoded_completions)
    if any(not isinstance(item, str) for item in completions):
        raise TypeError("decoded completions must be text")
    flags = tuple(truncation_flags)
    if any(not isinstance(item, bool) for item in flags):
        raise TypeError("truncation flags must be boolean")
    count = len(token_rows)
    lengths = tuple(completion_token_lengths)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in lengths):
        raise ValueError("completion token lengths must be positive integers")
    if (
        count != len(completions)
        or count != len(lengths)
        or count != len(flags)
        or count != len(reward_components)
    ):
        raise ValueError("generation evidence columns must have exactly equal counts")
    if any(length > len(row) for length, row in zip(lengths, token_rows, strict=True)):
        raise ValueError("completion token length exceeds its generated token-ID row")
    rewards = _reward_component_rows(reward_components)
    warnings = list(warning_messages)
    supplied_warning_hashes = list(warning_sha256s)
    if warnings and supplied_warning_hashes:
        raise ValueError("warning messages and prehashed warning SHA-256s are mutually exclusive")
    if any(not isinstance(item, str) for item in warnings):
        raise TypeError("warning messages must be text")
    logprob_values = (reference_logprobs, policy_logprobs, per_token_kl)
    if any(item is None for item in logprob_values) and any(
        item is not None for item in logprob_values
    ):
        raise ValueError("reference, policy, and KL tensors must be supplied together")
    reference = (
        None if reference_logprobs is None else tensor_evidence(reference_logprobs).as_dict()
    )
    policy = None if policy_logprobs is None else tensor_evidence(policy_logprobs).as_dict()
    kl = None if per_token_kl is None else tensor_evidence(per_token_kl).as_dict()
    completion_hashes = tuple(
        hashlib.sha256(item.encode("utf-8")).hexdigest() for item in completions
    )
    token_hashes = tuple(canonical_sha256(item) for item in token_rows)
    warning_hashes = (
        tuple(hashlib.sha256(item.encode("utf-8")).hexdigest() for item in warnings)
        if warnings
        else tuple(
            _require_sha256(item, f"warning_sha256s[{index}]")
            for index, item in enumerate(supplied_warning_hashes)
        )
    )
    core: dict[str, object] = {
        "group_id": group_id,
        "source_kind": source_kind,
        "prompt_sha256": prompt_sha256,
        "completion_count": count,
        "completion_token_lengths": list(lengths),
        "completion_token_lengths_sha256": canonical_sha256(lengths),
        "token_rows_sha256": canonical_sha256(token_rows),
        "token_row_sha256s": list(token_hashes),
        "decoded_completion_sha256": canonical_sha256(completion_hashes),
        "decoded_completion_sha256s": list(completion_hashes),
        "truncation_flags": list(flags),
        "truncation_sha256": canonical_sha256(flags),
        "reward_components": rewards,
        "reward_components_sha256": canonical_sha256(rewards),
        "rng_before_sha256": rng_before_sha256,
        "rng_after_sha256": rng_after_sha256,
        "rng_advanced": rng_before_sha256 != rng_after_sha256,
        "warning_sha256s": list(warning_hashes),
        "warning_sequence_sha256": canonical_sha256(warning_hashes),
        "reference_logprobs": reference,
        "policy_logprobs": policy,
        "per_token_kl": kl,
    }
    return GenerationEvidence(
        group_id=group_id,
        source_kind=source_kind,
        prompt_sha256=prompt_sha256,
        completion_count=count,
        completion_token_lengths=lengths,
        completion_token_lengths_sha256=cast(str, core["completion_token_lengths_sha256"]),
        token_rows_sha256=cast(str, core["token_rows_sha256"]),
        token_row_sha256s=token_hashes,
        decoded_completion_sha256=cast(str, core["decoded_completion_sha256"]),
        decoded_completion_sha256s=completion_hashes,
        truncation_flags=flags,
        truncation_sha256=cast(str, core["truncation_sha256"]),
        reward_components=tuple(rewards),
        reward_components_sha256=cast(str, core["reward_components_sha256"]),
        rng_before_sha256=rng_before_sha256,
        rng_after_sha256=rng_after_sha256,
        rng_advanced=rng_before_sha256 != rng_after_sha256,
        warning_sha256s=warning_hashes,
        warning_sequence_sha256=cast(str, core["warning_sequence_sha256"]),
        reference_logprobs=reference,
        policy_logprobs=policy,
        per_token_kl=kl,
        evidence_sha256=canonical_sha256(core),
    )


@dataclass(frozen=True)
class CompatibilityStepEvidence:
    """Exact evidence spanning one backward and optimizer update."""

    step: int
    generation: GenerationEvidence
    loss_hex: str
    loss_tensor: dict[str, object]
    mean_kl_hex: str
    mean_kl_tensor: dict[str, object]
    rng_before: dict[str, object]
    rng_after: dict[str, object]
    lora_before: dict[str, object]
    lora_after: dict[str, object]
    gradients_after_backward: dict[str, object]
    gradients_after_clipping: dict[str, object]
    optimizer_before: dict[str, object]
    optimizer_after: dict[str, object]
    scheduler_before: dict[str, object]
    scheduler_after: dict[str, object]
    strict_mode_evidence: dict[str, bool]
    evidence_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "generation": self.generation.as_dict(),
            "loss_hex": self.loss_hex,
            "loss_tensor": self.loss_tensor,
            "mean_kl_hex": self.mean_kl_hex,
            "mean_kl_tensor": self.mean_kl_tensor,
            "rng_before": self.rng_before,
            "rng_after": self.rng_after,
            "lora_before": self.lora_before,
            "lora_after": self.lora_after,
            "gradients_after_backward": self.gradients_after_backward,
            "gradients_after_clipping": self.gradients_after_clipping,
            "optimizer_before": self.optimizer_before,
            "optimizer_after": self.optimizer_after,
            "scheduler_before": self.scheduler_before,
            "scheduler_after": self.scheduler_after,
            "strict_mode_evidence": self.strict_mode_evidence,
            "evidence_sha256": self.evidence_sha256,
        }


def build_compatibility_step_evidence(
    *,
    step: int,
    generation: GenerationEvidence,
    loss: float,
    loss_tensor: object,
    mean_kl: float,
    mean_kl_tensor: object,
    rng_before: Mapping[str, object],
    rng_after: Mapping[str, object],
    lora_before: Mapping[str, object],
    lora_after: Mapping[str, object],
    gradients_after_backward: Mapping[str, object],
    gradients_after_clipping: Mapping[str, object],
    optimizer_before: Mapping[str, object],
    optimizer_after: Mapping[str, object],
    scheduler_before: Mapping[str, object],
    scheduler_after: Mapping[str, object],
    strict_mode_evidence: Mapping[str, bool],
) -> CompatibilityStepEvidence:
    """Build one self-hashed step record from already-captured states."""

    if isinstance(step, bool) or not isinstance(step, int) or step not in {1, 2}:
        raise ValueError("compatibility step must be 1 or 2")
    required_strict_points = {
        "after_backward",
        "before_optimizer",
        "after_optimizer",
        "after_scheduler",
    }
    if set(strict_mode_evidence) != required_strict_points or not all(
        value is True for value in strict_mode_evidence.values()
    ):
        raise ValueError("all frozen non-generation operations must record strict mode")
    fields = {
        "step": step,
        "generation": generation.as_dict(),
        "loss_hex": _float_hex(loss, "loss"),
        "loss_tensor": tensor_evidence(loss_tensor).as_dict(),
        "mean_kl_hex": _float_hex(mean_kl, "mean_kl"),
        "mean_kl_tensor": tensor_evidence(mean_kl_tensor).as_dict(),
        "rng_before": dict(rng_before),
        "rng_after": dict(rng_after),
        "lora_before": dict(lora_before),
        "lora_after": dict(lora_after),
        "gradients_after_backward": dict(gradients_after_backward),
        "gradients_after_clipping": dict(gradients_after_clipping),
        "optimizer_before": dict(optimizer_before),
        "optimizer_after": dict(optimizer_after),
        "scheduler_before": dict(scheduler_before),
        "scheduler_after": dict(scheduler_after),
        "strict_mode_evidence": dict(strict_mode_evidence),
    }
    return CompatibilityStepEvidence(
        step=step,
        generation=generation,
        loss_hex=cast(str, fields["loss_hex"]),
        loss_tensor=cast(dict[str, object], fields["loss_tensor"]),
        mean_kl_hex=cast(str, fields["mean_kl_hex"]),
        mean_kl_tensor=cast(dict[str, object], fields["mean_kl_tensor"]),
        rng_before=dict(rng_before),
        rng_after=dict(rng_after),
        lora_before=dict(lora_before),
        lora_after=dict(lora_after),
        gradients_after_backward=dict(gradients_after_backward),
        gradients_after_clipping=dict(gradients_after_clipping),
        optimizer_before=dict(optimizer_before),
        optimizer_after=dict(optimizer_after),
        scheduler_before=dict(scheduler_before),
        scheduler_after=dict(scheduler_after),
        strict_mode_evidence=dict(strict_mode_evidence),
        evidence_sha256=canonical_sha256(fields),
    )


def _build_packet(kind: ReplayKind, body: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": REPLAY_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": REPLAY_EVIDENCE_ID,
        "kind": kind,
        **dict(body),
    }
    payload["packet_sha256"] = canonical_sha256(payload)
    return payload


def build_generation_only_packet(
    *,
    run_contract: Mapping[str, object],
    generations: Sequence[GenerationEvidence],
    rng_before: Mapping[str, object],
    rng_after: Mapping[str, object],
    lora_state: Mapping[str, object],
) -> dict[str, object]:
    """Build one exact packet suitable for same- or fresh-process replay."""

    if not run_contract:
        raise ValueError("run contract cannot be empty")
    if len(generations) != 3:
        raise ValueError("generation-only packet requires exactly three groups")
    expected_kinds = ("synthetic", "synthetic", "base_replay")
    if tuple(item.source_kind for item in generations) != expected_kinds:
        raise ValueError(
            "generation-only groups must be ordered as two synthetic then one base replay"
        )
    if any(item.completion_count != 4 for item in generations):
        raise ValueError("each generation-only group must contain exactly four completions")
    if len({item.group_id for item in generations}) != 3:
        raise ValueError("generation-only group IDs must be unique")
    if len({item.prompt_sha256 for item in generations}) != 3:
        raise ValueError("generation-only prompt hashes must be unique")
    generation_records = [item.as_dict() for item in generations]
    return _build_packet(
        "generation_only",
        {
            "run_contract": dict(run_contract),
            "generations": generation_records,
            "generation_sequence_sha256": canonical_sha256(generation_records),
            "completion_count": sum(item.completion_count for item in generations),
            "rng_before": dict(rng_before),
            "rng_after": dict(rng_after),
            "lora_state": dict(lora_state),
        },
    )


def build_two_step_packet(
    *,
    run_contract: Mapping[str, object],
    steps: Sequence[CompatibilityStepEvidence],
    replay_generation: GenerationEvidence,
    initial_lora: Mapping[str, object],
    final_lora: Mapping[str, object],
    reloaded_lora: Mapping[str, object],
    base_before: Mapping[str, object],
    base_after: Mapping[str, object],
    reloaded_base: Mapping[str, object],
    final_optimizer: Mapping[str, object],
    final_scheduler: Mapping[str, object],
    adapter_artifact_sha256: str,
    adapter_directory_sha256: str,
) -> dict[str, object]:
    """Build an exact packet for the frozen two-update compatibility run."""

    if not run_contract:
        raise ValueError("run contract cannot be empty")
    if len(steps) != 2 or tuple(item.step for item in steps) != (1, 2):
        raise ValueError("two-step packet requires ordered steps 1 and 2 exactly once")
    if tuple(item.generation.source_kind for item in steps) != ("synthetic", "synthetic"):
        raise ValueError("two-step updates must use exactly two synthetic generations")
    if any(item.generation.completion_count != 4 for item in steps):
        raise ValueError("each compatibility update must contain exactly four completions")
    if replay_generation.source_kind != "base_replay" or replay_generation.completion_count != 4:
        raise ValueError("compatibility replay must contain four base-replay completions")
    group_ids = [item.generation.group_id for item in steps] + [replay_generation.group_id]
    if len(set(group_ids)) != 3:
        raise ValueError("compatibility group IDs must be unique")
    for field in (
        "trained_model_released_before_reload",
        "pre_reload_memory_gate_passed",
    ):
        if run_contract.get(field) is not True:
            raise ValueError(f"two-step run contract did not prove {field}")
    adapter_hash = _require_sha256(adapter_artifact_sha256, "adapter_artifact_sha256")
    directory_hash = _require_sha256(adapter_directory_sha256, "adapter_directory_sha256")
    base_states = [dict(base_before), dict(base_after), dict(reloaded_base)]
    base_hashes = [
        _require_sha256(
            state.get("base_parameter_state_sha256"),
            f"base_states[{index}].base_parameter_state_sha256",
        )
        for index, state in enumerate(base_states)
    ]
    if len(set(base_hashes)) != 1:
        raise ValueError("frozen base parameter bytes differ before, after, or after reload")
    final_lora_hash = _require_sha256(
        final_lora.get("lora_tensor_state_sha256"), "final_lora.lora_tensor_state_sha256"
    )
    initial_lora_hash = _require_sha256(
        initial_lora.get("lora_tensor_state_sha256"),
        "initial_lora.lora_tensor_state_sha256",
    )
    reloaded_lora_hash = _require_sha256(
        reloaded_lora.get("lora_tensor_state_sha256"),
        "reloaded_lora.lora_tensor_state_sha256",
    )
    if final_lora_hash != reloaded_lora_hash:
        raise ValueError("saved and reloaded LoRA tensor bytes differ")
    if initial_lora_hash == final_lora_hash:
        raise ValueError("two compatibility updates did not change the LoRA tensors")
    return _build_packet(
        "two_step_compatibility",
        {
            "run_contract": dict(run_contract),
            "steps": [item.as_dict() for item in steps],
            "replay_generation": replay_generation.as_dict(),
            "initial_lora": dict(initial_lora),
            "final_lora": dict(final_lora),
            "reloaded_lora": dict(reloaded_lora),
            "base_before": base_states[0],
            "base_after": base_states[1],
            "reloaded_base": base_states[2],
            "base_restoration_passed": True,
            "adapter_reload_passed": True,
            "final_optimizer": dict(final_optimizer),
            "final_scheduler": dict(final_scheduler),
            "adapter_artifact_sha256": adapter_hash,
            "adapter_directory_sha256": directory_hash,
        },
    )


def validate_replay_packet(value: Mapping[str, object], *, expected_kind: ReplayKind) -> str:
    """Validate schema, kind, and self-hash; return the packet SHA-256."""

    packet = dict(value)
    if packet.get("schema_version") != REPLAY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("replay evidence schema version differs")
    if packet.get("evidence_id") != REPLAY_EVIDENCE_ID:
        raise ValueError("replay evidence ID differs")
    if packet.get("kind") != expected_kind:
        raise ValueError("replay evidence kind differs")
    declared = _require_sha256(packet.pop("packet_sha256", None), "packet_sha256")
    if canonical_sha256(packet) != declared:
        raise ValueError("replay evidence self-hash differs")
    if expected_kind == "generation_only":
        _validate_generation_packet_body(packet)
    else:
        _validate_two_step_packet_body(packet)
    return declared


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed object")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _validate_generation_record(value: object, field: str) -> dict[str, object]:
    record = _object(value, field)
    declared = _require_sha256(record.get("evidence_sha256"), f"{field}.evidence_sha256")
    core = dict(record)
    core.pop("evidence_sha256")
    warning_count = core.pop("warning_count", None)
    warnings = _array(core.get("warning_sha256s"), f"{field}.warning_sha256s")
    if warning_count != len(warnings):
        raise ValueError(f"{field} warning count differs")
    if canonical_sha256(core) != declared:
        raise ValueError(f"{field} self-hash differs")
    count = record.get("completion_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{field} completion count is invalid")
    sequence_fields = (
        "completion_token_lengths",
        "token_row_sha256s",
        "decoded_completion_sha256s",
        "truncation_flags",
        "reward_components",
    )
    if any(len(_array(record.get(name), f"{field}.{name}")) != count for name in sequence_fields):
        raise ValueError(f"{field} completion evidence count differs")
    return record


def _validate_generation_packet_body(packet: Mapping[str, object]) -> None:
    rows = _array(packet.get("generations"), "generations")
    if len(rows) != 3:
        raise ValueError("generation-only packet must contain exactly three groups")
    generations = [
        _validate_generation_record(row, f"generations[{index}]") for index, row in enumerate(rows)
    ]
    if [item.get("source_kind") for item in generations] != [
        "synthetic",
        "synthetic",
        "base_replay",
    ]:
        raise ValueError("generation-only source composition differs")
    if any(item.get("completion_count") != 4 for item in generations):
        raise ValueError("generation-only completion count differs")
    if packet.get("completion_count") != 12:
        raise ValueError("generation-only aggregate completion count differs")
    if packet.get("generation_sequence_sha256") != canonical_sha256(generations):
        raise ValueError("generation-only sequence hash differs")


def _validate_two_step_packet_body(packet: Mapping[str, object]) -> None:
    run_contract = _object(packet.get("run_contract"), "run_contract")
    for field in (
        "trained_model_released_before_reload",
        "pre_reload_memory_gate_passed",
    ):
        if run_contract.get(field) is not True:
            raise ValueError(f"two-step run contract did not prove {field}")
    rows = _array(packet.get("steps"), "steps")
    if len(rows) != 2:
        raise ValueError("compatibility packet must contain exactly two steps")
    for expected_step, raw in enumerate(rows, start=1):
        row = _object(raw, f"steps[{expected_step - 1}]")
        if row.get("step") != expected_step:
            raise ValueError("compatibility step order differs")
        _validate_generation_record(row.get("generation"), f"steps[{expected_step - 1}].generation")
        declared = _require_sha256(
            row.get("evidence_sha256"), f"steps[{expected_step - 1}].evidence_sha256"
        )
        core = dict(row)
        core.pop("evidence_sha256")
        if canonical_sha256(core) != declared:
            raise ValueError(f"steps[{expected_step - 1}] self-hash differs")
        generation = _object(row.get("generation"), f"steps[{expected_step - 1}].generation")
        if generation.get("source_kind") != "synthetic" or generation.get("completion_count") != 4:
            raise ValueError("compatibility update generation contract differs")
        strict = _object(
            row.get("strict_mode_evidence"),
            f"steps[{expected_step - 1}].strict_mode_evidence",
        )
        if set(strict) != {
            "after_backward",
            "before_optimizer",
            "after_optimizer",
            "after_scheduler",
        } or any(value is not True for value in strict.values()):
            raise ValueError("compatibility strict-mode evidence differs")
        required_fields = {
            "loss_hex",
            "loss_tensor",
            "mean_kl_hex",
            "mean_kl_tensor",
            "rng_before",
            "rng_after",
            "lora_before",
            "lora_after",
            "gradients_after_backward",
            "gradients_after_clipping",
            "optimizer_before",
            "optimizer_after",
            "scheduler_before",
            "scheduler_after",
        }
        missing = sorted(required_fields - set(row))
        if missing:
            raise ValueError(f"compatibility step fields are missing: {missing}")
        for name in ("gradients_after_backward", "gradients_after_clipping"):
            gradient = _object(row.get(name), f"steps[{expected_step - 1}].{name}")
            if gradient.get("frozen_gradient_count") != 0:
                raise ValueError("compatibility frozen-base gradient evidence differs")
            count = gradient.get("lora_gradient_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("compatibility LoRA gradient evidence differs")
    replay = _validate_generation_record(packet.get("replay_generation"), "replay_generation")
    if replay.get("source_kind") != "base_replay" or replay.get("completion_count") != 4:
        raise ValueError("compatibility replay generation contract differs")
    generations = [
        _object(_object(row, f"steps[{index}]").get("generation"), f"steps[{index}].generation")
        for index, row in enumerate(rows)
    ]
    group_ids = [generation.get("group_id") for generation in generations] + [
        replay.get("group_id")
    ]
    if len(set(group_ids)) != 3:
        raise ValueError("compatibility group IDs differ")
    base_states = [
        _object(packet.get(name), name) for name in ("base_before", "base_after", "reloaded_base")
    ]
    base_hashes = [
        _require_sha256(
            state.get("base_parameter_state_sha256"),
            f"base_states[{index}].base_parameter_state_sha256",
        )
        for index, state in enumerate(base_states)
    ]
    if len(set(base_hashes)) != 1 or packet.get("base_restoration_passed") is not True:
        raise ValueError("compatibility frozen-base restoration differs")
    final_lora = _object(packet.get("final_lora"), "final_lora")
    reloaded_lora = _object(packet.get("reloaded_lora"), "reloaded_lora")
    initial_lora = _object(packet.get("initial_lora"), "initial_lora")
    initial_lora_hash = _require_sha256(
        initial_lora.get("lora_tensor_state_sha256"), "initial_lora hash"
    )
    final_lora_hash = _require_sha256(final_lora.get("lora_tensor_state_sha256"), "final_lora hash")
    if (
        final_lora_hash
        != (_require_sha256(reloaded_lora.get("lora_tensor_state_sha256"), "reloaded_lora hash"))
        or packet.get("adapter_reload_passed") is not True
    ):
        raise ValueError("compatibility adapter reload differs")
    if initial_lora_hash == final_lora_hash:
        raise ValueError("compatibility LoRA tensors did not update")
    _require_sha256(packet.get("adapter_artifact_sha256"), "adapter_artifact_sha256")
    _require_sha256(packet.get("adapter_directory_sha256"), "adapter_directory_sha256")


def write_replay_packet_new(path: Path, packet: Mapping[str, object], *, kind: ReplayKind) -> str:
    """Write one validated raw packet without overwriting prior evidence."""

    digest = validate_replay_packet(packet, expected_kind=kind)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite replay evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(packet), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return digest


def load_replay_packet(path: Path, *, expected_kind: ReplayKind) -> dict[str, object]:
    """Load and validate one packet produced by the current or a fresh process."""

    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load replay evidence {path}: {error}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("replay evidence must be a string-keyed object")
    packet = cast(dict[str, object], value)
    validate_replay_packet(packet, expected_kind=expected_kind)
    return packet


def _first_difference(left: object, right: object, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = sorted(str(key) for key in left)
        right_keys = sorted(str(key) for key in right)
        if left_keys != right_keys:
            return f"{path}: mapping keys differ"
        for key in left_keys:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, str | bytes | bytearray)
    ):
        if len(left) != len(right):
            return f"{path}: sequence length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path}: values differ"
    return None


def _diagnostic_projection(value: object) -> object:
    """Remove redundant enclosing self-hashes so diagnostics name root evidence."""

    if isinstance(value, Mapping):
        return {
            str(key): _diagnostic_projection(item)
            for key, item in value.items()
            if key not in {"packet_sha256", "evidence_sha256"}
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_diagnostic_projection(item) for item in value]
    return value


def assert_exact_replay(
    packets: Sequence[Mapping[str, object]], *, expected_kind: ReplayKind
) -> str:
    """Require byte-equivalent canonical packets across same or fresh processes."""

    if len(packets) < 2:
        raise ValueError("exact replay comparison requires at least two packets")
    normalized = [dict(packet) for packet in packets]
    hashes = [validate_replay_packet(packet, expected_kind=expected_kind) for packet in normalized]
    comparable = [_diagnostic_projection(packet) for packet in normalized]
    baseline = comparable[0]
    for index, packet in enumerate(comparable[1:], start=1):
        difference = _first_difference(baseline, packet)
        if difference is not None:
            raise RuntimeError(f"replay packet {index} differs: {difference}")
    if len(set(hashes)) != 1:
        raise RuntimeError("replay packet hashes differ despite equal payloads")
    return hashes[0]


def compare_fresh_process_packets(paths: Sequence[Path], *, expected_kind: ReplayKind) -> str:
    """Load independently written packets and require an exact common digest."""

    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("fresh-process replay paths must be distinct")
    packets = [load_replay_packet(path, expected_kind=expected_kind) for path in paths]
    return assert_exact_replay(packets, expected_kind=expected_kind)
