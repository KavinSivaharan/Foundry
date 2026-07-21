"""Narrow warning-only compatibility contract for frozen top-p generation."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import random
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import wraps
from typing import Any

from foundry.training.config import canonical_sha256
from foundry.training.lora_scaling import (
    adapter_state_sha256,
    base_parameter_signature_sha256,
)

CONTRACT_ID = "foundry-warning-only-top-p-replay-v1"
EXPECTED_GENERATION_SOURCE_SHA256 = (
    "40c8e7d6adef288cd86f567b40e91ec8e95e1e916cc774e14fa4c91ad9f1105f"
)
EXPECTED_TOP_P_SOURCE_SHA256 = "ac1f86dbf01f392e3fd068c434fa30b785d55ad2f8943209185319f081445698"
EXPECTED_GENERATION_FRAGMENTS = (
    "@torch.no_grad()",
    "generation_config",
    "logits_processor",
)
EXPECTED_TOP_P_FRAGMENTS = (
    "sorted_logits.softmax(dim=-1).cumsum(dim=-1)",
    "cumulative_probs <= (1 - self.top_p)",
)
WARNING_WHITELIST_FRAGMENTS = (
    "cumsum_cuda_kernel",
    "does not have a deterministic implementation",
)
CANONICAL_WARNING_CLASS_ID = "pytorch-cuda-cumsum-determinism-warning-v1"
CANONICAL_WARNING_NORMALIZED_TEXT = (
    "cumsum_cuda_kernel does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(true, warn_only=true)'."
)
WARNING_SUFFIX_MARKERS = (
    " you can file an issue at ",
    " (triggered internally at ",
)
WARNING_NORMALIZATION_CONFIG = {
    "version": 1,
    "strip": True,
    "collapse_whitespace": True,
    "casefold": True,
    "drop_suffix_markers": list(WARNING_SUFFIX_MARKERS),
}
FROZEN_SAMPLING = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
}
WARNING_NORMALIZATION_SHA256 = canonical_sha256(WARNING_NORMALIZATION_CONFIG)
WARNING_CLASSIFICATION_CONFIG = {
    "version": 1,
    "canonical_class_id": CANONICAL_WARNING_CLASS_ID,
    "canonical_category": "builtins.UserWarning",
    "canonical_normalized_text": CANONICAL_WARNING_NORMALIZED_TEXT,
    "normalization_sha256": WARNING_NORMALIZATION_SHA256,
    "unclassified_identity": "normalized-message-sha256",
    "maximum_distinct_classes_per_call": 1,
}
WARNING_CLASSIFICATION_SHA256 = canonical_sha256(WARNING_CLASSIFICATION_CONFIG)
WARNING_WHITELIST_SHA256 = canonical_sha256(
    {
        "canonical_class_id": CANONICAL_WARNING_CLASS_ID,
        "classification_sha256": WARNING_CLASSIFICATION_SHA256,
    }
)
RNG_STATE_CONFIG = {
    "version": 2,
    "components": ["python", "numpy", "torch_cpu", "torch_cuda_all"],
    "restore_after_generation": False,
}
RNG_STATE_CONTRACT_SHA256 = canonical_sha256(RNG_STATE_CONFIG)
SAMPLING_CONTRACT_SHA256 = canonical_sha256(FROZEN_SAMPLING)
FIXTURE_SHA256 = canonical_sha256(
    [
        "strict-entry-known-cumsum-warning-success",
        "canonical-warning-identity-fails-closed",
        "multiple-normalized-warning-classes-rejected",
        "warning-filters-restored-after-success-and-exception",
        "sampling-arguments-and-return-value-preserved",
        "python-numpy-torch-rng-transition-recorded-not-restored",
        "model-and-adapter-state-unchanged",
        "strict-state-restored-after-success",
        "strict-state-restored-after-generation-exception",
        "unexpected-warning-fails-closed",
        "warning-normalization-is-frozen",
        "missing-known-warning-fails-closed",
        "nested-context-rejected",
        "backward-and-optimizer-labels-rejected",
        "installed-source-drift-rejected",
    ]
)

_ACTIVE_CONTRACT: ContextVar[bool] = ContextVar(
    "foundry_top_p_warning_only_contract", default=False
)
_ACTIVE_INSTALL: ContextVar[bool] = ContextVar("foundry_top_p_warning_only_install", default=False)
_PATCH_LOCK = threading.Lock()
_PATCH_SENTINEL = "__foundry_top_p_warning_only_wrapper__"


@dataclass(frozen=True)
class ModelAdapterState:
    """Content-free state identities captured around one generation call."""

    model_sha256: str
    adapter_sha256: str
    active_adapters: tuple[str, ...] = ("default",)
    adapter_enabled: bool | str = True
    scaling_sha256: str = "c" * 64
    training_state_sha256: str = "d" * 64


@dataclass(frozen=True)
class GenerationCallEvidence:
    """One audited warning-only generation invocation."""

    call_index: int
    warning_count: int
    warning_sha256s: tuple[str, ...]
    warning_class_ids: tuple[str, ...]
    distinct_warning_class_count: int
    warnings_whitelisted: bool
    expected_warning_present: bool
    warning_filters_before_sha256: str
    warning_filters_after_sha256: str
    warning_filters_restored: bool
    rng_before_sha256: str
    rng_after_sha256: str
    rng_advanced: bool
    model_before_sha256: str
    model_after_sha256: str
    adapter_before_sha256: str
    adapter_after_sha256: str
    active_adapters_before: tuple[str, ...]
    active_adapters_after: tuple[str, ...]
    adapter_enabled_before: bool | str
    adapter_enabled_after: bool | str
    scaling_before_sha256: str
    scaling_after_sha256: str
    training_state_before_sha256: str
    training_state_after_sha256: str
    state_unchanged: bool
    strict_entry: bool
    strict_restored: bool
    generation_completed: bool
    error_type: str | None

    def as_dict(self) -> dict[str, object]:
        """Return content-free evidence for a tracked aggregate."""

        return {
            "call_index": self.call_index,
            "warning_count": self.warning_count,
            "warning_sha256s": list(self.warning_sha256s),
            "warning_class_ids": list(self.warning_class_ids),
            "distinct_warning_class_count": self.distinct_warning_class_count,
            "warnings_whitelisted": self.warnings_whitelisted,
            "expected_warning_present": self.expected_warning_present,
            "warning_filters_before_sha256": self.warning_filters_before_sha256,
            "warning_filters_after_sha256": self.warning_filters_after_sha256,
            "warning_filters_restored": self.warning_filters_restored,
            "rng_before_sha256": self.rng_before_sha256,
            "rng_after_sha256": self.rng_after_sha256,
            "rng_advanced": self.rng_advanced,
            "model_before_sha256": self.model_before_sha256,
            "model_after_sha256": self.model_after_sha256,
            "adapter_before_sha256": self.adapter_before_sha256,
            "adapter_after_sha256": self.adapter_after_sha256,
            "active_adapters_before": list(self.active_adapters_before),
            "active_adapters_after": list(self.active_adapters_after),
            "adapter_enabled_before": self.adapter_enabled_before,
            "adapter_enabled_after": self.adapter_enabled_after,
            "scaling_before_sha256": self.scaling_before_sha256,
            "scaling_after_sha256": self.scaling_after_sha256,
            "training_state_before_sha256": self.training_state_before_sha256,
            "training_state_after_sha256": self.training_state_after_sha256,
            "state_unchanged": self.state_unchanged,
            "strict_entry": self.strict_entry,
            "strict_restored": self.strict_restored,
            "generation_completed": self.generation_completed,
            "error_type": self.error_type,
        }


def callable_source_sha256(value: Callable[..., Any]) -> str:
    """Return the SHA-256 of inspect's exact callable source."""

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise ValueError("compatibility callable source is unavailable") from error
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _validate_source(
    value: Callable[..., Any],
    *,
    expected_sha256: str,
    required_fragments: Sequence[str],
    label: str,
) -> str:
    actual = callable_source_sha256(value)
    if actual != expected_sha256:
        raise ValueError(f"{label} source hash differs")
    source = inspect.getsource(value)
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise ValueError(f"{label} source semantics differ: {missing}")
    return actual


def source_contract_sha256(
    generation_source_sha256: str = EXPECTED_GENERATION_SOURCE_SHA256,
    top_p_source_sha256: str = EXPECTED_TOP_P_SOURCE_SHA256,
) -> str:
    """Hash the two pinned source identities and required semantic fragments."""

    return canonical_sha256(
        {
            "generation_source_sha256": generation_source_sha256,
            "generation_fragments": list(EXPECTED_GENERATION_FRAGMENTS),
            "top_p_source_sha256": top_p_source_sha256,
            "top_p_fragments": list(EXPECTED_TOP_P_FRAGMENTS),
        }
    )


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _one_active_adapter(model: Any) -> tuple[str, bool | str]:
    status_method = getattr(model, "get_model_status", None)
    if not callable(status_method):
        raise TypeError("PEFT model-status API is unavailable")
    status = status_method()
    enabled = getattr(status, "enabled", None)
    if not isinstance(enabled, bool | str):
        raise TypeError("PEFT enabled state must be bool or str")
    active_value = getattr(status, "active_adapters", None)
    active: tuple[str, ...]
    if isinstance(active_value, str):
        active = (active_value,)
    elif isinstance(active_value, Sequence):
        active = tuple(str(item) for item in active_value)
    else:
        raise TypeError("PEFT active-adapter state must be a string or sequence")
    if len(active) != 1:
        raise ValueError("generation requires exactly one active adapter")
    return active[0], enabled


def _scaling_sha256(model: Any, adapter_name: str) -> str:
    payload: list[dict[str, object]] = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            payload.append(
                {
                    "module": str(name),
                    "adapter": adapter_name,
                    "scaling": float(scaling[adapter_name]),
                }
            )
    if not payload:
        raise ValueError("active adapter has no LoRA scaling state")
    return canonical_sha256(payload)


def _training_state_sha256(model: Any) -> str:
    payload: list[dict[str, object]] = []
    for name, module in model.named_modules():
        training = getattr(module, "training", None)
        if not isinstance(training, bool):
            raise TypeError(f"module lacks a Boolean training state: {name}")
        payload.append({"module": str(name), "training": training})
    if not payload:
        raise ValueError("model has no modules for training-state evidence")
    return canonical_sha256(payload)


def model_adapter_state(model: Any) -> ModelAdapterState:
    """Hash base, adapter, activation, scaling, and module train/eval state."""

    adapter_name, enabled = _one_active_adapter(model)
    return ModelAdapterState(
        model_sha256=base_parameter_signature_sha256(model),
        adapter_sha256=adapter_state_sha256(model, adapter_name),
        active_adapters=(adapter_name,),
        adapter_enabled=enabled,
        scaling_sha256=_scaling_sha256(model, adapter_name),
        training_state_sha256=_training_state_sha256(model),
    )


def _state_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    current = value
    for method_name in ("detach", "cpu", "contiguous"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        return bytes(numpy_method().tobytes())
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        return json.dumps(tolist(), separators=(",", ":")).encode("utf-8")
    return repr(current).encode("utf-8")


def _update_digest_component(digest: Any, label: str, value: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(4, "big"))
    digest.update(label_bytes)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _numpy_rng_state_bytes(numpy_random: Any) -> bytes:
    get_state = getattr(numpy_random, "get_state", None)
    if not callable(get_state):
        raise TypeError("NumPy RNG state API is unavailable")
    state = get_state()
    if not isinstance(state, tuple) or len(state) != 5:
        raise TypeError("NumPy RNG state has an unsupported shape")
    generator_name, keys, position, has_gauss, cached_gaussian = state
    if not isinstance(generator_name, str):
        raise TypeError("NumPy RNG generator name is invalid")
    payload = hashlib.sha256()
    _update_digest_component(payload, "generator", generator_name.encode("utf-8"))
    _update_digest_component(payload, "keys", _state_bytes(keys))
    _update_digest_component(payload, "position", str(int(position)).encode("ascii"))
    _update_digest_component(payload, "has_gauss", str(int(has_gauss)).encode("ascii"))
    _update_digest_component(
        payload,
        "cached_gaussian",
        float(cached_gaussian).hex().encode("ascii"),
    )
    return payload.digest()


def rng_state_sha256(torch_module: Any, *, numpy_random: Any | None = None) -> str:
    """Hash complete Python, NumPy, CPU Torch, and available CUDA RNG state."""

    if numpy_random is None:
        numpy_random = importlib.import_module("numpy").random
    digest = hashlib.sha256()
    _update_digest_component(digest, "contract", RNG_STATE_CONTRACT_SHA256.encode("ascii"))
    _update_digest_component(
        digest,
        "python",
        repr(random.getstate()).encode("utf-8"),
    )
    _update_digest_component(digest, "numpy", _numpy_rng_state_bytes(numpy_random))
    _update_digest_component(digest, "torch_cpu", _state_bytes(torch_module.get_rng_state()))
    cuda = getattr(torch_module, "cuda", None)
    available = getattr(cuda, "is_available", None)
    if cuda is not None and callable(available) and bool(available()):
        get_all = getattr(cuda, "get_rng_state_all", None)
        if not callable(get_all):
            raise TypeError("CUDA RNG state API is unavailable")
        states = list(get_all())
        _update_digest_component(digest, "torch_cuda_count", str(len(states)).encode("ascii"))
        for index, state in enumerate(states):
            _update_digest_component(digest, f"torch_cuda_{index}", _state_bytes(state))
    else:
        _update_digest_component(digest, "torch_cuda_count", b"0")
    return digest.hexdigest()


def _strict_enabled(torch_module: Any) -> bool:
    return bool(torch_module.are_deterministic_algorithms_enabled()) and not bool(
        torch_module.is_deterministic_algorithms_warn_only_enabled()
    )


def normalize_warning_message(message: str) -> str:
    """Apply the frozen warning normalization before matching or hashing."""

    normalized = " ".join(message.strip().split()).casefold()
    for marker in WARNING_SUFFIX_MARKERS:
        if marker in normalized:
            normalized = normalized.split(marker, maxsplit=1)[0]
    return normalized


def _warning_class_id(message: str, category: type[Warning]) -> str | None:
    normalized = normalize_warning_message(message)
    if category is not UserWarning:
        return None
    if normalized != CANONICAL_WARNING_NORMALIZED_TEXT:
        return None
    return CANONICAL_WARNING_CLASS_ID


def _unclassified_warning_id(message: str, category: type[Warning]) -> str:
    normalized = normalize_warning_message(message)
    category_name = f"{category.__module__}.{category.__qualname__}"
    return "unclassified:" + canonical_sha256(
        {"category": category_name, "normalized_message": normalized}
    )


def _pattern_identity(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    pattern = getattr(value, "pattern", value if isinstance(value, str) else None)
    flags = getattr(value, "flags", 0 if isinstance(value, str) else None)
    if not isinstance(pattern, str) or not isinstance(flags, int):
        raise TypeError("warning filter contains an unsupported pattern")
    return {"pattern": pattern, "flags": flags}


def warning_filters_sha256(filters: Sequence[object] | None = None) -> str:
    """Hash a stable content projection of the process warning filters."""

    active_filters = warnings.filters if filters is None else filters
    payload: list[dict[str, object]] = []
    for entry in active_filters:
        if not isinstance(entry, tuple) or len(entry) != 5:
            raise TypeError("warning filter has an unsupported shape")
        action, message, category, module, lineno = entry
        if not isinstance(action, str) or not isinstance(category, type):
            raise TypeError("warning filter action or category is invalid")
        if not issubclass(category, Warning) or not isinstance(lineno, int):
            raise TypeError("warning filter category or line number is invalid")
        payload.append(
            {
                "action": action,
                "message": _pattern_identity(message),
                "category": f"{category.__module__}.{category.__qualname__}",
                "module": _pattern_identity(module),
                "lineno": lineno,
            }
        )
    return canonical_sha256(payload)


def _sampling_values(kwargs: Mapping[str, Any]) -> dict[str, object]:
    config = kwargs.get("generation_config")
    if config is None:
        raise ValueError("frozen top-p generation requires an explicit generation_config")
    return {name: getattr(config, name, None) for name in FROZEN_SAMPLING}


class TopPWarningOnlyGenerationContract:
    """Patch only ``GenerationMixin.generate`` and audit its warning-only call."""

    def __init__(
        self,
        *,
        torch_module: Any,
        generation_owner: type[Any],
        top_p_call: Callable[..., Any],
        state_probe: Callable[[], ModelAdapterState] | None = None,
        numpy_random: Any | None = None,
        expected_generation_sha256: str = EXPECTED_GENERATION_SOURCE_SHA256,
        expected_top_p_sha256: str = EXPECTED_TOP_P_SOURCE_SHA256,
        generation_fragments: Sequence[str] = EXPECTED_GENERATION_FRAGMENTS,
        top_p_fragments: Sequence[str] = EXPECTED_TOP_P_FRAGMENTS,
    ) -> None:
        self.torch_module = torch_module
        self.generation_owner = generation_owner
        self.top_p_call = top_p_call
        self.expected_generation_sha256 = expected_generation_sha256
        self.expected_top_p_sha256 = expected_top_p_sha256
        self.generation_fragments = tuple(generation_fragments)
        self.top_p_fragments = tuple(top_p_fragments)
        self._state_probe = state_probe
        self._numpy_random = (
            importlib.import_module("numpy").random if numpy_random is None else numpy_random
        )
        self._records: list[GenerationCallEvidence] = []
        self._source_contract_sha256: str | None = None

    def bind_state_probe(self, probe: Callable[[], ModelAdapterState]) -> None:
        """Bind one model/adapter state probe before the first generation call."""

        if self._state_probe is not None or self._records:
            raise RuntimeError("generation state probe is already bound or generation has started")
        self._state_probe = probe

    def release_state_probe(self) -> None:
        """Sever the model probe after all audited generation calls are complete."""

        if _ACTIVE_INSTALL.get() or _ACTIVE_CONTRACT.get():
            raise RuntimeError(
                "generation state probe cannot be released while the contract is active"
            )
        if self._state_probe is None:
            raise RuntimeError("generation state probe is not bound")
        if not self._records:
            raise RuntimeError("generation state probe cannot be released before generation")
        self._state_probe = None

    def _validate_contract_sources(self) -> None:
        original_generate = getattr(self.generation_owner, "generate", None)
        if not callable(original_generate):
            raise TypeError("generation owner has no callable generate method")
        generation_hash = _validate_source(
            original_generate,
            expected_sha256=self.expected_generation_sha256,
            required_fragments=self.generation_fragments,
            label="generation",
        )
        top_p_hash = _validate_source(
            self.top_p_call,
            expected_sha256=self.expected_top_p_sha256,
            required_fragments=self.top_p_fragments,
            label="top-p warper",
        )
        self._source_contract_sha256 = source_contract_sha256(generation_hash, top_p_hash)

    def _run_generate(
        self,
        original_generate: Callable[..., Any],
        model: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _ACTIVE_CONTRACT.get():
            raise RuntimeError("nested warning-only generation calls are prohibited")
        if self._state_probe is None:
            raise RuntimeError("model/adapter state probe must be bound before generation")
        if not _strict_enabled(self.torch_module):
            raise RuntimeError("warning-only generation requires strict deterministic entry")
        sampling = _sampling_values(kwargs)
        if sampling != FROZEN_SAMPLING:
            raise ValueError("generation sampling values differ from the frozen top-p contract")
        before_state = self._state_probe()
        _require_sha256(before_state.model_sha256, "model state")
        _require_sha256(before_state.adapter_sha256, "adapter state")
        _require_sha256(before_state.scaling_sha256, "adapter scaling state")
        _require_sha256(before_state.training_state_sha256, "training state")
        if len(before_state.active_adapters) != 1 or before_state.adapter_enabled is not True:
            raise RuntimeError("generation requires one uniformly enabled active adapter")
        rng_before = rng_state_sha256(self.torch_module, numpy_random=self._numpy_random)
        warning_filters_before = warning_filters_sha256()
        active_token: Token[bool] = _ACTIVE_CONTRACT.set(True)
        captured: list[warnings.WarningMessage] = []
        result: Any = None
        generation_error: BaseException | None = None
        try:
            self.torch_module.use_deterministic_algorithms(True, warn_only=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = original_generate(model, *args, **kwargs)
                except BaseException as error:  # pragma: no cover - re-raised after audit
                    generation_error = error
                captured = list(caught)
        finally:
            self.torch_module.use_deterministic_algorithms(True, warn_only=False)
            _ACTIVE_CONTRACT.reset(active_token)

        strict_restored = _strict_enabled(self.torch_module)
        after_state = self._state_probe()
        rng_after = rng_state_sha256(self.torch_module, numpy_random=self._numpy_random)
        warning_filters_after = warning_filters_sha256()
        messages = [str(item.message) for item in captured]
        warning_class_ids = tuple(
            _warning_class_id(message, item.category)
            or _unclassified_warning_id(message, item.category)
            for message, item in zip(messages, captured, strict=True)
        )
        distinct_warning_class_count = len(set(warning_class_ids))
        unexpected = [
            class_id for class_id in warning_class_ids if class_id != CANONICAL_WARNING_CLASS_ID
        ]
        state_unchanged = before_state == after_state
        warnings_whitelisted = not unexpected
        expected_warning_present = CANONICAL_WARNING_CLASS_ID in warning_class_ids
        warning_filters_restored = warning_filters_before == warning_filters_after
        evidence = GenerationCallEvidence(
            call_index=len(self._records) + 1,
            warning_count=len(messages),
            warning_sha256s=tuple(
                hashlib.sha256(normalize_warning_message(message).encode("utf-8")).hexdigest()
                for message in messages
            ),
            warning_class_ids=warning_class_ids,
            distinct_warning_class_count=distinct_warning_class_count,
            warnings_whitelisted=warnings_whitelisted,
            expected_warning_present=expected_warning_present,
            warning_filters_before_sha256=warning_filters_before,
            warning_filters_after_sha256=warning_filters_after,
            warning_filters_restored=warning_filters_restored,
            rng_before_sha256=rng_before,
            rng_after_sha256=rng_after,
            rng_advanced=rng_before != rng_after,
            model_before_sha256=before_state.model_sha256,
            model_after_sha256=after_state.model_sha256,
            adapter_before_sha256=before_state.adapter_sha256,
            adapter_after_sha256=after_state.adapter_sha256,
            active_adapters_before=before_state.active_adapters,
            active_adapters_after=after_state.active_adapters,
            adapter_enabled_before=before_state.adapter_enabled,
            adapter_enabled_after=after_state.adapter_enabled,
            scaling_before_sha256=before_state.scaling_sha256,
            scaling_after_sha256=after_state.scaling_sha256,
            training_state_before_sha256=before_state.training_state_sha256,
            training_state_after_sha256=after_state.training_state_sha256,
            state_unchanged=state_unchanged,
            strict_entry=True,
            strict_restored=strict_restored,
            generation_completed=generation_error is None,
            error_type=None if generation_error is None else type(generation_error).__name__,
        )
        self._records.append(evidence)
        if not strict_restored:
            raise RuntimeError("strict deterministic mode was not restored after generation")
        if not state_unchanged:
            raise RuntimeError("model or adapter state changed during generation")
        if not warning_filters_restored:
            raise RuntimeError("warning filters were not restored after generation")
        if distinct_warning_class_count > 1:
            raise RuntimeError("generation emitted multiple distinct normalized warning classes")
        if unexpected:
            raise RuntimeError("generation emitted a non-whitelisted warning")
        if generation_error is not None:
            raise generation_error
        if not messages:
            raise RuntimeError("top-p generation emitted no deterministic cumsum warning")
        return result

    @contextmanager
    def install(self, operation_label: str = "generation") -> Iterator[None]:
        """Install the audited generate wrapper for one stock trainer operation."""

        if operation_label != "generation":
            raise ValueError("warning-only compatibility is permitted only for generation")
        if _ACTIVE_INSTALL.get():
            raise RuntimeError("nested warning-only generation contexts are prohibited")
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError("a concurrent warning-only generation context is active")
        install_token: Token[bool] = _ACTIVE_INSTALL.set(True)
        original_generate = getattr(self.generation_owner, "generate", None)
        try:
            if not callable(original_generate):
                raise TypeError("generation owner has no callable generate method")
            if bool(getattr(original_generate, _PATCH_SENTINEL, False)):
                raise RuntimeError("generation owner is already patched")
            self._validate_contract_sources()
            if not _strict_enabled(self.torch_module):
                raise RuntimeError("warning-only context requires strict deterministic entry")

            @wraps(original_generate)
            def audited_generate(model: Any, *args: Any, **kwargs: Any) -> Any:
                return self._run_generate(original_generate, model, args, kwargs)

            setattr(audited_generate, _PATCH_SENTINEL, True)
            self.generation_owner.generate = audited_generate
            try:
                yield
            finally:
                self.generation_owner.generate = original_generate
                if not _strict_enabled(self.torch_module):
                    self.torch_module.use_deterministic_algorithms(True, warn_only=False)
                    raise RuntimeError("strict deterministic mode changed outside generation")
        finally:
            _ACTIVE_INSTALL.reset(install_token)
            _PATCH_LOCK.release()

    def evidence(self, *, require_calls: bool = True) -> dict[str, object]:
        """Return hash-bound aggregate evidence without warning or model content."""

        if self._source_contract_sha256 is None:
            raise RuntimeError("warning-only generation source contract was not installed")
        if require_calls and not self._records:
            raise RuntimeError("warning-only generation did not intercept any generate call")
        records = [record.as_dict() for record in self._records]
        payload: dict[str, object] = {
            "schema_version": 2,
            "contract_id": CONTRACT_ID,
            "source_contract_sha256": self._source_contract_sha256,
            "warning_normalization_sha256": WARNING_NORMALIZATION_SHA256,
            "warning_classification_sha256": WARNING_CLASSIFICATION_SHA256,
            "canonical_warning_class_id": CANONICAL_WARNING_CLASS_ID,
            "warning_whitelist_sha256": WARNING_WHITELIST_SHA256,
            "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
            "rng_state_contract_sha256": RNG_STATE_CONTRACT_SHA256,
            "fixture_sha256": FIXTURE_SHA256,
            "generation_calls": len(records),
            "warning_count": sum(record.warning_count for record in self._records),
            "all_warnings_whitelisted": all(
                record.warnings_whitelisted for record in self._records
            ),
            "all_expected_warnings_present": all(
                record.expected_warning_present for record in self._records
            ),
            "all_single_warning_class": all(
                record.distinct_warning_class_count == 1 for record in self._records
            ),
            "all_warning_filters_restored": all(
                record.warning_filters_restored for record in self._records
            ),
            "all_state_unchanged": all(record.state_unchanged for record in self._records),
            "all_adapter_runtime_unchanged": all(
                record.active_adapters_before == record.active_adapters_after
                and record.adapter_enabled_before == record.adapter_enabled_after
                for record in self._records
            ),
            "all_scaling_unchanged": all(
                record.scaling_before_sha256 == record.scaling_after_sha256
                for record in self._records
            ),
            "all_training_state_unchanged": all(
                record.training_state_before_sha256 == record.training_state_after_sha256
                for record in self._records
            ),
            "all_strict_entries": all(record.strict_entry for record in self._records),
            "all_strict_restorations": all(record.strict_restored for record in self._records),
            "sampling_preserved": True,
            "rng_restored": False,
            "rng_transitions_sha256": canonical_sha256(
                [
                    {
                        "before": record.rng_before_sha256,
                        "after": record.rng_after_sha256,
                    }
                    for record in self._records
                ]
            ),
            "call_evidence_sha256": canonical_sha256(records),
        }
        payload["evidence_sha256"] = canonical_sha256(payload)
        return payload

    def call_records(self) -> tuple[GenerationCallEvidence, ...]:
        """Return immutable ordered per-call records for group-level evidence."""

        return tuple(self._records)
