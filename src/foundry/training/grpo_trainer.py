"""Audited TRL hook exposing exact truncation flags to deterministic rewards."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Sequence
from contextvars import ContextVar, Token
from functools import wraps
from typing import Any

EXPECTED_GENERATE_AND_SCORE_SOURCE_SHA256 = (
    "688cb0ed965eee96bd9a985fdd185f63f984ee81eaab7bbfec2519f21e06331b"
)
EXPECTED_GENERATE_AND_SCORE_FRAGMENTS = (
    "is_eos = completion_ids == self.processing_class.eos_token_id",
    "truncated_completions = ~is_eos.any(dim=1)",
    "self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)",
    "output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)",
)

_ACTIVE_TRUNCATION_FLAGS: ContextVar[tuple[bool, ...] | None] = ContextVar(
    "foundry_grpo_truncation_flags", default=None
)


def get_active_truncation_flags(*, expected_count: int | None = None) -> tuple[bool, ...]:
    """Return exact EOS-absence flags while stock TRL reward callbacks are active."""

    if expected_count is not None and expected_count < 0:
        raise ValueError("expected truncation-flag count cannot be negative")
    flags = _ACTIVE_TRUNCATION_FLAGS.get()
    if flags is None:
        raise RuntimeError("exact GRPO truncation flags are unavailable outside reward scoring")
    if expected_count is not None:
        if len(flags) != expected_count:
            raise RuntimeError(
                "exact GRPO truncation-flag count differs: "
                f"expected {expected_count}, got {len(flags)}"
            )
    return flags


def _tensor_rows(value: Any) -> int:
    size = getattr(value, "size", None)
    if callable(size):
        rows = int(size(0))
    else:
        rows = len(value)
    if rows < 1:
        raise ValueError("completion IDs must contain at least one row")
    return rows


def _row_has_eos_values(completion_ids: Any, eos_token_id: int) -> tuple[bool, ...]:
    equality = completion_ids == eos_token_id
    any_method = getattr(equality, "any", None)
    if not callable(any_method):
        raise TypeError("completion-ID equality result does not support row-wise any")
    row_has_eos = any_method(dim=1)
    for method_name in ("detach", "cpu"):
        method = getattr(row_has_eos, method_name, None)
        if callable(method):
            row_has_eos = method()
    tolist = getattr(row_has_eos, "tolist", None)
    if not callable(tolist):
        raise TypeError("row-wise EOS result cannot be converted to a list")
    raw_values = tolist()
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, str | bytes | bytearray):
        raise TypeError("row-wise EOS result must be a flat sequence")
    values: list[bool] = []
    for value in raw_values:
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            raise TypeError("row-wise EOS result must not be nested")
        values.append(bool(value))
    if len(values) != _tensor_rows(completion_ids):
        raise RuntimeError("row-wise EOS result count differs from completion batch size")
    return tuple(values)


def exact_eos_absence_flags(completion_ids: Any, eos_token_id: int) -> tuple[bool, ...]:
    """Return one ordered flag per row; true means no EOS token was generated."""

    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise TypeError("EOS token ID must be an integer")
    return tuple(not has_eos for has_eos in _row_has_eos_values(completion_ids, eos_token_id))


def callable_source_sha256(value: Callable[..., Any]) -> str:
    """Hash the source returned by inspect for an installed trainer method."""

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise ValueError("trainer generation method source is unavailable") from error
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def make_truncation_aware_grpo_trainer(
    base_trainer_class: type[Any],
    *,
    expected_method_sha256: str = EXPECTED_GENERATE_AND_SCORE_SOURCE_SHA256,
    required_fragments: Sequence[str] = EXPECTED_GENERATE_AND_SCORE_FRAGMENTS,
) -> type[Any]:
    """Create a subclass that exposes truncation flags without changing stock TRL logic."""

    method = getattr(base_trainer_class, "_generate_and_score_completions", None)
    if not callable(method):
        raise TypeError("base trainer has no callable _generate_and_score_completions method")
    actual_sha256 = callable_source_sha256(method)
    if actual_sha256 != expected_method_sha256:
        raise ValueError("TRL generation-and-scoring method source hash differs")
    source = inspect.getsource(method)
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise ValueError(f"TRL generation-and-scoring semantics differ: {missing}")

    class TruncationAwareGRPOTrainer(base_trainer_class):  # type: ignore[misc]
        """Stock GRPOTrainer with a temporary decode interception for reward metadata."""

        def _generate_and_score_completions(self, inputs: Any) -> Any:
            processing_class = self.processing_class
            if _ACTIVE_TRUNCATION_FLAGS.get() is not None:
                raise RuntimeError("nested GRPO truncation contexts are prohibited")

            had_instance_decode = "batch_decode" in vars(processing_class)
            prior_instance_decode = vars(processing_class).get("batch_decode")
            original_decode = processing_class.batch_decode
            context_token: Token[tuple[bool, ...] | None] | None = None
            decode_calls = 0

            @wraps(original_decode)
            def audited_batch_decode(completion_ids: Any, *args: Any, **kwargs: Any) -> Any:
                nonlocal context_token, decode_calls
                decode_calls += 1
                if decode_calls != 1:
                    raise RuntimeError("stock TRL completion decoding occurred more than once")
                flags = exact_eos_absence_flags(completion_ids, int(processing_class.eos_token_id))
                context_token = _ACTIVE_TRUNCATION_FLAGS.set(flags)
                return original_decode(completion_ids, *args, **kwargs)

            processing_class.batch_decode = audited_batch_decode
            try:
                result = super()._generate_and_score_completions(inputs)
                if decode_calls != 1 or context_token is None:
                    raise RuntimeError("stock TRL did not expose one completion decode batch")
                return result
            finally:
                try:
                    if had_instance_decode:
                        processing_class.batch_decode = prior_instance_decode
                    else:
                        del processing_class.batch_decode
                finally:
                    if context_token is not None:
                        _ACTIVE_TRUNCATION_FLAGS.reset(context_token)

    TruncationAwareGRPOTrainer.__name__ = "TruncationAwareGRPOTrainer"
    return TruncationAwareGRPOTrainer
