from __future__ import annotations

from typing import Any

import pytest

from foundry.training import grpo_trainer as gt


class _BoolRows:
    def __init__(self, values: list[bool]) -> None:
        self.values = list(values)

    def detach(self) -> _BoolRows:
        return self

    def cpu(self) -> _BoolRows:
        return self

    def tolist(self) -> list[bool]:
        return list(self.values)


class _EqualityRows:
    def __init__(self, values: list[list[bool]]) -> None:
        self.values = [list(row) for row in values]

    def any(self, *, dim: int) -> _BoolRows:
        assert dim == 1
        return _BoolRows([any(row) for row in self.values])


class _CompletionIds:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = [list(row) for row in values]

    def __eq__(self, other: object) -> _EqualityRows:
        assert isinstance(other, int)
        return _EqualityRows([[token == other for token in row] for row in self.values])

    def __len__(self) -> int:
        return len(self.values)

    def size(self, dim: int) -> int:
        assert dim == 0
        return len(self.values)


class _ProcessingClass:
    eos_token_id = 2

    def __init__(self) -> None:
        self.decode_inputs: list[_CompletionIds] = []

    def batch_decode(
        self, completion_ids: _CompletionIds, *, skip_special_tokens: bool
    ) -> list[str]:
        assert skip_special_tokens is True
        self.decode_inputs.append(completion_ids)
        return ["decoded" for _ in completion_ids.values]


class _BaseTrainer:
    def __init__(self, *, fail_after_reward: bool = False, decode_twice: bool = False) -> None:
        self.processing_class = _ProcessingClass()
        self.fail_after_reward = fail_after_reward
        self.decode_twice = decode_twice
        self.flags_seen: tuple[bool, ...] | None = None
        self.completion_ids_seen: _CompletionIds | None = None

    def _generate_and_score_completions(self, inputs: dict[str, Any]) -> dict[str, Any]:
        completion_ids = inputs["completion_ids"]
        self.completion_ids_seen = completion_ids
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if self.decode_twice:
            self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        self.flags_seen = gt.get_active_truncation_flags(expected_count=len(completions))
        if self.fail_after_reward:
            raise RuntimeError("fixture reward failure")
        return {"completions": completions, "source": completion_ids}


def _trainer_class() -> type[Any]:
    method_hash = gt.callable_source_sha256(_BaseTrainer._generate_and_score_completions)
    return gt.make_truncation_aware_grpo_trainer(
        _BaseTrainer,
        expected_method_sha256=method_hash,
        required_fragments=("batch_decode", "get_active_truncation_flags"),
    )


def test_exact_eos_absence_flags_preserve_row_order_and_input() -> None:
    completion_ids = _CompletionIds([[5, 2, 7], [8, 9, 10], [2, 2, 2], [11, 12, 13]])
    before = [list(row) for row in completion_ids.values]
    assert gt.exact_eos_absence_flags(completion_ids, 2) == (False, True, False, True)
    assert completion_ids.values == before


def test_strict_getter_fails_outside_active_scoring() -> None:
    with pytest.raises(RuntimeError, match="unavailable outside reward scoring"):
        gt.get_active_truncation_flags()
    with pytest.raises(ValueError, match="cannot be negative"):
        gt.get_active_truncation_flags(expected_count=-1)


def test_trainer_exposes_exact_flags_during_stock_reward_and_cleans_context() -> None:
    trainer_class = _trainer_class()
    trainer = trainer_class()
    completion_ids = _CompletionIds([[4, 2], [4, 5], [2, 5], [6, 7]])
    before = [list(row) for row in completion_ids.values]
    had_instance_decode = "batch_decode" in vars(trainer.processing_class)

    result = trainer._generate_and_score_completions({"completion_ids": completion_ids})

    assert trainer.flags_seen == (False, True, False, True)
    assert result["source"] is completion_ids
    assert trainer.completion_ids_seen is completion_ids
    assert completion_ids.values == before
    assert ("batch_decode" in vars(trainer.processing_class)) is had_instance_decode
    assert trainer.processing_class.batch_decode.__func__ is _ProcessingClass.batch_decode
    with pytest.raises(RuntimeError, match="unavailable outside reward scoring"):
        gt.get_active_truncation_flags()


def test_context_and_method_restore_when_stock_reward_raises() -> None:
    trainer_class = _trainer_class()
    trainer = trainer_class(fail_after_reward=True)
    completion_ids = _CompletionIds([[1, 2], [3, 4]])
    before_method = trainer.processing_class.batch_decode

    with pytest.raises(RuntimeError, match="fixture reward failure"):
        trainer._generate_and_score_completions({"completion_ids": completion_ids})

    assert trainer.flags_seen == (False, True)
    assert trainer.processing_class.batch_decode == before_method
    assert "batch_decode" not in vars(trainer.processing_class)
    with pytest.raises(RuntimeError, match="unavailable outside reward scoring"):
        gt.get_active_truncation_flags()


def test_preexisting_instance_decode_is_restored_exactly() -> None:
    trainer_class = _trainer_class()
    trainer = trainer_class()
    calls: list[_CompletionIds] = []

    def instance_decode(completion_ids: _CompletionIds, **kwargs: Any) -> list[str]:
        assert kwargs == {"skip_special_tokens": True}
        calls.append(completion_ids)
        return ["instance" for _ in completion_ids.values]

    trainer.processing_class.batch_decode = instance_decode
    trainer._generate_and_score_completions({"completion_ids": _CompletionIds([[1, 2], [3, 4]])})
    assert trainer.processing_class.batch_decode is instance_decode
    assert len(calls) == 1


def test_repeated_stock_decode_fails_closed_and_restores_everything() -> None:
    trainer_class = _trainer_class()
    trainer = trainer_class(decode_twice=True)
    with pytest.raises(RuntimeError, match="more than once"):
        trainer._generate_and_score_completions(
            {"completion_ids": _CompletionIds([[1, 2], [3, 4]])}
        )
    assert "batch_decode" not in vars(trainer.processing_class)
    with pytest.raises(RuntimeError, match="unavailable outside reward scoring"):
        gt.get_active_truncation_flags()


def test_factory_fails_closed_on_source_or_semantic_drift() -> None:
    method_hash = gt.callable_source_sha256(_BaseTrainer._generate_and_score_completions)
    with pytest.raises(ValueError, match="source hash differs"):
        gt.make_truncation_aware_grpo_trainer(
            _BaseTrainer, expected_method_sha256="0" * 64, required_fragments=()
        )
    with pytest.raises(ValueError, match="semantics differ"):
        gt.make_truncation_aware_grpo_trainer(
            _BaseTrainer,
            expected_method_sha256=method_hash,
            required_fragments=("missing semantic marker",),
        )


def test_getter_rejects_count_mismatch_during_scoring() -> None:
    class _WrongCountBase(_BaseTrainer):
        def _generate_and_score_completions(self, inputs: dict[str, Any]) -> Any:
            completion_ids = inputs["completion_ids"]
            self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            return gt.get_active_truncation_flags(expected_count=99)

    method_hash = gt.callable_source_sha256(_WrongCountBase._generate_and_score_completions)
    trainer_class = gt.make_truncation_aware_grpo_trainer(
        _WrongCountBase,
        expected_method_sha256=method_hash,
        required_fragments=("batch_decode",),
    )
    with pytest.raises(RuntimeError, match="count differs"):
        trainer_class()._generate_and_score_completions(
            {"completion_ids": _CompletionIds([[1, 2], [3, 4]])}
        )
    with pytest.raises(RuntimeError, match="unavailable outside reward scoring"):
        gt.get_active_truncation_flags()
