from __future__ import annotations

from foundry.training.token_matching import (
    TokenCensusEntry,
    _balanced_fixed_steps,
    _select_method_a_extras,
    largest_remainder_quotas,
    measure_record,
)


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        return "|".join(message["content"] for message in messages)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int | None = None,
        padding: str | None = None,
    ) -> dict[str, list[int]]:
        assert not add_special_tokens
        values = list(range(1, len(text.split()) + 1))
        if not truncation:
            return {"input_ids": values}
        assert max_length is not None and padding == "max_length"
        values = values[:max_length]
        attention = [1] * len(values)
        return {
            "input_ids": values + [0] * (max_length - len(values)),
            "attention_mask": attention + [0] * (max_length - len(values)),
        }


def _entry(index: int, arm: str, tokens: int) -> TokenCensusEntry:
    return TokenCensusEntry(
        synthetic_id=f"{arm}-{index:03d}",
        dataset_arm=arm,
        family=("bookkeeping", "rate", "discrete")[index % 3],
        submode="mode",
        difficulty=("easy", "medium", "hard")[index % 3],
        output_contract_enabled=index % 5 == 0,
        formatted_input_tokens=tokens,
        loss_bearing_tokens=tokens,
        truncated_tokens=0,
        labels_entirely_masked=False,
        formatted_example_sha256=f"{index:064x}",
    )


def test_measure_record_uses_frozen_masking_and_reports_truncation() -> None:
    record = {
        "synthetic_id": "id-1",
        "group": "targeted",
        "family": "bookkeeping",
        "mode": "inventory",
        "difficulty": "easy",
        "output_contract_enabled": True,
        "rendered_question": "one two three four five",
        "training_completion": "six seven eight nine ten",
    }
    result = measure_record(record, tokenizer=_FakeTokenizer(), max_sequence_length=12)
    assert result.formatted_input_tokens > 12
    assert result.loss_bearing_tokens == 12
    assert result.truncated_tokens == result.formatted_input_tokens - 12
    assert not result.labels_entirely_masked


def test_largest_remainder_is_exact_and_stable() -> None:
    counts = {("a", "easy", False): 2, ("b", "easy", False): 1}
    result = largest_remainder_quotas(2, counts)
    assert result == {("a", "easy", False): 1, ("b", "easy", False): 1}


def test_method_a_uses_exact_repeat_counts_and_deterministic_steps() -> None:
    generic = tuple(_entry(index, "generic_control", 100 + index % 19) for index in range(450))
    targeted = tuple(_entry(index, "targeted", 130 + index % 17) for index in range(450))
    selected, quotas, proof = _select_method_a_extras(generic, targeted)
    assert proof == "target_minimum_minus_generic_maximum_exact_lower_bound"
    assert all(sum(values.values()) == 250 for values in quotas.values())
    assert all(len(values) == 250 for values in selected.values())
    schedule = _balanced_fixed_steps(generic, selected["generic_control"])
    assert schedule == _balanced_fixed_steps(generic, selected["generic_control"])
    assert len(schedule) == 200
    assert all(len(step.occurrences) == 8 for step in schedule)
    counts = {
        synthetic_id: sum(
            occurrence.synthetic_id == synthetic_id
            for step in schedule
            for occurrence in step.occurrences
        )
        for synthetic_id in {entry.synthetic_id for entry in generic}
    }
    assert set(counts.values()) == {3, 4}
    assert sum(value == 4 for value in counts.values()) == 250
