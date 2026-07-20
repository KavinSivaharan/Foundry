from __future__ import annotations

from pathlib import Path
from typing import Any

from foundry.training.assistant_only import (
    assistant_only_tokenize,
    normalized_assistant_completion,
)


def _record(*, old_terminal: bool) -> dict[str, object]:
    trace = ["Add 7 and 5 without rounding: 7 + 5 = 12."]
    completion = trace[0] + ("\nFinal answer: 12" if old_terminal else "")
    return {
        "rendered_question": "An original fixture asks for a sum.",
        "deterministic_solution_trace": trace,
        "canonical_final_answer": "12",
        "training_completion": completion,
        "latent_program_sha256": "a" * 64,
        "semantic_ir_sha256": "b" * 64,
        "semantic_frame": "fixture_frame",
        "template_id": "fixture_template",
        "sentence_plan_id": "fixture_plan",
    }


def _tokenizer() -> Any:
    from transformers import AutoTokenizer

    model = Path(
        "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_normalization_adds_or_preserves_one_terminal_line() -> None:
    without = normalized_assistant_completion(_record(old_terminal=False))
    with_line = normalized_assistant_completion(_record(old_terminal=True))
    assert without == with_line
    assert without.splitlines()[-1] == "Final answer: 12"
    assert sum(line.startswith("Final answer:") for line in without.splitlines()) == 1


def test_real_qwen_template_masks_only_assistant_completion_and_eos() -> None:
    value, evidence = assistant_only_tokenize(
        _record(old_terminal=False), _tokenizer(), max_length=512
    )
    assert evidence.system_user_header_loss_tokens == 0
    assert evidence.padding_loss_tokens == 0
    assert evidence.post_eos_loss_tokens == 0
    assert evidence.final_answer_line_count == 1
    assert evidence.decoded_labels_equal_completion_plus_eos is True
    assert evidence.question_in_decoded_labels is False
    assert evidence.metadata_in_decoded_labels is False
    assert sum(label != -100 for label in value["labels"]) == evidence.assistant_loss_tokens
    assert value["labels"][evidence.eos_token_index] != -100
