"""Assistant-only SFT v3 completion normalization and token labeling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from foundry.training.config import assistant_only_v3_messages


@dataclass(frozen=True)
class AssistantOnlyEvidence:
    """Content-free invariants for one corrected SFT record."""

    formatted_text_sha256: str
    normalized_completion_sha256: str
    formatted_tokens: int
    assistant_loss_tokens: int
    prefix_tokens: int
    eos_token_index: int
    truncated_tokens: int
    system_user_header_loss_tokens: int
    padding_loss_tokens: int
    post_eos_loss_tokens: int
    final_answer_line_count: int
    decoded_labels_equal_completion_plus_eos: bool
    question_in_decoded_labels: bool
    metadata_in_decoded_labels: bool


def normalized_assistant_completion(record: dict[str, Any]) -> str:
    """Preserve the deterministic trace and append exactly one canonical terminal line."""

    trace_value = record.get("deterministic_solution_trace")
    if not isinstance(trace_value, list) or not all(isinstance(item, str) for item in trace_value):
        raise ValueError("deterministic solution trace must be a string list")
    trace = "\n".join(cast(list[str], trace_value))
    if not trace.strip():
        raise ValueError("deterministic solution trace must be non-empty")
    answer = str(record["canonical_final_answer"])
    terminal = f"Final answer: {answer}"
    old_completion = str(record["training_completion"])
    if old_completion not in {trace, f"{trace}\n{terminal}"}:
        raise ValueError("stored completion differs from its frozen trace/answer contract")
    completion = f"{trace}\n{terminal}"
    if completion.splitlines().count(terminal) != 1 or completion.splitlines()[-1] != terminal:
        raise ValueError("normalized completion requires exactly one terminal line")
    return completion


def assistant_only_tokenize(
    record: dict[str, Any], tokenizer: Any, *, max_length: int
) -> tuple[dict[str, list[int]], AssistantOnlyEvidence]:
    """Tokenize a corrected record with loss only on assistant content and final EOS."""

    question = str(record["rendered_question"])
    completion = normalized_assistant_completion(record)
    messages = assistant_only_v3_messages(question, completion)
    generation_prefix = cast(
        str,
        tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True),
    )
    full_text = cast(
        str,
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
    )
    completion_text = f"{generation_prefix}{completion}"
    if not full_text.startswith(completion_text):
        raise ValueError("official chat template does not preserve assistant prefix/content")

    def ids(text: str) -> list[int]:
        return cast(
            list[int],
            tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"],
        )

    prefix_ids = ids(generation_prefix)
    completion_ids = ids(completion_text)
    original_ids = ids(full_text)
    if completion_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("assistant content token boundary is unstable")
    if original_ids[: len(completion_ids)] != completion_ids:
        raise ValueError("assistant EOS token boundary is unstable")
    eos_token_id = cast(int | None, tokenizer.eos_token_id)
    if eos_token_id is None or len(original_ids) <= len(completion_ids):
        raise ValueError("tokenizer lacks the required final assistant EOS")
    if original_ids[len(completion_ids)] != eos_token_id:
        raise ValueError("first post-completion token is not the final assistant EOS")
    if len(original_ids) > max_length:
        raise ValueError("assistant-only record would truncate its completion or EOS")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = cast(list[int], encoded["input_ids"])
    attention_mask = cast(list[int], encoded["attention_mask"])
    labels = [-100] * max_length
    for index in range(len(prefix_ids), len(completion_ids)):
        labels[index] = input_ids[index]
    labels[len(completion_ids)] = input_ids[len(completion_ids)]
    loss_ids = [label for label in labels if label != -100]
    intended_ids = [*original_ids[len(prefix_ids) : len(completion_ids)], eos_token_id]
    decoded = cast(str, tokenizer.decode(loss_ids, skip_special_tokens=False))
    metadata_values = [
        str(record[key])
        for key in (
            "latent_program_sha256",
            "semantic_ir_sha256",
            "semantic_frame",
            "template_id",
            "sentence_plan_id",
        )
        if isinstance(record.get(key), str) and len(str(record[key])) >= 8
    ]
    evidence = AssistantOnlyEvidence(
        formatted_text_sha256=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        normalized_completion_sha256=hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        formatted_tokens=len(original_ids),
        assistant_loss_tokens=len(loss_ids),
        prefix_tokens=len(prefix_ids),
        eos_token_index=len(completion_ids),
        truncated_tokens=0,
        system_user_header_loss_tokens=sum(label != -100 for label in labels[: len(prefix_ids)]),
        padding_loss_tokens=sum(
            label != -100 for label, mask in zip(labels, attention_mask, strict=True) if mask == 0
        ),
        post_eos_loss_tokens=sum(label != -100 for label in labels[len(completion_ids) + 1 :]),
        final_answer_line_count=sum(
            line.startswith("Final answer:") for line in completion.splitlines()
        ),
        decoded_labels_equal_completion_plus_eos=loss_ids == intended_ids,
        question_in_decoded_labels=question in decoded,
        metadata_in_decoded_labels=any(value in decoded for value in metadata_values),
    )
    if not (
        evidence.assistant_loss_tokens > 1
        and evidence.system_user_header_loss_tokens == 0
        and evidence.padding_loss_tokens == 0
        and evidence.post_eos_loss_tokens == 0
        and evidence.final_answer_line_count == 1
        and evidence.decoded_labels_equal_completion_plus_eos
        and not evidence.question_in_decoded_labels
        and not evidence.metadata_in_decoded_labels
    ):
        raise ValueError("assistant-only SFT invariants failed")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }, evidence


def format_and_tokenize_assistant_only(
    records: list[dict[str, Any]], tokenizer: Any, *, max_length: int
) -> tuple[list[dict[str, list[int]]], int, int, tuple[AssistantOnlyEvidence, ...]]:
    """Tokenize a stable record sequence and total only loss-bearing assistant tokens."""

    values: list[dict[str, list[int]]] = []
    evidence: list[AssistantOnlyEvidence] = []
    for record in records:
        value, item = assistant_only_tokenize(record, tokenizer, max_length=max_length)
        values.append(value)
        evidence.append(item)
    return (
        values,
        sum(item.assistant_loss_tokens for item in evidence),
        sum(item.truncated_tokens > 0 for item in evidence),
        tuple(evidence),
    )
