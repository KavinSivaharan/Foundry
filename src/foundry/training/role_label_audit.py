"""Role-aware reconstruction of the original all-nonpadding SFT label mask."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from foundry.training.config import SFT_SYSTEM_PROMPT, canonical_sha256, sft_messages

ROLE_NAMES = (
    "system_message",
    "user_message",
    "assistant_header",
    "assistant_reasoning",
    "assistant_final_answer",
    "eos",
    "padding",
)
INTERNAL_METADATA_KEYS = (
    "latent_program_sha256",
    "semantic_ir_sha256",
    "semantic_frame",
    "template_id",
    "sentence_plan_id",
    "quota_cell_id",
)


@dataclass(frozen=True)
class TokenSpan:
    """One half-open token range with a stable semantic role."""

    role: str
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of tokens in the span."""

        return self.end - self.start


def _ids(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return cast(list[int], value)


def _require_prefix(prefix: list[int], whole: list[int], name: str) -> None:
    if whole[: len(prefix)] != prefix:
        raise ValueError(f"chat-template token boundary is unstable at {name}")


def _token_boundary(tokenizer: Any, text: str, character_index: int, whole: list[int]) -> int:
    prefix = _ids(tokenizer, text[:character_index])
    _require_prefix(prefix, whole, f"character offset {character_index}")
    return len(prefix)


def reconstruct_original_record(
    record: dict[str, Any], tokenizer: Any, *, max_length: int
) -> dict[str, Any]:
    """Reproduce the original full-chat labels and assign every token a role."""

    question = str(record["rendered_question"])
    completion = str(record["training_completion"])
    messages = sft_messages(question, completion)
    system_text = cast(
        str,
        tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=False),
    )
    user_text = cast(
        str,
        tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=False),
    )
    generation_prefix = cast(
        str,
        tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True),
    )
    full_text = cast(
        str,
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
    )
    completion_text = f"{generation_prefix}{completion}"

    system_ids = _ids(tokenizer, system_text)
    user_ids = _ids(tokenizer, user_text)
    prefix_ids = _ids(tokenizer, generation_prefix)
    completion_ids = _ids(tokenizer, completion_text)
    original_ids = _ids(tokenizer, full_text)
    _require_prefix(system_ids, user_ids, "system/user")
    _require_prefix(user_ids, prefix_ids, "user/assistant header")
    _require_prefix(prefix_ids, completion_ids, "assistant content")
    _require_prefix(completion_ids, original_ids, "assistant EOS")
    if not full_text.startswith(completion_text):
        raise ValueError("full chat is not generation prefix plus completion and EOS")

    final_lines = [
        index
        for index, line in enumerate(completion.splitlines(keepends=True))
        if line.startswith("Final answer:")
    ]
    completion_lines = completion.splitlines(keepends=True)
    final_character_start = (
        sum(len(line) for line in completion_lines[: final_lines[0]])
        if len(final_lines) == 1
        else len(completion)
    )
    reasoning_text = f"{generation_prefix}{completion[:final_character_start]}"
    reasoning_ids = _ids(tokenizer, reasoning_text)
    _require_prefix(reasoning_ids, completion_ids, "reasoning/final answer")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = cast(list[int], encoded["input_ids"])
    attention_mask = cast(list[int], encoded["attention_mask"])
    labels = [
        token if mask else -100 for token, mask in zip(input_ids, attention_mask, strict=True)
    ]
    nonpadding = sum(attention_mask)

    raw_spans = (
        TokenSpan("system_message", 0, len(system_ids)),
        TokenSpan("user_message", len(system_ids), len(user_ids)),
        TokenSpan("assistant_header", len(user_ids), len(prefix_ids)),
        TokenSpan("assistant_reasoning", len(prefix_ids), len(reasoning_ids)),
        TokenSpan("assistant_final_answer", len(reasoning_ids), len(completion_ids)),
        TokenSpan("eos", len(completion_ids), len(original_ids)),
        TokenSpan("padding", nonpadding, max_length),
    )
    spans = [
        TokenSpan(span.role, min(span.start, nonpadding), min(span.end, nonpadding))
        if span.role != "padding"
        else span
        for span in raw_spans
    ]
    role_by_index = ["unassigned"] * max_length
    for span in spans:
        for index in range(span.start, span.end):
            if role_by_index[index] != "unassigned":
                raise ValueError("token spans overlap")
            role_by_index[index] = span.role
    if any(role == "unassigned" for role in role_by_index):
        raise ValueError("token spans do not cover the padded sequence")

    system_content_start = _token_boundary(
        tokenizer, full_text, full_text.index(SFT_SYSTEM_PROMPT), original_ids
    )
    system_content_end = _token_boundary(
        tokenizer,
        full_text,
        full_text.index(SFT_SYSTEM_PROMPT) + len(SFT_SYSTEM_PROMPT),
        original_ids,
    )
    user_content = str(messages[1]["content"])
    user_character_start = full_text.index(user_content, len(system_text))
    user_content_start = _token_boundary(tokenizer, full_text, user_character_start, original_ids)
    user_content_end = _token_boundary(
        tokenizer, full_text, user_character_start + len(user_content), original_ids
    )

    eos_token_id = cast(int | None, tokenizer.eos_token_id)
    eos_ids = original_ids[len(completion_ids) :]
    has_final_eos = eos_token_id is not None and eos_token_id in eos_ids
    loss_ids = [label for label in labels if label != -100]
    intended_ids = original_ids[len(prefix_ids) : len(completion_ids)]
    if eos_token_id is not None:
        intended_ids = [*intended_ids, eos_token_id]
    metadata_values = [
        str(record[key])
        for key in INTERNAL_METADATA_KEYS
        if key in record and isinstance(record[key], str) and len(str(record[key])) >= 8
    ]
    decoded_loss = cast(str, tokenizer.decode(loss_ids, skip_special_tokens=False))
    metadata_hits = sorted(value for value in metadata_values if value in decoded_loss)

    span_counts: dict[str, dict[str, int | bool]] = {}
    for span in spans:
        values = labels[span.start : span.end]
        loss_bearing = sum(label != -100 for label in values)
        span_counts[span.role] = {
            "tokens": span.length,
            "masked": span.length - loss_bearing,
            "loss_bearing": loss_bearing,
            "contributes_to_loss": loss_bearing > 0,
        }

    content_system_loss = sum(
        labels[index] != -100
        for index in range(system_content_start, min(system_content_end, max_length))
    )
    content_user_loss = sum(
        labels[index] != -100
        for index in range(user_content_start, min(user_content_end, max_length))
    )
    assistant_header_loss = cast(int, span_counts["assistant_header"]["loss_bearing"])
    padding_masked = cast(int, span_counts["padding"]["masked"])
    padding_tokens = cast(int, span_counts["padding"]["tokens"])
    template_consistent = [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]

    return {
        "synthetic_id_sha256": hashlib.sha256(
            str(record["synthetic_id"]).encode("utf-8")
        ).hexdigest(),
        "full_text_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "record": {
            "formatted_text": full_text,
            "decoded_loss_bearing_target": decoded_loss,
            "token_rows": [
                {
                    "index": index,
                    "token_id": token,
                    "token": cast(
                        str,
                        tokenizer.decode(
                            [token], clean_up_tokenization_spaces=False, skip_special_tokens=False
                        ),
                    ),
                    "role": role_by_index[index],
                    "label": labels[index],
                    "loss_bearing": labels[index] != -100,
                }
                for index, token in enumerate(input_ids)
            ],
        },
        "span_counts": span_counts,
        "original_tokens": len(original_ids),
        "nonpadding_tokens": nonpadding,
        "truncated": len(original_ids) > max_length,
        "system_content_loss_bearing": content_system_loss,
        "user_content_loss_bearing": content_user_loss,
        "assistant_header_loss_bearing": assistant_header_loss,
        "padding_correctly_masked": padding_masked == padding_tokens,
        "assistant_content_present": bool(completion.strip())
        and len(completion_ids) > len(prefix_ids),
        "assistant_target_ends_with_eos": has_final_eos,
        "completion_final_answer_line_count": len(final_lines),
        "decoded_loss_equals_intended_completion": loss_ids == intended_ids,
        "decoded_labels_contain_question": question in decoded_loss,
        "decoded_labels_contain_internal_metadata": bool(metadata_hits),
        "internal_metadata_hit_count": len(metadata_hits),
        "chat_template_and_role_order_consistent": template_consistent,
    }


def _load_training_records(path: Path, expected_group: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("training row must be an object")
            row = cast(dict[str, Any], value)
            if (
                row.get("group") != expected_group
                or row.get("future_split") != "training"
                or row.get("final_decision") != "accepted"
            ):
                raise ValueError("training row violates the frozen split contract")
            records.append(row)
    if len(records) != 450:
        raise ValueError(f"{expected_group} requires exactly 450 training records")
    return records


def audit_arm(
    *,
    group: str,
    path: Path,
    tokenizer: Any,
    packet_path: Path,
    max_length: int = 512,
) -> dict[str, Any]:
    """Audit all 450 rows in one arm and write a 30-row ignored packet."""

    records = _load_training_records(path, group)
    rows = [
        reconstruct_original_record(record, tokenizer, max_length=max_length) for record in records
    ]
    selected = sorted(rows, key=lambda row: str(row["synthetic_id_sha256"]))[:30]
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    with packet_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row["record"], sort_keys=True, ensure_ascii=False) + "\n")

    spans: dict[str, Counter[str]] = {name: Counter() for name in ROLE_NAMES}
    boolean_fields = (
        "truncated",
        "padding_correctly_masked",
        "assistant_content_present",
        "assistant_target_ends_with_eos",
        "decoded_loss_equals_intended_completion",
        "decoded_labels_contain_question",
        "decoded_labels_contain_internal_metadata",
        "chat_template_and_role_order_consistent",
    )
    flags = {field: sum(bool(row[field]) for row in rows) for field in boolean_fields}
    for row in rows:
        for role, counts in cast(dict[str, dict[str, int]], row["span_counts"]).items():
            spans[role].update(
                {
                    "tokens": int(counts["tokens"]),
                    "masked": int(counts["masked"]),
                    "loss_bearing": int(counts["loss_bearing"]),
                }
            )
    packet_sha = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    aggregate: dict[str, Any] = {
        "group": group,
        "records": len(rows),
        "span_totals": {name: dict(spans[name]) for name in ROLE_NAMES},
        "system_content_loss_bearing_tokens": sum(
            int(row["system_content_loss_bearing"]) for row in rows
        ),
        "system_content_loss_bearing_records": sum(
            int(row["system_content_loss_bearing"]) > 0 for row in rows
        ),
        "user_content_loss_bearing_tokens": sum(
            int(row["user_content_loss_bearing"]) for row in rows
        ),
        "user_content_loss_bearing_records": sum(
            int(row["user_content_loss_bearing"]) > 0 for row in rows
        ),
        "assistant_header_loss_bearing_records": sum(
            int(row["assistant_header_loss_bearing"]) > 0 for row in rows
        ),
        "flag_counts": flags,
        "internal_metadata_hit_total": sum(int(row["internal_metadata_hit_count"]) for row in rows),
        "nonpadding_tokens": sum(int(row["nonpadding_tokens"]) for row in rows),
        "original_tokens": sum(int(row["original_tokens"]) for row in rows),
        "packet_records": len(selected),
        "packet_sha256": packet_sha,
    }
    aggregate["aggregate_sha256"] = canonical_sha256(aggregate)
    return aggregate


def run_audit(
    *,
    model_path: Path,
    targeted_path: Path,
    generic_path: Path,
    raw_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Load the pinned tokenizer and audit both frozen training arms."""

    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    arms = {
        "generic_control": audit_arm(
            group="generic_control",
            path=generic_path,
            tokenizer=tokenizer,
            packet_path=raw_directory / "generic_control_token_audit.jsonl",
        ),
        "targeted": audit_arm(
            group="targeted",
            path=targeted_path,
            tokenizer=tokenizer,
            packet_path=raw_directory / "targeted_token_audit.jsonl",
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-original-sft-role-label-audit-v1",
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "label_rule": "every nonpadding token copied to labels; padding is -100",
        "arms": arms,
        "mandatory_answers": {
            "system_message_tokens_loss_bearing": any(
                arm["system_content_loss_bearing_records"] > 0 for arm in arms.values()
            ),
            "user_question_tokens_loss_bearing": any(
                arm["user_content_loss_bearing_records"] > 0 for arm in arms.values()
            ),
            "assistant_header_tokens_loss_bearing": any(
                arm["assistant_header_loss_bearing_records"] > 0 for arm in arms.values()
            ),
            "padding_correctly_masked_all_records": all(
                arm["flag_counts"]["padding_correctly_masked"] == arm["records"]
                for arm in arms.values()
            ),
            "assistant_content_present_all_records": all(
                arm["flag_counts"]["assistant_content_present"] == arm["records"]
                for arm in arms.values()
            ),
            "assistant_target_ends_with_eos_all_records": all(
                arm["flag_counts"]["assistant_target_ends_with_eos"] == arm["records"]
                for arm in arms.values()
            ),
            "decoded_loss_equals_intended_completion_all_records": all(
                arm["flag_counts"]["decoded_loss_equals_intended_completion"] == arm["records"]
                for arm in arms.values()
            ),
            "decoded_label_span_contains_original_question_any_record": any(
                arm["flag_counts"]["decoded_labels_contain_question"] > 0 for arm in arms.values()
            ),
            "decoded_label_span_contains_internal_metadata_any_record": any(
                arm["flag_counts"]["decoded_labels_contain_internal_metadata"] > 0
                for arm in arms.values()
            ),
            "different_chat_template_or_role_order_any_record": any(
                arm["flag_counts"]["chat_template_and_role_order_consistent"] != arm["records"]
                for arm in arms.values()
            ),
        },
    }
    summary["label_mask_gate_passed_defect_found"] = bool(
        summary["mandatory_answers"]["system_message_tokens_loss_bearing"]
        or summary["mandatory_answers"]["user_question_tokens_loss_bearing"]
    )
    summary["summary_sha256"] = canonical_sha256(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run the role-aware audit CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--targeted-path", type=Path, required=True)
    parser.add_argument("--generic-path", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    summary = run_audit(
        model_path=args.model_path,
        targeted_path=args.targeted_path,
        generic_path=args.generic_path,
        raw_directory=args.raw_directory,
        output_path=args.output_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
