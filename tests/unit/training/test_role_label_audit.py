from __future__ import annotations

from typing import Any

from foundry.training.role_label_audit import reconstruct_original_record


class CharacterTokenizer:
    eos_token_id = 900

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize
        text = "".join(f"<{item['role']}>{item['content']}</{item['role']}>" for item in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        ids = [
            900 if text[index : index + 12] == "</assistant>" else ord(char)
            for index, char in enumerate(text)
        ]
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and isinstance(max_length, int):
            ids = ids[:max_length]
        attention = [1] * len(ids)
        if kwargs.get("padding") == "max_length" and isinstance(max_length, int):
            padding = max_length - len(ids)
            ids.extend([0] * padding)
            attention.extend([0] * padding)
        return {"input_ids": ids, "attention_mask": attention}

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join("<eos>" if token == 900 else chr(token) for token in ids if token)


def _record() -> dict[str, object]:
    return {
        "synthetic_id": "original-fixture",
        "rendered_question": "How many original objects remain?",
        "training_completion": "Subtract the removed objects.\nFinal answer: 4",
        "latent_program_sha256": "abcdef0123456789",
    }


def test_original_mask_makes_system_user_and_header_loss_bearing() -> None:
    result = reconstruct_original_record(_record(), CharacterTokenizer(), max_length=512)
    assert result["system_content_loss_bearing"] > 0
    assert result["user_content_loss_bearing"] > 0
    assert result["assistant_header_loss_bearing"] > 0
    assert result["decoded_labels_contain_question"] is True
    assert result["decoded_loss_equals_intended_completion"] is False


def test_original_mask_still_masks_padding() -> None:
    result = reconstruct_original_record(_record(), CharacterTokenizer(), max_length=512)
    assert result["padding_correctly_masked"] is True
    padding = result["span_counts"]["padding"]
    assert padding["loss_bearing"] == 0
    assert padding["masked"] == padding["tokens"]


def test_role_partition_covers_entire_sequence() -> None:
    result = reconstruct_original_record(_record(), CharacterTokenizer(), max_length=512)
    assert sum(span["tokens"] for span in result["span_counts"].values()) == 512
    assert result["chat_template_and_role_order_consistent"] is True
