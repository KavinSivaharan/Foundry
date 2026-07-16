"""Stable prompt rendering and hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, TypedDict

from foundry.config import PromptConfig


class ChatMessage(TypedDict):
    """One role/content message compatible with chat-template tokenizers."""

    role: Literal["system", "user"]
    content: str


def render_messages(prompt: PromptConfig, question: str) -> tuple[ChatMessage, ChatMessage]:
    """Render a benchmark question without changing surrounding prompt text."""

    if not question.strip():
        raise ValueError("question must be non-empty")
    return (
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user_template.format(question=question)},
    )


def prompt_sha256(prompt: PromptConfig) -> str:
    """Hash the exact role/content template used by every evaluation."""

    payload = json.dumps(
        {"system": prompt.system, "user_template": prompt.user_template},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
