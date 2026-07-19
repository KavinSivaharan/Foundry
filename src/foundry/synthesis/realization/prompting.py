"""Frozen prompt serialization for constrained Qwen3 realization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from foundry.synthesis.realization.model_contracts import (
    SURFACE_REALIZATION_SYSTEM_PROMPT,
    RealizationRequest,
)

USER_PROMPT_PROTOCOL = """REALIZATION_PROTOCOL: foundry-slot-preserving-json-v1

Create natural English for the supplied value-blind semantic specification.
Each semantic event must be expressed exactly once. The final sentence must be the one question.
Use only the permitted discourse orders. Do not use pronouns.
Placeholder tokens are opaque words: copy each required token byte-for-byte exactly once.

Return exactly one JSON object with exactly these fields:
- question_template: string ending in one question mark
- placeholder_inventory: array containing every supplied placeholder token exactly once
- clause_to_semantic_nodes: array of objects with clause_index and semantic_node_ids
- requested_target_type: exact supplied target type
- requested_question_intent: exact supplied question intent
- style_id: exact supplied style identifier

clause_index is the zero-based sentence index in question_template. Map every semantic node exactly
once to a sentence that contains all placeholders required by that node. Do not emit Markdown,
explanations, calculations, equations, final answers, or prose outside the JSON object.

VALUE_BLIND_SPECIFICATION:
{request_payload}
"""

USER_PROMPT_PROTOCOL_SHA256 = hashlib.sha256(USER_PROMPT_PROTOCOL.encode("utf-8")).hexdigest()
COMBINED_PROMPT_PROTOCOL_SHA256 = hashlib.sha256(
    (SURFACE_REALIZATION_SYSTEM_PROMPT + "\n" + USER_PROMPT_PROTOCOL).encode("utf-8")
).hexdigest()


def serialize_value_blind_request(request: RealizationRequest) -> str:
    """Serialize only approved value-blind fields; omit the internal request ID."""

    payload = asdict(request)
    payload.pop("request_id")
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return USER_PROMPT_PROTOCOL.format(request_payload=rendered)


__all__ = [
    "COMBINED_PROMPT_PROTOCOL_SHA256",
    "USER_PROMPT_PROTOCOL_SHA256",
    "serialize_value_blind_request",
]
