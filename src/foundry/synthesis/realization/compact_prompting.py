"""Frozen concise prompt for the compact tagged realization protocol."""

from __future__ import annotations

import hashlib

from foundry.synthesis.realization.compact_contracts import (
    COMPACT_SURFACE_SYSTEM_PROMPT,
    CompactRealizationRequest,
)

COMPACT_USER_PROTOCOL = """TAGGED_REALIZATION_V1
Rules: output only the listed tags in ORDER; one tag per line; copy every listed token exactly once
inside its assigned tag; end E tags with a period and Q with one question mark; add only grammar and
punctuation; use no pronouns, numbers, calculations, answers, explanations, Markdown, or extra tags.

Syntax example only (never copy these example tokens):
ORDER=E1,Q
E1=<ENTITY_EXAMPLE>,<HOLDS_E1>,<QUANTITY_EXAMPLE>,<UNIT_EXAMPLE>
Q=<ASK_COUNT_Q>,<TARGET_EXAMPLE>
OUTPUT:
<E1><ENTITY_EXAMPLE> <HOLDS_E1> <QUANTITY_EXAMPLE> <UNIT_EXAMPLE>.</E1>
<Q><ASK_COUNT_Q> <TARGET_EXAMPLE>?</Q>

SPEC:
{specification}
"""

COMPACT_USER_PROTOCOL_SHA256 = hashlib.sha256(COMPACT_USER_PROTOCOL.encode("utf-8")).hexdigest()
COMPACT_COMBINED_PROTOCOL_SHA256 = hashlib.sha256(
    (COMPACT_SURFACE_SYSTEM_PROMPT + "\n" + COMPACT_USER_PROTOCOL).encode("utf-8")
).hexdigest()


def serialize_compact_request(request: CompactRealizationRequest) -> str:
    """Serialize only tags and their immutable token assignments."""

    lines = [f"ORDER={','.join(request.tag_order)}"]
    for segment in request.segments:
        lines.append(f"{segment.tag}={','.join(segment.required_tokens)}")
    return COMPACT_USER_PROTOCOL.format(specification="\n".join(lines))


__all__ = [
    "COMPACT_COMBINED_PROTOCOL_SHA256",
    "COMPACT_USER_PROTOCOL_SHA256",
    "serialize_compact_request",
]
