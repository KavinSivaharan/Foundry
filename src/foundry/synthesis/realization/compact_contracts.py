"""Typed contracts for the compact tagged Qwen realization protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from foundry.synthesis.realization.model_contracts import PlaceholderSpec

COMPACT_SURFACE_SYSTEM_PROMPT = (
    "You are a constrained surface realizer. Return only the requested tagged clauses.\n"
    "Copy every angle-bracket token exactly once in its assigned tag.\n"
    "Use semantic-anchor tokens as immutable predicates; do not restate or reverse them.\n"
    "Add only natural grammar, connecting words, and punctuation.\n"
    "Do not calculate, answer, explain, add facts, emit numbers, or use Markdown."
)
COMPACT_SYSTEM_PROMPT_SHA256 = hashlib.sha256(
    COMPACT_SURFACE_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class CompactAnchorSpec:
    """One immutable semantic predicate filled only after model validation."""

    token: str
    replacement: str

    def __post_init__(self) -> None:
        if not self.token.startswith("<") or not self.token.endswith(">"):
            raise ValueError("compact anchors must be angle-bracket tokens")
        if not self.replacement.strip() or "<" in self.replacement or ">" in self.replacement:
            raise ValueError("compact anchor replacements must be nonempty plain text")


@dataclass(frozen=True)
class CompactSegmentSpec:
    """One event or question tag with its exact immutable token assignment."""

    tag: str
    node_id: str
    placeholders: tuple[str, ...]
    anchors: tuple[CompactAnchorSpec, ...]

    def __post_init__(self) -> None:
        if self.tag != "Q" and not (self.tag.startswith("E") and self.tag[1:].isdigit()):
            raise ValueError("compact segment tags must be E<number> or Q")
        if not self.node_id.strip() or not self.placeholders or not self.anchors:
            raise ValueError("compact segments require a node, placeholders, and anchors")
        tokens = (*self.placeholders, *(anchor.token for anchor in self.anchors))
        if len(tokens) != len(set(tokens)):
            raise ValueError("compact segment tokens must be unique")

    @property
    def required_tokens(self) -> tuple[str, ...]:
        """Return the exact set of tokens assigned to this tag."""

        return (*self.placeholders, *(anchor.token for anchor in self.anchors))


@dataclass(frozen=True)
class CompactRealizationRequest:
    """Minimal value-blind contract sent to Qwen3."""

    request_id: str
    events: tuple[CompactSegmentSpec, ...]
    question: CompactSegmentSpec
    placeholders: tuple[PlaceholderSpec, ...]

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.events:
            raise ValueError("compact requests require an ID and at least one fact event")
        expected_tags = tuple(f"E{index}" for index in range(1, len(self.events) + 1))
        if tuple(event.tag for event in self.events) != expected_tags:
            raise ValueError("compact fact tags must be consecutive and ordered")
        if self.question.tag != "Q":
            raise ValueError("compact requests require one terminal Q segment")
        assigned = [token for segment in self.segments for token in segment.required_tokens]
        if len(assigned) != len(set(assigned)):
            raise ValueError("compact tokens may be assigned to only one segment")
        placeholder_tokens = {placeholder.token for placeholder in self.placeholders}
        assigned_placeholders = {
            token for segment in self.segments for token in segment.placeholders
        }
        if placeholder_tokens != assigned_placeholders:
            raise ValueError("compact request must assign every base placeholder exactly once")

    @property
    def segments(self) -> tuple[CompactSegmentSpec, ...]:
        return (*self.events, self.question)

    @property
    def tag_order(self) -> tuple[str, ...]:
        return tuple(segment.tag for segment in self.segments)

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompactTaggedSegment:
    tag: str
    body: str


@dataclass(frozen=True)
class CompactRealizationResponse:
    segments: tuple[CompactTaggedSegment, ...]

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FilledCompactRealization:
    question: str
    template_sha256: str
    replacement_sha256: str


__all__ = [
    "COMPACT_SURFACE_SYSTEM_PROMPT",
    "COMPACT_SYSTEM_PROMPT_SHA256",
    "CompactAnchorSpec",
    "CompactRealizationRequest",
    "CompactRealizationResponse",
    "CompactSegmentSpec",
    "CompactTaggedSegment",
    "FilledCompactRealization",
]
