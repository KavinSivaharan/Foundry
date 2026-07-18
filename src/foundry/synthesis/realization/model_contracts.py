"""Typed contracts for constrained local-model surface realization.

The language model receives semantic roles and immutable placeholders, never the
numeric values or the canonical answer.  These contracts are deliberately
independent of any model runtime.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

from foundry.synthesis.realization.ir import TargetKind

SURFACE_REALIZATION_SYSTEM_PROMPT = (
    "You are a constrained surface-realization component.\n"
    "Return exactly one JSON object matching the supplied schema.\n"
    "Express only the supplied semantic events as natural English.\n"
    "Preserve every placeholder token byte-for-byte and use each exactly as specified.\n"
    "Do not add, remove, duplicate, calculate, or reveal numbers, units, entities, "
    "events, targets, answers, explanations, or fields."
)

SYSTEM_PROMPT_SHA256 = hashlib.sha256(SURFACE_REALIZATION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()

_PLACEHOLDER_PATTERN = re.compile(r"^<[A-Z][A-Z0-9_]*>$")


class PlaceholderKind(StrEnum):
    """Immutable semantic slot types visible to the realization model."""

    ENTITY = "entity"
    QUANTITY = "quantity"
    UNIT = "unit"
    RATE_INTERVAL = "rate_interval"
    LOCATION = "location"
    CONSTRAINT = "constraint"
    TARGET_ENTITY = "target_entity"


@dataclass(frozen=True)
class PlaceholderSpec:
    """One opaque slot whose surface value is withheld until compilation."""

    token: str
    kind: PlaceholderKind
    semantic_node_id: str
    required_occurrences: int = 1

    def __post_init__(self) -> None:
        if not _PLACEHOLDER_PATTERN.fullmatch(self.token):
            raise ValueError(f"invalid immutable placeholder: {self.token!r}")
        if not self.semantic_node_id.strip():
            raise ValueError("placeholder semantic_node_id cannot be empty")
        if self.required_occurrences < 1:
            raise ValueError("placeholder required_occurrences must be positive")


@dataclass(frozen=True)
class SemanticEventSpec:
    """One ordered semantic event that must survive realization."""

    node_id: str
    event_kind: str
    required_placeholders: tuple[str, ...]
    repetition_authorized: bool = False

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.event_kind.strip():
            raise ValueError("semantic events require nonempty IDs and kinds")
        if not self.required_placeholders:
            raise ValueError("semantic events require at least one placeholder")
        if len(self.required_placeholders) != len(set(self.required_placeholders)):
            raise ValueError("semantic event placeholders must be unique")


@dataclass(frozen=True)
class StyleControls:
    """Bounded discourse choices; none may alter mathematical semantics."""

    style_id: str
    difficulty: str
    permitted_voices: tuple[str, ...]
    allow_safe_context: bool

    def __post_init__(self) -> None:
        if not self.style_id.strip() or not self.difficulty.strip():
            raise ValueError("style controls require nonempty identifiers")
        if not self.permitted_voices:
            raise ValueError("at least one grammatical voice must be permitted")


@dataclass(frozen=True)
class RealizationRequest:
    """Complete, value-blind request sent to a local realization model."""

    request_id: str
    category: str
    semantic_frame: str
    ordered_events: tuple[SemanticEventSpec, ...]
    placeholders: tuple[PlaceholderSpec, ...]
    target_type: TargetKind
    required_question_intent: str
    allowed_discourse_orders: tuple[tuple[str, ...], ...]
    forbidden_transformations: tuple[str, ...]
    style: StyleControls

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.request_id,
                self.category,
                self.semantic_frame,
                self.required_question_intent,
            )
        ):
            raise ValueError("realization request identifiers cannot be empty")
        node_ids = tuple(event.node_id for event in self.ordered_events)
        if not node_ids or len(node_ids) != len(set(node_ids)):
            raise ValueError("ordered semantic node IDs must be nonempty and unique")
        placeholder_tokens = tuple(slot.token for slot in self.placeholders)
        if not placeholder_tokens or len(placeholder_tokens) != len(set(placeholder_tokens)):
            raise ValueError("request placeholder tokens must be nonempty and unique")
        known_tokens = set(placeholder_tokens)
        for event in self.ordered_events:
            if not set(event.required_placeholders) <= known_tokens:
                raise ValueError("semantic event refers to an unknown placeholder")
        if not self.allowed_discourse_orders:
            raise ValueError("at least one discourse order must be permitted")
        expected_nodes = set(node_ids)
        if any(
            len(order) != len(node_ids) or set(order) != expected_nodes
            for order in self.allowed_discourse_orders
        ):
            raise ValueError("each discourse order must cover every event exactly once")
        if not self.forbidden_transformations:
            raise ValueError("forbidden transformations must be explicit")


@dataclass(frozen=True)
class ClauseNodeMap:
    """Model-declared mapping from a rendered clause to semantic nodes."""

    clause_index: int
    semantic_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.clause_index < 0 or not self.semantic_node_ids:
            raise ValueError("clause maps require a valid index and semantic nodes")


@dataclass(frozen=True)
class RealizationResponse:
    """Strict model output; it intentionally contains no answer field."""

    question_template: str
    placeholder_inventory: tuple[str, ...]
    clause_to_semantic_nodes: tuple[ClauseNodeMap, ...]
    requested_target_type: TargetKind
    requested_question_intent: str
    style_id: str


@dataclass(frozen=True)
class FilledRealization:
    """Compiler-filled question created only after deterministic validation."""

    question: str
    request_id: str
    template_sha256: str
    replacement_sha256: str


def placeholder_pattern() -> re.Pattern[str]:
    """Expose the exact placeholder grammar without exposing mutable state."""

    return _PLACEHOLDER_PATTERN
