"""Typed, fail-closed contracts for the offline template bank."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from foundry.synthesis.realization.ir import TargetKind
from foundry.synthesis.schema import DifficultyLevel

_ID = re.compile(r"^[a-z0-9][a-z0-9._-]+$")
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_FORBIDDEN = ("answer", "gsm", "benchmark")


@dataclass(frozen=True)
class SentencePlanSpec:
    """One independently authored grammatical plan within a semantic frame."""

    plan_id: str
    clause_order: tuple[str, ...]
    opening_form: str
    event_form: str
    question_form: str
    temporal_framing: str
    grammatical_construction: str

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.plan_id):
            raise ValueError("sentence-plan ID is invalid")
        if len(self.clause_order) < 2 or len(set(self.clause_order)) != len(self.clause_order):
            raise ValueError("sentence plans require a unique multi-clause ordering")
        values = (
            self.opening_form,
            self.event_form,
            self.question_form,
            self.temporal_framing,
            self.grammatical_construction,
        )
        if any(not value.strip() for value in values):
            raise ValueError("sentence-plan fields must be nonempty")
        if any(word in " ".join(values).lower() for word in _FORBIDDEN):
            raise ValueError("sentence plans cannot contain benchmark or answer language")

    @property
    def normalized_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TemplateSpec:
    """A versioned semantic frame with explicit compatibility and provenance."""

    template_id: str
    template_version: str
    reasoning_category: str
    semantic_frame: str
    compatible_target_types: tuple[TargetKind, ...]
    required_semantic_event_types: tuple[str, ...]
    required_placeholder_roles: tuple[str, ...]
    allowed_units: tuple[str, ...]
    allowed_object_families: tuple[str, ...]
    supported_difficulty_levels: tuple[DifficultyLevel, ...]
    clause_order_plan: tuple[str, ...]
    question_form: str
    sentence_plan_variants: tuple[SentencePlanSpec, ...]
    optional_context_policy: str
    output_contract_compatible: bool
    provenance: str
    review_status: str

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.template_id) or not _ID.fullmatch(self.template_version):
            raise ValueError("template ID or version is invalid")
        if len(self.sentence_plan_variants) < 4:
            raise ValueError("each semantic frame requires at least four sentence plans")
        if len({plan.plan_id for plan in self.sentence_plan_variants}) != len(
            self.sentence_plan_variants
        ):
            raise ValueError("sentence-plan IDs must be unique within a frame")
        if not self.compatible_target_types or not self.required_semantic_event_types:
            raise ValueError("template compatibility cannot be empty")
        if not self.required_placeholder_roles or not self.allowed_units:
            raise ValueError("typed placeholders and units are required")
        if not self.allowed_object_families or not self.supported_difficulty_levels:
            raise ValueError("object and difficulty compatibility is required")
        if self.review_status != "human_review_pending":
            raise ValueError("new template-bank plans must remain human-review pending")
        if self.provenance != "original_hand_authored_foundry_v1":
            raise ValueError("template provenance differs from the approved source")
        if not self.output_contract_compatible:
            raise ValueError("every initial-bank template must support the shared output track")
        if any("{" in value or "}" in value for value in self.required_placeholder_roles):
            raise ValueError("placeholder roles are names, not free-form template slots")
        for plan in self.sentence_plan_variants:
            used = set(
                _PLACEHOLDER.findall(
                    " ".join((plan.opening_form, plan.event_form, plan.question_form))
                )
            )
            if not used <= set(self.required_placeholder_roles):
                raise ValueError("sentence plan contains an untyped placeholder")

    @property
    def normalized_template_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def render_signature_hash(self, plan: SentencePlanSpec) -> str:
        if plan not in self.sentence_plan_variants:
            raise ValueError("plan does not belong to template")
        payload = f"{self.template_id}:{self.template_version}:{plan.normalized_sha256}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
