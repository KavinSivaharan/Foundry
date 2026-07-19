"""Compile trusted procedural IR requests into compact tagged contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.realization.compact_contracts import (
    CompactAnchorSpec,
    CompactRealizationRequest,
    CompactSegmentSpec,
)
from foundry.synthesis.realization.ir import TargetKind
from foundry.synthesis.realization.request_builder import (
    PreparedRealizationRequest,
    prepare_realization_request,
)


@dataclass(frozen=True)
class PreparedCompactRequest:
    """Compact model contract plus deterministic post-validation replacements."""

    base: PreparedRealizationRequest
    request: CompactRealizationRequest
    replacements: dict[str, str]
    realization_signature: str

    @property
    def draft(self) -> CandidateDraft:
        return self.base.draft

    @property
    def semantic_frame(self) -> str:
        return self.base.semantic_frame

    @property
    def request_sha256(self) -> str:
        return self.request.sha256


def _anchor_key(description: str) -> tuple[tuple[str, str], ...]:
    lowered = description.lower()
    if "begins with" in lowered:
        return (("BEGINS_WITH", "begins with"),)
    if "transfer into" in lowered:
        return (("MOVED_INTO_FROM", "are moved from"),)
    if "transfer out" in lowered:
        return (("MOVED_OUT_FROM", "are moved from"),)
    if " handles " in lowered:
        return (("HANDLES", "handles"),)
    if "continues for" in lowered:
        return (("CONTINUES_FOR", "continues for"),)
    if "ratio of" in lowered:
        return (("RATIO_OF", "have a ratio of"),)
    if "represented by" in lowered:
        return (("REPRESENTED_BY", "is represented by"),)
    if "selected share" in lowered:
        return (("SELECTED_SHARE", "represents"),)
    if "averaging" in lowered:
        return (("CONTAINS", "contains"), ("AVERAGES_PER", "and averages"))
    if " provides " in lowered:
        return (("PROVIDES", "provides"),)
    if "consist of" in lowered:
        return (("CONSIST_OF", "consist of"),)
    if "all items together use" in lowered:
        return (("TOTAL_RESOURCE_USE", "use"),)
    if "distributed equally" in lowered:
        return (("DISTRIBUTED_EQUALLY", "are distributed equally among"),)
    if "requires exactly" in lowered:
        return (("REQUIRES_EXACTLY", "requires exactly"),)
    if " requires " in lowered:
        return (("REQUIRES", "requires"),)
    if " uses " in lowered:
        return (("USES", "uses"),)
    if " contains " in lowered:
        return (("CONTAINS", "contains"),)
    if "operate for" in lowered:
        return (("OPERATE_FOR", "operate for"),)
    raise ValueError(f"unsupported compact semantic relation: {description}")


def _question_anchor(kind: TargetKind) -> tuple[str, str]:
    return {
        TargetKind.COUNT: ("ASK_COUNT", "How many"),
        TargetKind.TOTAL_QUANTITY: ("ASK_TOTAL", "What is the total number of"),
        TargetKind.REMAINING_QUANTITY: ("ASK_REMAINING", "How many"),
        TargetKind.RATE: ("ASK_RATE", "What is the rate for"),
        TargetKind.PERCENTAGE: ("ASK_PERCENT_SHARE", "How many"),
        TargetKind.RATIO: ("ASK_RATIO_RESULT", "How many"),
        TargetKind.WEIGHTED_MEAN: ("ASK_WEIGHTED_MEAN", "What is the weighted mean for"),
        TargetKind.VALID_ASSIGNMENT_COUNT: ("ASK_VALID_ASSIGNMENTS", "How many"),
        TargetKind.CAPACITY: ("ASK_CAPACITY", "What is the maximum number of"),
        TargetKind.GROUP_COUNT: ("ASK_COMPLETE_GROUPS", "How many"),
    }[kind]


def prepare_compact_request(draft: CandidateDraft, *, style_variant: int) -> PreparedCompactRequest:
    """Reuse the trusted value-blind request and remove redundant model echoes."""

    base = prepare_realization_request(draft, style_variant=style_variant)
    events = base.request.ordered_events
    fact_events = events[:-1]
    target_event = events[-1]
    compact_events: list[CompactSegmentSpec] = []
    replacements = dict(base.replacements)
    for index, event in enumerate(fact_events, start=1):
        tag = f"E{index}"
        anchors: list[CompactAnchorSpec] = []
        for anchor_index, (key, replacement) in enumerate(_anchor_key(event.event_kind), start=1):
            suffix = "" if anchor_index == 1 else f"_{anchor_index}"
            token = f"<{key}_{tag}{suffix}>"
            anchors.append(CompactAnchorSpec(token, replacement))
            replacements[token] = replacement
        compact_events.append(
            CompactSegmentSpec(tag, event.node_id, event.required_placeholders, tuple(anchors))
        )
    question_key, question_replacement = _question_anchor(base.request.target_type)
    question_token = f"<{question_key}_Q>"
    replacements[question_token] = question_replacement
    question = CompactSegmentSpec(
        "Q",
        target_event.node_id,
        target_event.required_placeholders,
        (CompactAnchorSpec(question_token, question_replacement),),
    )
    request = CompactRealizationRequest(
        request_id=base.request.request_id,
        events=tuple(compact_events),
        question=question,
        placeholders=base.request.placeholders,
    )
    signature_material = f"compact-v1:{base.semantic_frame}:" + ",".join(
        anchor.token for segment in request.segments for anchor in segment.anchors
    )
    return PreparedCompactRequest(
        base=base,
        request=request,
        replacements=replacements,
        realization_signature=hashlib.sha256(signature_material.encode("utf-8")).hexdigest(),
    )


__all__ = ["PreparedCompactRequest", "prepare_compact_request"]
