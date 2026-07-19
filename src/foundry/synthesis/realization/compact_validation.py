"""Deterministic parser, admission checks, and fill for compact tagged output."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from foundry.synthesis.realization.compact_contracts import (
    CompactRealizationRequest,
    CompactRealizationResponse,
    CompactTaggedSegment,
    FilledCompactRealization,
)

_BLOCK = re.compile(r"<(?P<tag>E[1-9]\d*|Q)>(?P<body>.*?)</(?P=tag)>", re.DOTALL)
_TOKEN = re.compile(r"<[A-Z][A-Z0-9_]*>")
_NUMERIC = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:[.,/]\d+)?")
_FORBIDDEN_MARKUP = re.compile(r"```|<think>|</think>", re.IGNORECASE)
_CALCULATION = re.compile(
    r"(?:\b(?:therefore|thus|equals|answer\s+is|result\s+is|final\s+answer)\b|=)",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|never|except|excluding|instead|without|less|fewer|opposite|reverse)\b",
    re.IGNORECASE,
)
_BAD_PUNCTUATION = re.compile(r"(?:\.\.|;;|,,|\s+[?.!,;])")
_PRONOUN = re.compile(
    r"\b(?:he|she|it|they|them|their|theirs|his|hers|this|that|these|those)\b",
    re.IGNORECASE,
)
_ALLOWED_CONNECTING_WORDS = {
    "a",
    "an",
    "the",
    "at",
    "in",
    "on",
    "from",
    "to",
    "per",
    "for",
    "during",
    "after",
    "before",
    "and",
    "or",
    "together",
    "each",
    "every",
    "all",
    "with",
    "of",
    "into",
    "out",
    "how",
    "many",
    "what",
    "which",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "can",
    "be",
    "formed",
    "remain",
    "remaining",
    "total",
    "average",
    "mean",
    "number",
    "valid",
    "possible",
    "maximum",
    "available",
    "there",
    "then",
    "later",
    "initially",
    "finally",
    "across",
    "among",
    "through",
    "over",
    "by",
    "now",
    "altogether",
    "still",
    "complete",
    "groups",
    "group",
    "items",
    "item",
    "rate",
}


class CompactContractError(ValueError):
    """Raised when Qwen output violates the compact tagged protocol."""


def parse_compact_response(raw: str) -> CompactRealizationResponse:
    """Parse a complete tag-only response and reject text outside its blocks."""

    if not raw.strip():
        raise CompactContractError("empty_response")
    if _FORBIDDEN_MARKUP.search(raw):
        raise CompactContractError("forbidden_markup")
    segments: list[CompactTaggedSegment] = []
    cursor = 0
    for match in _BLOCK.finditer(raw):
        if raw[cursor : match.start()].strip():
            raise CompactContractError("text_outside_tags")
        body = match.group("body").strip()
        if not body:
            raise CompactContractError("empty_tag_body")
        segments.append(CompactTaggedSegment(match.group("tag"), body))
        cursor = match.end()
    if raw[cursor:].strip():
        raise CompactContractError("text_after_q_or_unparsed_tag")
    if not segments:
        raise CompactContractError("no_tagged_segments")
    return CompactRealizationResponse(tuple(segments))


def validate_compact_response(
    request: CompactRealizationRequest, response: CompactRealizationResponse
) -> tuple[str, ...]:
    """Return stable rejection reasons for every deterministic compact layer."""

    reasons: list[str] = []
    observed_tags = tuple(segment.tag for segment in response.segments)
    expected_tags = request.tag_order
    tag_counts = Counter(observed_tags)
    if set(observed_tags) != set(expected_tags):
        reasons.append("event_tag_set_mismatch")
    if any(tag_counts[tag] != 1 for tag in expected_tags):
        reasons.append("event_tag_occurrence_mismatch")
    if tag_counts["Q"] != 1:
        reasons.append("question_tag_occurrence_mismatch")
    if observed_tags != expected_tags:
        reasons.append("event_order_changed")

    expected_by_tag = {segment.tag: segment for segment in request.segments}
    expected_all = {token for segment in request.segments for token in segment.required_tokens}
    observed_all: list[str] = []
    for segment in response.segments:
        expected = expected_by_tag.get(segment.tag)
        observed = _TOKEN.findall(segment.body)
        observed_all.extend(observed)
        if expected is None:
            continue
        counts = Counter(observed)
        expected_tokens = set(expected.required_tokens)
        if set(observed) != expected_tokens:
            reasons.append("placeholder_assignment_mismatch")
        if any(counts[token] != 1 for token in expected.required_tokens):
            reasons.append("placeholder_occurrence_mismatch")
        anchor_tokens = {anchor.token for anchor in expected.anchors}
        if not anchor_tokens <= set(observed):
            reasons.append("semantic_anchor_missing")
        stripped = _TOKEN.sub("", segment.body)
        if _NUMERIC.search(stripped):
            reasons.append("raw_numeric_literal")
        if _CALCULATION.search(stripped):
            reasons.append("answer_or_calculation_content")
        if _NEGATION.search(stripped):
            reasons.append("semantic_anchor_reversal")
        if _PRONOUN.search(stripped):
            reasons.append("unlicensed_pronoun")
        words = {word.lower() for word in re.findall(r"[A-Za-z]+", stripped)}
        if words - _ALLOWED_CONNECTING_WORDS:
            reasons.append("unlicensed_semantic_content")
        if _BAD_PUNCTUATION.search(segment.body):
            reasons.append("malformed_punctuation")
        if segment.tag == "Q":
            if not segment.body.endswith("?") or segment.body.count("?") != 1:
                reasons.append("malformed_question_tag")
        elif not segment.body.endswith(".") or "?" in segment.body:
            reasons.append("malformed_event_punctuation")

    observed_counts = Counter(observed_all)
    if set(observed_all) != expected_all:
        reasons.append("placeholder_set_mismatch")
    if any(observed_counts[token] != 1 for token in expected_all):
        reasons.append("altered_or_duplicated_placeholder")
    question = next((segment for segment in response.segments if segment.tag == "Q"), None)
    if question is not None:
        question_tokens = set(_TOKEN.findall(question.body))
        if not set(request.question.required_tokens) <= question_tokens:
            reasons.append("target_placeholder_missing")
    return tuple(dict.fromkeys(reasons))


def fill_compact_response(
    request: CompactRealizationRequest,
    response: CompactRealizationResponse,
    replacements: dict[str, str],
) -> FilledCompactRealization:
    """Fill anchors, entities, quantities, and units only after strict validation."""

    reasons = validate_compact_response(request, response)
    if reasons:
        raise CompactContractError(f"compact response rejected: {', '.join(reasons)}")
    expected = {token for segment in request.segments for token in segment.required_tokens}
    if set(replacements) != expected:
        raise CompactContractError("replacement_set_mismatch")
    if any(not value.strip() or _TOKEN.search(value) for value in replacements.values()):
        raise CompactContractError("invalid_replacement_surface")
    bodies: list[str] = []
    for segment in response.segments:
        body = segment.body
        for token in sorted(replacements, key=len, reverse=True):
            body = body.replace(token, replacements[token])
        if _TOKEN.search(body):
            raise CompactContractError("unresolved_placeholder")
        bodies.append(body)
    question = " ".join(bodies)
    template_payload = "\n".join(
        f"<{segment.tag}>{segment.body}</{segment.tag}>" for segment in response.segments
    )
    replacement_payload = json.dumps(replacements, sort_keys=True, separators=(",", ":"))
    return FilledCompactRealization(
        question=question,
        template_sha256=hashlib.sha256(template_payload.encode("utf-8")).hexdigest(),
        replacement_sha256=hashlib.sha256(replacement_payload.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "CompactContractError",
    "fill_compact_response",
    "parse_compact_response",
    "validate_compact_response",
]
