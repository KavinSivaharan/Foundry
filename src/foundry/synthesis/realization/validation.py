"""Deterministic admission checks for model-produced realization templates."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from typing import cast

from foundry.synthesis.realization.ir import TargetKind
from foundry.synthesis.realization.model_contracts import (
    ClauseNodeMap,
    FilledRealization,
    PlaceholderKind,
    RealizationRequest,
    RealizationResponse,
)

_RESPONSE_KEYS = {
    "question_template",
    "placeholder_inventory",
    "clause_to_semantic_nodes",
    "requested_target_type",
    "requested_question_intent",
    "style_id",
}
_CLAUSE_MAP_KEYS = {"clause_index", "semantic_node_ids"}
_PLACEHOLDER_IN_TEXT = re.compile(r"<[A-Z][A-Z0-9_]*>")
_CLAUSE_PATTERN = re.compile(r"[^.?!]+[.?!]")
_NUMERIC_LITERAL = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:[.,/]\d+)?")
_BAD_PUNCTUATION = re.compile(r"(?:\.\.|;;|,,|\s+[?.!,;])")


class RealizationContractError(ValueError):
    """Raised when model output violates the frozen realization contract."""


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealizationContractError(f"{location} must be a nonempty string")
    return value


def _string_tuple(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RealizationContractError(f"{location} must be a nonempty array")
    return tuple(_string(item, f"{location}[]") for item in value)


def parse_realization_response(raw: str) -> RealizationResponse:
    """Parse one exact JSON object and reject missing or invented fields."""

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RealizationContractError("response is not valid JSON") from exc
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise RealizationContractError("response root must be a JSON object")
    values = cast(dict[str, object], loaded)
    if set(values) != _RESPONSE_KEYS:
        raise RealizationContractError("response fields do not exactly match the schema")

    raw_maps = values["clause_to_semantic_nodes"]
    if not isinstance(raw_maps, list) or not raw_maps:
        raise RealizationContractError("clause_to_semantic_nodes must be a nonempty array")
    maps: list[ClauseNodeMap] = []
    for index, item in enumerate(raw_maps):
        if not isinstance(item, dict) or not all(isinstance(key, str) for key in item):
            raise RealizationContractError(f"clause map {index} must be an object")
        mapping = cast(dict[str, object], item)
        if set(mapping) != _CLAUSE_MAP_KEYS:
            raise RealizationContractError(f"clause map {index} fields are invalid")
        clause_index = mapping["clause_index"]
        if isinstance(clause_index, bool) or not isinstance(clause_index, int):
            raise RealizationContractError(f"clause map {index} index must be an integer")
        try:
            maps.append(
                ClauseNodeMap(
                    clause_index=clause_index,
                    semantic_node_ids=_string_tuple(
                        mapping["semantic_node_ids"],
                        f"clause_to_semantic_nodes[{index}].semantic_node_ids",
                    ),
                )
            )
        except ValueError as exc:
            raise RealizationContractError(str(exc)) from exc

    target = _string(values["requested_target_type"], "requested_target_type")
    try:
        target_type = TargetKind(target)
    except ValueError as exc:
        raise RealizationContractError("requested_target_type is unknown") from exc
    return RealizationResponse(
        question_template=_string(values["question_template"], "question_template"),
        placeholder_inventory=_string_tuple(
            values["placeholder_inventory"], "placeholder_inventory"
        ),
        clause_to_semantic_nodes=tuple(maps),
        requested_target_type=target_type,
        requested_question_intent=_string(
            values["requested_question_intent"], "requested_question_intent"
        ),
        style_id=_string(values["style_id"], "style_id"),
    )


def split_template_clauses(template: str) -> tuple[str, ...]:
    """Split a fully punctuated template into stable clause units."""

    clauses = tuple(match.group(0).strip() for match in _CLAUSE_PATTERN.finditer(template))
    if " ".join(clauses) != " ".join(template.split()):
        return ()
    return clauses


def validate_realization_response(
    request: RealizationRequest, response: RealizationResponse
) -> tuple[str, ...]:
    """Return stable rejection reasons for a model-produced template."""

    reasons: list[str] = []
    expected_slots = {slot.token: slot for slot in request.placeholders}
    actual_inventory = response.placeholder_inventory
    if len(actual_inventory) != len(set(actual_inventory)):
        reasons.append("duplicated_placeholder_inventory")
    if set(actual_inventory) != set(expected_slots):
        reasons.append("placeholder_set_mismatch")

    observed_tokens = _PLACEHOLDER_IN_TEXT.findall(response.question_template)
    observed_counts = Counter(observed_tokens)
    if set(observed_tokens) != set(expected_slots):
        reasons.append("template_placeholder_set_mismatch")
    if any(
        observed_counts[token] != slot.required_occurrences
        for token, slot in expected_slots.items()
    ):
        reasons.append("placeholder_occurrence_mismatch")
    stripped = _PLACEHOLDER_IN_TEXT.sub("", response.question_template)
    if _NUMERIC_LITERAL.search(stripped):
        reasons.append("invented_numeric_literal")
    if "final answer" in response.question_template.lower():
        reasons.append("answer_content_forbidden")

    clauses = split_template_clauses(response.question_template)
    if not clauses or not response.question_template.rstrip().endswith("?"):
        reasons.append("malformed_question_template")
    if response.question_template.count("?") != 1:
        reasons.append("conflicting_question_intent")
    if _BAD_PUNCTUATION.search(response.question_template):
        reasons.append("malformed_punctuation")
    normalized_clauses = tuple(" ".join(clause.lower().split()) for clause in clauses)
    if len(normalized_clauses) != len(set(normalized_clauses)):
        reasons.append("duplicated_clause")

    event_by_id = {event.node_id: event for event in request.ordered_events}
    mapped_nodes: list[str] = []
    for mapping in response.clause_to_semantic_nodes:
        if mapping.clause_index >= len(clauses):
            reasons.append("invalid_clause_map_index")
            continue
        clause = clauses[mapping.clause_index]
        for node_id in mapping.semantic_node_ids:
            mapped_nodes.append(node_id)
            event = event_by_id.get(node_id)
            if event is None:
                reasons.append("invented_semantic_node")
            elif not set(event.required_placeholders) <= set(_PLACEHOLDER_IN_TEXT.findall(clause)):
                reasons.append("semantic_node_placeholder_mismatch")
    mapped_counts = Counter(mapped_nodes)
    for event in request.ordered_events:
        expected = 1
        if mapped_counts[event.node_id] < expected:
            reasons.append("missing_semantic_node")
        if mapped_counts[event.node_id] > expected and not event.repetition_authorized:
            reasons.append("duplicated_semantic_node")
    first_occurrence_order = tuple(dict.fromkeys(mapped_nodes))
    if first_occurrence_order not in request.allowed_discourse_orders:
        reasons.append("discourse_order_changed")
    if response.requested_target_type is not request.target_type:
        reasons.append("target_type_changed")
    if response.requested_question_intent != request.required_question_intent:
        reasons.append("question_intent_changed")
    if response.style_id != request.style.style_id:
        reasons.append("style_id_changed")
    rate_slots = {
        slot.token for slot in request.placeholders if slot.kind is PlaceholderKind.RATE_INTERVAL
    }
    if rate_slots and not rate_slots <= set(observed_tokens):
        reasons.append("rate_denominator_missing")
    return tuple(dict.fromkeys(reasons))


def fill_validated_template(
    request: RealizationRequest,
    response: RealizationResponse,
    replacements: dict[str, str],
) -> FilledRealization:
    """Fill opaque values only after the model template passes every check."""

    reasons = validate_realization_response(request, response)
    if reasons:
        raise RealizationContractError(f"template rejected: {', '.join(reasons)}")
    expected = {slot.token for slot in request.placeholders}
    if set(replacements) != expected:
        raise RealizationContractError("replacement keys must exactly match placeholder tokens")
    if any(
        not value.strip() or _PLACEHOLDER_IN_TEXT.search(value) for value in replacements.values()
    ):
        raise RealizationContractError("replacement surfaces must be nonempty and placeholder-free")
    question = response.question_template
    for token in sorted(replacements, key=len, reverse=True):
        question = question.replace(token, replacements[token])
    if _PLACEHOLDER_IN_TEXT.search(question):
        raise RealizationContractError("unresolved placeholder after deterministic filling")
    template_hash = hashlib.sha256(response.question_template.encode("utf-8")).hexdigest()
    replacement_payload = json.dumps(replacements, sort_keys=True, separators=(",", ":"))
    return FilledRealization(
        question=question,
        request_id=request.request_id,
        template_sha256=template_hash,
        replacement_sha256=hashlib.sha256(replacement_payload.encode("utf-8")).hexdigest(),
    )


def response_sha256(response: RealizationResponse) -> str:
    """Hash a validated response representation for replay evidence."""

    payload = json.dumps(asdict(response), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
