"""Design-only tests for the constrained local-model realization boundary."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from foundry.synthesis.realization.ir import TargetKind
from foundry.synthesis.realization.model_contracts import (
    SYSTEM_PROMPT_SHA256,
    ClauseNodeMap,
    PlaceholderKind,
    PlaceholderSpec,
    RealizationRequest,
    RealizationResponse,
    SemanticEventSpec,
    StyleControls,
)
from foundry.synthesis.realization.validation import (
    RealizationContractError,
    fill_validated_template,
    parse_realization_response,
    validate_realization_response,
)


def _request(*, include_rate: bool = False) -> RealizationRequest:
    slots = [
        PlaceholderSpec("<ENTITY_1>", PlaceholderKind.ENTITY, "state"),
        PlaceholderSpec("<QUANTITY_1>", PlaceholderKind.QUANTITY, "state"),
        PlaceholderSpec("<UNIT_1>", PlaceholderKind.UNIT, "state"),
        PlaceholderSpec("<TARGET_ENTITY>", PlaceholderKind.TARGET_ENTITY, "target"),
    ]
    state_slots = ["<ENTITY_1>", "<QUANTITY_1>", "<UNIT_1>"]
    if include_rate:
        slots.append(PlaceholderSpec("<RATE_INTERVAL_1>", PlaceholderKind.RATE_INTERVAL, "state"))
        state_slots.append("<RATE_INTERVAL_1>")
    return RealizationRequest(
        request_id="fixture-request",
        category="bookkeeping",
        semantic_frame="initial-state",
        ordered_events=(
            SemanticEventSpec("state", "initial_state", tuple(state_slots)),
            SemanticEventSpec("target", "question_target", ("<TARGET_ENTITY>",)),
        ),
        placeholders=tuple(slots),
        target_type=TargetKind.COUNT,
        required_question_intent="ask for the target count",
        allowed_discourse_orders=(("state", "target"),),
        forbidden_transformations=("invent values", "change the target"),
        style=StyleControls("plain-active", "medium", ("active",), False),
    )


def _response(*, include_rate: bool = False) -> RealizationResponse:
    rate = " per <RATE_INTERVAL_1>" if include_rate else ""
    inventory = ["<ENTITY_1>", "<QUANTITY_1>", "<UNIT_1>", "<TARGET_ENTITY>"]
    if include_rate:
        inventory.append("<RATE_INTERVAL_1>")
    return RealizationResponse(
        question_template=(
            f"<ENTITY_1> records <QUANTITY_1> <UNIT_1>{rate}. "
            "How many <TARGET_ENTITY> are requested?"
        ),
        placeholder_inventory=tuple(inventory),
        clause_to_semantic_nodes=(
            ClauseNodeMap(0, ("state",)),
            ClauseNodeMap(1, ("target",)),
        ),
        requested_target_type=TargetKind.COUNT,
        requested_question_intent="ask for the target count",
        style_id="plain-active",
    )


def test_system_prompt_hash_is_frozen() -> None:
    assert SYSTEM_PROMPT_SHA256 == (
        "9d3e808d5c887d974919728d5afb51df9daa5760d467d3c08648799aeddcc393"
    )


def test_strict_json_response_parses_without_answer_field() -> None:
    response = _response()
    raw = json.dumps(
        {
            "question_template": response.question_template,
            "placeholder_inventory": list(response.placeholder_inventory),
            "clause_to_semantic_nodes": [
                {
                    "clause_index": entry.clause_index,
                    "semantic_node_ids": list(entry.semantic_node_ids),
                }
                for entry in response.clause_to_semantic_nodes
            ],
            "requested_target_type": response.requested_target_type.value,
            "requested_question_intent": response.requested_question_intent,
            "style_id": response.style_id,
        }
    )
    assert parse_realization_response(raw) == response
    with pytest.raises(RealizationContractError, match="exactly match"):
        parse_realization_response(raw[:-1] + ', "answer": 12}')


def test_valid_template_fills_values_only_after_validation() -> None:
    request = _request()
    response = _response()
    assert validate_realization_response(request, response) == ()
    filled = fill_validated_template(
        request,
        response,
        {
            "<ENTITY_1>": "Avery",
            "<QUANTITY_1>": "18",
            "<UNIT_1>": "tickets",
            "<TARGET_ENTITY>": "tickets",
        },
    )
    assert filled.question == "Avery records 18 tickets. How many tickets are requested?"
    assert len(filled.template_sha256) == len(filled.replacement_sha256) == 64


@pytest.mark.parametrize(
    ("response", "reason"),
    (
        (
            replace(_response(), placeholder_inventory=("<ENTITY_1>",)),
            "placeholder_set_mismatch",
        ),
        (
            replace(
                _response(),
                question_template=(
                    "<ENTITY_1> records <QUANTITY_1> <UNIT_1> and <UNIT_1>. "
                    "How many <TARGET_ENTITY> are requested?"
                ),
            ),
            "placeholder_occurrence_mismatch",
        ),
        (
            replace(
                _response(),
                question_template=(
                    "<ENTITY_1> records 12 <UNIT_1>. How many <TARGET_ENTITY> are requested?"
                ),
            ),
            "invented_numeric_literal",
        ),
        (
            replace(
                _response(),
                clause_to_semantic_nodes=(
                    ClauseNodeMap(0, ("state",)),
                    ClauseNodeMap(1, ("invented",)),
                ),
            ),
            "invented_semantic_node",
        ),
        (
            replace(
                _response(),
                clause_to_semantic_nodes=(ClauseNodeMap(0, ("state",)),),
            ),
            "missing_semantic_node",
        ),
        (
            replace(
                _response(),
                clause_to_semantic_nodes=(
                    ClauseNodeMap(0, ("state",)),
                    ClauseNodeMap(1, ("target", "target")),
                ),
            ),
            "duplicated_semantic_node",
        ),
        (
            replace(_response(), requested_target_type=TargetKind.WEIGHTED_MEAN),
            "target_type_changed",
        ),
        (
            replace(_response(), requested_question_intent="ask for a different value"),
            "question_intent_changed",
        ),
        (
            replace(_response(), style_id="invented-style"),
            "style_id_changed",
        ),
    ),
)
def test_unsafe_realizations_are_rejected(response: RealizationResponse, reason: str) -> None:
    assert reason in validate_realization_response(_request(), response)


def test_rate_interval_is_immutable_and_required() -> None:
    request = _request(include_rate=True)
    assert validate_realization_response(request, _response(include_rate=True)) == ()
    missing = replace(
        _response(include_rate=True),
        question_template=(
            "<ENTITY_1> records <QUANTITY_1> <UNIT_1>. How many <TARGET_ENTITY> are requested?"
        ),
    )
    reasons = validate_realization_response(request, missing)
    assert "template_placeholder_set_mismatch" in reasons
    assert "rate_denominator_missing" in reasons


def test_event_cannot_claim_placeholders_from_another_clause() -> None:
    response = replace(
        _response(),
        clause_to_semantic_nodes=(
            ClauseNodeMap(1, ("state",)),
            ClauseNodeMap(0, ("target",)),
        ),
    )
    assert "semantic_node_placeholder_mismatch" in validate_realization_response(
        _request(), response
    )


def test_unapproved_discourse_order_is_rejected() -> None:
    response = replace(
        _response(),
        clause_to_semantic_nodes=(
            ClauseNodeMap(0, ("target",)),
            ClauseNodeMap(1, ("state",)),
        ),
    )
    assert "discourse_order_changed" in validate_realization_response(_request(), response)


def test_invalid_replacement_inventory_cannot_compile() -> None:
    with pytest.raises(RealizationContractError, match="exactly match"):
        fill_validated_template(_request(), _response(), {"<ENTITY_1>": "Avery"})
