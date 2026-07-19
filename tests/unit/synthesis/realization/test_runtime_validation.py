"""Strict rejection tests for model-produced realization text."""

from __future__ import annotations

import pytest

from foundry.synthesis.realization.ir import TargetKind
from foundry.synthesis.realization.model_contracts import (
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
    parse_realization_response,
    validate_filled_question,
    validate_realization_response,
)


def _request() -> RealizationRequest:
    return RealizationRequest(
        request_id="fixture",
        category="original_fixture",
        semantic_frame="rate_total",
        ordered_events=(
            SemanticEventSpec("fact", "state a rate", ("<QUANTITY_A>", "<UNIT_A>")),
            SemanticEventSpec("target", "ask for a total", ("<TARGET_ENTITY>",)),
        ),
        placeholders=(
            PlaceholderSpec("<QUANTITY_A>", PlaceholderKind.QUANTITY, "fact"),
            PlaceholderSpec("<UNIT_A>", PlaceholderKind.UNIT, "fact"),
            PlaceholderSpec("<TARGET_ENTITY>", PlaceholderKind.TARGET_ENTITY, "target"),
        ),
        target_type=TargetKind.TOTAL_QUANTITY,
        required_question_intent="ask for the total quantity produced or available",
        allowed_discourse_orders=(("fact", "target"),),
        forbidden_transformations=("do not alter facts",),
        style=StyleControls("formal_neutral", "easy", ("active",), False),
    )


def test_thinking_or_markdown_is_rejected_before_json_repair() -> None:
    with pytest.raises(RealizationContractError, match="forbidden markup"):
        parse_realization_response("<think>hidden</think> {}")
    with pytest.raises(RealizationContractError, match="forbidden markup"):
        parse_realization_response("```json\n{}\n```")


def test_numeric_answer_calculation_and_pronoun_content_reject() -> None:
    request = _request()
    response = RealizationResponse(
        question_template=(
            "It handles <QUANTITY_A> <UNIT_A>. Therefore, how many <TARGET_ENTITY> equal 42?"
        ),
        placeholder_inventory=("<QUANTITY_A>", "<UNIT_A>", "<TARGET_ENTITY>"),
        clause_to_semantic_nodes=(ClauseNodeMap(0, ("fact",)), ClauseNodeMap(1, ("target",))),
        requested_target_type=TargetKind.TOTAL_QUANTITY,
        requested_question_intent=request.required_question_intent,
        style_id=request.style.style_id,
    )
    reasons = validate_realization_response(request, response)
    assert "invented_numeric_literal" in reasons
    assert "calculation_content_forbidden" in reasons
    assert "unlicensed_pronoun" in reasons


def test_filled_question_quality_rejects_unresolved_or_ambiguous_surface() -> None:
    reasons = validate_filled_question("they move <QUANTITY_A> boxes.. How many remain?")
    assert "unresolved_placeholder" in reasons
    assert "filled_question_malformed_punctuation" in reasons
    assert "filled_question_unlicensed_pronoun" in reasons
    assert "filled_question_incomplete" in reasons
