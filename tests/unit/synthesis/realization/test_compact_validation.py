from __future__ import annotations

from dataclasses import replace

import pytest

from foundry.synthesis.realization.compact_contracts import (
    CompactAnchorSpec,
    CompactRealizationRequest,
    CompactSegmentSpec,
)
from foundry.synthesis.realization.compact_validation import (
    CompactContractError,
    fill_compact_response,
    parse_compact_response,
    validate_compact_response,
)
from foundry.synthesis.realization.model_contracts import PlaceholderKind, PlaceholderSpec


def _request() -> tuple[CompactRealizationRequest, dict[str, str]]:
    placeholders = (
        PlaceholderSpec("<ENTITY_A>", PlaceholderKind.ENTITY, "start"),
        PlaceholderSpec("<QUANTITY_A>", PlaceholderKind.QUANTITY, "start"),
        PlaceholderSpec("<UNIT_A>", PlaceholderKind.UNIT, "start"),
        PlaceholderSpec("<LOCATION_A>", PlaceholderKind.LOCATION, "change"),
        PlaceholderSpec("<QUANTITY_B>", PlaceholderKind.QUANTITY, "change"),
        PlaceholderSpec("<UNIT_B>", PlaceholderKind.UNIT, "change"),
        PlaceholderSpec("<TARGET_A>", PlaceholderKind.TARGET_ENTITY, "target"),
    )
    request = CompactRealizationRequest(
        request_id="original-fixture-compact-001",
        events=(
            CompactSegmentSpec(
                "E1",
                "start",
                ("<ENTITY_A>", "<QUANTITY_A>", "<UNIT_A>"),
                (CompactAnchorSpec("<HOLDS_E1>", "holds"),),
            ),
            CompactSegmentSpec(
                "E2",
                "change",
                ("<LOCATION_A>", "<QUANTITY_B>", "<UNIT_B>"),
                (CompactAnchorSpec("<RECEIVES_E2>", "receives"),),
            ),
        ),
        question=CompactSegmentSpec(
            "Q",
            "target",
            ("<TARGET_A>",),
            (CompactAnchorSpec("<ASK_REMAINING_Q>", "How many"),),
        ),
        placeholders=placeholders,
    )
    replacements = {
        "<ENTITY_A>": "Nora",
        "<QUANTITY_A>": "5",
        "<UNIT_A>": "tokens",
        "<LOCATION_A>": "the depot",
        "<QUANTITY_B>": "3",
        "<UNIT_B>": "tokens",
        "<TARGET_A>": "tokens",
        "<HOLDS_E1>": "holds",
        "<RECEIVES_E2>": "receives",
        "<ASK_REMAINING_Q>": "How many",
    }
    return request, replacements


VALID = """<E1><ENTITY_A> <HOLDS_E1> <QUANTITY_A> <UNIT_A>.</E1>
<E2><LOCATION_A> <RECEIVES_E2> <QUANTITY_B> <UNIT_B>.</E2>
<Q><ASK_REMAINING_Q> <TARGET_A> remain?</Q>"""


def test_valid_compact_response_round_trips() -> None:
    request, replacements = _request()
    response = parse_compact_response(VALID)
    assert validate_compact_response(request, response) == ()
    filled = fill_compact_response(request, response, replacements)
    assert filled.question == (
        "Nora holds 5 tokens. the depot receives 3 tokens. How many tokens remain?"
    )


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("\n".join((VALID.splitlines()[0], VALID.splitlines()[2])), "event_tag_set_mismatch"),
        (VALID.replace("</E1>", "</E1><E1><HOLDS_E1>.</E1>", 1), "event_tag_occurrence_mismatch"),
        (
            "\n".join((VALID.splitlines()[1], VALID.splitlines()[0], VALID.splitlines()[2])),
            "event_order_changed",
        ),
        (
            VALID.replace("<QUANTITY_A>", "<TEMP_TOKEN>", 1)
            .replace("<QUANTITY_B>", "<QUANTITY_A>", 1)
            .replace("<TEMP_TOKEN>", "<QUANTITY_B>", 1),
            "placeholder_assignment_mismatch",
        ),
        (VALID.replace(" <HOLDS_E1>", "", 1), "semantic_anchor_missing"),
        (VALID.replace("<QUANTITY_A>", "<QUANTITY_X>", 1), "placeholder_assignment_mismatch"),
        (VALID.replace("<QUANTITY_A>", "7 <QUANTITY_A>", 1), "raw_numeric_literal"),
        (VALID.replace("<HOLDS_E1>", "<HOLDS_E1> therefore", 1), "answer_or_calculation_content"),
    ],
)
def test_compact_validation_rejects_structural_defects(raw: str, reason: str) -> None:
    request, _ = _request()
    response = parse_compact_response(raw)
    assert reason in validate_compact_response(request, response)


@pytest.mark.parametrize(
    "raw",
    [
        f"Explanation first.\n{VALID}",
        f"{VALID}\nTrailing explanation.",
        VALID.rsplit("</Q>", 1)[0],
        VALID.replace("</E2>", "", 1),
        f"```\n{VALID}\n```",
    ],
)
def test_compact_parser_rejects_extra_or_truncated_text(raw: str) -> None:
    with pytest.raises(CompactContractError):
        parse_compact_response(raw)


def test_compact_request_rejects_cross_assigned_placeholder() -> None:
    request, _ = _request()
    crossed = replace(
        request.events[0], placeholders=(*request.events[0].placeholders, "<LOCATION_A>")
    )
    with pytest.raises(ValueError, match="only one segment"):
        replace(request, events=(crossed, request.events[1]))
