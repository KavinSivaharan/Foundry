from __future__ import annotations

from dataclasses import dataclass

from foundry.synthesis.realization.compact_contracts import (
    CompactAnchorSpec,
    CompactRealizationRequest,
    CompactSegmentSpec,
)
from foundry.synthesis.realization.compact_validation import (
    fill_compact_response,
    parse_compact_response,
    validate_compact_response,
)
from foundry.synthesis.realization.model_contracts import PlaceholderKind, PlaceholderSpec


@dataclass(frozen=True)
class FakeTaggedModel:
    beams: tuple[str, ...]

    def generate(self, _: CompactRealizationRequest) -> tuple[str, ...]:
        return self.beams


def test_fake_model_first_passing_beam_is_selected_without_repair() -> None:
    placeholders = (
        PlaceholderSpec("<ENTITY_A>", PlaceholderKind.ENTITY, "fact"),
        PlaceholderSpec("<QUANTITY_A>", PlaceholderKind.QUANTITY, "fact"),
        PlaceholderSpec("<UNIT_A>", PlaceholderKind.UNIT, "fact"),
        PlaceholderSpec("<TARGET_A>", PlaceholderKind.TARGET_ENTITY, "target"),
    )
    request = CompactRealizationRequest(
        "original-fake-integration",
        (
            CompactSegmentSpec(
                "E1",
                "fact",
                ("<ENTITY_A>", "<QUANTITY_A>", "<UNIT_A>"),
                (CompactAnchorSpec("<HOLDS_E1>", "holds"),),
            ),
        ),
        CompactSegmentSpec(
            "Q",
            "target",
            ("<TARGET_A>",),
            (CompactAnchorSpec("<ASK_COUNT_Q>", "How many"),),
        ),
        placeholders,
    )
    fake = FakeTaggedModel(
        (
            "<E1><ENTITY_A> <HOLDS_E1> 12 <QUANTITY_A> <UNIT_A>.</E1>"
            "<Q><ASK_COUNT_Q> <TARGET_A> remain?</Q>",
            "<E1><ENTITY_A> <HOLDS_E1> <QUANTITY_A> <UNIT_A>.</E1>"
            "<Q><ASK_COUNT_Q> <TARGET_A> remain?</Q>",
            "<E1><ENTITY_A> <HOLDS_E1> <QUANTITY_A> <UNIT_A>.</E1>",
        )
    )
    selected: tuple[int, str] | None = None
    for index, raw in enumerate(fake.generate(request), start=1):
        try:
            response = parse_compact_response(raw)
        except ValueError:
            continue
        if validate_compact_response(request, response):
            continue
        filled = fill_compact_response(
            request,
            response,
            {
                "<ENTITY_A>": "Rin",
                "<QUANTITY_A>": "4",
                "<UNIT_A>": "tiles",
                "<TARGET_A>": "tiles",
                "<HOLDS_E1>": "holds",
                "<ASK_COUNT_Q>": "How many",
            },
        )
        selected = (index, filled.question)
        break
    assert selected == (2, "Rin holds 4 tiles. How many tiles remain?")
