"""Typed object, unit, transfer, and grammar safety tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from foundry.synthesis.generators import exact_value
from foundry.synthesis.object_units import (
    Countability,
    LedgerOperationKind,
    LocationSpec,
    ObjectKind,
    QuantityUnit,
    TypedLedgerOperation,
    TypedQuantity,
    validate_combination,
    validate_ledger_plan,
    validate_transfer,
)
from foundry.synthesis.quality import (
    NounFormEvidence,
    RenderQualityMetadata,
    UnitTransitionEvidence,
    validate_rendered_candidate,
)


def _kind(
    family: str = "archive_card",
    *,
    unit: QuantityUnit = QuantityUnit.ITEM,
) -> ObjectKind:
    return ObjectKind(
        family=family,
        singular=family.replace("_", " "),
        plural=f"{family.replace('_', ' ')}s",
        countability=Countability.DISCRETE,
        unit=unit,
        combination_key=f"{family}:{unit}",
        transferable=True,
        supported_verbs=("move",),
        supported_containers=("inventory",),
    )


def _locations(family: str) -> tuple[LocationSpec, LocationSpec]:
    return (
        LocationSpec("source", "intake shelf", "inventory", (family,)),
        LocationSpec("ledger", "archive cabinet", "inventory", (family,)),
    )


def _metadata(kind: ObjectKind) -> RenderQualityMetadata:
    return RenderQualityMetadata(
        scenario_id="sanitized-regression",
        renderer_family="fixture",
        clauses=("A typed inventory is recorded.",),
        declared_entity_ids=("ledger",),
        referenced_entity_ids=("ledger",),
        pronoun_referent_ids=(),
        noun_forms=(NounFormEvidence(2, kind.plural, kind),),
        combination_groups=((kind, kind),),
        unit_transitions=(),
        target_symbol="ending",
        target_mentions=1,
        conclusion="What is the ending inventory?",
        constraints_tied=False,
        grammar_complete=True,
    )


def _quality_reasons(metadata: RenderQualityMetadata) -> tuple[str, ...]:
    return validate_rendered_candidate(
        question="A typed inventory is recorded. What is the ending inventory?",
        completion="A deterministic trace.",
        answer=exact_value(4),
        output_contract_enabled=False,
        metadata=metadata,
    )


def test_compatible_inventory_and_valid_transfer_pass() -> None:
    kind = _kind()
    source, ledger = _locations(kind.family)
    quantity = TypedQuantity("delivered", 4, kind, "source")
    operation = TypedLedgerOperation(
        LedgerOperationKind.TRANSFER_IN,
        "delivered",
        "move",
        "source",
        "ledger",
    )

    assert validate_combination((kind, kind)) == ()
    assert validate_transfer(object_kind=kind, origin=source, destination=ledger, verb="move") == ()
    assert (
        validate_ledger_plan(
            ledger_kind=kind,
            quantities=(quantity,),
            operations=(operation,),
            locations=(source, ledger),
        )
        == ()
    )


def test_incompatible_object_and_unit_combinations_fail() -> None:
    cards = _kind()
    tags = _kind("signal_tag")
    packages = _kind(unit=QuantityUnit.PACKAGE)

    assert validate_combination((cards, tags)) == ("incompatible_object_or_unit_combination",)
    assert validate_combination((cards, packages)) == ("incompatible_object_or_unit_combination",)


def test_invalid_transfer_endpoint_and_item_type_fail() -> None:
    cards = _kind()
    tags = _kind("signal_tag")
    source, ledger = _locations(cards.family)

    assert "transfer_location_is_incompatible" in validate_transfer(
        object_kind=tags, origin=source, destination=ledger, verb="move"
    )
    assert "transfer_endpoints_are_identical" in validate_transfer(
        object_kind=cards, origin=source, destination=source, verb="move"
    )


@pytest.mark.parametrize(
    ("quantity", "rendered"),
    ((1, "archive card"), (0, "archive cards"), (2, "archive cards")),
)
def test_singular_and_plural_forms_render_correctly(quantity: int, rendered: str) -> None:
    assert _kind().render_noun(quantity) == rendered


@pytest.mark.parametrize("other_family", ("marker", "coil", "parcel", "module"))
def test_four_sanitized_bookkeeping_regressions_are_rejected(
    other_family: str,
) -> None:
    ledger_kind = _kind()
    other_kind = _kind(other_family)
    metadata = replace(_metadata(ledger_kind), combination_groups=((ledger_kind, other_kind),))

    assert "incompatible_object_or_unit_combination" in _quality_reasons(metadata)


def test_sanitized_discrete_capacity_regression_is_rejected() -> None:
    kind = _kind()
    metadata = replace(
        _metadata(kind),
        noun_forms=(NounFormEvidence(2, kind.singular, kind),),
        constraints_tied=True,
    )

    reasons = _quality_reasons(metadata)
    assert "singular_plural_mismatch" in reasons
    assert "tied_or_unintended_constraints" in reasons


def test_quality_rejects_implicit_unit_change_and_unresolved_reference() -> None:
    kind = _kind()
    metadata = replace(
        _metadata(kind),
        referenced_entity_ids=("missing",),
        pronoun_referent_ids=("unknown",),
        unit_transitions=(UnitTransitionEvidence(QuantityUnit.ITEM, QuantityUnit.PACKAGE, False),),
    )

    reasons = _quality_reasons(metadata)
    assert "missing_entity_reference" in reasons
    assert "unresolved_pronoun" in reasons
    assert "implicit_unit_change" in reasons
