"""Deterministic rendered-language quality checks."""

from __future__ import annotations

from dataclasses import replace

from foundry.synthesis.generators import exact_value
from foundry.synthesis.object_units import (
    Countability,
    ObjectKind,
    QuantityUnit,
)
from foundry.synthesis.quality import RenderQualityMetadata, validate_rendered_candidate

_KIND = ObjectKind(
    family="test_token",
    singular="test token",
    plural="test tokens",
    countability=Countability.DISCRETE,
    unit=QuantityUnit.ITEM,
    combination_key="test_token:item",
    transferable=True,
    supported_verbs=("move",),
    supported_containers=("inventory",),
)


def _metadata() -> RenderQualityMetadata:
    return RenderQualityMetadata(
        scenario_id="original-fixture",
        renderer_family="controlled",
        clauses=("A complete premise is stated.",),
        declared_entity_ids=("inventory",),
        referenced_entity_ids=("inventory",),
        pronoun_referent_ids=(),
        noun_forms=(),
        combination_groups=((_KIND,),),
        unit_transitions=(),
        target_symbol="answer",
        target_mentions=1,
        conclusion="Determine the final quantity.",
        constraints_tied=False,
        grammar_complete=True,
    )


def _validate(question: str, metadata: RenderQualityMetadata) -> tuple[str, ...]:
    return validate_rendered_candidate(
        question=question,
        completion="Reasoning trace.",
        answer=exact_value(3),
        output_contract_enabled=False,
        metadata=metadata,
    )


def test_complete_imperative_question_is_valid() -> None:
    question = "A complete premise is stated. Determine the final quantity."
    assert _validate(question, _metadata()) == ()


def test_rule_based_quality_rejects_malformed_or_duplicate_language() -> None:
    metadata = replace(
        _metadata(),
        clauses=("Repeated premise.", "Repeated premise."),
        conclusion="Determine the final quantity.",
    )
    question = "Repeated premise. Repeated premise.. Determine the final quantity."

    reasons = _validate(question, metadata)
    assert "malformed_punctuation" in reasons
    assert "duplicated_clause" in reasons


def test_target_and_conclusion_must_be_unique_and_consistent() -> None:
    metadata = replace(_metadata(), target_mentions=2, conclusion="Find a missing conclusion.")
    reasons = _validate("A complete premise is stated. Determine the final quantity.", metadata)

    assert "target_is_not_unique" in reasons
    assert "conclusion_inconsistent" in reasons
