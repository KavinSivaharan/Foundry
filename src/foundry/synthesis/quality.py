"""Typed, deterministic rendering-quality validation without language models."""

from __future__ import annotations

import re
from dataclasses import dataclass

from foundry.synthesis.object_units import (
    ObjectKind,
    TypedLedgerOperation,
    TypedQuantity,
    validate_combination,
)
from foundry.synthesis.schema import ExactValue

_MALFORMED_PUNCTUATION = re.compile(r"(?:\.\.|;;|,,|\s+[?.!,;])")
_SENTENCE_START = re.compile(r"(?<=[.!?])\s+")
_REQUEST_CUE = re.compile(
    r"^(?:what|how|which|calculate|compute|determine|find|report)", re.IGNORECASE
)


@dataclass(frozen=True)
class NounFormEvidence:
    """A rendered noun tied to the quantity that selects its grammatical number."""

    quantity: int
    rendered_noun: str
    object_kind: ObjectKind


@dataclass(frozen=True)
class UnitTransitionEvidence:
    """One possible unit change and whether an explicit conversion authorizes it."""

    source_unit: str
    target_unit: str
    conversion_explicit: bool


@dataclass(frozen=True)
class RenderQualityMetadata:
    """Auditable metadata assembled by a controlled renderer."""

    scenario_id: str
    renderer_family: str
    clauses: tuple[str, ...]
    declared_entity_ids: tuple[str, ...]
    referenced_entity_ids: tuple[str, ...]
    pronoun_referent_ids: tuple[str, ...]
    noun_forms: tuple[NounFormEvidence, ...]
    combination_groups: tuple[tuple[ObjectKind, ...], ...]
    unit_transitions: tuple[UnitTransitionEvidence, ...]
    target_symbol: str
    target_mentions: int
    conclusion: str
    constraints_tied: bool
    grammar_complete: bool
    quantities: tuple[TypedQuantity, ...] = ()
    operations: tuple[TypedLedgerOperation, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.renderer_family.strip():
            raise ValueError("quality metadata requires scenario and renderer IDs")


def _has_repeated_adjacent_phrase(text: str) -> bool:
    words = re.findall(r"[a-z0-9]+", text.lower())
    for width in range(2, 6):
        for index in range(0, len(words) - (2 * width) + 1):
            if words[index : index + width] == words[index + width : index + (2 * width)]:
                return True
    return False


def validate_rendered_candidate(
    *,
    question: str,
    completion: str,
    answer: ExactValue,
    output_contract_enabled: bool,
    metadata: RenderQualityMetadata,
) -> tuple[str, ...]:
    """Return stable quality failures in pipeline order."""

    reasons: list[str] = []
    conclusion = metadata.conclusion.strip()
    if (
        not metadata.grammar_complete
        or not question.strip().endswith(("?", "."))
        or (not conclusion.endswith("?") and _REQUEST_CUE.match(conclusion) is None)
    ):
        reasons.append("grammar_or_sentence_incomplete")
    if _MALFORMED_PUNCTUATION.search(question):
        reasons.append("malformed_punctuation")
    sentences = [item.strip() for item in _SENTENCE_START.split(question.strip()) if item.strip()]
    if any(sentence[0].isalpha() and not sentence[0].isupper() for sentence in sentences):
        reasons.append("sentence_capitalization_failure")
    normalized_clauses = [" ".join(clause.lower().split()) for clause in metadata.clauses]
    if not normalized_clauses or any(not clause for clause in normalized_clauses):
        reasons.append("empty_clause")
    if len(normalized_clauses) != len(set(normalized_clauses)):
        reasons.append("duplicated_clause")
    if _has_repeated_adjacent_phrase(question):
        reasons.append("repeated_adjacent_phrase")
    declared = set(metadata.declared_entity_ids)
    if not set(metadata.referenced_entity_ids) <= declared:
        reasons.append("missing_entity_reference")
    if not set(metadata.pronoun_referent_ids) <= declared:
        reasons.append("unresolved_pronoun")
    for evidence in metadata.noun_forms:
        expected = evidence.object_kind.render_noun(evidence.quantity)
        if evidence.rendered_noun != expected:
            reasons.append("singular_plural_mismatch")
    for group in metadata.combination_groups:
        reasons.extend(validate_combination(group))
    for transition in metadata.unit_transitions:
        if transition.source_unit != transition.target_unit and not transition.conversion_explicit:
            reasons.append("implicit_unit_change")
    if metadata.target_mentions != 1 or not metadata.target_symbol.strip():
        reasons.append("target_is_not_unique")
    if not conclusion or conclusion not in question:
        reasons.append("conclusion_inconsistent")
    if metadata.constraints_tied:
        reasons.append("tied_or_unintended_constraints")
    expected_line = f"Final answer: {answer.render()}"
    lines = completion.strip().splitlines()
    if output_contract_enabled:
        if not lines or lines[-1] != expected_line or lines.count(expected_line) != 1:
            reasons.append("output_contract_failure")
    elif any(line.startswith("Final answer:") for line in lines):
        reasons.append("unexpected_output_contract")
    return tuple(dict.fromkeys(reasons))
