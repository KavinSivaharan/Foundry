"""Typed English composition and auditable surface provenance for the template bank."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum

from foundry.synthesis.realization.ir import (
    BookkeepingProblemIR,
    CompiledRealization,
    DiscreteProblemIR,
    LexemeSpec,
    MorphologyUse,
    RateProblemIR,
)
from foundry.synthesis.realization.morphology import noun_form
from foundry.synthesis.template_bank.contracts import TemplateSpec

ProblemIR = BookkeepingProblemIR | RateProblemIR | DiscreteProblemIR
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:st|nd|rd|th)?|\d+|%|[^\w\s]")
_RAW_IDENTIFIER = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class SurfaceSourceKind(StrEnum):
    """Allowed origins for every emitted surface token."""

    FIXED_GRAMMAR = "fixed_grammar"
    APPROVED_LEXEME = "approved_lexeme"
    ENTITY_SLOT = "entity_slot"
    QUANTITY_SLOT = "quantity_slot"
    UNIT_SLOT = "unit_slot"
    MORPHOLOGY_OUTPUT = "morphology_output"
    PUNCTUATION = "punctuation"
    OPTIONAL_CONTEXT = "optional_context"


@dataclass(frozen=True)
class SurfaceTokenEvidence:
    """One rendered token and its deterministic source."""

    token: str
    source_kind: SurfaceSourceKind
    source_id: str


@dataclass(frozen=True)
class SurfaceProvenanceReport:
    """Complete, hashable proof that surface text came from licensed sources."""

    entries: tuple[SurfaceTokenEvidence, ...]
    semantic_node_counts: tuple[tuple[str, int], ...]
    reasons: tuple[str, ...]

    @property
    def provenance_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NounPhraseSpec:
    """One typed noun phrase with a single explicit head."""

    head: LexemeSpec
    quantity: int
    attributive_modifiers: tuple[LexemeSpec, ...] = ()
    grouping_noun: LexemeSpec | None = None
    container_noun: LexemeSpec | None = None

    def render(self) -> tuple[str, MorphologyUse]:
        if self.quantity < 0:
            raise ValueError("noun-phrase quantities cannot be negative")
        role_lexemes = (
            *self.attributive_modifiers,
            *((self.grouping_noun,) if self.grouping_noun else ()),
            *((self.container_noun,) if self.container_noun else ()),
            self.head,
        )
        role_ids = [item.lexeme_id for item in role_lexemes]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("noun phrases cannot repeat a lexical head or modifier")
        modifiers = [item.attributive for item in self.attributive_modifiers]
        if self.grouping_noun is not None:
            modifiers.append(self.grouping_noun.attributive)
        if self.container_noun is not None:
            modifiers.append(self.container_noun.attributive)
        head, morphology = noun_form(self.head, self.quantity)
        words = [*modifiers, head]
        lowered = [word.lower() for word in words]
        if any(left == right for left, right in zip(lowered, lowered[1:], strict=False)):
            raise ValueError("noun phrases cannot contain adjacent duplicate heads")
        return " ".join(words), morphology


def numeric_ordinal(value: int) -> str:
    """Render an English numeric ordinal without guessing a universal ``th`` suffix."""

    if value <= 0:
        raise ValueError("ordinals must be positive")
    last_two = value % 100
    if 11 <= last_two <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


_WORD_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
}


def word_ordinal(value: int) -> str:
    """Return only explicitly approved word-form ordinals."""

    try:
        return _WORD_ORDINALS[value]
    except KeyError as error:
        raise ValueError("word-form ordinal is unsupported") from error


def validate_surface_text(text: str, internal_identifiers: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Reject identifier leakage, invalid ordinals, and untyped noun repetition."""

    reasons: list[str] = []
    lowered = " ".join(text.lower().split())
    if _RAW_IDENTIFIER.search(text):
        reasons.append("internal_identifier_leak")
    for identifier in internal_identifiers:
        normalized = " ".join(identifier.lower().replace("_", " ").replace(".", " ").split())
        leaf = " ".join(identifier.lower().split(".")[-1].replace("_", " ").split())
        normalized_pattern = (
            r"(?<!\w)" + r"\s+".join(re.escape(word) for word in normalized.split()) + r"(?!\w)"
        )
        leaf_pattern = (
            r"(?<!\w)" + r"\s+".join(re.escape(word) for word in leaf.split()) + r"(?!\w)"
        )
        if (len(normalized.split()) > 1 and re.search(normalized_pattern, lowered)) or (
            len(leaf.split()) > 1 and re.search(leaf_pattern, lowered)
        ):
            reasons.append("internal_frame_label_leak")
            break
    words = [match.group(0).lower() for match in _WORD.finditer(text)]
    if any(left == right for left, right in zip(words, words[1:], strict=False)):
        reasons.append("adjacent_duplicate_noun")
    for match in _ORDINAL.finditer(text):
        value = int(match.group(1))
        if match.group(0).lower() != numeric_ordinal(value):
            reasons.append("invalid_ordinal_morphology")
            break
    return tuple(dict.fromkeys(reasons))


def _domain_phrases(problem: ProblemIR) -> dict[str, tuple[SurfaceSourceKind, str]]:
    domain = problem.domain
    phrases: dict[str, tuple[SurfaceSourceKind, str]] = {}
    for entity in (
        domain.actor,
        domain.item,
        domain.primary_location,
        domain.secondary_location,
        domain.destination_location,
        domain.container,
    ):
        if entity.proper_name:
            phrases[entity.proper_name.lower()] = (SurfaceSourceKind.ENTITY_SLOT, entity.entity_id)
        for form in (entity.lexeme.singular, entity.lexeme.plural, entity.lexeme.attributive):
            phrases[form.lower()] = (SurfaceSourceKind.APPROVED_LEXEME, entity.lexeme.lexeme_id)
    phrases[domain.setting.lower()] = (SurfaceSourceKind.APPROVED_LEXEME, domain.domain_id)
    if domain.safe_context:
        phrases[domain.safe_context.lower()] = (
            SurfaceSourceKind.OPTIONAL_CONTEXT,
            domain.domain_id,
        )
    return phrases


def audit_surface_provenance(
    problem: ProblemIR,
    realization: CompiledRealization,
    template: TemplateSpec,
) -> SurfaceProvenanceReport:
    """Account for every token and validate one-to-one semantic-node realization."""

    reasons = list(validate_surface_text(realization.text, (template.semantic_frame,)))
    expected = tuple(problem.required_node_ids)
    counts = tuple(
        sorted(
            (node, sum(entry.node_id == node for entry in realization.coverage))
            for node in expected
        )
    )
    if any(count != 1 for _, count in counts):
        reasons.append("semantic_node_realization_count")
    if any(entry.node_id not in expected for entry in realization.coverage):
        reasons.append("unapproved_semantic_node")

    phrases = _domain_phrases(problem)
    phrases[template.surface_lexeme.text.lower()] = (
        SurfaceSourceKind.APPROVED_LEXEME,
        template.surface_lexeme.lexeme_id,
    )
    morphology_forms = {
        use.rendered.lower(): use.lexeme.lexeme_id for use in realization.morphology_uses
    }
    unit_forms = {use.numerator_rendered.lower(): use.unit_id for use in realization.unit_uses}
    unit_forms.update(
        {
            use.denominator_rendered.lower(): use.unit_id
            for use in realization.unit_uses
            if use.denominator_rendered
        }
    )
    entries: list[SurfaceTokenEvidence] = []
    for token in _TOKEN.findall(realization.text):
        lowered = token.lower()
        if token.isdigit() or _ORDINAL.fullmatch(token):
            source = (SurfaceSourceKind.QUANTITY_SLOT, "typed_numeric_surface")
        elif not any(character.isalnum() for character in token):
            source = (SurfaceSourceKind.PUNCTUATION, "compiler_punctuation")
        elif lowered in morphology_forms:
            source = (SurfaceSourceKind.MORPHOLOGY_OUTPUT, morphology_forms[lowered])
        elif lowered in unit_forms:
            source = (SurfaceSourceKind.UNIT_SLOT, unit_forms[lowered])
        else:
            source = next(
                (
                    value
                    for phrase, value in sorted(phrases.items(), key=lambda item: -len(item[0]))
                    if lowered in phrase.split()
                ),
                (SurfaceSourceKind.FIXED_GRAMMAR, f"{template.template_id}:grammar"),
            )
        entries.append(SurfaceTokenEvidence(token, source[0], source[1]))
    if not entries or len(entries) != len(_TOKEN.findall(realization.text)):
        reasons.append("surface_token_unaccounted")
    return SurfaceProvenanceReport(tuple(entries), counts, tuple(dict.fromkeys(reasons)))
