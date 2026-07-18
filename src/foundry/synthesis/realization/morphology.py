"""Centralized deterministic English morphology and agreement."""

from __future__ import annotations

from foundry.synthesis.realization.ir import CountBehavior, LexemeSpec, MorphologyUse, VerbSpec


def count_lexeme(
    lexeme_id: str,
    singular: str,
    plural: str,
    *,
    attributive: str | None = None,
    prepositions: tuple[str, ...] = (),
) -> LexemeSpec:
    """Construct a fully explicit count-noun lexeme."""

    return LexemeSpec(
        lexeme_id=lexeme_id,
        singular=singular,
        plural=plural,
        attributive=singular if attributive is None else attributive,
        count_behavior=CountBehavior.COUNT,
        supported_prepositions=prepositions,
    )


def mass_lexeme(
    lexeme_id: str,
    form: str,
    *,
    attributive: str | None = None,
    prepositions: tuple[str, ...] = (),
) -> LexemeSpec:
    """Construct a fully explicit mass-noun lexeme."""

    return LexemeSpec(
        lexeme_id=lexeme_id,
        singular=form,
        plural=form,
        attributive=form if attributive is None else attributive,
        count_behavior=CountBehavior.MASS,
        supported_prepositions=prepositions,
    )


def noun_form(lexeme: LexemeSpec, quantity: int) -> tuple[str, MorphologyUse]:
    """Render quantity agreement without guessing morphology."""

    if quantity < 0:
        raise ValueError("noun quantities cannot be negative")
    if lexeme.count_behavior is CountBehavior.MASS:
        rendered = lexeme.singular
    else:
        rendered = lexeme.singular if quantity == 1 else lexeme.plural
    return rendered, MorphologyUse(lexeme, "head", quantity, rendered)


def attributive_form(lexeme: LexemeSpec) -> tuple[str, MorphologyUse]:
    """Render an explicit attributive form, never a plural head form."""

    if not lexeme.attributive.strip():
        raise ValueError("attributive morphology is unavailable")
    return lexeme.attributive, MorphologyUse(lexeme, "attributive", None, lexeme.attributive)


def indefinite_article(lexeme: LexemeSpec) -> str:
    """Return a deterministic article for an explicit singular form."""

    if lexeme.count_behavior is CountBehavior.MASS:
        raise ValueError("mass nouns cannot take an indefinite article")
    return "an" if lexeme.singular[0].lower() in "aeiou" else "a"


def verb_form(verb: VerbSpec, *, subject_quantity: int, tense: str) -> str:
    """Render explicit subject-verb agreement."""

    if tense == "present":
        return verb.third_person_singular if subject_quantity == 1 else verb.base
    if tense == "past":
        return verb.past
    if tense == "participle":
        return verb.past_participle
    raise ValueError("unsupported verb tense")


SHELF = count_lexeme("shelf", "shelf", "shelves")
PERSON = count_lexeme("person", "person", "people")
BOX = count_lexeme("box", "box", "boxes")
ITEM = count_lexeme("item", "item", "items")
CAPACITY = count_lexeme("capacity", "capacity", "capacities")
GROUP = count_lexeme("group", "group", "groups")
INTERVAL = count_lexeme("interval", "interval", "intervals")
PART = count_lexeme("part", "part", "parts")
PANEL = count_lexeme("panel", "panel", "panels")
MARK = count_lexeme("mark", "mark", "marks")

TRANSFER = VerbSpec("transfer", "transfer", "transfers", "transferred", "transferred")
RECEIVE = VerbSpec("receive", "receive", "receives", "received", "received")
REMOVE = VerbSpec("remove", "remove", "removes", "removed", "removed")
HOLD = VerbSpec("hold", "hold", "holds", "held", "held")
PRODUCE = VerbSpec("produce", "produce", "produces", "produced", "produced")
DISTRIBUTE = VerbSpec("distribute", "distribute", "distributes", "distributed", "distributed")
ASSEMBLE = VerbSpec("assemble", "assemble", "assembles", "assembled", "assembled")
