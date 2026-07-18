"""Ordered exact, template, structure, and lexical duplicate screening."""

from __future__ import annotations

from dataclasses import dataclass

from foundry.synthesis.contamination import (
    DevelopmentQuestion,
    normalize_text,
    normalized_text_sha256,
    numeric_template_sha256,
)
from foundry.synthesis.generators import CandidateDraft


@dataclass(frozen=True)
class LexicalScreenResult:
    """Content-free outcome of the four non-semantic duplicate stages."""

    rejection_reason: str | None
    matched_scope: str | None
    matched_identifier_prefix: str | None
    maximum_ngram_jaccard: float


@dataclass(frozen=True)
class _Reference:
    scope: str
    identifier: str
    exact_hash: str
    template_hash: str
    structure_hash: str | None
    ngrams: frozenset[tuple[str, ...]]


def token_ngrams(text: str, size: int = 5) -> frozenset[tuple[str, ...]]:
    """Create frozen number-neutral token n-grams."""

    if size != 5:
        raise ValueError("the synthesis contamination contract requires five-token n-grams")
    tokens = normalize_text(text, replace_numbers=True).split()
    if not tokens:
        return frozenset()
    if len(tokens) < size:
        return frozenset((tuple(tokens),))
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _jaccard(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


class DeduplicationIndex:
    """Stateful ordered comparison against development and prior generated questions."""

    def __init__(self, development_questions: tuple[DevelopmentQuestion, ...]) -> None:
        self._references: list[_Reference] = [
            _Reference(
                scope="development",
                identifier=question.stable_id,
                exact_hash=normalized_text_sha256(question.question),
                template_hash=numeric_template_sha256(question.question),
                structure_hash=None,
                ngrams=token_ngrams(question.question),
            )
            for question in development_questions
        ]

    @property
    def reference_count(self) -> int:
        return len(self._references)

    def screen(self, draft: CandidateDraft) -> LexicalScreenResult:
        """Run exact, numeric-template, structure, then five-gram checks in order."""

        exact_hash = normalized_text_sha256(draft.rendered_question)
        template_hash = numeric_template_sha256(draft.rendered_question)
        ngrams = token_ngrams(draft.rendered_question)
        for reference in self._references:
            if reference.exact_hash == exact_hash:
                return LexicalScreenResult(
                    "exact_normalized_text", reference.scope, reference.identifier[:12], 1.0
                )
        for reference in self._references:
            if reference.template_hash == template_hash:
                return LexicalScreenResult(
                    "numeric_template_copy", reference.scope, reference.identifier[:12], 1.0
                )
        for reference in self._references:
            if (
                reference.structure_hash is not None
                and reference.structure_hash == draft.structure_sha256
            ):
                return LexicalScreenResult(
                    "latent_structure_copy", reference.scope, reference.identifier[:12], 1.0
                )
        maximum = 0.0
        matched: _Reference | None = None
        for reference in self._references:
            similarity = _jaccard(ngrams, reference.ngrams)
            if similarity > maximum:
                maximum = similarity
                matched = reference
        if maximum >= 0.35 and matched is not None:
            return LexicalScreenResult(
                "token_ngram_overlap", matched.scope, matched.identifier[:12], maximum
            )
        return LexicalScreenResult(
            None,
            None if matched is None else matched.scope,
            None if matched is None else matched.identifier[:12],
            maximum,
        )

    def add_candidate(self, draft: CandidateDraft) -> None:
        """Add every attempted candidate so later attempts compare against it."""

        self._references.append(
            _Reference(
                scope="generated",
                identifier=draft.candidate_id,
                exact_hash=normalized_text_sha256(draft.rendered_question),
                template_hash=numeric_template_sha256(draft.rendered_question),
                structure_hash=draft.structure_sha256,
                ngrams=token_ngrams(draft.rendered_question),
            )
        )
