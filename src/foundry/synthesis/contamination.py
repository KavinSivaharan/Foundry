"""Design-only deduplication and contamination-screening primitives."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+/\d+|\d+(?:,\d{3})*(?:\.\d+)?)")
_TOKEN = re.compile(r"[a-z0-9]+|<num>")


class ContaminationOutcome(StrEnum):
    """Pairwise screening result."""

    PASS = "pass"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class ContaminationPolicy:
    """Thresholds that must be frozen before pilot generation."""

    token_ngram_size: int = 5
    reject_ngram_jaccard: float = 0.35
    manual_semantic_similarity: float = 0.75
    reject_semantic_similarity: float = 0.82
    semantic_backend: str = "pinned-local-sentence-encoder-required-before-pilot"


@dataclass(frozen=True)
class ContaminationDecision:
    """Content-free reason for accepting, rejecting, or escalating one pair."""

    outcome: ContaminationOutcome
    reason: str
    ngram_jaccard: float
    semantic_similarity: float | None


FROZEN_CONTAMINATION_POLICY = ContaminationPolicy()


def normalize_text(text: str, *, replace_numbers: bool) -> str:
    """Normalize case, Unicode, punctuation, spacing, and optionally numeric values."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    if replace_numbers:
        normalized = _NUMBER.sub(" <num> ", normalized)
    tokens = _TOKEN.findall(normalized)
    return " ".join(tokens)


def normalized_text_sha256(text: str) -> str:
    """Hash exact normalized surface text."""

    return hashlib.sha256(normalize_text(text, replace_numbers=False).encode("utf-8")).hexdigest()


def numeric_template_sha256(text: str) -> str:
    """Hash wording after replacing all numeric values."""

    return hashlib.sha256(normalize_text(text, replace_numbers=True).encode("utf-8")).hexdigest()


def latent_structure_sha256(structure: object) -> str:
    """Hash a canonical latent-program structure without rendered wording."""

    rendered = json.dumps(structure, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def token_ngram_jaccard(left: str, right: str, *, size: int) -> float:
    """Return deterministic token n-gram Jaccard similarity."""

    if size <= 0:
        raise ValueError("n-gram size must be positive")

    def ngrams(text: str) -> set[tuple[str, ...]]:
        tokens = normalize_text(text, replace_numbers=True).split()
        if len(tokens) < size:
            return {tuple(tokens)} if tokens else set()
        return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}

    left_ngrams = ngrams(left)
    right_ngrams = ngrams(right)
    union = left_ngrams | right_ngrams
    if not union:
        return 1.0
    return len(left_ngrams & right_ngrams) / len(union)


def assess_pair(
    generated_text: str,
    comparison_text: str,
    *,
    generated_structure_sha256: str,
    comparison_structure_sha256: str | None,
    semantic_similarity: float | None,
    policy: ContaminationPolicy = FROZEN_CONTAMINATION_POLICY,
) -> ContaminationDecision:
    """Apply staged exact, numeric, structural, lexical, and semantic gates."""

    similarity = token_ngram_jaccard(
        generated_text,
        comparison_text,
        size=policy.token_ngram_size,
    )
    if normalized_text_sha256(generated_text) == normalized_text_sha256(comparison_text):
        return ContaminationDecision(
            ContaminationOutcome.REJECT, "exact_normalized_text", similarity, semantic_similarity
        )
    if numeric_template_sha256(generated_text) == numeric_template_sha256(comparison_text):
        return ContaminationDecision(
            ContaminationOutcome.REJECT, "numeric_template_copy", similarity, semantic_similarity
        )
    if comparison_structure_sha256 is not None and (
        generated_structure_sha256 == comparison_structure_sha256
    ):
        return ContaminationDecision(
            ContaminationOutcome.REJECT, "latent_structure_copy", similarity, semantic_similarity
        )
    if similarity >= policy.reject_ngram_jaccard:
        return ContaminationDecision(
            ContaminationOutcome.REJECT, "token_ngram_overlap", similarity, semantic_similarity
        )
    if semantic_similarity is None:
        return ContaminationDecision(
            ContaminationOutcome.MANUAL_REVIEW, "semantic_check_not_run", similarity, None
        )
    if semantic_similarity >= policy.reject_semantic_similarity:
        return ContaminationDecision(
            ContaminationOutcome.REJECT, "semantic_similarity", similarity, semantic_similarity
        )
    if semantic_similarity >= policy.manual_semantic_similarity:
        return ContaminationDecision(
            ContaminationOutcome.MANUAL_REVIEW,
            "semantic_similarity_band",
            similarity,
            semantic_similarity,
        )
    return ContaminationDecision(
        ContaminationOutcome.PASS, "all_pairwise_checks_passed", similarity, semantic_similarity
    )
