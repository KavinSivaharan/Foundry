"""Design-only deduplication and contamination-screening primitives."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from foundry.config import load_config

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


@dataclass(frozen=True)
class DevelopmentQuestion:
    """Development-only question exposed solely to contamination screening."""

    stable_id: str
    row_index: int
    question: str


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


def _development_manifest_entries(path: Path) -> tuple[tuple[str, int], ...]:
    """Validate the identifier-only 904-row development manifest without sealed access."""

    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load development manifest: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("development manifest must be an object")
    manifest = cast(dict[str, object], raw)
    if manifest.get("partition") != "development":
        raise ValueError("contamination loading requires the development partition")
    if manifest.get("dataset_id") != "ScaleAI/gsm1k" or manifest.get("dataset_revision") != (
        "bc09569d09a614b9b530edc7f076fb214ac10493"
    ):
        raise ValueError("development manifest dataset pin changed")
    entries_raw = manifest.get("entries")
    if not isinstance(entries_raw, list) or len(entries_raw) != 904:
        raise ValueError("development manifest must contain exactly 904 identifiers")
    unsigned = dict(manifest)
    digest = unsigned.pop("manifest_sha256", None)
    recomputed = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != recomputed:
        raise ValueError("development manifest hash is invalid")
    entries: list[tuple[str, int]] = []
    for raw_entry in entries_raw:
        if not isinstance(raw_entry, dict):
            raise ValueError("development manifest entry must be an object")
        entry = cast(dict[str, object], raw_entry)
        stable_id = entry.get("stable_id")
        row_index = entry.get("row_index")
        if (
            not isinstance(stable_id, str)
            or isinstance(row_index, bool)
            or not isinstance(row_index, int)
        ):
            raise ValueError("development manifest entry has invalid fields")
        identity = (
            f"ScaleAI/gsm1k@bc09569d09a614b9b530edc7f076fb214ac10493:default:test:{row_index}"
        )
        if hashlib.sha256(identity.encode("utf-8")).hexdigest() != stable_id:
            raise ValueError("development stable identifier mismatch")
        entries.append((stable_id, row_index))
    if len({row for _, row in entries}) != 904 or len({item for item, _ in entries}) != 904:
        raise ValueError("development manifest identifiers must be unique")
    return tuple(entries)


def load_development_questions_for_contamination(
    *,
    evaluation_config_path: Path,
    development_manifest_path: Path,
) -> tuple[DevelopmentQuestion, ...]:
    """Load only 904 development questions; never return labels or sealed rows."""

    config = load_config(evaluation_config_path)
    if config.dataset.repo_id != "ScaleAI/gsm1k" or config.dataset.revision != (
        "bc09569d09a614b9b530edc7f076fb214ac10493"
    ):
        raise ValueError("evaluation dataset differs from the frozen contamination source")
    entries = _development_manifest_entries(development_manifest_path)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("development contamination loading requires datasets") from error
    dataset: Any = load_dataset(
        config.dataset.repo_id,
        config.dataset.config_name,
        split=config.dataset.source_split,
        revision=config.dataset.revision,
    )
    if len(dataset) != config.dataset.expected_examples:
        raise ValueError("pinned development source length changed")
    selected: Any = dataset.select([row for _, row in entries]).select_columns(["question"])
    questions: list[DevelopmentQuestion] = []
    for (stable_id, row_index), raw_row in zip(entries, selected, strict=True):
        if not isinstance(raw_row, dict):
            raise ValueError("selected development row is not a mapping")
        question = raw_row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("selected development question is invalid")
        questions.append(DevelopmentQuestion(stable_id, row_index, question))
    if len(questions) != 904:
        raise ValueError("contamination loader did not return exactly 904 development questions")
    return tuple(questions)
