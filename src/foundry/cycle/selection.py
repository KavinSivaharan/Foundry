"""Deterministic verifier filtering, coverage, and trace selection."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from foundry.cycle.contract import normalized_completion, text_sha256
from foundry.evaluation.answer_extraction import (
    CanonicalExtractionError,
    extract_canonical_number,
)
from foundry.training.config import canonical_sha256
from foundry.training.retention import RetentionItem, score_response


@dataclass(frozen=True)
class CandidateDecision:
    """Content-free verifier decision for one generated completion."""

    source_id: str
    family: str
    completion_index: int
    completion_tokens: int
    raw_sha256: str
    normalized_sha256: str
    extractable: bool
    exact_answer: bool
    exact_format: bool
    truncated: bool
    prompt_echo: bool
    question_generation: bool
    malformed: bool
    backend_error: bool
    verifier_disagreement: bool
    eligible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "family": self.family,
            "completion_index": self.completion_index,
            "completion_tokens": self.completion_tokens,
            "raw_sha256": self.raw_sha256,
            "normalized_sha256": self.normalized_sha256,
            "extractable": self.extractable,
            "exact_answer": self.exact_answer,
            "exact_format": self.exact_format,
            "truncated": self.truncated,
            "prompt_echo": self.prompt_echo,
            "question_generation": self.question_generation,
            "malformed": self.malformed,
            "backend_error": self.backend_error,
            "verifier_disagreement": self.verifier_disagreement,
            "eligible": self.eligible,
        }


def verify_candidate(
    *,
    source_id: str,
    family: str,
    prompt: str,
    canonical_answer: str,
    completion_index: int,
    completion: str,
    completion_tokens: int,
    truncated: bool,
    backend_error_type: str | None,
) -> CandidateDecision:
    """Apply the existing extractor/scorer and an independent exact-answer check."""

    raw_sha256 = text_sha256(completion)
    normalized_sha256 = text_sha256(normalized_completion(completion))
    item = RetentionItem(
        item_id=source_id,
        section="arithmetic",
        skill=family,
        kind="numeric_terminal",
        prompt=prompt,
        expected=canonical_answer,
    )
    scored = score_response(item, completion)
    independent_correct = False
    independent_extractable = False
    try:
        extracted = extract_canonical_number(completion)
        independent_extractable = True
        independent_correct = Fraction(extracted) == Fraction(canonical_answer)
    except (CanonicalExtractionError, ValueError, ZeroDivisionError):
        pass
    scorer_correct = bool(scored["correct"])
    disagreement = (
        independent_extractable != bool(scored["extractable"])
        or independent_correct != scorer_correct
    )
    backend_error = backend_error_type is not None
    eligible = bool(
        independent_extractable
        and independent_correct
        and scorer_correct
        and bool(scored["exact_format"])
        and not truncated
        and not bool(scored["prompt_echo"])
        and not bool(scored["question_generation"])
        and not bool(scored["malformed"])
        and not backend_error
        and not disagreement
        and 1 <= completion_tokens <= 256
    )
    return CandidateDecision(
        source_id=source_id,
        family=family,
        completion_index=completion_index,
        completion_tokens=completion_tokens,
        raw_sha256=raw_sha256,
        normalized_sha256=normalized_sha256,
        extractable=independent_extractable,
        exact_answer=independent_correct and scorer_correct,
        exact_format=bool(scored["exact_format"]),
        truncated=truncated,
        prompt_echo=bool(scored["prompt_echo"]),
        question_generation=bool(scored["question_generation"]),
        malformed=bool(scored["malformed"]),
        backend_error=backend_error,
        verifier_disagreement=disagreement,
        eligible=eligible,
    )


def select_candidate(decisions: list[CandidateDecision]) -> CandidateDecision | None:
    """Select at most one eligible trace using the frozen hierarchy."""

    eligible = [item for item in decisions if item.eligible]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item.completion_tokens,
            item.normalized_sha256,
            item.raw_sha256,
            item.completion_index,
        ),
    )


def build_selection_summary(
    *,
    decisions: list[CandidateDecision],
    expected_source_ids: set[str],
    family_by_source_id: dict[str, str],
    original_target_sha256: dict[str, str],
) -> dict[str, Any]:
    """Build the complete content-free verifier coverage decision."""

    grouped: defaultdict[str, list[CandidateDecision]] = defaultdict(list)
    for decision in decisions:
        grouped[decision.source_id].append(decision)
    selected = {
        source_id: selected
        for source_id, values in grouped.items()
        if (selected := select_candidate(values)) is not None
    }
    expected_family = Counter(family_by_source_id.values())
    selected_family = Counter(item.family for item in selected.values())
    family_coverage = {
        family: {
            "accepted": selected_family[family],
            "total": total,
            "fraction": selected_family[family] / total,
            "passed": selected_family[family] / total >= 0.5,
        }
        for family, total in sorted(expected_family.items())
    }
    all_processed_once = set(grouped) == expected_source_ids
    attempts_exact = all(
        sorted(item.completion_index for item in grouped[source_id]) == list(range(8))
        for source_id in expected_source_ids
    )
    changed = sum(
        item.raw_sha256 != original_target_sha256[source_id] for source_id, item in selected.items()
    )
    checks = {
        "all_prompts_processed_once": all_processed_once,
        "exactly_eight_attempts_per_prompt": attempts_exact,
        "zero_backend_failures": not any(item.backend_error for item in decisions),
        "zero_verifier_disagreements": not any(item.verifier_disagreement for item in decisions),
        "minimum_overall_coverage": len(selected) / len(expected_source_ids) >= 0.5,
        "minimum_each_family_coverage": all(
            bool(value["passed"]) for value in family_coverage.values()
        ),
        "minimum_changed_selected_traces": changed >= 60,
        "zero_selected_prompt_echo": not any(item.prompt_echo for item in selected.values()),
        "zero_selected_question_generation": not any(
            item.question_generation for item in selected.values()
        ),
        "zero_selected_malformed": not any(item.malformed for item in selected.values()),
        "zero_selected_answer_disagreements": all(item.exact_answer for item in selected.values()),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "selection_id": "foundry-cycle1-verifier-selection-v1",
        "expected_prompts": len(expected_source_ids),
        "attempted_completions": len(decisions),
        "selected_prompts": len(selected),
        "fallback_prompts": len(expected_source_ids) - len(selected),
        "changed_selected_traces": changed,
        "family_coverage": family_coverage,
        "checks_before_overlap_audit": checks,
        "passed_before_overlap_audit": all(checks.values()),
        "selected": [
            {
                "source_id": source_id,
                "family": item.family,
                "completion_index": item.completion_index,
                "completion_tokens": item.completion_tokens,
                "raw_sha256": item.raw_sha256,
                "normalized_sha256": item.normalized_sha256,
            }
            for source_id, item in sorted(selected.items())
        ],
    }
    summary["selection_sha256"] = canonical_sha256(summary)
    return summary


def contiguous_windows(value: str, size: int) -> set[str]:
    tokens = normalized_completion(value).casefold().split()
    return {
        hashlib.sha256(" ".join(tokens[index : index + size]).encode("utf-8")).hexdigest()
        for index in range(max(0, len(tokens) - size + 1))
    }


def audit_gsm1k_overlap(
    *,
    training_prompts: dict[str, str],
    selected_traces: dict[str, str],
    gsm1k_questions: list[str],
    window_tokens: int = 12,
) -> dict[str, Any]:
    """Fail closed on exact, normalized, or contiguous-window GSM1K overlap."""

    gsm_raw = set(gsm1k_questions)
    gsm_normalized = {normalized_completion(item).casefold() for item in gsm1k_questions}
    gsm_windows: set[str] = set()
    for question in gsm1k_questions:
        gsm_windows.update(contiguous_windows(question, window_tokens))
    exact = 0
    normalized = 0
    windows = 0
    audited = {f"prompt:{key}": value for key, value in training_prompts.items()}
    audited.update({f"trace:{key}": value for key, value in selected_traces.items()})
    for value in audited.values():
        exact += int(value in gsm_raw)
        normalized += int(normalized_completion(value).casefold() in gsm_normalized)
        windows += len(contiguous_windows(value, window_tokens).intersection(gsm_windows))
    result: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-cycle1-gsm1k-overlap-v1",
        "training_prompts": len(training_prompts),
        "selected_traces": len(selected_traces),
        "gsm1k_questions": len(gsm1k_questions),
        "window_tokens": window_tokens,
        "exact_overlap": exact,
        "normalized_exact_overlap": normalized,
        "contiguous_window_overlap": windows,
        "passed": exact == 0 and normalized == 0 and windows == 0,
    }
    result["audit_sha256"] = canonical_sha256(result)
    return result


def selection_index(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["source_id"]): item for item in cast(list[dict[str, Any]], summary["selected"])
    }
