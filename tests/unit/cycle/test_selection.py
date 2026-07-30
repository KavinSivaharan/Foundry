from __future__ import annotations

import hashlib
from dataclasses import replace

from foundry.cycle.selection import (
    CandidateDecision,
    audit_gsm1k_overlap,
    build_selection_summary,
    select_candidate,
    verify_candidate,
)


def _decision(
    source_id: str,
    family: str,
    completion_index: int,
    *,
    eligible: bool,
    tokens: int = 12,
) -> CandidateDecision:
    raw = hashlib.sha256(f"raw:{source_id}:{completion_index}".encode()).hexdigest()
    normalized = hashlib.sha256(f"normalized:{source_id}:{completion_index}".encode()).hexdigest()
    return CandidateDecision(
        source_id=source_id,
        family=family,
        completion_index=completion_index,
        completion_tokens=tokens,
        raw_sha256=raw,
        normalized_sha256=normalized,
        extractable=eligible,
        exact_answer=eligible,
        exact_format=eligible,
        truncated=False,
        prompt_echo=False,
        question_generation=False,
        malformed=not eligible,
        backend_error=False,
        verifier_disagreement=False,
        eligible=eligible,
    )


def test_exact_answer_and_format_candidate_is_eligible() -> None:
    decision = verify_candidate(
        source_id="example-1",
        family="arithmetic",
        prompt="Compute six times seven.",
        canonical_answer="42",
        completion_index=0,
        completion="Six groups of seven make 42.\nFinal answer: 42",
        completion_tokens=14,
        truncated=False,
        backend_error_type=None,
    )

    assert decision.eligible is True
    assert decision.exact_answer is True
    assert decision.exact_format is True
    assert decision.verifier_disagreement is False


def test_selection_uses_length_then_normalized_and_raw_hash() -> None:
    long = _decision("id", "family", 0, eligible=True, tokens=13)
    short_high = replace(
        _decision("id", "family", 1, eligible=True, tokens=12),
        normalized_sha256="f" * 64,
    )
    short_low = replace(
        _decision("id", "family", 2, eligible=True, tokens=12),
        normalized_sha256="0" * 64,
    )

    assert select_candidate([long, short_high, short_low]) == short_low


def _coverage_fixture(
    selected_per_family: tuple[int, int, int],
) -> tuple[
    list[CandidateDecision],
    set[str],
    dict[str, str],
    dict[str, str],
]:
    families = ("family-a", "family-b", "family-c")
    decisions: list[CandidateDecision] = []
    source_ids: set[str] = set()
    family_by_id: dict[str, str] = {}
    original_hashes: dict[str, str] = {}
    for family_index, family in enumerate(families):
        for item_index in range(60):
            source_id = f"{family}-{item_index:03d}"
            source_ids.add(source_id)
            family_by_id[source_id] = family
            original_hashes[source_id] = "original"
            accepted = item_index < selected_per_family[family_index]
            for completion_index in range(8):
                decisions.append(
                    _decision(
                        source_id,
                        family,
                        completion_index,
                        eligible=accepted and completion_index == 0,
                    )
                )
    return decisions, source_ids, family_by_id, original_hashes


def test_exact_minimum_verifier_coverage_passes() -> None:
    decisions, source_ids, families, originals = _coverage_fixture((30, 30, 30))

    summary = build_selection_summary(
        decisions=decisions,
        expected_source_ids=source_ids,
        family_by_source_id=families,
        original_target_sha256=originals,
    )

    assert summary["selected_prompts"] == 90
    assert summary["changed_selected_traces"] == 90
    assert summary["passed_before_overlap_audit"] is True
    assert all(item["fraction"] == 0.5 for item in summary["family_coverage"].values())


def test_one_family_below_half_rejects_coverage() -> None:
    decisions, source_ids, families, originals = _coverage_fixture((29, 31, 30))

    summary = build_selection_summary(
        decisions=decisions,
        expected_source_ids=source_ids,
        family_by_source_id=families,
        original_target_sha256=originals,
    )

    assert summary["selected_prompts"] == 90
    assert summary["checks_before_overlap_audit"]["minimum_overall_coverage"] is True
    assert summary["checks_before_overlap_audit"]["minimum_each_family_coverage"] is False
    assert summary["passed_before_overlap_audit"] is False


def test_overlap_audit_detects_normalized_and_contiguous_overlap() -> None:
    question = "one two three four five six seven eight nine ten eleven twelve thirteen"
    result = audit_gsm1k_overlap(
        training_prompts={"id": question.upper()},
        selected_traces={},
        gsm1k_questions=[question],
    )

    assert result["normalized_exact_overlap"] == 1
    assert result["contiguous_window_overlap"] > 0
    assert result["passed"] is False
