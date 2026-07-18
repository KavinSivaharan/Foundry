"""Fast deterministic checks for stress construction (without loading embeddings)."""

from __future__ import annotations

from foundry.synthesis.realization.stress import (
    construct_stress_drafts,
    deterministic_manual_sample,
)


def test_stress_construction_is_deterministic_and_bounded() -> None:
    first = construct_stress_drafts(20)
    second = construct_stress_drafts(20)
    assert len(first) == len(second) == 60
    assert [draft.candidate_id for draft in first] == [draft.candidate_id for draft in second]
    assert [draft.semantic_ir_sha256 for draft in first] == [
        draft.semantic_ir_sha256 for draft in second
    ]
    assert [draft.render_signature_sha256 for draft in first] == [
        draft.render_signature_sha256 for draft in second
    ]


def test_manual_sample_selects_twenty_per_family() -> None:
    drafts = construct_stress_drafts(40)
    sample = deterministic_manual_sample(drafts)
    assert len(sample) == 60
    counts: dict[str, int] = {}
    for draft in sample:
        counts[draft.generator_id] = counts.get(draft.generator_id, 0) + 1
    assert sorted(counts.values()) == [20, 20, 20]
