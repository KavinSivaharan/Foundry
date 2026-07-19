"""Original-fixture and policy tests for internal realization diversity."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from foundry.synthesis.contamination import ContaminationOutcome
from foundry.synthesis.realization.diversity import (
    PairEvidence,
    classify_internal_pair,
    fixture_set_sha256,
    load_candidate_policies,
    load_frozen_internal_policy,
    load_internal_diversity_fixtures,
)

FIXTURES = Path("tests/fixtures/synthesis/internal_diversity_v1.json")
CANDIDATES = Path("configs/synthesis/internal_diversity_candidates_v1.yaml")
FROZEN = Path("configs/synthesis/internal_diversity_v1.yaml")
SUMMARY = Path("results/synthesis_smoke/internal_diversity_calibration.json")


def test_fixture_set_is_complete_original_and_frozen() -> None:
    fixtures = load_internal_diversity_fixtures(FIXTURES)
    assert len(fixtures) == 24
    assert fixture_set_sha256(fixtures) == (
        "e5ba09dc45c6afd58c2c6f9435a33756cb0bb5c20ab73d41313bc49e09c17b89"
    )
    assert Counter(fixture.relationship for fixture in fixtures) == {
        "exact_duplicate": 3,
        "number_swapped_copy": 3,
        "structurally_equivalent_rewrite": 3,
        "close_paraphrase": 3,
        "same_skill_distinct_scenario": 4,
        "related_but_distinct": 3,
        "unrelated": 3,
        "ambiguous_similarity": 2,
    }
    assert all(
        fixture.ambiguity_note is not None
        for fixture in fixtures
        if fixture.expected_outcome is ContaminationOutcome.MANUAL_REVIEW
    )


def test_exactly_three_candidates_and_one_frozen_selection() -> None:
    candidates = load_candidate_policies(CANDIDATES)
    selected = load_frozen_internal_policy(FROZEN)
    assert len(candidates) == 3
    assert selected == candidates[1]
    assert selected.policy_id == "evidence-gated-balanced-v1"
    assert selected.sha256 == ("26c030e8497c4727e286ff3e89d4720cee1c2681a224b8a93b8c515ef521cc90")


def test_hard_duplicate_controls_precede_semantic_policy() -> None:
    policy = load_frozen_internal_policy(FROZEN)
    base = PairEvidence(False, False, False, 0.0, 0.1, False, False)
    assert (
        classify_internal_pair(
            policy, PairEvidence(True, False, False, 0.0, 0.1, False, False)
        ).reason
        == "exact_normalized_text"
    )
    assert (
        classify_internal_pair(
            policy, PairEvidence(False, True, False, 0.0, 0.1, False, False)
        ).reason
        == "numeric_template_copy"
    )
    assert (
        classify_internal_pair(
            policy, PairEvidence(False, False, True, 0.0, 0.1, False, False)
        ).reason
        == "latent_structure_copy"
    )
    assert (
        classify_internal_pair(
            policy, PairEvidence(False, False, False, 0.35, 0.1, False, False)
        ).reason
        == "token_ngram_overlap"
    )
    assert classify_internal_pair(policy, base).outcome is ContaminationOutcome.PASS


def test_supported_semantic_similarity_rejects_or_routes_review() -> None:
    policy = load_frozen_internal_policy(FROZEN)
    rejected = classify_internal_pair(
        policy, PairEvidence(False, False, False, 0.0, 0.93, True, True)
    )
    reviewed = classify_internal_pair(
        policy, PairEvidence(False, False, False, 0.12, 0.86, False, False)
    )
    same_skill_distinct = classify_internal_pair(
        policy, PairEvidence(False, False, False, 0.02, 0.88, True, False)
    )
    assert rejected.outcome is ContaminationOutcome.REJECT
    assert reviewed.outcome is ContaminationOutcome.MANUAL_REVIEW
    assert same_skill_distinct.outcome is ContaminationOutcome.PASS


def test_content_free_calibration_summary_matches_frozen_hashes() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    selected = summary["selected_policy"]
    assert summary["fixture_count"] == 24
    assert summary["fixture_set_sha256"] == (
        "e5ba09dc45c6afd58c2c6f9435a33756cb0bb5c20ab73d41313bc49e09c17b89"
    )
    assert summary["calibration_sha256"] == (
        "e855e29a953cbb6b0563e73def3ee4bceb3bbbf20a6f0c5dc7f18573d43063ab"
    )
    assert selected["duplicate_escapes"] == 0
    assert selected["distinct_auto_rejections"] == 0
    assert summary["benchmark_content_used"] is False
    assert summary["qwen_output_used"] is False
