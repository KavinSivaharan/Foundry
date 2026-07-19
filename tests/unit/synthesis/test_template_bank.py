"""Contract and renderer tests for the offline template bank."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from foundry.synthesis.pipeline import _generate, build_attempt_plan, load_smoke_config
from foundry.synthesis.quality import validate_rendered_candidate
from foundry.synthesis.realization import validate_realization
from foundry.synthesis.realization.ir import TargetKind
from foundry.synthesis.template_bank import build_template_bank
from foundry.synthesis.template_bank.contracts import SentencePlanSpec
from foundry.synthesis.template_bank.finalize import finalize_summary
from foundry.synthesis.template_bank.policy import load_policy
from foundry.synthesis.template_bank.renderer import render_with_template
from foundry.synthesis.template_bank.smoke import _select_template


def test_initial_bank_has_required_capacity_and_review_state() -> None:
    bank = build_template_bank()
    counts: dict[str, int] = {}
    signatures: set[str] = set()
    for template in bank:
        counts[template.reasoning_category] = counts.get(template.reasoning_category, 0) + 1
        assert template.review_status == "human_review_pending"
        assert template.normalized_template_hash
        for plan in template.sentence_plan_variants:
            signatures.add(template.render_signature_hash(plan))
    assert sorted(counts.values()) == [18, 20, 20]
    assert len(bank) == 58
    assert len(signatures) == 232


def test_bank_fails_closed_on_untyped_placeholder() -> None:
    template = build_template_bank()[0]
    plan = template.sentence_plan_variants[0]
    damaged = replace(plan, opening_form=plan.opening_form + " {free_text}")
    with pytest.raises(ValueError, match="untyped placeholder"):
        replace(template, sentence_plan_variants=(damaged, *template.sentence_plan_variants[1:]))


def test_bank_fails_closed_on_target_mismatch() -> None:
    config = load_smoke_config(Path("configs/synthesis/template_bank_smoke.yaml"))
    plan = build_attempt_plan(config)[0]
    draft = _generate(plan)
    template, sentence_plan = _select_template(plan, draft, build_template_bank())
    damaged = replace(template, compatible_target_types=(TargetKind.WEIGHTED_MEAN,))
    with pytest.raises(ValueError, match="target type"):
        render_with_template(draft, damaged, sentence_plan)


def test_all_120_plans_render_uniquely_and_pass_deterministic_language_rules() -> None:
    config = load_smoke_config(Path("configs/synthesis/template_bank_smoke.yaml"))
    bank = build_template_bank()
    signatures: set[str] = set()
    latent_ids: set[str] = set()
    for attempt in build_attempt_plan(config):
        original = _generate(attempt)
        template, sentence_plan = _select_template(attempt, original, bank)
        draft = render_with_template(original, template, sentence_plan)
        assert not validate_realization(
            problem=draft.problem_ir,
            realization=draft.realization,
            answer=draft.canonical_final_answer,
        )
        assert not validate_rendered_candidate(
            question=draft.rendered_question,
            completion=draft.training_completion,
            answer=draft.canonical_final_answer,
            output_contract_enabled=draft.output_contract_enabled,
            metadata=draft.quality_metadata,
        )
        signatures.add(template.render_signature_hash(sentence_plan))
        latent_ids.add(draft.candidate_id)
    assert len(signatures) == 120
    assert len(latent_ids) == 120


def test_internal_policy_is_frozen_before_smoke() -> None:
    policy = load_policy(Path("configs/synthesis/template_bank_internal_diversity.yaml"))
    assert policy.review_ngram_at == 0.35
    assert policy.review_semantic_at == 0.82
    assert policy.benchmark_policy_unchanged
    assert not policy.topical_similarity_auto_reject
    assert len(policy.fixture_set_sha256) == len(policy.policy_sha256) == 64


def test_sentence_plan_requires_meaningful_clause_order() -> None:
    with pytest.raises(ValueError, match="multi-clause"):
        SentencePlanSpec("bad.plan", ("only",), "opening", "event", "question", "now", "active")


def test_codex_inspection_can_only_fail_the_automatic_gate(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    inspection = tmp_path / "inspection.json"
    summary.write_text('{"attempted":120,"technical_gate_passed":true}', encoding="utf-8")
    inspection.write_text(
        '{"attempts_inspected":120,"inspection_kind":"codex_non_human_surface_inspection",'
        '"invalid_or_unnatural_count":1,"systematic_template_defect":true,'
        '"human_review_status":"pending_user_review",'
        '"findings":[{"attempt_index":7,"defect":"invalid_ordinal_inflection"}]}',
        encoding="utf-8",
    )
    finalized = finalize_summary(summary, inspection)
    assert finalized["technical_gate_passed"] is False
    assert finalized["technical_status"] == "TECHNICAL GATE FAILED"
