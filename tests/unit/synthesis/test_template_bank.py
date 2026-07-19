"""Contract and renderer tests for the offline template bank."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from foundry.synthesis.pipeline import _generate, build_attempt_plan, load_smoke_config
from foundry.synthesis.quality import validate_rendered_candidate
from foundry.synthesis.realization import validate_realization
from foundry.synthesis.realization.ir import (
    BookkeepingProblemIR,
    RateProblemIR,
    RateRelationKind,
    TargetKind,
)
from foundry.synthesis.realization.morphology import count_lexeme
from foundry.synthesis.template_bank import TEMPLATE_BANK_VERSION, build_template_bank
from foundry.synthesis.template_bank.composition import (
    NounPhraseSpec,
    audit_surface_provenance,
    numeric_ordinal,
    validate_surface_text,
    word_ordinal,
)
from foundry.synthesis.template_bank.contracts import SentencePlanSpec
from foundry.synthesis.template_bank.expansion import run_static_expansion
from foundry.synthesis.template_bank.finalize import finalize_summary
from foundry.synthesis.template_bank.policy import load_policy
from foundry.synthesis.template_bank.renderer import render_with_template
from foundry.synthesis.template_bank.review_import import (
    EXPECTED_REVIEW_SHA256,
    PLAN_DISPOSITIONS,
)
from foundry.synthesis.template_bank.smoke import (
    _select_template,
    _write_html_review_packet,
    _write_review_packets_if_ready,
)


def test_initial_bank_has_required_capacity_and_review_state() -> None:
    assert TEMPLATE_BANK_VERSION == "foundry-template-bank-v3"
    bank = build_template_bank()
    counts: dict[str, int] = {}
    signatures: set[str] = set()
    for template in bank:
        counts[template.reasoning_category] = counts.get(template.reasoning_category, 0) + 1
        assert template.review_status == "human_review_pending"
        assert template.provenance == "human_review_reauthored_foundry_v2"
        assert template.normalized_template_hash
        assert "_" not in template.surface_lexeme.text
        assert template.surface_lexeme.text != template.semantic_frame.replace("_", " ")
        for plan in template.sentence_plan_variants:
            signatures.add(template.render_signature_hash(plan))
    assert sorted(counts.values()) == [18, 20, 20]
    assert len(bank) == 58
    assert len(signatures) == 232


def test_human_review_manifest_is_content_free_and_complete() -> None:
    manifest = json.loads(
        Path("configs/synthesis/template_bank_human_review_v2.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        Path("results/synthesis_smoke/template_bank_human_review_v2_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["review_export_sha256"] == EXPECTED_REVIEW_SHA256
    assert summary["review_export_sha256"] == EXPECTED_REVIEW_SHA256
    assert summary["attempted"] == 120
    assert summary["approved"] == summary["rejected"] == 60
    assert summary["unsure"] == 0
    assert not summary["full_generation_gate_passed"]
    assert len(manifest["approved_reviewed_plan_keys"]) == 60
    assert len(manifest["quarantined_reviewed_plan_keys"]) == 60
    assert len(PLAN_DISPOSITIONS) == 12
    serialized = json.dumps((manifest, summary)).lower()
    assert "rendered_question" not in serialized
    assert "canonical_answer" not in serialized


def test_reauthored_bank_replaces_all_review_derived_defective_plan_families() -> None:
    old_ids = {item.plan_id for item in PLAN_DISPOSITIONS if item.replacement_plan_id is not None}
    replacement_ids = {
        item.replacement_plan_id
        for item in PLAN_DISPOSITIONS
        if item.replacement_plan_id is not None
    }
    bank_ids = {
        plan.plan_id
        for template in build_template_bank()
        for plan in template.sentence_plan_variants
    }
    assert old_ids.isdisjoint(bank_ids)
    assert replacement_ids <= bank_ids


def test_v3_smoke_blocks_every_recurring_human_rejected_language_pattern() -> None:
    config = load_smoke_config(Path("configs/synthesis/template_bank_smoke_v3.yaml"))
    bank = build_template_bank()
    forbidden = (
        "update 1 has",
        "a transfer of",
        "before the scheduled movements",
        "using every listed movement",
        "when the register closes",
        "panel-weighted average",
        "assigned to the sample",
        "first side",
        "second side",
        "the exact condition is",
        "the task can proceed",
        "completely filled",
        "belongs in the completed order",
        "in total, a total of",
    )
    for attempt in build_attempt_plan(config):
        original = _generate(attempt)
        template, sentence_plan = _select_template(attempt, original, bank)
        rendered = render_with_template(original, template, sentence_plan)
        lowered = rendered.rendered_question.lower()
        assert not any(phrase in lowered for phrase in forbidden)
        if isinstance(rendered.problem_ir, BookkeepingProblemIR):
            ledger = rendered.problem_ir.domain.primary_location.lexeme.singular
            assert ledger in rendered.realization.question_clause
        if (
            isinstance(rendered.problem_ir, RateProblemIR)
            and rendered.problem_ir.relation_kind is RateRelationKind.RATIO_SCALE
        ):
            assert "second collection" in rendered.realization.question_clause


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
        provenance = audit_surface_provenance(draft.problem_ir, draft.realization, template)
        assert not provenance.reasons
        assert provenance.provenance_sha256
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


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (101, "101st"),
    ),
)
def test_numeric_ordinals_are_morphologically_correct(value: int, expected: str) -> None:
    assert numeric_ordinal(value) == expected


def test_ordinal_words_fail_closed_outside_approved_mapping() -> None:
    assert word_ordinal(1) == "first"
    assert word_ordinal(13) == "thirteenth"
    with pytest.raises(ValueError, match="positive"):
        numeric_ordinal(0)
    with pytest.raises(ValueError, match="unsupported"):
        word_ordinal(21)


def test_noun_phrase_has_one_typed_head_and_rejects_repetition() -> None:
    shelf = count_lexeme("shelf", "shelf", "shelves", attributive="shelf")
    box = count_lexeme("box", "box", "boxes", attributive="box")
    assert NounPhraseSpec(head=shelf, quantity=1).render()[0] == "shelf"
    assert NounPhraseSpec(head=shelf, quantity=2).render()[0] == "shelves"
    assert NounPhraseSpec(head=box, quantity=2).render()[0] == "boxes"
    with pytest.raises(ValueError, match="repeat"):
        NounPhraseSpec(head=shelf, quantity=2, grouping_noun=shelf).render()


@pytest.mark.parametrize(
    ("sanitized_surface", "internal_identifier", "expected_reason"),
    (
        ("dispatch record record", "dispatch_record", "adjacent_duplicate_noun"),
        ("receiving record record", "receiving_record", "adjacent_duplicate_noun"),
        ("paired collections collections", "paired_collections", "adjacent_duplicate_noun"),
        ("selected share inventory", "selected_share", "internal_frame_label_leak"),
        ("the 1th group", "weighted_readings", "invalid_ordinal_morphology"),
        ("two resource capacity inventory", "two_resource_capacity", "internal_frame_label_leak"),
        ("materials register register", "materials_register", "adjacent_duplicate_noun"),
        ("equipment register register", "equipment_register", "adjacent_duplicate_noun"),
        ("parallel channels process", "parallel_channels", "internal_frame_label_leak"),
        ("matched batches collections", "matched_batches", "internal_frame_label_leak"),
        ("the 2th group", "grouped_measurements", "invalid_ordinal_morphology"),
        ("paired supply limit inventory", "paired_supply_limit", "internal_frame_label_leak"),
        ("dual recipe plan plan", "dual_recipe_plan", "adjacent_duplicate_noun"),
    ),
)
def test_all_thirteen_milestone_6a_surface_defects_are_blocked(
    sanitized_surface: str, internal_identifier: str, expected_reason: str
) -> None:
    reasons = validate_surface_text(sanitized_surface, (internal_identifier,))
    assert expected_reason in reasons


def test_internal_identifiers_cannot_reach_surface_text() -> None:
    assert "internal_identifier_leak" in validate_surface_text("raw_frame_id appears here")
    assert "internal_frame_label_leak" in validate_surface_text(
        "a raw frame label appears here", ("raw_frame_label",)
    )


def test_full_bank_expansion_exercises_ten_fixtures_per_plan(tmp_path: Path) -> None:
    summary = run_static_expansion(tmp_path)
    assert summary["total_expansions_attempted"] == 2320
    assert summary["valid_renders"] == 2320
    assert summary["failure_counts"] == {}
    assert summary["distinct_render_signatures"] == 232
    assert summary["training_dataset_created"] is False


def test_html_review_packet_is_local_and_exports_user_decisions(tmp_path: Path) -> None:
    packet = tmp_path / "human_review.html"
    _write_html_review_packet(packet, ())
    html = packet.read_text(encoding="utf-8")
    assert "localStorage" in html
    assert '["Approve", "Reject", "Unsure"]' in html
    assert "Export review JSON" in html
    assert "genuine_user_human_review" in html
    assert "foundry-template-bank-smoke-v2-review.json" in html


def test_failed_technical_gate_does_not_create_review_packets(tmp_path: Path) -> None:
    config = load_smoke_config(Path("configs/synthesis/template_bank_smoke_v3.yaml"))
    metadata = _write_review_packets_if_ready(
        repository_root=tmp_path,
        config=config,
        records=(),
        technical_gate=False,
    )
    assert metadata == {
        "human_review_status": "not_created_technical_gate_failed",
        "human_review_packet": None,
        "human_review_html_packet": None,
        "human_review_export_filename": None,
    }
    assert not (tmp_path / config.manual_audit_path).exists()
    assert not (tmp_path / config.raw_directory / "human_review.html").exists()


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
