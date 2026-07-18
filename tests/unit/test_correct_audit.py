from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from foundry.evaluation.correct_audit import (
    CorrectAuditError,
    FrozenAuditClassification,
    LabelBlindAuditView,
    _classification_jsonl_bytes,
    _context_values,
    _normalize_classifications,
    _suspicious_patterns,
    _trace_extraction,
    audit_configuration_sha256,
)


def _audit_view() -> LabelBlindAuditView:
    return LabelBlindAuditView(
        schema_version=1,
        stable_id="a" * 64,
        response="Therefore, the answer is 42.",
        strict_parser_accepted=False,
        canonical_extractor_accepted=True,
        extraction_rule="explicit_answer_cue",
        extracted_value=42,
        source_span_start=25,
        source_span_end=27,
        source_span_text="42",
        rule_context="Therefore, the answer is 42.",
        competing_terminal_values=False,
        terminal_context_values=(42,),
        output_complete=True,
        output_tokens=8,
        suspicious_patterns=("canonical_only",),
    )


def _working_record() -> dict[str, object]:
    view = _audit_view()
    return {
        "audit_classification": "confirmed_intended_answer",
        "canonical_extractor_accepted": view.canonical_extractor_accepted,
        "competing_terminal_values": view.competing_terminal_values,
        "extracted_value": view.extracted_value,
        "extraction_rule": view.extraction_rule,
        "output_complete": view.output_complete,
        "output_tokens": view.output_tokens,
        "sanitized_rationale": "Explicit answer cue identifies the terminal value.",
        "stable_id": view.stable_id,
        "strict_parser_accepted": view.strict_parser_accepted,
    }


def test_trace_records_exact_numeric_source_span_and_rule() -> None:
    response = "Reasoning with 10 and 20.\nFinal answer: -42"

    rule, start, end, span, context, values = _trace_extraction(response, -42)

    assert rule == "literal_final_answer_line"
    assert response[start:end] == span == "-42"
    assert context == "Final answer: -42"
    assert values == (-42,)


def test_terminal_context_exposes_competing_values_without_labels() -> None:
    context = "Therefore, after a 20% discount, the final cost is **$480**."

    assert _context_values(context) == (20, 480)


def test_suspicion_rules_cover_previous_false_acceptance_shapes() -> None:
    percentage_currency = _suspicious_patterns(
        strict_parser_accepted=False,
        rule_name="conclusion_verb_prose",
        extracted_value=20,
        rule_context="Therefore, after a 20% discount, the final cost is **$480**.",
        context_values=(20, 480),
        output_tokens=300,
    )
    semantic_sign = _suspicious_patterns(
        strict_parser_accepted=False,
        rule_name="conclusion_terminal_prose",
        extracted_value=20,
        rule_context="Therefore, the total loss is $20.",
        context_values=(20,),
        output_tokens=300,
    )

    assert "percentage_currency_mixture" in percentage_currency
    assert "multiple_values_in_terminal_context" in percentage_currency
    assert "negative_intent_without_negative_value" in semantic_sign


def test_audit_configuration_hash_is_stable_sha256() -> None:
    digest = audit_configuration_sha256()

    assert digest == "e50df38364b88d4900dfecc948cd56d0d552e050971ca47a7a21264699ee4122"
    assert len(bytes.fromhex(digest)) == hashlib.sha256().digest_size


def test_trace_rejects_value_not_selected_by_terminal_rule() -> None:
    with pytest.raises(CorrectAuditError, match="could not be traced"):
        _trace_extraction("Therefore, the answer is 42.", 7)


def test_no_audit_fixture_contains_reference_fields(tmp_path: Path) -> None:
    payload = {
        "stable_id": "a" * 64,
        "response": "Final answer: 42",
        "extracted_value": 42,
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert "reference_answer" not in json.loads(path.read_text(encoding="utf-8"))


def test_label_blind_classification_normalizes_to_stable_schema() -> None:
    classifications = _normalize_classifications((_audit_view(),), (_working_record(),))

    assert classifications == (
        FrozenAuditClassification(
            schema_version=1,
            stable_id="a" * 64,
            strict_parser_accepted=False,
            canonical_extractor_accepted=True,
            extraction_rule="explicit_answer_cue",
            extracted_value=42,
            competing_terminal_values=False,
            output_complete=True,
            output_tokens=8,
            audit_classification="confirmed_intended_answer",
            sanitized_rationale="Explicit answer cue identifies the terminal value.",
        ),
    )
    assert _classification_jsonl_bytes(classifications) == _classification_jsonl_bytes(
        classifications
    )


def test_label_blind_classification_rejects_changed_extraction_evidence() -> None:
    working = _working_record()
    working["extracted_value"] = 7

    with pytest.raises(CorrectAuditError, match="changed evidence field extracted_value"):
        _normalize_classifications((_audit_view(),), (working,))
