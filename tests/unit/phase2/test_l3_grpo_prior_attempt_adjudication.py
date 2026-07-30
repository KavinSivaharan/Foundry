from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from foundry.phase2.l3_grpo_prior_attempt_adjudication import (
    ADJUDICATION_PATH,
    CASE_1,
    CASE_2,
    CASE_3,
    CASE_REQUIREMENTS,
    PRIOR_FREEZE_PATH,
    audit_historical_manifests,
    build_prior_attempt_adjudication,
    build_prior_attempt_freeze,
    classify_prior_attempt,
)
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]


def _case_evidence(case: str) -> dict[str, bool]:
    names = set(CASE_REQUIREMENTS[CASE_1]).union(CASE_REQUIREMENTS[CASE_2])
    evidence = dict.fromkeys(names, False)
    evidence["terminal_result_scientifically_separable"] = True
    evidence["execution_source_independently_certified"] = True
    evidence["manifest_structure_did_not_affect_numeric_result"] = True
    if case == CASE_1:
        evidence["terminal_result_scientifically_separable"] = False
        evidence["execution_source_independently_certified"] = False
        evidence["terminal_result_depended_on_invalid_binding_state"] = True
        evidence["manifest_structure_did_not_affect_numeric_result"] = False
    for name in CASE_REQUIREMENTS[case]:
        evidence[name] = True
    return evidence


def test_all_three_adjudication_cases_are_exclusive() -> None:
    assert classify_prior_attempt(_case_evidence(CASE_1)) == CASE_1
    assert classify_prior_attempt(_case_evidence(CASE_2)) == CASE_2
    ambiguous = _case_evidence(CASE_2)
    ambiguous["terminal_result_valid_under_unchanged_contract"] = False
    assert classify_prior_attempt(ambiguous) == CASE_3


def test_inconsistent_or_incomplete_adjudication_evidence_fails() -> None:
    incomplete = _case_evidence(CASE_2)
    incomplete.pop("scientific_identities_exact")
    with pytest.raises(ValueError, match="fields differ"):
        classify_prior_attempt(incomplete)

    inconsistent = _case_evidence(CASE_2)
    inconsistent["terminal_result_not_scientifically_separable"] = True
    with pytest.raises(ValueError, match="internally inconsistent"):
        classify_prior_attempt(inconsistent)


def test_historical_manifests_reproduce_revised_contract_violations() -> None:
    result = audit_historical_manifests(ROOT)
    violations = cast(dict[str, bool], result["violations"])
    assert violations == {
        "layer1_self_hash_field": True,
        "layer2_self_hash_field": True,
        "layer2_command_template_fields": True,
        "layer2_argv_hash_fields": True,
        "tracked_active_runtime_manifests": True,
        "primary_repository_import_root": True,
        "revised_non_circular_contract_passed": False,
    }
    assert cast(dict[str, Any], result["layer2"])["ordered_path_count"] == 60


def test_prior_attempt_freeze_is_content_free_and_reconstructs() -> None:
    result = build_prior_attempt_freeze(ROOT)
    supplied = cast(str, result.pop("prior_attempt_freeze_sha256"))
    assert supplied == canonical_sha256(result)
    encoded = json.dumps(result, sort_keys=True)
    for prohibited in (
        '"prompt"',
        '"decoded_completion"',
        '"generated_token_ids":',
        '"reward_components"',
    ):
        assert prohibited not in encoded
    assert result["raw_prompt_or_completion_content_in_record"] is False
    assert cast(dict[str, Any], result["model_execution"])["generated_completions"] == 8


def test_published_evidence_uniquely_supports_scientifically_counted_case() -> None:
    freeze = build_prior_attempt_freeze(ROOT)
    result = build_prior_attempt_adjudication(ROOT, freeze)
    decision = cast(dict[str, Any], result["decision"])
    assert decision["classification"] == CASE_2
    assert decision["scientifically_counted"] is True
    assert decision["non_counted_diagnostic_classification"] is False
    assert decision["new_compatibility_campaign_authorized"] is False
    assert decision["source_binding_correction_authorized"] is False
    assert decision["l3_verifier_grpo_compatibility_line"] == "closed"
    evidence = cast(dict[str, Any], result["evidence"])
    inputs = cast(dict[str, bool], evidence["case_inputs"])
    assert inputs["active_source_binding_structurally_invalid"] is True
    assert inputs["execution_source_independently_certified"] is True
    assert inputs["terminal_result_scientifically_separable"] is True
    assert inputs["manifest_structure_did_not_affect_numeric_result"] is True
    assert result["evidence_sha256"] == canonical_sha256(evidence)
    assert result["decision_sha256"] == canonical_sha256(decision)


def test_published_records_match_deterministic_builders_when_present() -> None:
    freeze_path = ROOT / PRIOR_FREEZE_PATH
    adjudication_path = ROOT / ADJUDICATION_PATH
    if not freeze_path.exists() or not adjudication_path.exists():
        pytest.skip("R4 adjudication records have not been published yet")
    assert json.loads(freeze_path.read_text(encoding="utf-8")) == build_prior_attempt_freeze(ROOT)
    freeze = build_prior_attempt_freeze(ROOT)
    assert json.loads(adjudication_path.read_text(encoding="utf-8")) == (
        build_prior_attempt_adjudication(ROOT, freeze)
    )
