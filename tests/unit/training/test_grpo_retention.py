from __future__ import annotations

import copy
from typing import Any

import pytest

from foundry.training.base_conditioned_retention import wilson_lower_bound
from foundry.training.config import canonical_sha256
from foundry.training.grpo_retention import (
    ARMS,
    CHECKPOINTS,
    SELECTION_SUBSETS,
    SUBSET_CONTRACTS,
    RetentionCell,
    build_checkpoint_selection,
    build_final_decision,
    validate_assessment,
)


def _assessment(subset: str, adapter: str, *, passed: bool = True) -> dict[str, Any]:
    contract = SUBSET_CONTRACTS[subset]  # type: ignore[index]
    totals = dict(contract.section_totals)
    if passed:
        section_preserved = dict(totals)
        echo = 0
        question_generation = 0
        backend_failures = 0
        family_failures: dict[str, int] = {}
    else:
        section_preserved = dict(totals)
        section_preserved["format"] = 0
        echo = 0
        question_generation = 0
        backend_failures = 0
        family_failures = {}
    preserved = sum(section_preserved.values())
    broken = contract.total - preserved
    gate_checks = {
        "overall_preservation_at_least_90_percent": preserved / contract.total >= 0.90,
        "arithmetic_preservation_at_least_90_percent": (
            section_preserved["arithmetic"] / totals["arithmetic"] >= 0.90
        ),
        "format_preservation_at_least_90_percent": (
            section_preserved["format"] / totals["format"] >= 0.90
        ),
        "instruction_preservation_at_least_90_percent": (
            section_preserved["instruction"] / totals["instruction"] >= 0.90
        ),
        "overall_wilson_lower_bound_at_least_85_percent": (
            wilson_lower_bound(preserved, contract.total) >= 0.85
        ),
        "prompt_echo_at_most_2_percent": echo / contract.total <= 0.02,
        "zero_question_generation": question_generation == 0,
        "zero_backend_failures": backend_failures == 0,
        "instruction_family_adapter_only_failures_at_most_3": True,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "assessment_id": "foundry-base-conditioned-retention-assessment-v1",
        "instrument_id": "foundry-base-conditioned-retention-v1",
        "suite_id": contract.suite_id,
        "suite_sha256": contract.suite_sha256,
        "subset_id": contract.subset_id,
        "subset_sha256": contract.subset_sha256,
        "adapter_sha256": adapter,
        "evaluation_summary_sha256": "e" * 64,
        "raw_packet_sha256": "f" * 64,
        "total": contract.total,
        "preserved": preserved,
        "broken": broken,
        "overall_preservation": preserved / contract.total,
        "overall_wilson_95_lower_bound": wilson_lower_bound(preserved, contract.total),
        "section_preservation": {
            section: {
                "preserved": section_preserved[section],
                "total": totals[section],
                "rate": section_preserved[section] / totals[section],
            }
            for section in ("arithmetic", "format", "instruction")
        },
        "extractable": contract.total,
        "extractability": 1.0,
        "prompt_echo": echo,
        "prompt_echo_rate": echo / contract.total,
        "question_generation": question_generation,
        "malformed_outputs": 0,
        "backend_failures": backend_failures,
        "paired_transitions": {
            "base_pass_adapter_pass": preserved,
            "base_pass_adapter_fail": broken,
        },
        "broken_item_ids": [f"broken-{index}" for index in range(broken)],
        "instruction_family_adapter_only_failures": family_failures,
        "maximum_instruction_family_adapter_only_failures": 0,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def _matrix(
    variant: str,
    *,
    failing: set[tuple[str, int, str]] | None = None,
) -> list[RetentionCell]:
    failures = failing or set()
    result = []
    for arm_index, arm in enumerate(ARMS):
        for checkpoint in CHECKPOINTS:
            adapter = f"{arm_index + checkpoint // 16:x}" * 64
            adapter = adapter[:64]
            for subset in SELECTION_SUBSETS:
                result.append(
                    RetentionCell(
                        variant_id=variant,  # type: ignore[arg-type]
                        arm=arm,
                        checkpoint=checkpoint,
                        subset_name=subset,
                        checkpoint_adapter_sha256=adapter,
                        assessment=_assessment(
                            subset,
                            adapter,
                            passed=(arm, checkpoint, subset) not in failures,
                        ),
                    )
                )
    return result


def test_valid_assessment_recomputes_every_gate() -> None:
    assessment = _assessment("adjudication", "a" * 64)
    result = validate_assessment(
        assessment,
        contract=SUBSET_CONTRACTS["adjudication"],
        expected_adapter_sha256="a" * 64,
    )
    assert result["gate_passed"] is True
    assert result["total"] == 187


def test_assessment_rejects_forged_gate_and_adapter_identity() -> None:
    assessment = _assessment("adjudication", "a" * 64)
    assessment["gate_checks"]["zero_backend_failures"] = False
    assessment["gate_passed"] = False
    assessment["summary_sha256"] = canonical_sha256(
        {key: value for key, value in assessment.items() if key != "summary_sha256"}
    )
    with pytest.raises(ValueError, match="recomputed metrics"):
        validate_assessment(
            assessment,
            contract=SUBSET_CONTRACTS["adjudication"],
            expected_adapter_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="adapter differs"):
        validate_assessment(
            _assessment("adjudication", "a" * 64),
            contract=SUBSET_CONTRACTS["adjudication"],
            expected_adapter_sha256="b" * 64,
        )


def test_selection_chooses_latest_common_g1_checkpoint() -> None:
    cells = _matrix(
        "G1",
        failing={
            ("targeted", 64, "anchor_holdout"),
        },
    )
    decision = build_checkpoint_selection(cells)
    assert decision["selected_variant"] == "G1"
    assert decision["selected_checkpoint"] == 32
    assert decision["g2_training_authorized"] is False
    assert decision["gsm1k_authorized"] is False
    assert decision["summary_sha256"] == canonical_sha256(
        {key: value for key, value in decision.items() if key != "summary_sha256"}
    )


def test_g1_failure_authorizes_g2_and_forbids_running_g2_after_g1_pass() -> None:
    every_step_fails = {
        ("generic_control", checkpoint, "adjudication") for checkpoint in CHECKPOINTS
    }
    g1 = _matrix("G1", failing=every_step_fails)
    decision = build_checkpoint_selection(g1)
    assert decision["selection_passed"] is False
    assert decision["g2_training_authorized"] is True
    selected_g2 = build_checkpoint_selection([*g1, *_matrix("G2")])
    assert selected_g2["selected_variant"] == "G2"
    assert selected_g2["selected_checkpoint"] == 64
    with pytest.raises(ValueError, match="forbidden after G1"):
        build_checkpoint_selection([*_matrix("G1"), *_matrix("G2")])


def test_incomplete_or_wrong_subset_matrix_rejects() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        build_checkpoint_selection(_matrix("G1")[:-1])
    cells = _matrix("G1")
    cells[0] = RetentionCell(
        variant_id="G1",
        arm="generic_control",
        checkpoint=16,
        subset_name="final_holdout",
        checkpoint_adapter_sha256=cells[0].checkpoint_adapter_sha256,
        assessment=_assessment("final_holdout", cells[0].checkpoint_adapter_sha256),
    )
    with pytest.raises(ValueError, match="wrong subset"):
        build_checkpoint_selection(cells)


def test_independent_final_decision_is_one_shot_and_cannot_reselect() -> None:
    selection = build_checkpoint_selection(_matrix("G1"))
    selected = selection["selected_adapter_sha256s"]
    cells = [
        RetentionCell(
            variant_id="G1",
            arm=arm,
            checkpoint=64,
            subset_name="final_holdout",
            checkpoint_adapter_sha256=selected[arm],
            assessment=_assessment("final_holdout", selected[arm]),
        )
        for arm in ARMS
    ]
    decision = build_final_decision(selection=selection, cells=cells)
    assert decision["retention_passed"] is True
    assert decision["gsm1k_authorized"] is True
    assert decision["checkpoint_reselection_allowed"] is False
    assert decision["beta_reselection_allowed"] is False

    failed_cells = copy.deepcopy(cells)
    failed_cells[1] = RetentionCell(
        variant_id="G1",
        arm="targeted",
        checkpoint=64,
        subset_name="final_holdout",
        checkpoint_adapter_sha256=selected["targeted"],
        assessment=_assessment("final_holdout", selected["targeted"], passed=False),
    )
    failed = build_final_decision(selection=selection, cells=failed_cells)
    assert failed["retention_passed"] is False
    assert failed["gsm1k_authorized"] is False
    assert failed["verifier_grpo_route_stopped"] is True


def test_final_decision_rejects_checkpoint_or_selection_tampering() -> None:
    selection = build_checkpoint_selection(_matrix("G1"))
    selected = selection["selected_adapter_sha256s"]
    cells = [
        RetentionCell(
            variant_id="G1",
            arm=arm,
            checkpoint=64,
            subset_name="final_holdout",
            checkpoint_adapter_sha256=selected[arm],
            assessment=_assessment("final_holdout", selected[arm]),
        )
        for arm in ARMS
    ]
    cells[0] = RetentionCell(
        variant_id="G1",
        arm="generic_control",
        checkpoint=32,
        subset_name="final_holdout",
        checkpoint_adapter_sha256=selected["generic_control"],
        assessment=cells[0].assessment,
    )
    with pytest.raises(ValueError, match="differs from selection"):
        build_final_decision(selection=selection, cells=cells)
    selection["selected_checkpoint"] = 32
    selection["summary_sha256"] = canonical_sha256(
        {key: value for key, value in selection.items() if key != "summary_sha256"}
    )
    with pytest.raises(ValueError, match="variant or checkpoint differs"):
        build_final_decision(selection=selection, cells=cells)
