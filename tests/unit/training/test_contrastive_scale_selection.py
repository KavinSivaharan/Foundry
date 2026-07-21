import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import foundry.training.contrastive_scale_selection as subject
from foundry.training.config import canonical_sha256
from foundry.training.contrastive_scale_selection import (
    build_final_validation,
    build_selection,
)

ADAPTER_SHA256 = "a" * 64
CONFIG_SHA256 = "c" * 64
RAW_SHA256 = "d" * 64
SELECTION_IDENTITIES = (
    {
        "suite_id": "foundry-retention-adjudication-v2",
        "suite_sha256": "5caf23be79fa01151af6f7db8d45c2b85bfe24b03a29589e482d51731c8358af",
        "subset_id": "retention-adjudication-v2-base-correct",
        "subset_sha256": "c76df74b911b96ca43c2663a123e41347fd544bf6644f15522ccaad7b77099e1",
        "evaluation_id": "foundry-retention-adjudication-evaluation-v2",
    },
    {
        "suite_id": "foundry-retention-anchor-holdout-v1",
        "suite_sha256": "bff18b434a284d848387262dde201601278e5c8b573937b3486bed2bf925696e",
        "subset_id": "retention-anchor-holdout-v1-base-correct",
        "subset_sha256": "36be91d08f2ab0e05c491094c53965d1aa4f989a730347768877a2548a62c7a9",
        "evaluation_id": "foundry-retention-anchor-holdout-evaluation-v1",
    },
)
FINAL_IDENTITY = {
    "suite_id": "foundry-retention-scale-final-holdout-v1",
    "suite_sha256": "b856c8ce8e56d98eb7e3fbffdead07ffde7091ab2a20abe5a22ada598136353e",
    "subset_id": "retention-scale-final-holdout-v1-base-correct",
    "subset_sha256": "0884923ce7ab39f1080282dab0ce51aff7063270d6c97f5c1d70370256012ded",
    "evaluation_id": "foundry-retention-scale-final-holdout-evaluation-v1",
}


def _resign(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("summary_sha256", None)
    value["summary_sha256"] = canonical_sha256(value)
    return value


def _construction(*, adapter_sha256: str = ADAPTER_SHA256, passed: bool = True) -> dict[str, Any]:
    return _resign(
        {
            "construction_id": "foundry-targeted-minus-generic-task-vector-construction-v1",
            "protocol_sha256": CONFIG_SHA256,
            "contrastive_adapter_sha256": adapter_sha256,
            "gate_passed": passed,
            "sealed_final_accessed": False,
            "unmerged_and_reversible": True,
            "merged_module_count": 0,
            "base_state_unchanged": True,
            "source_directories_unchanged": True,
            "dense_equivalence": {"gate_passed": True},
            "functional_logit_equivalence": {"gate_passed": True},
            "scale_zero_sanity": {"matches_untouched_base": True},
            "scale_one_sanity": {"matches_unscaled_contrastive": True},
        }
    )


def _scale_evidence(scale: float) -> dict[str, Any]:
    return {
        "implementation_id": "foundry-common-lora-runtime-scaling-v1",
        "scale": scale,
        "original_scaling_restored": True,
        "adapter_state_unchanged": True,
        "base_parameter_signature_unchanged": True,
        "adapter_state_sha256_before": "1" * 64,
        "adapter_state_sha256_after": "1" * 64,
        "base_parameter_signature_before": "2" * 64,
        "base_parameter_signature_after": "2" * 64,
        "original_scaling_sha256": "3" * 64,
        "applied_scaling_sha256": "4" * 64,
    }


def _evaluation(
    *,
    scale: float,
    passed: bool,
    identity: dict[str, str],
    adapter_sha256: str = ADAPTER_SHA256,
) -> dict[str, Any]:
    question_generation = 0 if passed else 1
    return _resign(
        {
            "evaluation_id": identity["evaluation_id"],
            "base_conditioned_instrument_id": "foundry-base-conditioned-retention-v1",
            "suite_sha256": identity["suite_sha256"],
            "base_conditioned_subset_id": identity["subset_id"],
            "base_conditioned_subset_sha256": identity["subset_sha256"],
            "adapter_sha256": adapter_sha256,
            "adapter_scale": scale,
            "adapter_scale_evidence": _scale_evidence(scale),
            "total": 300,
            "section_metrics": {
                section: {"correct": 100, "total": 100, "accuracy": 1.0}
                for section in ("arithmetic", "format", "instruction")
            },
            "extractable": 300,
            "extractability": 1.0,
            "exact_format": 300,
            "exact_format_rate": 1.0,
            "prompt_echo": 0,
            "prompt_echo_rate": 0.0,
            "question_generation": question_generation,
            "malformed_outputs": 0,
            "backend_failures": 0,
            "raw_packet_sha256": RAW_SHA256,
        }
    )


def _assessment_from_evaluation(
    evaluation: dict[str, Any], *, identity: dict[str, str]
) -> dict[str, Any]:
    question_generation = int(evaluation["question_generation"])
    gate_checks = {
        "overall_preservation_at_least_90_percent": True,
        "arithmetic_preservation_at_least_90_percent": True,
        "format_preservation_at_least_90_percent": True,
        "instruction_preservation_at_least_90_percent": True,
        "overall_wilson_lower_bound_at_least_85_percent": True,
        "prompt_echo_at_most_2_percent": True,
        "zero_question_generation": question_generation == 0,
        "zero_backend_failures": True,
        "instruction_family_adapter_only_failures_at_most_3": True,
    }
    evidence = evaluation["adapter_scale_evidence"]
    assert isinstance(evidence, dict)
    return _resign(
        {
            "assessment_id": "foundry-base-conditioned-retention-assessment-v1",
            "instrument_id": "foundry-base-conditioned-retention-v1",
            "scaling_implementation_id": "foundry-common-lora-runtime-scaling-v1",
            "adapter_sha256": evaluation["adapter_sha256"],
            "adapter_scale": evaluation["adapter_scale"],
            "adapter_scale_evidence_sha256": canonical_sha256(evidence),
            "state_restoration_verified": True,
            "evaluation_summary_sha256": evaluation["summary_sha256"],
            "raw_packet_sha256": evaluation["raw_packet_sha256"],
            "suite_id": identity["suite_id"],
            "suite_sha256": identity["suite_sha256"],
            "subset_id": identity["subset_id"],
            "subset_sha256": identity["subset_sha256"],
            "total": 300,
            "preserved": 300,
            "broken": 0,
            "broken_item_ids": [],
            "overall_preservation": 1.0,
            "overall_wilson_95_lower_bound": subject._wilson_lower_bound(300, 300),
            "section_preservation": {
                section: {"preserved": 100, "total": 100, "rate": 1.0}
                for section in ("arithmetic", "format", "instruction")
            },
            "extractable": 300,
            "extractability": 1.0,
            "prompt_echo": 0,
            "prompt_echo_rate": 0.0,
            "question_generation": question_generation,
            "malformed_outputs": 0,
            "backend_failures": 0,
            "paired_transitions": {
                "base_pass_adapter_pass": 300,
                "base_pass_adapter_fail": 0,
            },
            "instruction_family_adapter_only_failures": {},
            "maximum_instruction_family_adapter_only_failures": 0,
            "gate_checks": gate_checks,
            "gate_passed": all(gate_checks.values()),
        }
    )


def _matrix(
    scale: float,
    passed: bool,
    *,
    adapter_sha256: str = ADAPTER_SHA256,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluations = [
        _evaluation(
            scale=scale,
            passed=passed,
            identity=identity,
            adapter_sha256=adapter_sha256,
        )
        for identity in SELECTION_IDENTITIES
    ]
    assessments = [
        _assessment_from_evaluation(evaluation, identity=identity)
        for evaluation, identity in zip(evaluations, SELECTION_IDENTITIES, strict=True)
    ]
    return assessments, evaluations


def _build(
    cells: list[tuple[float, bool]],
    *,
    construction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = [(scale, *_matrix(scale, passed)) for scale, passed in cells]
    return build_selection(
        construction=_construction() if construction is None else construction,
        matrices=[(scale, assessments) for scale, assessments, _ in evidence],
        evaluation_matrices=[(scale, evaluations) for scale, _, evaluations in evidence],
        scale_config_sha256=CONFIG_SHA256,
    )


def _selection(scale: float = 0.75) -> dict[str, Any]:
    cells = [(1.0, False)]
    if scale == 1.0:
        cells = [(1.0, True)]
    else:
        cells.append((scale, True))
    return _build(cells)


def _final_evidence(
    *, scale: float = 0.75, passed: bool = True, adapter_sha256: str = ADAPTER_SHA256
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = _evaluation(
        scale=scale,
        passed=passed,
        identity=FINAL_IDENTITY,
        adapter_sha256=adapter_sha256,
    )
    return _assessment_from_evaluation(evaluation, identity=FINAL_IDENTITY), evaluation


def test_selection_uses_first_largest_pass_and_binds_construction() -> None:
    construction = _construction()
    decision = _build([(1.0, False), (0.75, True)], construction=construction)

    assert decision["construction_summary_sha256"] == construction["summary_sha256"]
    assert decision["adapter_sha256"] == ADAPTER_SHA256
    assert decision["selected_contrastive_scale"] == 0.75
    assert decision["selection_used_gsm1k"] is False
    assert decision["gsm1k_authorized"] is False
    assert decision["sealed_final_accessed"] is False


def test_selection_rejects_failed_or_wrong_adapter_construction() -> None:
    with pytest.raises(ValueError, match="construction did not pass"):
        _build([(1.0, True)], construction=_construction(passed=False))
    with pytest.raises(ValueError, match="adapter hash differs from construction"):
        _build([(1.0, True)], construction=_construction(adapter_sha256="b" * 64))


def test_selection_requires_explicit_scale_and_restoration() -> None:
    assessments, evaluations = _matrix(1.0, True)
    evaluations[0]["adapter_scale"] = 0.75
    _resign(evaluations[0])
    assessments[0]["evaluation_summary_sha256"] = evaluations[0]["summary_sha256"]
    _resign(assessments[0])
    with pytest.raises(ValueError, match="exact explicit adapter scale"):
        build_selection(
            construction=_construction(),
            matrices=[(1.0, assessments)],
            evaluation_matrices=[(1.0, evaluations)],
            scale_config_sha256=CONFIG_SHA256,
        )

    assessments, evaluations = _matrix(1.0, True)
    scale_evidence = evaluations[0]["adapter_scale_evidence"]
    assert isinstance(scale_evidence, dict)
    scale_evidence["original_scaling_restored"] = False
    _resign(evaluations[0])
    assessments[0]["evaluation_summary_sha256"] = evaluations[0]["summary_sha256"]
    assessments[0]["adapter_scale_evidence_sha256"] = canonical_sha256(scale_evidence)
    _resign(assessments[0])
    with pytest.raises(ValueError, match="did not restore"):
        build_selection(
            construction=_construction(),
            matrices=[(1.0, assessments)],
            evaluation_matrices=[(1.0, evaluations)],
            scale_config_sha256=CONFIG_SHA256,
        )


def test_selection_rejects_evaluation_linkage_and_count_tampering() -> None:
    assessments, evaluations = _matrix(1.0, True)
    assessments[0]["evaluation_summary_sha256"] = "f" * 64
    _resign(assessments[0])
    with pytest.raises(ValueError, match="evaluation-summary linkage"):
        build_selection(
            construction=_construction(),
            matrices=[(1.0, assessments)],
            evaluation_matrices=[(1.0, evaluations)],
            scale_config_sha256=CONFIG_SHA256,
        )

    assessments, evaluations = _matrix(1.0, True)
    assessments[0]["preserved"] = 299
    _resign(assessments[0])
    with pytest.raises(ValueError, match="preserved count differs"):
        build_selection(
            construction=_construction(),
            matrices=[(1.0, assessments)],
            evaluation_matrices=[(1.0, evaluations)],
            scale_config_sha256=CONFIG_SHA256,
        )


def test_selection_recomputes_exact_gate_keys_and_decisions() -> None:
    assessments, evaluations = _matrix(1.0, False)
    gate_checks = assessments[0]["gate_checks"]
    assert isinstance(gate_checks, dict)
    gate_checks["invented_check"] = True
    _resign(assessments[0])
    with pytest.raises(ValueError, match="key set differs"):
        build_selection(
            construction=_construction(),
            matrices=[(1.0, assessments)],
            evaluation_matrices=[(1.0, evaluations)],
            scale_config_sha256=CONFIG_SHA256,
        )

    assessments, evaluations = _matrix(1.0, False)
    assessments[0]["gate_passed"] = True
    _resign(assessments[0])
    with pytest.raises(ValueError, match="aggregate gate decision differs"):
        build_selection(
            construction=_construction(),
            matrices=[(1.0, assessments)],
            evaluation_matrices=[(1.0, evaluations)],
            scale_config_sha256=CONFIG_SHA256,
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("assessment", "assessment_id", "wrong", "assessment ID differs"),
        ("assessment", "instrument_id", "wrong", "assessment instrument ID differs"),
        (
            "assessment",
            "scaling_implementation_id",
            "wrong",
            "assessment scaling implementation differs",
        ),
        (
            "evaluation",
            "base_conditioned_instrument_id",
            "wrong",
            "evaluation instrument ID differs",
        ),
    ],
)
def test_selection_requires_exact_evidence_ids(
    target: str, field: str, value: object, message: str
) -> None:
    assessments, evaluations = _matrix(1.0, True)
    evidence = assessments[0] if target == "assessment" else evaluations[0]
    evidence[field] = value
    _resign(evidence)
    if target == "evaluation":
        assessments[0]["evaluation_summary_sha256"] = evidence["summary_sha256"]
        _resign(assessments[0])
    with pytest.raises(ValueError, match=message):
        build_selection(
            construction=_construction(),
            matrices=[(1.0, assessments)],
            evaluation_matrices=[(1.0, evaluations)],
            scale_config_sha256=CONFIG_SHA256,
        )


def test_selection_is_independent_of_cell_input_order() -> None:
    assessments, evaluations = _matrix(1.0, True)
    normal = build_selection(
        construction=_construction(),
        matrices=[(1.0, assessments)],
        evaluation_matrices=[(1.0, evaluations)],
        scale_config_sha256=CONFIG_SHA256,
    )
    reversed_input = build_selection(
        construction=_construction(),
        matrices=[(1.0, list(reversed(assessments)))],
        evaluation_matrices=[(1.0, list(reversed(evaluations)))],
        scale_config_sha256=CONFIG_SHA256,
    )
    independently_reversed = build_selection(
        construction=_construction(),
        matrices=[(1.0, list(reversed(assessments)))],
        evaluation_matrices=[(1.0, evaluations)],
        scale_config_sha256=CONFIG_SHA256,
    )

    assert reversed_input == normal
    assert independently_reversed == normal
    assert reversed_input["summary_sha256"] == normal["summary_sha256"]


def test_selection_rejects_partial_search_and_post_pass_scales() -> None:
    with pytest.raises(ValueError, match="failed partial search"):
        _build([(1.0, False), (0.75, False)])
    with pytest.raises(ValueError, match="after a contrastive scale passed"):
        _build([(1.0, False), (0.75, True), (0.5, True)])

    decision = _build([(scale, False) for scale in subject.SCALE_ORDER])
    assert decision["selected_contrastive_scale"] is None
    assert decision["contrastive_adapter_route_stopped"] is True


def test_final_validation_approves_matching_independent_evidence() -> None:
    construction = _construction()
    assessment, evaluation = _final_evidence()
    result = build_final_validation(
        construction=construction,
        selection=_selection(),
        assessment=assessment,
        evaluation_summary=evaluation,
    )

    assert result["construction_summary_sha256"] == construction["summary_sha256"]
    assert result["evaluation_summary_sha256"] == evaluation["summary_sha256"]
    assert result["retention_decision"] == ("retention_approved_targeted_minus_generic_task_vector")
    assert result["gsm1k_authorized"] is True
    assert result["sealed_final_accessed"] is False


def test_final_validation_rejects_construction_or_evaluation_drift() -> None:
    assessment, evaluation = _final_evidence()
    selection = _selection()
    selection["construction_summary_sha256"] = "f" * 64
    _resign(selection)
    with pytest.raises(ValueError, match="construction evidence differs"):
        build_final_validation(
            construction=_construction(),
            selection=selection,
            assessment=assessment,
            evaluation_summary=evaluation,
        )

    assessment, evaluation = _final_evidence(adapter_sha256="b" * 64)
    with pytest.raises(ValueError, match="adapter hash differs from construction"):
        build_final_validation(
            construction=_construction(),
            selection=_selection(),
            assessment=assessment,
            evaluation_summary=evaluation,
        )


def test_final_validation_failure_closes_route_without_gsm1k() -> None:
    assessment, evaluation = _final_evidence(passed=False)
    result = build_final_validation(
        construction=_construction(),
        selection=_selection(),
        assessment=assessment,
        evaluation_summary=evaluation,
    )

    assert result["retention_decision"] == "failed_independent_contrastive_scale_validation"
    assert result["retention_passed"] is False
    assert result["gsm1k_authorized"] is False
    assert result["contrastive_adapter_route_stopped"] is True


def test_config_loader_requires_canonical_hash_and_frozen_controls(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scale_selection": {
            "scales_descending": list(subject.SCALE_ORDER),
            "zero_sanity_control": 0.0,
            "select_first_passing": True,
            "selection_subset_sha256s": [
                identity["subset_sha256"] for identity in SELECTION_IDENTITIES
            ],
            "independent_final_subset_sha256": FINAL_IDENTITY["subset_sha256"],
        },
        "sealed_final_allowed": False,
    }
    config = {**payload, "config_sha256": canonical_sha256(payload)}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    assert subject._load_config(path)["config_sha256"] == config["config_sha256"]

    config["sealed_final_allowed"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration hash differs"):
        subject._load_config(path)


def test_tampered_selection_hash_fails_before_final_validation() -> None:
    selection = deepcopy(_selection())
    selection["adapter_sha256"] = "b" * 64
    assessment, evaluation = _final_evidence()

    with pytest.raises(ValueError, match="summary hash differs"):
        build_final_validation(
            construction=_construction(),
            selection=selection,
            assessment=assessment,
            evaluation_summary=evaluation,
        )
