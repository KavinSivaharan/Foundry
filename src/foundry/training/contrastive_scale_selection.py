"""Freeze contrastive LoRA-scale selection and independent validation decisions."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256

SCALE_ORDER = (1.0, 0.75, 0.5, 0.25)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ASSESSMENT_ID = "foundry-base-conditioned-retention-assessment-v1"
_INSTRUMENT_ID = "foundry-base-conditioned-retention-v1"
_SCALING_IMPLEMENTATION_ID = "foundry-common-lora-runtime-scaling-v1"
_CONSTRUCTION_ID = "foundry-targeted-minus-generic-task-vector-construction-v1"
_SECTION_ORDER = ("arithmetic", "format", "instruction")
_GATE_CHECK_KEYS = frozenset(
    {
        "overall_preservation_at_least_90_percent",
        "arithmetic_preservation_at_least_90_percent",
        "format_preservation_at_least_90_percent",
        "instruction_preservation_at_least_90_percent",
        "overall_wilson_lower_bound_at_least_85_percent",
        "prompt_echo_at_most_2_percent",
        "zero_question_generation",
        "zero_backend_failures",
        "instruction_family_adapter_only_failures_at_most_3",
    }
)


@dataclass(frozen=True)
class _SuiteIdentity:
    suite_id: str
    suite_sha256: str
    subset_id: str
    subset_sha256: str
    evaluation_id: str

    def as_dict(self) -> dict[str, str]:
        """Return the content-free identity recorded in a decision artifact."""

        return {
            "suite_id": self.suite_id,
            "suite_sha256": self.suite_sha256,
            "subset_id": self.subset_id,
            "subset_sha256": self.subset_sha256,
        }


_SELECTION_SUITES = (
    _SuiteIdentity(
        suite_id="foundry-retention-adjudication-v2",
        suite_sha256="5caf23be79fa01151af6f7db8d45c2b85bfe24b03a29589e482d51731c8358af",
        subset_id="retention-adjudication-v2-base-correct",
        subset_sha256="c76df74b911b96ca43c2663a123e41347fd544bf6644f15522ccaad7b77099e1",
        evaluation_id="foundry-retention-adjudication-evaluation-v2",
    ),
    _SuiteIdentity(
        suite_id="foundry-retention-anchor-holdout-v1",
        suite_sha256="bff18b434a284d848387262dde201601278e5c8b573937b3486bed2bf925696e",
        subset_id="retention-anchor-holdout-v1-base-correct",
        subset_sha256="36be91d08f2ab0e05c491094c53965d1aa4f989a730347768877a2548a62c7a9",
        evaluation_id="foundry-retention-anchor-holdout-evaluation-v1",
    ),
)
_FINAL_SUITE = _SuiteIdentity(
    suite_id="foundry-retention-scale-final-holdout-v1",
    suite_sha256="b856c8ce8e56d98eb7e3fbffdead07ffde7091ab2a20abe5a22ada598136353e",
    subset_id="retention-scale-final-holdout-v1-base-correct",
    subset_sha256="0884923ce7ab39f1080282dab0ce51aff7063270d6c97f5c1d70370256012ded",
    evaluation_id="foundry-retention-scale-final-holdout-evaluation-v1",
)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


def _require_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _require_dict(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _same_rate(actual: object, expected: float, field: str) -> None:
    if not math.isclose(_require_float(actual, field), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field} differs from its counts")


def _validate_summary_hash(summary: dict[str, Any], *, label: str) -> None:
    expected = _require_sha256(summary.get("summary_sha256"), f"{label} summary_sha256")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if expected != canonical_sha256(payload):
        raise ValueError(f"{label} summary hash differs")


def _load(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    result = cast(dict[str, Any], value)
    _validate_summary_hash(result, label=str(path))
    return result


def _load_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    result = cast(dict[str, Any], value)
    expected = _require_sha256(result.get("config_sha256"), "config_sha256")
    payload = {key: item for key, item in result.items() if key != "config_sha256"}
    if expected != canonical_sha256(payload):
        raise ValueError("contrastive scale configuration hash differs")
    scale_selection = _require_dict(result.get("scale_selection"), "scale_selection")
    if scale_selection.get("scales_descending") != list(SCALE_ORDER):
        raise ValueError("contrastive scale configuration order differs")
    if scale_selection.get("zero_sanity_control") != 0.0:
        raise ValueError("contrastive scale-zero sanity control differs")
    if scale_selection.get("select_first_passing") is not True:
        raise ValueError("contrastive selection must stop after its first pass")
    if scale_selection.get("selection_subset_sha256s") != [
        suite.subset_sha256 for suite in _SELECTION_SUITES
    ]:
        raise ValueError("contrastive selection subset identities differ")
    if scale_selection.get("independent_final_subset_sha256") != _FINAL_SUITE.subset_sha256:
        raise ValueError("contrastive final-holdout identity differs")
    if result.get("sealed_final_allowed") is not False:
        raise ValueError("sealed-final access must remain disabled")
    return result


def _validate_construction(
    construction: dict[str, Any], *, protocol_sha256: str
) -> tuple[str, str]:
    _validate_summary_hash(construction, label="contrastive construction")
    if construction.get("construction_id") != _CONSTRUCTION_ID:
        raise ValueError("contrastive construction ID differs")
    if construction.get("protocol_sha256") != protocol_sha256:
        raise ValueError("contrastive construction protocol hash differs")
    if construction.get("gate_passed") is not True:
        raise ValueError("contrastive construction did not pass")
    if construction.get("sealed_final_accessed") is not False:
        raise ValueError("contrastive construction sealed-final marker differs")
    if construction.get("unmerged_and_reversible") is not True:
        raise ValueError("contrastive construction is not unmerged and reversible")
    if construction.get("merged_module_count") != 0:
        raise ValueError("contrastive construction contains merged modules")
    if construction.get("base_state_unchanged") is not True:
        raise ValueError("contrastive construction changed the base state")
    if construction.get("source_directories_unchanged") is not True:
        raise ValueError("contrastive construction changed a source adapter")
    for field in ("dense_equivalence", "functional_logit_equivalence"):
        evidence = _require_dict(construction.get(field), field)
        if evidence.get("gate_passed") is not True:
            raise ValueError(f"contrastive construction {field} did not pass")
    scale_zero = _require_dict(construction.get("scale_zero_sanity"), "scale_zero_sanity")
    scale_one = _require_dict(construction.get("scale_one_sanity"), "scale_one_sanity")
    if scale_zero.get("matches_untouched_base") is not True:
        raise ValueError("contrastive scale-zero sanity did not pass")
    if scale_one.get("matches_unscaled_contrastive") is not True:
        raise ValueError("contrastive scale-one sanity did not pass")
    adapter_sha256 = _require_sha256(
        construction.get("contrastive_adapter_sha256"), "contrastive_adapter_sha256"
    )
    return adapter_sha256, cast(str, construction["summary_sha256"])


def _wilson_lower_bound(successes: int, total: int) -> float:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson inputs are outside their valid range")
    z = 1.959963984540054
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    centre = proportion + z_squared / (2.0 * total)
    spread = z * math.sqrt((proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total)
    return (centre - spread) / denominator


def _identity_for_evaluation(evaluation: dict[str, Any]) -> _SuiteIdentity:
    candidates = (*_SELECTION_SUITES, _FINAL_SUITE)
    for identity in candidates:
        if (
            evaluation.get("evaluation_id") == identity.evaluation_id
            and evaluation.get("suite_sha256") == identity.suite_sha256
            and evaluation.get("base_conditioned_subset_id") == identity.subset_id
            and evaluation.get("base_conditioned_subset_sha256") == identity.subset_sha256
        ):
            return identity
    raise ValueError("retention evaluation suite or subset identity differs")


def _validate_scale_evidence(evaluation: dict[str, Any], *, scale: float) -> str:
    evidence = _require_dict(evaluation.get("adapter_scale_evidence"), "adapter_scale_evidence")
    if evidence.get("implementation_id") != _SCALING_IMPLEMENTATION_ID:
        raise ValueError("retention evaluation scaling implementation differs")
    actual_scale = _require_float(evidence.get("scale"), "adapter scale evidence scale")
    if actual_scale != scale:
        raise ValueError("retention evaluation scale evidence differs")
    if not all(
        evidence.get(key) is True
        for key in (
            "original_scaling_restored",
            "adapter_state_unchanged",
            "base_parameter_signature_unchanged",
        )
    ):
        raise ValueError("retention evaluation did not restore model state")
    if evidence.get("adapter_state_sha256_before") != evidence.get("adapter_state_sha256_after"):
        raise ValueError("retention evaluation adapter state hashes differ")
    if evidence.get("base_parameter_signature_before") != evidence.get(
        "base_parameter_signature_after"
    ):
        raise ValueError("retention evaluation base state hashes differ")
    for field in (
        "adapter_state_sha256_before",
        "adapter_state_sha256_after",
        "base_parameter_signature_before",
        "base_parameter_signature_after",
        "original_scaling_sha256",
        "applied_scaling_sha256",
    ):
        _require_sha256(evidence.get(field), f"adapter_scale_evidence.{field}")
    return canonical_sha256(evidence)


def _validate_evaluation(
    evaluation: dict[str, Any],
    *,
    scale: float,
    expected_identity: _SuiteIdentity,
    expected_adapter_sha256: str,
) -> str:
    _validate_summary_hash(evaluation, label="retention evaluation")
    if _identity_for_evaluation(evaluation) != expected_identity:
        raise ValueError("retention evaluation identity differs")
    if evaluation.get("base_conditioned_instrument_id") != _INSTRUMENT_ID:
        raise ValueError("retention evaluation instrument ID differs")
    if evaluation.get("adapter_sha256") != expected_adapter_sha256:
        raise ValueError("retention evaluation adapter hash differs from construction")
    actual_scale = _require_float(evaluation.get("adapter_scale"), "evaluation adapter_scale")
    if actual_scale != scale:
        raise ValueError("retention evaluation lacks the exact explicit adapter scale")
    if evaluation.get("sealed_final_accessed") is True:
        raise ValueError("sealed-final evidence cannot enter contrastive scale selection")
    return _validate_scale_evidence(evaluation, scale=scale)


def _validate_metric_consistency(
    assessment: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, bool]:
    total = _require_int(assessment.get("total"), "assessment total", minimum=1)
    if total != _require_int(evaluation.get("total"), "evaluation total", minimum=1):
        raise ValueError("assessment and evaluation totals differ")
    assessment_sections = _require_dict(
        assessment.get("section_preservation"), "section_preservation"
    )
    evaluation_sections = _require_dict(evaluation.get("section_metrics"), "section_metrics")
    if set(assessment_sections) != set(_SECTION_ORDER) or set(evaluation_sections) != set(
        _SECTION_ORDER
    ):
        raise ValueError("retention section key set differs")

    preserved = 0
    section_rates: dict[str, float] = {}
    section_totals = 0
    for section in _SECTION_ORDER:
        assessment_metric = _require_dict(assessment_sections[section], f"{section} assessment")
        evaluation_metric = _require_dict(evaluation_sections[section], f"{section} evaluation")
        if set(assessment_metric) != {"preserved", "total", "rate"}:
            raise ValueError(f"{section} assessment metric keys differ")
        if set(evaluation_metric) != {"correct", "total", "accuracy"}:
            raise ValueError(f"{section} evaluation metric keys differ")
        count = _require_int(assessment_metric.get("preserved"), f"{section} preserved")
        evaluation_count = _require_int(evaluation_metric.get("correct"), f"{section} correct")
        section_total = _require_int(assessment_metric.get("total"), f"{section} total", minimum=1)
        evaluation_total = _require_int(
            evaluation_metric.get("total"), f"{section} evaluation total", minimum=1
        )
        if count != evaluation_count or section_total != evaluation_total:
            raise ValueError(f"{section} assessment and evaluation counts differ")
        if count > section_total:
            raise ValueError(f"{section} preserved count exceeds its total")
        rate = count / section_total
        _same_rate(assessment_metric.get("rate"), rate, f"{section} preservation rate")
        _same_rate(evaluation_metric.get("accuracy"), rate, f"{section} evaluation accuracy")
        preserved += count
        section_totals += section_total
        section_rates[section] = rate
    if section_totals != total:
        raise ValueError("retention section totals differ from overall total")
    if preserved != _require_int(assessment.get("preserved"), "assessment preserved"):
        raise ValueError("assessment preserved count differs from section counts")
    broken = _require_int(assessment.get("broken"), "assessment broken")
    if broken != total - preserved:
        raise ValueError("assessment broken count differs")
    broken_ids = assessment.get("broken_item_ids")
    if (
        not isinstance(broken_ids, list)
        or any(not isinstance(item, str) for item in broken_ids)
        or len(broken_ids) != broken
        or len(set(broken_ids)) != len(broken_ids)
    ):
        raise ValueError("assessment broken item IDs differ from its broken count")
    _same_rate(assessment.get("overall_preservation"), preserved / total, "overall preservation")
    _same_rate(
        assessment.get("overall_wilson_95_lower_bound"),
        _wilson_lower_bound(preserved, total),
        "overall Wilson lower bound",
    )
    transitions = _require_dict(assessment.get("paired_transitions"), "paired_transitions")
    if transitions != {
        "base_pass_adapter_pass": preserved,
        "base_pass_adapter_fail": broken,
    }:
        raise ValueError("paired transition counts differ")

    extractable = _require_int(assessment.get("extractable"), "assessment extractable")
    if extractable != _require_int(evaluation.get("extractable"), "evaluation extractable"):
        raise ValueError("assessment and evaluation extractable counts differ")
    if extractable > total:
        raise ValueError("extractable count exceeds total")
    _same_rate(assessment.get("extractability"), extractable / total, "assessment extractability")
    _same_rate(evaluation.get("extractability"), extractable / total, "evaluation extractability")
    prompt_echo = _require_int(assessment.get("prompt_echo"), "assessment prompt echo")
    if prompt_echo != _require_int(evaluation.get("prompt_echo"), "evaluation prompt echo"):
        raise ValueError("assessment and evaluation prompt-echo counts differ")
    if prompt_echo > total:
        raise ValueError("prompt-echo count exceeds total")
    _same_rate(assessment.get("prompt_echo_rate"), prompt_echo / total, "prompt echo rate")
    _same_rate(
        evaluation.get("prompt_echo_rate"), prompt_echo / total, "evaluation prompt echo rate"
    )
    for field in ("question_generation", "malformed_outputs", "backend_failures"):
        assessment_value = _require_int(assessment.get(field), f"assessment {field}")
        evaluation_value = _require_int(evaluation.get(field), f"evaluation {field}")
        if assessment_value != evaluation_value:
            raise ValueError(f"assessment and evaluation {field} differ")
        if assessment_value > total:
            raise ValueError(f"{field} count exceeds total")
    if assessment["malformed_outputs"] != total - extractable:
        raise ValueError("malformed-output and extractable counts are inconsistent")
    exact_format = _require_int(evaluation.get("exact_format"), "evaluation exact_format")
    if exact_format > total:
        raise ValueError("exact-format count exceeds total")
    _same_rate(
        evaluation.get("exact_format_rate"), exact_format / total, "evaluation exact-format rate"
    )
    if assessment.get("raw_packet_sha256") != evaluation.get("raw_packet_sha256"):
        raise ValueError("assessment and evaluation raw-packet hashes differ")
    _require_sha256(assessment.get("raw_packet_sha256"), "raw_packet_sha256")

    family_failures = _require_dict(
        assessment.get("instruction_family_adapter_only_failures"),
        "instruction_family_adapter_only_failures",
    )
    if any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in family_failures.items()
    ):
        raise ValueError("instruction-family failure counts are invalid")
    maximum_family_failures = max(cast(dict[str, int], family_failures).values(), default=0)
    if sum(cast(dict[str, int], family_failures).values()) > broken:
        raise ValueError("instruction-family failures exceed total broken items")
    if (
        assessment.get("maximum_instruction_family_adapter_only_failures")
        != maximum_family_failures
    ):
        raise ValueError("maximum instruction-family failure count differs")

    return {
        "overall_preservation_at_least_90_percent": preserved / total >= 0.90,
        "arithmetic_preservation_at_least_90_percent": section_rates["arithmetic"] >= 0.90,
        "format_preservation_at_least_90_percent": section_rates["format"] >= 0.90,
        "instruction_preservation_at_least_90_percent": section_rates["instruction"] >= 0.90,
        "overall_wilson_lower_bound_at_least_85_percent": _wilson_lower_bound(preserved, total)
        >= 0.85,
        "prompt_echo_at_most_2_percent": prompt_echo / total <= 0.02,
        "zero_question_generation": assessment["question_generation"] == 0,
        "zero_backend_failures": assessment["backend_failures"] == 0,
        "instruction_family_adapter_only_failures_at_most_3": maximum_family_failures <= 3,
    }


def _validate_assessment(
    assessment: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    scale: float,
    expected_identity: _SuiteIdentity,
    expected_adapter_sha256: str,
) -> None:
    _validate_summary_hash(assessment, label="retention assessment")
    if assessment.get("assessment_id") != _ASSESSMENT_ID:
        raise ValueError("retention assessment ID differs")
    if assessment.get("instrument_id") != _INSTRUMENT_ID:
        raise ValueError("retention assessment instrument ID differs")
    if assessment.get("scaling_implementation_id") != _SCALING_IMPLEMENTATION_ID:
        raise ValueError("retention assessment scaling implementation differs")
    if assessment.get("adapter_sha256") != expected_adapter_sha256:
        raise ValueError("retention assessment adapter hash differs from construction")
    actual_scale = _require_float(assessment.get("adapter_scale"), "assessment adapter_scale")
    if actual_scale != scale:
        raise ValueError("retention assessment lacks the exact explicit adapter scale")
    if assessment.get("state_restoration_verified") is not True:
        raise ValueError("retention assessment lacks state-restoration evidence")
    if assessment.get("suite_id") != expected_identity.suite_id or any(
        assessment.get(field) != expected
        for field, expected in (
            ("suite_sha256", expected_identity.suite_sha256),
            ("subset_id", expected_identity.subset_id),
            ("subset_sha256", expected_identity.subset_sha256),
        )
    ):
        raise ValueError("retention assessment suite or subset identity differs")
    if assessment.get("sealed_final_accessed") is True:
        raise ValueError("sealed-final evidence cannot enter contrastive scale selection")
    if any(
        assessment.get(key) is True
        for key in ("selection_used_gsm1k", "gsm1k_used_for_selection", "gsm1k_accessed")
    ):
        raise ValueError("GSM1K evidence cannot enter contrastive scale selection")

    scale_evidence_sha256 = _validate_evaluation(
        evaluation,
        scale=scale,
        expected_identity=expected_identity,
        expected_adapter_sha256=expected_adapter_sha256,
    )
    if assessment.get("evaluation_summary_sha256") != evaluation.get("summary_sha256"):
        raise ValueError("retention assessment evaluation-summary linkage differs")
    if assessment.get("adapter_scale_evidence_sha256") != scale_evidence_sha256:
        raise ValueError("retention assessment scale-evidence hash differs")

    expected_checks = _validate_metric_consistency(assessment, evaluation)
    gate_checks = _require_dict(assessment.get("gate_checks"), "gate_checks")
    if set(gate_checks) != _GATE_CHECK_KEYS:
        raise ValueError("retention gate check key set differs")
    if gate_checks != expected_checks:
        raise ValueError("retention gate decisions differ from recomputed metrics")
    expected_passed = all(expected_checks.values())
    if assessment.get("gate_passed") is not expected_passed:
        raise ValueError("retention aggregate gate decision differs from recomputed checks")


def _canonical_matrix(
    *,
    scale: float,
    assessments: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    adapter_sha256: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(assessments) != len(_SELECTION_SUITES) or len(evaluations) != len(_SELECTION_SUITES):
        raise ValueError(
            "each contrastive scale requires exactly two assessments and two evaluations"
        )
    assessments_by_suite = {str(item.get("suite_id")): item for item in assessments}
    if len(assessments_by_suite) != len(assessments):
        raise ValueError("contrastive scale assessment suites are duplicated")
    evaluations_by_suite: dict[str, dict[str, Any]] = {}
    for evaluation in evaluations:
        identity = _identity_for_evaluation(evaluation)
        if identity.suite_id in evaluations_by_suite:
            raise ValueError("contrastive scale evaluation suites are duplicated")
        evaluations_by_suite[identity.suite_id] = evaluation
    expected_suites = {identity.suite_id for identity in _SELECTION_SUITES}
    if set(assessments_by_suite) != expected_suites or set(evaluations_by_suite) != expected_suites:
        raise ValueError("contrastive scale evidence matrix is incomplete")

    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for identity in _SELECTION_SUITES:
        assessment = assessments_by_suite[identity.suite_id]
        evaluation = evaluations_by_suite[identity.suite_id]
        _validate_assessment(
            assessment,
            evaluation,
            scale=scale,
            expected_identity=identity,
            expected_adapter_sha256=adapter_sha256,
        )
        result.append((assessment, evaluation))
    return result


def build_selection(
    *,
    construction: dict[str, Any],
    matrices: list[tuple[float, list[dict[str, Any]]]],
    evaluation_matrices: list[tuple[float, list[dict[str, Any]]]],
    scale_config_sha256: str,
) -> dict[str, Any]:
    """Select the first passing contrastive scale in the frozen descending order."""

    protocol_sha256 = _require_sha256(scale_config_sha256, "scale_config_sha256")
    adapter_sha256, construction_summary_sha256 = _validate_construction(
        construction, protocol_sha256=protocol_sha256
    )
    scales = [scale for scale, _ in matrices]
    evaluation_scales = [scale for scale, _ in evaluation_matrices]
    if scales != list(SCALE_ORDER[: len(scales)]):
        raise ValueError("scales were not evaluated in the frozen descending order")
    if evaluation_scales != scales:
        raise ValueError("assessment and evaluation scale matrices differ")
    if not matrices:
        raise ValueError("contrastive scale selection cannot be empty")

    selected: float | None = None
    records: list[dict[str, Any]] = []
    for (scale, assessments), (_, evaluations) in zip(matrices, evaluation_matrices, strict=True):
        cells = _canonical_matrix(
            scale=scale,
            assessments=assessments,
            evaluations=evaluations,
            adapter_sha256=adapter_sha256,
        )
        passed = all(assessment["gate_passed"] is True for assessment, _ in cells)
        records.append(
            {
                "scale": scale,
                "matrix_passed": passed,
                "assessment_summary_sha256s": [
                    str(assessment["summary_sha256"]) for assessment, _ in cells
                ],
                "evaluation_summary_sha256s": [
                    str(evaluation["summary_sha256"]) for _, evaluation in cells
                ],
                "cell_gate_passed": [bool(assessment["gate_passed"]) for assessment, _ in cells],
            }
        )
        if passed:
            selected = scale
            break

    if selected is not None and len(records) != len(matrices):
        raise ValueError("lower scales were supplied after a contrastive scale passed")
    if selected is None and len(matrices) != len(SCALE_ORDER):
        raise ValueError("a failed partial search cannot close contrastive scale selection")

    result: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": "foundry-contrastive-lora-scale-selection-v1",
        "scale_config_sha256": protocol_sha256,
        "construction_summary_sha256": construction_summary_sha256,
        "adapter_sha256": adapter_sha256,
        "selection_suites": [identity.as_dict() for identity in _SELECTION_SUITES],
        "scales": records,
        "selected_contrastive_scale": selected,
        "scale_zero_selectable": False,
        "lower_scales_skipped_after_first_pass": selected is not None,
        "selection_passed": selected is not None,
        "selection_used_gsm1k": False,
        "independent_final_holdout_used_for_selection": False,
        "gsm1k_authorized": False,
        "contrastive_adapter_route_stopped": selected is None,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def build_final_validation(
    *,
    construction: dict[str, Any],
    selection: dict[str, Any],
    assessment: dict[str, Any],
    evaluation_summary: dict[str, Any],
) -> dict[str, Any]:
    """Freeze one independent final-holdout validation of the selected scale."""

    _validate_summary_hash(selection, label="contrastive scale selection")
    if selection.get("decision_id") != "foundry-contrastive-lora-scale-selection-v1":
        raise ValueError("final validation received the wrong selection decision")
    if selection.get("selection_used_gsm1k") is not False:
        raise ValueError("contrastive selection must not use GSM1K")
    if selection.get("sealed_final_accessed") is not False:
        raise ValueError("contrastive selection sealed-final marker differs")
    protocol_sha256 = _require_sha256(selection.get("scale_config_sha256"), "scale_config_sha256")
    adapter_sha256, construction_summary_sha256 = _validate_construction(
        construction, protocol_sha256=protocol_sha256
    )
    if selection.get("construction_summary_sha256") != construction_summary_sha256:
        raise ValueError("final validation construction evidence differs from selection")
    if selection.get("adapter_sha256") != adapter_sha256:
        raise ValueError("final validation construction adapter differs from selection")
    selected = selection.get("selected_contrastive_scale")
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int | float)
        or float(selected) not in SCALE_ORDER
        or selection.get("selection_passed") is not True
    ):
        raise ValueError("final validation requires one selected contrastive scale")
    _validate_assessment(
        assessment,
        evaluation_summary,
        scale=float(selected),
        expected_identity=_FINAL_SUITE,
        expected_adapter_sha256=adapter_sha256,
    )

    passed = assessment["gate_passed"] is True
    result: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": "foundry-contrastive-lora-scale-final-validation-v1",
        "selection_summary_sha256": selection["summary_sha256"],
        "construction_summary_sha256": construction_summary_sha256,
        "selected_contrastive_scale": float(selected),
        "adapter_sha256": adapter_sha256,
        "assessment_summary_sha256": assessment["summary_sha256"],
        "evaluation_summary_sha256": evaluation_summary["summary_sha256"],
        **_FINAL_SUITE.as_dict(),
        "retention_decision": (
            "retention_approved_targeted_minus_generic_task_vector"
            if passed
            else "failed_independent_contrastive_scale_validation"
        ),
        "retention_passed": passed,
        "gsm1k_authorized": passed,
        "contrastive_adapter_route_stopped": not passed,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def _parse_matrix_values(values: list[str]) -> list[tuple[float, list[dict[str, Any]]]]:
    matrices: list[tuple[float, list[dict[str, Any]]]] = []
    for value in values:
        scale_text, paths_text = value.split("=", 1)
        paths = [Path(path) for path in paths_text.split(",")]
        matrices.append((float(scale_text), [_load(path) for path in paths]))
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--config", required=True, type=Path)
    select.add_argument("--construction", required=True, type=Path)
    select.add_argument("--scale-matrix", required=True, action="append")
    select.add_argument("--evaluation-matrix", required=True, action="append")
    select.add_argument("--output", required=True, type=Path)
    final = subparsers.add_parser("final")
    final.add_argument("--construction", required=True, type=Path)
    final.add_argument("--selection", required=True, type=Path)
    final.add_argument("--assessment", required=True, type=Path)
    final.add_argument("--evaluation-summary", required=True, type=Path)
    final.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        config = _load_config(args.config)
        result = build_selection(
            construction=_load(args.construction),
            matrices=_parse_matrix_values(args.scale_matrix),
            evaluation_matrices=_parse_matrix_values(args.evaluation_matrix),
            scale_config_sha256=str(config["config_sha256"]),
        )
    else:
        result = build_final_validation(
            construction=_load(args.construction),
            selection=_load(args.selection),
            assessment=_load(args.assessment),
            evaluation_summary=_load(args.evaluation_summary),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
