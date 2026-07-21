"""Strict retention-only selection and final validation for verifier GRPO."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from foundry.training.base_conditioned_retention import wilson_lower_bound
from foundry.training.config import canonical_sha256

VariantId = Literal["G1", "G2"]
Arm = Literal["generic_control", "targeted"]
SubsetName = Literal["adjudication", "anchor_holdout", "final_holdout"]

PROTOCOL_ID = "foundry-verifier-grpo-v1"
SELECTION_DECISION_ID = "foundry-verifier-grpo-retention-selection-v1"
FINAL_DECISION_ID = "foundry-verifier-grpo-independent-final-retention-v1"
INSTRUMENT_ID = "foundry-base-conditioned-retention-v1"
ASSESSMENT_ID = "foundry-base-conditioned-retention-assessment-v1"
CHECKPOINTS = (16, 32, 64)
VARIANT_ORDER: tuple[VariantId, ...] = ("G1", "G2")
ARMS: tuple[Arm, ...] = ("generic_control", "targeted")
SELECTION_SUBSETS: tuple[SubsetName, ...] = ("adjudication", "anchor_holdout")
SECTION_ORDER = ("arithmetic", "format", "instruction")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GATE_KEYS = frozenset(
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
class RetentionSubsetContract:
    """Frozen identity and composition of one base-correct retention subset."""

    name: SubsetName
    suite_id: str
    suite_sha256: str
    subset_id: str
    subset_sha256: str
    total: int
    section_totals: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the content-free contract recorded in decision evidence."""

        return {
            "name": self.name,
            "suite_id": self.suite_id,
            "suite_sha256": self.suite_sha256,
            "subset_id": self.subset_id,
            "subset_sha256": self.subset_sha256,
            "total": self.total,
            "section_totals": dict(self.section_totals),
        }


SUBSET_CONTRACTS: Mapping[SubsetName, RetentionSubsetContract] = {
    "adjudication": RetentionSubsetContract(
        name="adjudication",
        suite_id="foundry-retention-adjudication-v2",
        suite_sha256="5caf23be79fa01151af6f7db8d45c2b85bfe24b03a29589e482d51731c8358af",
        subset_id="retention-adjudication-v2-base-correct",
        subset_sha256="c76df74b911b96ca43c2663a123e41347fd544bf6644f15522ccaad7b77099e1",
        total=187,
        section_totals=(("arithmetic", 84), ("format", 48), ("instruction", 55)),
    ),
    "anchor_holdout": RetentionSubsetContract(
        name="anchor_holdout",
        suite_id="foundry-retention-anchor-holdout-v1",
        suite_sha256="bff18b434a284d848387262dde201601278e5c8b573937b3486bed2bf925696e",
        subset_id="retention-anchor-holdout-v1-base-correct",
        subset_sha256="36be91d08f2ab0e05c491094c53965d1aa4f989a730347768877a2548a62c7a9",
        total=210,
        section_totals=(("arithmetic", 96), ("format", 60), ("instruction", 54)),
    ),
    "final_holdout": RetentionSubsetContract(
        name="final_holdout",
        suite_id="foundry-retention-replay-final-holdout-v1",
        suite_sha256="4f49c42cbae8ce7b5029192786f8ff493a4cc445f940063298e0bd22392b6ef9",
        subset_id="retention-replay-final-holdout-v1-base-correct",
        subset_sha256="f56845076a1a59e5ca1a95466541339b56f026e945f86118caec307a690ee4ec",
        total=141,
        section_totals=(("arithmetic", 84), ("format", 27), ("instruction", 30)),
    ),
}


@dataclass(frozen=True)
class RetentionCell:
    """One arm/checkpoint/subset assessment with its checkpoint adapter identity."""

    variant_id: VariantId
    arm: Arm
    checkpoint: int
    subset_name: SubsetName
    checkpoint_adapter_sha256: str
    assessment: Mapping[str, Any]


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _require_int(value: object, field: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} exceeds {maximum}")
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


def _same_float(actual: object, expected: float, field: str) -> None:
    if not math.isclose(_require_float(actual, field), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field} differs from recomputed counts")


def _validate_summary_hash(summary: Mapping[str, Any], label: str) -> str:
    expected = _require_sha256(summary.get("summary_sha256"), f"{label} summary_sha256")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if expected != canonical_sha256(payload):
        raise ValueError(f"{label} summary hash differs")
    return expected


def _expected_gate_checks(
    *,
    preserved: int,
    total: int,
    section_preserved: Mapping[str, int],
    section_totals: Mapping[str, int],
    prompt_echo: int,
    question_generation: int,
    backend_failures: int,
    maximum_instruction_family_failures: int,
) -> dict[str, bool]:
    return {
        "overall_preservation_at_least_90_percent": preserved / total >= 0.90,
        "arithmetic_preservation_at_least_90_percent": (
            section_preserved["arithmetic"] / section_totals["arithmetic"] >= 0.90
        ),
        "format_preservation_at_least_90_percent": (
            section_preserved["format"] / section_totals["format"] >= 0.90
        ),
        "instruction_preservation_at_least_90_percent": (
            section_preserved["instruction"] / section_totals["instruction"] >= 0.90
        ),
        "overall_wilson_lower_bound_at_least_85_percent": (
            wilson_lower_bound(preserved, total) >= 0.85
        ),
        "prompt_echo_at_most_2_percent": prompt_echo / total <= 0.02,
        "zero_question_generation": question_generation == 0,
        "zero_backend_failures": backend_failures == 0,
        "instruction_family_adapter_only_failures_at_most_3": (
            maximum_instruction_family_failures <= 3
        ),
    }


def validate_assessment(
    assessment: Mapping[str, Any],
    *,
    contract: RetentionSubsetContract,
    expected_adapter_sha256: str,
) -> dict[str, Any]:
    """Recompute and validate one tracked base-conditioned assessment."""

    summary_sha256 = _validate_summary_hash(assessment, "retention assessment")
    adapter_sha256 = _require_sha256(assessment.get("adapter_sha256"), "assessment adapter_sha256")
    if adapter_sha256 != _require_sha256(expected_adapter_sha256, "checkpoint adapter_sha256"):
        raise ValueError("retention assessment adapter differs from checkpoint adapter")
    expected_identity = {
        "schema_version": 1,
        "assessment_id": ASSESSMENT_ID,
        "instrument_id": INSTRUMENT_ID,
        "suite_id": contract.suite_id,
        "suite_sha256": contract.suite_sha256,
        "subset_id": contract.subset_id,
        "subset_sha256": contract.subset_sha256,
    }
    if any(assessment.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("retention assessment suite or subset identity differs")
    if "adapter_scale" in assessment and assessment.get("adapter_scale") is not None:
        raise ValueError("verifier GRPO retention cannot use runtime adapter scaling")
    for field in ("evaluation_summary_sha256", "raw_packet_sha256"):
        _require_sha256(assessment.get(field), f"assessment {field}")

    total = _require_int(assessment.get("total"), "assessment total")
    if total != contract.total:
        raise ValueError("retention assessment total differs from frozen subset")
    preserved = _require_int(assessment.get("preserved"), "assessment preserved", maximum=total)
    broken = _require_int(assessment.get("broken"), "assessment broken", maximum=total)
    if preserved + broken != total:
        raise ValueError("retention assessment preserved and broken counts differ")
    _same_float(assessment.get("overall_preservation"), preserved / total, "overall preservation")
    _same_float(
        assessment.get("overall_wilson_95_lower_bound"),
        wilson_lower_bound(preserved, total),
        "overall Wilson lower bound",
    )

    section_metrics = _require_dict(assessment.get("section_preservation"), "section preservation")
    if set(section_metrics) != set(SECTION_ORDER):
        raise ValueError("retention assessment section set differs")
    section_totals = dict(contract.section_totals)
    section_preserved: dict[str, int] = {}
    for section in SECTION_ORDER:
        metric = _require_dict(section_metrics[section], f"{section} preservation")
        if set(metric) != {"preserved", "total", "rate"}:
            raise ValueError(f"{section} preservation fields differ")
        section_total = _require_int(metric.get("total"), f"{section} total")
        if section_total != section_totals[section]:
            raise ValueError(f"{section} total differs from frozen subset")
        value = _require_int(metric.get("preserved"), f"{section} preserved", maximum=section_total)
        section_preserved[section] = value
        _same_float(metric.get("rate"), value / section_total, f"{section} rate")
    if sum(section_preserved.values()) != preserved:
        raise ValueError("section preservation counts do not sum to overall preservation")

    broken_ids = assessment.get("broken_item_ids")
    if (
        not isinstance(broken_ids, list)
        or any(not isinstance(item, str) or not item for item in broken_ids)
        or len(broken_ids) != broken
        or len(set(broken_ids)) != broken
    ):
        raise ValueError("broken retention item IDs differ from the broken count")
    transitions = _require_dict(assessment.get("paired_transitions"), "paired transitions")
    if transitions != {
        "base_pass_adapter_pass": preserved,
        "base_pass_adapter_fail": broken,
    }:
        raise ValueError("paired transition counts differ")

    prompt_echo = _require_int(assessment.get("prompt_echo"), "prompt echo", maximum=total)
    _same_float(assessment.get("prompt_echo_rate"), prompt_echo / total, "prompt echo rate")
    question_generation = _require_int(
        assessment.get("question_generation"), "question generation", maximum=total
    )
    backend_failures = _require_int(
        assessment.get("backend_failures"), "backend failures", maximum=total
    )
    extractable = _require_int(assessment.get("extractable"), "extractable", maximum=total)
    _same_float(assessment.get("extractability"), extractable / total, "extractability")
    malformed = _require_int(
        assessment.get("malformed_outputs"), "malformed outputs", maximum=total
    )
    family_failures = _require_dict(
        assessment.get("instruction_family_adapter_only_failures"),
        "instruction family failures",
    )
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for name, value in family_failures.items()
    ):
        raise ValueError("instruction family failures are malformed")
    maximum_family_failures = max(cast(dict[str, int], family_failures).values(), default=0)
    if sum(cast(dict[str, int], family_failures).values()) != (
        section_totals["instruction"] - section_preserved["instruction"]
    ):
        raise ValueError("instruction family failures do not cover instruction failures")
    recorded_maximum = _require_int(
        assessment.get("maximum_instruction_family_adapter_only_failures"),
        "maximum instruction-family failure concentration",
    )
    if recorded_maximum != maximum_family_failures:
        raise ValueError("maximum instruction-family failure concentration differs")

    expected_checks = _expected_gate_checks(
        preserved=preserved,
        total=total,
        section_preserved=section_preserved,
        section_totals=section_totals,
        prompt_echo=prompt_echo,
        question_generation=question_generation,
        backend_failures=backend_failures,
        maximum_instruction_family_failures=maximum_family_failures,
    )
    gate_checks = _require_dict(assessment.get("gate_checks"), "retention gate checks")
    if (
        set(gate_checks) != _GATE_KEYS
        or any(not isinstance(value, bool) for value in gate_checks.values())
        or gate_checks != expected_checks
    ):
        raise ValueError("retention gate decisions differ from recomputed metrics")
    gate_passed = all(expected_checks.values())
    if assessment.get("gate_passed") is not gate_passed:
        raise ValueError("retention aggregate gate decision differs")

    return {
        "assessment_summary_sha256": summary_sha256,
        "evaluation_summary_sha256": assessment["evaluation_summary_sha256"],
        "raw_packet_sha256": assessment["raw_packet_sha256"],
        "adapter_sha256": adapter_sha256,
        "total": total,
        "preserved": preserved,
        "broken": broken,
        "overall_preservation": preserved / total,
        "overall_wilson_95_lower_bound": wilson_lower_bound(preserved, total),
        "section_preservation": {
            section: {
                "preserved": section_preserved[section],
                "total": section_totals[section],
                "rate": section_preserved[section] / section_totals[section],
            }
            for section in SECTION_ORDER
        },
        "extractable": extractable,
        "extractability": extractable / total,
        "prompt_echo": prompt_echo,
        "question_generation": question_generation,
        "malformed_outputs": malformed,
        "backend_failures": backend_failures,
        "maximum_instruction_family_adapter_only_failures": maximum_family_failures,
        "gate_checks": expected_checks,
        "gate_passed": gate_passed,
    }


def _validate_cell_identity(cell: RetentionCell, *, selection: bool) -> None:
    if cell.variant_id not in VARIANT_ORDER:
        raise ValueError("unknown GRPO variant")
    if cell.arm not in ARMS:
        raise ValueError("unknown GRPO arm")
    if cell.checkpoint not in CHECKPOINTS:
        raise ValueError("retention checkpoint is not frozen")
    allowed = SELECTION_SUBSETS if selection else ("final_holdout",)
    if cell.subset_name not in allowed:
        raise ValueError("retention cell uses the wrong subset stage")
    _require_sha256(cell.checkpoint_adapter_sha256, "checkpoint adapter_sha256")


def _variant_result(variant: VariantId, cells: list[RetentionCell]) -> dict[str, Any]:
    expected_keys = {
        (arm, checkpoint, subset)
        for arm in ARMS
        for checkpoint in CHECKPOINTS
        for subset in SELECTION_SUBSETS
    }
    actual_keys: set[tuple[Arm, int, SubsetName]] = set()
    validated: dict[tuple[Arm, int, SubsetName], dict[str, Any]] = {}
    for cell in cells:
        _validate_cell_identity(cell, selection=True)
        if cell.variant_id != variant:
            raise ValueError("retention cell variant differs from its matrix")
        key = (cell.arm, cell.checkpoint, cell.subset_name)
        if key in actual_keys:
            raise ValueError("retention cell is duplicated")
        actual_keys.add(key)
        validated[key] = validate_assessment(
            cell.assessment,
            contract=SUBSET_CONTRACTS[cell.subset_name],
            expected_adapter_sha256=cell.checkpoint_adapter_sha256,
        )
    if actual_keys != expected_keys:
        raise ValueError("retention variant matrix is incomplete")

    checkpoints: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        adapter_sha256s: dict[str, str] = {}
        cells_record: list[dict[str, Any]] = []
        for arm in ARMS:
            arm_hashes = {
                cast(str, validated[(arm, checkpoint, subset)]["adapter_sha256"])
                for subset in SELECTION_SUBSETS
            }
            if len(arm_hashes) != 1:
                raise ValueError("one arm/checkpoint used different adapters across subsets")
            adapter_sha256s[arm] = arm_hashes.pop()
            for subset in SELECTION_SUBSETS:
                evidence = validated[(arm, checkpoint, subset)]
                cells_record.append(
                    {
                        "arm": arm,
                        "subset": subset,
                        **evidence,
                    }
                )
        passed = all(item["gate_passed"] for item in cells_record)
        checkpoints[str(checkpoint)] = {
            "adapter_sha256s": adapter_sha256s,
            "cells": cells_record,
            "common_checkpoint_passed": passed,
        }
    passing = [
        checkpoint
        for checkpoint in CHECKPOINTS
        if checkpoints[str(checkpoint)]["common_checkpoint_passed"]
    ]
    latest = max(passing) if passing else None
    return {
        "variant_id": variant,
        "checkpoints": checkpoints,
        "latest_common_passing_checkpoint": latest,
    }


def build_checkpoint_selection(cells: list[RetentionCell]) -> dict[str, Any]:
    """Apply G1-first, latest-common-checkpoint selection using retention only."""

    by_variant = {
        variant: [cell for cell in cells if cell.variant_id == variant] for variant in VARIANT_ORDER
    }
    if not by_variant["G1"]:
        raise ValueError("G1 retention evidence is required")
    g1 = _variant_result("G1", by_variant["G1"])
    g1_checkpoint = g1["latest_common_passing_checkpoint"]
    if g1_checkpoint is not None and by_variant["G2"]:
        raise ValueError("G2 evidence is forbidden after G1 has a common passing checkpoint")

    variants = {"G1": g1}
    selected_variant: VariantId | None = None
    selected_checkpoint: int | None = None
    g2_authorized = False
    route_stopped = False
    if g1_checkpoint is not None:
        selected_variant = "G1"
        selected_checkpoint = cast(int, g1_checkpoint)
    elif not by_variant["G2"]:
        g2_authorized = True
    else:
        g2 = _variant_result("G2", by_variant["G2"])
        variants["G2"] = g2
        g2_checkpoint = g2["latest_common_passing_checkpoint"]
        if g2_checkpoint is None:
            route_stopped = True
        else:
            selected_variant = "G2"
            selected_checkpoint = cast(int, g2_checkpoint)

    selected_adapters: dict[str, str] | None = None
    if selected_variant is not None and selected_checkpoint is not None:
        selected_adapters = cast(
            dict[str, str],
            variants[selected_variant]["checkpoints"][str(selected_checkpoint)]["adapter_sha256s"],
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": SELECTION_DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "variant_order": list(VARIANT_ORDER),
        "checkpoint_order": list(CHECKPOINTS),
        "selection_rule": "latest_common_passing_checkpoint_with_g1_precedence",
        "selection_subsets": [SUBSET_CONTRACTS[name].as_dict() for name in SELECTION_SUBSETS],
        "independent_final_subset": SUBSET_CONTRACTS["final_holdout"].as_dict(),
        "variants": variants,
        "selected_variant": selected_variant,
        "selected_checkpoint": selected_checkpoint,
        "selected_adapter_sha256s": selected_adapters,
        "selection_passed": selected_variant is not None,
        "g2_training_authorized": g2_authorized,
        "verifier_grpo_route_stopped": route_stopped,
        "independent_final_holdout_used_for_selection": False,
        "gsm1k_used_for_selection": False,
        "gsm1k_authorized": False,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def _validate_selection(selection: Mapping[str, Any]) -> None:
    _validate_summary_hash(selection, "GRPO retention selection")
    if (
        selection.get("schema_version") != 1
        or selection.get("decision_id") != SELECTION_DECISION_ID
        or selection.get("protocol_id") != PROTOCOL_ID
        or selection.get("variant_order") != list(VARIANT_ORDER)
        or selection.get("checkpoint_order") != list(CHECKPOINTS)
        or selection.get("selection_rule") != "latest_common_passing_checkpoint_with_g1_precedence"
        or selection.get("selection_passed") is not True
        or selection.get("g2_training_authorized") is not False
        or selection.get("gsm1k_authorized") is not False
        or selection.get("gsm1k_used_for_selection") is not False
        or selection.get("independent_final_holdout_used_for_selection") is not False
        or selection.get("verifier_grpo_route_stopped") is not False
        or selection.get("sealed_final_accessed") is not False
    ):
        raise ValueError("GRPO retention selection contract differs")
    if (
        selection.get("selection_subsets")
        != [SUBSET_CONTRACTS[name].as_dict() for name in SELECTION_SUBSETS]
        or selection.get("independent_final_subset") != SUBSET_CONTRACTS["final_holdout"].as_dict()
    ):
        raise ValueError("GRPO retention selection subset contracts differ")

    variants = _require_dict(selection.get("variants"), "retention selection variants")
    if "G1" not in variants or any(name not in VARIANT_ORDER for name in variants):
        raise ValueError("GRPO retention selection variant set differs")
    latest_by_variant: dict[str, int | None] = {}
    for variant, raw_result in variants.items():
        result = _require_dict(raw_result, f"{variant} retention result")
        if result.get("variant_id") != variant:
            raise ValueError("GRPO retention result variant identity differs")
        checkpoints = _require_dict(result.get("checkpoints"), f"{variant} checkpoints")
        if set(checkpoints) != {str(checkpoint) for checkpoint in CHECKPOINTS}:
            raise ValueError("GRPO retention checkpoint set differs")
        passing: list[int] = []
        for checkpoint in CHECKPOINTS:
            record = _require_dict(checkpoints[str(checkpoint)], "checkpoint result")
            adapters = _require_dict(record.get("adapter_sha256s"), "checkpoint adapter hashes")
            if set(adapters) != set(ARMS):
                raise ValueError("checkpoint adapter arms differ")
            for arm in ARMS:
                _require_sha256(adapters[arm], f"{arm} checkpoint adapter_sha256")
            raw_cells = record.get("cells")
            if not isinstance(raw_cells, list) or len(raw_cells) != 4:
                raise ValueError("checkpoint retention cell count differs")
            cell_keys: set[tuple[str, str]] = set()
            for raw_cell in raw_cells:
                cell = _require_dict(raw_cell, "checkpoint retention cell")
                raw_arm = cell.get("arm")
                subset = cell.get("subset")
                if raw_arm not in ARMS or subset not in SELECTION_SUBSETS:
                    raise ValueError("checkpoint retention cell identity differs")
                key = (cast(str, raw_arm), cast(str, subset))
                if key in cell_keys:
                    raise ValueError("checkpoint retention cell is duplicated")
                cell_keys.add(key)
                if cell.get("adapter_sha256") != adapters[cast(str, raw_arm)]:
                    raise ValueError("checkpoint retention cell adapter differs")
                _require_sha256(
                    cell.get("assessment_summary_sha256"),
                    "checkpoint assessment summary_sha256",
                )
                if not isinstance(cell.get("gate_passed"), bool):
                    raise ValueError("checkpoint retention cell gate decision is not Boolean")
            expected_passed = all(bool(cell["gate_passed"]) for cell in raw_cells)
            if record.get("common_checkpoint_passed") is not expected_passed:
                raise ValueError("common-checkpoint decision differs from its cells")
            if expected_passed:
                passing.append(checkpoint)
        latest = max(passing) if passing else None
        if result.get("latest_common_passing_checkpoint") != latest:
            raise ValueError("latest common passing checkpoint differs")
        latest_by_variant[variant] = latest

    expected_checkpoint: int | None
    if latest_by_variant["G1"] is not None:
        if set(variants) != {"G1"} or selection.get("g2_training_authorized") is not False:
            raise ValueError("G2 evidence or authorization exists after G1 passed")
        expected_variant: str | None = "G1"
        expected_checkpoint = latest_by_variant["G1"]
    else:
        if "G2" not in variants:
            raise ValueError("a selected retention decision requires completed G2 evidence")
        expected_variant = "G2" if latest_by_variant["G2"] is not None else None
        expected_checkpoint = latest_by_variant["G2"]
    if (
        selection.get("selected_variant") != expected_variant
        or selection.get("selected_checkpoint") != expected_checkpoint
        or expected_variant is None
    ):
        raise ValueError("selected GRPO variant or checkpoint differs from retention evidence")
    selected_adapters = _require_dict(
        selection.get("selected_adapter_sha256s"), "selected adapter hashes"
    )
    selected_result = _require_dict(variants[expected_variant], "selected variant result")
    selected_checkpoints = _require_dict(
        selected_result["checkpoints"], "selected variant checkpoints"
    )
    selected_checkpoint_result = _require_dict(
        selected_checkpoints[str(expected_checkpoint)], "selected checkpoint result"
    )
    expected_adapters = _require_dict(
        selected_checkpoint_result["adapter_sha256s"], "selected checkpoint adapter hashes"
    )
    if selected_adapters != expected_adapters:
        raise ValueError("selected GRPO adapter hashes differ from checkpoint evidence")


def build_final_decision(
    *, selection: Mapping[str, Any], cells: list[RetentionCell]
) -> dict[str, Any]:
    """Freeze the one-shot independent final-holdout decision for both arms."""

    _validate_selection(selection)
    selected_variant = selection.get("selected_variant")
    selected_checkpoint = selection.get("selected_checkpoint")
    selected_adapters = _require_dict(
        selection.get("selected_adapter_sha256s"), "selected adapter hashes"
    )
    if selected_variant not in VARIANT_ORDER or selected_checkpoint not in CHECKPOINTS:
        raise ValueError("GRPO retention selection lacks a frozen variant/checkpoint")
    if set(selected_adapters) != set(ARMS):
        raise ValueError("GRPO retention selection adapter arms differ")

    if len(cells) != 2 or {cell.arm for cell in cells} != set(ARMS):
        raise ValueError("independent final retention requires exactly both arms")
    records: list[dict[str, Any]] = []
    for arm in ARMS:
        cell = next(item for item in cells if item.arm == arm)
        _validate_cell_identity(cell, selection=False)
        if (
            cell.variant_id != selected_variant
            or cell.checkpoint != selected_checkpoint
            or cell.subset_name != "final_holdout"
            or cell.checkpoint_adapter_sha256 != selected_adapters[arm]
        ):
            raise ValueError("independent final-retention cell differs from selection")
        evidence = validate_assessment(
            cell.assessment,
            contract=SUBSET_CONTRACTS["final_holdout"],
            expected_adapter_sha256=cell.checkpoint_adapter_sha256,
        )
        records.append({"arm": arm, **evidence})
    passed = all(record["gate_passed"] for record in records)
    result: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": FINAL_DECISION_ID,
        "protocol_id": PROTOCOL_ID,
        "selection_summary_sha256": selection["summary_sha256"],
        "selected_variant": selected_variant,
        "selected_checkpoint": selected_checkpoint,
        "selected_adapter_sha256s": selected_adapters,
        "final_subset": SUBSET_CONTRACTS["final_holdout"].as_dict(),
        "arm_results": records,
        "retention_decision": (
            "retention_approved_verifier_grpo_adapters"
            if passed
            else "failed_independent_verifier_grpo_retention"
        ),
        "retention_passed": passed,
        "gsm1k_authorized": passed,
        "verifier_grpo_route_stopped": not passed,
        "one_shot_independent_validation": True,
        "checkpoint_reselection_allowed": False,
        "beta_reselection_allowed": False,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def _load_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return cast(dict[str, Any], value)


def _parse_cell(values: list[str]) -> RetentionCell:
    variant, arm, checkpoint, subset, adapter_sha256, assessment_path = values
    if variant not in VARIANT_ORDER or arm not in ARMS or subset not in SUBSET_CONTRACTS:
        raise ValueError("retention cell tag is unknown")
    return RetentionCell(
        variant_id=cast(VariantId, variant),
        arm=cast(Arm, arm),
        checkpoint=int(checkpoint),
        subset_name=cast(SubsetName, subset),
        checkpoint_adapter_sha256=adapter_sha256,
        assessment=_load_object(Path(assessment_path)),
    )


def main() -> None:
    """Build a checkpoint-selection or independent-final decision artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument(
        "--cell",
        action="append",
        nargs=6,
        metavar=("VARIANT", "ARM", "STEP", "SUBSET", "ADAPTER_SHA256", "ASSESSMENT"),
        required=True,
    )
    select.add_argument("--output", required=True, type=Path)
    final = subparsers.add_parser("final")
    final.add_argument("--selection", required=True, type=Path)
    final.add_argument(
        "--cell",
        action="append",
        nargs=6,
        metavar=("VARIANT", "ARM", "STEP", "SUBSET", "ADAPTER_SHA256", "ASSESSMENT"),
        required=True,
    )
    final.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cells = [_parse_cell(value) for value in args.cell]
    if args.command == "select":
        result = build_checkpoint_selection(cells)
    else:
        result = build_final_decision(selection=_load_object(args.selection), cells=cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
