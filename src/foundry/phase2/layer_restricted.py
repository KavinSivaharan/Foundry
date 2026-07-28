"""Freeze and adjudicate the Milestone 13E layer-restricted LoRA experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

ARMS = ("generic", "targeted")
DEVELOPMENT_SUITES = ("adjudication", "anchor")
CHECKPOINTS_DESCENDING = (64, 32, 16)
DATASET_SHA256 = "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
V1_LORA_SHA256 = "3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"
V1_RECIPE_DECISION_SHA256 = "b03dfc9d6f66843f2a83e9b4ff5b82e133fb9c6a4a27ab1783d64991a0f7d118"
ENVIRONMENT_SHA256 = "9244dd7aa9d4d5138ef01f1b4fb20b911fc390e034e5704ded4ba8fcd967244b"
COMBINED_CHILD_ENVIRONMENT_SHA256 = (
    "1d402ec0cb661adeb50a3d3bd9510895f3f9068cbb393fb381565a5670de995b"
)
SCHEDULES = {
    "generic": "4bc00d29d5cf308c12c77111d7943567521cc533b13440dc06c3d8b39c74e9df",
    "targeted": "88c5378cac7efe927b29d3f421d97777cd6d917187c71c8388b60bbe7b57e259",
}
HOLDOUT_V2_SUITE_SHA256 = "b8b978ba69b501187b984d4631e8d7f5a41f39efeb292cb391a845c3e61e1b18"
HOLDOUT_V2_SUBSET_SHA256 = "a23b1014d92e9f98b74da3b29913a430bdaebf8e07a16b31b4c3dcc831f1f420"


@dataclass(frozen=True)
class LayerScope:
    """One immutable layer-selection cell."""

    label: str
    top_layer_count: int
    layer_indices: tuple[int, ...]

    @property
    def adapted_module_count(self) -> int:
        return self.top_layer_count * 4

    @property
    def trainable_tensor_count(self) -> int:
        return self.top_layer_count * 8

    @property
    def trainable_parameter_count(self) -> int:
        return self.top_layer_count * 77_824

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "definition": f"top_{self.top_layer_count}_layers",
            "top_layer_count": self.top_layer_count,
            "layer_indices": list(self.layer_indices),
            "adapted_module_count": self.adapted_module_count,
            "trainable_tensor_count": self.trainable_tensor_count,
            "trainable_parameter_count": self.trainable_parameter_count,
        }


LAYER_SCOPES = (
    LayerScope("L1", 4, tuple(range(24, 28))),
    LayerScope("L2", 8, tuple(range(20, 28))),
    LayerScope("L3", 14, tuple(range(14, 28))),
)
SCOPE_LABELS = tuple(scope.label for scope in LAYER_SCOPES)


def scope_for_label(label: str) -> LayerScope:
    """Return one predeclared scope and reject every other layer selection."""

    for scope in LAYER_SCOPES:
        if scope.label == label:
            return scope
    raise ValueError("layer scope is not predeclared")


def select_largest_passing(scope_passes: Mapping[str, bool]) -> str | None:
    """Select L3, then L2, then L1, from one complete frozen ladder."""

    if set(scope_passes) != set(SCOPE_LABELS):
        raise ValueError("scope pass map must cover the exact frozen ladder")
    for scope in reversed(LAYER_SCOPES):
        if scope_passes[scope.label]:
            return scope.label
    return None


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify(value: dict[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _freeze(value: dict[str, Any], key: str) -> dict[str, Any]:
    if key in value:
        raise ValueError(f"{key} already exists")
    value[key] = canonical_sha256(value)
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_contract(root: Path) -> dict[str, Any]:
    """Build the content-free Stage A/B contract without model or holdout access."""

    dataset = _read(root / "results/phase2_vetted_corpus/dataset_summary.json")
    recipe = _read(root / "results/phase2_vetted_corpus/milestone13c_r3_v1_kl_recipe.json")
    environment = _read(root / "results/phase2_vetted_corpus/windows_operational_environment.json")
    if dataset["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("vetted dataset identity differs")
    _verify(recipe, "final_recipe_decision_sha256")
    if (
        recipe["final_recipe_decision_sha256"] != V1_RECIPE_DECISION_SHA256
        or recipe["canonical_lora_configuration"]["canonical_kl_lora_configuration_sha256"]
        != V1_LORA_SHA256
    ):
        raise ValueError("frozen V1 LoRA recipe differs")
    if (
        environment["environment_evidence_sha256"] != ENVIRONMENT_SHA256
        or environment["combined_child_environment_sha256"] != COMBINED_CHILD_ENVIRONMENT_SHA256
    ):
        raise ValueError("frozen deterministic environment differs")
    source_paths = (
        "src/foundry/phase2/layer_restricted.py",
        "src/foundry/phase2/layer_restricted_campaign.py",
        "src/foundry/phase2/vetted_qlora_layer_restricted.py",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": "foundry-milestone13e-layer-restricted-lora-v1",
        "authorization_starting_commit": "4b3421e4312c95c564947e1886fb8aab151d12ad",
        "base_model": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": MODEL_REVISION,
        },
        "dataset_sha256": DATASET_SHA256,
        "arms": list(ARMS),
        "v1_lora_configuration": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "target_modules_in_construction_order": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            "total_transformer_layers": 28,
            "v1_lora_configuration_sha256": V1_LORA_SHA256,
            "recipe_decision_sha256": V1_RECIPE_DECISION_SHA256,
        },
        "sole_scientific_intervention": "layers_to_transform",
        "unchanged_fields": [
            "dataset",
            "generic_and_targeted_schedules",
            "replay_ratio",
            "seed",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "scheduler_horizon",
            "gradient_clipping",
            "assistant_only_masking",
            "quantization",
            "gradient_checkpointing",
            "lora_rank_alpha_dropout_and_projection_targets",
        ],
        "layer_ladder": [scope.as_dict() for scope in LAYER_SCOPES],
        "calibration_order": [f"{scope.label}/{arm}" for scope in LAYER_SCOPES for arm in ARMS],
        "selection_order": [scope.label for scope in reversed(LAYER_SCOPES)],
        "selection_rule": "largest_scope_where_both_arms_pass_all_development_gates",
        "calibration": {
            "optimizer_steps_per_run": 16,
            "loss_bearing_tokens_per_run": 16_000,
            "run_count": 6,
            "total_optimizer_steps": 96,
            "total_loss_bearing_tokens": 96_000,
            "recorded_metrics": [
                "validation_ce",
                "adjudication_development_retention",
                "anchor_development_retention",
                "failure_family_counts",
                "diagnostic_only_replay_kl",
            ],
            "eligibility": [
                "all_adjudication_retention_thresholds_pass",
                "all_anchor_retention_thresholds_pass",
                "zero_backend_failures",
            ],
        },
        "full_training": {
            "optimizer_steps_per_arm": 64,
            "loss_bearing_tokens_per_arm": 64_000,
            "checkpoint_order": list(CHECKPOINTS_DESCENDING),
            "checkpoint_selection_rule": "latest_common_development_passing_checkpoint",
        },
        "holdout_v2": {
            "adapter_unexposed_before_final_checkpoint_selection": True,
            "adapter_evaluations_per_arm": 1,
            "suite_sha256": HOLDOUT_V2_SUITE_SHA256,
            "subset_sha256": HOLDOUT_V2_SUBSET_SHA256,
            "gate": {
                "overall_minimum": 0.90,
                "arithmetic_minimum": 0.90,
                "format_minimum": 0.90,
                "instruction_minimum": 0.90,
                "maximum_instruction_family_failures": 3,
                "question_generation": 0,
                "backend_failures": 0,
            },
        },
        "gsm1k": {
            "authorized_only_if_both_holdout_v2_arms_pass": True,
            "base_result_source": "frozen_814_example_base_evaluation",
            "success_rule": "targeted_correct_greater_than_generic_and_base",
        },
        "schedules": SCHEDULES,
        "seed": 20260720,
        "environment_evidence_sha256": ENVIRONMENT_SHA256,
        "combined_child_environment_sha256": COMBINED_CHILD_ENVIRONMENT_SHA256,
        "source_file_sha256": {path: file_sha256(root / path) for path in source_paths},
        "pre_contract_counts": {
            "model_processes": 0,
            "optimizer_steps": 0,
            "holdout_v2_adapter_evaluations": 0,
            "gsm1k_adapter_evaluations": 0,
            "sealed_final_accesses": 0,
        },
    }
    return _freeze(payload, "contract_sha256")


def _assessment(path: Path, adapter_sha256: str) -> dict[str, Any]:
    value = _read(path)
    _verify(value, "summary_sha256")
    if value["adapter_sha256"] != adapter_sha256:
        raise ValueError("retention assessment adapter identity differs")
    return {
        "assessment_sha256": value["summary_sha256"],
        "assessment_file_sha256": file_sha256(path),
        "suite_sha256": value["suite_sha256"],
        "subset_sha256": value["subset_sha256"],
        "preserved": value["preserved"],
        "total": value["total"],
        "overall_preservation": value["overall_preservation"],
        "section_preservation": value["section_preservation"],
        "broken": value["broken"],
        "broken_item_ids": value["broken_item_ids"],
        "instruction_family_adapter_only_failures": value[
            "instruction_family_adapter_only_failures"
        ],
        "maximum_instruction_family_adapter_only_failures": value[
            "maximum_instruction_family_adapter_only_failures"
        ],
        "question_generation": value["question_generation"],
        "backend_failures": value["backend_failures"],
        "gate_checks": value["gate_checks"],
        "gate_passed": value["gate_passed"],
    }


def _training(
    summary_path: Path,
    adapter_path: Path,
    scope: LayerScope,
    arm: str,
    steps: int,
) -> tuple[dict[str, Any], str]:
    value = _read(summary_path)
    _verify(value, "result_sha256")
    adapter_sha256 = directory_sha256(adapter_path)
    checkpoint = cast(dict[str, Any], value["checkpoints"])[str(steps)]
    if (
        value["scope_label"] != scope.label
        or value["layer_indices"] != list(scope.layer_indices)
        or value["arm"] != arm
        or value["optimizer_steps"] != steps
        or value["loss_bearing_tokens"] != steps * 1_000
        or value["schedule_sha256"] != SCHEDULES[arm]
        or checkpoint["adapter_sha256"] != adapter_sha256
        or value["trainable_inventory"]["trainable_tensor_count"] != scope.trainable_tensor_count
        or value["trainable_inventory"]["trainable_parameter_count"]
        != scope.trainable_parameter_count
        or value["reload_inventory"] != value["trainable_inventory"]
    ):
        raise ValueError("layer-restricted training evidence differs from contract")
    return value, adapter_sha256


def calibration_record(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Aggregate all six calibration cells and apply only the authorized gates."""

    _verify(contract, "contract_sha256")
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13e/calibration"
    rows: list[dict[str, Any]] = []
    for scope in LAYER_SCOPES:
        for arm in ARMS:
            run_root = raw_root / scope.label / arm
            summary_path = run_root / "training_summary.json"
            adapter_path = run_root / "training/checkpoint-16/adapter"
            training, adapter_sha256 = _training(summary_path, adapter_path, scope, arm, 16)
            retention = {
                suite: _assessment(
                    run_root / f"retention/{suite}_assessment.json",
                    adapter_sha256,
                )
                for suite in DEVELOPMENT_SUITES
            }
            measurement = cast(dict[str, Any], training["final_measurement"])
            finite_metrics = all(
                math.isfinite(float(measurement[key]))
                for key in (
                    "vetted_ce",
                    "replay_ce",
                    "replay_token_kl",
                    "vetted_validation_ce",
                )
            )
            criteria = {
                "exactly_16_optimizer_steps": training["optimizer_steps"] == 16,
                "exactly_16000_loss_bearing_tokens": (training["loss_bearing_tokens"] == 16_000),
                "frozen_schedule": training["schedule_sha256"] == SCHEDULES[arm],
                "exact_layer_scope": (training["layer_indices"] == list(scope.layer_indices)),
                "v1_lora_fields_except_layer_selection": (
                    training["v1_lora_configuration_sha256"] == V1_LORA_SHA256
                ),
                "ce_only_training_with_diagnostic_only_replay_kl": (
                    training["training_objective"]["replay_kl"] == "post_training_diagnostic_only"
                ),
                "finite_training_and_diagnostics": (
                    training["finite_gradients"] is True and finite_metrics
                ),
                "lora_updated": training["lora_updated"] is True,
                "base_parameters_unchanged": (
                    training["base_parameters_unchanged"] is True
                    and training["base_restoration"] is True
                ),
                "cuda_only_and_offline_reload": (
                    training["cuda_only"] is True
                    and training["cpu_offload"] is False
                    and training["offline_reload"] is True
                ),
                "all_development_retention_thresholds_pass": all(
                    retention[suite]["gate_passed"] is True for suite in DEVELOPMENT_SUITES
                ),
                "zero_backend_failures": all(
                    retention[suite]["backend_failures"] == 0 for suite in DEVELOPMENT_SUITES
                ),
                "holdout_v2_unexposed": training["holdout_v2_use"] is False,
                "gsm1k_unused": training["gsm1k_use"] is False,
            }
            row: dict[str, Any] = {
                "scope_label": scope.label,
                "top_layer_count": scope.top_layer_count,
                "layer_indices": list(scope.layer_indices),
                "arm": arm,
                "optimizer_steps": training["optimizer_steps"],
                "loss_bearing_tokens": training["loss_bearing_tokens"],
                "schedule_sha256": training["schedule_sha256"],
                "schedule_prefix_sha256": training["schedule_prefix_sha256"],
                "adapter_sha256": adapter_sha256,
                "adapter_bytes": training["checkpoints"]["16"]["bytes"],
                "training_result_sha256": training["result_sha256"],
                "training_file_sha256": file_sha256(summary_path),
                "validation_ce": measurement["vetted_validation_ce"],
                "replay_kl_diagnostic": measurement["replay_token_kl"],
                "final_measurement": measurement,
                "development_retention": retention,
                "criteria": criteria,
                "eligible": all(criteria.values()),
                "runtime_seconds": training["runtime_seconds"],
                "peak_allocated_vram_bytes": training["peak_allocated_vram_bytes"],
                "peak_reserved_vram_bytes": training["peak_reserved_vram_bytes"],
                "peak_process_rss_bytes": training["peak_process_rss_bytes"],
            }
            row["calibration_run_sha256"] = canonical_sha256(row)
            rows.append(row)
    scope_results = []
    common_passes: dict[str, bool] = {}
    for scope in reversed(LAYER_SCOPES):
        pair = [row for row in rows if row["scope_label"] == scope.label]
        common_pass = len(pair) == 2 and all(row["eligible"] is True for row in pair)
        common_passes[scope.label] = common_pass
        scope_results.append(
            {
                "scope_label": scope.label,
                "top_layer_count": scope.top_layer_count,
                "common_pass": common_pass,
                "eligible_by_arm": {str(row["arm"]): row["eligible"] for row in pair},
                "failed_criteria_by_arm": {
                    str(row["arm"]): [
                        name
                        for name, passed in cast(dict[str, bool], row["criteria"]).items()
                        if not passed
                    ]
                    for row in pair
                },
                "run_sha256_by_arm": {
                    str(row["arm"]): row["calibration_run_sha256"] for row in pair
                },
            }
        )
    selected_scope = select_largest_passing(common_passes)
    payload = {
        "schema_version": 1,
        "calibration_id": "foundry-milestone13e-layer-calibration-v1",
        "contract_sha256": contract["contract_sha256"],
        "scope_order": list(SCOPE_LABELS),
        "selection_order": [scope.label for scope in reversed(LAYER_SCOPES)],
        "arm_order": list(ARMS),
        "run_count": len(rows),
        "optimizer_step_count": sum(int(row["optimizer_steps"]) for row in rows),
        "loss_bearing_token_count": sum(int(row["loss_bearing_tokens"]) for row in rows),
        "runs": rows,
        "scope_results": scope_results,
        "selected_scope": selected_scope,
        "decision": (
            "largest_common_passing_scope_selected"
            if selected_scope is not None
            else "no_common_passing_layer_scope"
        ),
        "full_training_authorized": selected_scope is not None,
        "all_backend_failures_zero": all(row["criteria"]["zero_backend_failures"] for row in rows),
        "holdout_v2_adapter_evaluations": 0,
        "gsm1k_adapter_evaluations": 0,
        "sealed_final_accessed": False,
    }
    return _freeze(payload, "calibration_summary_sha256")


def full_selection_record(
    root: Path,
    contract: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Adjudicate full training and freeze the latest common development checkpoint."""

    _verify(contract, "contract_sha256")
    _verify(calibration, "calibration_summary_sha256")
    selected_label = calibration["selected_scope"]
    if not isinstance(selected_label, str):
        raise ValueError("full training requires a selected calibration scope")
    scope = scope_for_label(selected_label)
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13e/full"
    training_rows: dict[str, Any] = {}
    adapters: dict[tuple[str, int], str] = {}
    for arm in ARMS:
        summary_path = raw_root / arm / "training_summary.json"
        final_adapter = raw_root / arm / "training/checkpoint-64/adapter"
        training, _ = _training(summary_path, final_adapter, scope, arm, 64)
        for step in CHECKPOINTS_DESCENDING:
            adapter = raw_root / arm / f"training/checkpoint-{step}/adapter"
            digest = directory_sha256(adapter)
            if training["checkpoints"][str(step)]["adapter_sha256"] != digest:
                raise ValueError("full checkpoint adapter identity differs")
            adapters[(arm, step)] = digest
        training_rows[arm] = {
            "training_result_sha256": training["result_sha256"],
            "training_file_sha256": file_sha256(summary_path),
            "optimizer_steps": training["optimizer_steps"],
            "loss_bearing_tokens": training["loss_bearing_tokens"],
            "checkpoint_validation_ce": {
                step: training["checkpoints"][str(step)]["validation_ce"]
                for step in ("16", "32", "64")
            },
            "final_replay_kl_diagnostic": training["final_measurement"]["replay_token_kl"],
            "runtime_seconds": training["runtime_seconds"],
        }
    checkpoint_rows = []
    selected_checkpoint: int | None = None
    for step in CHECKPOINTS_DESCENDING:
        step_root = raw_root / "development_selection" / f"checkpoint-{step}"
        if not step_root.exists():
            continue
        by_arm: dict[str, Any] = {}
        for arm in ARMS:
            retention = {
                suite: _assessment(
                    step_root / arm / f"retention/{suite}_assessment.json",
                    adapters[(arm, step)],
                )
                for suite in DEVELOPMENT_SUITES
            }
            by_arm[arm] = {
                "adapter_sha256": adapters[(arm, step)],
                "development_retention": retention,
                "passes": all(
                    retention[suite]["gate_passed"] is True
                    and retention[suite]["backend_failures"] == 0
                    for suite in DEVELOPMENT_SUITES
                ),
            }
        common_pass = all(by_arm[arm]["passes"] is True for arm in ARMS)
        checkpoint_rows.append(
            {
                "checkpoint": step,
                "by_arm": by_arm,
                "common_pass": common_pass,
            }
        )
        if selected_checkpoint is None and common_pass:
            selected_checkpoint = step
    payload = {
        "schema_version": 1,
        "selection_id": "foundry-milestone13e-full-development-selection-v1",
        "contract_sha256": contract["contract_sha256"],
        "calibration_summary_sha256": calibration["calibration_summary_sha256"],
        "selected_scope": selected_label,
        "training_runs": training_rows,
        "checkpoint_order": list(CHECKPOINTS_DESCENDING),
        "evaluated_checkpoints": [row["checkpoint"] for row in checkpoint_rows],
        "checkpoint_results": checkpoint_rows,
        "selected_checkpoint": selected_checkpoint,
        "selected_adapter_sha256_by_arm": (
            {arm: adapters[(arm, selected_checkpoint)] for arm in ARMS}
            if selected_checkpoint is not None
            else {}
        ),
        "decision": (
            "latest_common_development_passing_checkpoint_selected"
            if selected_checkpoint is not None
            else "no_common_development_passing_checkpoint"
        ),
        "holdout_v2_authorized": selected_checkpoint is not None,
        "holdout_v2_adapter_evaluations": 0,
        "gsm1k_adapter_evaluations": 0,
        "sealed_final_accessed": False,
    }
    return _freeze(payload, "full_selection_sha256")


def holdout_record(
    root: Path,
    contract: dict[str, Any],
    full_selection: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen gate after exactly one holdout-v2 evaluation per arm."""

    _verify(contract, "contract_sha256")
    _verify(full_selection, "full_selection_sha256")
    checkpoint = full_selection["selected_checkpoint"]
    if not isinstance(checkpoint, int):
        raise ValueError("holdout v2 requires a common selected checkpoint")
    adapters = cast(dict[str, str], full_selection["selected_adapter_sha256_by_arm"])
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13e/holdout_v2"
    arms: dict[str, Any] = {}
    for arm in ARMS:
        assessment = _assessment(
            raw_root / arm / "assessment.json",
            adapters[arm],
        )
        sections = cast(dict[str, dict[str, Any]], assessment["section_preservation"])
        criteria = {
            "overall_at_least_90_percent": (float(assessment["overall_preservation"]) >= 0.90),
            "arithmetic_at_least_90_percent": (float(sections["arithmetic"]["rate"]) >= 0.90),
            "format_at_least_90_percent": (float(sections["format"]["rate"]) >= 0.90),
            "instruction_at_least_90_percent": (float(sections["instruction"]["rate"]) >= 0.90),
            "failure_family_at_most_3": (
                int(assessment["maximum_instruction_family_adapter_only_failures"]) <= 3
            ),
            "question_generation_zero": assessment["question_generation"] == 0,
            "backend_failures_zero": assessment["backend_failures"] == 0,
            "complete_frozen_retention_gate": assessment["gate_passed"] is True,
            "exact_holdout_v2_suite": (assessment["suite_sha256"] == HOLDOUT_V2_SUITE_SHA256),
            "exact_holdout_v2_subset": (assessment["subset_sha256"] == HOLDOUT_V2_SUBSET_SHA256),
        }
        arms[arm] = {
            "adapter_sha256": adapters[arm],
            "assessment": assessment,
            "criteria": criteria,
            "passed": all(criteria.values()),
        }
    both_pass = all(arms[arm]["passed"] is True for arm in ARMS)
    payload = {
        "schema_version": 1,
        "decision_id": "foundry-milestone13e-holdout-v2-decision-v1",
        "contract_sha256": contract["contract_sha256"],
        "full_selection_sha256": full_selection["full_selection_sha256"],
        "selected_scope": full_selection["selected_scope"],
        "selected_checkpoint": checkpoint,
        "adapter_evaluation_order": list(ARMS),
        "adapter_evaluations": 2,
        "arms": arms,
        "both_arms_pass": both_pass,
        "decision": (
            "holdout_v2_passed_gsm1k_authorized"
            if both_pass
            else "holdout_v2_failed_stop_before_gsm1k"
        ),
        "gsm1k_authorized": both_pass,
        "gsm1k_adapter_evaluations": 0,
        "sealed_final_accessed": False,
    }
    return _freeze(payload, "holdout_decision_sha256")


def gsm1k_record(
    root: Path,
    contract: dict[str, Any],
    holdout: dict[str, Any],
) -> dict[str, Any]:
    """Compare the two newly evaluated adapters with the frozen base result."""

    _verify(contract, "contract_sha256")
    _verify(holdout, "holdout_decision_sha256")
    if holdout["gsm1k_authorized"] is not True:
        raise ValueError("GSM1K requires both holdout-v2 arms to pass")
    phase1 = _read(root / "results/phase1_summary.json")
    base = cast(dict[str, Any], phase1["base_evaluation"])
    summaries: dict[str, Any] = {}
    for arm in ARMS:
        path = root / f"results/raw/phase2_vetted_corpus/milestone13e/gsm1k/{arm}/summary.json"
        value = _read(path)
        if (
            value["processed_examples"] != 814
            or value["generation_failures"] != 0
            or value["adapter_sha256"] != holdout["arms"][arm]["adapter_sha256"]
        ):
            raise ValueError("GSM1K adapter summary differs from frozen contract")
        summaries[arm] = {
            "summary_file_sha256": file_sha256(path),
            "adapter_sha256": value["adapter_sha256"],
            "correct": value["correct_examples"],
            "processed": value["processed_examples"],
            "accuracy": value["accuracy"],
            "extractable": value["extractable_examples"],
            "backend_failures": value["generation_failures"],
            "runtime_seconds": value["total_runtime_seconds"],
            "input_tokens": value["total_input_tokens"],
            "output_tokens": value["total_output_tokens"],
        }
    base_correct = int(base["correct"])
    generic_correct = int(summaries["generic"]["correct"])
    targeted_correct = int(summaries["targeted"]["correct"])
    success = targeted_correct > generic_correct and targeted_correct > base_correct
    payload = {
        "schema_version": 1,
        "decision_id": "foundry-milestone13e-gsm1k-decision-v1",
        "contract_sha256": contract["contract_sha256"],
        "holdout_decision_sha256": holdout["holdout_decision_sha256"],
        "base": {
            "result_source": "frozen_phase1_814_example_base_evaluation",
            "correct": base_correct,
            "processed": base["processed"],
            "accuracy": base["accuracy"],
            "manifest_sha256": base["manifest_sha256"],
            "config_sha256": base["config_sha256"],
        },
        "generic": summaries["generic"],
        "targeted": summaries["targeted"],
        "targeted_delta_correct_vs_generic": targeted_correct - generic_correct,
        "targeted_delta_correct_vs_base": targeted_correct - base_correct,
        "success_condition": {
            "targeted_greater_than_generic": targeted_correct > generic_correct,
            "targeted_greater_than_base": targeted_correct > base_correct,
        },
        "success": success,
        "decision": (
            "layer_restricted_targeted_training_succeeded"
            if success
            else "layer_restricted_targeted_training_did_not_beat_both_comparators"
        ),
        "adapter_evaluations": 2,
        "sealed_final_accessed": False,
    }
    return _freeze(payload, "gsm1k_decision_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("contract", "calibration", "full", "holdout", "gsm1k"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command in {"calibration", "full", "holdout", "gsm1k"}:
            child.add_argument("--contract", type=Path, required=True)
        if command == "full":
            child.add_argument("--calibration", type=Path, required=True)
        if command == "holdout":
            child.add_argument("--full-selection", type=Path, required=True)
        if command == "gsm1k":
            child.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "contract":
        value = build_contract(root)
    else:
        contract = _read(args.contract)
        if args.command == "calibration":
            value = calibration_record(root, contract)
        elif args.command == "full":
            value = full_selection_record(root, contract, _read(args.calibration))
        elif args.command == "holdout":
            value = holdout_record(root, contract, _read(args.full_selection))
        else:
            value = gsm1k_record(root, contract, _read(args.holdout))
    _write_new(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
