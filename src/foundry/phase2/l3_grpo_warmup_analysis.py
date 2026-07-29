"""Content-free gate analysis for Milestone 14B-R2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from foundry.phase2.l3_grpo_analysis import (
    BASE_CORRECT,
    GSM1K_EXAMPLES,
    MINIMUM_TARGETED_EXTRACTABILITY,
    STARTING_CORRECT,
    STARTING_HOLDOUT_CORRECT,
    _category_effects,
    _paired,
    _predictions,
    _taxonomy,
)
from foundry.phase2.l3_grpo_schedule import (
    COMPLETIONS_PER_ARM,
    GROUPS_PER_ARM,
    PROMPT_TOKEN_PARITY_MAXIMUM,
)
from foundry.phase2.l3_grpo_warmup_compatibility_campaign import (
    OUTPUT_NAME as COMPATIBILITY_OUTPUT,
)
from foundry.phase2.l3_grpo_warmup_prepare import (
    CONTRACT_OUTPUT as WARMUP_CONTRACT_OUTPUT,
)
from foundry.phase2.l3_grpo_warmup_update import (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    EXPECTED_ZERO_LR_WARMUP_NOOP,
    INVALID_OR_AMBIGUOUS,
    NONZERO_POLICY_UPDATE,
    UNEXPECTED_POSITIVE_LR_NO_UPDATE,
    UNEXPECTED_ZERO_GRADIENT,
)
from foundry.training.config import canonical_sha256
from foundry.training.paired_analysis import paired_bootstrap_interval
from foundry.training.qlora import directory_sha256, file_sha256

CHECKPOINTS = (8, 16, 32)
ARMS = ("generic", "targeted")
DEVELOPMENT_SUITES = ("adjudication", "anchor")
RAW_ROOT = "results/raw/phase2_vetted_corpus/milestone14b_r2"
TRACKED_ROOT = "results/phase2_vetted_corpus"
TRAINING_OUTPUT = "milestone14b_r2_counted_training.json"
DEVELOPMENT_OUTPUT = "milestone14b_r2_development_selection.json"
HOLDOUT_OUTPUT = "milestone14b_r2_holdout_v2_decision.json"
GSM1K_ANALYSIS_OUTPUT = "milestone14b_r2_gsm1k_analysis.json"


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify(value: Mapping[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite scientific evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_counted_training_result(root: Path) -> dict[str, object]:
    """Project both successful warmup-aware training summaries."""

    tracked = root / TRACKED_ROOT
    paired = _read(tracked / "milestone14a_paired_schedule.json")
    compatibility = _read(tracked / COMPATIBILITY_OUTPUT)
    warmup = _read(tracked / WARMUP_CONTRACT_OUTPUT)
    _verify(paired, "paired_schedule_sha256")
    _verify(compatibility, "compatibility_sha256")
    _verify(warmup, "warmup_update_contract_sha256")
    if (
        compatibility.get("gate_passed") is not True
        or compatibility.get("decision") != "pass"
        or compatibility.get("warmup_update_contract_sha256")
        != warmup.get("warmup_update_contract_sha256")
    ):
        raise ValueError("counted training is not compatibility-authorized")

    arms: dict[str, object] = {}
    all_classifications = (
        EXPECTED_ZERO_ADVANTAGE_NOOP,
        EXPECTED_ZERO_LR_WARMUP_NOOP,
        NONZERO_POLICY_UPDATE,
        UNEXPECTED_ZERO_GRADIENT,
        UNEXPECTED_POSITIVE_LR_NO_UPDATE,
        INVALID_OR_AMBIGUOUS,
    )
    for arm in ARMS:
        path = root / RAW_ROOT / "training" / f"{arm}_summary.json"
        partial_path = root / RAW_ROOT / "training" / f"{arm}_partial_evidence.json"
        summary = _read(path)
        partial = _read(partial_path)
        _verify(summary, "summary_sha256")
        _verify(partial, "partial_evidence_sha256")
        trajectory = cast(dict[str, Any], summary["trajectory"])
        gate = cast(dict[str, Any], summary["counted_update_gate"])
        trajectory_gate = cast(dict[str, Any], trajectory["counted_update_gate"])
        counts = cast(dict[str, int], gate["classification_counts"])
        if (
            summary.get("gate_passed") is not True
            or summary.get("groups") != GROUPS_PER_ARM
            or summary.get("completions") != COMPLETIONS_PER_ARM
            or summary.get("optimizer_steps") != GROUPS_PER_ARM
            or summary.get("base_unchanged") is not True
            or summary.get("reference_unchanged") is not True
            or summary.get("policy_updated") is not True
            or summary.get("cpu_offload") is not False
            or summary.get("warmup_update_contract_sha256")
            != warmup.get("warmup_update_contract_sha256")
            or summary.get("partial_evidence_file_sha256") != file_sha256(partial_path)
            or set(cast(dict[str, object], summary["checkpoint_evidence"])) != {"8", "16", "32"}
            or gate != trajectory_gate
            or gate.get("passed") is not True
            or gate.get("step_count") != GROUPS_PER_ARM
            or gate.get("optimizer_call_count") != GROUPS_PER_ARM
            or gate.get("scheduler_advance_count") != GROUPS_PER_ARM
            or gate.get("learning_rate_trajectory_exact") is not True
            or gate.get("reference_update_count") != 0
            or gate.get("base_update_count") != 0
            or counts.get(UNEXPECTED_ZERO_GRADIENT) != 0
            or counts.get(UNEXPECTED_POSITIVE_LR_NO_UPDATE) != 0
            or counts.get(INVALID_OR_AMBIGUOUS) != 0
            or counts.get(NONZERO_POLICY_UPDATE, 0) < 1
            or set(counts) != set(all_classifications)
        ):
            raise RuntimeError(f"{arm} counted warmup-aware training gate failed")
        arms[arm] = {
            "summary_sha256": summary["summary_sha256"],
            "summary_file_sha256": file_sha256(path),
            "partial_evidence_sha256": partial["partial_evidence_sha256"],
            "partial_evidence_file_sha256": file_sha256(partial_path),
            "schedule_packet_sha256": summary["schedule_packet_sha256"],
            "schedule_manifest_sha256": summary["schedule_manifest_sha256"],
            "warmup_update_contract_sha256": summary["warmup_update_contract_sha256"],
            "scheduler_contract_sha256": summary["scheduler_contract_sha256"],
            "optimizer_steps": summary["optimizer_steps"],
            "groups": summary["groups"],
            "completions": summary["completions"],
            "reward": summary["reward"],
            "loss_trajectory": trajectory["loss_trajectory"],
            "kl_trajectory": trajectory["kl_trajectory"],
            "gradient_trajectory": trajectory["gradient_trajectory"],
            "learning_rate_trajectory": trajectory["learning_rate_trajectory"],
            "policy_state_trajectory": trajectory["policy_state_trajectory"],
            "update_classification_counts": counts,
            "counted_update_gate": gate,
            "trajectory_sha256": trajectory["trajectory_sha256"],
            "checkpoint_evidence": summary["checkpoint_evidence"],
            "final_adapter": summary["final_adapter"],
            "base_parameter_state_sha256": summary["base_parameter_state_sha256"],
            "reference_state_sha256": summary["final_reference"]["normalized_tensor_state_sha256"],
            "optimizer_ownership": summary["optimizer_ownership"],
            "warning_evidence_sha256": summary["warning_evidence"]["evidence_sha256"],
            "offline_reload_passed": summary["offline_reload_passed"],
            "runtime_seconds": summary["runtime_seconds"],
            "training_seconds": summary["training_seconds"],
            "peak_allocated_vram_bytes": summary["peak_allocated_vram_bytes"],
            "peak_reserved_vram_bytes": summary["peak_reserved_vram_bytes"],
            "physical_vram_bytes": summary["physical_vram_bytes"],
            "peak_process_rss_bytes": summary["peak_process_rss_bytes"],
            "output_disk_bytes": summary["output_disk_bytes"],
        }

    generic = cast(dict[str, Any], arms["generic"])
    targeted = cast(dict[str, Any], arms["targeted"])
    parity = {
        "groups_equal": generic["groups"] == targeted["groups"] == GROUPS_PER_ARM,
        "completions_equal": (
            generic["completions"] == targeted["completions"] == COMPLETIONS_PER_ARM
        ),
        "optimizer_steps_equal": (
            generic["optimizer_steps"] == targeted["optimizer_steps"] == GROUPS_PER_ARM
        ),
        "replay_groups_equal": paired["replay_groups_per_arm"] == 8,
        "checkpoint_positions_equal": (
            set(generic["checkpoint_evidence"])
            == set(targeted["checkpoint_evidence"])
            == {"8", "16", "32"}
        ),
        "maximum_completion_budget_equal": True,
        "prompt_token_parity_ratio": paired["prompt_token_parity_ratio"],
        "prompt_token_parity_passed": paired["prompt_token_parity_passed"],
    }
    parity_passed = (
        all(
            cast(bool, parity[key])
            for key in (
                "groups_equal",
                "completions_equal",
                "optimizer_steps_equal",
                "replay_groups_equal",
                "checkpoint_positions_equal",
                "maximum_completion_budget_equal",
                "prompt_token_parity_passed",
            )
        )
        and float(parity["prompt_token_parity_ratio"]) <= PROMPT_TOKEN_PARITY_MAXIMUM
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "training_id": "foundry-milestone14b-r2-counted-training-v1",
        "compatibility_sha256": compatibility["compatibility_sha256"],
        "warmup_update_contract_sha256": warmup["warmup_update_contract_sha256"],
        "arms": arms,
        "parity": parity,
        "both_arms_passed": parity_passed
        and all(cast(dict[str, Any], arms[arm])["offline_reload_passed"] is True for arm in ARMS),
        "sealed_content_use": 0,
    }
    if result["both_arms_passed"] is not True:
        raise RuntimeError("counted generic-targeted parity failed")
    result["training_result_sha256"] = canonical_sha256(result)
    return result


def write_counted_training_result(root: Path) -> dict[str, object]:
    result = build_counted_training_result(root)
    _write_new(root / TRACKED_ROOT / TRAINING_OUTPUT, result)
    return result


def _assessment(root: Path, checkpoint: int, arm: str, suite: str) -> dict[str, Any]:
    path = (
        root / RAW_ROOT / "development" / f"checkpoint-{checkpoint}/{arm}/{suite}/assessment.json"
    )
    value = _read(path)
    _verify(value, "summary_sha256")
    if value.get("backend_failures") != 0:
        raise RuntimeError("development retention backend failure")
    return value


def build_development_selection(root: Path) -> dict[str, object]:
    """Select the latest common passing checkpoint using development retention."""

    training = _read(root / TRACKED_ROOT / TRAINING_OUTPUT)
    _verify(training, "training_result_sha256")
    if training.get("both_arms_passed") is not True:
        raise ValueError("development retention is not training-authorized")
    rows: list[dict[str, object]] = []
    passing: list[int] = []
    for checkpoint in CHECKPOINTS:
        arm_passes: dict[str, bool] = {}
        by_arm: dict[str, object] = {}
        for arm in ARMS:
            suites = {
                suite: _assessment(root, checkpoint, arm, suite) for suite in DEVELOPMENT_SUITES
            }
            arm_passes[arm] = all(
                suites[suite]["gate_passed"] is True for suite in DEVELOPMENT_SUITES
            )
            by_arm[arm] = {
                suite: {
                    "preserved": suites[suite]["preserved"],
                    "total": suites[suite]["total"],
                    "overall_preservation": suites[suite]["overall_preservation"],
                    "wilson_lower_bound": suites[suite]["overall_wilson_95_lower_bound"],
                    "section_preservation": suites[suite]["section_preservation"],
                    "maximum_failure_family": suites[suite][
                        "maximum_instruction_family_adapter_only_failures"
                    ],
                    "prompt_echo": suites[suite]["prompt_echo"],
                    "question_generation": suites[suite]["question_generation"],
                    "backend_failures": suites[suite]["backend_failures"],
                    "gate_passed": suites[suite]["gate_passed"],
                    "assessment_sha256": suites[suite]["summary_sha256"],
                }
                for suite in DEVELOPMENT_SUITES
            }
        common = all(arm_passes.values())
        if common:
            passing.append(checkpoint)
        rows.append(
            {
                "checkpoint": checkpoint,
                "by_arm": by_arm,
                "arm_passes": arm_passes,
                "common_pass": common,
            }
        )
    selected = max(passing) if passing else None
    adapter_hashes: dict[str, str] = {}
    if selected is not None:
        for arm in ARMS:
            adapter_hashes[arm] = directory_sha256(
                root / RAW_ROOT / f"training/{arm}/checkpoint-{selected}/adapter"
            )
    result: dict[str, object] = {
        "schema_version": 1,
        "selection_id": ("foundry-milestone14b-r2-development-retention-selection-v1"),
        "training_result_sha256": training["training_result_sha256"],
        "evaluated_checkpoints": list(CHECKPOINTS),
        "evaluation_count": (len(CHECKPOINTS) * len(ARMS) * len(DEVELOPMENT_SUITES)),
        "results": rows,
        "selection_basis": ["adjudication", "anchor"],
        "excluded_selection_signals": [
            "grpo_training_loss",
            "grpo_reward",
            "holdout_v2",
            "gsm1k",
        ],
        "latest_common_passing_checkpoint": selected,
        "selected_adapter_sha256_by_arm": adapter_hashes,
        "development_retention_passed": selected is not None,
        "holdout_v2_authorized": selected is not None,
        "sealed_content_use": 0,
    }
    result["development_selection_sha256"] = canonical_sha256(result)
    return result


def write_development_selection(root: Path) -> dict[str, object]:
    result = build_development_selection(root)
    _write_new(root / TRACKED_ROOT / DEVELOPMENT_OUTPUT, result)
    return result


def build_holdout_decision(root: Path) -> dict[str, object]:
    """Apply the unchanged one-shot holdout-v2 gate to the selected pair."""

    selection = _read(root / TRACKED_ROOT / DEVELOPMENT_OUTPUT)
    _verify(selection, "development_selection_sha256")
    checkpoint = selection.get("latest_common_passing_checkpoint")
    if not isinstance(checkpoint, int):
        raise ValueError("holdout v2 is unauthorized without a common checkpoint")
    arms: dict[str, object] = {}
    passes: list[bool] = []
    for arm in ARMS:
        path = root / RAW_ROOT / f"holdout_v2/{arm}/assessment.json"
        assessment = _read(path)
        _verify(assessment, "summary_sha256")
        expected_adapter = selection["selected_adapter_sha256_by_arm"][arm]
        if assessment.get("adapter_sha256") != expected_adapter:
            raise ValueError("holdout adapter differs from development selection")
        passed = assessment.get("gate_passed") is True
        passes.append(passed)
        arms[arm] = {
            "adapter_sha256": assessment["adapter_sha256"],
            "preserved": assessment["preserved"],
            "total": assessment["total"],
            "overall_preservation": assessment["overall_preservation"],
            "section_preservation": assessment["section_preservation"],
            "maximum_failure_family": assessment[
                "maximum_instruction_family_adapter_only_failures"
            ],
            "prompt_echo": assessment["prompt_echo"],
            "question_generation": assessment["question_generation"],
            "backend_failures": assessment["backend_failures"],
            "gate_passed": passed,
            "starting_l3_preserved": STARTING_HOLDOUT_CORRECT[arm],
            "change_vs_starting_l3": (int(assessment["preserved"]) - STARTING_HOLDOUT_CORRECT[arm]),
            "assessment_sha256": assessment["summary_sha256"],
        }
    both = all(passes)
    result: dict[str, object] = {
        "schema_version": 1,
        "decision_id": "foundry-milestone14b-r2-holdout-v2-decision-v1",
        "selected_checkpoint": checkpoint,
        "development_selection_sha256": selection["development_selection_sha256"],
        "frozen_subset_sha256": (
            "a23b1014d92e9f98b74da3b29913a430bdaebf8e07a16b31b4c3dcc831f1f420"
        ),
        "adapter_evaluations": 2,
        "evaluated_exactly_once": True,
        "arms": arms,
        "both_arms_pass": both,
        "gsm1k_authorized": both,
        "holdout_used_for_checkpoint_selection": False,
        "holdout_previously_seen_by_starting_sft_policies": True,
        "sealed_content_use": 0,
    }
    result["holdout_decision_sha256"] = canonical_sha256(result)
    return result


def write_holdout_decision(root: Path) -> dict[str, object]:
    result = build_holdout_decision(root)
    _write_new(root / TRACKED_ROOT / HOLDOUT_OUTPUT, result)
    return result


def build_gsm1k_analysis(root: Path) -> dict[str, object]:
    """Run the predeclared paired analysis and unchanged ten-check signal gate."""

    tracked = root / TRACKED_ROOT
    holdout = _read(tracked / HOLDOUT_OUTPUT)
    selection = _read(tracked / DEVELOPMENT_OUTPUT)
    paired_schedule = _read(tracked / "milestone14a_paired_schedule.json")
    _verify(holdout, "holdout_decision_sha256")
    _verify(selection, "development_selection_sha256")
    _verify(paired_schedule, "paired_schedule_sha256")
    if holdout.get("gsm1k_authorized") is not True:
        raise ValueError("GSM1K analysis is unauthorized")
    summaries = {arm: _read(tracked / f"milestone14b_r2_gsm1k_{arm}.json") for arm in ARMS}
    prediction_paths = {
        "base": (root / "results/raw/development_baseline/qwen2_5_1_5b/raw/predictions.jsonl"),
        "starting_generic": (
            root
            / "results/raw/phase2_vetted_corpus/milestone13e/gsm1k/generic"
            / "output/raw/predictions.jsonl"
        ),
        "starting_targeted": (
            root
            / "results/raw/phase2_vetted_corpus/milestone13e/gsm1k/targeted"
            / "output/raw/predictions.jsonl"
        ),
        "grpo_generic": (root / RAW_ROOT / "gsm1k/generic/output/raw/predictions.jsonl"),
        "grpo_targeted": (root / RAW_ROOT / "gsm1k/targeted/output/raw/predictions.jsonl"),
    }
    models = {name: _predictions(path) for name, path in prediction_paths.items()}
    if len({frozenset(values) for values in models.values()}) != 1:
        raise ValueError("GSM1K model packets have different stable-ID sets")
    if sum(models["base"].values()) != BASE_CORRECT:
        raise ValueError("frozen base GSM1K packet differs")
    for arm in ARMS:
        if sum(models[f"starting_{arm}"].values()) != STARTING_CORRECT[arm]:
            raise ValueError(f"starting {arm} GSM1K packet differs")
        summary = summaries[arm]
        if (
            summary.get("processed_examples") != GSM1K_EXAMPLES
            or summary.get("generation_failures") != 0
            or summary.get("adapter_sha256") != selection["selected_adapter_sha256_by_arm"][arm]
            or summary.get("correct_examples") != sum(models[f"grpo_{arm}"].values())
        ):
            raise RuntimeError(f"{arm} GSM1K completeness or identity gate failed")
    taxonomy_path = root / "results/raw/failure_taxonomy/current/classifications.jsonl"
    taxonomy = _taxonomy(taxonomy_path, set(models["base"]))
    category_effects, untargeted = _category_effects(taxonomy, models)
    ids = sorted(models["base"])
    differences = tuple(
        int(models["grpo_targeted"][item]) - int(models["grpo_generic"][item]) for item in ids
    )
    bootstrap = paired_bootstrap_interval(
        differences,
        replicates=10_000,
        seed=20260720,
    )
    generic_correct = sum(models["grpo_generic"].values())
    targeted_correct = sum(models["grpo_targeted"].values())
    checks = {
        "targeted_correct_greater_than_521": targeted_correct > BASE_CORRECT,
        "targeted_correct_greater_than_generic": (targeted_correct > generic_correct),
        "targeted_extractability_at_least_91_38_percent": (
            float(summaries["targeted"]["extractable_answer_rate"])
            >= MINIMUM_TARGETED_EXTRACTABILITY
        ),
        "zero_backend_failures": all(
            int(summaries[arm]["generation_failures"]) == 0 for arm in ARMS
        ),
        "untargeted_decline_no_greater_than_two_points": untargeted["gate_passed"],
        "development_retention_passed": selection["development_retention_passed"],
        "holdout_v2_retention_passed": holdout["both_arms_pass"],
        "exact_group_and_completion_parity": (
            paired_schedule["exact_group_and_completion_parity"] is True
            and paired_schedule["groups_per_arm"] == GROUPS_PER_ARM
            and paired_schedule["completions_per_arm"] == COMPLETIONS_PER_ARM
        ),
        "prompt_token_difference_zero_percent": (
            float(paired_schedule["prompt_token_parity_ratio"]) == 0.0
            and float(paired_schedule["prompt_token_parity_ratio"]) <= PROMPT_TOKEN_PARITY_MAXIMUM
        ),
        "frozen_inputs_and_contracts": (
            paired_schedule["model_outputs_observed_during_scheduling"] is False
            and paired_schedule["gsm1k_prompt_use"] == 0
            and paired_schedule["holdout_v2_prompt_use"] == 0
            and paired_schedule["sealed_content_use"] == 0
        ),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "analysis_id": "foundry-milestone14b-r2-paired-signal-analysis-v1",
        "examples": GSM1K_EXAMPLES,
        "frozen_comparisons": {
            "base_correct": BASE_CORRECT,
            "starting_generic_correct": STARTING_CORRECT["generic"],
            "starting_targeted_correct": STARTING_CORRECT["targeted"],
        },
        "generic_grpo": {
            "correct": generic_correct,
            "accuracy": summaries["generic"]["accuracy"],
            "extractable": summaries["generic"]["extractable_examples"],
            "extractability": summaries["generic"]["extractable_answer_rate"],
            "exact_format": summaries["generic"]["exact_format_compliant_examples"],
            "extractable_but_wrong": summaries["generic"]["extractable_incorrect_examples"],
            "unextractable": summaries["generic"]["unextractable_examples"],
            "truncated": summaries["generic"]["truncated_examples"],
            "backend_failures": summaries["generic"]["generation_failures"],
        },
        "targeted_grpo": {
            "correct": targeted_correct,
            "accuracy": summaries["targeted"]["accuracy"],
            "extractable": summaries["targeted"]["extractable_examples"],
            "extractability": summaries["targeted"]["extractable_answer_rate"],
            "exact_format": summaries["targeted"]["exact_format_compliant_examples"],
            "extractable_but_wrong": summaries["targeted"]["extractable_incorrect_examples"],
            "unextractable": summaries["targeted"]["unextractable_examples"],
            "truncated": summaries["targeted"]["truncated_examples"],
            "backend_failures": summaries["targeted"]["generation_failures"],
        },
        "paired_generic_vs_targeted": _paired(
            models["grpo_generic"],
            models["grpo_targeted"],
        ),
        "targeted_vs_starting_targeted": _paired(
            models["starting_targeted"],
            models["grpo_targeted"],
        ),
        "generic_vs_starting_generic": _paired(
            models["starting_generic"],
            models["grpo_generic"],
        ),
        "targeted_vs_untouched_base": _paired(
            models["base"],
            models["grpo_targeted"],
        ),
        "category_effects": category_effects,
        "aggregate_untargeted_change": untargeted,
        "paired_bootstrap": {
            "replicates": 10_000,
            "seed": 20260720,
            "estimand": "targeted_minus_generic_accuracy",
            "observed": fmean(differences),
            "percentile_95_interval": list(bootstrap),
        },
        "signal_checks": checks,
        "signal_gate_passed": all(bool(value) for value in checks.values()),
        "input_file_sha256": {
            **{name: file_sha256(path) for name, path in prediction_paths.items()},
            "taxonomy": file_sha256(taxonomy_path),
            **{
                f"{arm}_summary": file_sha256(tracked / f"milestone14b_r2_gsm1k_{arm}.json")
                for arm in ARMS
            },
        },
        "second_seed_started": False,
        "sealed_final_accessed": False,
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def write_gsm1k_analysis(root: Path) -> dict[str, object]:
    result = build_gsm1k_analysis(root)
    _write_new(root / TRACKED_ROOT / GSM1K_ANALYSIS_OUTPUT, result)
    return result
