"""Content-free gate analysis for Milestone 14A."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from foundry.phase2.l3_grpo_contract import FIXED_LIBRARY_NOTICE_CLASSES
from foundry.phase2.l3_grpo_schedule import (
    COMPLETIONS_PER_ARM,
    GROUPS_PER_ARM,
    PROMPT_TOKEN_PARITY_MAXIMUM,
)
from foundry.training.config import canonical_sha256
from foundry.training.paired_analysis import paired_bootstrap_interval
from foundry.training.qlora import directory_sha256, file_sha256

CHECKPOINTS = (8, 16, 32)
ARMS = ("generic", "targeted")
DEVELOPMENT_SUITES = ("adjudication", "anchor")
BASE_CORRECT = 521
STARTING_CORRECT = {"generic": 517, "targeted": 519}
STARTING_HOLDOUT_CORRECT = {"generic": 315, "targeted": 315}
GSM1K_EXAMPLES = 814
MINIMUM_TARGETED_EXTRACTABILITY = 0.9138
MAXIMUM_UNTARGETED_DECLINE = 0.02
TARGETED_CATEGORIES = frozenset(
    {
        "multi_step_bookkeeping_or_omission",
        "rate_ratio_percentage_or_average",
        "constraint_distribution_or_discrete_reasoning",
    }
)


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


def _notice_evidence(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    classes = [
        {
            **notice,
            "count": sum(notice["required_substring"] in line for line in lines),
        }
        for notice in FIXED_LIBRARY_NOTICE_CLASSES
    ]
    payload: dict[str, object] = {
        "stderr_file_sha256": file_sha256(path),
        "stderr_bytes": path.stat().st_size,
        "stderr_line_count": len(lines),
        "predeclared_classes": classes,
        "broad_warning_suppression": False,
        "generation_warning_recorded_separately": True,
    }
    payload["notice_evidence_sha256"] = canonical_sha256(payload)
    return payload


def build_compatibility_result(root: Path) -> dict[str, object]:
    """Require exact official smoke equality and publish one shared packet."""

    raw = root / "results/raw/phase2_vetted_corpus/milestone14a"
    summaries = [_read(raw / f"compatibility/run-{index}/summary.json") for index in (1, 2)]
    for summary in summaries:
        _verify(summary, "summary_sha256")
        if (
            summary.get("gate_passed") is not True
            or summary.get("optimizer_steps") != 2
            or summary.get("completions") != 8
            or summary.get("arm") != "generic"
        ):
            raise RuntimeError("official compatibility smoke gate failed")
    packets = [cast(dict[str, Any], summary["exact_packet"]) for summary in summaries]
    if packets[0] != packets[1]:
        raise RuntimeError("official compatibility smoke exact packets differ")
    if packets[0].get("packet_sha256") != canonical_sha256(
        {key: value for key, value in packets[0].items() if key != "packet_sha256"}
    ):
        raise ValueError("official compatibility packet self-hash differs")
    notices = [
        _notice_evidence(raw / f"compatibility/logs/run-{index}.stderr.txt") for index in (1, 2)
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "compatibility_id": "foundry-l3-verifier-grpo-compatibility-v1",
        "official_smoke_runs": 2,
        "official_smoke_retries": 0,
        "fresh_processes": True,
        "arm": "generic",
        "optimizer_steps_per_run": 2,
        "completions_per_run": 8,
        "one_task_and_one_replay_group_per_run": True,
        "exact_packet": packets[0],
        "exact_packet_sha256": packets[0]["packet_sha256"],
        "run_summary_sha256s": [summary["summary_sha256"] for summary in summaries],
        "run_summary_file_sha256s": [
            file_sha256(raw / f"compatibility/run-{index}/summary.json") for index in (1, 2)
        ],
        "fixed_library_notices": notices,
        "peak_reserved_vram_bytes": [summary["peak_reserved_vram_bytes"] for summary in summaries],
        "peak_process_rss_bytes": [summary["peak_process_rss_bytes"] for summary in summaries],
        "runtime_seconds": [summary["runtime_seconds"] for summary in summaries],
        "exact_match": True,
        "gate_passed": True,
        "sealed_content_use": 0,
    }
    result["compatibility_sha256"] = canonical_sha256(result)
    return result


def write_compatibility_result(root: Path) -> dict[str, object]:
    result = build_compatibility_result(root)
    _write_new(
        root / "results/phase2_vetted_corpus/milestone14a_compatibility.json",
        result,
    )
    return result


def build_counted_training_result(root: Path) -> dict[str, object]:
    """Project both successful raw training summaries into tracked evidence."""

    paired = _read(root / "results/phase2_vetted_corpus/milestone14a_paired_schedule.json")
    _verify(paired, "paired_schedule_sha256")
    arms: dict[str, object] = {}
    for arm in ARMS:
        path = root / f"results/raw/phase2_vetted_corpus/milestone14a/training/{arm}_summary.json"
        summary = _read(path)
        _verify(summary, "summary_sha256")
        trajectory = cast(dict[str, Any], summary["trajectory"])
        if (
            summary.get("gate_passed") is not True
            or summary.get("groups") != GROUPS_PER_ARM
            or summary.get("completions") != COMPLETIONS_PER_ARM
            or summary.get("optimizer_steps") != GROUPS_PER_ARM
            or summary.get("base_unchanged") is not True
            or summary.get("reference_unchanged") is not True
            or summary.get("policy_updated") is not True
            or set(cast(dict[str, object], summary["checkpoint_evidence"])) != {"8", "16", "32"}
        ):
            raise RuntimeError(f"{arm} counted training gate failed")
        arms[arm] = {
            "summary_sha256": summary["summary_sha256"],
            "summary_file_sha256": file_sha256(path),
            "schedule_packet_sha256": summary["schedule_packet_sha256"],
            "schedule_manifest_sha256": summary["schedule_manifest_sha256"],
            "optimizer_steps": summary["optimizer_steps"],
            "groups": summary["groups"],
            "completions": summary["completions"],
            "reward": summary["reward"],
            "loss_trajectory": trajectory["loss_trajectory"],
            "kl_trajectory": trajectory["kl_trajectory"],
            "gradient_trajectory": trajectory["gradient_trajectory"],
            "learning_rate_trajectory": trajectory["learning_rate_trajectory"],
            "policy_state_trajectory": trajectory["policy_state_trajectory"],
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
                "maximum_completion_budget_equal",
                "prompt_token_parity_passed",
            )
        )
        and float(parity["prompt_token_parity_ratio"]) <= PROMPT_TOKEN_PARITY_MAXIMUM
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "training_id": "foundry-milestone14a-counted-training-v1",
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
    _write_new(
        root / "results/phase2_vetted_corpus/milestone14a_counted_training.json",
        result,
    )
    return result


def _assessment(root: Path, checkpoint: int, arm: str, suite: str) -> dict[str, Any]:
    path = (
        root
        / "results/raw/phase2_vetted_corpus/milestone14a/development"
        / f"checkpoint-{checkpoint}/{arm}/{suite}/assessment.json"
    )
    value = _read(path)
    _verify(value, "summary_sha256")
    if value.get("backend_failures") != 0:
        raise RuntimeError("development retention backend failure")
    return value


def build_development_selection(root: Path) -> dict[str, object]:
    """Evaluate the already-produced six checkpoints without using reward or loss."""

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
                root
                / f"results/raw/phase2_vetted_corpus/milestone14a/training/{arm}"
                / f"checkpoint-{selected}/adapter"
            )
    result: dict[str, object] = {
        "schema_version": 1,
        "selection_id": "foundry-milestone14a-development-retention-selection-v1",
        "evaluated_checkpoints": list(CHECKPOINTS),
        "evaluation_count": len(CHECKPOINTS) * len(ARMS) * len(DEVELOPMENT_SUITES),
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
    _write_new(
        root / "results/phase2_vetted_corpus/milestone14a_development_selection.json",
        result,
    )
    return result


def build_holdout_decision(root: Path) -> dict[str, object]:
    """Apply the unchanged one-shot holdout-v2 gate to the selected pair."""

    selection = _read(root / "results/phase2_vetted_corpus/milestone14a_development_selection.json")
    _verify(selection, "development_selection_sha256")
    checkpoint = selection.get("latest_common_passing_checkpoint")
    if not isinstance(checkpoint, int):
        raise ValueError("holdout v2 is unauthorized without a common checkpoint")
    arms: dict[str, object] = {}
    passes: list[bool] = []
    for arm in ARMS:
        path = (
            root
            / f"results/raw/phase2_vetted_corpus/milestone14a/holdout_v2/{arm}"
            / "assessment.json"
        )
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
        "decision_id": "foundry-milestone14a-holdout-v2-decision-v1",
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
    _write_new(
        root / "results/phase2_vetted_corpus/milestone14a_holdout_v2_decision.json",
        result,
    )
    return result


def _predictions(path: Path) -> dict[str, bool]:
    values: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row: object = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("GSM1K prediction row must be an object")
        stable_id = row.get("stable_id")
        correct = row.get("correct")
        if (
            not isinstance(stable_id, str)
            or len(stable_id) != 64
            or not isinstance(correct, bool)
            or stable_id in values
        ):
            raise ValueError("GSM1K prediction identity or correctness differs")
        values[stable_id] = correct
    if len(values) != GSM1K_EXAMPLES:
        raise ValueError("GSM1K prediction packet must contain 814 unique rows")
    return values


def _taxonomy(path: Path, valid_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row: object = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("taxonomy row must be an object")
        stable_id = row.get("stable_id")
        category = row.get("primary_category")
        if (
            not isinstance(stable_id, str)
            or stable_id not in valid_ids
            or not isinstance(category, str)
            or not category
            or stable_id in result
        ):
            raise ValueError("frozen taxonomy identity differs")
        result[stable_id] = category
    if len(result) != 293:
        raise ValueError("frozen failure taxonomy must contain 293 base failures")
    return result


def _paired(left: Mapping[str, bool], right: Mapping[str, bool]) -> dict[str, int]:
    ids = sorted(left)
    if set(ids) != set(right):
        raise ValueError("paired GSM1K stable-ID sets differ")
    left_wins = sum(left[item] and not right[item] for item in ids)
    right_wins = sum(right[item] and not left[item] for item in ids)
    return {
        "left_wins": left_wins,
        "right_wins": right_wins,
        "net_right_advantage": right_wins - left_wins,
        "both_correct": sum(left[item] and right[item] for item in ids),
        "both_wrong": sum(not left[item] and not right[item] for item in ids),
    }


def _category_effects(
    taxonomy: Mapping[str, str],
    models: Mapping[str, Mapping[str, bool]],
) -> tuple[dict[str, object], dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for stable_id, category in taxonomy.items():
        grouped[category].append(stable_id)
    categories: dict[str, object] = {}
    for category, ids in sorted(grouped.items()):
        categories[category] = {
            "examples": len(ids),
            **{
                f"{name}_correct": sum(values[item] for item in ids)
                for name, values in models.items()
            },
        }
    untargeted_ids = [
        stable_id for stable_id, category in taxonomy.items() if category not in TARGETED_CATEGORIES
    ]
    starting = sum(models["starting_targeted"][item] for item in untargeted_ids)
    current = sum(models["grpo_targeted"][item] for item in untargeted_ids)
    delta = (current - starting) / len(untargeted_ids)
    untargeted: dict[str, object] = {
        "definition": (
            "frozen base-failure taxonomy primary categories outside the three "
            "targeted curriculum families"
        ),
        "examples": len(untargeted_ids),
        "starting_targeted_correct": starting,
        "grpo_targeted_correct": current,
        "accuracy_change_vs_starting_targeted": delta,
        "maximum_allowed_decline": MAXIMUM_UNTARGETED_DECLINE,
        "gate_passed": delta >= -MAXIMUM_UNTARGETED_DECLINE,
    }
    return categories, untargeted


def build_gsm1k_analysis(root: Path) -> dict[str, object]:
    """Run the predeclared paired analysis and exact ten-check signal gate."""

    holdout = _read(root / "results/phase2_vetted_corpus/milestone14a_holdout_v2_decision.json")
    _verify(holdout, "holdout_decision_sha256")
    if holdout.get("gsm1k_authorized") is not True:
        raise ValueError("GSM1K analysis is unauthorized")
    selection = _read(root / "results/phase2_vetted_corpus/milestone14a_development_selection.json")
    _verify(selection, "development_selection_sha256")
    paired_schedule = _read(root / "results/phase2_vetted_corpus/milestone14a_paired_schedule.json")
    _verify(paired_schedule, "paired_schedule_sha256")
    summaries = {
        arm: _read(root / f"results/phase2_vetted_corpus/milestone14a_gsm1k_{arm}.json")
        for arm in ARMS
    }
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
        "grpo_generic": (
            root
            / "results/raw/phase2_vetted_corpus/milestone14a/gsm1k/generic"
            / "output/raw/predictions.jsonl"
        ),
        "grpo_targeted": (
            root
            / "results/raw/phase2_vetted_corpus/milestone14a/gsm1k/targeted"
            / "output/raw/predictions.jsonl"
        ),
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
    bootstrap = paired_bootstrap_interval(differences, replicates=10_000, seed=20260720)
    generic_correct = sum(models["grpo_generic"].values())
    targeted_correct = sum(models["grpo_targeted"].values())
    checks = {
        "targeted_correct_greater_than_521": targeted_correct > BASE_CORRECT,
        "targeted_correct_greater_than_generic": targeted_correct > generic_correct,
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
        "prompt_token_difference_at_most_one_percent": (
            float(paired_schedule["prompt_token_parity_ratio"]) <= PROMPT_TOKEN_PARITY_MAXIMUM
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
        "analysis_id": "foundry-milestone14a-paired-signal-analysis-v1",
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
        "paired_generic_vs_targeted": _paired(models["grpo_generic"], models["grpo_targeted"]),
        "targeted_vs_starting_targeted": _paired(
            models["starting_targeted"], models["grpo_targeted"]
        ),
        "generic_vs_starting_generic": _paired(models["starting_generic"], models["grpo_generic"]),
        "targeted_vs_untouched_base": _paired(models["base"], models["grpo_targeted"]),
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
                f"{arm}_summary": file_sha256(
                    root / f"results/phase2_vetted_corpus/milestone14a_gsm1k_{arm}.json"
                )
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
    _write_new(
        root / "results/phase2_vetted_corpus/milestone14a_gsm1k_analysis.json",
        result,
    )
    return result
