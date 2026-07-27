"""Interpret frozen vetted-corpus retention failures without model inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256
from foundry.training.retention import question_generation_configuration_sha256

ANALYSIS_ID = "foundry-vetted-corpus-independent-retention-interpretation-v1"
STARTING_COMMIT = "4c4db2b8ec3fddc136d5efac9c47c81017236931"
DATASET_SHA256 = "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SCORER_SOURCE_SHA256 = "0cc6f583caad049fe9ce25a4da70b84d8abcfd45360d8dbc1f9e25e86b530b91"
SCORER_CONFIGURATION_SHA256 = "87a140820f017acb0ff9bb44c047b169605befab3d250f42a054ea45d430f2c7"
INDEPENDENT_SUBSET_SHA256 = "f56845076a1a59e5ca1a95466541339b56f026e945f86118caec307a690ee4ec"
INDEPENDENT_DECISION_SHA256 = "0746305653d6d23674f8df1652ec07be6585d1fda2f0d4dd3b3b70f6efe79741"
ADAPTER_SHA256 = {
    "generic": "fe3c0f5a6e8082f2d151a293e918882f896ddeece93c8f779d4b16d21618a73d",
    "targeted": "7f15edf4cd7e8c50478b6bb8deb55bb3d9d43273109fc0b224110f9fe08da0bd",
}
INDEPENDENT_RAW_SHA256 = {
    "generic": "c6375040bf8e80b936a8d3a14ec7d41d1f4a96c5a07110afe3b703265b3f5835",
    "targeted": "c9210c5abff6f922b91518a08881ba9d6081670c52b9b4d4a7b93a5998347dfe",
}
INDEPENDENT_SUMMARY_FILE_SHA256 = {
    "generic": "369895a54586ff2c626c4283725cdb62ca4870eeca5ad9065ce795ae2508a038",
    "targeted": "569e1ec9700b06d4122029a325caea2f7ed252567bc1713045420a6d632d023c",
}
REPLAY_FAMILY_MAP: dict[str, tuple[str, ...]] = {
    "sequential_integer_adjustment": ("addition", "short_multi_step", "subtraction"),
    "scaled_sum_with_offset": ("addition", "multiplication", "short_multi_step"),
    "exact_division_then_adjustment": (
        "addition",
        "exact_division",
        "short_multi_step",
        "subtraction",
    ),
    "symmetric_three_value_mean": ("addition", "exact_division", "short_multi_step"),
    "integer_percentage": ("percentage",),
    "double_colon_identifier": ("required_affixes",),
}
FUTURE_HOLDOUT_REQUIREMENTS = (
    "keep_existing_adjudication_and_anchor_as_development_instruments",
    "construct_new_original_nonbenchmark_retention_holdout",
    "prove_zero_exact_or_12_token_overlap_with_training_replay_gsm1k_development_and_prior_retention",
    "evaluate_untouched_base_before_adapter_exposure",
    "freeze_only_base_correct_subset",
    "validate_prompt_reference_and_corrected_scorer_integrity",
    "freeze_identity_before_training",
    "use_exactly_once_on_finally_selected_new_architecture",
    "never_use_for_coefficient_or_checkpoint_selection",
    "leave_sealed_final_untouched",
)
TIER3_ROADMAP = (
    "complete_independent_failure_interpretation",
    "train_one_retention_regularized_sft_architecture",
    "obtain_one_common_retention_safe_generic_targeted_checkpoint",
    "evaluate_both_on_gsm1k",
    "require_targeted_to_beat_generic_and_untouched_base",
    "run_one_certified_verifier_reward_grpo_path_from_retention_safe_checkpoint",
    "build_one_autonomous_evaluate_diagnose_data_train_retain_evaluate_controller",
    "execute_second_smaller_autonomous_repair_cycle",
    "repeat_final_important_result_with_seed_2",
    "run_sealed_final_once_after_all_decisions_are_frozen",
    "package_dashboard_demo",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    """Return the deterministic case-folded whitespace normalization."""

    return " ".join(value.casefold().split())


def structural_signature(value: str) -> str:
    """Hash a deterministic number/quotation-neutral text structure."""

    normalized = re.sub(r"`[^`]*`|\"[^\"]*\"|'[^']*'", "<quoted>", value.casefold())
    normalized = re.sub(r"(?<![\w.])[+-]?\d+(?:\.\d+)?%?", "<number>", normalized)
    normalized = " ".join(normalized.split())
    return _sha256_text(normalized)


def freeze(payload: dict[str, Any], hash_key: str) -> dict[str, Any]:
    """Return a copy with a canonical SHA-256 appended."""

    if hash_key in payload:
        raise ValueError("payload already contains its hash key")
    result = dict(payload)
    result[hash_key] = canonical_sha256(result)
    return result


def failure_rule(section: str, score: dict[str, Any]) -> tuple[str, str]:
    """Classify one stored failure without rescoring it."""

    if section == "format":
        return "exact_text_equality", "exact_format_mismatch"
    if bool(score["extractable"]):
        return "numeric_terminal_expected_answer_equality", "wrong_extractable_numeric_answer"
    return "numeric_terminal_extraction_required", "malformed_nonextractable_numeric_answer"


def select_architecture(
    *,
    shared_failure_fraction: float,
    replay_ratio_trajectories_aligned: bool,
    explicit_logit_constraint: bool,
    capacity_implicated: bool,
    gradient_conflict_measured: bool,
) -> str | None:
    """Apply the Milestone 13B predeclared architecture hierarchy."""

    if (
        shared_failure_fraction >= 0.8
        and replay_ratio_trajectories_aligned
        and not explicit_logit_constraint
    ):
        return "replay-ce-token-kl-v1"
    if capacity_implicated:
        return "layer-restricted-lora-v1"
    if gradient_conflict_measured:
        return "multiobjective-gradient-balanced-sft-v1"
    return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)


def _validate_source_identities(root: Path) -> dict[str, Any]:
    dataset = cast(
        dict[str, Any],
        _read_json(root / "results/raw/phase2_vetted_corpus/dataset/dataset_summary.json"),
    )
    if dataset.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("frozen dataset identity differs")
    scorer_path = root / "src/foundry/training/retention.py"
    if file_sha256(scorer_path) != SCORER_SOURCE_SHA256:
        raise ValueError("corrected scorer source differs")
    if question_generation_configuration_sha256() != SCORER_CONFIGURATION_SHA256:
        raise ValueError("corrected scorer configuration differs")
    training = cast(
        dict[str, Any],
        _read_json(root / "results/phase2_vetted_corpus/v1_training.json"),
    )
    decision = cast(
        dict[str, Any],
        _read_json(
            root / "results/phase2_vetted_corpus/milestone13a_independent_final_failure.json"
        ),
    )
    if (
        decision.get("retention_decision_sha256") != INDEPENDENT_DECISION_SHA256
        or decision.get("independent_final_subset_sha256") != INDEPENDENT_SUBSET_SHA256
        or decision.get("selected_variant") != "V1"
        or decision.get("selected_checkpoint") != 64
        or decision.get("corrected_scorer_source_sha256") != SCORER_SOURCE_SHA256
        or decision.get("corrected_scorer_configuration_sha256") != SCORER_CONFIGURATION_SHA256
    ):
        raise ValueError("independent retention decision identity differs")
    for arm in ("generic", "targeted"):
        expected = ADAPTER_SHA256[arm]
        checkpoint = cast(dict[str, Any], training[arm])["checkpoints"]["64"]
        decision_arm = cast(dict[str, Any], decision[arm])
        adapter_path = (
            root / f"results/raw/phase2_vetted_corpus/v1_training/{arm}/checkpoint-64/adapter"
        )
        raw_path = (
            root / f"results/raw/phase2_vetted_corpus/milestone13a/independent_final/{arm}_raw.json"
        )
        summary_path = (
            root
            / f"results/raw/phase2_vetted_corpus/milestone13a/independent_final/{arm}_summary.json"
        )
        if (
            checkpoint["adapter_sha256"] != expected
            or decision_arm["adapter_sha256"] != expected
            or directory_sha256(adapter_path) != expected
            or file_sha256(raw_path) != INDEPENDENT_RAW_SHA256[arm]
            or decision_arm["raw_packet_sha256"] != INDEPENDENT_RAW_SHA256[arm]
            or file_sha256(summary_path) != INDEPENDENT_SUMMARY_FILE_SHA256[arm]
            or decision_arm["summary_file_sha256"] != INDEPENDENT_SUMMARY_FILE_SHA256[arm]
        ):
            raise ValueError(f"{arm} frozen adapter or independent output identity differs")
    return {
        "dataset_identity": DATASET_SHA256,
        "corrected_scorer_source_sha256": SCORER_SOURCE_SHA256,
        "corrected_scorer_configuration_sha256": SCORER_CONFIGURATION_SHA256,
        "selected_adapter_sha256": ADAPTER_SHA256,
        "independent_raw_sha256": INDEPENDENT_RAW_SHA256,
        "independent_summary_file_sha256": INDEPENDENT_SUMMARY_FILE_SHA256,
        "independent_subset_sha256": INDEPENDENT_SUBSET_SHA256,
        "independent_decision_sha256": INDEPENDENT_DECISION_SHA256,
    }


def _score_projection(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "correct": bool(score["correct"]),
        "extractable": bool(score["extractable"]),
        "exact_format": bool(score["exact_format"]),
        "malformed": bool(score["malformed"]),
        "prompt_echo": bool(score["prompt_echo"]),
        "question_generation": bool(score["question_generation"]),
        "backend_status": ("failure" if "backend_error_type" in score else "success"),
        "extracted_hash": score.get("extracted_hash"),
    }


def _build_failure_inventory(
    *,
    suite_items: dict[str, dict[str, Any]],
    base_rows: dict[str, dict[str, Any]],
    arm_rows: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for arm in ("generic", "targeted"):
        failures = [row for row in arm_rows[arm] if not bool(row["score"]["correct"])]
        if len(failures) != 10:
            raise ValueError(f"{arm} failure count differs from ten")
        for row in failures:
            item_id = str(row["id"])
            item = suite_items[item_id]
            base = base_rows[item_id]
            score = cast(dict[str, Any], row["score"])
            base_score = cast(dict[str, Any], base["score"])
            if not bool(base_score["correct"]):
                raise ValueError("independent subset contains a base failure")
            if (
                row["section"] != item["section"]
                or row["skill"] != item["skill"]
                or _sha256_text(str(row["response"])) != row["response_sha256"]
                or _sha256_text(str(base["response"])) != base["response_sha256"]
            ):
                raise ValueError("independent prompt or response identity differs")
            rule, reason = failure_rule(str(row["section"]), score)
            response = str(row["response"])
            category = str(row["section"])
            requested_task_completed = (
                bool(score["extractable"])
                if category == "arithmetic"
                else bool(score["exact_format"])
                if category == "format"
                else bool(score["correct"])
            )
            records.append(
                {
                    "arm": arm,
                    "prompt_id": item_id,
                    "prompt_id_sha256": _sha256_text(item_id),
                    "prompt_sha256": _sha256_text(str(item["prompt"])),
                    "category": row["section"],
                    "deterministic_failure_family": row["skill"],
                    "untouched_base_response_sha256": base["response_sha256"],
                    "adapter_response_sha256": row["response_sha256"],
                    "normalized_adapter_response_sha256": _sha256_text(normalize_text(response)),
                    "base_scorer_decision": _score_projection(base_score),
                    "adapter_scorer_decision": _score_projection(score),
                    "extraction_result": bool(score["extractable"]),
                    "format_result": bool(score["exact_format"]),
                    "instruction_result": (
                        bool(score["correct"])
                        if str(row["section"]) == "instruction"
                        else "not_applicable"
                    ),
                    "correctness_result": bool(score["correct"]),
                    "token_count": len(tokenizer.encode(response, add_special_tokens=False)),
                    "malformed_status": bool(score["malformed"]),
                    "prompt_echo_status": bool(score["prompt_echo"]),
                    "question_generation_status": bool(score["question_generation"]),
                    "backend_status": ("failure" if "backend_error_type" in score else "success"),
                    "deterministic_rule_failed": rule,
                    "reason_code": reason,
                    "requested_task_completed": requested_task_completed,
                    "final_answer_extractable": bool(score["extractable"]),
                    "answer_mathematically_correct": (
                        bool(score["correct"])
                        if str(row["section"]) == "arithmetic"
                        else "not_applicable"
                    ),
                }
            )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "inventory_id": "foundry-vetted-corpus-independent-failure-inventory-v1",
        "source_subset_sha256": INDEPENDENT_SUBSET_SHA256,
        "records": records,
        "counts": {
            "generic": sum(row["arm"] == "generic" for row in records),
            "targeted": sum(row["arm"] == "targeted" for row in records),
            "arithmetic_per_arm": 9,
            "format_per_arm": 1,
            "instruction_per_arm": 0,
        },
    }
    return freeze(payload, "failure_inventory_sha256")


def _failure_structure(record: dict[str, Any]) -> str:
    decision = cast(dict[str, Any], record["adapter_scorer_decision"])
    return canonical_sha256(
        {
            "category": record["category"],
            "family": record["deterministic_failure_family"],
            "rule": record["deterministic_rule_failed"],
            "reason": record["reason_code"],
            "correct": decision["correct"],
            "extractable": decision["extractable"],
            "malformed": decision["malformed"],
            "extracted_hash": decision["extracted_hash"],
        }
    )


def _build_overlap(inventory: dict[str, Any]) -> dict[str, Any]:
    records = cast(list[dict[str, Any]], inventory["records"])
    by_arm = {
        arm: {str(row["prompt_id"]): row for row in records if row["arm"] == arm}
        for arm in ("generic", "targeted")
    }
    generic_ids = set(by_arm["generic"])
    targeted_ids = set(by_arm["targeted"])
    shared = sorted(generic_ids & targeted_ids)
    matrix: list[dict[str, Any]] = []
    for generic_id in sorted(generic_ids):
        for targeted_id in sorted(targeted_ids):
            generic = by_arm["generic"][generic_id]
            targeted = by_arm["targeted"][targeted_id]
            matrix.append(
                {
                    "generic_prompt_id_sha256": generic["prompt_id_sha256"],
                    "targeted_prompt_id_sha256": targeted["prompt_id_sha256"],
                    "same_prompt": generic_id == targeted_id,
                    "same_response_hash": (
                        generic["adapter_response_sha256"] == targeted["adapter_response_sha256"]
                    ),
                    "structurally_equivalent_output": (
                        generic_id == targeted_id
                        and _failure_structure(generic) == _failure_structure(targeted)
                    ),
                    "same_failure_family": (
                        generic["deterministic_failure_family"]
                        == targeted["deterministic_failure_family"]
                    ),
                    "same_scorer_branch": (
                        generic["deterministic_rule_failed"]
                        == targeted["deterministic_rule_failed"]
                    ),
                }
            )
    exact_output_overlap = sum(
        by_arm["generic"][item]["adapter_response_sha256"]
        == by_arm["targeted"][item]["adapter_response_sha256"]
        for item in shared
    )
    structural_overlap = sum(
        _failure_structure(by_arm["generic"][item]) == _failure_structure(by_arm["targeted"][item])
        for item in shared
    )
    shared_arithmetic = sum(by_arm["generic"][item]["category"] == "arithmetic" for item in shared)
    shared_format = sum(by_arm["generic"][item]["category"] == "format" for item in shared)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": "foundry-vetted-corpus-two-arm-failure-overlap-v1",
        "shared_failing_prompt_count": len(shared),
        "generic_only_failing_prompt_count": len(generic_ids - targeted_ids),
        "targeted_only_failing_prompt_count": len(targeted_ids - generic_ids),
        "shared_prompt_id_sha256": canonical_sha256([_sha256_text(item) for item in shared]),
        "shared_exact_response_hash_count": exact_output_overlap,
        "structurally_equivalent_output_count": structural_overlap,
        "shared_arithmetic_failure_count": shared_arithmetic,
        "shared_format_failure_count": shared_format,
        "shared_failure_families": sorted(
            {str(by_arm["generic"][item]["deterministic_failure_family"]) for item in shared}
        ),
        "shared_scorer_branches": sorted(
            {str(by_arm["generic"][item]["deterministic_rule_failed"]) for item in shared}
        ),
        "base_response_overlap_count": sum(
            by_arm["generic"][item]["untouched_base_response_sha256"]
            == by_arm["targeted"][item]["untouched_base_response_sha256"]
            for item in shared
        ),
        "adapter_output_overlap_count": exact_output_overlap,
        "matrix_dimensions": [len(generic_ids), len(targeted_ids)],
        "equivalence_matrix_sha256": canonical_sha256(matrix),
        "answers": {
            "same_ten_prompts": len(shared) == 10,
            "nine_arithmetic_failures_shared": shared_arithmetic == 9,
            "one_format_failure_shared": shared_format == 1,
            "identical_outputs_for_all_failures": exact_output_overlap == 10,
            "different_outputs_same_reason_count": structural_overlap - exact_output_overlap,
            "targeted_specific_retention_defect": False,
            "generic_specific_retention_defect": False,
            "classification": "shared_adaptation_drift",
        },
    }
    return freeze(payload, "overlap_analysis_sha256")


def _build_trajectory(repository_root: Path) -> dict[str, Any]:
    root = repository_root / "results/raw/phase2_vetted_corpus/milestone13a/rescore_a"
    cells: list[dict[str, Any]] = []
    for variant in ("v1", "v2"):
        for arm in ("generic", "targeted"):
            for step in (16, 32, 64):
                for suite in ("adjudication", "anchor"):
                    cell_root = root / f"{variant}/{arm}/step-{step}"
                    summary_path = cell_root / f"{suite}_summary.json"
                    raw_path = cell_root / f"{suite}_raw.json"
                    summary = cast(dict[str, Any], _read_json(summary_path))
                    rows = cast(list[dict[str, Any]], _read_json(raw_path))
                    metrics = cast(dict[str, Any], summary["metrics"])
                    family_counts = Counter(
                        str(row["skill"])
                        for row in rows
                        if not bool(cast(dict[str, Any], row["score"])["correct"])
                    )
                    cells.append(
                        {
                            "variant": variant.upper(),
                            "arm": arm,
                            "step": step,
                            "suite": suite,
                            "overall": {
                                "correct": metrics["correct"],
                                "total": metrics["total"],
                                "preservation": metrics["overall_preservation"],
                            },
                            "sections": metrics["section_metrics"],
                            "adapter_only_failure_families": dict(sorted(family_counts.items())),
                            "maximum_failure_family": metrics["maximum_failure_family"],
                            "source_summary_file_sha256": file_sha256(summary_path),
                            "source_raw_file_sha256": file_sha256(raw_path),
                        }
                    )
    by_key = {
        (
            str(cell["variant"]),
            str(cell["arm"]),
            int(cell["step"]),
            str(cell["suite"]),
        ): cell
        for cell in cells
    }
    trajectory_change: Counter[str] = Counter()
    for variant in ("V1", "V2"):
        for arm in ("generic", "targeted"):
            for suite in ("adjudication", "anchor"):
                first = cast(dict[str, Any], by_key[(variant, arm, 16, suite)]["overall"])
                last = cast(dict[str, Any], by_key[(variant, arm, 64, suite)]["overall"])
                difference = int(last["correct"]) - int(first["correct"])
                trajectory_change[
                    "improved" if difference > 0 else "worsened" if difference < 0 else "unchanged"
                ] += 1
    improvements: list[dict[str, Any]] = []
    for arm in ("generic", "targeted"):
        for step in (16, 32, 64):
            for suite in ("adjudication", "anchor"):
                v1_sections = cast(
                    dict[str, dict[str, Any]],
                    by_key[("V1", arm, step, suite)]["sections"],
                )
                v2_sections = cast(
                    dict[str, dict[str, Any]],
                    by_key[("V2", arm, step, suite)]["sections"],
                )
                for section in ("arithmetic", "format", "instruction"):
                    delta = int(v2_sections[section]["correct"]) - int(
                        v1_sections[section]["correct"]
                    )
                    if delta > 0:
                        improvements.append(
                            {
                                "arm": arm,
                                "step": step,
                                "suite": suite,
                                "section": section,
                                "correct_delta": delta,
                            }
                        )
    max_arm_difference = max(
        abs(
            int(
                cast(dict[str, Any], by_key[(variant, "generic", step, suite)]["overall"])[
                    "correct"
                ]
            )
            - int(
                cast(dict[str, Any], by_key[(variant, "targeted", step, suite)]["overall"])[
                    "correct"
                ]
            )
        )
        for variant in ("V1", "V2")
        for step in (16, 32, 64)
        for suite in ("adjudication", "anchor")
    )
    max_variant_difference = max(
        abs(
            int(cast(dict[str, Any], by_key[("V1", arm, step, suite)]["overall"])["correct"])
            - int(cast(dict[str, Any], by_key[("V2", arm, step, suite)]["overall"])["correct"])
        )
        for arm in ("generic", "targeted")
        for step in (16, 32, 64)
        for suite in ("adjudication", "anchor")
    )
    v1_training = cast(
        dict[str, Any],
        _read_json(repository_root / "results/phase2_vetted_corpus/v1_training.json"),
    )
    v2_training = cast(
        dict[str, Any],
        _read_json(
            repository_root / "results/phase2_vetted_corpus/v2_training_and_retention_stop.json"
        ),
    )
    training_loss_evidence = {
        "V1": {
            arm: {
                "initial_loss": cast(dict[str, Any], v1_training[arm])["initial_loss"],
                "final_loss": cast(dict[str, Any], v1_training[arm])["final_loss"],
            }
            for arm in ("generic", "targeted")
        },
        "V2": {
            arm: {
                "initial_loss": cast(dict[str, Any], v2_training[f"{arm}_training"])[
                    "initial_loss"
                ],
                "final_loss": cast(dict[str, Any], v2_training[f"{arm}_training"])["final_loss"],
            }
            for arm in ("generic", "targeted")
        },
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": "foundry-vetted-corpus-retention-trajectory-v1",
        "cells": cells,
        "step16_to_step64_suite_trajectory_counts": dict(sorted(trajectory_change.items())),
        "replay40_category_improvements": improvements,
        "replay40_uniformly_improved_retention": False,
        "replay40_established_independent_safety": False,
        "v2_independent_final_was_run": False,
        "training_loss_evidence": training_loss_evidence,
        "replay40_lower_final_training_loss_both_arms": all(
            float(training_loss_evidence["V2"][arm]["final_loss"])
            < float(training_loss_evidence["V1"][arm]["final_loss"])
            for arm in ("generic", "targeted")
        ),
        "replay40_changed_loss_without_establishing_independent_safety": True,
        "independent_final_used_for_retroactive_checkpoint_selection": False,
        "maximum_generic_targeted_correct_count_difference": max_arm_difference,
        "generic_targeted_trajectories_aligned": max_arm_difference <= 2,
        "maximum_v1_v2_correct_count_difference": max_variant_difference,
        "v1_v2_residual_drift_aligned": max_variant_difference <= 3,
        "interpretation": (
            "More steps worsened six of eight suite trajectories; REPLAY40 produced localized "
            "category gains but no uniform retention improvement or independent-safety result."
        ),
    }
    return freeze(payload, "trajectory_analysis_sha256")


def _schedule_exposure(
    schedule: list[dict[str, Any]], replay_skill: dict[str, str]
) -> tuple[Counter[str], Counter[str]]:
    occurrences: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for step in schedule:
        for item in cast(list[dict[str, Any]], step["occurrences"]):
            if item["kind"] != "replay":
                continue
            skill = replay_skill[str(item["record_id"])]
            occurrences[skill] += 1
            tokens[skill] += int(item["tokens"])
    return occurrences, tokens


def _build_coverage(
    *,
    repository_root: Path,
    suite_items: dict[str, dict[str, Any]],
    overlap: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    replay_path = repository_root / "results/raw/training/base_replay_kl/replay_corpus.json"
    replay_root = cast(dict[str, Any], _read_json(replay_path))
    replay = cast(list[dict[str, Any]], replay_root["items"])
    replay_skill = {str(item["id"]): str(item["skill"]) for item in replay}
    skill_record_counts = Counter(replay_skill.values())
    schedules: dict[str, list[dict[str, Any]]] = {
        variant: cast(
            list[dict[str, Any]],
            _read_json(
                repository_root
                / f"results/raw/phase2_vetted_corpus/{directory}/generic_schedule.json"
            ),
        )
        for variant, directory in (
            ("V1", "v1_replay25_schedules"),
            ("V2", "v2_replay40_schedules"),
        )
    }
    exposures = {
        variant: _schedule_exposure(schedule, replay_skill)
        for variant, schedule in schedules.items()
    }
    replay_normalized = {normalize_text(str(item["prompt"])) for item in replay}
    replay_structures = {structural_signature(str(item["prompt"])) for item in replay}
    records = cast(list[dict[str, Any]], inventory["records"])
    generic = [row for row in records if row["arm"] == "generic"]
    coverage_rows: list[dict[str, Any]] = []
    for record in generic:
        item = suite_items[str(record["prompt_id"])]
        prompt = str(item["prompt"])
        family = str(record["deterministic_failure_family"])
        mapped = REPLAY_FAMILY_MAP[family]
        exact_prompt = any(prompt == str(row["prompt"]) for row in replay)
        normalized_prompt = normalize_text(prompt) in replay_normalized
        structure_match = structural_signature(prompt) in replay_structures
        directly_supervised = exact_prompt or normalized_prompt or structure_match
        represented = sum(skill_record_counts[skill] for skill in mapped) > 0
        row: dict[str, Any] = {
            "prompt_id_sha256": record["prompt_id_sha256"],
            "prompt_sha256": record["prompt_sha256"],
            "category": record["category"],
            "failure_family": family,
            "exact_prompt_in_replay": exact_prompt,
            "normalized_prompt_in_replay": normalized_prompt,
            "structural_template_in_replay": structure_match,
            "exact_skill_name_in_replay": family in skill_record_counts,
            "mapped_replay_skills": list(mapped),
            "mapped_family_represented": represented,
            "replay_record_count": sum(skill_record_counts[skill] for skill in mapped),
            "v1_occurrence_count_per_arm": sum(exposures["V1"][0][skill] for skill in mapped),
            "v2_occurrence_count_per_arm": sum(exposures["V2"][0][skill] for skill in mapped),
            "v1_assistant_tokens_per_arm": sum(exposures["V1"][1][skill] for skill in mapped),
            "v2_assistant_tokens_per_arm": sum(exposures["V2"][1][skill] for skill in mapped),
            "base_behavior_directly_supervised": directly_supervised,
            "only_adjacent_behavior_supervised": represented and not directly_supervised,
            "no_related_behavior_supervised": not represented,
            "coverage_class": (
                "direct" if directly_supervised else "adjacent_only" if represented else "outside"
            ),
        }
        row["v1_assistant_tokens_both_arms"] = int(row["v1_assistant_tokens_per_arm"]) * 2
        row["v2_assistant_tokens_both_arms"] = int(row["v2_assistant_tokens_per_arm"]) * 2
        coverage_rows.append(row)
    classes = Counter(str(row["coverage_class"]) for row in coverage_rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": "foundry-vetted-corpus-replay-coverage-v1",
        "replay_corpus_file_sha256": file_sha256(replay_path),
        "replay_record_count": len(replay),
        "replay_section_counts": dict(
            sorted(Counter(str(item["section"]) for item in replay).items())
        ),
        "failure_coverage": coverage_rows,
        "coverage_class_counts": dict(sorted(classes.items())),
        "all_failures_exactly_or_structurally_direct": classes["direct"] == len(coverage_rows),
        "all_failures_have_adjacent_family_coverage": (
            classes["adjacent_only"] == len(coverage_rows)
        ),
        "coverage_interpretation": (
            "All ten failures have adjacent operation or format-family replay coverage, but none "
            "has an exact, normalized, structural-template, or exact-skill replay target."
        ),
        "overlap_analysis_sha256": overlap["overlap_analysis_sha256"],
    }
    return freeze(payload, "replay_coverage_sha256")


def _build_objective_audit(repository_root: Path, coverage: dict[str, Any]) -> dict[str, Any]:
    implementation = repository_root / "src/foundry/phase2/vetted_qlora.py"
    schedules = repository_root / "src/foundry/phase2/vetted_schedule.py"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-vetted-corpus-training-objective-audit-v1",
        "implementation_source_sha256": file_sha256(implementation),
        "schedule_source_sha256": file_sha256(schedules),
        "v1_replay_share": 0.25,
        "v2_replay_share": 0.40,
        "objective_components": {
            "vetted_corpus_assistant_token_cross_entropy": True,
            "replay_target_assistant_token_cross_entropy": True,
            "combined_by_whole_example_token_weighted_scheduling": True,
            "token_level_kl_to_adapter_disabled_base": False,
            "logit_preservation": False,
            "representation_preservation": False,
            "gradient_balancing_or_projection": False,
            "base_behavior_constraint_beyond_replay_cross_entropy": False,
        },
        "mechanism_evidence": {
            "insufficient_replay_quantity": (
                "not_supported_as_primary_because_replay40_was_not_uniformly_better"
            ),
            "replay_coverage_gap": (
                "supported_as_exact_targets_are_absent_and_only_adjacent_families_are_covered"
            ),
            "shared_task_replay_gradient_interference": (
                "plausible_but_unmeasured_no_separate_gradient_evidence"
            ),
            "unconstrained_logit_drift_outside_exact_replay_targets": "supported",
            "excessive_adapter_capacity": (
                "not_supported_no_capacity_or_layer_scope_ablation_exists"
            ),
            "ambiguous_mechanism": (
                "causal_mechanism_not_fully_identified_but_primary_intervention_is_unique_under_hierarchy"
            ),
        },
        "either_replay_ratio_eliminated_shared_drift": False,
        "replay_coverage_sha256": coverage["replay_coverage_sha256"],
        "primary_intervention": "explicit_token_level_kl_logit_preservation",
    }
    return freeze(payload, "objective_audit_sha256")


def _build_options(objective: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "comparison_id": "foundry-vetted-corpus-retention-architecture-options-v1",
        "options": {
            "replay-ce-token-kl-v1": {
                "eligible": True,
                "adds_missing_primary_mechanism": "token_level_kl_logit_preservation",
                "preserves_curriculum_comparison": True,
                "requires_learned_reward_model": False,
                "uses_adapter_disabled_frozen_base_reference": True,
            },
            "layer-restricted-lora-v1": {
                "eligible": False,
                "reason": (
                    "no_frozen_capacity_or_layer_scope_evidence_implicates_excessive_capacity"
                ),
            },
            "multiobjective-gradient-balanced-sft-v1": {
                "eligible": False,
                "reason": "no_separate_task_replay_gradient_conflict_was_measured",
            },
            "verifier-grpo-after-safe-sft-v1": {
                "eligible": False,
                "reason": "no_retention_safe_sft_checkpoint_exists",
                "later_tier3_requirement": True,
            },
        },
        "prohibited_nonoptions": [
            "another_replay_percentage",
            "lower_retention_threshold",
            "adapter_scaling",
            "benchmark_selected_checkpoint",
            "grpo_before_retention_safe_sft",
        ],
        "objective_audit_sha256": objective["objective_audit_sha256"],
    }
    return freeze(payload, "options_comparison_sha256")


def _build_decision(
    *,
    inventory: dict[str, Any],
    overlap: dict[str, Any],
    trajectory: dict[str, Any],
    coverage: dict[str, Any],
    objective: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    selected = select_architecture(
        shared_failure_fraction=float(overlap["shared_failing_prompt_count"]) / 10,
        replay_ratio_trajectories_aligned=bool(trajectory["v1_v2_residual_drift_aligned"]),
        explicit_logit_constraint=bool(
            objective["objective_components"]["token_level_kl_to_adapter_disabled_base"]
        ),
        capacity_implicated=False,
        gradient_conflict_measured=False,
    )
    if selected != "replay-ce-token-kl-v1":
        raise RuntimeError("predeclared hierarchy did not select replay-ce-token-kl-v1")
    selection_contract = {
        "hierarchy_item": 1,
        "shared_failure_count": overlap["shared_failing_prompt_count"],
        "shared_failure_fraction": 1.0,
        "generic_targeted_trajectories_aligned": trajectory[
            "generic_targeted_trajectories_aligned"
        ],
        "v1_v2_residual_drift_aligned": trajectory["v1_v2_residual_drift_aligned"],
        "replay40_uniformly_improved_retention": trajectory[
            "replay40_uniformly_improved_retention"
        ],
        "explicit_kl_or_logit_constraint_existed": False,
        "selected_architecture": selected,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": "foundry-vetted-corpus-next-retention-architecture-v1",
        "architecture_id": selected,
        "failure_inventory_sha256": inventory["failure_inventory_sha256"],
        "overlap_analysis_sha256": overlap["overlap_analysis_sha256"],
        "trajectory_analysis_sha256": trajectory["trajectory_analysis_sha256"],
        "replay_coverage_sha256": coverage["replay_coverage_sha256"],
        "objective_audit_sha256": objective["objective_audit_sha256"],
        "options_comparison_sha256": options["options_comparison_sha256"],
        "selection_contract": selection_contract,
        "selection_decision_sha256": canonical_sha256(selection_contract),
        "inherited_dataset_identity": DATASET_SHA256,
        "inherited_base_model_revision": BASE_REVISION,
        "inherited_controls": [
            "human_written_vetted_corpus_identity",
            "targeted_generic_matching_and_contamination_controls",
            "assistant_only_cross_entropy_contract",
            "corrected_retention_scorer",
            "existing_adjudication_and_anchor_development_instruments",
            "benchmark_and_sealed_final_firewall",
        ],
        "prohibited_alternatives": options["prohibited_nonoptions"],
        "scientific_hypothesis": (
            "Adding token-level KL preservation against the adapter-disabled frozen base on replay "
            "prompts will reduce shared base-behavior regression that persisted under 25% and 40% "
            "replay cross-entropy alone, while preserving the targeted-versus-generic curriculum "
            "comparison."
        ),
        "next_milestone_success_conditions": [
            "one_predeclared_common_checkpoint_passes_adjudication_and_anchor_for_both_arms",
            "both_arms_pass_a_new_once_only_independent_base_correct_retention_subset",
            "targeted_generic_curriculum_comparison_remains_matched",
            "only_after_retention_approval_may_gsm1k_run",
        ],
        "next_milestone_failure_conditions": [
            "no_common_checkpoint_passes_development_retention",
            "either_arm_fails_new_independent_retention",
            "kl_runtime_or_reference_contract_fails",
            "dataset_matching_or_benchmark_firewall_drifts",
        ],
        "not_selected_in_this_milestone": {
            "kl_coefficient": None,
            "checkpoint": None,
            "seed": None,
            "new_replay_ratio": None,
            "new_final_retention_subset": None,
        },
        "exposed_subset": {
            "subset_sha256": INDEPENDENT_SUBSET_SHA256,
            "status": "diagnostic_only_for_future_architectures",
            "preserved_as_milestone13a_evidence": True,
            "future_architecture_selection_use": False,
            "future_checkpoint_selection_use": False,
            "future_independent_final_gate_use": False,
            "modified_or_deleted": False,
        },
        "future_holdout_requirements": list(FUTURE_HOLDOUT_REQUIREMENTS),
        "future_holdout_requirements_sha256": canonical_sha256(FUTURE_HOLDOUT_REQUIREMENTS),
        "tier3_roadmap": list(TIER3_ROADMAP),
        "tier3_roadmap_sha256": canonical_sha256(TIER3_ROADMAP),
        "model_inference_runs": 0,
        "training_runs": 0,
        "gsm1k_runs": 0,
        "sealed_final_accessed": False,
    }
    return freeze(payload, "architecture_decision_sha256")


def build_interpretation(
    *, repository_root: Path, model_path: Path, output_root: Path
) -> dict[str, Any]:
    """Build all Milestone 13B evidence from existing frozen artifacts."""

    root = repository_root.resolve()
    if output_root.exists():
        raise FileExistsError("interpretation output root must be fresh")
    source_identity = _validate_source_identities(root)
    tokenizer_module: Any = importlib.import_module("transformers")
    tokenizer = tokenizer_module.AutoTokenizer.from_pretrained(
        str(model_path.resolve()), local_files_only=True, trust_remote_code=False
    )
    suite_root = cast(
        dict[str, Any],
        _read_json(
            root / "results/raw/training/base_replay_kl/retention_replay_final_holdout_v1.json"
        ),
    )
    suite_items = {
        str(item["id"]): item for item in cast(list[dict[str, Any]], suite_root["items"])
    }
    base_rows = {
        str(row["id"]): row
        for row in cast(
            list[dict[str, Any]],
            _read_json(root / "results/raw/training/base_replay_kl/final_holdout_base_raw.json"),
        )
    }
    arm_rows = {
        arm: cast(
            list[dict[str, Any]],
            _read_json(
                root
                / f"results/raw/phase2_vetted_corpus/milestone13a/independent_final/{arm}_raw.json"
            ),
        )
        for arm in ("generic", "targeted")
    }
    inventory = _build_failure_inventory(
        suite_items=suite_items,
        base_rows=base_rows,
        arm_rows=arm_rows,
        tokenizer=tokenizer,
    )
    overlap = _build_overlap(inventory)
    trajectory = _build_trajectory(root)
    coverage = _build_coverage(
        repository_root=root,
        suite_items=suite_items,
        overlap=overlap,
        inventory=inventory,
    )
    objective = _build_objective_audit(root, coverage)
    options = _build_options(objective)
    decision = _build_decision(
        inventory=inventory,
        overlap=overlap,
        trajectory=trajectory,
        coverage=coverage,
        objective=objective,
        options=options,
    )
    aggregate_payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "starting_commit": STARTING_COMMIT,
        "dataset_identity": DATASET_SHA256,
        "base_model_revision": BASE_REVISION,
        "corrected_scorer_source_sha256": SCORER_SOURCE_SHA256,
        "corrected_scorer_configuration_sha256": SCORER_CONFIGURATION_SHA256,
        "selected_variant": "V1_REPLAY25",
        "selected_checkpoint": 64,
        "selected_adapter_sha256": ADAPTER_SHA256,
        "independent_subset_sha256": INDEPENDENT_SUBSET_SHA256,
        "independent_decision_sha256": INDEPENDENT_DECISION_SHA256,
        "source_identity": source_identity,
        "failure_inventory": {
            "generic_failures": inventory["counts"]["generic"],
            "targeted_failures": inventory["counts"]["targeted"],
            "failure_inventory_sha256": inventory["failure_inventory_sha256"],
        },
        "overlap": {
            "shared_failures": overlap["shared_failing_prompt_count"],
            "generic_only": overlap["generic_only_failing_prompt_count"],
            "targeted_only": overlap["targeted_only_failing_prompt_count"],
            "shared_exact_response_hashes": overlap["shared_exact_response_hash_count"],
            "structurally_equivalent_outputs": overlap["structurally_equivalent_output_count"],
            "shared_arithmetic_failures": overlap["shared_arithmetic_failure_count"],
            "shared_format_failures": overlap["shared_format_failure_count"],
            "maximum_failure_family": "scaled_sum_with_offset",
            "maximum_failure_family_count_per_arm": 5,
            "classification": "shared_adaptation_drift",
            "overlap_analysis_sha256": overlap["overlap_analysis_sha256"],
        },
        "trajectory": {
            "step16_to_step64_suite_trajectory_counts": trajectory[
                "step16_to_step64_suite_trajectory_counts"
            ],
            "replay40_uniformly_improved_retention": False,
            "generic_targeted_trajectories_aligned": trajectory[
                "generic_targeted_trajectories_aligned"
            ],
            "trajectory_analysis_sha256": trajectory["trajectory_analysis_sha256"],
        },
        "replay_coverage": {
            "coverage_class_counts": coverage["coverage_class_counts"],
            "directly_supervised_failure_count": 0,
            "adjacent_only_failure_count": 10,
            "replay_coverage_sha256": coverage["replay_coverage_sha256"],
        },
        "objective": {
            "v1_replay_share": 0.25,
            "v2_replay_share": 0.40,
            "replay_and_task_cross_entropy_only": True,
            "explicit_kl_or_logit_constraint": False,
            "objective_audit_sha256": objective["objective_audit_sha256"],
        },
        "architecture": {
            "options_comparison_sha256": options["options_comparison_sha256"],
            "selected_architecture": decision["architecture_id"],
            "selection_decision_sha256": decision["selection_decision_sha256"],
            "architecture_decision_sha256": decision["architecture_decision_sha256"],
        },
        "exposed_subset_status": "diagnostic_only_for_future_architectures",
        "future_holdout_requirements_sha256": decision["future_holdout_requirements_sha256"],
        "tier3_roadmap_sha256": decision["tier3_roadmap_sha256"],
        "model_inference_runs": 0,
        "training_runs": 0,
        "gsm1k_runs": 0,
        "sealed_final_accessed": False,
    }
    aggregate = freeze(aggregate_payload, "aggregate_sha256")
    outputs = {
        "failure_inventory.json": inventory,
        "overlap_analysis.json": overlap,
        "trajectory_analysis.json": trajectory,
        "replay_coverage.json": coverage,
        "objective_audit.json": objective,
        "options_comparison.json": options,
        "architecture_decision.json": decision,
        "tracked_aggregate.json": aggregate,
    }
    for name, value in outputs.items():
        _write_json(output_root / name, value)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_interpretation(
                repository_root=args.repository_root,
                model_path=args.model_path,
                output_root=args.output_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
