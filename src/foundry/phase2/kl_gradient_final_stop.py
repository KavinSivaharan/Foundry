"""Freeze the terminal Milestone 13D gradient-calibration stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256

STARTING_COMMIT = "f41021c01c9fd03279717473339bcea4b4274b32"
LADDER_COMMIT = "4c054bf6541367f9fbfbb766ca247891537a67a0"


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


def build(root: Path) -> dict[str, Any]:
    """Bind every published 13D decision into the required terminal stop."""

    results = root / "results/phase2_vetted_corpus"
    recipe = _read(results / "milestone13c_r3_v1_kl_recipe.json")
    objective = _read(results / "milestone13c_r3_kl_objective.json")
    historical = _read(results / "milestone13c_r3_historical_comparator.json")
    for value, key in (
        (recipe, "final_recipe_decision_sha256"),
        (objective, "objective_contract_sha256"),
        (historical, "historical_comparator_summary_sha256"),
    ):
        _verify(value, key)
    evidence_specs = (
        ("raw_scale", "milestone13d_raw_scale_analysis.json", "raw_scale_analysis_sha256"),
        (
            "objective_graph",
            "milestone13d_objective_graph_integrity.json",
            "objective_graph_integrity_sha256",
        ),
        (
            "measurement_manifest",
            "milestone13d_gradient_measurement_manifest.json",
            "measurement_manifest_sha256",
        ),
        (
            "historical_gradient",
            "milestone13d_historical_gradient_audit.json",
            "gradient_audit_sha256",
        ),
        (
            "previous_ladder_scale",
            "milestone13d_previous_ladder_gradient_scale.json",
            "previous_ladder_gradient_scale_sha256",
        ),
        ("ladder", "milestone13d_gradient_ladder.json", "ladder_sha256"),
        ("smoke", "milestone13d_kl_smoke.json", "smoke_summary_sha256"),
        ("calibration", "milestone13d_kl_calibration.json", "calibration_summary_sha256"),
        (
            "selection",
            "milestone13d_kl_coefficient_selection.json",
            "coefficient_selection_sha256",
        ),
        (
            "blocker",
            "milestone13d_kl_calibration_blocker.json",
            "calibration_blocker_sha256",
        ),
    )
    evidence: dict[str, dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    for name, filename, hash_key in evidence_specs:
        value = _read(results / filename)
        _verify(value, hash_key)
        evidence[name] = value
        evidence_hashes[name] = cast(str, value[hash_key])

    smoke = evidence["smoke"]
    calibration = evidence["calibration"]
    selection = evidence["selection"]
    blocker = evidence["blocker"]
    if (
        evidence["raw_scale"]["objective_contract_sha256"] != objective["objective_contract_sha256"]
        or evidence["historical_gradient"]["objective_contract_sha256"]
        != objective["objective_contract_sha256"]
        or calibration["historical_comparator_sha256"]
        != historical["historical_comparator_summary_sha256"]
        or historical["recipe_sha256"] != recipe["historical_v1_configuration_sha256"]
        or recipe["dataset_sha256"] != evidence["raw_scale"]["dataset_sha256"]
        or smoke.get("run_count") != 8
        or calibration.get("run_count") != 8
        or selection.get("selected_rho") is not None
        or selection.get("selected_coefficient_exact") is not None
        or selection.get("decision") != "no_common_eligible_gradient_scaled_coefficient"
        or blocker.get("required_stop") != "before_full_training"
    ):
        raise ValueError("published 13D evidence does not require the terminal stop")
    calibration_runs = cast(list[dict[str, Any]], calibration["runs"])
    smoke_runs = cast(list[dict[str, Any]], smoke["runs"])
    if any(
        row["criteria"]["replay_token_kl_at_most_75_percent_historical"] is not False
        for row in calibration_runs
    ):
        raise ValueError("every bounded calibration must fail the unchanged replay-KL gate")
    if any(
        any(
            passed is not True
            for name, passed in cast(dict[str, bool], row["criteria"]).items()
            if name != "replay_token_kl_at_most_75_percent_historical"
        )
        for row in calibration_runs
    ):
        raise ValueError("a calibration failed a gate other than replay-KL reduction")

    record: dict[str, Any] = {
        "schema_version": 1,
        "record_id": "foundry-milestone13d-gradient-kl-terminal-stop-v1",
        "result": "token_level_replay_kl_closed_for_v1_equivalent_architecture",
        "reason": blocker["reason"],
        "starting_commit": STARTING_COMMIT,
        "published_commits": {
            "gradient_ladder": LADDER_COMMIT,
            "terminal_result_subject": "analysis: stop gradient-calibrated KL adaptation",
        },
        "dataset_sha256": "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31",
        "architecture_decision_sha256": (
            "74907ea92b2217b6f9ca39044feab6c6452600e7774a2b442e0ec9e29b6899a5"
        ),
        "environment_v2": {
            "operational_environment_sha256": (
                "76afee8390e73ef9274d4bc4b91d8a99735f66efb9137c4909e8619d3f9d244a"
            ),
            "combined_child_environment_sha256": (
                "1d402ec0cb661adeb50a3d3bd9510895f3f9068cbb393fb381565a5670de995b"
            ),
            "v2_contract_sha256": (
                "c9faa8afafafb20b84fcd0cb5e7de1b57749e822adfa27c8b401bbaf8f0153dc"
            ),
            "replay_passed": True,
        },
        "holdout_v2": {
            "suite_sha256": "b8b978ba69b501187b984d4631e8d7f5a41f39efeb292cb391a845c3e61e1b18",
            "base_correct_subset_sha256": (
                "a23b1014d92e9f98b74da3b29913a430bdaebf8e07a16b31b4c3dcc831f1f420"
            ),
            "decision_sha256": "970df977248195d71e3d5a54a20d24b5c2cc07ad5c3c7add10d1bf13dcb0ae60",
            "adapter_evaluations": 0,
            "status": "not_evaluated_and_adapter_unexposed",
        },
        "recipe_decision_sha256": recipe["final_recipe_decision_sha256"],
        "objective_contract_sha256": objective["objective_contract_sha256"],
        "historical_comparator_sha256": historical["historical_comparator_summary_sha256"],
        "evidence_sha256": evidence_hashes,
        "smoke_counts": {
            "runs": smoke["run_count"],
            "optimizer_steps": sum(int(row["optimizer_steps"]) for row in smoke_runs),
            "assistant_tokens": sum(int(row["loss_bearing_tokens"]) for row in smoke_runs),
            "common_eligible_coefficients": sum(
                int(row["common_eligible"])
                for row in cast(list[dict[str, Any]], smoke["coefficient_results"])
            ),
        },
        "calibration_counts": {
            "coefficients": 4,
            "arms": 2,
            "runs": calibration["run_count"],
            "optimizer_steps": calibration["optimizer_step_count"],
            "assistant_tokens": calibration["loss_bearing_token_count"],
            "development_retention_evaluations": 16,
            "backend_failures": 0,
        },
        "calibration_resource_use": {
            "training_runtime_seconds": sum(
                float(row["runtime_seconds"]) for row in calibration_runs
            ),
            "peak_reserved_vram_bytes": max(
                int(row["peak_reserved_vram_bytes"]) for row in calibration_runs
            ),
            "peak_process_rss_bytes": max(
                int(row["peak_process_rss_bytes"]) for row in calibration_runs
            ),
            "saved_adapter_bytes": sum(int(row["adapter_bytes"]) for row in calibration_runs),
        },
        "replay_kl_ratio_by_run": blocker["replay_kl_ratio_by_run"],
        "selected_rho": None,
        "selected_coefficient_exact": None,
        "full_training_runs": 0,
        "full_training_optimizer_steps": 0,
        "independent_holdout_adapter_evaluations": 0,
        "gsm1k_adapter_evaluations": 0,
        "verification": {
            "ruff_formatted_file_count": 309,
            "ruff_lint_passed": True,
            "strict_mypy_source_file_count": 175,
            "strict_mypy_passed": True,
            "full_test_count": 973,
            "full_tests_passed": True,
            "dependency_environment_count": 2,
            "dependency_checks_passed": True,
            "all_required_reconstructions_passed": True,
            "publication_candidate_count": 23,
            "development_reference_count": 904,
            "exact_development_hits": 0,
            "normalized_exact_development_hits": 0,
            "contiguous_12_token_development_hits": 0,
            "high_confidence_secret_hits": 0,
            "content_bearing_result_hits": 0,
            "raw_tracked_artifacts": 0,
            "protected_path_changes": 0,
            "candidate_files_at_or_above_1_mib": 0,
            "git_diff_check_passed": True,
        },
        "sealed_boundary_status": "metadata_accessed_example_content_unseen",
        "sealed_paths_accessed": False,
        "next_action": (
            "project_level_selection_between_layer_restricted_lora_and_"
            "multi_objective_gradient_balanced_sft"
        ),
    }
    record["terminal_stop_sha256"] = canonical_sha256(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("terminal stop output already exists")
    result = build(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
