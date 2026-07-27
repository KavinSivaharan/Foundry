"""Freeze the terminal Milestone 13C-R3 calibration stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256


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
    """Bind every published R3 decision into the required terminal stop."""

    results = root / "results/phase2_vetted_corpus"
    recipe = _read(results / "milestone13c_r3_v1_kl_recipe.json")
    objective = _read(results / "milestone13c_r3_kl_objective.json")
    historical = _read(results / "milestone13c_r3_historical_comparator.json")
    calibration = _read(results / "milestone13c_r3_kl_calibration.json")
    selection = _read(results / "milestone13c_r3_kl_coefficient_selection.json")
    blocker = _read(results / "milestone13c_r3_kl_calibration_blocker.json")
    for value, key in (
        (recipe, "final_recipe_decision_sha256"),
        (objective, "objective_contract_sha256"),
        (historical, "historical_comparator_summary_sha256"),
        (calibration, "calibration_summary_sha256"),
        (selection, "coefficient_selection_sha256"),
        (blocker, "calibration_blocker_sha256"),
    ):
        _verify(value, key)
    if (
        selection.get("selected_coefficient") is not None
        or selection.get("stop_before_full_training") is not True
        or blocker.get("all_replay_kl_gates_failed") is not True
    ):
        raise ValueError("published calibration does not require the terminal stop")
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_id": "foundry-milestone13c-r3-terminal-stop-v1",
        "result": "stopped_at_calibration",
        "reason": blocker["reason"],
        "starting_commit": "91aea71492a2b1925da6bfb235e31a46e0a47665",
        "published_commits": {
            "recipe": "c055fe6834d001381f7e0e9aea8ea0f89494afe5",
            "objective": "f9d150a3f41f83016c0162588a6b20d77cefbe59",
            "historical_comparator": "9443a26a6e27b7b3f5aaf72dff2c3792039b794b",
            "coefficient_selection": "36ec0ca54383de7d8e56f0b0d7de0b0d768f6a5a",
        },
        "dataset_sha256": ("ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"),
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
            "suite_sha256": ("b8b978ba69b501187b984d4631e8d7f5a41f39efeb292cb391a845c3e61e1b18"),
            "base_correct_subset_sha256": (
                "a23b1014d92e9f98b74da3b29913a430bdaebf8e07a16b31b4c3dcc831f1f420"
            ),
            "decision_sha256": ("970df977248195d71e3d5a54a20d24b5c2cc07ad5c3c7add10d1bf13dcb0ae60"),
            "base_correct_counts": {
                "arithmetic": 79,
                "format": 89,
                "instruction": 149,
                "total": 317,
            },
            "adapter_evaluations": 0,
        },
        "recipe_decision_sha256": recipe["final_recipe_decision_sha256"],
        "objective_contract_sha256": objective["objective_contract_sha256"],
        "historical_comparator_summary_sha256": historical["historical_comparator_summary_sha256"],
        "calibration_summary_sha256": calibration["calibration_summary_sha256"],
        "coefficient_selection_sha256": selection["coefficient_selection_sha256"],
        "calibration_blocker_sha256": blocker["calibration_blocker_sha256"],
        "calibration_counts": {
            "coefficients": 4,
            "arms": 2,
            "runs": calibration["run_count"],
            "optimizer_steps": calibration["optimizer_step_count"],
            "assistant_tokens": calibration["loss_bearing_token_count"],
            "development_retention_evaluations": 16,
            "backend_failures": 0,
        },
        "selected_coefficient": None,
        "full_training_runs": 0,
        "full_training_optimizer_steps": 0,
        "independent_holdout_adapter_evaluations": 0,
        "gsm1k_adapter_evaluations": 0,
        "verification": {
            "ruff_formatted_file_count": 297,
            "ruff_lint_passed": True,
            "strict_mypy_source_file_count": 169,
            "strict_mypy_passed": True,
            "full_test_count": 959,
            "full_tests_passed": True,
            "focused_reconstruction_test_count": 61,
            "focused_reconstruction_tests_passed": True,
            "dependency_environment_count": 2,
            "dependency_checks_passed": True,
            "recipe_replay_passed": True,
            "objective_replay_passed": True,
            "historical_replay_passed": True,
            "calibration_replay_passed": True,
            "selection_replay_passed": True,
            "blocker_replay_passed": True,
            "environment_v2_replay_passed": True,
            "publication_candidate_count": 22,
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
        "next_action": "project_level_interpretation_of_v1_equivalent_kl_result",
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
