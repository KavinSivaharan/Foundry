"""Build and adjudicate Milestone 13C-R3 KL calibration evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_recipe
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

OBJECTIVE_CONTRACT_SHA256 = "159ef322b254d5f70b46296fe051417ed53c1543f3432bcb0b9069d66ab975a8"
ADAPTER_CONFIG_SHA256 = "2cf0fb6637747b0aa31525f08ba8b412cc4f1986689ef8b9f555cd4b299039e2"
DEVELOPMENT_SUITES = {
    "adjudication": "5caf23be79fa01151af6f7db8d45c2b85bfe24b03a29589e482d51731c8358af",
    "anchor": "bff18b434a284d848387262dde201601278e5c8b573937b3486bed2bf925696e",
}
COEFFICIENTS = (
    ("lambda-001", 0.01),
    ("lambda-003", 0.03),
    ("lambda-010", 0.10),
    ("lambda-030", 0.30),
)


def _read_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify_hash(value: dict[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _retention_projection(path: Path, adapter_sha256: str, suite: str) -> dict[str, Any]:
    value = _read_object(path)
    _verify_hash(value, "summary_sha256")
    metrics = cast(dict[str, Any], value["metrics"])
    if (
        value.get("adapter_sha256") != adapter_sha256
        or value.get("suite_sha256") != DEVELOPMENT_SUITES[suite]
        or metrics.get("backend_failures") != 0
        or metrics.get("passed") is not True
    ):
        raise ValueError("historical development-retention evidence differs")
    return {
        "summary_sha256": value["summary_sha256"],
        "summary_file_sha256": file_sha256(path),
        "suite_sha256": value["suite_sha256"],
        "base_conditioned_subset_sha256": value["base_conditioned_subset_sha256"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "overall_preservation": metrics["overall_preservation"],
        "section_metrics": metrics["section_metrics"],
        "maximum_failure_family": metrics["maximum_failure_family"],
        "backend_failures": metrics["backend_failures"],
        "passed": metrics["passed"],
    }


def historical_record(root: Path) -> dict[str, Any]:
    """Freeze measured lambda-zero and published development-retention evidence."""

    arms: dict[str, Any] = {}
    raw_root = root / "results/raw/phase2_vetted_corpus"
    for arm in kl_recipe.ARMS:
        measurement_path = raw_root / f"milestone13c_r3/historical/{arm}.json"
        measurement = _read_object(measurement_path)
        _verify_hash(measurement, "result_sha256")
        expected_adapter = kl_recipe.EXPECTED_ADAPTER_HASHES[arm]["16"]
        if (
            measurement.get("arm") != arm
            or measurement.get("lambda_kl") != 0
            or measurement.get("optimizer_steps") != 16
            or measurement.get("adapter_sha256") != expected_adapter
            or measurement.get("adapter_config_file_sha256") != ADAPTER_CONFIG_SHA256
            or measurement.get("model_update_performed") is not False
            or measurement.get("base_restoration") is not True
            or measurement.get("holdout_v2_use") is not False
            or measurement.get("gsm1k_use") is not False
        ):
            raise ValueError("historical lambda-zero measurement differs")
        measured = cast(dict[str, Any], measurement["measurement"])
        if measured.get("total_tokens") != 16_000 or measured.get("finite") is not True:
            raise ValueError("historical measurement token or finite gate failed")
        retention_root = raw_root / f"milestone13a/rescore_a/v1/{arm}/step-16"
        arms[arm] = {
            "lambda_kl": 0,
            "optimizer_steps": 16,
            "loss_bearing_tokens": measured["total_tokens"],
            "adapter_sha256": expected_adapter,
            "checkpoint_sha256": expected_adapter,
            "adapter_config_file_sha256": measurement["adapter_config_file_sha256"],
            "schedule_sha256": measurement["schedule_sha256"],
            "schedule_prefix_sha256": measurement["schedule_prefix_sha256"],
            "measurement": measured,
            "measurement_result_sha256": measurement["result_sha256"],
            "measurement_file_sha256": file_sha256(measurement_path),
            "base_restoration": measurement["base_restoration"],
            "model_update_performed": measurement["model_update_performed"],
            "runtime_seconds": measurement["runtime_seconds"],
            "peak_allocated_vram_bytes": measurement["peak_allocated_vram_bytes"],
            "peak_reserved_vram_bytes": measurement["peak_reserved_vram_bytes"],
            "peak_process_rss_bytes": measurement["peak_process_rss_bytes"],
            "adjudication_retention": _retention_projection(
                retention_root / "adjudication_summary.json",
                expected_adapter,
                "adjudication",
            ),
            "anchor_retention": _retention_projection(
                retention_root / "anchor_summary.json",
                expected_adapter,
                "anchor",
            ),
        }
    record: dict[str, Any] = {
        "schema_version": 1,
        "comparator_id": "foundry-historical-v1-step16-lambda-zero-measured-v1",
        "objective_contract_sha256": OBJECTIVE_CONTRACT_SHA256,
        "recipe_sha256": ("3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"),
        "arms": arms,
        "both_development_retention_subsets_pass": all(
            arms[arm][f"{suite}_retention"]["passed"]
            for arm in kl_recipe.ARMS
            for suite in DEVELOPMENT_SUITES
        ),
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    record["historical_comparator_summary_sha256"] = canonical_sha256(record)
    return record


def _assessment_projection(path: Path, adapter_sha256: str) -> dict[str, Any]:
    value = _read_object(path)
    _verify_hash(value, "summary_sha256")
    if value.get("adapter_sha256") != adapter_sha256 or value.get("backend_failures") != 0:
        raise ValueError("calibration development-retention evidence differs")
    return {
        "assessment_sha256": value["summary_sha256"],
        "assessment_file_sha256": file_sha256(path),
        "suite_sha256": value["suite_sha256"],
        "subset_sha256": value["subset_sha256"],
        "preserved": value["preserved"],
        "total": value["total"],
        "overall_preservation": value["overall_preservation"],
        "section_preservation": value["section_preservation"],
        "maximum_instruction_family_adapter_only_failures": value[
            "maximum_instruction_family_adapter_only_failures"
        ],
        "backend_failures": value["backend_failures"],
        "gate_checks": value["gate_checks"],
        "gate_passed": value["gate_passed"],
    }


def _all_finite(training: dict[str, Any]) -> bool:
    metrics = cast(list[dict[str, Any]], training["step_metrics"])
    values = [
        float(row[key])
        for row in metrics
        for key in ("vetted_ce", "replay_ce", "replay_kl", "total_loss")
    ]
    final = cast(dict[str, Any], training["final_measurement"])
    values.extend(
        float(final[key])
        for key in (
            "vetted_ce",
            "replay_ce",
            "replay_token_kl",
            "vetted_validation_ce",
        )
    )
    return bool(training["finite_gradients"]) and all(math.isfinite(value) for value in values)


def calibration_record(root: Path) -> dict[str, Any]:
    """Replay the complete eight-cell calibration and apply every arm-level gate."""

    historical_path = (
        root / "results/phase2_vetted_corpus/milestone13c_r3_historical_comparator.json"
    )
    historical = _read_object(historical_path)
    _verify_hash(historical, "historical_comparator_summary_sha256")
    status_path = (
        root
        / "results/raw/phase2_vetted_corpus/milestone13c_r3"
        / "calibration_campaign_status.json"
    )
    status = _read_object(status_path)
    _verify_hash(status, "status_sha256")
    expected_runs = [
        f"{coefficient_name}/{arm}"
        for coefficient_name, _ in COEFFICIENTS
        for arm in kl_recipe.ARMS
    ]
    if (
        status.get("completed_runs") != expected_runs
        or status.get("complete") is not True
        or status.get("failed") is not False
    ):
        raise ValueError("calibration campaign did not complete exactly")
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13c_r3/calibration"
    rows: list[dict[str, Any]] = []
    for coefficient_name, coefficient in COEFFICIENTS:
        pair_tokens: dict[str, int] = {}
        pair_rows: list[dict[str, Any]] = []
        for arm in kl_recipe.ARMS:
            run_root = raw_root / coefficient_name / arm
            training_path = run_root / "training_summary.json"
            training = _read_object(training_path)
            _verify_hash(training, "result_sha256")
            measurement = cast(dict[str, Any], training["final_measurement"])
            _verify_hash(measurement, "measurement_sha256")
            adapter_path = run_root / "training/checkpoint-16/adapter"
            adapter_sha256 = str(training["checkpoints"]["16"]["adapter_sha256"])
            adapter_replay = directory_sha256(adapter_path)
            if adapter_replay != adapter_sha256:
                raise ValueError("calibration adapter directory hash differs")
            retention = {
                suite: _assessment_projection(
                    run_root / f"retention/{suite}_assessment.json",
                    adapter_sha256,
                )
                for suite in DEVELOPMENT_SUITES
            }
            historical_measurement = historical["arms"][arm]["measurement"]
            historical_kl = float(historical_measurement["replay_token_kl"])
            historical_validation = float(historical_measurement["vetted_validation_ce"])
            replay_kl = float(measurement["replay_token_kl"])
            validation_ce = float(measurement["vetted_validation_ce"])
            pair_tokens[arm] = int(training["loss_bearing_tokens"])
            criteria: dict[str, bool] = {
                "exactly_16_optimizer_steps": training["optimizer_steps"] == 16,
                "exact_16000_token_schedule_prefix": (
                    training["loss_bearing_tokens"] == 16_000
                    and measurement["total_tokens"] == 16_000
                ),
                "generic_targeted_token_parity": True,
                "finite_ce_kl_total_and_gradients": _all_finite(training),
                "lora_parameters_updated": training["lora_updated"] is True,
                "base_parameters_unchanged": (
                    training["base_parameters_unchanged"] is True
                    and training["base_parameter_fingerprint_before"]
                    == training["base_parameter_fingerprint_after"]
                    and training["base_restoration"] is True
                ),
                "no_cpu_offload": (
                    training["cpu_offload"] is False
                    and training["cuda_only"] is True
                    and training["offline_reload"] is True
                ),
                "both_development_retention_subsets_pass": all(
                    retention[suite]["gate_passed"] is True for suite in DEVELOPMENT_SUITES
                ),
                "replay_token_kl_at_most_75_percent_historical": (
                    replay_kl <= 0.75 * historical_kl
                ),
                "validation_ce_no_more_than_15_percent_worse": (
                    validation_ce <= 1.15 * historical_validation
                ),
                "zero_backend_failures": all(
                    retention[suite]["backend_failures"] == 0 for suite in DEVELOPMENT_SUITES
                ),
                "deterministic_evidence_replay_passed": True,
                "exact_historical_v1_lora_configuration": (
                    training["recipe_sha256"]
                    == "3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"
                    and training["trainable_inventory"]["tensor_inventory_sha256"]
                    == ("d3edea65d6d09226eb743182474ea51b2af1c0f94b163812ce67913ffc865e78")
                    and training["reload_inventory"] == training["trainable_inventory"]
                ),
            }
            row: dict[str, Any] = {
                "coefficient_name": coefficient_name,
                "coefficient": coefficient,
                "arm": arm,
                "optimizer_steps": training["optimizer_steps"],
                "loss_bearing_tokens": training["loss_bearing_tokens"],
                "schedule_sha256": training["schedule_sha256"],
                "schedule_prefix_sha256": training["schedule_prefix_sha256"],
                "adapter_sha256": adapter_sha256,
                "checkpoint_sha256": adapter_sha256,
                "checkpoint_bytes": training["checkpoints"]["16"]["bytes"],
                "training_result_sha256": training["result_sha256"],
                "training_result_file_sha256": file_sha256(training_path),
                "final_measurement": measurement,
                "historical_replay_token_kl": historical_kl,
                "replay_kl_ratio_to_historical": replay_kl / historical_kl,
                "replay_kl_eligibility_ceiling": 0.75 * historical_kl,
                "historical_validation_ce": historical_validation,
                "validation_ce_eligibility_ceiling": 1.15 * historical_validation,
                "step_metrics_sha256": canonical_sha256(training["step_metrics"]),
                "finite_gradients": training["finite_gradients"],
                "lora_updated": training["lora_updated"],
                "base_parameters_unchanged": training["base_parameters_unchanged"],
                "cpu_offload": training["cpu_offload"],
                "offline_reload": training["offline_reload"],
                "runtime_seconds": training["runtime_seconds"],
                "peak_allocated_vram_bytes": training["peak_allocated_vram_bytes"],
                "peak_reserved_vram_bytes": training["peak_reserved_vram_bytes"],
                "peak_process_rss_bytes": training["peak_process_rss_bytes"],
                "development_retention": retention,
                "eligibility_criteria": criteria,
                "eligible": False,
                "holdout_v2_use": training["holdout_v2_use"],
                "gsm1k_use": training["gsm1k_use"],
            }
            pair_rows.append(row)
        parity = pair_tokens["generic"] == pair_tokens["targeted"] == 16_000
        for row in pair_rows:
            row["eligibility_criteria"]["generic_targeted_token_parity"] = parity
            row["eligible"] = all(row["eligibility_criteria"].values())
            row["calibration_run_sha256"] = canonical_sha256(row)
            rows.append(row)
    record: dict[str, Any] = {
        "schema_version": 1,
        "calibration_id": "foundry-replay-ce-token-kl-v1-calibration-v1",
        "objective_contract_sha256": OBJECTIVE_CONTRACT_SHA256,
        "historical_comparator_summary_sha256": historical["historical_comparator_summary_sha256"],
        "campaign_status_sha256": status["status_sha256"],
        "coefficient_order": [value for _, value in COEFFICIENTS],
        "arm_order": list(kl_recipe.ARMS),
        "run_count": len(rows),
        "optimizer_step_count": sum(int(row["optimizer_steps"]) for row in rows),
        "loss_bearing_token_count": sum(int(row["loss_bearing_tokens"]) for row in rows),
        "runs": rows,
        "all_backend_failures_zero": all(
            row["eligibility_criteria"]["zero_backend_failures"] for row in rows
        ),
        "all_evidence_replay_passed": all(
            row["eligibility_criteria"]["deterministic_evidence_replay_passed"] for row in rows
        ),
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    record["calibration_summary_sha256"] = canonical_sha256(record)
    return record


def selection_record(calibration: dict[str, Any]) -> dict[str, Any]:
    """Select the smallest common eligible coefficient or freeze no selection."""

    _verify_hash(calibration, "calibration_summary_sha256")
    coefficient_results: list[dict[str, Any]] = []
    selected: float | None = None
    for _, coefficient in COEFFICIENTS:
        rows = [
            row
            for row in cast(list[dict[str, Any]], calibration["runs"])
            if row["coefficient"] == coefficient
        ]
        common = len(rows) == 2 and all(row["eligible"] is True for row in rows)
        failed = {
            str(row["arm"]): [
                name
                for name, passed in cast(dict[str, bool], row["eligibility_criteria"]).items()
                if not passed
            ]
            for row in rows
        }
        coefficient_results.append(
            {
                "coefficient": coefficient,
                "common_eligible": common,
                "failed_criteria_by_arm": failed,
                "run_sha256_by_arm": {
                    str(row["arm"]): row["calibration_run_sha256"] for row in rows
                },
            }
        )
        if selected is None and common:
            selected = coefficient
    result: dict[str, Any] = {
        "schema_version": 1,
        "selection_id": "foundry-vetted-corpus-kl-coefficient-selection-v1",
        "calibration_summary_sha256": calibration["calibration_summary_sha256"],
        "selection_order": [value for _, value in COEFFICIENTS],
        "coefficient_results": coefficient_results,
        "selected_coefficient": selected,
        "decision": (
            "smallest_common_eligible_coefficient_selected"
            if selected is not None
            else "no_common_eligible_coefficient"
        ),
        "full_training_authorized": selected is not None,
        "stop_before_full_training": selected is None,
        "holdout_v2_used_for_selection": False,
        "gsm1k_used_for_selection": False,
        "targeted_vs_generic_benchmark_used_for_selection": False,
        "final_training_loss_only_used_for_selection": False,
        "sealed_paths_accessed": False,
    }
    result["coefficient_selection_sha256"] = canonical_sha256(result)
    return result


def blocker_record(calibration: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Freeze the mandatory no-coefficient calibration stop."""

    _verify_hash(calibration, "calibration_summary_sha256")
    _verify_hash(selection, "coefficient_selection_sha256")
    if selection["selected_coefficient"] is not None:
        raise ValueError("calibration blocker requires no selected coefficient")
    runs = cast(list[dict[str, Any]], calibration["runs"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "blocker_id": "foundry-milestone13c-r3-kl-calibration-blocker-v1",
        "reason": "no_common_coefficient_met_replay_kl_at_most_75_percent",
        "calibration_summary_sha256": calibration["calibration_summary_sha256"],
        "coefficient_selection_sha256": selection["coefficient_selection_sha256"],
        "coefficient_count": 4,
        "arm_count": 2,
        "run_count": 8,
        "all_development_retention_passed": all(
            row["eligibility_criteria"]["both_development_retention_subsets_pass"] for row in runs
        ),
        "all_validation_ce_gates_passed": all(
            row["eligibility_criteria"]["validation_ce_no_more_than_15_percent_worse"]
            for row in runs
        ),
        "all_replay_kl_gates_failed": all(
            not row["eligibility_criteria"]["replay_token_kl_at_most_75_percent_historical"]
            for row in runs
        ),
        "replay_kl_ratio_by_run": {
            f"{row['coefficient_name']}/{row['arm']}": row["replay_kl_ratio_to_historical"]
            for row in runs
        },
        "full_training_runs": 0,
        "holdout_v2_adapter_evaluations": 0,
        "gsm1k_adapter_evaluations": 0,
        "sealed_paths_accessed": False,
        "required_stop": "before_full_training",
    }
    result["calibration_blocker_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    historical = subparsers.add_parser("historical")
    historical.add_argument("--root", type=Path, required=True)
    historical.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--root", type=Path, required=True)
    finalize.add_argument("--calibration-output", type=Path, required=True)
    finalize.add_argument("--selection-output", type=Path, required=True)
    finalize.add_argument("--blocker-output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "historical":
        if args.output.exists():
            raise FileExistsError("historical comparator output already exists")
        result = historical_record(args.root.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True))
        return
    outputs = (
        args.calibration_output,
        args.selection_output,
        args.blocker_output,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("calibration publication output already exists")
    calibration = calibration_record(args.root.resolve())
    selection = selection_record(calibration)
    blocker = blocker_record(calibration, selection)
    for path, value in zip(outputs, (calibration, selection, blocker), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(blocker, sort_keys=True))


if __name__ == "__main__":
    main()
