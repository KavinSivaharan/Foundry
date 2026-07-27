"""Replay and adjudicate Milestone 13D smoke and calibration evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

ARMS = ("generic", "targeted")
RHO_LABELS = (
    ("rho-010", "0.10"),
    ("rho-030", "0.30"),
    ("rho-100", "1.00"),
    ("rho-300", "3.00"),
)
DEVELOPMENT_SUITES = {
    "adjudication": "5caf23be79fa01151af6f7db8d45c2b85bfe24b03a29589e482d51731c8358af",
    "anchor": "bff18b434a284d848387262dde201601278e5c8b573937b3486bed2bf925696e",
}
HISTORICAL_COMPARATOR_SHA256 = "06053527fdb5786ace22972aab642fd824b8471ee72e7e19ef849520f5d33324"


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


def _ladder(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    value = _read(root / "results/phase2_vetted_corpus/milestone13d_gradient_ladder.json")
    _verify(value, "ladder_sha256")
    coefficients = {
        str(row["rho_exact"]): str(row["lambda_common_exact"])
        for row in cast(list[dict[str, Any]], value["ladder"])
    }
    if list(coefficients) != [rho for _, rho in RHO_LABELS]:
        raise ValueError("frozen rho ladder order differs")
    return value, coefficients


def _training_finite(training: dict[str, Any]) -> bool:
    step_metrics = cast(list[dict[str, Any]], training["step_metrics"])
    measurement = cast(dict[str, Any], training["final_measurement"])
    values = [
        float(row[key])
        for row in step_metrics
        for key in ("vetted_ce", "replay_ce", "replay_kl", "total_loss")
    ]
    values.extend(
        float(measurement[key])
        for key in (
            "vetted_ce",
            "replay_ce",
            "replay_token_kl",
            "vetted_validation_ce",
        )
    )
    return (
        training.get("finite_gradients") is True
        and training.get("finite_losses") is True
        and all(math.isfinite(value) for value in values)
    )


def smoke_record(root: Path) -> dict[str, Any]:
    """Replay every two-step cell and mark unsafe coefficients ineligible."""

    ladder, coefficients = _ladder(root)
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13d/smoke"
    rows: list[dict[str, Any]] = []
    for label, rho in RHO_LABELS:
        for arm in ARMS:
            run_root = raw_root / label / arm
            summary_path = run_root / "training_summary.json"
            training = _read(summary_path)
            _verify(training, "result_sha256")
            adapter_path = run_root / "training/checkpoint-2/adapter"
            adapter_sha256 = directory_sha256(adapter_path)
            if adapter_sha256 != training["checkpoint"]["adapter_sha256"]:
                raise ValueError("smoke adapter directory hash differs")
            criteria: dict[str, bool] = {
                "exactly_two_optimizer_steps": training["optimizer_steps"] == 2,
                "exact_first_replay_containing_schedule_segment": (
                    training["loss_bearing_tokens"] == training["expected_loss_bearing_tokens"]
                ),
                "finite_ce_kl_total_and_gradients": _training_finite(training),
                "first_positive_lr_step_updates_lora": (
                    training["first_positive_lr_step_updated_lora"] is True
                ),
                "base_parameters_unchanged": (
                    training["base_parameters_unchanged"] is True
                    and training["base_parameter_fingerprint_before"]
                    == training["base_parameter_fingerprint_after"]
                    and training["base_restoration"] is True
                ),
                "no_cpu_offload": (
                    training["cpu_offload"] is False and training["cuda_only"] is True
                ),
                "no_overflow_or_nan": training["overflow_or_nan"] is False,
                "replay_kl_did_not_increase_above_10x_historical": (
                    training["smoke_replay_kl_within_10x_historical"] is True
                ),
                "adapter_save_and_offline_reload": (
                    training["offline_reload"] is True
                    and training["reload_inventory"] == training["trainable_inventory"]
                ),
                "deterministic_summary_replay": True,
                "exact_v1_equivalent_configuration": (
                    training["recipe_sha256"]
                    == "3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"
                    and training["trainable_inventory"]["tensor_inventory_sha256"]
                    == "d3edea65d6d09226eb743182474ea51b2af1c0f94b163812ce67913ffc865e78"
                ),
                "holdout_v2_unexposed": training["holdout_v2_use"] is False,
                "gsm1k_unused": training["gsm1k_use"] is False,
            }
            row: dict[str, Any] = {
                "rho_label": label,
                "rho_exact": rho,
                "coefficient_exact": coefficients[rho],
                "arm": arm,
                "optimizer_steps": training["optimizer_steps"],
                "loss_bearing_tokens": training["loss_bearing_tokens"],
                "adapter_sha256": adapter_sha256,
                "adapter_bytes": training["checkpoint"]["bytes"],
                "training_result_sha256": training["result_sha256"],
                "training_file_sha256": file_sha256(summary_path),
                "step_metrics": training["step_metrics"],
                "gradient_measurements": training["gradient_measurements"],
                "smoke_post_step_measurements": training["smoke_post_step_measurements"],
                "final_measurement": training["final_measurement"],
                "runtime_seconds": training["runtime_seconds"],
                "peak_allocated_vram_bytes": training["peak_allocated_vram_bytes"],
                "peak_reserved_vram_bytes": training["peak_reserved_vram_bytes"],
                "peak_process_rss_bytes": training["peak_process_rss_bytes"],
                "criteria": criteria,
                "eligible": all(criteria.values()),
            }
            row["smoke_run_sha256"] = canonical_sha256(row)
            rows.append(row)
    coefficient_results = []
    for label, rho in RHO_LABELS:
        pair = [row for row in rows if row["rho_exact"] == rho]
        coefficient_results.append(
            {
                "rho_label": label,
                "rho_exact": rho,
                "coefficient_exact": coefficients[rho],
                "common_eligible": len(pair) == 2 and all(row["eligible"] for row in pair),
                "run_sha256_by_arm": {str(row["arm"]): row["smoke_run_sha256"] for row in pair},
                "failed_criteria_by_arm": {
                    str(row["arm"]): [
                        name
                        for name, passed in cast(dict[str, bool], row["criteria"]).items()
                        if not passed
                    ]
                    for row in pair
                },
            }
        )
    record: dict[str, Any] = {
        "schema_version": 1,
        "smoke_id": "foundry-milestone13d-gradient-kl-compatibility-smoke-v1",
        "ladder_sha256": ladder["ladder_sha256"],
        "rho_order": [rho for _, rho in RHO_LABELS],
        "arm_order": list(ARMS),
        "run_count": len(rows),
        "runs": rows,
        "coefficient_results": coefficient_results,
        "every_coefficient_unsafe": all(not row["common_eligible"] for row in coefficient_results),
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    record["smoke_summary_sha256"] = canonical_sha256(record)
    return record


def _assessment(path: Path, adapter_sha256: str) -> dict[str, Any]:
    value = _read(path)
    _verify(value, "summary_sha256")
    if value.get("adapter_sha256") != adapter_sha256:
        raise ValueError("development-retention adapter hash differs")
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


def calibration_record(root: Path, smoke: dict[str, Any]) -> dict[str, Any]:
    """Replay every smoke-eligible 16-step cell and apply unchanged gates."""

    _verify(smoke, "smoke_summary_sha256")
    ladder, coefficients = _ladder(root)
    historical = _read(
        root / "results/phase2_vetted_corpus/milestone13c_r3_historical_comparator.json"
    )
    _verify(historical, "historical_comparator_summary_sha256")
    if historical["historical_comparator_summary_sha256"] != HISTORICAL_COMPARATOR_SHA256:
        raise ValueError("historical comparator identity differs")
    common_smoke = {
        str(row["rho_exact"]): bool(row["common_eligible"])
        for row in cast(list[dict[str, Any]], smoke["coefficient_results"])
    }
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13d/calibration"
    rows: list[dict[str, Any]] = []
    for label, rho in RHO_LABELS:
        if not common_smoke[rho]:
            continue
        pair: list[dict[str, Any]] = []
        for arm in ARMS:
            run_root = raw_root / label / arm
            summary_path = run_root / "training_summary.json"
            training = _read(summary_path)
            _verify(training, "result_sha256")
            adapter_path = run_root / "training/checkpoint-16/adapter"
            adapter_sha256 = directory_sha256(adapter_path)
            if adapter_sha256 != training["checkpoint"]["adapter_sha256"]:
                raise ValueError("calibration adapter directory hash differs")
            retention = {
                suite: _assessment(
                    run_root / f"retention/{suite}_assessment.json",
                    adapter_sha256,
                )
                for suite in DEVELOPMENT_SUITES
            }
            measured = cast(dict[str, Any], training["final_measurement"])
            historical_measured = historical["arms"][arm]["measurement"]
            replay_kl = float(measured["replay_token_kl"])
            historical_kl = float(historical_measured["replay_token_kl"])
            validation_ce = float(measured["vetted_validation_ce"])
            historical_validation = float(historical_measured["vetted_validation_ce"])
            gradient_steps = cast(dict[str, Any], training["gradient_measurements"])
            criteria: dict[str, bool] = {
                "exactly_16_optimizer_steps": training["optimizer_steps"] == 16,
                "exact_16000_token_schedule_prefix": (
                    training["loss_bearing_tokens"] == 16_000 and measured["total_tokens"] == 16_000
                ),
                "generic_targeted_assistant_token_parity": True,
                "finite_ce_kl_total_and_gradients": _training_finite(training),
                "lora_parameters_update": training["lora_updated"] is True,
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
                "exact_v1_equivalent_lora_configuration": (
                    training["recipe_sha256"]
                    == "3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"
                    and training["trainable_inventory"]["tensor_inventory_sha256"]
                    == "d3edea65d6d09226eb743182474ea51b2af1c0f94b163812ce67913ffc865e78"
                    and training["reload_inventory"] == training["trainable_inventory"]
                ),
                "holdout_v2_unexposed": training["holdout_v2_use"] is False,
                "gsm1k_unused": training["gsm1k_use"] is False,
                "gradient_ratio_measurements_complete": (
                    set(gradient_steps) == {"2", "8", "16"}
                    and all(
                        gradient_steps[step]["finite_gradients"] is True
                        and gradient_steps[step]["base_gradient_count"] == 0
                        and gradient_steps[step]["reference_gradient_count"] == 0
                        for step in ("2", "8", "16")
                    )
                ),
            }
            row: dict[str, Any] = {
                "rho_label": label,
                "rho_exact": rho,
                "coefficient_exact": coefficients[rho],
                "arm": arm,
                "optimizer_steps": training["optimizer_steps"],
                "loss_bearing_tokens": training["loss_bearing_tokens"],
                "schedule_sha256": training["schedule_sha256"],
                "schedule_prefix_sha256": training["schedule_prefix_sha256"],
                "adapter_sha256": adapter_sha256,
                "adapter_bytes": training["checkpoint"]["bytes"],
                "training_result_sha256": training["result_sha256"],
                "training_file_sha256": file_sha256(summary_path),
                "final_measurement": measured,
                "historical_replay_token_kl": historical_kl,
                "replay_kl_ratio_to_historical": replay_kl / historical_kl,
                "historical_validation_ce": historical_validation,
                "validation_ce_ratio_to_historical": (validation_ce / historical_validation),
                "gradient_measurements": gradient_steps,
                "step_metrics_sha256": canonical_sha256(training["step_metrics"]),
                "development_retention": retention,
                "criteria": criteria,
                "eligible": False,
                "runtime_seconds": training["runtime_seconds"],
                "peak_allocated_vram_bytes": training["peak_allocated_vram_bytes"],
                "peak_reserved_vram_bytes": training["peak_reserved_vram_bytes"],
                "peak_process_rss_bytes": training["peak_process_rss_bytes"],
                "holdout_v2_use": False,
                "gsm1k_use": False,
            }
            pair.append(row)
        parity = all(row["loss_bearing_tokens"] == 16_000 for row in pair)
        for row in pair:
            row["criteria"]["generic_targeted_assistant_token_parity"] = parity
            row["eligible"] = all(row["criteria"].values())
            row["calibration_run_sha256"] = canonical_sha256(row)
            rows.append(row)
    record: dict[str, Any] = {
        "schema_version": 1,
        "calibration_id": "foundry-milestone13d-gradient-kl-calibration-v1",
        "ladder_sha256": ladder["ladder_sha256"],
        "smoke_summary_sha256": smoke["smoke_summary_sha256"],
        "historical_comparator_sha256": HISTORICAL_COMPARATOR_SHA256,
        "rho_order": [rho for _, rho in RHO_LABELS],
        "arm_order": list(ARMS),
        "run_count": len(rows),
        "optimizer_step_count": sum(int(row["optimizer_steps"]) for row in rows),
        "loss_bearing_token_count": sum(int(row["loss_bearing_tokens"]) for row in rows),
        "runs": rows,
        "all_backend_failures_zero": all(row["criteria"]["zero_backend_failures"] for row in rows),
        "all_evidence_replay_passed": all(
            row["criteria"]["deterministic_evidence_replay_passed"] for row in rows
        ),
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    record["calibration_summary_sha256"] = canonical_sha256(record)
    return record


def selection_record(
    calibration: dict[str, Any],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    """Select the smallest common eligible rho without holdout or GSM1K."""

    _verify(calibration, "calibration_summary_sha256")
    _verify(smoke, "smoke_summary_sha256")
    results = []
    selected_rho: str | None = None
    selected_coefficient: str | None = None
    for _, rho in RHO_LABELS:
        rows = [
            row
            for row in cast(list[dict[str, Any]], calibration["runs"])
            if row["rho_exact"] == rho
        ]
        common = len(rows) == 2 and all(row["eligible"] is True for row in rows)
        coefficient = (
            str(rows[0]["coefficient_exact"])
            if rows
            else next(
                str(row["coefficient_exact"])
                for row in smoke["coefficient_results"]
                if row["rho_exact"] == rho
            )
        )
        results.append(
            {
                "rho_exact": rho,
                "coefficient_exact": coefficient,
                "common_eligible": common,
                "failed_criteria_by_arm": {
                    str(row["arm"]): [
                        name
                        for name, passed in cast(dict[str, bool], row["criteria"]).items()
                        if not passed
                    ]
                    for row in rows
                },
                "run_sha256_by_arm": {
                    str(row["arm"]): row["calibration_run_sha256"] for row in rows
                },
            }
        )
        if common and selected_rho is None:
            selected_rho = rho
            selected_coefficient = coefficient
    record: dict[str, Any] = {
        "schema_version": 1,
        "selection_id": "foundry-milestone13d-gradient-kl-coefficient-selection-v1",
        "smoke_summary_sha256": smoke["smoke_summary_sha256"],
        "calibration_summary_sha256": calibration["calibration_summary_sha256"],
        "selection_order": [rho for _, rho in RHO_LABELS],
        "coefficient_results": results,
        "selected_rho": selected_rho,
        "selected_coefficient_exact": selected_coefficient,
        "decision": (
            "smallest_common_eligible_rho_selected"
            if selected_rho is not None
            else "no_common_eligible_gradient_scaled_coefficient"
        ),
        "full_training_run_in_milestone": False,
        "holdout_v2_used_for_selection": False,
        "gsm1k_used_for_selection": False,
        "sealed_paths_accessed": False,
    }
    record["coefficient_selection_sha256"] = canonical_sha256(record)
    return record


def blocker_record(
    calibration: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Freeze token-level KL closure only when no common rho is eligible."""

    _verify(calibration, "calibration_summary_sha256")
    _verify(selection, "coefficient_selection_sha256")
    if selection["selected_rho"] is not None:
        raise ValueError("no-coefficient blocker requires no selected rho")
    runs = cast(list[dict[str, Any]], calibration["runs"])
    record: dict[str, Any] = {
        "schema_version": 1,
        "blocker_id": "foundry-milestone13d-gradient-kl-calibration-blocker-v1",
        "reason": "no_common_gradient_scaled_coefficient_met_all_unchanged_gates",
        "calibration_summary_sha256": calibration["calibration_summary_sha256"],
        "coefficient_selection_sha256": selection["coefficient_selection_sha256"],
        "run_count": len(runs),
        "replay_kl_ratio_by_run": {
            f"{row['rho_label']}/{row['arm']}": row["replay_kl_ratio_to_historical"] for row in runs
        },
        "full_training_runs": 0,
        "holdout_v2_adapter_evaluations": 0,
        "gsm1k_adapter_evaluations": 0,
        "sealed_paths_accessed": False,
        "required_stop": "before_full_training",
    }
    record["calibration_blocker_sha256"] = canonical_sha256(record)
    return record


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--root", type=Path, required=True)
    smoke_parser.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--root", type=Path, required=True)
    finalize.add_argument("--smoke", type=Path, required=True)
    finalize.add_argument("--calibration-output", type=Path, required=True)
    finalize.add_argument("--selection-output", type=Path, required=True)
    finalize.add_argument("--blocker-output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "smoke":
        value = smoke_record(root)
        _write_new(args.output, value)
        print(json.dumps(value, sort_keys=True))
        return
    smoke = _read(args.smoke)
    calibration = calibration_record(root, smoke)
    selection = selection_record(calibration, smoke)
    _write_new(args.calibration_output, calibration)
    _write_new(args.selection_output, selection)
    if selection["selected_rho"] is None:
        if args.blocker_output is None:
            raise ValueError("no selection requires a blocker output")
        _write_new(args.blocker_output, blocker_record(calibration, selection))
    print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
