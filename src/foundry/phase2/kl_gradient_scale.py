"""Build content-free Milestone 13D KL gradient-scale evidence."""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_objective_contract
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

DATASET_SHA256 = "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"
ARCHITECTURE_DECISION_SHA256 = "74907ea92b2217b6f9ca39044feab6c6452600e7774a2b442e0ec9e29b6899a5"
RECIPE_SHA256 = "3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"
RECIPE_DECISION_SHA256 = "b03dfc9d6f66843f2a83e9b4ff5b82e133fb9c6a4a27ab1783d64991a0f7d118"
OBJECTIVE_SOURCE_SHA256 = "daacbe91d4d6352166ec857e4074f2cfb6261b97319bf0ffa62c6298a7197515"
OBJECTIVE_CONFIGURATION_SHA256 = "b00a0c2c8fbc23031c8fa5c433fe6bea42f0fb1e9dc48f81d0a8148f9526cbd7"
MASKING_SHA256 = "3858c6db0da9dbcf3a1149be2aa677ba0cdc52ce22c19de08a8d47aeaab8df4d"
REFERENCE_SHA256 = "c694138040fab03b4d98d1089eacf4306f8cca438b12061a9b815eea033a9b43"
FIXTURE_SHA256 = "351476cf2ba3780914a2f3a8aea926256c305ff8f2c6751dde997d6de1591e0b"
OBJECTIVE_CONTRACT_SHA256 = "159ef322b254d5f70b46296fe051417ed53c1543f3432bcb0b9069d66ab975a8"
HISTORICAL_COMPARATOR_SHA256 = "06053527fdb5786ace22972aab642fd824b8471ee72e7e19ef849520f5d33324"
PREVIOUS_CALIBRATION_SHA256 = "e1d0af0211bcb3c4e60c983ef3f2511abe88fb5ef3bf225a3d5d936b9e22305e"
PREVIOUS_SELECTION_SHA256 = "3db161b50423115f8737f52ff1d174ee943a1d013a8855fc729e27786399a3f3"
PREVIOUS_BLOCKER_SHA256 = "8f25722d6c7b1a48f5f41e20edbece94d69b25d86c50e3d9d62336f59fb4cf75"
ARMS = ("generic", "targeted")
PREVIOUS_COEFFICIENTS = (0.01, 0.03, 0.10, 0.30)
RHO_TARGETS = ("0.10", "0.30", "1.00", "3.00")
EXPECTED_ADAPTERS = {
    "generic": "cf230487953cf347824a40faa36ad6b1b93ef667119f83766dce1d26e72ba63e",
    "targeted": "6915af1d2ab6bd8ac75b538417e1b3af7395924e77c1150138bc06bc4773c5e9",
}


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify(value: dict[str, Any], key: str, expected: str | None = None) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")
    if expected is not None and supplied != expected:
        raise ValueError(f"{key} differs from its frozen identity")


def raw_scale_analysis(root: Path) -> dict[str, Any]:
    """Describe the original coefficient ladder without selecting from loss scale."""

    results = root / "results/phase2_vetted_corpus"
    historical = _read(results / "milestone13c_r3_historical_comparator.json")
    calibration = _read(results / "milestone13c_r3_kl_calibration.json")
    _verify(
        historical,
        "historical_comparator_summary_sha256",
        HISTORICAL_COMPARATOR_SHA256,
    )
    _verify(calibration, "calibration_summary_sha256", PREVIOUS_CALIBRATION_SHA256)
    rows: dict[str, Any] = {}
    for arm in ARMS:
        measured = cast(dict[str, Any], historical["arms"][arm]["measurement"])
        replay_ce = float(measured["replay_ce"])
        replay_kl = float(measured["replay_token_kl"])
        validation_ce = float(measured["vetted_validation_ce"])
        if not all(math.isfinite(value) and value > 0.0 for value in (replay_ce, replay_kl)):
            raise ValueError("historical replay CE or KL is not finite and positive")
        coefficient_rows: list[dict[str, Any]] = []
        for coefficient in PREVIOUS_COEFFICIENTS:
            matches = [
                row
                for row in cast(list[dict[str, Any]], calibration["runs"])
                if row["arm"] == arm and float(row["coefficient"]) == coefficient
            ]
            if len(matches) != 1:
                raise ValueError("previous calibration cell is absent or duplicated")
            run = matches[0]
            final = cast(dict[str, Any], run["final_measurement"])
            weighted = coefficient * replay_kl
            coefficient_rows.append(
                {
                    "coefficient": coefficient,
                    "historical_weighted_kl_loss": weighted,
                    "historical_weighted_kl_percent_of_replay_ce": (100.0 * weighted / replay_ce),
                    "calibration_replay_kl_ratio_to_historical": (
                        float(final["replay_token_kl"]) / replay_kl
                    ),
                    "calibration_validation_ce_ratio_to_historical": (
                        float(final["vetted_validation_ce"]) / validation_ce
                    ),
                    "calibration_run_sha256": run["calibration_run_sha256"],
                }
            )
        equality = replay_ce / replay_kl
        rows[arm] = {
            "historical_replay_ce": replay_ce,
            "historical_replay_token_kl": replay_kl,
            "historical_validation_ce": validation_ce,
            "previous_coefficients": coefficient_rows,
            "lambda_for_weighted_kl_equals_replay_ce": equality,
            "lambda_by_replay_ce_fraction": {
                "0.01": 0.01 * equality,
                "0.10": 0.10 * equality,
                "0.30": 0.30 * equality,
                "1.00": equality,
            },
        }
    record: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": "foundry-milestone13d-original-kl-raw-scale-v1",
        "dataset_sha256": DATASET_SHA256,
        "architecture_decision_sha256": ARCHITECTURE_DECISION_SHA256,
        "objective_contract_sha256": OBJECTIVE_CONTRACT_SHA256,
        "historical_comparator_sha256": HISTORICAL_COMPARATOR_SHA256,
        "previous_calibration_sha256": PREVIOUS_CALIBRATION_SHA256,
        "arms": rows,
        "selection_from_raw_loss_scale": False,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    record["raw_scale_analysis_sha256"] = canonical_sha256(record)
    return record


def gradient_audit_record(root: Path) -> dict[str, Any]:
    """Validate both ignored raw historical gradient audits and publish their metrics."""

    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13d/gradient"
    arms: dict[str, Any] = {}
    manifests: list[dict[str, Any]] = []
    for arm in ARMS:
        path = raw_root / f"{arm}.json"
        value = _read(path)
        _verify(value, "result_sha256")
        if (
            value.get("arm") != arm
            or value.get("adapter_sha256") != EXPECTED_ADAPTERS[arm]
            or value.get("adapter_unchanged") is not True
            or value.get("model_state_unchanged") is not True
            or value.get("duplicate_measurement_identical") is not True
            or value.get("holdout_v2_use") is not False
            or value.get("gsm1k_use") is not False
        ):
            raise ValueError("historical gradient audit identity or boundary differs")
        measurements = cast(list[dict[str, Any]], value["measurements"])
        if len(measurements) != 2 or measurements[0] != measurements[1]:
            raise ValueError("historical duplicate gradient measurements differ")
        measurement = measurements[0]
        required = (
            bool(measurement["finite_gradients"])
            and float(measurement["ce_global_l2_norm"]) > 0.0
            and float(measurement["kl_global_l2_norm"]) > 0.0
            and int(measurement["base_gradient_count"]) == 0
            and int(measurement["reference_gradient_count"]) == 0
        )
        if not required:
            raise ValueError("historical gradient validity gate failed")
        manifest = cast(dict[str, Any], value["measurement_manifest"])
        manifests.append(manifest)
        arms[arm] = {
            "adapter_sha256": value["adapter_sha256"],
            "adapter_config_file_sha256": value["adapter_config_file_sha256"],
            "raw_result_sha256": value["result_sha256"],
            "raw_file_sha256": file_sha256(path),
            "measurement": measurement,
            "runtime_seconds": value["runtime_seconds"],
            "peak_allocated_vram_bytes": value["peak_allocated_vram_bytes"],
            "peak_reserved_vram_bytes": value["peak_reserved_vram_bytes"],
            "peak_process_rss_bytes": value["peak_process_rss_bytes"],
            "launch_evidence": value["launch_evidence"],
        }
    if manifests[0] != manifests[1]:
        raise ValueError("generic and targeted replay gradient corpora differ")
    manifest = manifests[0]
    if (
        manifest.get("replay_assistant_token_count") != 4_000
        or manifest.get("source_schedule_prefix_assistant_tokens") != 16_000
        or manifest.get("generic_targeted_identical") is not True
    ):
        raise ValueError("replay gradient measurement corpus differs")
    manifest_record: dict[str, Any] = {
        "schema_version": 1,
        "manifest_id": "foundry-milestone13d-replay-gradient-measurement-corpus-v1",
        **manifest,
        "vetted_corpus_use": False,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    manifest_record["measurement_manifest_sha256"] = canonical_sha256(manifest_record)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-milestone13d-historical-gradient-scale-audit-v1",
        "dataset_sha256": DATASET_SHA256,
        "architecture_decision_sha256": ARCHITECTURE_DECISION_SHA256,
        "recipe_sha256": RECIPE_SHA256,
        "objective_contract_sha256": OBJECTIVE_CONTRACT_SHA256,
        "measurement_manifest_sha256": manifest_record["measurement_manifest_sha256"],
        "arms": arms,
        "all_gradients_finite_positive_and_connected": True,
        "all_base_and_reference_gradient_counts_zero": True,
        "all_duplicate_measurements_identical": True,
        "all_adapters_and_model_states_unchanged": True,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    audit["gradient_audit_sha256"] = canonical_sha256(audit)
    return {"manifest": manifest_record, "audit": audit}


def objective_graph_integrity(root: Path, gradient_audit: dict[str, Any]) -> dict[str, Any]:
    """Bind the frozen graph contract to the observed nonzero LoRA KL gradients."""

    objective = _read(root / "results/phase2_vetted_corpus/milestone13c_r3_kl_objective.json")
    kl_objective_contract.validate(root, objective)
    if (
        objective["objective_contract_sha256"] != OBJECTIVE_CONTRACT_SHA256
        or objective["source"]["objective_source_sha256"] != OBJECTIVE_SOURCE_SHA256
        or objective["configuration"]["objective_configuration_sha256"]
        != OBJECTIVE_CONFIGURATION_SHA256
        or objective["masking"]["masking_sha256"] != MASKING_SHA256
        or objective["reference"]["reference_mechanism_sha256"] != REFERENCE_SHA256
        or objective["fixture"]["fixture_sha256"] != FIXTURE_SHA256
    ):
        raise ValueError("frozen objective identities differ")
    _verify(gradient_audit, "gradient_audit_sha256")
    record: dict[str, Any] = {
        "schema_version": 1,
        "integrity_id": "foundry-milestone13d-kl-computation-graph-integrity-v1",
        "objective_source_sha256": OBJECTIVE_SOURCE_SHA256,
        "objective_configuration_sha256": OBJECTIVE_CONFIGURATION_SHA256,
        "masking_sha256": MASKING_SHA256,
        "reference_mechanism_sha256": REFERENCE_SHA256,
        "objective_fixture_sha256": FIXTURE_SHA256,
        "objective_contract_sha256": OBJECTIVE_CONTRACT_SHA256,
        "gradient_audit_sha256": gradient_audit["gradient_audit_sha256"],
        "policy_logits_source": "active_adapter",
        "reference_logits_source": "same_model_adapter_disabled",
        "active_reference_distinct_state_fixture_passed": True,
        "policy_logits_detached": False,
        "reference_logits_detached": True,
        "kl_tensor_detached_before_coefficient": False,
        "kl_detachment_failure_fixture_passed": True,
        "kl_direction": "KL(base||adapter)",
        "mask": "labels[:,1:]!=-100",
        "excluded_positions": [
            "system",
            "user",
            "assistant_header",
            "padding",
            "post_eos",
        ],
        "vocabulary_reduction_before_token_average": True,
        "token_average_before_coefficient": True,
        "probability_dtype": "float32",
        "vetted_corpus_kl": False,
        "base_gradient_count": 0,
        "reference_gradient_count": 0,
        "nonzero_lora_kl_gradient": True,
        "base_parameters_unchanged": True,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    record["objective_graph_integrity_sha256"] = canonical_sha256(record)
    return record


def _gradient_band(value: float) -> str:
    if value < 0.01:
        return "less_than_1_percent"
    if value < 0.10:
        return "1_to_10_percent"
    if value < 0.30:
        return "10_to_30_percent"
    if value <= 1.0:
        return "30_to_100_percent"
    return "greater_than_100_percent"


def previous_ladder_interpretation(gradient_audit: dict[str, Any]) -> dict[str, Any]:
    """Interpret the failed ladder using measured LoRA-gradient norms."""

    _verify(gradient_audit, "gradient_audit_sha256")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        measurement = gradient_audit["arms"][arm]["measurement"]
        ratio = float(measurement["kl_to_ce_gradient_norm_ratio"])
        rows = []
        for coefficient in PREVIOUS_COEFFICIENTS:
            contribution = coefficient * ratio
            rows.append(
                {
                    "coefficient": coefficient,
                    "weighted_kl_to_ce_gradient_ratio": contribution,
                    "classification": _gradient_band(contribution),
                }
            )
        arms[arm] = {
            "unweighted_kl_to_ce_gradient_norm_ratio": ratio,
            "coefficients": rows,
        }
    record: dict[str, Any] = {
        "schema_version": 1,
        "interpretation_id": "foundry-milestone13d-previous-kl-ladder-gradient-scale-v1",
        "gradient_audit_sha256": gradient_audit["gradient_audit_sha256"],
        "previous_coefficients": list(PREVIOUS_COEFFICIENTS),
        "arms": arms,
        "previous_result_redefined": False,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    record["previous_ladder_gradient_scale_sha256"] = canonical_sha256(record)
    return record


def derive_common_ladder(gradient_audit: dict[str, Any]) -> dict[str, Any]:
    """Derive the frozen common ladder using Decimal arithmetic."""

    _verify(gradient_audit, "gradient_audit_sha256")
    arm_ratios: dict[str, str] = {}
    with localcontext() as context:
        context.prec = 50
        for arm in ARMS:
            measurement = gradient_audit["arms"][arm]["measurement"]
            ce = Decimal(str(measurement["ce_global_l2_norm"]))
            kl = Decimal(str(measurement["kl_global_l2_norm"]))
            if not ce.is_finite() or not kl.is_finite() or ce <= 0 or kl <= 0:
                raise ValueError("gradient norm is not finite and positive")
            arm_ratios[arm] = str(ce / kl)
        ladder: list[dict[str, Any]] = []
        for rho_text in RHO_TARGETS:
            rho = Decimal(rho_text)
            lambdas = {arm: rho * Decimal(arm_ratios[arm]) for arm in ARMS}
            common = max(lambdas.values())
            if common > Decimal("1000000"):
                raise ValueError("derived common coefficient exceeds 1,000,000")
            ladder.append(
                {
                    "rho_exact": rho_text,
                    "lambda_by_arm_exact": {arm: str(lambdas[arm]) for arm in ARMS},
                    "lambda_common_exact": str(common),
                    "lambda_common_rendered": format(common, ".12g"),
                    "guaranteed_minimum_weighted_kl_to_ce_gradient_ratio": rho_text,
                }
            )
    exact_values = [Decimal(str(row["lambda_common_exact"])) for row in ladder]
    if any(not value.is_finite() or value <= 0 for value in exact_values) or any(
        right <= left for left, right in zip(exact_values, exact_values[1:], strict=False)
    ):
        raise ValueError("derived common ladder is not finite, positive, and strictly increasing")
    derivation_contract: dict[str, Any] = {
        "formula_by_arm": "lambda_arm(rho)=rho*||g_CE||/||g_KL||",
        "common_formula": "lambda_common(rho)=max(lambda_generic,lambda_targeted)",
        "numeric_type": "Decimal",
        "decimal_precision": 50,
        "rho_targets_in_order": list(RHO_TARGETS),
        "arm_order": list(ARMS),
        "maximum_allowed_coefficient": "1000000",
        "architecture_unchanged": "replay-ce-token-kl-v1",
    }
    derivation_contract["derivation_contract_sha256"] = canonical_sha256(derivation_contract)
    record: dict[str, Any] = {
        "schema_version": 1,
        "ladder_id": "foundry-milestone13d-common-gradient-calibrated-kl-ladder-v1",
        "gradient_audit_sha256": gradient_audit["gradient_audit_sha256"],
        "derivation_contract": derivation_contract,
        "ce_to_kl_gradient_norm_ratio_exact": arm_ratios,
        "ladder": ladder,
        "frozen_before_coefficient_execution": True,
        "coefficient_add_remove_reorder_after_freeze": False,
        "architecture": "replay-ce-token-kl-v1",
        "objective_direction_changed": False,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
    }
    record["ladder_sha256"] = canonical_sha256(record)
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
    raw = subparsers.add_parser("raw-scale")
    raw.add_argument("--root", type=Path, required=True)
    raw.add_argument("--output", type=Path, required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--manifest-output", type=Path, required=True)
    freeze.add_argument("--audit-output", type=Path, required=True)
    freeze.add_argument("--graph-output", type=Path, required=True)
    freeze.add_argument("--previous-output", type=Path, required=True)
    freeze.add_argument("--ladder-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "raw-scale":
        result = raw_scale_analysis(root)
        _write_new(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return
    values = gradient_audit_record(root)
    manifest = values["manifest"]
    audit = values["audit"]
    graph = objective_graph_integrity(root, audit)
    previous = previous_ladder_interpretation(audit)
    ladder = derive_common_ladder(audit)
    for path, value in (
        (args.manifest_output, manifest),
        (args.audit_output, audit),
        (args.graph_output, graph),
        (args.previous_output, previous),
        (args.ladder_output, ladder),
    ):
        _write_new(path, value)
    print(json.dumps(ladder, sort_keys=True))


if __name__ == "__main__":
    main()
