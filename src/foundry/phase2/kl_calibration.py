"""Build and adjudicate Milestone 13C-R3 KL calibration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_recipe
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

OBJECTIVE_CONTRACT_SHA256 = "159ef322b254d5f70b46296fe051417ed53c1543f3432bcb0b9069d66ab975a8"
ADAPTER_CONFIG_SHA256 = "2cf0fb6637747b0aa31525f08ba8b412cc4f1986689ef8b9f555cd4b299039e2"
DEVELOPMENT_SUITES = {
    "adjudication": "5caf23be79fa01151af6f7db8d45c2b85bfe24b03a29589e482d51731c8358af",
    "anchor": "bff18b434a284d848387262dde201601278e5c8b573937b3486bed2bf925696e",
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    historical = subparsers.add_parser("historical")
    historical.add_argument("--root", type=Path, required=True)
    historical.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command != "historical":
        raise ValueError("unsupported command")
    if args.output.exists():
        raise FileExistsError("historical comparator output already exists")
    result = historical_record(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
