"""Freeze the Milestone 14A-R1 Case-1 compatibility correction source."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

PRIMARY_ROOT = Path(r"C:\Users\Admin\Projects\Foundry")
CORRECTED_FILES = (
    "src/foundry/phase2/l3_grpo_runtime.py",
    "src/foundry/phase2/l3_grpo_zero_gradient.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_analysis.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_campaign.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_compatibility.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_correction.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_diagnostic.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_prepare.py",
    "tests/unit/phase2/test_l3_grpo_runtime.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_campaign.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_compatibility.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_correction.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_diagnostic.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_prepare.py",
)
FOCUSED_TEST_FILES = tuple(
    relative for relative in CORRECTED_FILES if relative.startswith("tests/")
)


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify(value: Mapping[str, Any], key: str) -> None:
    supplied = value.get(key)
    projected = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(projected):
        raise ValueError(f"{key} does not reconstruct")


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"correction freeze already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _file_rows(root: Path, files: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in files:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"corrected implementation path is invalid: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def freeze_correction(root: Path) -> dict[str, object]:
    """Write the corrected implementation and semantic contract exactly once."""

    root = root.resolve()
    if root != PRIMARY_ROOT.resolve():
        raise ValueError("Milestone 14A-R1 is attached to the wrong repository")
    tracked = root / "results/phase2_vetted_corpus"
    old = _read(tracked / "milestone14a_implementation.json")
    experiment = _read(tracked / "milestone14a_experiment_contract.json")
    freeze = _read(tracked / "milestone14a_r1_zero_gradient_freeze.json")
    decision = _read(tracked / "milestone14a_r1_zero_gradient_decision.json")
    _verify(old, "implementation_sha256")
    _verify(experiment, "experiment_contract_sha256")
    _verify(freeze, "freeze_sha256")
    _verify(decision, "diagnostic_decision_sha256")
    if (
        old["implementation_sha256"]
        != "63631afa3abb5d0faf2a73decf9a96285d9cbfff789a47806b009c3d8b672eae"
        or experiment["implementation_sha256"] != old["implementation_sha256"]
        or freeze["original_implementation_sha256"] != old["implementation_sha256"]
        or decision["classification"] != "expected_zero_advantage_noop"
        or decision["original_exception_classification"] != "overstrict_per_group_update_gate"
        or decision["scientific_grpo_contract_changed"] is not False
    ):
        raise ValueError("Case-1 correction authorization evidence differs")
    frozen_classification = cast(dict[str, Any], freeze["classification_contract"])
    frozen_fixture = cast(dict[str, Any], freeze["fixture_contract"])
    if (
        file_sha256(root / "src/foundry/phase2/l3_grpo_zero_gradient.py")
        != (frozen_classification["source_sha256"])
    ):
        raise ValueError("frozen classification source changed after diagnostic generation")

    rows = _file_rows(root, CORRECTED_FILES)
    tests = [row for row in rows if cast(str, row["path"]) in FOCUSED_TEST_FILES]
    focused_test_sha256 = canonical_sha256(tests)
    corrected: dict[str, object] = {
        "schema_version": 1,
        "implementation_id": "foundry-l3-verifier-grpo-implementation-r1-v1",
        "old_implementation_sha256": old["implementation_sha256"],
        "files": rows,
        "source_file_count": len(rows) - len(tests),
        "test_file_count": len(tests),
        "focused_test_files": list(FOCUSED_TEST_FILES),
        "focused_test_sha256": focused_test_sha256,
        "scientific_settings_changed": False,
        "counted_training_gradient_gate_changed": False,
        "compatibility_group_classification_added": True,
        "complete_smoke_update_gate_preserved": True,
        "official_smoke_retry_allowed": False,
    }
    corrected["corrected_implementation_sha256"] = canonical_sha256(corrected)

    old_runtime = next(
        row
        for row in cast(list[dict[str, Any]], old["files"])
        if row["path"] == "src/foundry/phase2/l3_grpo_runtime.py"
    )
    corrected_runtime = next(
        row for row in rows if row["path"] == "src/foundry/phase2/l3_grpo_runtime.py"
    )
    correction: dict[str, object] = {
        "schema_version": 1,
        "correction_id": "foundry-l3-zero-advantage-compatibility-correction-v1",
        "correction_case": "expected_zero_advantage_noop",
        "original_exception_classification": "overstrict_per_group_update_gate",
        "starting_commit": freeze["starting_commit"],
        "old_implementation_sha256": old["implementation_sha256"],
        "old_runtime_source_sha256": old_runtime["sha256"],
        "corrected_implementation_sha256": corrected["corrected_implementation_sha256"],
        "corrected_runtime_source_sha256": corrected_runtime["sha256"],
        "experiment_contract_sha256": experiment["experiment_contract_sha256"],
        "classification_contract_sha256": frozen_classification["classification_contract_sha256"],
        "classification_source_sha256": frozen_classification["source_sha256"],
        "fixture_sha256": frozen_fixture["fixture_sha256"],
        "focused_test_sha256": focused_test_sha256,
        "diagnostic_decision_sha256": decision["diagnostic_decision_sha256"],
        "diagnostic_partial_sha256s": decision["diagnostic_partial_sha256s"],
        "original_blocker_sha256": decision["original_blocker_sha256"],
        "frozen_scientific_contracts": freeze["frozen_scientific_contracts"],
        "process_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "20260720",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "individual_expected_noop_may_have_zero_gradient": True,
        "individual_expected_noop_may_have_zero_parameter_delta": True,
        "noop_group_skipped_or_resampled": False,
        "optimizer_scheduler_semantics_changed": False,
        "complete_smoke_requires": {
            "groups": 2,
            "completions": 8,
            "nonzero_variance_groups_minimum": 1,
            "nonzero_policy_gradient_groups_minimum": 1,
            "nonzero_policy_updates_minimum": 1,
            "optimizer_steps": 2,
            "scheduler_steps": 2,
            "reference_updates": 0,
            "base_updates": 0,
        },
        "counted_training_gradient_gate_changed": False,
        "reward_changed": False,
        "schedule_changed": False,
        "model_or_adapter_changed": False,
        "scientific_settings_changed": False,
        "official_smoke_runs_authorized": 2,
        "official_smoke_retries_authorized": 0,
        "source_commit_required_before_official_smoke": True,
        "counted_training_authorized": False,
        "retention_authorized": False,
        "holdout_v2_authorized": False,
        "gsm1k_authorized": False,
        "sealed_content_use": 0,
    }
    correction["correction_contract_sha256"] = canonical_sha256(correction)
    _write_new(
        tracked / "milestone14a_r1_corrected_implementation.json",
        corrected,
    )
    _write_new(
        tracked / "milestone14a_r1_correction_contract.json",
        correction,
    )
    return {
        "corrected_implementation_sha256": corrected["corrected_implementation_sha256"],
        "correction_contract_sha256": correction["correction_contract_sha256"],
        "classification_contract_sha256": correction["classification_contract_sha256"],
        "focused_test_sha256": focused_test_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(freeze_correction(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
