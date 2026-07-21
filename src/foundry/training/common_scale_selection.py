"""Freeze common LoRA-scale selection and independent validation decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256

SCALE_ORDER = (1.0, 0.75, 0.5, 0.25)


def _load(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    result = cast(dict[str, Any], value)
    expected = result.get("summary_sha256")
    payload = {key: item for key, item in result.items() if key != "summary_sha256"}
    if not isinstance(expected, str) or expected != canonical_sha256(payload):
        raise ValueError(f"{path} summary hash differs")
    return result


def _load_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    result = cast(dict[str, Any], value)
    expected = result.get("config_sha256")
    payload = {key: item for key, item in result.items() if key != "config_sha256"}
    if not isinstance(expected, str) or expected != canonical_sha256(payload):
        raise ValueError("scale configuration hash differs")
    return result


def _validate_matrix(scale: float, assessments: list[dict[str, Any]]) -> None:
    if len(assessments) != 4:
        raise ValueError("each scale requires four retention assessments")
    keys = {(str(item.get("adapter_sha256")), str(item.get("suite_id"))) for item in assessments}
    if len(keys) != 4 or len({key[0] for key in keys}) != 2 or len({key[1] for key in keys}) != 2:
        raise ValueError("scale assessment matrix is incomplete")
    if scale == 1.0:
        if any(item.get("adapter_scale") is not None for item in assessments):
            raise ValueError("historical scale 1 evidence must remain unmodified")
    elif any(
        item.get("adapter_scale") != scale or item.get("state_restoration_verified") is not True
        for item in assessments
    ):
        raise ValueError("scaled assessment factor or restoration evidence differs")


def build_selection(
    *,
    matrices: list[tuple[float, list[dict[str, Any]]]],
    scale_config_sha256: str,
) -> dict[str, Any]:
    """Select the first passing common factor in the predeclared descending order."""

    scales = [scale for scale, _ in matrices]
    if scales != list(SCALE_ORDER[: len(scales)]):
        raise ValueError("scales were not evaluated in the frozen descending order")
    selected: float | None = None
    records = []
    for scale, assessments in matrices:
        _validate_matrix(scale, assessments)
        passed = all(item.get("gate_passed") is True for item in assessments)
        records.append(
            {
                "scale": scale,
                "matrix_passed": passed,
                "assessment_summary_sha256s": [str(item["summary_sha256"]) for item in assessments],
                "cell_gate_passed": [bool(item["gate_passed"]) for item in assessments],
            }
        )
        if passed:
            selected = scale
            break
    if selected is not None and len(records) != len(matrices):
        raise ValueError("lower scales were supplied after a scale passed")
    if selected is None and len(matrices) != len(SCALE_ORDER):
        raise ValueError("a failed partial search cannot close scale selection")
    result: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": "foundry-common-lora-scale-selection-v1",
        "scale_config_sha256": scale_config_sha256,
        "scales": records,
        "selected_common_scale": selected,
        "scale_zero_selectable": False,
        "lower_scales_skipped_after_first_pass": selected is not None,
        "selection_passed": selected is not None,
        "selection_used_gsm1k": False,
        "new_final_holdout_used_for_selection": False,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def build_final_validation(
    *, selection: dict[str, Any], assessments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Freeze independent two-arm validation of the selected common scale."""

    selected = selection.get("selected_common_scale")
    if not isinstance(selected, int | float) or isinstance(selected, bool):
        raise ValueError("final validation requires one selected common scale")
    if len(assessments) != 2 or len({item.get("adapter_sha256") for item in assessments}) != 2:
        raise ValueError("final validation requires exactly two distinct adapters")
    if len({item.get("suite_id") for item in assessments}) != 1:
        raise ValueError("final validation assessments must use one independent suite")
    if any(
        item.get("adapter_scale") != float(selected)
        or item.get("state_restoration_verified") is not True
        for item in assessments
    ):
        raise ValueError("final validation scale or restoration evidence differs")
    passed = all(item.get("gate_passed") is True for item in assessments)
    result: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": "foundry-common-lora-scale-final-validation-v1",
        "selection_summary_sha256": selection["summary_sha256"],
        "selected_common_scale": float(selected),
        "assessment_summary_sha256s": [str(item["summary_sha256"]) for item in assessments],
        "adapter_sha256s": [str(item["adapter_sha256"]) for item in assessments],
        "suite_id": assessments[0]["suite_id"],
        "suite_sha256": assessments[0]["suite_sha256"],
        "retention_decision": (
            "retention_approved_common_scaled_short_run_adapters"
            if passed
            else "failed_independent_common_scale_validation"
        ),
        "retention_passed": passed,
        "gsm1k_authorized": passed,
        "sft_adaptation_line_stopped": not passed,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--config", required=True, type=Path)
    select.add_argument("--scale-matrix", required=True, action="append")
    select.add_argument("--output", required=True, type=Path)
    final = subparsers.add_parser("final")
    final.add_argument("--selection", required=True, type=Path)
    final.add_argument("--assessment", required=True, action="append", type=Path)
    final.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        config = _load_config(args.config)
        matrices = []
        for value in args.scale_matrix:
            scale_text, paths_text = value.split("=", 1)
            paths = [Path(path) for path in paths_text.split(",")]
            matrices.append((float(scale_text), [_load(path) for path in paths]))
        result = build_selection(
            matrices=matrices,
            scale_config_sha256=str(config["config_sha256"]),
        )
    else:
        result = build_final_validation(
            selection=_load(args.selection),
            assessments=[_load(path) for path in args.assessment],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
