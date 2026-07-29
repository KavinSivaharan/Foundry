"""Publish the Milestone 14A-R1 exact two-smoke compatibility decision."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_contract import FIXED_LIBRARY_NOTICE_CLASSES
from foundry.phase2.l3_grpo_zero_gradient import (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    NONZERO_GRADIENT_UPDATE,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256


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
        raise FileExistsError(f"refusing to overwrite compatibility evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.stdout.strip()


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


def _require_clean_synchronized_main(root: Path) -> str:
    source_commit = _git(root, "rev-parse", "HEAD")
    if (
        _git(root, "branch", "--show-current") != "main"
        or source_commit != _git(root, "rev-parse", "origin/main")
        or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
        != ["0", "0"]
        or _git(root, "status", "--porcelain")
    ):
        raise RuntimeError("compatibility publication requires synchronized clean main")
    return source_commit


def build_compatibility_result(root: Path) -> dict[str, object]:
    """Require byte-exact raw evidence and exact packet equality across both smokes."""

    root = root.resolve()
    source_commit = _require_clean_synchronized_main(root)
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a_r1"
    summaries = [_read(raw / f"compatibility/run-{index}/summary.json") for index in (1, 2)]
    partials = [_read(raw / f"compatibility/run-{index}/partial_evidence.json") for index in (1, 2)]
    for index, (summary, partial) in enumerate(
        zip(summaries, partials, strict=True),
        start=1,
    ):
        _verify(summary, "summary_sha256")
        _verify(partial, "partial_evidence_sha256")
        gate = cast(dict[str, Any], summary.get("complete_smoke_gate"))
        if (
            summary.get("gate_passed") is not True
            or summary.get("optimizer_steps") != 2
            or summary.get("groups") != 2
            or summary.get("completions") != 8
            or summary.get("arm") != "generic"
            or gate.get("passed") is not True
            or partial.get("stage") != "complete_smoke_gate_persisted"
            or partial.get("error") is not None
            or summary.get("partial_evidence_file_sha256")
            != file_sha256(raw / f"compatibility/run-{index}/partial_evidence.json")
        ):
            raise RuntimeError(f"official compatibility smoke {index} gate failed")
    packets = [cast(dict[str, Any], summary["exact_packet"]) for summary in summaries]
    if packets[0] != packets[1]:
        raise RuntimeError("official compatibility smoke exact packets differ")
    _verify(packets[0], "packet_sha256")
    if partials[0] != partials[1]:
        raise RuntimeError("official compatibility smoke raw evidence differs")
    classifications = cast(list[dict[str, Any]], packets[0]["classification_steps"])
    if len(classifications) != 2:
        raise RuntimeError("official compatibility classification count differs")
    for row in classifications:
        _verify(row, "classification_evidence_sha256")
    classification_values = [row["classification"] for row in classifications]
    if (
        any(
            value not in {EXPECTED_ZERO_ADVANTAGE_NOOP, NONZERO_GRADIENT_UPDATE}
            for value in classification_values
        )
        or NONZERO_GRADIENT_UPDATE not in classification_values
    ):
        raise RuntimeError("official compatibility classifications fail the complete gate")
    complete_gate = cast(dict[str, Any], packets[0]["complete_smoke_gate"])
    if complete_gate.get("passed") is not True:
        raise RuntimeError("official complete-smoke update gate failed")

    correction = _read(
        root / "results/phase2_vetted_corpus/milestone14a_r1_correction_contract.json"
    )
    implementation = _read(
        root / "results/phase2_vetted_corpus/milestone14a_r1_corrected_implementation.json"
    )
    decision = _read(
        root / "results/phase2_vetted_corpus/milestone14a_r1_zero_gradient_decision.json"
    )
    _verify(correction, "correction_contract_sha256")
    _verify(implementation, "corrected_implementation_sha256")
    _verify(decision, "diagnostic_decision_sha256")
    if (
        packets[0].get("correction_contract_sha256") != correction["correction_contract_sha256"]
        or packets[0].get("corrected_implementation_sha256")
        != implementation["corrected_implementation_sha256"]
        or packets[0].get("classification_contract_sha256")
        != correction["classification_contract_sha256"]
    ):
        raise RuntimeError("official packet source binding differs")

    notices = [
        _notice_evidence(raw / f"compatibility/logs/run-{index}.stderr.txt") for index in (1, 2)
    ]
    final_policy = cast(dict[str, Any], packets[0]["final_policy"])
    final_reference = cast(dict[str, Any], packets[0]["final_reference"])
    base_before = cast(dict[str, Any], packets[0]["base_before"])
    base_after = cast(dict[str, Any], packets[0]["base_after"])
    result: dict[str, object] = {
        "schema_version": 1,
        "compatibility_id": "foundry-l3-verifier-grpo-compatibility-r1-v1",
        "decision": "pass",
        "corrected_source_commit": source_commit,
        "official_smoke_runs": 2,
        "official_smoke_retries": 0,
        "fresh_processes": True,
        "arm": "generic",
        "optimizer_steps_per_run": 2,
        "groups_per_run": 2,
        "completions_per_run": 8,
        "one_task_and_one_replay_group_per_run": True,
        "classification_steps": classifications,
        "complete_smoke_gate": complete_gate,
        "zero_variance_group_count": complete_gate["zero_variance_group_count"],
        "nonzero_variance_group_count": complete_gate["nonzero_variance_group_count"],
        "expected_noop_group_count": complete_gate["expected_noop_group_count"],
        "nonzero_gradient_group_count": complete_gate["nonzero_gradient_group_count"],
        "policy_update_count": complete_gate["policy_update_count"],
        "optimizer_step_count": complete_gate["optimizer_step_count"],
        "scheduler_step_count": complete_gate["scheduler_step_count"],
        "exact_packet": packets[0],
        "exact_packet_sha256": packets[0]["packet_sha256"],
        "raw_partial_evidence_exact_match": True,
        "raw_partial_evidence_sha256": partials[0]["partial_evidence_sha256"],
        "raw_partial_file_sha256": file_sha256(raw / "compatibility/run-1/partial_evidence.json"),
        "run_summary_sha256s": [summary["summary_sha256"] for summary in summaries],
        "run_summary_file_sha256s": [
            file_sha256(raw / f"compatibility/run-{index}/summary.json") for index in (1, 2)
        ],
        "fixed_library_notices": notices,
        "final_policy_adapter_sha256": final_policy["normalized_tensor_state_sha256"],
        "final_adapter_directory_sha256": cast(dict[str, Any], packets[0]["final_adapter"])[
            "directory_sha256"
        ],
        "reference_state_sha256": final_reference["normalized_tensor_state_sha256"],
        "reference_unchanged": (
            final_reference["normalized_tensor_state_sha256"]
            == cast(dict[str, Any], packets[0]["initial_reference"])[
                "normalized_tensor_state_sha256"
            ]
        ),
        "base_parameter_state_sha256": base_before["base_parameter_state_sha256"],
        "base_unchanged": (
            base_before["base_parameter_state_sha256"] == base_after["base_parameter_state_sha256"]
        ),
        "correction_contract_sha256": correction["correction_contract_sha256"],
        "corrected_implementation_sha256": implementation["corrected_implementation_sha256"],
        "classification_contract_sha256": correction["classification_contract_sha256"],
        "diagnostic_decision_sha256": decision["diagnostic_decision_sha256"],
        "completion_tokens_per_run": [
            cast(dict[str, Any], summary["reward"])["completion_tokens"] for summary in summaries
        ],
        "peak_allocated_vram_bytes": [
            summary["peak_allocated_vram_bytes"] for summary in summaries
        ],
        "peak_reserved_vram_bytes": [summary["peak_reserved_vram_bytes"] for summary in summaries],
        "peak_process_rss_bytes": [summary["peak_process_rss_bytes"] for summary in summaries],
        "output_disk_bytes": [summary["output_disk_bytes"] for summary in summaries],
        "runtime_seconds": [summary["runtime_seconds"] for summary in summaries],
        "training_seconds": [summary["training_seconds"] for summary in summaries],
        "exact_match": True,
        "gate_passed": True,
        "counted_training_started": False,
        "retention_started": False,
        "holdout_v2_started": False,
        "gsm1k_started": False,
        "sealed_content_use": 0,
        "next_action": (
            "resume Milestone 14A at counted generic and targeted verifier-GRPO training"
        ),
    }
    if result["reference_unchanged"] is not True or result["base_unchanged"] is not True:
        raise RuntimeError("official compatibility reference or base integrity failed")
    result["compatibility_sha256"] = canonical_sha256(result)
    return result


def write_compatibility_result(root: Path) -> dict[str, object]:
    result = build_compatibility_result(root)
    _write_new(
        root / "results/phase2_vetted_corpus/milestone14a_r1_compatibility.json",
        result,
    )
    return result
