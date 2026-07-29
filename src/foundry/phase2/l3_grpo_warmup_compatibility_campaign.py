"""Run and publish four exact warmup-aware official compatibility smokes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_campaign import (
    _environment,
    _require_clean_synchronized_main,
    _run,
)
from foundry.phase2.l3_grpo_contract import (
    COMBINED_CHILD_ENVIRONMENT_SHA256,
    FIXED_LIBRARY_NOTICE_CLASSES,
    INTERPRETER_SHA256,
    PACKAGE_INVENTORY_SHA256,
    adapter_path,
)
from foundry.phase2.l3_grpo_signal_audit import ARMS
from foundry.phase2.l3_grpo_signal_qualification import (
    CONTRACT_OUTPUT as QUALIFICATION_OUTPUT,
)
from foundry.phase2.l3_grpo_signal_qualification import (
    SELECTION_OUTPUT,
)
from foundry.phase2.l3_grpo_source_binding import (
    CONTRACT_OUTPUT as SOURCE_BINDING_OUTPUT,
)
from foundry.phase2.l3_grpo_source_binding import (
    LAYER1_OUTPUT,
    LAYER2_OUTPUT,
    argv_projection_sha256,
)
from foundry.phase2.l3_grpo_warmup_prepare import (
    CONTRACT_OUTPUT as WARMUP_CONTRACT_OUTPUT,
)
from foundry.phase2.l3_grpo_warmup_prepare import (
    ORDER_OUTPUT,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

OUTPUT_NAME = "milestone14b_r3_source_bound_compatibility.json"


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


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite compatibility result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _runtime_command(
    root: Path,
    *,
    arm: str,
    run_index: int,
    run_root: Path,
) -> list[str]:
    training_python = root / ".venv-training/Scripts/python.exe"
    if file_sha256(training_python) != INTERPRETER_SHA256:
        raise ValueError("authorized model interpreter differs")
    tracked = root / "results/phase2_vetted_corpus"
    layer1 = _read(tracked / LAYER1_OUTPUT)
    layer2 = _read(tracked / LAYER2_OUTPUT)
    binding = _read(tracked / SOURCE_BINDING_OUTPUT)
    for value, key in (
        (layer1, "layer1_manifest_sha256"),
        (layer2, "layer2_manifest_sha256"),
        (binding, "source_binding_contract_sha256"),
    ):
        _verify(value, key)
    command = [
        str(training_python),
        "-m",
        "foundry.phase2.l3_grpo_warmup_compatibility_runtime",
        "--root",
        str(root),
        "--arm",
        arm,
        "--run-index",
        str(run_index),
        "--packet",
        str(
            root / f"results/raw/phase2_vetted_corpus/milestone14a/schedules/"
            f"{arm}_prompt_packet.json"
        ),
        "--manifest",
        str(tracked / f"milestone14a_{arm}_schedule.json"),
        "--experiment-contract",
        str(tracked / "milestone14a_experiment_contract.json"),
        "--qualification-contract",
        str(tracked / QUALIFICATION_OUTPUT),
        "--selection",
        str(tracked / SELECTION_OUTPUT),
        "--warmup-update-contract",
        str(tracked / WARMUP_CONTRACT_OUTPUT),
        "--compatibility-order",
        str(tracked / ORDER_OUTPUT),
        "--layer1-manifest",
        str(tracked / LAYER1_OUTPUT),
        "--layer1-sha256",
        str(layer1["layer1_manifest_sha256"]),
        "--layer2-manifest",
        str(tracked / LAYER2_OUTPUT),
        "--layer2-sha256",
        str(layer2["layer2_manifest_sha256"]),
        "--source-binding-contract",
        str(tracked / SOURCE_BINDING_OUTPUT),
        "--source-binding-sha256",
        str(binding["source_binding_contract_sha256"]),
        "--expected-source-commit",
        str(layer2["source_commit"]),
        "--expected-source-tree",
        str(layer2["source_tree"]),
        "--expected-package-sha256",
        PACKAGE_INVENTORY_SHA256,
        "--expected-environment-sha256",
        COMBINED_CHILD_ENVIRONMENT_SHA256,
        "--expected-qualification-decision-sha256",
        str(binding["qualification_decision_sha256"]),
        "--expected-argv-sha256",
        "PENDING",
        "--starting-adapter",
        str(adapter_path(root, arm)),
        "--output-dir",
        str(run_root / "artifacts"),
        "--raw-evidence",
        str(run_root / "raw_evidence.json"),
        "--summary",
        str(run_root / "summary.json"),
        "--envelope",
        str(run_root / "qualification_envelope.json"),
    ]
    expected_index = command.index("--expected-argv-sha256") + 1
    command[expected_index] = argv_projection_sha256(command)
    return command


def _notice_evidence(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    classes = [
        {
            **notice,
            "count": sum(notice["required_substring"] in line for line in lines),
        }
        for notice in FIXED_LIBRARY_NOTICE_CLASSES
    ]
    unrecognized = [
        line
        for line in lines
        if not any(notice["required_substring"] in line for notice in FIXED_LIBRARY_NOTICE_CLASSES)
    ]
    payload: dict[str, object] = {
        "stderr_file_sha256": file_sha256(path),
        "stderr_bytes": path.stat().st_size,
        "stderr_line_count": len(lines),
        "predeclared_classes": classes,
        "unrecognized_line_count": len(unrecognized),
        "only_authorized_notices": not unrecognized,
        "broad_warning_suppression": False,
        "generation_warning_recorded_separately": True,
    }
    payload["notice_evidence_sha256"] = canonical_sha256(payload)
    return payload


def _validated_run(
    *,
    arm: str,
    run_index: int,
    run_root: Path,
    log_path: Path,
    selection_sha256: str,
    warmup_sha256: str,
    selected_group_ids: Sequence[str],
    source_binding_sha256: str,
) -> dict[str, Any]:
    summary_path = run_root / "summary.json"
    raw_path = run_root / "raw_evidence.json"
    partial_path = run_root / "partial_evidence.json"
    envelope_path = run_root / "qualification_envelope.json"
    summary = _read(summary_path)
    raw = _read(raw_path)
    partial = _read(partial_path)
    envelope = _read(envelope_path)
    for value, key in (
        (summary, "summary_sha256"),
        (raw, "raw_evidence_sha256"),
        (partial, "partial_evidence_sha256"),
        (envelope, "envelope_sha256"),
    ):
        _verify(value, key)
    packet = cast(dict[str, Any], summary.get("exact_packet"))
    _verify(packet, "packet_sha256")
    gate = cast(Mapping[str, Any], summary.get("complete_smoke_gate"))
    warning = cast(Mapping[str, Any], summary.get("warning_evidence"))
    steps = cast(list[Mapping[str, Any]], summary.get("classification_steps"))
    notices = _notice_evidence(log_path)
    adapter_hash = directory_sha256(run_root / "artifacts/final_adapter")
    if (
        summary.get("arm") != arm
        or summary.get("gate_passed") is not True
        or summary.get("optimizer_steps") != 2
        or summary.get("groups") != 2
        or summary.get("completions") != 8
        or summary.get("policy_updated") is not True
        or summary.get("reference_unchanged") is not True
        or summary.get("base_unchanged") is not True
        or summary.get("cpu_offload") is not False
        or summary.get("offline_reload_passed") is not True
        or summary.get("adapter_disabled_base_restoration_passed") is not True
        or summary.get("peak_reserved_below_physical") is not True
        or summary.get("warmup_update_contract_sha256") != warmup_sha256
        or summary.get("source_binding_contract_sha256") != source_binding_sha256
        or gate.get("passed") is not True
        or gate.get("policy_update_count") != 1
        or gate.get("optimizer_call_count") != 2
        or gate.get("scheduler_advance_count") != 2
        or gate.get("reference_update_count") != 0
        or gate.get("base_update_count") != 0
        or len(steps) != 2
        or steps[1].get("classification") != "nonzero_policy_update"
        or warning.get("all_warnings_whitelisted") is not True
        or warning.get("all_state_unchanged") is not True
        or packet.get("group_ids") != list(selected_group_ids)
        or packet.get("source_kinds") != ["base_replay", "task"]
        or packet.get("packet_sha256") != summary.get("exact_packet_sha256")
        or cast(Mapping[str, Any], packet.get("final_adapter")).get("directory_sha256")
        != adapter_hash
        or summary.get("raw_evidence_file_sha256") != file_sha256(raw_path)
        or summary.get("partial_evidence_file_sha256") != file_sha256(partial_path)
        or envelope.get("arm") != arm
        or envelope.get("run_index") != run_index
        or envelope.get("selection_decision_sha256") != selection_sha256
        or envelope.get("warmup_update_contract_sha256") != warmup_sha256
        or envelope.get("source_binding_contract_sha256") != source_binding_sha256
        or envelope.get("selected_group_ids") != list(selected_group_ids)
        or envelope.get("exact_packet_sha256") != packet.get("packet_sha256")
        or envelope.get("final_adapter_directory_sha256") != adapter_hash
        or notices.get("only_authorized_notices") is not True
    ):
        raise RuntimeError(f"{arm} compatibility run {run_index} gate failed")
    return {
        "summary": summary,
        "raw": raw,
        "partial": partial,
        "envelope": envelope,
        "packet": packet,
        "notices": notices,
        "adapter_hash": adapter_hash,
        "summary_file_sha256": file_sha256(summary_path),
        "raw_file_sha256": file_sha256(raw_path),
        "partial_file_sha256": file_sha256(partial_path),
    }


def build_compatibility_result(
    *,
    root: Path,
    source_commit: str,
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, object]:
    """Require exact duplicate scientific evidence for both selected arms."""

    tracked = root / "results/phase2_vetted_corpus"
    selection = _read(tracked / SELECTION_OUTPUT)
    qualification = _read(tracked / QUALIFICATION_OUTPUT)
    warmup = _read(tracked / WARMUP_CONTRACT_OUTPUT)
    order = _read(tracked / ORDER_OUTPUT)
    layer1 = _read(tracked / LAYER1_OUTPUT)
    layer2 = _read(tracked / LAYER2_OUTPUT)
    source_binding = _read(tracked / SOURCE_BINDING_OUTPUT)
    for value, key in (
        (selection, "selection_decision_sha256"),
        (qualification, "qualification_contract_sha256"),
        (warmup, "warmup_update_contract_sha256"),
        (order, "compatibility_order_sha256"),
        (layer1, "layer1_manifest_sha256"),
        (layer2, "layer2_manifest_sha256"),
        (source_binding, "source_binding_contract_sha256"),
    ):
        _verify(value, key)
    if (
        selection.get("qualification_contract_sha256")
        != qualification.get("qualification_contract_sha256")
        or selection.get("compatibility_authorized") is not True
        or warmup.get("compatibility_order_sha256") != order.get("compatibility_order_sha256")
    ):
        raise ValueError("compatibility publication source binding differs")

    arm_results: dict[str, object] = {}
    all_rows: list[Mapping[str, Any]] = []
    for arm in ARMS:
        rows = runs.get(arm)
        if rows is None or len(rows) != 2:
            raise ValueError(f"{arm} requires exactly two compatibility runs")
        first, second = rows
        if (
            first["packet"] != second["packet"]
            or first["raw"] != second["raw"]
            or first["partial"] != second["partial"]
            or first["adapter_hash"] != second["adapter_hash"]
        ):
            raise RuntimeError(f"{arm} compatibility duplicate evidence differs")
        packet = cast(Mapping[str, Any], first["packet"])
        summary = cast(Mapping[str, Any], first["summary"])
        gate = cast(Mapping[str, Any], summary["complete_smoke_gate"])
        steps = cast(list[Mapping[str, Any]], summary["classification_steps"])
        final_policy = cast(Mapping[str, Any], packet["final_policy"])
        final_reference = cast(Mapping[str, Any], packet["final_reference"])
        initial_reference = cast(Mapping[str, Any], packet["initial_reference"])
        base_before = cast(Mapping[str, Any], packet["base_before"])
        base_after = cast(Mapping[str, Any], packet["base_after"])
        classifications = [str(step["classification"]) for step in steps]
        arm_results[arm] = {
            "official_smoke_runs": 2,
            "official_smoke_retries": 0,
            "exact_match": True,
            "exact_packet_sha256": packet["packet_sha256"],
            "raw_evidence_sha256": cast(Mapping[str, Any], first["raw"])["raw_evidence_sha256"],
            "partial_evidence_sha256": cast(Mapping[str, Any], first["partial"])[
                "partial_evidence_sha256"
            ],
            "selected_group_ids": packet["group_ids"],
            "effective_learning_rates": [step["effective_learning_rates"] for step in steps],
            "step_classifications": classifications,
            "classification_counts": dict(Counter(classifications)),
            "complete_smoke_gate": dict(gate),
            "final_policy_adapter_sha256": final_policy["normalized_tensor_state_sha256"],
            "final_adapter_directory_sha256": first["adapter_hash"],
            "reference_state_sha256": final_reference["normalized_tensor_state_sha256"],
            "reference_unchanged": (
                final_reference["normalized_tensor_state_sha256"]
                == initial_reference["normalized_tensor_state_sha256"]
            ),
            "base_parameter_state_sha256": base_before["base_parameter_state_sha256"],
            "base_unchanged": (
                base_before["base_parameter_state_sha256"]
                == base_after["base_parameter_state_sha256"]
            ),
            "run_summary_sha256s": [
                cast(Mapping[str, Any], row["summary"])["summary_sha256"] for row in rows
            ],
            "run_summary_file_sha256s": [row["summary_file_sha256"] for row in rows],
            "raw_file_sha256s": [row["raw_file_sha256"] for row in rows],
            "partial_file_sha256s": [row["partial_file_sha256"] for row in rows],
            "fixed_library_notices": [row["notices"] for row in rows],
            "completion_tokens_per_run": [
                cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], row["summary"])["reward"],
                )["completion_tokens"]
                for row in rows
            ],
            "peak_allocated_vram_bytes": [
                cast(Mapping[str, Any], row["summary"])["peak_allocated_vram_bytes"] for row in rows
            ],
            "peak_reserved_vram_bytes": [
                cast(Mapping[str, Any], row["summary"])["peak_reserved_vram_bytes"] for row in rows
            ],
            "peak_process_rss_bytes": [
                cast(Mapping[str, Any], row["summary"])["peak_process_rss_bytes"] for row in rows
            ],
            "output_disk_bytes": [
                cast(Mapping[str, Any], row["summary"])["output_disk_bytes"] for row in rows
            ],
            "runtime_seconds": [
                cast(Mapping[str, Any], row["summary"])["runtime_seconds"] for row in rows
            ],
            "training_seconds": [
                cast(Mapping[str, Any], row["summary"])["training_seconds"] for row in rows
            ],
            "gate_passed": True,
        }
        if (
            cast(Mapping[str, Any], arm_results[arm]).get("reference_unchanged") is not True
            or cast(Mapping[str, Any], arm_results[arm]).get("base_unchanged") is not True
        ):
            raise RuntimeError(f"{arm} compatibility integrity differs")
        all_rows.extend(rows)

    policy_updates = sum(
        cast(
            int,
            cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], row["summary"])["complete_smoke_gate"],
            )["policy_update_count"],
        )
        for row in all_rows
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "compatibility_id": "foundry-l3-grpo-source-bound-warmup-compatibility-v1",
        "decision": "pass",
        "source_commit": source_commit,
        "layer2_source_commit": layer2["source_commit"],
        "layer2_source_tree": layer2["source_tree"],
        "layer1_manifest_sha256": layer1["layer1_manifest_sha256"],
        "layer2_manifest_sha256": layer2["layer2_manifest_sha256"],
        "source_binding_contract_sha256": source_binding["source_binding_contract_sha256"],
        "qualification_contract_sha256": qualification["qualification_contract_sha256"],
        "selection_decision_sha256": selection["selection_decision_sha256"],
        "warmup_update_contract_sha256": warmup["warmup_update_contract_sha256"],
        "scheduler_contract_sha256": warmup["scheduler_contract_sha256"],
        "compatibility_order_sha256": order["compatibility_order_sha256"],
        "expected_effective_learning_rates": warmup["compatibility_effective_learning_rates"],
        "arm_order": list(ARMS),
        "arms": arm_results,
        "official_smoke_runs": 4,
        "official_smoke_retries": 0,
        "fresh_processes": True,
        "optimizer_steps": 8,
        "groups": 8,
        "completions": 32,
        "policy_update_count": policy_updates,
        "reference_update_count": 0,
        "base_update_count": 0,
        "all_scientific_and_tensor_evidence_exact": True,
        "all_final_adapter_directory_hashes_exact": True,
        "only_authorized_warnings": True,
        "gate_passed": True,
        "counted_training_started": False,
        "retention_started": False,
        "holdout_v2_started": False,
        "gsm1k_started": False,
        "sealed_content_use": 0,
        "next_action": (
            "begin counted generic and targeted L3 verifier-GRPO training under "
            "the unchanged frozen schedules"
        ),
    }
    result["compatibility_sha256"] = canonical_sha256(result)
    return result


def run_campaign(root: Path) -> dict[str, object]:
    """Run generic and duplicate, then targeted and duplicate, without retries."""

    root = root.resolve()
    environment = _environment(root)
    source_commit = _require_clean_synchronized_main(root)
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone14b_r3/compatibility"
    tracked = root / "results/phase2_vetted_corpus"
    selection = _read(tracked / SELECTION_OUTPUT)
    warmup = _read(tracked / WARMUP_CONTRACT_OUTPUT)
    source_binding = _read(tracked / SOURCE_BINDING_OUTPUT)
    _verify(selection, "selection_decision_sha256")
    _verify(warmup, "warmup_update_contract_sha256")
    _verify(source_binding, "source_binding_contract_sha256")
    selected = cast(Mapping[str, Mapping[str, Any]], selection["arms"])
    runs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        selected_ids = [
            cast(str, selected[arm]["replay_group_id"]),
            cast(str, selected[arm]["task_group_id"]),
        ]
        for run_index in (1, 2):
            _require_clean_synchronized_main(root)
            run_root = raw_root / arm / f"run-{run_index}"
            log_path = raw_root / "logs" / f"{arm}-run-{run_index}.stderr.txt"
            _run(
                _runtime_command(
                    root,
                    arm=arm,
                    run_index=run_index,
                    run_root=run_root,
                ),
                root=root,
                environment=environment,
                stdout=(raw_root / "logs" / f"{arm}-run-{run_index}.stdout.txt"),
                stderr=log_path,
            )
            runs[arm].append(
                _validated_run(
                    arm=arm,
                    run_index=run_index,
                    run_root=run_root,
                    log_path=log_path,
                    selection_sha256=cast(
                        str,
                        selection["selection_decision_sha256"],
                    ),
                    warmup_sha256=cast(
                        str,
                        warmup["warmup_update_contract_sha256"],
                    ),
                    selected_group_ids=selected_ids,
                    source_binding_sha256=cast(
                        str,
                        source_binding["source_binding_contract_sha256"],
                    ),
                )
            )
    result = build_compatibility_result(
        root=root,
        source_commit=source_commit,
        runs=runs,
    )
    _write_new(tracked / OUTPUT_NAME, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_campaign(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
