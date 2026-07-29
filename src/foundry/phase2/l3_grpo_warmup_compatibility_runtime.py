"""Run one warmup-aware selected-group smoke through the official GRPO runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from foundry.phase2 import l3_grpo_runtime as official
from foundry.phase2.l3_grpo_runtime import RuntimeSchedule
from foundry.phase2.l3_grpo_signal_audit import ARMS
from foundry.phase2.l3_grpo_signal_qualification import verify_qualification_contract
from foundry.phase2.l3_grpo_warmup_prepare import verify_warmup_update_contract
from foundry.phase2.l3_grpo_warmup_update import (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    EXPECTED_ZERO_LR_WARMUP_NOOP,
    NONZERO_POLICY_UPDATE,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

Arm = Literal["generic", "targeted"]
RUNTIME_ID = "foundry-l3-grpo-warmup-compatibility-wrapper-v1"


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
        raise FileExistsError(f"refusing to overwrite compatibility envelope: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.strip()


def _restricted_schedule(
    full: RuntimeSchedule,
    selected: Mapping[str, Any],
) -> RuntimeSchedule:
    task = next(
        (group for group in full.groups if group.group_id == selected.get("task_group_id")),
        None,
    )
    replay = next(
        (group for group in full.groups if group.group_id == selected.get("replay_group_id")),
        None,
    )
    if (
        task is None
        or replay is None
        or task.source_kind != "task"
        or replay.source_kind != "base_replay"
        or task.position != selected.get("task_schedule_position")
        or replay.position != selected.get("replay_schedule_position")
        or task.prompt_sha256 != selected.get("task_prompt_sha256")
        or replay.prompt_sha256 != selected.get("replay_prompt_sha256")
    ):
        raise ValueError("selected compatibility groups do not reconstruct")
    return RuntimeSchedule(
        arm=full.arm,
        groups=(replay, task),
        packet_sha256=full.packet_sha256,
        manifest_sha256=full.manifest_sha256,
    )


def run(
    *,
    root: Path,
    arm: Arm,
    run_index: int,
    packet_path: Path,
    manifest_path: Path,
    experiment_contract_path: Path,
    qualification_contract_path: Path,
    selection_path: Path,
    warmup_update_contract_path: Path,
    compatibility_order_path: Path,
    starting_adapter: Path,
    output_dir: Path,
    raw_evidence_path: Path,
    summary_path: Path,
    envelope_path: Path,
) -> dict[str, object]:
    """Restrict group order, then delegate the complete official smoke."""

    root = root.resolve()
    if root != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("warmup compatibility is attached to the wrong repository")
    if arm not in ARMS or run_index not in (1, 2):
        raise ValueError("warmup compatibility arm or run index differs")
    if envelope_path.exists():
        raise FileExistsError("compatibility envelope must start unused")

    qualification = _read(qualification_contract_path)
    verify_qualification_contract(root, qualification, require_clean_synchronized=True)
    selection = _read(selection_path)
    _verify(selection, "selection_decision_sha256")
    warmup = _read(warmup_update_contract_path)
    verify_warmup_update_contract(root, warmup, require_clean_synchronized=True)
    order = _read(compatibility_order_path)
    _verify(order, "compatibility_order_sha256")
    if (
        selection.get("qualification_contract_sha256")
        != qualification.get("qualification_contract_sha256")
        or selection.get("both_arms_pass") is not True
        or selection.get("selection_frozen") is not True
        or selection.get("compatibility_authorized") is not True
        or selection.get("counted_training_authorized") is not False
        or order.get("compatibility_order_sha256") != warmup.get("compatibility_order_sha256")
    ):
        raise ValueError("warmup compatibility source decision differs")
    selected = cast(Mapping[str, Any], cast(Mapping[str, Any], selection["arms"])[arm])
    selected_order = cast(Mapping[str, Any], cast(Mapping[str, Any], order["arms"])[arm])
    first_order = cast(Mapping[str, Any], selected_order["optimizer_call_1"])
    second_order = cast(Mapping[str, Any], selected_order["optimizer_call_2"])
    if (
        selected.get("fresh_process_replay_status") != "passed"
        or selected.get("gradient_projection_status") != "passed"
        or selected.get("exact_duplicate_evidence") is not True
        or first_order.get("group_id") != selected.get("replay_group_id")
        or first_order.get("source_kind") != "base_replay"
        or second_order.get("group_id") != selected.get("task_group_id")
        or second_order.get("source_kind") != "task"
    ):
        raise ValueError("selected arm or compatibility order is not qualified")

    original_loader = official.load_schedule
    full = original_loader(packet_path, manifest_path, arm)
    restricted = _restricted_schedule(full, selected)
    expected_packet_path = packet_path.resolve()
    expected_manifest_path = manifest_path.resolve()
    expected_arm = arm

    def load_restricted(
        packet_path: Path,
        manifest_path: Path,
        arm: Arm,
    ) -> RuntimeSchedule:
        if (
            packet_path.resolve() != expected_packet_path
            or manifest_path.resolve() != expected_manifest_path
            or arm != expected_arm
        ):
            raise ValueError("official compatibility loader arguments differ")
        return restricted

    official.load_schedule = load_restricted
    try:
        summary = official.run(
            root=root,
            arm=arm,
            mode="compatibility",
            packet_path=packet_path,
            manifest_path=manifest_path,
            experiment_contract_path=experiment_contract_path,
            warmup_update_contract_path=warmup_update_contract_path,
            starting_adapter=starting_adapter,
            output_dir=output_dir,
            raw_evidence_path=raw_evidence_path,
            partial_evidence_path=raw_evidence_path.with_name("partial_evidence.json"),
            summary_path=summary_path,
        )
    finally:
        official.load_schedule = original_loader

    _verify(summary, "summary_sha256")
    packet = cast(dict[str, Any], summary.get("exact_packet"))
    _verify(packet, "packet_sha256")
    gate = cast(Mapping[str, Any], summary.get("complete_smoke_gate"))
    steps = cast(list[Mapping[str, Any]], summary.get("classification_steps"))
    final_adapter = cast(Mapping[str, Any], packet.get("final_adapter"))
    expected_group_ids = [selected["replay_group_id"], selected["task_group_id"]]
    expected_lrs = cast(list[float], warmup["compatibility_effective_learning_rates"])
    if (
        summary.get("gate_passed") is not True
        or summary.get("arm") != arm
        or summary.get("groups") != 2
        or summary.get("completions") != 8
        or summary.get("optimizer_steps") != 2
        or summary.get("policy_updated") is not True
        or summary.get("reference_unchanged") is not True
        or summary.get("base_unchanged") is not True
        or summary.get("cpu_offload") is not False
        or summary.get("offline_reload_passed") is not True
        or summary.get("adapter_disabled_base_restoration_passed") is not True
        or summary.get("peak_reserved_below_physical") is not True
        or summary.get("warmup_update_contract_sha256")
        != warmup.get("warmup_update_contract_sha256")
        or summary.get("expected_effective_learning_rates") != expected_lrs
        or gate.get("passed") is not True
        or gate.get("optimizer_call_count") != 2
        or gate.get("scheduler_advance_count") != 2
        or gate.get("policy_update_count") != 1
        or gate.get("reference_update_count") != 0
        or gate.get("base_update_count") != 0
        or len(steps) != 2
        or steps[0].get("classification")
        not in {EXPECTED_ZERO_ADVANTAGE_NOOP, EXPECTED_ZERO_LR_WARMUP_NOOP}
        or steps[1].get("classification") != NONZERO_POLICY_UPDATE
        or steps[0].get("effective_learning_rates") != [expected_lrs[0]]
        or steps[1].get("effective_learning_rates") != [expected_lrs[1]]
        or packet.get("group_ids") != expected_group_ids
        or packet.get("source_kinds") != ["base_replay", "task"]
        or packet.get("packet_sha256") != summary.get("exact_packet_sha256")
        or not isinstance(final_adapter.get("directory_sha256"), str)
    ):
        raise RuntimeError("warmup-aware official compatibility smoke gate failed")

    envelope: dict[str, object] = {
        "schema_version": 1,
        "wrapper_id": RUNTIME_ID,
        "arm": arm,
        "run_index": run_index,
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "qualification_contract_sha256": qualification["qualification_contract_sha256"],
        "selection_decision_sha256": selection["selection_decision_sha256"],
        "warmup_update_contract_sha256": warmup["warmup_update_contract_sha256"],
        "compatibility_order_sha256": order["compatibility_order_sha256"],
        "official_runtime_id": summary["runtime_id"],
        "official_summary_sha256": summary["summary_sha256"],
        "official_summary_file_sha256": file_sha256(summary_path),
        "exact_packet_sha256": packet["packet_sha256"],
        "selected_group_ids": expected_group_ids,
        "expected_effective_learning_rates": expected_lrs,
        "step_classifications": [step["classification"] for step in steps],
        "final_adapter_directory_sha256": final_adapter["directory_sha256"],
        "optimizer_steps": 2,
        "groups": 2,
        "completions": 8,
        "gate_passed": True,
    }
    envelope["envelope_sha256"] = canonical_sha256(envelope)
    _write_new(envelope_path, envelope)
    return envelope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-contract", type=Path, required=True)
    parser.add_argument("--qualification-contract", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--warmup-update-contract", type=Path, required=True)
    parser.add_argument("--compatibility-order", type=Path, required=True)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        root=args.root,
        arm=cast(Arm, args.arm),
        run_index=args.run_index,
        packet_path=args.packet,
        manifest_path=args.manifest,
        experiment_contract_path=args.experiment_contract,
        qualification_contract_path=args.qualification_contract,
        selection_path=args.selection,
        warmup_update_contract_path=args.warmup_update_contract,
        compatibility_order_path=args.compatibility_order,
        starting_adapter=args.starting_adapter,
        output_dir=args.output_dir,
        raw_evidence_path=args.raw_evidence,
        summary_path=args.summary,
        envelope_path=args.envelope,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
