"""Run the four independent L3 GRPO replay/gradient qualifications sequentially."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_campaign import (
    _environment,
    _require_clean_synchronized_main,
    _run,
)
from foundry.phase2.l3_grpo_contract import INTERPRETER_SHA256, adapter_path
from foundry.phase2.l3_grpo_signal_audit import ARMS, build_signal_summary
from foundry.phase2.l3_grpo_signal_qualification import (
    CONTRACT_OUTPUT,
    build_selection_and_gradient_decision,
    write_selection_and_signal_decision,
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


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite qualification campaign: {path}")
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
    return [
        str(training_python),
        "-m",
        "foundry.phase2.l3_grpo_signal_qualification_runtime",
        "--root",
        str(root),
        "--arm",
        arm,
        "--run-index",
        str(run_index),
        "--packet",
        str(
            root
            / f"results/raw/phase2_vetted_corpus/milestone14a/schedules/{arm}_prompt_packet.json"
        ),
        "--manifest",
        str(tracked / f"milestone14a_{arm}_schedule.json"),
        "--audit-contract",
        str(tracked / "milestone14b_r1_signal_audit_contract.json"),
        "--advantage-contract",
        str(tracked / "milestone14b_r1_advantage_equivalence_contract.json"),
        "--prior-diagnostic-manifest",
        str(tracked / "milestone14b_r1_prior_diagnostic_manifest.json"),
        "--qualification-contract",
        str(tracked / CONTRACT_OUTPUT),
        "--starting-adapter",
        str(adapter_path(root, arm)),
        "--raw-evidence",
        str(run_root / "raw_evidence.json"),
        "--summary",
        str(run_root / "summary.json"),
    ]


def _validate_pair(
    *,
    arm: str,
    run_roots: Sequence[Path],
) -> tuple[list[dict[str, Any]], str]:
    if len(run_roots) != 2:
        raise ValueError("qualification requires exactly two run roots")
    summaries: list[dict[str, Any]] = []
    exact_values: list[dict[str, Any]] = []
    for index, run_root in enumerate(run_roots, start=1):
        summary = _read(run_root / "summary.json")
        raw = _read(run_root / "raw_evidence.json")
        _verify(summary, "summary_sha256")
        _verify(raw, "raw_projection_sha256")
        exact = cast(dict[str, Any], raw.get("exact_scientific_tensor_evidence"))
        _verify(exact, "exact_projection_sha256")
        if (
            summary.get("arm") != arm
            or summary.get("run_index") != index
            or summary.get("gate_passed") is not True
            or summary.get("fresh_process_replay_exact") is not True
            or summary.get("optimizer_created") is not False
            or summary.get("optimizer_steps") != 0
            or summary.get("reference_gradient_count") != 0
            or summary.get("base_gradient_count") != 0
            or summary.get("exact_projection_sha256") != exact.get("exact_projection_sha256")
            or summary.get("raw_projection_file_sha256")
            != file_sha256(run_root / "raw_evidence.json")
        ):
            raise RuntimeError(f"{arm} qualification run {index} gate failed")
        summaries.append(summary)
        exact_values.append(exact)
    if exact_values[0] != exact_values[1]:
        raise RuntimeError(f"{arm} independent projection evidence differs")
    return summaries, cast(str, exact_values[0]["exact_projection_sha256"])


def run_campaign(root: Path) -> dict[str, object]:
    """Run generic twice then targeted twice, without retries or optimizers."""

    root = root.resolve()
    environment = _environment(root)
    source_commit = _require_clean_synchronized_main(root)
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone14b_r1/qualification"
    run_roots: dict[str, list[Path]] = {arm: [] for arm in ARMS}
    process_order: list[str] = []
    for arm in ARMS:
        for run_index in (1, 2):
            _require_clean_synchronized_main(root)
            run_root = raw_root / arm / f"run-{run_index}"
            _run(
                _runtime_command(
                    root,
                    arm=arm,
                    run_index=run_index,
                    run_root=run_root,
                ),
                root=root,
                environment=environment,
                stdout=raw_root / "logs" / f"{arm}-run-{run_index}.stdout.txt",
                stderr=raw_root / "logs" / f"{arm}-run-{run_index}.stderr.txt",
            )
            run_roots[arm].append(run_root)
            process_order.append(f"{arm}-run-{run_index}")

    summaries: dict[str, list[dict[str, Any]]] = {}
    exact_hashes: dict[str, str] = {}
    for arm in ARMS:
        summaries[arm], exact_hashes[arm] = _validate_pair(
            arm=arm,
            run_roots=run_roots[arm],
        )
    tracked = root / "results/phase2_vetted_corpus"
    contract = _read(tracked / CONTRACT_OUTPUT)
    _verify(contract, "qualification_contract_sha256")
    selected = cast(Mapping[str, Mapping[str, Any]], contract["selected_candidates"])
    deterministic_replays = {
        arm: {
            "passed": True,
            "group_id": selected[arm]["task_group_id"],
            "independently_reset_processes": 2,
            "exact_projection_sha256": exact_hashes[arm],
        }
        for arm in ARMS
    }
    audit_root = root / "results/raw/phase2_vetted_corpus/milestone14b_r1/signal_audit"
    signal_summary = build_signal_summary(
        _read(audit_root / "generic/raw_evidence.json"),
        _read(audit_root / "targeted/raw_evidence.json"),
        deterministic_replays=deterministic_replays,
    )
    if (
        signal_summary.get("decision") != "schedule_viable"
        or signal_summary.get("viability_passed") is not True
    ):
        raise RuntimeError("fresh-process replay did not complete the viability gate")
    decision = build_selection_and_gradient_decision(
        contract=contract,
        signal_summary=signal_summary,
        summaries=summaries,
    )
    write_selection_and_signal_decision(
        root,
        signal_summary=signal_summary,
        decision=decision,
    )
    campaign: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": "foundry-l3-grpo-signal-qualification-campaign-v1",
        "source_commit": source_commit,
        "process_order": process_order,
        "fresh_processes": 4,
        "retries": 0,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "exact_projection_sha256s": exact_hashes,
        "signal_summary_sha256": signal_summary["signal_summary_sha256"],
        "selection_decision_sha256": decision["selection_decision_sha256"],
        "gate_passed": True,
    }
    campaign["campaign_sha256"] = canonical_sha256(campaign)
    _write_new(raw_root / "campaign_summary.json", campaign)
    return campaign


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_campaign(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
