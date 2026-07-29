"""Run both Milestone 14B full-schedule signal audits sequentially."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_campaign import (
    _environment,
    _require_clean_synchronized_main,
    _run,
)
from foundry.phase2.l3_grpo_contract import INTERPRETER_SHA256, adapter_path
from foundry.phase2.l3_grpo_signal_audit import (
    ARMS,
    COMPLETIONS_PER_ARM,
    GROUPS_PER_ARM,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256


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


def _runtime_command(root: Path, arm: str, run_root: Path) -> list[str]:
    training_python = root / ".venv-training/Scripts/python.exe"
    if file_sha256(training_python) != INTERPRETER_SHA256:
        raise ValueError("authorized model interpreter differs")
    return [
        str(training_python),
        "-m",
        "foundry.phase2.l3_grpo_signal_runtime",
        "--root",
        str(root),
        "--arm",
        arm,
        "--packet",
        str(
            root
            / f"results/raw/phase2_vetted_corpus/milestone14a/schedules/{arm}_prompt_packet.json"
        ),
        "--manifest",
        str(root / f"results/phase2_vetted_corpus/milestone14a_{arm}_schedule.json"),
        "--audit-contract",
        str(root / "results/phase2_vetted_corpus/milestone14b_signal_audit_contract.json"),
        "--starting-adapter",
        str(adapter_path(root, arm)),
        "--raw-evidence",
        str(run_root / "raw_evidence.json"),
        "--summary",
        str(run_root / "summary.json"),
    ]


def run_campaign(root: Path) -> dict[str, object]:
    """Run generic then targeted once each with no optimizer or retry path."""

    root = root.resolve()
    environment = _environment(root)
    source_commit = _require_clean_synchronized_main(root)
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone14b/signal_audit"
    results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        _require_clean_synchronized_main(root)
        run_root = raw_root / arm
        _run(
            _runtime_command(root, arm, run_root),
            root=root,
            environment=environment,
            stdout=raw_root / "logs" / f"{arm}.stdout.txt",
            stderr=raw_root / "logs" / f"{arm}.stderr.txt",
        )
        summary = _read(run_root / "summary.json")
        _verify(summary, "summary_sha256")
        if (
            summary.get("groups") != GROUPS_PER_ARM
            or summary.get("completions") != COMPLETIONS_PER_ARM
            or summary.get("optimizer_created") is not False
            or summary.get("backward_calls") != 0
            or summary.get("scheduler_created") is not False
            or summary.get("adapter_saved") is not False
            or summary.get("gate_passed") is not True
        ):
            raise RuntimeError(f"{arm} signal-audit gate failed")
        results[arm] = summary
    campaign: dict[str, object] = {
        "schema_version": 1,
        "campaign_id": "foundry-l3-grpo-signal-audit-campaign-v1",
        "source_commit": source_commit,
        "arm_order": list(ARMS),
        "processes": len(ARMS),
        "groups": sum(cast(int, results[arm]["groups"]) for arm in ARMS),
        "completions": sum(cast(int, results[arm]["completions"]) for arm in ARMS),
        "optimizer_created": False,
        "backward_calls": 0,
        "scheduler_created": False,
        "adapter_saved": False,
        "summaries": {
            arm: {
                "summary_sha256": results[arm]["summary_sha256"],
                "raw_evidence_file_sha256": results[arm]["raw_evidence_file_sha256"],
            }
            for arm in ARMS
        },
    }
    campaign["campaign_sha256"] = canonical_sha256(campaign)
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
