"""Run the predeclared Milestone 13C-R3 calibration matrix sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from foundry.phase2.windows_environment import (
    load_frozen_child_environment,
    validate_child_environment,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

RUNS = (
    ("lambda-001", "generic", "0.01"),
    ("lambda-001", "targeted", "0.01"),
    ("lambda-003", "generic", "0.03"),
    ("lambda-003", "targeted", "0.03"),
    ("lambda-010", "generic", "0.10"),
    ("lambda-010", "targeted", "0.10"),
    ("lambda-030", "generic", "0.30"),
    ("lambda-030", "targeted", "0.30"),
)
SCHEDULES = {
    "generic": "4bc00d29d5cf308c12c77111d7943567521cc533b13440dc06c3d8b39c74e9df",
    "targeted": "88c5378cac7efe927b29d3f421d97777cd6d917187c71c8388b60bbe7b57e259",
}


def _read_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verified(path: Path, key: str) -> dict[str, Any]:
    value = _read_object(path)
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{path} has an invalid {key}")
    return value


def _write_status(path: Path, value: dict[str, Any]) -> None:
    value["status_sha256"] = canonical_sha256(
        {name: item for name, item in value.items() if name != "status_sha256"}
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(
    *,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("campaign command log already exists")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            command,
            shell=False,
            env=environment,
            cwd=cwd,
            check=False,
            stdout=stdout,
            stderr=stderr,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    if result.returncode != 0:
        raise RuntimeError(f"campaign command failed with {result.returncode}: {stderr_path}")


def _training_command(root: Path, arm: str, coefficient: str, run_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "foundry.phase2.vetted_qlora_kl",
        "train",
        "--arm",
        arm,
        "--coefficient",
        coefficient,
        "--max-steps",
        "16",
        "--model-path",
        str(
            root
            / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
            / "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        ),
        "--vetted-path",
        str(root / f"results/raw/phase2_vetted_corpus/dataset/{arm}_training.jsonl"),
        "--validation-path",
        str(root / f"results/raw/phase2_vetted_corpus/dataset/{arm}_validation.jsonl"),
        "--replay-path",
        str(root / "results/raw/training/base_replay_kl/replay_corpus.json"),
        "--schedule-path",
        str(root / f"results/raw/phase2_vetted_corpus/v1_replay25_schedules/{arm}_schedule.json"),
        "--schedule-sha256",
        SCHEDULES[arm],
        "--recipe-path",
        str(root / "results/phase2_vetted_corpus/milestone13c_r3_v1_kl_recipe.json"),
        "--output-directory",
        str(run_root / "training"),
        "--summary-path",
        str(run_root / "training_summary.json"),
    ]


def _retention_command(root: Path, run_root: Path, suite: str, adapter: Path) -> list[str]:
    names = {
        "adjudication": (
            "retention_adjudication_v2.json",
            "retention_adjudication_v2_base_correct_subset.json",
        ),
        "anchor": (
            "retention_anchor_holdout_v1.json",
            "retention_anchor_holdout_v1_base_correct_subset.json",
        ),
    }
    suite_name, subset_name = names[suite]
    return [
        sys.executable,
        "-m",
        "foundry.training.retention",
        "--suite",
        str(root / f"results/raw/training/retention_powered_adjudication/{suite_name}"),
        "--model-path",
        str(
            root
            / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
            / "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        ),
        "--adapter",
        str(adapter),
        "--raw-path",
        str(run_root / f"retention/{suite}_raw.json"),
        "--output-path",
        str(run_root / f"retention/{suite}_summary.json"),
        "--subset-manifest",
        str(root / f"results/training/{subset_name}"),
    ]


def _assessment_command(root: Path, run_root: Path, suite: str) -> list[str]:
    names = {
        "adjudication": (
            "retention_adjudication_v2.json",
            "retention_adjudication_v2_base_correct_subset.json",
        ),
        "anchor": (
            "retention_anchor_holdout_v1.json",
            "retention_anchor_holdout_v1_base_correct_subset.json",
        ),
    }
    suite_name, subset_name = names[suite]
    return [
        sys.executable,
        "-m",
        "foundry.training.base_conditioned_retention",
        "assess",
        "--suite",
        str(root / f"results/raw/training/retention_powered_adjudication/{suite_name}"),
        "--subset",
        str(root / f"results/training/{subset_name}"),
        "--summary",
        str(run_root / f"retention/{suite}_summary.json"),
        "--raw",
        str(run_root / f"retention/{suite}_raw.json"),
        "--output",
        str(run_root / f"retention/{suite}_assessment.json"),
    ]


def _complete(run_root: Path) -> bool:
    paths = [
        run_root / "training_summary.json",
        run_root / "retention/adjudication_assessment.json",
        run_root / "retention/anchor_assessment.json",
    ]
    if not all(path.exists() for path in paths):
        return False
    training = _verified(paths[0], "result_sha256")
    assessments = [_verified(path, "summary_sha256") for path in paths[1:]]
    return (
        training.get("optimizer_steps") == 16
        and training.get("loss_bearing_tokens") == 16_000
        and all(value.get("backend_failures") == 0 for value in assessments)
    )


def run_campaign(root: Path, status_path: Path) -> dict[str, Any]:
    """Execute unfinished matrix cells, one model child at a time."""

    raw_environment_path = (
        root
        / "results/raw/phase2_vetted_corpus/milestone13c_r1"
        / "windows_operational_environment_v2_raw.json"
    )
    tracked_environment_path = (
        root / "results/phase2_vetted_corpus/windows_operational_environment.json"
    )
    environment = load_frozen_child_environment(
        raw_environment_path=raw_environment_path,
        tracked_evidence_path=tracked_environment_path,
    )
    validate_child_environment(dict(os.environ), _read_object(tracked_environment_path))
    if dict(os.environ) != environment:
        raise RuntimeError("campaign parent differs from exact frozen child environment")
    root_raw = root / "results/raw/phase2_vetted_corpus/milestone13c_r3"
    logs = root_raw / "calibration_logs"
    status: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": "foundry-milestone13c-r3-calibration-campaign-v1",
        "completed_runs": [],
        "current_run": None,
        "current_stage": None,
        "failed": False,
        "complete": False,
        "interpreter_sha256": file_sha256(Path(sys.executable)),
    }
    _write_status(status_path, status)
    try:
        for coefficient_name, arm, coefficient in RUNS:
            run_name = f"{coefficient_name}/{arm}"
            run_root = root_raw / f"calibration/{run_name}"
            if _complete(run_root):
                cast(list[str], status["completed_runs"]).append(run_name)
                _write_status(status_path, status)
                continue
            if run_root.exists():
                raise FileExistsError(f"incomplete calibration run already exists: {run_name}")
            status["current_run"] = run_name
            status["current_stage"] = "training"
            _write_status(status_path, status)
            print(json.dumps({"run": run_name, "stage": "training"}), flush=True)
            _run(
                command=_training_command(root, arm, coefficient, run_root),
                cwd=root / "src",
                environment=environment,
                stdout_path=logs / f"{coefficient_name}-{arm}.stdout.txt",
                stderr_path=logs / f"{coefficient_name}-{arm}.stderr.txt",
            )
            training = _verified(run_root / "training_summary.json", "result_sha256")
            if (
                training.get("optimizer_steps") != 16
                or training.get("loss_bearing_tokens") != 16_000
            ):
                raise RuntimeError("calibration training completion gate failed")
            adapter = run_root / "training/checkpoint-16/adapter"
            for suite in ("adjudication", "anchor"):
                status["current_stage"] = f"{suite}_evaluation"
                _write_status(status_path, status)
                print(
                    json.dumps({"run": run_name, "stage": f"{suite}_evaluation"}),
                    flush=True,
                )
                _run(
                    command=_retention_command(root, run_root, suite, adapter),
                    cwd=root / "src",
                    environment=environment,
                    stdout_path=logs / f"{coefficient_name}-{arm}-{suite}.stdout.txt",
                    stderr_path=logs / f"{coefficient_name}-{arm}-{suite}.stderr.txt",
                )
                status["current_stage"] = f"{suite}_assessment"
                _write_status(status_path, status)
                _run(
                    command=_assessment_command(root, run_root, suite),
                    cwd=root / "src",
                    environment=environment,
                    stdout_path=logs / f"{coefficient_name}-{arm}-{suite}-assessment.stdout.txt",
                    stderr_path=logs / f"{coefficient_name}-{arm}-{suite}-assessment.stderr.txt",
                )
                assessment = _verified(
                    run_root / f"retention/{suite}_assessment.json",
                    "summary_sha256",
                )
                if assessment.get("backend_failures") != 0:
                    raise RuntimeError("calibration development retention has backend failures")
            if not _complete(run_root):
                raise RuntimeError("calibration run did not complete")
            cast(list[str], status["completed_runs"]).append(run_name)
            print(json.dumps({"run": run_name, "stage": "complete"}), flush=True)
            _write_status(status_path, status)
        status["current_run"] = None
        status["current_stage"] = None
        status["complete"] = True
    except Exception as error:
        status["failed"] = True
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        _write_status(status_path, status)
        raise
    _write_status(status_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    if args.status.exists():
        raise FileExistsError("campaign status already exists")
    result = run_campaign(args.root.resolve(), args.status.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
