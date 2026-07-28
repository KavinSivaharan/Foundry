"""Run Milestone 13E training and evaluation stages sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from foundry.phase2.layer_restricted import (
    ARMS,
    CHECKPOINTS_DESCENDING,
    DEVELOPMENT_SUITES,
    LAYER_SCOPES,
    SCHEDULES,
)
from foundry.phase2.windows_environment import (
    load_frozen_child_environment,
    validate_child_environment,
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


def _write_status(path: Path, value: dict[str, Any]) -> None:
    payload = {name: item for name, item in value.items() if name != "status_sha256"}
    value["status_sha256"] = canonical_sha256(payload)
    temporary = path.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
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


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        shell=False,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.stdout.strip()


def _repository_commit(root: Path) -> str:
    if _git(root, "branch", "--show-current") != "main":
        raise RuntimeError("campaign requires local main")
    head = _git(root, "rev-parse", "HEAD")
    if head != _git(root, "rev-parse", "origin/main"):
        raise RuntimeError("local main and origin/main differ")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("campaign requires a clean repository")
    return head


def _model_path(root: Path) -> Path:
    return (
        root
        / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
        / "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )


def _training_command(
    root: Path,
    *,
    arm: str,
    scope_label: str,
    max_steps: int,
    run_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "foundry.phase2.vetted_qlora_layer_restricted",
        "--arm",
        arm,
        "--scope-label",
        scope_label,
        "--max-steps",
        str(max_steps),
        "--model-path",
        str(_model_path(root)),
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


def _development_paths(root: Path, suite: str) -> tuple[Path, Path]:
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
    return (
        root / "results/raw/training/retention_powered_adjudication" / suite_name,
        root / "results/training" / subset_name,
    )


def _retention_command(
    root: Path,
    run_root: Path,
    suite_path: Path,
    subset_path: Path,
    adapter: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "foundry.training.retention",
        "--suite",
        str(suite_path),
        "--model-path",
        str(_model_path(root)),
        "--adapter",
        str(adapter),
        "--raw-path",
        str(run_root / "raw.json"),
        "--output-path",
        str(run_root / "summary.json"),
        "--subset-manifest",
        str(subset_path),
    ]


def _assessment_command(
    run_root: Path,
    suite_path: Path,
    subset_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "foundry.training.base_conditioned_retention",
        "assess",
        "--suite",
        str(suite_path),
        "--subset",
        str(subset_path),
        "--summary",
        str(run_root / "summary.json"),
        "--raw",
        str(run_root / "raw.json"),
        "--output",
        str(run_root / "assessment.json"),
    ]


def _gsm1k_command(
    root: Path,
    run_root: Path,
    adapter: Path,
    adapter_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "foundry.training.adapter_evaluation",
        "--base-config",
        str(root / "configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml"),
        "--config",
        str(root / "configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml"),
        "--development-manifest",
        str(root / "configs/eval/manifests/gsm1k_development.json"),
        "--source-pool-manifest",
        str(root / "configs/eval/manifests/gsm1k_development_baseline.json"),
        "--source-baseline-manifest",
        str(root / "configs/eval/manifests/gsm1k_development_baseline_844.json"),
        "--baseline-manifest",
        str(root / "configs/eval/manifests/gsm1k_development_baseline_814.json"),
        "--model-path",
        str(_model_path(root)),
        "--adapter",
        str(adapter),
        "--adapter-sha256",
        adapter_sha256,
        "--adapter-scale",
        "1.0",
        "--output-dir",
        str(run_root / "output"),
        "--tracked-summary",
        str(run_root / "summary.json"),
    ]


def _training_complete(run_root: Path, steps: int, scope_label: str, arm: str) -> bool:
    summary_path = run_root / "training_summary.json"
    adapter = run_root / f"training/checkpoint-{steps}/adapter"
    if not summary_path.exists() or not adapter.exists():
        return False
    value = _read(summary_path)
    _verify(value, "result_sha256")
    return (
        value.get("optimizer_steps") == steps
        and value.get("scope_label") == scope_label
        and value.get("arm") == arm
    )


def _assessment_pass(path: Path) -> bool:
    value = _read(path)
    _verify(value, "summary_sha256")
    if value["backend_failures"] != 0:
        raise RuntimeError("retention backend failure violates the campaign contract")
    return value["gate_passed"] is True


def _evaluation_complete(run_root: Path) -> bool:
    return all(
        (run_root / name).exists() for name in ("raw.json", "summary.json", "assessment.json")
    )


def _environment(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    raw = (
        root
        / "results/raw/phase2_vetted_corpus/milestone13c_r1"
        / "windows_operational_environment_v2_raw.json"
    )
    tracked = root / "results/phase2_vetted_corpus/windows_operational_environment.json"
    evidence = _read(tracked)
    environment = load_frozen_child_environment(
        raw_environment_path=raw,
        tracked_evidence_path=tracked,
    )
    validate_child_environment(dict(os.environ), evidence)
    if dict(os.environ) != environment:
        raise RuntimeError("campaign parent differs from exact frozen child environment")
    return environment, evidence


def _base_status(
    root: Path,
    stage: str,
    repository_commit: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": f"foundry-milestone13e-{stage}-campaign-v1",
        "stage": stage,
        "repository_commit": repository_commit,
        "environment_evidence_sha256": environment["environment_evidence_sha256"],
        "combined_child_environment_sha256": environment["combined_child_environment_sha256"],
        "interpreter_sha256": file_sha256(Path(sys.executable)),
        "completed_runs": [],
        "current_run": None,
        "current_stage": None,
        "failed": False,
        "complete": False,
    }


def _evaluate(
    *,
    root: Path,
    environment: dict[str, str],
    adapter: Path,
    suite_path: Path,
    subset_path: Path,
    run_root: Path,
    log_root: Path,
    log_label: str,
) -> None:
    if run_root.exists():
        raise FileExistsError(f"retention output already exists: {run_root}")
    _run(
        command=_retention_command(
            root,
            run_root,
            suite_path,
            subset_path,
            adapter,
        ),
        cwd=root / "src",
        environment=environment,
        stdout_path=log_root / f"{log_label}.stdout.txt",
        stderr_path=log_root / f"{log_label}.stderr.txt",
    )
    _run(
        command=_assessment_command(run_root, suite_path, subset_path),
        cwd=root / "src",
        environment=environment,
        stdout_path=log_root / f"{log_label}-assessment.stdout.txt",
        stderr_path=log_root / f"{log_label}-assessment.stderr.txt",
    )
    if not _evaluation_complete(run_root):
        raise RuntimeError("retention evaluation did not produce complete evidence")
    _assessment_pass(run_root / "assessment.json")


def _run_calibration(
    root: Path,
    status_path: Path,
    environment: dict[str, str],
    status: dict[str, Any],
) -> None:
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13e/calibration"
    logs = root / "results/raw/phase2_vetted_corpus/milestone13e/calibration_logs"
    for scope in LAYER_SCOPES:
        for arm in ARMS:
            run_name = f"{scope.label}/{arm}"
            run_root = raw_root / scope.label / arm
            if _training_complete(run_root, 16, scope.label, arm):
                if not all(
                    _evaluation_complete(run_root / "retention" / suite)
                    for suite in DEVELOPMENT_SUITES
                ):
                    raise FileExistsError(f"incomplete calibration run: {run_name}")
            elif run_root.exists():
                raise FileExistsError(f"incomplete calibration run: {run_name}")
            else:
                status["current_run"] = run_name
                status["current_stage"] = "training"
                _write_status(status_path, status)
                print(json.dumps({"run": run_name, "stage": "training"}), flush=True)
                _run(
                    command=_training_command(
                        root,
                        arm=arm,
                        scope_label=scope.label,
                        max_steps=16,
                        run_root=run_root,
                    ),
                    cwd=root / "src",
                    environment=environment,
                    stdout_path=logs / f"{scope.label}-{arm}.stdout.txt",
                    stderr_path=logs / f"{scope.label}-{arm}.stderr.txt",
                )
                if not _training_complete(run_root, 16, scope.label, arm):
                    raise RuntimeError("calibration training completion gate failed")
                adapter = run_root / "training/checkpoint-16/adapter"
                for suite in DEVELOPMENT_SUITES:
                    status["current_stage"] = f"{suite}_evaluation"
                    _write_status(status_path, status)
                    print(
                        json.dumps({"run": run_name, "stage": f"{suite}_evaluation"}),
                        flush=True,
                    )
                    suite_path, subset_path = _development_paths(root, suite)
                    _evaluate(
                        root=root,
                        environment=environment,
                        adapter=adapter,
                        suite_path=suite_path,
                        subset_path=subset_path,
                        run_root=run_root / "retention" / suite,
                        log_root=logs,
                        log_label=f"{scope.label}-{arm}-{suite}",
                    )
            cast(list[str], status["completed_runs"]).append(run_name)
            status["current_stage"] = "complete"
            _write_status(status_path, status)
            print(json.dumps({"run": run_name, "stage": "complete"}), flush=True)


def _run_full(
    root: Path,
    status_path: Path,
    environment: dict[str, str],
    status: dict[str, Any],
    calibration_path: Path,
) -> None:
    calibration = _read(calibration_path)
    _verify(calibration, "calibration_summary_sha256")
    selected_scope = calibration["selected_scope"]
    if not isinstance(selected_scope, str):
        raise ValueError("no layer scope passed calibration")
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13e/full"
    logs = root / "results/raw/phase2_vetted_corpus/milestone13e/full_logs"
    status["selected_scope"] = selected_scope
    status["evaluated_checkpoints"] = []
    status["selected_checkpoint"] = None
    _write_status(status_path, status)
    for arm in ARMS:
        run_root = raw_root / arm
        if not _training_complete(run_root, 64, selected_scope, arm):
            if run_root.exists():
                raise FileExistsError(f"incomplete full training run: {arm}")
            status["current_run"] = arm
            status["current_stage"] = "training"
            _write_status(status_path, status)
            print(json.dumps({"run": arm, "stage": "training"}), flush=True)
            _run(
                command=_training_command(
                    root,
                    arm=arm,
                    scope_label=selected_scope,
                    max_steps=64,
                    run_root=run_root,
                ),
                cwd=root / "src",
                environment=environment,
                stdout_path=logs / f"{arm}.stdout.txt",
                stderr_path=logs / f"{arm}.stderr.txt",
            )
            if not _training_complete(run_root, 64, selected_scope, arm):
                raise RuntimeError("full training completion gate failed")
        cast(list[str], status["completed_runs"]).append(f"training/{arm}")
        _write_status(status_path, status)
    for checkpoint in CHECKPOINTS_DESCENDING:
        common_pass = True
        for arm in ARMS:
            adapter = raw_root / arm / f"training/checkpoint-{checkpoint}/adapter"
            for suite in DEVELOPMENT_SUITES:
                status["current_run"] = f"checkpoint-{checkpoint}/{arm}"
                status["current_stage"] = f"{suite}_evaluation"
                _write_status(status_path, status)
                print(
                    json.dumps(
                        {
                            "run": status["current_run"],
                            "stage": status["current_stage"],
                        }
                    ),
                    flush=True,
                )
                suite_path, subset_path = _development_paths(root, suite)
                run_root = (
                    raw_root
                    / "development_selection"
                    / f"checkpoint-{checkpoint}"
                    / arm
                    / "retention"
                    / suite
                )
                if not _evaluation_complete(run_root):
                    _evaluate(
                        root=root,
                        environment=environment,
                        adapter=adapter,
                        suite_path=suite_path,
                        subset_path=subset_path,
                        run_root=run_root,
                        log_root=logs,
                        log_label=f"checkpoint-{checkpoint}-{arm}-{suite}",
                    )
                common_pass = _assessment_pass(run_root / "assessment.json") and common_pass
        cast(list[int], status["evaluated_checkpoints"]).append(checkpoint)
        _write_status(status_path, status)
        if common_pass:
            status["selected_checkpoint"] = checkpoint
            break
    if status["selected_checkpoint"] is None:
        status["stop_reason"] = "no_common_development_passing_checkpoint"


def _run_holdout(
    root: Path,
    status_path: Path,
    environment: dict[str, str],
    status: dict[str, Any],
    full_selection_path: Path,
) -> None:
    selection = _read(full_selection_path)
    _verify(selection, "full_selection_sha256")
    checkpoint = selection["selected_checkpoint"]
    scope = selection["selected_scope"]
    if not isinstance(checkpoint, int) or not isinstance(scope, str):
        raise ValueError("holdout v2 requires a selected full checkpoint")
    adapters = cast(dict[str, str], selection["selected_adapter_sha256_by_arm"])
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13e/holdout_v2"
    logs = root / "results/raw/phase2_vetted_corpus/milestone13e/holdout_v2_logs"
    suite_path = (
        root
        / "results/raw/phase2_vetted_corpus/milestone13c_r2"
        / "combined_holdout/candidate_suite.json"
    )
    subset_path = (
        root / "results/phase2_vetted_corpus" / "milestone13c_r2_combined_retention_subset.json"
    )
    status["selected_scope"] = scope
    status["selected_checkpoint"] = checkpoint
    status["adapter_evaluation_order"] = list(ARMS)
    _write_status(status_path, status)
    arm_passes: dict[str, bool] = {}
    for arm in ARMS:
        run_root = raw_root / arm
        if run_root.exists():
            raise FileExistsError(f"holdout-v2 arm already has evidence; repeat prohibited: {arm}")
        status["current_run"] = arm
        status["current_stage"] = "holdout_v2_evaluation"
        _write_status(status_path, status)
        print(
            json.dumps({"run": arm, "stage": "holdout_v2_evaluation"}),
            flush=True,
        )
        adapter = (
            root
            / f"results/raw/phase2_vetted_corpus/milestone13e/full/{arm}"
            / f"training/checkpoint-{checkpoint}/adapter"
        )
        _evaluate(
            root=root,
            environment=environment,
            adapter=adapter,
            suite_path=suite_path,
            subset_path=subset_path,
            run_root=run_root,
            log_root=logs,
            log_label=arm,
        )
        value = _read(run_root / "assessment.json")
        if value["adapter_sha256"] != adapters[arm]:
            raise RuntimeError("holdout-v2 adapter identity differs")
        arm_passes[arm] = value["gate_passed"] is True
        cast(list[str], status["completed_runs"]).append(arm)
        _write_status(status_path, status)
    status["both_arms_pass"] = all(arm_passes.values())
    if not status["both_arms_pass"]:
        status["stop_reason"] = "holdout_v2_failed"


def _run_gsm1k(
    root: Path,
    status_path: Path,
    environment: dict[str, str],
    status: dict[str, Any],
    full_selection_path: Path,
    holdout_path: Path,
) -> None:
    selection = _read(full_selection_path)
    _verify(selection, "full_selection_sha256")
    holdout = _read(holdout_path)
    _verify(holdout, "holdout_decision_sha256")
    if holdout["gsm1k_authorized"] is not True:
        raise ValueError("GSM1K is unauthorized because holdout v2 did not pass")
    checkpoint = selection["selected_checkpoint"]
    if not isinstance(checkpoint, int):
        raise ValueError("GSM1K requires a selected full checkpoint")
    adapters = cast(dict[str, str], selection["selected_adapter_sha256_by_arm"])
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone13e/gsm1k"
    logs = root / "results/raw/phase2_vetted_corpus/milestone13e/gsm1k_logs"
    status["base_result"] = "frozen_phase1_521_of_814"
    status["adapter_evaluation_order"] = list(ARMS)
    _write_status(status_path, status)
    for arm in ARMS:
        run_root = raw_root / arm
        if run_root.exists():
            raise FileExistsError(f"GSM1K arm already has evidence; repeat prohibited: {arm}")
        status["current_run"] = arm
        status["current_stage"] = "gsm1k_evaluation"
        _write_status(status_path, status)
        print(
            json.dumps({"run": arm, "stage": "gsm1k_evaluation"}),
            flush=True,
        )
        adapter = (
            root
            / f"results/raw/phase2_vetted_corpus/milestone13e/full/{arm}"
            / f"training/checkpoint-{checkpoint}/adapter"
        )
        _run(
            command=_gsm1k_command(root, run_root, adapter, adapters[arm]),
            cwd=root / "src",
            environment=environment,
            stdout_path=logs / f"{arm}.stdout.txt",
            stderr_path=logs / f"{arm}.stderr.txt",
        )
        if not (run_root / "summary.json").exists():
            raise RuntimeError("GSM1K evaluation summary is absent")
        value = _read(run_root / "summary.json")
        if value["generation_failures"] != 0 or value["processed_examples"] != 814:
            raise RuntimeError("GSM1K evaluator completeness gate failed")
        cast(list[str], status["completed_runs"]).append(arm)
        _write_status(status_path, status)


def run_campaign(
    root: Path,
    stage: str,
    status_path: Path,
    *,
    calibration_path: Path | None = None,
    full_selection_path: Path | None = None,
    holdout_path: Path | None = None,
) -> dict[str, Any]:
    """Run exactly one authorized milestone stage under the frozen child environment."""

    if stage not in {"calibration", "full", "holdout", "gsm1k"}:
        raise ValueError("campaign stage is not authorized")
    environment, evidence = _environment(root)
    repository_commit = _repository_commit(root)
    status = _base_status(root, stage, repository_commit, evidence)
    _write_status(status_path, status)
    try:
        if stage == "calibration":
            _run_calibration(root, status_path, environment, status)
        elif stage == "full":
            if calibration_path is None:
                raise ValueError("full stage requires calibration evidence")
            _run_full(
                root,
                status_path,
                environment,
                status,
                calibration_path,
            )
        elif stage == "holdout":
            if full_selection_path is None:
                raise ValueError("holdout stage requires full-selection evidence")
            _run_holdout(
                root,
                status_path,
                environment,
                status,
                full_selection_path,
            )
        else:
            if full_selection_path is None or holdout_path is None:
                raise ValueError("GSM1K requires full-selection and holdout evidence")
            _run_gsm1k(
                root,
                status_path,
                environment,
                status,
                full_selection_path,
                holdout_path,
            )
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
    parser.add_argument(
        "--stage",
        choices=("calibration", "full", "holdout", "gsm1k"),
        required=True,
    )
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--full-selection", type=Path)
    parser.add_argument("--holdout-decision", type=Path)
    args = parser.parse_args()
    if args.status.exists():
        raise FileExistsError("campaign status already exists")
    result = run_campaign(
        args.root.resolve(),
        args.stage,
        args.status.resolve(),
        calibration_path=(args.calibration.resolve() if args.calibration is not None else None),
        full_selection_path=(
            args.full_selection.resolve() if args.full_selection is not None else None
        ),
        holdout_path=(
            args.holdout_decision.resolve() if args.holdout_decision is not None else None
        ),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
