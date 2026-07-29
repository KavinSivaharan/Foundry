"""Run each Milestone 14A model stage sequentially under the frozen environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from foundry.phase2.l3_grpo_analysis import (
    ARMS,
    CHECKPOINTS,
    DEVELOPMENT_SUITES,
    write_compatibility_result,
    write_counted_training_result,
    write_development_selection,
    write_gsm1k_analysis,
    write_holdout_decision,
)
from foundry.phase2.l3_grpo_contract import adapter_path
from foundry.phase2.windows_environment import (
    load_frozen_child_environment,
    validate_child_environment,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

Stage = Literal[
    "smoke",
    "train-generic",
    "train-targeted",
    "development",
    "holdout",
    "gsm1k",
]


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


def _environment(root: Path) -> dict[str, str]:
    raw = (
        root
        / "results/raw/phase2_vetted_corpus/milestone13c_r1"
        / "windows_operational_environment_v2_raw.json"
    )
    tracked = root / "results/phase2_vetted_corpus/windows_operational_environment.json"
    evidence = _read(tracked)
    child = load_frozen_child_environment(
        raw_environment_path=raw,
        tracked_evidence_path=tracked,
    )
    validate_child_environment(dict(os.environ), evidence)
    if dict(os.environ) != child:
        raise RuntimeError("campaign parent differs from the exact frozen child environment")
    if file_sha256(Path(sys.executable)) != (
        "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
    ):
        raise RuntimeError("campaign interpreter differs")
    return child


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


def _require_clean_synchronized_main(root: Path) -> str:
    if _git(root, "branch", "--show-current") != "main":
        raise RuntimeError("counted stage requires local main")
    head = _git(root, "rev-parse", "HEAD")
    if (
        head != _git(root, "rev-parse", "origin/main")
        or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
        != ["0", "0"]
        or _git(root, "status", "--porcelain")
    ):
        raise RuntimeError("counted stage requires synchronized clean main")
    return head


def _run(
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str],
    stdout: Path,
    stderr: Path,
) -> None:
    if stdout.exists() or stderr.exists():
        raise FileExistsError("campaign command log already exists")
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout.open("w", encoding="utf-8") as stdout_handle,
        stderr.open("w", encoding="utf-8") as stderr_handle,
    ):
        result = subprocess.run(
            command,
            cwd=root / "src",
            env=environment,
            shell=False,
            check=False,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    if result.returncode != 0:
        raise RuntimeError(f"campaign command failed with {result.returncode}; stderr={stderr}")


def _runtime_command(
    root: Path,
    *,
    arm: str,
    mode: str,
    output_dir: Path,
    raw_evidence: Path,
    summary: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "foundry.phase2.l3_grpo_runtime",
        "--root",
        str(root),
        "--arm",
        arm,
        "--mode",
        mode,
        "--packet",
        str(
            root
            / f"results/raw/phase2_vetted_corpus/milestone14a/schedules/{arm}_prompt_packet.json"
        ),
        "--manifest",
        str(root / f"results/phase2_vetted_corpus/milestone14a_{arm}_schedule.json"),
        "--experiment-contract",
        str(root / "results/phase2_vetted_corpus/milestone14a_experiment_contract.json"),
        "--starting-adapter",
        str(adapter_path(root, arm)),
        "--output-dir",
        str(output_dir),
        "--raw-evidence",
        str(raw_evidence),
        "--summary",
        str(summary),
    ]


def _model_path(root: Path) -> Path:
    return (
        root
        / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
        / "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )


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
    *,
    suite: Path,
    subset: Path,
    adapter: Path,
    run_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "foundry.training.retention",
        "--suite",
        str(suite),
        "--model-path",
        str(_model_path(root)),
        "--adapter",
        str(adapter),
        "--raw-path",
        str(run_root / "raw.json"),
        "--output-path",
        str(run_root / "summary.json"),
        "--subset-manifest",
        str(subset),
    ]


def _assessment_command(
    *,
    suite: Path,
    subset: Path,
    run_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "foundry.training.base_conditioned_retention",
        "assess",
        "--suite",
        str(suite),
        "--subset",
        str(subset),
        "--summary",
        str(run_root / "summary.json"),
        "--raw",
        str(run_root / "raw.json"),
        "--output",
        str(run_root / "assessment.json"),
    ]


def _evaluate_retention(
    root: Path,
    environment: dict[str, str],
    *,
    adapter: Path,
    suite: Path,
    subset: Path,
    run_root: Path,
    log_root: Path,
    label: str,
) -> None:
    if run_root.exists():
        raise FileExistsError(f"retention evaluation path already exists: {run_root}")
    _run(
        _retention_command(
            root,
            suite=suite,
            subset=subset,
            adapter=adapter,
            run_root=run_root,
        ),
        root=root,
        environment=environment,
        stdout=log_root / f"{label}.stdout.txt",
        stderr=log_root / f"{label}.stderr.txt",
    )
    _run(
        _assessment_command(suite=suite, subset=subset, run_root=run_root),
        root=root,
        environment=environment,
        stdout=log_root / f"{label}-assessment.stdout.txt",
        stderr=log_root / f"{label}-assessment.stderr.txt",
    )
    assessment = _read(run_root / "assessment.json")
    _verify(assessment, "summary_sha256")
    if assessment.get("backend_failures") != 0:
        raise RuntimeError("retention backend failure")


def _gsm1k_command(
    root: Path,
    *,
    adapter: Path,
    adapter_sha256: str,
    output_dir: Path,
    summary: Path,
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
        str(output_dir),
        "--tracked-summary",
        str(summary),
    ]


def _run_smoke(root: Path, environment: dict[str, str]) -> dict[str, object]:
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a"
    for index in (1, 2):
        run_root = raw / f"compatibility/run-{index}"
        _run(
            _runtime_command(
                root,
                arm="generic",
                mode="compatibility",
                output_dir=run_root / "artifacts",
                raw_evidence=run_root / "raw_evidence.json",
                summary=run_root / "summary.json",
            ),
            root=root,
            environment=environment,
            stdout=raw / f"compatibility/logs/run-{index}.stdout.txt",
            stderr=raw / f"compatibility/logs/run-{index}.stderr.txt",
        )
    return write_compatibility_result(root)


def _run_training(
    root: Path,
    environment: dict[str, str],
    arm: str,
) -> dict[str, Any]:
    _require_clean_synchronized_main(root)
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a"
    summary = raw / f"training/{arm}_summary.json"
    _run(
        _runtime_command(
            root,
            arm=arm,
            mode="train",
            output_dir=raw / f"training/{arm}",
            raw_evidence=raw / f"training/{arm}_raw_evidence.json",
            summary=summary,
        ),
        root=root,
        environment=environment,
        stdout=raw / f"training/logs/{arm}.stdout.txt",
        stderr=raw / f"training/logs/{arm}.stderr.txt",
    )
    value = _read(summary)
    _verify(value, "summary_sha256")
    if (
        value.get("gate_passed") is not True
        or value.get("groups") != 32
        or value.get("completions") != 128
        or value.get("optimizer_steps") != 32
    ):
        raise RuntimeError(f"{arm} counted training gate failed")
    return value


def _run_development(
    root: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    _require_clean_synchronized_main(root)
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a"
    logs = raw / "development/logs"
    for arm in ARMS:
        training = _read(raw / f"training/{arm}_summary.json")
        _verify(training, "summary_sha256")
        for checkpoint in CHECKPOINTS:
            adapter = raw / f"training/{arm}/checkpoint-{checkpoint}/adapter"
            expected = training["checkpoint_evidence"][str(checkpoint)]["directory_sha256"]
            if directory_sha256(adapter) != expected:
                raise ValueError("development checkpoint adapter identity differs")
            for suite_name in DEVELOPMENT_SUITES:
                suite, subset = _development_paths(root, suite_name)
                _evaluate_retention(
                    root,
                    environment,
                    adapter=adapter,
                    suite=suite,
                    subset=subset,
                    run_root=(raw / f"development/checkpoint-{checkpoint}/{arm}/{suite_name}"),
                    log_root=logs,
                    label=f"{arm}-checkpoint-{checkpoint}-{suite_name}",
                )
    return write_development_selection(root)


def _run_holdout(
    root: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    selection = _read(root / "results/phase2_vetted_corpus/milestone14a_development_selection.json")
    _verify(selection, "development_selection_sha256")
    checkpoint = selection.get("latest_common_passing_checkpoint")
    if not isinstance(checkpoint, int):
        raise ValueError("holdout is unauthorized without a common development checkpoint")
    suite = (
        root
        / "results/raw/phase2_vetted_corpus/milestone13c_r2"
        / "combined_holdout/candidate_suite.json"
    )
    subset = root / "results/phase2_vetted_corpus/milestone13c_r2_combined_retention_subset.json"
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a"
    for arm in ARMS:
        adapter = raw / f"training/{arm}/checkpoint-{checkpoint}/adapter"
        if directory_sha256(adapter) != selection["selected_adapter_sha256_by_arm"][arm]:
            raise ValueError("holdout adapter identity differs")
        _evaluate_retention(
            root,
            environment,
            adapter=adapter,
            suite=suite,
            subset=subset,
            run_root=raw / f"holdout_v2/{arm}",
            log_root=raw / "holdout_v2/logs",
            label=arm,
        )
    training = write_counted_training_result(root)
    holdout = write_holdout_decision(root)
    return {
        "training_result_sha256": training["training_result_sha256"],
        "holdout_decision_sha256": holdout["holdout_decision_sha256"],
        "both_arms_pass": holdout["both_arms_pass"],
    }


def _run_gsm1k(root: Path, environment: dict[str, str]) -> dict[str, object]:
    _require_clean_synchronized_main(root)
    holdout = _read(root / "results/phase2_vetted_corpus/milestone14a_holdout_v2_decision.json")
    _verify(holdout, "holdout_decision_sha256")
    if holdout.get("gsm1k_authorized") is not True:
        raise ValueError("GSM1K is unauthorized")
    checkpoint = cast(int, holdout["selected_checkpoint"])
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a"
    for arm in ARMS:
        adapter_sha256 = cast(str, holdout["arms"][arm]["adapter_sha256"])
        _run(
            _gsm1k_command(
                root,
                adapter=raw / f"training/{arm}/checkpoint-{checkpoint}/adapter",
                adapter_sha256=adapter_sha256,
                output_dir=raw / f"gsm1k/{arm}/output",
                summary=(root / f"results/phase2_vetted_corpus/milestone14a_gsm1k_{arm}.json"),
            ),
            root=root,
            environment=environment,
            stdout=raw / f"gsm1k/logs/{arm}.stdout.txt",
            stderr=raw / f"gsm1k/logs/{arm}.stderr.txt",
        )
    return write_gsm1k_analysis(root)


def run_stage(root: Path, stage: Stage) -> dict[str, object]:
    root = root.resolve()
    environment = _environment(root)
    if stage == "smoke":
        return _run_smoke(root, environment)
    if stage == "train-generic":
        return cast(dict[str, object], _run_training(root, environment, "generic"))
    if stage == "train-targeted":
        return cast(dict[str, object], _run_training(root, environment, "targeted"))
    if stage == "development":
        return _run_development(root, environment)
    if stage == "holdout":
        return _run_holdout(root, environment)
    return _run_gsm1k(root, environment)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "train-generic",
            "train-targeted",
            "development",
            "holdout",
            "gsm1k",
        ),
        required=True,
    )
    args = parser.parse_args(argv)
    result = run_stage(args.root, cast(Stage, args.stage))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
