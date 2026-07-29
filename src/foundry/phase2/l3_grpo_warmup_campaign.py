"""Run Milestone 14B-R2 stages sequentially under the frozen environment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from foundry.phase2.l3_grpo_campaign import (
    _environment,
    _require_clean_synchronized_main,
    _run,
)
from foundry.phase2.l3_grpo_contract import (
    INTERPRETER_SHA256,
    adapter_path,
)
from foundry.phase2.l3_grpo_warmup_analysis import (
    ARMS,
    CHECKPOINTS,
    DEVELOPMENT_OUTPUT,
    DEVELOPMENT_SUITES,
    HOLDOUT_OUTPUT,
    RAW_ROOT,
    TRACKED_ROOT,
    TRAINING_OUTPUT,
    write_counted_training_result,
    write_development_selection,
    write_gsm1k_analysis,
    write_holdout_decision,
)
from foundry.phase2.l3_grpo_warmup_compatibility_campaign import (
    OUTPUT_NAME as COMPATIBILITY_OUTPUT,
)
from foundry.phase2.l3_grpo_warmup_prepare import (
    CONTRACT_OUTPUT as WARMUP_CONTRACT_OUTPUT,
)
from foundry.phase2.l3_grpo_warmup_update import (
    INVALID_OR_AMBIGUOUS,
    NONZERO_POLICY_UPDATE,
    UNEXPECTED_POSITIVE_LR_NO_UPDATE,
    UNEXPECTED_ZERO_GRADIENT,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

Stage = Literal[
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


def _verify(value: Mapping[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _training_python(root: Path) -> Path:
    value = root / ".venv-training/Scripts/python.exe"
    if file_sha256(value) != INTERPRETER_SHA256:
        raise ValueError("authorized model/training interpreter differs")
    return value


def _runtime_command(
    root: Path,
    *,
    arm: str,
    output_dir: Path,
    raw_evidence: Path,
    partial_evidence: Path,
    summary: Path,
) -> list[str]:
    return [
        str(_training_python(root)),
        "-m",
        "foundry.phase2.l3_grpo_runtime",
        "--root",
        str(root),
        "--arm",
        arm,
        "--mode",
        "train",
        "--packet",
        str(
            root / f"results/raw/phase2_vetted_corpus/milestone14a/schedules/"
            f"{arm}_prompt_packet.json"
        ),
        "--manifest",
        str(root / TRACKED_ROOT / f"milestone14a_{arm}_schedule.json"),
        "--experiment-contract",
        str(root / TRACKED_ROOT / "milestone14a_experiment_contract.json"),
        "--warmup-update-contract",
        str(root / TRACKED_ROOT / WARMUP_CONTRACT_OUTPUT),
        "--starting-adapter",
        str(adapter_path(root, arm)),
        "--output-dir",
        str(output_dir),
        "--raw-evidence",
        str(raw_evidence),
        "--partial-evidence",
        str(partial_evidence),
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
        str(_training_python(root)),
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
    root: Path,
    *,
    suite: Path,
    subset: Path,
    run_root: Path,
) -> list[str]:
    return [
        str(_training_python(root)),
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
        _assessment_command(
            root,
            suite=suite,
            subset=subset,
            run_root=run_root,
        ),
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
        str(_training_python(root)),
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


def _require_compatibility(root: Path) -> dict[str, Any]:
    value = _read(root / TRACKED_ROOT / COMPATIBILITY_OUTPUT)
    _verify(value, "compatibility_sha256")
    if value.get("gate_passed") is not True or value.get("decision") != "pass":
        raise ValueError("counted training is not compatibility-authorized")
    return value


def _run_training(
    root: Path,
    environment: dict[str, str],
    arm: str,
) -> dict[str, Any]:
    _require_clean_synchronized_main(root)
    compatibility = _require_compatibility(root)
    raw = root / RAW_ROOT
    summary_path = raw / "training" / f"{arm}_summary.json"
    partial_path = raw / "training" / f"{arm}_partial_evidence.json"
    _run(
        _runtime_command(
            root,
            arm=arm,
            output_dir=raw / "training" / arm,
            raw_evidence=raw / "training" / f"{arm}_raw_evidence.json",
            partial_evidence=partial_path,
            summary=summary_path,
        ),
        root=root,
        environment=environment,
        stdout=raw / "training/logs" / f"{arm}.stdout.txt",
        stderr=raw / "training/logs" / f"{arm}.stderr.txt",
    )
    value = _read(summary_path)
    partial = _read(partial_path)
    _verify(value, "summary_sha256")
    _verify(partial, "partial_evidence_sha256")
    gate = cast(Mapping[str, Any], value.get("counted_update_gate"))
    counts = cast(Mapping[str, int], gate.get("classification_counts"))
    if (
        value.get("gate_passed") is not True
        or value.get("groups") != 32
        or value.get("completions") != 128
        or value.get("optimizer_steps") != 32
        or value.get("warmup_update_contract_sha256")
        != compatibility.get("warmup_update_contract_sha256")
        or value.get("partial_evidence_file_sha256") != file_sha256(partial_path)
        or gate.get("passed") is not True
        or gate.get("step_count") != 32
        or gate.get("optimizer_call_count") != 32
        or gate.get("scheduler_advance_count") != 32
        or gate.get("learning_rate_trajectory_exact") is not True
        or counts.get(UNEXPECTED_ZERO_GRADIENT) != 0
        or counts.get(UNEXPECTED_POSITIVE_LR_NO_UPDATE) != 0
        or counts.get(INVALID_OR_AMBIGUOUS) != 0
        or cast(int, counts.get(NONZERO_POLICY_UPDATE, 0)) < 1
    ):
        raise RuntimeError(f"{arm} counted training gate failed")
    return value


def _run_development(
    root: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    _require_clean_synchronized_main(root)
    raw = root / RAW_ROOT
    training = write_counted_training_result(root)
    if training.get("both_arms_passed") is not True:
        raise RuntimeError("counted training pair did not pass")
    logs = raw / "development/logs"
    for arm in ARMS:
        arm_training = _read(raw / "training" / f"{arm}_summary.json")
        _verify(arm_training, "summary_sha256")
        for checkpoint in CHECKPOINTS:
            adapter = raw / f"training/{arm}/checkpoint-{checkpoint}/adapter"
            expected = arm_training["checkpoint_evidence"][str(checkpoint)]["directory_sha256"]
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
    selection = _read(root / TRACKED_ROOT / DEVELOPMENT_OUTPUT)
    _verify(selection, "development_selection_sha256")
    checkpoint = selection.get("latest_common_passing_checkpoint")
    if not isinstance(checkpoint, int):
        raise ValueError("holdout is unauthorized without a common development checkpoint")
    suite = (
        root
        / "results/raw/phase2_vetted_corpus/milestone13c_r2"
        / "combined_holdout/candidate_suite.json"
    )
    subset = root / TRACKED_ROOT / "milestone13c_r2_combined_retention_subset.json"
    raw = root / RAW_ROOT
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
    holdout = write_holdout_decision(root)
    return {
        "training_result_sha256": _read(root / TRACKED_ROOT / TRAINING_OUTPUT)[
            "training_result_sha256"
        ],
        "holdout_decision_sha256": holdout["holdout_decision_sha256"],
        "both_arms_pass": holdout["both_arms_pass"],
    }


def _run_gsm1k(
    root: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    _require_clean_synchronized_main(root)
    holdout = _read(root / TRACKED_ROOT / HOLDOUT_OUTPUT)
    _verify(holdout, "holdout_decision_sha256")
    if holdout.get("gsm1k_authorized") is not True:
        raise ValueError("GSM1K is unauthorized")
    checkpoint = cast(int, holdout["selected_checkpoint"])
    raw = root / RAW_ROOT
    for arm in ARMS:
        adapter_sha256 = cast(str, holdout["arms"][arm]["adapter_sha256"])
        _run(
            _gsm1k_command(
                root,
                adapter=(raw / f"training/{arm}/checkpoint-{checkpoint}/adapter"),
                adapter_sha256=adapter_sha256,
                output_dir=raw / f"gsm1k/{arm}/output",
                summary=(root / TRACKED_ROOT / f"milestone14b_r2_gsm1k_{arm}.json"),
            ),
            root=root,
            environment=environment,
            stdout=raw / "gsm1k/logs" / f"{arm}.stdout.txt",
            stderr=raw / "gsm1k/logs" / f"{arm}.stderr.txt",
        )
    return write_gsm1k_analysis(root)


def run_stage(root: Path, stage: Stage) -> dict[str, object]:
    root = root.resolve()
    environment = _environment(root)
    if stage == "train-generic":
        return cast(
            dict[str, object],
            _run_training(root, environment, "generic"),
        )
    if stage == "train-targeted":
        return cast(
            dict[str, object],
            _run_training(root, environment, "targeted"),
        )
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
