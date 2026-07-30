"""Controller-owned retention and benchmark orchestration using existing evaluators."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

from foundry.cycle.contract import (
    CycleConfig,
    CycleContractError,
    validate_file_identity,
    validate_process_environment,
    verified_payload,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _run(
    *,
    config: CycleConfig,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, int]:
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("evaluation command log already exists")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("x", encoding="utf-8") as stdout,
        stderr_path.open("x", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=config.source_root / "src",
            env=dict(__import__("os").environ),
            shell=False,
            text=True,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        psutil: Any = importlib.import_module("psutil")
        monitored = psutil.Process(process.pid)
        peak_rss = 0
        while process.poll() is None:
            try:
                memory = monitored.memory_info()
                peak_rss = max(
                    peak_rss,
                    int(getattr(memory, "peak_wset", memory.rss)),
                )
            except psutil.Error:
                pass
            time.sleep(0.05)
        returncode = int(process.returncode)
    if returncode != 0:
        raise RuntimeError(f"evaluation command failed with {returncode}: {stderr_path}")
    return {"peak_process_rss_bytes": peak_rss}


def _retention_pair(
    *,
    config: CycleConfig,
    adapter_path: Path,
    suite_path: Path,
    subset_path: Path,
    output_directory: Path,
    holdout: bool,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError("retention output must be fresh")
    output_directory.mkdir(parents=True, exist_ok=False)
    raw_path = output_directory / "raw.json"
    summary_path = output_directory / "summary.json"
    assessment_path = output_directory / "assessment.json"
    module = "foundry.cycle.holdout" if holdout else "foundry.training.retention"
    evaluate_command = [sys.executable, "-m", module]
    if holdout:
        evaluate_command.extend(["--config", str(config.path), "evaluate"])
    evaluate_command.extend(
        [
            "--suite",
            str(suite_path),
            "--model-path",
            str(config.resolve_artifact(str(config.section("model")["snapshot_relative_path"]))),
            "--adapter",
            str(adapter_path),
            "--raw-path",
            str(raw_path),
            "--output-path",
            str(summary_path),
            "--subset-manifest",
            str(subset_path),
        ]
    )
    _run(
        config=config,
        command=evaluate_command,
        stdout_path=output_directory / "evaluate.stdout.txt",
        stderr_path=output_directory / "evaluate.stderr.txt",
    )
    assess_module = (
        [
            "-m",
            "foundry.cycle.holdout",
            "--config",
            str(config.path),
            "assess",
        ]
        if holdout
        else ["-m", "foundry.training.base_conditioned_retention", "assess"]
    )
    _run(
        config=config,
        command=[
            sys.executable,
            *assess_module,
            "--suite",
            str(suite_path),
            "--subset",
            str(subset_path),
            "--summary",
            str(summary_path),
            "--raw",
            str(raw_path),
            "--output",
            str(assessment_path),
        ],
        stdout_path=output_directory / "assess.stdout.txt",
        stderr_path=output_directory / "assess.stderr.txt",
    )
    assessment = _read(assessment_path)
    if assessment.get("backend_failures") != 0:
        raise RuntimeError("retention backend failure violates Cycle 1")
    return assessment


def run_development_retention(
    *,
    config: CycleConfig,
    training_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate all 8/16/32 checkpoints and select the latest full pass."""

    validate_process_environment(config=config)
    if output_directory.exists():
        raise FileExistsError("development retention output must be fresh")
    output_directory.mkdir(parents=True, exist_ok=False)
    training_summary = verified_payload(
        training_directory / "summary.json",
        "training_sha256",
    )
    checkpoint_contracts = cast(
        dict[str, dict[str, Any]],
        training_summary["checkpoints"],
    )
    if set(checkpoint_contracts) != {"8", "16", "32"}:
        raise CycleContractError("development checkpoint identities differ")
    retention = config.section("retention")
    development = cast(dict[str, dict[str, Any]], retention["development"])
    trajectory: dict[str, Any] = {}
    passing: list[int] = []
    started = time.perf_counter()
    for checkpoint in (8, 16, 32):
        adapter = training_directory / f"checkpoint-{checkpoint}" / "adapter"
        adapter_hash = directory_sha256(adapter)
        if adapter_hash != checkpoint_contracts[str(checkpoint)]["adapter_sha256"]:
            raise CycleContractError("development adapter artifact identity differs")
        suites: dict[str, Any] = {}
        for name in ("adjudication", "anchor"):
            contract = development[name]
            suite = validate_file_identity(
                config,
                str(contract["suite_relative_path"]),
                str(contract["suite_file_sha256"]),
            )
            subset = validate_file_identity(
                config,
                str(contract["subset_relative_path"]),
                str(contract["subset_file_sha256"]),
            )
            assessment = _retention_pair(
                config=config,
                adapter_path=adapter,
                suite_path=suite,
                subset_path=subset,
                output_directory=output_directory / f"checkpoint-{checkpoint}" / name,
                holdout=False,
            )
            suites[name] = {
                "gate_passed": assessment["gate_passed"],
                "preserved": assessment["preserved"],
                "total": assessment["total"],
                "overall_preservation": assessment["overall_preservation"],
                "section_preservation": assessment["section_preservation"],
                "question_generation": assessment["question_generation"],
                "backend_failures": assessment["backend_failures"],
                "maximum_instruction_family_adapter_only_failures": assessment[
                    "maximum_instruction_family_adapter_only_failures"
                ],
                "assessment_sha256": assessment["summary_sha256"],
                "assessment_file_sha256": file_sha256(
                    output_directory / f"checkpoint-{checkpoint}" / name / "assessment.json"
                ),
            }
        passed = all(bool(item["gate_passed"]) for item in suites.values())
        if passed:
            passing.append(checkpoint)
        trajectory[str(checkpoint)] = {
            "adapter_sha256": adapter_hash,
            "suites": suites,
            "passed": passed,
        }
    selected = max(passing) if passing else None
    result: dict[str, Any] = {
        "schema_version": 1,
        "development_retention_id": "foundry-cycle1-development-retention-v1",
        "trajectory": trajectory,
        "passing_checkpoints": passing,
        "selected_checkpoint": selected,
        "passed": selected is not None,
        "runtime_seconds": time.perf_counter() - started,
    }
    result["development_retention_sha256"] = canonical_sha256(result)
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run_holdout_retention(
    *,
    config: CycleConfig,
    adapter_path: Path,
    expected_adapter_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate the selected candidate exactly once on holdout v2."""

    validate_process_environment(config=config)
    if directory_sha256(adapter_path) != expected_adapter_sha256:
        raise CycleContractError("holdout candidate adapter identity differs")
    holdout = cast(dict[str, Any], config.section("retention")["holdout_v2"])
    suite = validate_file_identity(
        config,
        str(holdout["suite_relative_path"]),
        str(holdout["suite_file_sha256"]),
    )
    subset = validate_file_identity(
        config,
        str(holdout["subset_relative_path"]),
        str(holdout["subset_file_sha256"]),
    )
    started = time.perf_counter()
    assessment = _retention_pair(
        config=config,
        adapter_path=adapter_path,
        suite_path=suite,
        subset_path=subset,
        output_directory=output_directory / "evaluation",
        holdout=True,
    )
    if (
        assessment.get("subset_sha256") != holdout["subset_sha256"]
        or assessment.get("total") != holdout["total"]
    ):
        raise CycleContractError("holdout-v2 identity differs after evaluation")
    result: dict[str, Any] = {
        "schema_version": 1,
        "holdout_retention_id": "foundry-cycle1-holdout-v2-v1",
        "adapter_sha256": directory_sha256(adapter_path),
        "assessment": {
            "preserved": assessment["preserved"],
            "total": assessment["total"],
            "overall_preservation": assessment["overall_preservation"],
            "section_preservation": assessment["section_preservation"],
            "question_generation": assessment["question_generation"],
            "backend_failures": assessment["backend_failures"],
            "maximum_instruction_family_adapter_only_failures": assessment[
                "maximum_instruction_family_adapter_only_failures"
            ],
            "gate_passed": assessment["gate_passed"],
            "subset_sha256": assessment["subset_sha256"],
            "suite_sha256": assessment["suite_sha256"],
            "assessment_sha256": assessment["summary_sha256"],
        },
        "assessment_file_sha256": file_sha256(output_directory / "evaluation" / "assessment.json"),
        "starting_targeted_preserved": holdout["starting_preserved"],
        "passed": assessment["gate_passed"] is True,
        "runtime_seconds": time.perf_counter() - started,
    }
    result["holdout_retention_sha256"] = canonical_sha256(result)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _category_effects(
    *,
    config: CycleConfig,
    candidate_predictions_path: Path,
) -> dict[str, Any]:
    benchmark = config.section("benchmark")
    base_path = validate_file_identity(
        config,
        str(benchmark["base_predictions_relative_path"]),
        str(benchmark["base_predictions_sha256"]),
    )
    taxonomy_path = validate_file_identity(
        config,
        str(benchmark["taxonomy_relative_path"]),
        str(benchmark["taxonomy_file_sha256"]),
    )
    base = {str(item["stable_id"]): bool(item["correct"]) for item in _jsonl(base_path)}
    candidate = {
        str(item["stable_id"]): bool(item["correct"]) for item in _jsonl(candidate_predictions_path)
    }
    if set(base) != set(candidate) or len(base) != 814:
        raise ValueError("candidate and frozen base prediction IDs differ")
    taxonomy = _read(taxonomy_path)
    prefixes = cast(
        dict[str, list[str]],
        taxonomy["stable_identifier_prefixes_by_primary_category"],
    )
    categories: dict[str, Any] = {}
    ids_by_category: dict[str, set[str]] = {}
    for name, values in prefixes.items():
        matched = {
            stable_id
            for stable_id in base
            if any(stable_id.startswith(prefix) for prefix in values)
        }
        if len(matched) != len(values):
            raise ValueError(f"taxonomy prefix reconstruction differs: {name}")
        ids_by_category[name] = matched
        categories[name] = {
            "items": len(matched),
            "base_correct": sum(base[item] for item in matched),
            "candidate_correct": sum(candidate[item] for item in matched),
        }
        categories[name]["delta_correct"] = (
            categories[name]["candidate_correct"] - categories[name]["base_correct"]
        )
    selected_categories = set(cast(list[str], taxonomy["selected_reasoning_categories"]))
    targeted_ids = set().union(*(ids_by_category[name] for name in selected_categories))
    untargeted_ids = set(base) - targeted_ids
    base_rate = sum(base[item] for item in untargeted_ids) / len(untargeted_ids)
    candidate_rate = sum(candidate[item] for item in untargeted_ids) / len(untargeted_ids)
    decline_points = max(0.0, (base_rate - candidate_rate) * 100)
    result: dict[str, Any] = {
        "schema_version": 1,
        "category_effect_id": "foundry-cycle1-published-taxonomy-effects-v1",
        "selected_reasoning_categories": sorted(selected_categories),
        "categories": categories,
        "untargeted_population_items": len(untargeted_ids),
        "untargeted_base_accuracy": base_rate,
        "untargeted_candidate_accuracy": candidate_rate,
        "untargeted_decline_percentage_points": decline_points,
        "maximum_untargeted_decline_percentage_points": float(
            benchmark["maximum_untargeted_aggregate_decline_percentage_points"]
        ),
        "untargeted_gate_passed": decline_points
        <= float(benchmark["maximum_untargeted_aggregate_decline_percentage_points"]),
    }
    result["category_effect_sha256"] = canonical_sha256(result)
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def run_benchmark(
    *,
    config: CycleConfig,
    adapter_path: Path,
    expected_adapter_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate the candidate exactly once on the frozen 814-example GSM1K development set."""

    validate_process_environment(config=config)
    if output_directory.exists():
        raise FileExistsError("GSM1K candidate output must be fresh")
    benchmark = config.section("benchmark")
    adapter_sha256 = directory_sha256(adapter_path)
    if adapter_sha256 != expected_adapter_sha256:
        raise CycleContractError("GSM1K candidate adapter identity differs")
    started = time.perf_counter()
    raw_output = output_directory / "output"
    summary_path = output_directory / "evaluator_summary.json"
    process_metrics = _run(
        config=config,
        command=[
            sys.executable,
            "-m",
            "foundry.training.adapter_evaluation",
            "--base-config",
            str(config.artifact_root / "configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml"),
            "--config",
            str(config.resolve_artifact(str(benchmark["evaluator_config_relative_path"]))),
            "--development-manifest",
            str(config.resolve_artifact(str(benchmark["development_manifest_relative_path"]))),
            "--source-pool-manifest",
            str(config.artifact_root / "configs/eval/manifests/gsm1k_development_baseline.json"),
            "--source-baseline-manifest",
            str(
                config.artifact_root / "configs/eval/manifests/gsm1k_development_baseline_844.json"
            ),
            "--baseline-manifest",
            str(
                config.artifact_root / "configs/eval/manifests/gsm1k_development_baseline_814.json"
            ),
            "--model-path",
            str(config.resolve_artifact(str(config.section("model")["snapshot_relative_path"]))),
            "--adapter",
            str(adapter_path),
            "--adapter-sha256",
            adapter_sha256,
            "--adapter-scale",
            "1.0",
            "--output-dir",
            str(raw_output),
            "--tracked-summary",
            str(summary_path),
        ],
        stdout_path=output_directory / "evaluate.stdout.txt",
        stderr_path=output_directory / "evaluate.stderr.txt",
    )
    summary = _read(summary_path)
    if (
        summary.get("processed_examples") != benchmark["total"]
        or summary.get("generation_failures") != 0
        or summary.get("manifest_sha256") != benchmark["frozen_manifest_sha256"]
    ):
        raise RuntimeError("GSM1K evaluator completeness or identity gate failed")
    category_effects = _category_effects(
        config=config,
        candidate_predictions_path=raw_output / "raw" / "predictions.jsonl",
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "foundry-cycle1-gsm1k-development-v1",
        "adapter_sha256": adapter_sha256,
        "correct": summary["correct_examples"],
        "total": summary["processed_examples"],
        "accuracy": summary["accuracy"],
        "extractable": summary["extractable_examples"],
        "extractability": summary["extractable_answer_rate"],
        "exact_format_compliant": summary["exact_format_compliant_examples"],
        "exact_format_compliance_rate": summary["exact_format_compliance_rate"],
        "extractable_but_wrong": summary["extractable_incorrect_examples"],
        "unextractable": summary["unextractable_examples"],
        "truncated": summary["truncated_examples"],
        "backend_failures": summary["generation_failures"],
        "extraction_failure_categories": summary["extraction_failure_categories"],
        "input_tokens": summary["total_input_tokens"],
        "output_tokens": summary["total_output_tokens"],
        "peak_vram_allocated_bytes": summary["backend_metrics"]["peak_vram_allocated_bytes"],
        "peak_vram_reserved_bytes": summary["backend_metrics"]["peak_vram_reserved_bytes"],
        "peak_process_rss_bytes": process_metrics["peak_process_rss_bytes"],
        "evaluation_runtime_seconds": summary["total_runtime_seconds"],
        "controller_runtime_seconds": time.perf_counter() - started,
        "category_effects": category_effects,
        "evaluator_summary_file_sha256": file_sha256(summary_path),
        "candidate_predictions_file_sha256": file_sha256(raw_output / "raw" / "predictions.jsonl"),
    }
    result["benchmark_sha256"] = canonical_sha256(result)
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
