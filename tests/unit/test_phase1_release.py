from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "results" / "phase1_summary.json"
FIGURE_DATA_PATH = ROOT / "results" / "phase1_figure_data.json"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_phase1_summary_self_hash_and_terminal_status() -> None:
    summary = load_json(SUMMARY_PATH)
    recorded = summary.pop("summary_sha256")
    assert recorded == canonical_sha256(summary)
    assert summary["schema_version"] == 1
    assert summary["repository"]["evidence_snapshot_commit"] == (
        "20409ba41dc99bb1e6300b53d9ad9b3db1431722"
    )
    assert summary["sealed_final"] == {"accessed": False, "evaluation_occurred": False}
    assert summary["human_review"]["status"] == "pending"
    assert summary["human_review"]["export_found_at_closeout"] is False
    assert summary["grpo"]["optimizer_steps_completed"] == 0
    assert summary["grpo"]["counted_training_started"] is False
    assert summary["grpo"]["training_completed"] is False


def test_result_table_arithmetic() -> None:
    summary = load_json(SUMMARY_PATH)
    results = summary["results"]
    total = summary["benchmark"]["development_examples"]
    assert total == 814
    for arm in ("base", "generic_control", "targeted"):
        row = results[arm]
        assert row["accuracy"] == pytest.approx(row["correct"] / total)
        assert row["accuracy_percent"] == pytest.approx(100 * row["correct"] / total)
        assert row["delta_correct_vs_base"] == row["correct"] - results["base"]["correct"]
        assert row["delta_percentage_points_vs_base"] == pytest.approx(
            100 * (row["correct"] - results["base"]["correct"]) / total
        )
    assert results["targeted"]["correct"] - results["generic_control"]["correct"] == 27
    assert results["base"]["correct"] - results["targeted"]["correct"] == 107


def test_paired_result_matches_source_analysis() -> None:
    summary = load_json(SUMMARY_PATH)
    source = load_json(ROOT / "results" / "training" / "common_scale_0_50_paired_analysis.json")
    paired = summary["paired_comparison"]
    source_paired = source["paired_bootstrap"]
    assert paired["point_estimate"] == source_paired["point_estimate"]
    assert paired["bootstrap_interval"] == source_paired["interval"]
    assert paired["bootstrap_replicates"] == source_paired["replicates"] == 10000
    assert paired["bootstrap_seed"] == source_paired["seed"] == 20260720
    assert paired["targeted_wins"] == source["paired_changes"]["targeted_wins_over_generic"]
    assert paired["generic_wins"] == source["paired_changes"]["generic_wins_over_targeted"]
    assert paired["targeted_net_wins"] == source["paired_changes"]["targeted_net_wins"]
    assert paired["point_estimate_percentage_points"] == pytest.approx(
        100 * paired["point_estimate"]
    )
    assert paired["bootstrap_interval_percentage_points"] == pytest.approx(
        [100 * value for value in paired["bootstrap_interval"]]
    )


@pytest.mark.parametrize(
    ("relative_path", "hash_field", "additional_exclusions"),
    [
        (
            "results/synthesis_smoke/matched_signal_dataset_manifest.json",
            "manifest_sha256",
            (),
        ),
        (
            "results/synthesis_smoke/matched_signal_dataset_summary.json",
            "summary_sha256",
            ("runtime_seconds",),
        ),
        ("results/training/token_matched_v2_final_parity.json", "summary_sha256", ()),
        ("results/training/common_scale_training_parity.json", "summary_sha256", ()),
        ("results/training/common_lora_scale_selection.json", "summary_sha256", ()),
        ("results/training/common_lora_scale_final_validation.json", "summary_sha256", ()),
        ("results/training/common_scale_0_50_paired_analysis.json", "analysis_sha256", ()),
        ("results/training/common_scale_0_50_signal_decision.json", "summary_sha256", ()),
        ("results/training/contrastive_scale_selection.json", "summary_sha256", ()),
        ("results/training/verifier_grpo_v1_schedule_summary.json", "summary_sha256", ()),
        ("results/training/verifier_grpo_v1_v4_replay_failure.json", "summary_sha256", ()),
        (
            "results/training/verifier_grpo_v1_v4_training_warning_audit.json",
            "warning_audit_sha256",
            (),
        ),
    ],
)
def test_referenced_evidence_self_hashes(
    relative_path: str,
    hash_field: str,
    additional_exclusions: tuple[str, ...],
) -> None:
    value = load_json(ROOT / relative_path)
    recorded = value.pop(hash_field)
    for key in additional_exclusions:
        value.pop(key)
    assert recorded == canonical_sha256(value)


def test_source_evidence_file_hashes() -> None:
    summary = load_json(SUMMARY_PATH)
    paths = {
        "base_summary": "results/development_baseline/qwen2_5_1_5b/summary.json",
        "common_scale_final_validation": (
            "results/training/common_lora_scale_final_validation.json"
        ),
        "contrastive_construction": "results/training/contrastive_task_vector_construction.json",
        "contrastive_selection": "results/training/contrastive_scale_selection.json",
        "failure_taxonomy": (
            "results/development_baseline/qwen2_5_1_5b/complete_failure_taxonomy.json"
        ),
        "generic_development": "results/training/common_scale_0_50_generic_development.json",
        "grpo_generic_schedule": "results/training/verifier_grpo_v1_generic_schedule.json",
        "grpo_replay_failure": "results/training/verifier_grpo_v1_v4_replay_failure.json",
        "grpo_schedule_summary": "results/training/verifier_grpo_v1_schedule_summary.json",
        "grpo_targeted_schedule": "results/training/verifier_grpo_v1_targeted_schedule.json",
        "grpo_warning_audit": ("results/training/verifier_grpo_v1_v4_training_warning_audit.json"),
        "language_audit": "results/synthesis_smoke/matched_signal_language_audit_summary.json",
        "matched_dataset_manifest": "results/synthesis_smoke/matched_signal_dataset_manifest.json",
        "matched_dataset_summary": "results/synthesis_smoke/matched_signal_dataset_summary.json",
        "paired_analysis": "results/training/common_scale_0_50_paired_analysis.json",
        "retention_safe_ladder": "results/training/retention_safe_ladder_results.json",
        "signal_decision": "results/training/common_scale_0_50_signal_decision.json",
        "targeted_development": "results/training/common_scale_0_50_targeted_development.json",
    }
    expected = summary["source_evidence_file_sha256"]
    assert set(expected) == set(paths)
    assert {key: file_sha256(ROOT / path) for key, path in paths.items()} == expected


def test_dataset_retention_contrastive_and_grpo_aggregates() -> None:
    summary = load_json(SUMMARY_PATH)
    datasets = summary["matched_datasets"]
    dataset_source = load_json(
        ROOT / "results" / "synthesis_smoke" / "matched_signal_dataset_summary.json"
    )
    assert datasets["generic_control"]["examples"] == 500
    assert datasets["targeted"]["examples"] == 500
    assert (
        datasets["generic_control"]["dataset_sha256"]
        == dataset_source["dataset_hashes"]["generic_control"]
    )
    assert datasets["targeted"]["dataset_sha256"] == dataset_source["dataset_hashes"]["targeted"]
    assert (
        datasets["generic_control"]["training_sha256"]
        == dataset_source["split_hashes"]["generic_control"]["training"]
    )
    assert (
        datasets["targeted"]["training_sha256"]
        == dataset_source["split_hashes"]["targeted"]["training"]
    )
    assert datasets["global_exact_duplicates"] == 0
    assert datasets["global_latent_program_duplicates"] == 0
    assert datasets["verifier_disagreements"] == 0
    assert datasets["unresolved_contamination_cases"] == 0
    assert datasets["deterministic_reconstruction_match"] is True

    retention = summary["retention"]
    retention_source = load_json(
        ROOT / "results" / "training" / "common_lora_scale_final_validation.json"
    )
    assert retention["common_scale"] == 0.5
    assert retention["retention_passed"] is True
    assert retention["common_scale"] == retention_source["selected_common_scale"]
    assert retention["validation_summary_sha256"] == retention_source["summary_sha256"]
    assert retention["final_holdout"]["generic_control"]["preserved"] == 314
    assert retention["final_holdout"]["targeted"]["preserved"] == 315

    contrastive = summary["contrastive_adapter"]
    contrastive_source = load_json(
        ROOT / "results" / "training" / "contrastive_scale_selection.json"
    )
    assert contrastive["dense_equivalence_gate_passed"] is True
    assert contrastive["retention_selection_passed"] is False
    assert contrastive["gsm1k_authorized"] is False
    assert not any(contrastive["retention_matrix_passed_by_scale"].values())
    assert contrastive["adapter_sha256"] == contrastive_source["adapter_sha256"]
    assert contrastive["selection_summary_sha256"] == contrastive_source["summary_sha256"]

    grpo = summary["grpo"]
    grpo_source = load_json(
        ROOT / "results" / "training" / "verifier_grpo_v1_schedule_summary.json"
    )
    assert grpo["same_process_replay_passed"] is True
    assert grpo["fresh_process_replay_passed"] is True
    assert grpo["same_process_common_packet_sha256"] == (grpo["fresh_process_common_packet_sha256"])
    assert grpo["completed_generation_replays"] == 6
    assert grpo["optimizer_steps_completed"] == 0
    assert grpo["prompt_tokens_per_arm"] == grpo_source["generic_prompt_tokens"]
    assert grpo_source["generic_prompt_tokens"] == grpo_source["targeted_prompt_tokens"]
    assert grpo["schedule_summary_sha256"] == grpo_source["summary_sha256"]


def test_figure_source_consistency_and_deterministic_assets() -> None:
    summary = load_json(SUMMARY_PATH)
    data = load_json(FIGURE_DATA_PATH)
    recorded = data.pop("data_sha256")
    assert recorded == canonical_sha256(data)
    assert data["sealed_final_accessed"] is False
    assert sum(item["count"] for item in data["taxonomy"]) == data["taxonomy_total"] == 293
    assert [item["correct"] for item in data["accuracy"]] == [521, 387, 414]
    assert [item["accuracy_percent"] for item in data["accuracy"]] == pytest.approx(
        [
            summary["results"]["base"]["accuracy_percent"],
            summary["results"]["generic_control"]["accuracy_percent"],
            summary["results"]["targeted"]["accuracy_percent"],
        ]
    )
    assert data["paired_difference"]["estimate_percentage_points"] == pytest.approx(
        summary["paired_comparison"]["point_estimate_percentage_points"]
    )
    assert data["paired_difference"]["interval_percentage_points"] == pytest.approx(
        summary["paired_comparison"]["bootstrap_interval_percentage_points"]
    )

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "20260720"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_phase1_figures.py"), "--check"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    paths = sorted((ROOT / "docs" / "assets" / "phase1").glob("*.svg"))
    assert len(paths) == 5
    for path in paths:
        assert f'data-source-sha256="{recorded}"' in path.read_text(encoding="utf-8")


def test_timeline_and_consistency_table_align() -> None:
    timeline = read_csv(ROOT / "results" / "phase1_experiment_timeline.csv")
    consistency = read_csv(ROOT / "results" / "phase1_evidence_consistency.csv")
    required_timeline_columns = {
        "milestone",
        "date",
        "commit",
        "experiment",
        "gate",
        "outcome",
        "key metric",
        "stopped before",
        "next decision",
    }
    required_consistency_columns = {
        "milestone",
        "commit",
        "gate",
        "result",
        "next decision",
        "model generation occurred",
        "optimizer steps occurred",
        "GSM1K evaluation occurred",
        "sealed-final content accessed",
    }
    assert set(timeline[0]) == required_timeline_columns
    assert set(consistency[0]) == required_consistency_columns
    assert len(timeline) == len(consistency) >= 45
    assert [(row["milestone"], row["commit"]) for row in timeline] == [
        (row["milestone"], row["commit"]) for row in consistency
    ]
    for row in timeline:
        assert len(row["commit"]) == 40
        assert row["date"].startswith("2026-07-")
    for row in consistency:
        assert row["sealed-final content accessed"] == "false"
        for field in (
            "model generation occurred",
            "optimizer steps occurred",
            "GSM1K evaluation occurred",
        ):
            assert row[field] in {"true", "false"}
    final = consistency[-1]
    assert final["milestone"] == "10J"
    assert final["commit"] == "20409ba41dc99bb1e6300b53d9ad9b3db1431722"
    assert final["optimizer steps occurred"] == "false"


def test_readme_result_claims_match_summary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "521/814",
        "64.0049%",
        "387/814",
        "47.5430%",
        "414/814",
        "50.8600%",
        "−13.1450 points",
        "+3.3170 percentage points",
        "[+1.3514, +5.2826] points",
        "107 questions below the untouched base",
        "No sealed-final evaluation occurred",
        "no backward",
        "optimizer step was certified",
    )
    for claim in required:
        assert claim in readme
    lowered = readme.casefold()
    for unsupported in (
        "autonomously improves",
        "successful grpo",
        "beats the base",
        "state of the art",
        "production ready",
        "fully autonomous",
    ):
        assert unsupported not in lowered


def test_final_report_has_required_paper_structure_and_caveats() -> None:
    report = (ROOT / "docs" / "PHASE1_FINAL_REPORT.md").read_text(encoding="utf-8")
    sections = [line for line in report.splitlines() if line.startswith("## ")]
    assert len(sections) == 20
    assert sections[0] == "## 1. Abstract"
    assert sections[-1] == "## 20. Conclusion"
    for required in (
        "Targeted exceeded generic",
        "Neither adapter outperformed the untouched base",
        "GRPO training was not completed",
        "No sealed-final evaluation occurred",
        "Human status is therefore `pending`",
    ):
        assert required in report


def test_no_forbidden_artifact_path_is_tracked() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    tracked = [Path(value.decode("utf-8")) for value in completed.stdout.split(b"\0") if value]
    forbidden_roots = {
        ".venv",
        ".venv-training",
        ".venv-realization",
        "checkpoints",
        "data",
        "datasets",
        "outputs",
        "wandb",
    }
    forbidden_suffixes = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
    for path in tracked:
        parts = path.as_posix().split("/")
        assert parts[0] not in forbidden_roots
        assert path.as_posix() != "results/raw"
        assert not path.as_posix().startswith("results/raw/")
        assert path.suffix.casefold() not in forbidden_suffixes


def test_no_tracked_result_records_sealed_access() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "results"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for relative in completed.stdout.splitlines():
        path = ROOT / relative
        if path.suffix == ".json":
            assert (
                '"sealed_final_accessed": true' not in path.read_text(encoding="utf-8").casefold()
            )
