"""Freeze and analyze signal-qualified L3 GRPO replay/projection evidence."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_contract import DETERMINISTIC_ENVIRONMENT, STARTING_ADAPTER_SHA256
from foundry.phase2.l3_grpo_signal_audit import ARMS, build_signal_summary
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

SOURCE_PARENT_COMMIT = "bdffaf81180098736bff116074ede67f8fa43d89"
IMPLEMENTATION_OUTPUT = "milestone14b_r1_qualification_implementation.json"
CONTRACT_OUTPUT = "milestone14b_r1_qualification_contract.json"
SELECTION_OUTPUT = "milestone14b_r1_selection_and_gradient_decision.json"
SIGNAL_SUMMARY_OUTPUT = "milestone14b_r1_signal_summary.json"

IMPLEMENTATION_FILES = (
    "src/foundry/phase2/l3_grpo_signal_qualification.py",
    "src/foundry/phase2/l3_grpo_signal_qualification_runtime.py",
    "src/foundry/phase2/l3_grpo_signal_qualification_campaign.py",
    "src/foundry/phase2/l3_grpo_signal_compatibility_runtime.py",
    "src/foundry/phase2/l3_grpo_signal_compatibility_campaign.py",
    "tests/unit/phase2/test_l3_grpo_signal_qualification.py",
    "tests/unit/phase2/test_l3_grpo_signal_qualification_runtime.py",
    "tests/unit/phase2/test_l3_grpo_signal_qualification_campaign.py",
    "tests/unit/phase2/test_l3_grpo_signal_compatibility_runtime.py",
    "tests/unit/phase2/test_l3_grpo_signal_compatibility_campaign.py",
)
FROZEN_DEPENDENCY_FILES = (
    "src/foundry/phase2/l3_grpo_campaign.py",
    "src/foundry/phase2/l3_grpo_signal_audit.py",
    "src/foundry/phase2/l3_grpo_signal_runtime.py",
    "src/foundry/phase2/l3_grpo_signal_continuity.py",
    "src/foundry/phase2/l3_grpo_advantage_equivalence.py",
    "src/foundry/phase2/l3_grpo_runtime.py",
    "src/foundry/phase2/l3_grpo_zero_gradient.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_diagnostic.py",
    "src/foundry/phase2/l3_grpo_contract.py",
    "src/foundry/phase2/l3_grpo_reference.py",
    "src/foundry/phase2/l3_grpo_schedule.py",
    "src/foundry/phase2/windows_environment.py",
    "src/foundry/training/grpo_compatibility.py",
    "src/foundry/training/grpo_replay_evidence.py",
    "src/foundry/training/grpo_runtime.py",
    "src/foundry/training/grpo_trainer.py",
)


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


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.rstrip()


def _file_rows(root: Path, paths: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"qualification source is missing: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def _require_freeze_boundary(root: Path) -> None:
    if root.resolve() != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("Milestone 14B-R1 qualification is attached to the wrong repository")
    dirty = _git(root, "status", "--porcelain")
    allowed_prefixes = (
        "src/foundry/phase2/l3_grpo_signal_qualification",
        "src/foundry/phase2/l3_grpo_signal_compatibility",
        "tests/unit/phase2/test_l3_grpo_signal_qualification",
        "tests/unit/phase2/test_l3_grpo_signal_compatibility",
        "results/phase2_vetted_corpus/milestone14b_r1_qualification",
        "docs/DEVLOG.md",
        "docs/VERIFIER_GRPO_RESULT.md",
    )
    dirty_paths = [line[3:].replace("\\", "/") for line in dirty.splitlines() if len(line) >= 4]
    if (
        _git(root, "branch", "--show-current") != "main"
        or _git(root, "rev-parse", "HEAD") != SOURCE_PARENT_COMMIT
        or _git(root, "rev-parse", "origin/main") != SOURCE_PARENT_COMMIT
        or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
        != ["0", "0"]
        or any(not path.startswith(allowed_prefixes) for path in dirty_paths)
    ):
        raise RuntimeError("Milestone 14B-R1 qualification source-freeze boundary differs")
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone14b_r1"
    if (raw_root / "qualification").exists() or (raw_root / "compatibility").exists():
        raise RuntimeError("qualification model evidence exists before its source freeze")


def _audit_evidence(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, object]]:
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone14b_r1/signal_audit"
    raw_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        raw = _read(raw_root / arm / "raw_evidence.json")
        summary = _read(raw_root / arm / "summary.json")
        _verify(raw, "raw_audit_sha256")
        _verify(summary, "summary_sha256")
        if (
            raw.get("source_commit") != SOURCE_PARENT_COMMIT
            or summary.get("source_commit") != SOURCE_PARENT_COMMIT
            or summary.get("groups") != 32
            or summary.get("completions") != 128
            or summary.get("advantage_equivalence_passed") is not True
            or summary.get("prior_partial_continuity_passed") is not True
            or summary.get("backend_failures") != 0
        ):
            raise ValueError(f"{arm} corrected audit evidence differs")
        raw_by_arm[arm] = raw
    density = build_signal_summary(raw_by_arm["generic"], raw_by_arm["targeted"])
    if (
        density.get("quantitative_viability_passed") is not True
        or density.get("decision") != "deterministic_replay_required"
    ):
        raise ValueError("corrected audit did not reach deterministic replay")
    return raw_by_arm, density


def _selected_groups(
    raw_by_arm: Mapping[str, Mapping[str, Any]],
    density: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    expected_positions = {"generic": 5, "targeted": 1}
    for arm in ARMS:
        arm_summary = cast(Mapping[str, Any], cast(Mapping[str, Any], density["arms"])[arm])
        candidates = cast(list[dict[str, Any]], arm_summary["usable_informative_candidates"])
        if not candidates:
            raise ValueError(f"{arm} has no usable informative candidate")
        candidate = candidates[0]
        if candidate.get("schedule_position") != expected_positions[arm]:
            raise ValueError(f"{arm} earliest informative position differs")
        groups = cast(list[dict[str, Any]], raw_by_arm[arm]["groups"])
        task = groups[cast(int, candidate["schedule_position"]) - 1]
        replay = next(group for group in groups if group.get("source_kind") == "base_replay")
        for group in (task, replay):
            equivalence = cast(Mapping[str, Any], group["advantage_equivalence"])
            if (
                equivalence.get("passed") is not True
                or group.get("maximum_cpu_cuda_advantage_difference") != 0.0
                or group.get("canonical_cuda_advantages") != group.get("cpu_diagnostic_advantages")
                or group.get("backend_failure_count") != 0
            ):
                raise ValueError(f"{arm} selected smoke group lacks exact canonical projection")
        selected[arm] = {
            "selection_rule": (
                "earliest_schedule_position_with_nonzero_reward_variance_then_lowest_prompt_hash"
            ),
            "task_schedule_position": task["schedule_position"],
            "task_group_id": task["group_id"],
            "task_family": task["task_family"],
            "task_prompt_sha256": task["prompt_sha256"],
            "task_group_record_sha256": task["group_record_sha256"],
            "task_reward_variance": task["reward_variance"],
            "task_nonzero_advantage_count": task["nonzero_advantage_count"],
            "task_valid_completion_token_count": task["valid_completion_token_count"],
            "replay_schedule_position": replay["schedule_position"],
            "replay_group_id": replay["group_id"],
            "replay_prompt_sha256": replay["prompt_sha256"],
            "replay_group_record_sha256": replay["group_record_sha256"],
            "fresh_process_replay_status": "pending",
            "gradient_projection_status": "pending",
        }
    return selected


def freeze_qualification(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Freeze the post-audit candidate selection and all later runtime source."""

    root = root.resolve()
    _require_freeze_boundary(root)
    raw_by_arm, density = _audit_evidence(root)
    selected = _selected_groups(raw_by_arm, density)
    files = _file_rows(root, (*IMPLEMENTATION_FILES, *FROZEN_DEPENDENCY_FILES))
    implementation: dict[str, object] = {
        "schema_version": 1,
        "implementation_id": "foundry-l3-grpo-signal-qualification-implementation-v1",
        "source_parent_commit": SOURCE_PARENT_COMMIT,
        "implementation_files": list(IMPLEMENTATION_FILES),
        "frozen_dependency_files": list(FROZEN_DEPENDENCY_FILES),
        "files": files,
        "model_generation_calls_during_freeze": 0,
        "optimizer_steps_during_freeze": 0,
        "sealed_content_use": 0,
    }
    implementation["implementation_sha256"] = canonical_sha256(implementation)
    correction = _read(
        root / "results/phase2_vetted_corpus/milestone14b_r1_signal_audit_contract.json"
    )
    advantage = _read(
        root / "results/phase2_vetted_corpus/milestone14b_r1_advantage_equivalence_contract.json"
    )
    family = _read(
        root / "results/phase2_vetted_corpus/milestone14b_r1_family_aggregation_contract.json"
    )
    prior = _read(
        root / "results/phase2_vetted_corpus/milestone14b_r1_prior_diagnostic_manifest.json"
    )
    for value, key in (
        (correction, "signal_audit_contract_sha256"),
        (advantage, "advantage_equivalence_contract_sha256"),
        (family, "family_aggregation_contract_sha256"),
        (prior, "prior_diagnostic_manifest_sha256"),
    ):
        _verify(value, key)
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone14b_r1/signal_audit"
    contract: dict[str, object] = {
        "schema_version": 1,
        "contract_id": "foundry-l3-grpo-signal-qualification-v1",
        "source_parent_commit": SOURCE_PARENT_COMMIT,
        "implementation_sha256": implementation["implementation_sha256"],
        "corrected_signal_audit_contract_sha256": correction["signal_audit_contract_sha256"],
        "advantage_equivalence_contract_sha256": advantage["advantage_equivalence_contract_sha256"],
        "family_aggregation_contract_sha256": family["family_aggregation_contract_sha256"],
        "prior_diagnostic_manifest_sha256": prior["prior_diagnostic_manifest_sha256"],
        "audit_evidence": {
            arm: {
                "raw_audit_sha256": raw_by_arm[arm]["raw_audit_sha256"],
                "raw_evidence_file_sha256": file_sha256(raw_root / arm / "raw_evidence.json"),
                "runtime_summary_sha256": _read(raw_root / arm / "summary.json")["summary_sha256"],
                "runtime_summary_file_sha256": file_sha256(raw_root / arm / "summary.json"),
            }
            for arm in ARMS
        },
        "quantitative_signal_density_sha256": density["signal_summary_sha256"],
        "quantitative_viability_passed": True,
        "selected_candidates": selected,
        "projection_runs_per_arm": 2,
        "projection_prefix_policy": ("replay_all_frozen_schedule_positions_through_selected_task"),
        "gradient_requirements": {
            "nonzero_reward_variance": True,
            "nonzero_canonical_advantage": True,
            "nonempty_completion_masks": True,
            "connected_policy_objective": True,
            "finite_nonzero_policy_lora_gradient": True,
            "reference_gradient_count": 0,
            "base_gradient_count": 0,
            "duplicate_exact_evidence": True,
            "optimizer_construction": False,
            "optimizer_steps": 0,
        },
        "compatibility_runs_per_arm": 2,
        "compatibility_groups_per_run": 2,
        "compatibility_completions_per_run": 8,
        "compatibility_optimizer_steps_per_run": 2,
        "process_environment": DETERMINISTIC_ENVIRONMENT,
        "starting_adapters": STARTING_ADAPTER_SHA256,
        "counted_training_authorized": False,
        "retention_authorized": False,
        "holdout_v2_authorized": False,
        "gsm1k_authorized": False,
        "sealed_content_use": 0,
    }
    contract["qualification_contract_sha256"] = canonical_sha256(contract)
    return implementation, contract


def verify_qualification_contract(
    root: Path,
    value: Mapping[str, Any],
    *,
    require_clean_synchronized: bool,
) -> None:
    _verify(value, "qualification_contract_sha256")
    if (
        value.get("contract_id") != "foundry-l3-grpo-signal-qualification-v1"
        or value.get("source_parent_commit") != SOURCE_PARENT_COMMIT
        or value.get("quantitative_viability_passed") is not True
        or value.get("projection_runs_per_arm") != 2
        or value.get("compatibility_runs_per_arm") != 2
        or value.get("process_environment") != DETERMINISTIC_ENVIRONMENT
        or value.get("starting_adapters") != STARTING_ADAPTER_SHA256
        or value.get("counted_training_authorized") is not False
        or value.get("retention_authorized") is not False
        or value.get("holdout_v2_authorized") is not False
        or value.get("gsm1k_authorized") is not False
        or set(cast(Mapping[str, Any], value.get("selected_candidates"))) != set(ARMS)
    ):
        raise ValueError("signal-qualification contract differs")
    implementation = _read(root / "results/phase2_vetted_corpus" / IMPLEMENTATION_OUTPUT)
    _verify(implementation, "implementation_sha256")
    if implementation.get("implementation_sha256") != value.get("implementation_sha256"):
        raise ValueError("signal-qualification implementation differs")
    for row_value in cast(list[object], implementation.get("files")):
        if not isinstance(row_value, dict):
            raise ValueError("qualification implementation row differs")
        row = cast(dict[str, object], row_value)
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ValueError("qualification source path differs")
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or file_sha256(path) != row.get("sha256")
            or path.stat().st_size != row.get("bytes")
        ):
            raise ValueError("qualification implementation source differs")
    if require_clean_synchronized:
        head = _git(root, "rev-parse", "HEAD")
        if (
            _git(root, "branch", "--show-current") != "main"
            or head != _git(root, "rev-parse", "origin/main")
            or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
            != ["0", "0"]
            or _git(root, "status", "--porcelain")
        ):
            raise RuntimeError("qualification requires synchronized clean main")


def build_selection_and_gradient_decision(
    *,
    contract: Mapping[str, Any],
    signal_summary: Mapping[str, Any],
    summaries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, object]:
    """Finalize representative selection only after two exact projections per arm."""

    _verify(contract, "qualification_contract_sha256")
    _verify(signal_summary, "signal_summary_sha256")
    signal_arms = cast(Mapping[str, Mapping[str, Any]], signal_summary.get("arms"))
    if (
        signal_summary.get("decision") != "schedule_viable"
        or signal_summary.get("viability_passed") is not True
        or set(signal_arms) != set(ARMS)
    ):
        raise ValueError("fresh-process replay did not complete the viability gate")
    selected_contract = cast(Mapping[str, Mapping[str, Any]], contract["selected_candidates"])
    arms: dict[str, object] = {}
    source_commits: set[str] = set()
    for arm in ARMS:
        rows = summaries.get(arm)
        if rows is None or len(rows) != 2:
            raise ValueError(f"{arm} requires exactly two independently reset projections")
        for row in rows:
            _verify(row, "summary_sha256")
            source_commit = row.get("source_commit")
            if not isinstance(source_commit, str) or len(source_commit) != 40:
                raise ValueError(f"{arm} projection source commit differs")
            source_commits.add(source_commit)
        exact_hashes = {cast(str, row["exact_projection_sha256"]) for row in rows}
        selected = selected_contract[arm]
        replay_status = cast(
            Mapping[str, Any],
            signal_arms[arm].get("deterministic_replay_status"),
        )
        policy_norms = [row.get("policy_gradient_global_norm") for row in rows]
        combined_norms = [row.get("combined_gradient_global_norm") for row in rows]
        if (
            len(exact_hashes) != 1
            or any(row.get("gate_passed") is not True for row in rows)
            or any(row.get("optimizer_created") is not False for row in rows)
            or any(row.get("optimizer_steps") != 0 for row in rows)
            or any(row.get("task_group_id") != selected["task_group_id"] for row in rows)
            or any(row.get("reference_gradient_count") != 0 for row in rows)
            or any(row.get("base_gradient_count") != 0 for row in rows)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in (*policy_norms, *combined_norms)
            )
            or any(
                isinstance(row.get("policy_nonzero_gradient_tensor_count"), bool)
                or not isinstance(row.get("policy_nonzero_gradient_tensor_count"), int)
                or cast(int, row["policy_nonzero_gradient_tensor_count"]) <= 0
                for row in rows
            )
            or replay_status.get("passed") is not True
            or replay_status.get("group_id") != selected["task_group_id"]
        ):
            raise ValueError(f"{arm} projection qualification failed")
        arms[arm] = {
            **dict(selected),
            "fresh_process_replay_status": "passed",
            "gradient_projection_status": "passed",
            "independently_reset_processes": 2,
            "exact_duplicate_evidence": True,
            "exact_projection_sha256": next(iter(exact_hashes)),
            "policy_gradient_global_norm": rows[0]["policy_gradient_global_norm"],
            "combined_gradient_global_norm": rows[0]["combined_gradient_global_norm"],
            "policy_nonzero_gradient_tensor_count": rows[0]["policy_nonzero_gradient_tensor_count"],
            "reference_gradient_count": rows[0]["reference_gradient_count"],
            "base_gradient_count": rows[0]["base_gradient_count"],
            "runtime_summary_sha256s": [row["summary_sha256"] for row in rows],
            "raw_projection_file_sha256s": [row["raw_projection_file_sha256"] for row in rows],
        }
    if len(source_commits) != 1:
        raise ValueError("projection source commits differ")
    decision: dict[str, object] = {
        "schema_version": 1,
        "decision_id": "foundry-l3-grpo-signal-selection-and-gradient-v1",
        "qualification_contract_sha256": contract["qualification_contract_sha256"],
        "source_commit": next(iter(source_commits)),
        "signal_summary_sha256": signal_summary["signal_summary_sha256"],
        "arms": arms,
        "both_arms_pass": True,
        "selection_frozen": True,
        "compatibility_authorized": True,
        "counted_training_authorized": False,
        "optimizer_steps": 0,
        "holdout_v2_status": "not_run",
        "gsm1k_status": "not_run",
        "sealed_content_use": 0,
    }
    decision["selection_decision_sha256"] = canonical_sha256(decision)
    return decision


def _write_new_or_identical(path: Path, value: object) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"existing qualification artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def write_qualification_freeze(
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    implementation, contract = freeze_qualification(root)
    output = root / "results/phase2_vetted_corpus"
    _write_new_or_identical(output / IMPLEMENTATION_OUTPUT, implementation)
    _write_new_or_identical(output / CONTRACT_OUTPUT, contract)
    return implementation, contract


def write_selection_and_signal_decision(
    root: Path,
    *,
    signal_summary: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    """Publish content-free replay, selection, and gradient decisions once."""

    _verify(signal_summary, "signal_summary_sha256")
    _verify(decision, "selection_decision_sha256")
    output = root / "results/phase2_vetted_corpus"
    _write_new_or_identical(output / SIGNAL_SUMMARY_OUTPUT, signal_summary)
    _write_new_or_identical(output / SELECTION_OUTPUT, decision)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    implementation, contract = write_qualification_freeze(args.root)
    print(
        json.dumps(
            {
                "implementation_sha256": implementation["implementation_sha256"],
                "qualification_contract_sha256": contract["qualification_contract_sha256"],
                "selected_candidates": contract["selected_candidates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
