"""Publish the content-free Milestone 14B incomplete-audit blocker."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_contract import FIXED_LIBRARY_NOTICE_CLASSES
from foundry.phase2.l3_grpo_signal_audit import TASK_FAMILIES
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

BLOCKER_ID = "foundry-l3-grpo-signal-audit-blocker-v1"
OUTPUT_NAME = "milestone14b_signal_audit_blocker.json"
EXPECTED_EXCEPTION = "stock TRL advantages differ from frozen reward projection"
OBSERVED_FAMILIES = (
    "constraint_distribution_or_discrete_reasoning",
    "multi_step_bookkeeping_or_omission",
    "rate_ratio_percentage_or_average",
)
FIXTURE_VALUES = (
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    -0.1,
    0.0,
    0.05,
    0.1,
    0.15,
    0.9,
    1.0,
    1.05,
    1.1,
    1.15,
    1.2,
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


def cross_device_projection_fixture(torch: Any) -> dict[str, object]:
    """Demonstrate why CPU/CUDA float32 advantages cannot use exact equality."""

    if not bool(torch.cuda.is_available()):
        raise RuntimeError("cross-device projection fixture requires CUDA")
    torch.use_deterministic_algorithms(True, warn_only=False)
    rows = torch.tensor(
        list(itertools.product(FIXTURE_VALUES, repeat=4)),
        dtype=torch.float32,
    )
    cpu = rows - rows.mean(dim=1, keepdim=True)
    cuda_rows = rows.to("cuda:0")
    cuda = (cuda_rows - cuda_rows.mean(dim=1, keepdim=True)).cpu()
    differing = torch.any(cpu != cuda, dim=1)
    indices = torch.nonzero(differing).flatten()
    if int(indices.numel()) == 0:
        raise RuntimeError("cross-device projection fixture did not expose a mismatch")
    first = int(indices[0].item())
    result: dict[str, object] = {
        "fixture_id": "foundry-float32-cpu-cuda-advantage-projection-v1",
        "input_value_count": len(FIXTURE_VALUES),
        "fixture_vector_count": int(rows.shape[0]),
        "exact_mismatch_vector_count": int(indices.numel()),
        "first_mismatch_input": [float(value) for value in rows[first].tolist()],
        "first_cpu_advantages": [float(value) for value in cpu[first].tolist()],
        "first_cuda_advantages": [float(value) for value in cuda[first].tolist()],
        "maximum_absolute_difference": float(torch.max(torch.abs(cpu - cuda)).item()),
        "strict_determinism": bool(torch.are_deterministic_algorithms_enabled()),
        "model_loaded": False,
        "generation_calls": 0,
    }
    result["fixture_sha256"] = canonical_sha256(result)
    return result


def _family_summary(groups: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    task_groups = [group for group in groups if group.get("source_kind") == "task"]
    names = sorted({cast(str, group["task_family"]) for group in task_groups})
    return {
        name: {
            "audited_groups": sum(group.get("task_family") == name for group in task_groups),
            "nonzero_variance_groups": sum(
                group.get("task_family") == name
                and float(cast(float, group["reward_variance"])) > 0.0
                for group in task_groups
            ),
            "zero_variance_groups": sum(
                group.get("task_family") == name and group.get("reward_variance") == 0.0
                for group in task_groups
            ),
        }
        for name in names
    }


def _resource_observation(raw_root: Path) -> dict[str, object]:
    files = sorted(path for path in raw_root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError("Milestone 14B raw blocker evidence is missing")
    rows = [
        {
            "path": path.relative_to(raw_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    created = min(path.stat().st_ctime for path in files)
    modified = max(path.stat().st_mtime for path in files)
    return {
        "files": rows,
        "disk_bytes": sum(path.stat().st_size for path in files),
        "observed_start_utc": datetime.fromtimestamp(created, tz=UTC).isoformat(),
        "observed_end_utc": datetime.fromtimestamp(modified, tz=UTC).isoformat(),
        "observed_wall_clock_seconds": modified - created,
        "peak_allocated_vram_bytes": None,
        "peak_reserved_vram_bytes": None,
        "peak_process_rss_bytes": None,
        "resource_metrics_status": "final_runtime_summary_not_reached",
    }


def build_blocker(root: Path, torch: Any) -> dict[str, object]:
    """Build the terminal blocker from ignored partial evidence and logs."""

    root = root.resolve()
    if root != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("Milestone 14B blocker is attached to the wrong repository")
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone14b/signal_audit"
    partial_path = raw_root / "generic/partial_evidence.json"
    generic_stderr = raw_root / "logs/generic.stderr.txt"
    campaign_stderr = raw_root / "campaign.stderr.txt"
    partial = _read(partial_path)
    _verify(partial, "partial_audit_sha256")
    groups_value = partial.get("groups")
    if not isinstance(groups_value, list):
        raise ValueError("generic partial group evidence differs")
    groups = cast(list[dict[str, Any]], groups_value)
    for index, group in enumerate(groups, start=1):
        _verify(group, "group_record_sha256")
        if group.get("schedule_position") != index:
            raise ValueError("generic partial schedule order differs")
    stderr_text = generic_stderr.read_text(encoding="utf-8")
    campaign_text = campaign_stderr.read_text(encoding="utf-8")
    if (
        EXPECTED_EXCEPTION not in stderr_text
        or "campaign command failed with 1" not in campaign_text
        or (raw_root / "targeted").exists()
        or (raw_root / "generic/raw_evidence.json").exists()
        or (raw_root / "generic/summary.json").exists()
        or len(groups) != 12
        or partial.get("completed_completion_count") != 48
        or partial.get("optimizer_created") is not False
        or partial.get("backward_calls") != 0
        or partial.get("scheduler_created") is not False
        or partial.get("adapter_saved") is not False
    ):
        raise ValueError("Milestone 14B incomplete-audit blocker differs")
    observed_families = tuple(
        sorted({cast(str, group["task_family"]) for group in groups if group["task_family"]})
    )
    if observed_families != OBSERVED_FAMILIES:
        raise ValueError("partial audit task-family identities differ")
    classifications = Counter(
        cast(str, group["zero_variance_classification"])
        for group in groups
        if group.get("zero_variance_classification") is not None
    )
    notice_counts = {
        notice["class_id"]: stderr_text.count(notice["required_substring"])
        for notice in FIXED_LIBRARY_NOTICE_CLASSES
    }
    fixture = cross_device_projection_fixture(torch)
    result: dict[str, object] = {
        "schema_version": 1,
        "blocker_id": BLOCKER_ID,
        "decision": "signal_audit_implementation_blocker",
        "failure_classification": "cross_device_exact_advantage_projection_assertion",
        "failure_exception": EXPECTED_EXCEPTION,
        "failure_arm": "generic",
        "failed_schedule_position": 13,
        "failure_location": ("post_generation_and_reward_scoring_before_group_record_publication"),
        "source_commit": partial["source_commit"],
        "signal_audit_contract_sha256": partial["signal_audit_contract_sha256"],
        "schedule_manifest_sha256": partial["schedule_manifest_sha256"],
        "partial_audit_sha256": partial["partial_audit_sha256"],
        "partial_evidence_file_sha256": file_sha256(partial_path),
        "generic_stderr_file_sha256": file_sha256(generic_stderr),
        "campaign_stderr_file_sha256": file_sha256(campaign_stderr),
        "published_group_count": len(groups),
        "published_completion_count": len(groups) * 4,
        "generated_group_count": len(groups) + 1,
        "generated_completion_count": (len(groups) + 1) * 4,
        "failed_group_evidence_published": False,
        "partial_task_group_count": sum(group.get("source_kind") == "task" for group in groups),
        "partial_replay_group_count": sum(
            group.get("source_kind") == "base_replay" for group in groups
        ),
        "partial_nonzero_variance_group_count": sum(
            float(cast(float, group["reward_variance"])) > 0.0 for group in groups
        ),
        "partial_nonzero_variance_task_group_count": sum(
            group.get("source_kind") == "task"
            and float(cast(float, group["reward_variance"])) > 0.0
            for group in groups
        ),
        "partial_nonzero_variance_replay_group_count": sum(
            group.get("source_kind") == "base_replay"
            and float(cast(float, group["reward_variance"])) > 0.0
            for group in groups
        ),
        "partial_zero_variance_group_count": sum(
            group.get("reward_variance") == 0.0 for group in groups
        ),
        "partial_zero_variance_classifications": {
            name: classifications[name] for name in sorted(classifications)
        },
        "partial_family_signal_density": _family_summary(groups),
        "partial_backend_failure_count": sum(
            cast(int, group["backend_failure_count"]) for group in groups
        ),
        "partial_reward_contract_inconsistency_count": sum(
            group.get("reward_contract_consistent") is not True for group in groups
        ),
        "partial_completion_tokens": sum(
            sum(cast(list[int], group["completion_token_counts"])) for group in groups
        ),
        "optimizer_created": False,
        "backward_calls": 0,
        "scheduler_created": False,
        "adapter_saved": False,
        "final_policy_reference_base_integrity": ("not_published_due_prepublication_failure"),
        "cross_device_projection_fixture": fixture,
        "root_cause": {
            "stock_trl_advantage_device": "cuda",
            "frozen_helper_advantage_device": "cpu",
            "comparison": "torch.equal_after_cpu_projection_copy_to_cuda",
            "exact_cross_device_equality_valid": False,
            "scientific_setting_changed": False,
        },
        "secondary_analysis_defect": {
            "classification": "task_family_identifier_alias_mismatch",
            "frozen_helper_family_ids": list(TASK_FAMILIES),
            "observed_schedule_family_ids": list(observed_families),
            "raw_group_family_evidence_affected": False,
            "aggregation_ready": False,
        },
        "fixed_library_notice_counts": notice_counts,
        "resource_usage": _resource_observation(raw_root),
        "generic_complete_audit_status": "failed_at_group_13",
        "targeted_audit_status": "not_started",
        "viability_gate_status": "not_reached",
        "representative_selection_status": "not_reached",
        "gradient_projection_status": "not_reached",
        "official_smoke_status": "not_reached",
        "duplicate_smoke_status": "not_reached",
        "counted_training_status": "not_run",
        "holdout_v2_status": "not_run",
        "gsm1k_status": "not_run",
        "sealed_content_use": 0,
        "retry_performed": False,
        "next_action": "authorize_a_corrected_fresh_signal_audit_or_close_grpo",
    }
    result["blocker_sha256"] = canonical_sha256(result)
    return result


def _write_new_or_identical(path: Path, value: object) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"existing signal-audit blocker differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    blocker = build_blocker(args.root, torch)
    output = args.root / "results/phase2_vetted_corpus" / OUTPUT_NAME
    _write_new_or_identical(output, blocker)
    print(json.dumps(blocker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
