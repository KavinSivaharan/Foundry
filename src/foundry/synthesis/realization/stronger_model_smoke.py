"""Run the exact M5C compact IRs with the approved Qwen3-4B model only."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from foundry.synthesis.realization.compact_smoke import (
    CompactIrRecord,
    deterministic_compact_sha256,
    execute_compact_smoke,
)
from foundry.synthesis.realization.stronger_model_contract import (
    StrongerModelComparisonConfig,
    load_stronger_model_config,
    verify_controlled_irs,
)


def execute_stronger_model_smoke(
    *,
    repository_root: Path,
    comparison: StrongerModelComparisonConfig,
    write_artifacts: bool,
) -> tuple[tuple[CompactIrRecord, ...], dict[str, object]]:
    """Verify the controls, then change only the realization model."""

    control = verify_controlled_irs(comparison)
    probe: dict[str, object] = json.loads(
        comparison.memory_probe.raw_path.read_text(encoding="utf-8")
    )
    if probe.get("status") != "passed":
        raise RuntimeError("the approved three-beam stronger-model probe did not pass")
    records, summary = execute_compact_smoke(
        repository_root=repository_root,
        config=comparison.compact,
        write_artifacts=write_artifacts,
    )
    summary["controlled_comparison"] = {
        "only_changed_variable": "realization_model",
        "m5c_model": "Qwen/Qwen3-1.7B",
        "m5d_model": comparison.artifact.repo_id,
        "m5c_deterministic_run_sha256": comparison.m5c_deterministic_run_sha256,
        "m5c_raw_sha256": comparison.m5c_raw_sha256,
        "control_verification": asdict(control),
        "combined_experiment_sha256": comparison.combined_experiment_sha256,
    }
    summary["model_artifact"] = asdict(comparison.artifact)
    summary["memory_probe"] = {key: value for key, value in probe.items() if key != "beams"}
    if write_artifacts:
        comparison.summary_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return tuple(records), summary


def _main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    comparison = load_stronger_model_config(args.config)
    records, summary = execute_stronger_model_smoke(
        repository_root=repository_root,
        comparison=comparison,
        write_artifacts=not args.replay,
    )
    run_hash = deterministic_compact_sha256(records)
    if args.replay:
        original: dict[str, Any] = json.loads(comparison.summary_path.read_text(encoding="utf-8"))
        expected = original.get("deterministic_run_sha256")
        result = {
            "status": "passed" if run_hash == expected else "failed",
            "expected_sha256": expected,
            "actual_sha256": run_hash,
            "irs": len(records),
            "beams": sum(len(record.beams) for record in records),
            "replay_measurements": summary["measurements"],
        }
        replay_path = comparison.raw_directory / "replay.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    print(
        json.dumps(
            {
                "status": "completed",
                "attempted_irs": summary["attempted_irs"],
                "generated_beams": summary["generated_beams"],
                "automatic_selected_irs": summary["automatic_selected_irs"],
                "deterministic_run_sha256": run_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["execute_stronger_model_smoke"]
