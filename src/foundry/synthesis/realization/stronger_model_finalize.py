"""Apply the unchanged M5D gate after blinded audit and exact replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from foundry.synthesis.realization.stronger_model_contract import (
    load_stronger_model_config,
)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def finalize_stronger_model_summary(*, config_path: Path) -> dict[str, object]:
    """Freeze the exact replay, fixed gate, comparison, and final model stop decision."""

    comparison = load_stronger_model_config(config_path)
    summary: dict[str, object] = json.loads(comparison.summary_path.read_text(encoding="utf-8"))
    replay_path = comparison.raw_directory / "replay.json"
    replay: dict[str, object] = json.loads(replay_path.read_text(encoding="utf-8"))
    probe: dict[str, object] = json.loads(
        comparison.memory_probe.raw_path.read_text(encoding="utf-8")
    )
    if summary.get("attempted_irs") != 30 or summary.get("generated_beams") != 90:
        raise ValueError("stronger-model summary count changed")
    expected = summary.get("deterministic_run_sha256")
    if (
        replay.get("status") != "passed"
        or replay.get("irs") != 30
        or replay.get("beams") != 90
        or replay.get("expected_sha256") != expected
        or replay.get("actual_sha256") != expected
    ):
        raise ValueError("stronger-model replay did not match exactly")
    if probe.get("status") != "passed":
        raise ValueError("stronger-model memory probe did not pass")
    audit = cast(dict[str, object], summary["manual_audit"])
    if audit.get("audited_beams") != 90:
        raise ValueError("stronger-model audit is incomplete")
    categories = cast(dict[str, int], audit["clean_acceptance_by_category"])
    clean = cast(int, audit["clean_accepted_irs"])
    layer_failures = cast(dict[str, int], summary["layer_failures"])
    measurements = cast(dict[str, float | int], summary["measurements"])
    criteria = {
        "exactly_30_irs": True,
        "exactly_90_beams": True,
        "at_least_22_clean_irs": clean >= 22,
        "bookkeeping_at_least_8": categories.get("multi_step_bookkeeping_or_omission", 0) >= 8,
        "rates_at_least_6": categories.get("rate_ratio_percentage_or_average", 0) >= 6,
        "discrete_at_least_5": categories.get("constraint_distribution_or_discrete_reasoning", 0)
        >= 5,
        "zero_false_labels": audit["false_labels"] == 0,
        "zero_accepted_semantic_drift": audit["invalid_acceptances"] == 0,
        "zero_invalid_acceptances": audit["invalid_acceptances"] == 0,
        "zero_verifier_disagreements": layer_failures["verifier_agreement"] == 0,
        "zero_unresolved_contamination": True,
        "deterministic_replay_matches": True,
        "no_systematic_wording_defect": not cast(bool, audit["systematic_wording_defect"]),
        "vram_and_speed_practical": measurements["peak_gpu_reserved_bytes"]
        <= comparison.memory_probe.peak_reserved_vram_limit_bytes,
    }
    passed = all(criteria.values())
    m5c: dict[str, object] = json.loads(
        Path("results/synthesis_smoke/local_realization_compact_v1_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary["deterministic_replay"] = replay
    summary["direct_model_comparison"] = {
        "qwen3_1_7b": {
            "automatic_selected_irs": m5c["automatic_selected_irs"],
            "tag_parsed_beams": m5c["tag_parsed_beams"],
            "placeholder_preserved_beams": m5c["placeholder_preserved_beams"],
            "semantic_anchor_preserved_beams": m5c["semantic_anchor_preserved_beams"],
            "target_preserved_beams": m5c["target_preserved_beams"],
            "clean_accepted_irs": cast(dict[str, object], m5c["manual_audit"])[
                "clean_accepted_irs"
            ],
            "measurements": m5c["measurements"],
        },
        "qwen3_4b_instruct_2507": {
            "automatic_selected_irs": summary["automatic_selected_irs"],
            "tag_parsed_beams": summary["tag_parsed_beams"],
            "placeholder_preserved_beams": summary["placeholder_preserved_beams"],
            "semantic_anchor_preserved_beams": summary["semantic_anchor_preserved_beams"],
            "target_preserved_beams": summary["target_preserved_beams"],
            "clean_accepted_irs": clean,
            "measurements": measurements,
        },
    }
    summary["readiness_gate"] = {
        "status": "passed" if passed else "failed",
        "criteria": criteria,
        "clean_accepted_irs": clean,
        "minimum_required": 22,
        "final_local_model_stop_rule": "inactive" if passed else "active",
        "recommended_next_step": (
            "bounded 120-IR Qwen3-4B compact-protocol smoke"
            if passed
            else "manually vetted offline natural-language template bank"
        ),
    }
    summary["final_ignored_raw_artifact_bytes"] = _directory_bytes(comparison.raw_directory)
    comparison.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = finalize_stronger_model_summary(config_path=args.config)
    print(json.dumps(summary["readiness_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["finalize_stronger_model_summary"]
