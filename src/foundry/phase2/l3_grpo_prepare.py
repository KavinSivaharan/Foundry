"""Freeze Milestone 14A schedules and contracts before model generation."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path

from foundry.phase2.l3_grpo_contract import (
    build_experiment_contract,
    build_starting_state,
    write_json_new_or_identical,
)
from foundry.phase2.l3_grpo_reward import serialize_reward_contract
from foundry.phase2.l3_grpo_schedule import (
    build_production_schedules,
    write_schedule_bundle,
)
from foundry.phase2.launch_contract import validate_preimport
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

SOURCE_FILES = (
    "src/foundry/phase2/l3_grpo_contract.py",
    "src/foundry/phase2/l3_grpo_reference.py",
    "src/foundry/phase2/l3_grpo_reward.py",
    "src/foundry/phase2/l3_grpo_runtime.py",
    "src/foundry/phase2/l3_grpo_schedule.py",
    "src/foundry/phase2/l3_grpo_prepare.py",
    "src/foundry/phase2/l3_grpo_analysis.py",
    "src/foundry/phase2/l3_grpo_campaign.py",
)
TEST_FILES = (
    "tests/unit/phase2/test_l3_grpo_contract.py",
    "tests/unit/phase2/test_l3_grpo_reference.py",
    "tests/unit/phase2/test_l3_grpo_reward.py",
    "tests/unit/phase2/test_l3_grpo_runtime.py",
    "tests/unit/phase2/test_l3_grpo_schedule.py",
    "tests/unit/phase2/test_l3_grpo_analysis.py",
    "tests/unit/phase2/test_l3_grpo_campaign.py",
)


def _implementation_manifest(root: Path) -> dict[str, object]:
    rows = [
        {
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": file_sha256(root / relative),
        }
        for relative in (*SOURCE_FILES, *TEST_FILES)
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "implementation_id": "foundry-l3-verifier-grpo-implementation-v1",
        "files": rows,
        "source_file_count": len(SOURCE_FILES),
        "test_file_count": len(TEST_FILES),
        "scientific_settings_frozen_before_generation": True,
        "official_smoke_retry_allowed": False,
    }
    payload["implementation_sha256"] = canonical_sha256(payload)
    return payload


def prepare(root: Path) -> dict[str, object]:
    """Write all prompt-bearing and content-free pre-generation artifacts once."""

    root = root.resolve()
    preimport = validate_preimport()
    starting_state = build_starting_state(
        root,
        allow_milestone_implementation_changes=True,
    )
    transformers = importlib.import_module("transformers")
    model_path = (
        root
        / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
        / "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bundle = build_production_schedules(root, tokenizer)
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a/schedules"
    tracked = root / "results/phase2_vetted_corpus"
    paths = write_schedule_bundle(
        root,
        bundle,
        raw_directory=raw,
        tracked_directory=tracked,
    )
    reward_contract: object = json.loads(serialize_reward_contract())
    implementation = _implementation_manifest(root)
    experiment = build_experiment_contract(
        starting_state,
        bundle.summary,
        str(implementation["implementation_sha256"]),
    )
    outputs = {
        "starting_state": tracked / "milestone14a_starting_state.json",
        "reward_contract": tracked / "milestone14a_reward_contract.json",
        "experiment_contract": tracked / "milestone14a_experiment_contract.json",
        "implementation": tracked / "milestone14a_implementation.json",
    }
    write_json_new_or_identical(outputs["starting_state"], starting_state)
    write_json_new_or_identical(outputs["reward_contract"], reward_contract)
    write_json_new_or_identical(outputs["experiment_contract"], experiment)
    write_json_new_or_identical(outputs["implementation"], implementation)
    result: dict[str, object] = {
        "schema_version": 1,
        "preparation_id": "foundry-milestone14a-pre-generation-freeze-v1",
        "preimport_evidence_sha256": preimport["preimport_evidence_sha256"],
        "starting_state_sha256": starting_state["starting_state_sha256"],
        "paired_schedule_sha256": bundle.summary["paired_schedule_sha256"],
        "generic_manifest_sha256": bundle.generic.manifest["manifest_sha256"],
        "targeted_manifest_sha256": bundle.targeted.manifest["manifest_sha256"],
        "shared_replay_sha256": bundle.shared_replay["shared_replay_sha256"],
        "reward_contract_sha256": experiment["reward"]["contract_sha256"],  # type: ignore[index]
        "reference_mechanism_sha256": experiment["reference"]["reference_mechanism_sha256"],  # type: ignore[index]
        "recipe_sha256": experiment["recipe_sha256"],
        "experiment_contract_sha256": experiment["experiment_contract_sha256"],
        "implementation_sha256": implementation["implementation_sha256"],
        "prompt_packets": {
            "generic": file_sha256(paths["generic_packet"]),
            "targeted": file_sha256(paths["targeted_packet"]),
        },
        "model_generation_calls": 0,
        "scientific_settings_frozen": True,
    }
    result["preparation_sha256"] = canonical_sha256(result)
    write_json_new_or_identical(
        tracked / "milestone14a_preparation.json",
        result,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
