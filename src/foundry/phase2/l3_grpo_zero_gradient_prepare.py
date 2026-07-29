"""Freeze Milestone 14A-R1 classification, fixtures, and diagnostic source."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from foundry.phase2.l3_grpo_zero_gradient import (
    classification_contract,
    run_deterministic_fixtures,
)
from foundry.phase2.launch_contract import validate_preimport
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

STARTING_COMMIT = "b0635a7c0f551dfb8efd846da5cfe83b28f7af18"
ORIGINAL_IMPLEMENTATION_SHA256 = "63631afa3abb5d0faf2a73decf9a96285d9cbfff789a47806b009c3d8b672eae"
ORIGINAL_BLOCKER_SHA256 = "d4b23d898ef3c53db46882a4a218c2a43cd85298ebdfa75139eaf3a7c08e8752"
FREEZE_ID = "foundry-milestone14a-r1-zero-gradient-freeze-v1"
OUTPUT_NAME = "milestone14a_r1_zero_gradient_freeze.json"
IMPLEMENTATION_FILES = (
    "src/foundry/phase2/l3_grpo_zero_gradient.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_diagnostic.py",
    "src/foundry/phase2/l3_grpo_zero_gradient_prepare.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_diagnostic.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_prepare.py",
)
FROZEN_DEPENDENCY_FILES = (
    "src/foundry/phase2/l3_grpo_runtime.py",
    "src/foundry/phase2/l3_grpo_reference.py",
    "src/foundry/phase2/l3_grpo_reward.py",
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"zero-gradient freeze already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _source_manifest(root: Path) -> dict[str, object]:
    rows = [
        {
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": file_sha256(root / relative),
        }
        for relative in (*IMPLEMENTATION_FILES, *FROZEN_DEPENDENCY_FILES)
    ]
    payload: dict[str, object] = {
        "manifest_id": "foundry-milestone14a-r1-prediagnostic-source-v1",
        "implementation_files": list(IMPLEMENTATION_FILES),
        "frozen_dependency_files": list(FROZEN_DEPENDENCY_FILES),
        "files": rows,
    }
    payload["source_manifest_sha256"] = canonical_sha256(payload)
    return payload


def freeze(root: Path) -> dict[str, object]:
    """Write the pre-model classification freeze once."""

    root = root.resolve()
    if root != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("Milestone 14A-R1 is attached to the wrong repository")
    if (
        _git(root, "branch", "--show-current") != "main"
        or _git(root, "rev-parse", "HEAD") != STARTING_COMMIT
        or _git(root, "rev-parse", "main") != STARTING_COMMIT
        or _git(root, "rev-parse", "origin/main") != STARTING_COMMIT
        or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
        != ["0", "0"]
    ):
        raise RuntimeError("Milestone 14A-R1 starting Git state differs")
    dirty = _git(root, "status", "--porcelain")
    allowed = (
        "src/foundry/phase2/l3_grpo_zero_gradient",
        "tests/unit/phase2/test_l3_grpo_zero_gradient",
    )
    dirty_paths = [line[3:].replace("\\", "/") for line in dirty.splitlines() if len(line) >= 4]
    if not dirty_paths or any(not path.startswith(allowed) for path in dirty_paths):
        raise RuntimeError("prediagnostic changes exceed the Milestone 14A-R1 implementation")
    blocker = json.loads(
        (root / "results/phase2_vetted_corpus/milestone14a_compatibility_blocker.json").read_text(
            encoding="utf-8"
        )
    )
    blocker_projected = dict(blocker)
    blocker_sha256 = blocker_projected.pop("blocker_sha256")
    if blocker_sha256 != ORIGINAL_BLOCKER_SHA256 or blocker_sha256 != canonical_sha256(
        blocker_projected
    ):
        raise ValueError("published Milestone 14A blocker differs")
    validate_preimport()
    torch = importlib.import_module("torch")
    contract = classification_contract()
    fixtures = run_deterministic_fixtures(torch)
    source = _source_manifest(root)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "freeze_id": FREEZE_ID,
        "starting_commit": STARTING_COMMIT,
        "original_implementation_sha256": ORIGINAL_IMPLEMENTATION_SHA256,
        "original_blocker_sha256": ORIGINAL_BLOCKER_SHA256,
        "classification_contract": contract,
        "fixture_contract": fixtures,
        "prediagnostic_source": source,
        "frozen_scientific_contracts": {
            "generic_starting_adapter_sha256": (
                "67c6f1dd34c0fa1ddebb354dfe14c43e61c48fdd90c687ba1a9290d2401479cd"
            ),
            "targeted_starting_adapter_sha256": (
                "4e195ff2cb32c4faa6858915b95507862c911bb2eb853b060717416d825df91d"
            ),
            "generic_manifest_sha256": (
                "ff1005a1d7381acd52dd28b3d054b2979986c47595ed09c944880ea5fc5f5ff3"
            ),
            "targeted_manifest_sha256": (
                "8326c1b91ba127c4734527abfed2f8bca41ecbb3a0bb7bc62a5bf940ac24f0c4"
            ),
            "shared_replay_sha256": (
                "19e27fecde5349b6a7a9a24d8a0a8211a3b0da877282a51ece6b616688904181"
            ),
            "paired_schedule_sha256": (
                "ed99aa38f77961fa1f669ba110cd86b3af092e027a60b8e92096a9a68bdfc8e3"
            ),
            "reward_contract_sha256": (
                "441933982c2b51b49195763440c318893cea22af947c9efc50b732d05fee7b61"
            ),
            "reference_mechanism_sha256": (
                "674b368105f08b0e1eb00f54c6912f611da730ed070f60c63989de996ecb0316"
            ),
            "recipe_sha256": ("0c3280d81d60e76e3f58e9ae44a62b378540ae9d92d277606ba021e688403641"),
        },
        "model_generation_calls": 0,
        "optimizer_steps": 0,
        "scientific_settings_changed": False,
        "sealed_content_use": 0,
    }
    payload["freeze_sha256"] = canonical_sha256(payload)
    _write_new(root / f"results/phase2_vetted_corpus/{OUTPUT_NAME}", payload)
    return {
        "freeze_sha256": payload["freeze_sha256"],
        "classification_contract_sha256": contract["classification_contract_sha256"],
        "fixture_sha256": fixtures["fixture_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "model_generation_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(freeze(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
