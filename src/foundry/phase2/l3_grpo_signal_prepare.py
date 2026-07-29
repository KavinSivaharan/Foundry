"""Freeze the content-free Milestone 14B signal-audit source and contract."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_contract import (
    DETERMINISTIC_ENVIRONMENT,
    MODEL_REVISION,
    STARTING_ADAPTER_SHA256,
)
from foundry.phase2.l3_grpo_signal_audit import signal_audit_method_contract
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

STARTING_COMMIT = "d1c4edf15510128413735a19f937d5451137ae0b"
IMPLEMENTATION_OUTPUT = "milestone14b_signal_audit_implementation.json"
CONTRACT_OUTPUT = "milestone14b_signal_audit_contract.json"

IMPLEMENTATION_FILES = (
    "src/foundry/phase2/l3_grpo_signal_audit.py",
    "src/foundry/phase2/l3_grpo_signal_analysis.py",
    "src/foundry/phase2/l3_grpo_signal_runtime.py",
    "src/foundry/phase2/l3_grpo_signal_campaign.py",
    "src/foundry/phase2/l3_grpo_signal_prepare.py",
    "tests/unit/phase2/test_l3_grpo_signal_audit.py",
    "tests/unit/phase2/test_l3_grpo_signal_analysis.py",
    "tests/unit/phase2/test_l3_grpo_signal_runtime.py",
    "tests/unit/phase2/test_l3_grpo_signal_campaign.py",
    "tests/unit/phase2/test_l3_grpo_signal_prepare.py",
)
FROZEN_DEPENDENCY_FILES = (
    "src/foundry/phase2/l3_grpo_contract.py",
    "src/foundry/phase2/l3_grpo_runtime.py",
    "src/foundry/phase2/l3_grpo_schedule.py",
    "src/foundry/phase2/l3_grpo_reward.py",
    "src/foundry/phase2/l3_grpo_reference.py",
    "src/foundry/phase2/l3_grpo_zero_gradient.py",
    "src/foundry/phase2/launch_contract.py",
    "src/foundry/phase2/vetted_qlora_kl.py",
    "src/foundry/training/grpo_compatibility.py",
    "src/foundry/training/grpo_replay_evidence.py",
    "src/foundry/training/grpo_runtime.py",
    "src/foundry/training/grpo_trainer.py",
    "src/foundry/training/qlora.py",
)


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


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.strip()


def _require_freeze_boundary(root: Path) -> None:
    if root.resolve() != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("Milestone 14B is attached to the wrong repository")
    dirty = _git(root, "status", "--porcelain")
    allowed_prefixes = (
        "src/foundry/phase2/l3_grpo_signal_",
        "tests/unit/phase2/test_l3_grpo_signal_",
    )
    dirty_paths = [line[3:].replace("\\", "/") for line in dirty.splitlines() if len(line) >= 4]
    if (
        _git(root, "branch", "--show-current") != "main"
        or _git(root, "rev-parse", "HEAD") != STARTING_COMMIT
        or _git(root, "rev-parse", "origin/main") != STARTING_COMMIT
        or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
        != ["0", "0"]
        or any(not path.startswith(allowed_prefixes) for path in dirty_paths)
    ):
        raise RuntimeError("Milestone 14B source-freeze boundary differs")
    raw_output = root / "results/raw/phase2_vetted_corpus/milestone14b"
    if raw_output.exists():
        raise RuntimeError("Milestone 14B model evidence exists before source freeze")


def _file_rows(root: Path, paths: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"signal-audit source is missing: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def _frozen_inputs(root: Path) -> dict[str, object]:
    generic = _read(root / "results/phase2_vetted_corpus/milestone14a_generic_schedule.json")
    targeted = _read(root / "results/phase2_vetted_corpus/milestone14a_targeted_schedule.json")
    replay = _read(root / "results/phase2_vetted_corpus/milestone14a_shared_replay.json")
    paired = _read(root / "results/phase2_vetted_corpus/milestone14a_paired_schedule.json")
    blocker = _read(
        root / "results/phase2_vetted_corpus/milestone14a_r1_compatibility_blocker.json"
    )
    for value, key in (
        (generic, "manifest_sha256"),
        (targeted, "manifest_sha256"),
        (replay, "shared_replay_sha256"),
        (paired, "paired_schedule_sha256"),
        (blocker, "blocker_sha256"),
    ):
        _verify(value, key)
    expected = {
        "generic_schedule_sha256": (
            "ff1005a1d7381acd52dd28b3d054b2979986c47595ed09c944880ea5fc5f5ff3"
        ),
        "targeted_schedule_sha256": (
            "8326c1b91ba127c4734527abfed2f8bca41ecbb3a0bb7bc62a5bf940ac24f0c4"
        ),
        "shared_replay_sha256": (
            "19e27fecde5349b6a7a9a24d8a0a8211a3b0da877282a51ece6b616688904181"
        ),
        "paired_schedule_sha256": (
            "ed99aa38f77961fa1f669ba110cd86b3af092e027a60b8e92096a9a68bdfc8e3"
        ),
        "r1_compatibility_blocker_sha256": (
            "2a5fadb4071ac9f2be1371881744dcaacb1a696ed654befc220a29f59499b25b"
        ),
    }
    actual = {
        "generic_schedule_sha256": generic["manifest_sha256"],
        "targeted_schedule_sha256": targeted["manifest_sha256"],
        "shared_replay_sha256": replay["shared_replay_sha256"],
        "paired_schedule_sha256": paired["paired_schedule_sha256"],
        "r1_compatibility_blocker_sha256": blocker["blocker_sha256"],
    }
    if actual != expected:
        raise ValueError("Milestone 14B frozen input identity differs")
    return actual


def freeze_signal_audit(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Build the source manifest and complete pre-generation audit contract."""

    root = root.resolve()
    _require_freeze_boundary(root)
    files = _file_rows(root, (*IMPLEMENTATION_FILES, *FROZEN_DEPENDENCY_FILES))
    implementation: dict[str, object] = {
        "schema_version": 1,
        "implementation_id": "foundry-l3-grpo-signal-audit-implementation-v1",
        "starting_commit": STARTING_COMMIT,
        "implementation_files": list(IMPLEMENTATION_FILES),
        "frozen_dependency_files": list(FROZEN_DEPENDENCY_FILES),
        "files": files,
        "model_generation_calls": 0,
        "optimizer_steps": 0,
        "sealed_content_use": 0,
    }
    implementation["implementation_sha256"] = canonical_sha256(implementation)
    inputs = _frozen_inputs(root)
    method = signal_audit_method_contract()
    contract: dict[str, object] = {
        "schema_version": 1,
        "contract_id": "foundry-l3-grpo-signal-audit-v1",
        "starting_commit": STARTING_COMMIT,
        "base_model_revision": MODEL_REVISION,
        "dataset_sha256": ("ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"),
        "starting_adapters": STARTING_ADAPTER_SHA256,
        "schedules": {
            "generic": inputs["generic_schedule_sha256"],
            "targeted": inputs["targeted_schedule_sha256"],
            "shared_replay": inputs["shared_replay_sha256"],
            "paired": inputs["paired_schedule_sha256"],
        },
        "reward": {
            "implementation_sha256": (
                "448574e61ff74c40b026dd493b9773023c92eb7f4dbaccb5dfbf511be1e68e66"
            ),
            "configuration_sha256": (
                "701ac381bf01337f706dfaf46ebfa91839147bd600e23dc21a3f1d00eb0f5df5"
            ),
            "fixture_sha256": ("ca7ea72ae288234eb769486f1a1dd0893f9c14b1c14487ae043336af30318199"),
            "calibration_sha256": (
                "e0952f0034424f7817998300207ec3eecf2ce4f8443a87405899decff3fb65e7"
            ),
            "contract_sha256": ("441933982c2b51b49195763440c318893cea22af947c9efc50b732d05fee7b61"),
        },
        "reference_mechanism_sha256": (
            "674b368105f08b0e1eb00f54c6912f611da730ed070f60c63989de996ecb0316"
        ),
        "r1_compatibility_blocker_sha256": inputs["r1_compatibility_blocker_sha256"],
        "method_contract": method,
        "implementation_sha256": implementation["implementation_sha256"],
        "process_environment": DETERMINISTIC_ENVIRONMENT,
        "authorized_general_interpreter": ".venv/Scripts/python.exe",
        "authorized_model_interpreter": ".venv-training/Scripts/python.exe",
        "source_frozen_before_generation": True,
        "complete_schedule_order_preserved": True,
        "scientific_settings_changed": False,
        "optimizer_creation_authorized": False,
        "backward_authorized": False,
        "scheduler_creation_or_advancement_authorized": False,
        "adapter_mutation_authorized": False,
        "adapter_save_authorized": False,
        "counted_training_authorized": False,
        "holdout_v2_authorized": False,
        "gsm1k_authorized": False,
        "sealed_content_use": 0,
    }
    contract["signal_audit_contract_sha256"] = canonical_sha256(contract)
    return implementation, contract


def _write_new_or_identical(path: Path, value: object) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"existing signal-audit freeze differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def write_signal_audit_freeze(
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    implementation, contract = freeze_signal_audit(root)
    output = root / "results/phase2_vetted_corpus"
    _write_new_or_identical(output / IMPLEMENTATION_OUTPUT, implementation)
    _write_new_or_identical(output / CONTRACT_OUTPUT, contract)
    return implementation, contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    implementation, contract = write_signal_audit_freeze(args.root)
    print(
        json.dumps(
            {
                "implementation_sha256": implementation["implementation_sha256"],
                "signal_audit_contract_sha256": contract["signal_audit_contract_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
