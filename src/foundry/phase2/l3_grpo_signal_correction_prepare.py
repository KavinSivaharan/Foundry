"""Freeze the Milestone 14B-R1 corrected signal-audit source and contracts."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_advantage_equivalence import (
    advantage_equivalence_contract,
    exhaustive_cross_device_fixture,
)
from foundry.phase2.l3_grpo_contract import (
    DETERMINISTIC_ENVIRONMENT,
    MODEL_REVISION,
    STARTING_ADAPTER_SHA256,
)
from foundry.phase2.l3_grpo_signal_audit import (
    family_aggregation_contract,
    signal_audit_method_contract,
)
from foundry.phase2.l3_grpo_signal_continuity import build_prior_diagnostic_manifest
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

STARTING_COMMIT = "de4a02c5e98de9e7cf13bc07f1b6fe40aa348b9f"
ORIGINAL_AUDIT_CONTRACT_SHA256 = "5af65428836a359f332fe6866f2b31c6d0433d6e4e5f0ab28868baefda0beec6"
ORIGINAL_IMPLEMENTATION_SHA256 = "fe68e0676f59ed56027ca935f9092eded69203987808c532fb056dacc28e1c6c"
PUBLISHED_BLOCKER_SHA256 = "0597d2a628bf6acfa320cd0f567c4d3a6e296067bad1767974cb446b7c184ae4"

PRIOR_MANIFEST_OUTPUT = "milestone14b_r1_prior_diagnostic_manifest.json"
ADVANTAGE_CONTRACT_OUTPUT = "milestone14b_r1_advantage_equivalence_contract.json"
FAMILY_CONTRACT_OUTPUT = "milestone14b_r1_family_aggregation_contract.json"
IMPLEMENTATION_OUTPUT = "milestone14b_r1_signal_audit_implementation.json"
CONTRACT_OUTPUT = "milestone14b_r1_signal_audit_contract.json"

IMPLEMENTATION_FILES = (
    "src/foundry/phase2/l3_grpo_advantage_equivalence.py",
    "src/foundry/phase2/l3_grpo_signal_continuity.py",
    "src/foundry/phase2/l3_grpo_signal_audit.py",
    "src/foundry/phase2/l3_grpo_signal_analysis.py",
    "src/foundry/phase2/l3_grpo_signal_runtime.py",
    "src/foundry/phase2/l3_grpo_signal_campaign.py",
    "src/foundry/phase2/l3_grpo_signal_correction_prepare.py",
    "tests/unit/phase2/test_l3_grpo_advantage_equivalence.py",
    "tests/unit/phase2/test_l3_grpo_signal_continuity.py",
    "tests/unit/phase2/test_l3_grpo_signal_audit.py",
    "tests/unit/phase2/test_l3_grpo_signal_analysis.py",
    "tests/unit/phase2/test_l3_grpo_signal_runtime.py",
    "tests/unit/phase2/test_l3_grpo_signal_campaign.py",
    "tests/unit/phase2/test_l3_grpo_signal_correction_prepare.py",
)
FROZEN_DEPENDENCY_FILES = (
    "src/foundry/phase2/l3_grpo_contract.py",
    "src/foundry/phase2/l3_grpo_runtime.py",
    "src/foundry/phase2/l3_grpo_schedule.py",
    "src/foundry/phase2/l3_grpo_reward.py",
    "src/foundry/phase2/l3_grpo_reference.py",
    "src/foundry/phase2/l3_grpo_zero_gradient.py",
    "src/foundry/phase2/l3_grpo_signal_blocker.py",
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


def _require_freeze_boundary(root: Path) -> None:
    if root.resolve() != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("Milestone 14B-R1 is attached to the wrong repository")
    dirty = _git(root, "status", "--porcelain")
    allowed_prefixes = (
        "src/foundry/phase2/l3_grpo_signal_",
        "src/foundry/phase2/l3_grpo_advantage_",
        "tests/unit/phase2/test_l3_grpo_signal_",
        "tests/unit/phase2/test_l3_grpo_advantage_",
        "results/phase2_vetted_corpus/milestone14b_r1_",
        "docs/DEVLOG.md",
        "docs/VERIFIER_GRPO_RESULT.md",
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
        raise RuntimeError("Milestone 14B-R1 source-freeze boundary differs")
    new_raw = root / "results/raw/phase2_vetted_corpus/milestone14b_r1"
    if new_raw.exists():
        raise RuntimeError("Milestone 14B-R1 model evidence exists before source freeze")


def _file_rows(root: Path, paths: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"corrected signal-audit source is missing: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def _original_evidence(root: Path) -> dict[str, object]:
    original_contract = _read(
        root / "results/phase2_vetted_corpus/milestone14b_signal_audit_contract.json"
    )
    original_implementation = _read(
        root / "results/phase2_vetted_corpus/milestone14b_signal_audit_implementation.json"
    )
    blocker = _read(root / "results/phase2_vetted_corpus/milestone14b_signal_audit_blocker.json")
    _verify(original_contract, "signal_audit_contract_sha256")
    _verify(original_implementation, "implementation_sha256")
    _verify(blocker, "blocker_sha256")
    if (
        original_contract["signal_audit_contract_sha256"] != ORIGINAL_AUDIT_CONTRACT_SHA256
        or original_implementation["implementation_sha256"] != ORIGINAL_IMPLEMENTATION_SHA256
        or blocker["blocker_sha256"] != PUBLISHED_BLOCKER_SHA256
    ):
        raise ValueError("published Milestone 14B evidence differs")
    return {
        "original_signal_audit_contract_sha256": ORIGINAL_AUDIT_CONTRACT_SHA256,
        "original_signal_audit_implementation_sha256": ORIGINAL_IMPLEMENTATION_SHA256,
        "published_blocker_sha256": PUBLISHED_BLOCKER_SHA256,
    }


def _frozen_inputs(root: Path) -> dict[str, object]:
    paths_and_keys = (
        ("milestone14a_generic_schedule.json", "manifest_sha256"),
        ("milestone14a_targeted_schedule.json", "manifest_sha256"),
        ("milestone14a_shared_replay.json", "shared_replay_sha256"),
        ("milestone14a_paired_schedule.json", "paired_schedule_sha256"),
    )
    values: list[dict[str, Any]] = []
    for name, key in paths_and_keys:
        value = _read(root / "results/phase2_vetted_corpus" / name)
        _verify(value, key)
        values.append(value)
    actual = {
        "generic_schedule_sha256": values[0]["manifest_sha256"],
        "targeted_schedule_sha256": values[1]["manifest_sha256"],
        "shared_replay_sha256": values[2]["shared_replay_sha256"],
        "paired_schedule_sha256": values[3]["paired_schedule_sha256"],
    }
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
    }
    if actual != expected:
        raise ValueError("Milestone 14B-R1 frozen schedule identity differs")
    return actual


def freeze_correction(
    root: Path,
    *,
    torch_module: Any,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Build every tracked, content-free pre-generation correction artifact."""

    root = root.resolve()
    _require_freeze_boundary(root)
    original = _original_evidence(root)
    inputs = _frozen_inputs(root)
    prior = build_prior_diagnostic_manifest(root)
    fixture_first = exhaustive_cross_device_fixture(torch_module)
    fixture_second = exhaustive_cross_device_fixture(torch_module)
    if fixture_first != fixture_second:
        raise RuntimeError("65,536-vector advantage fixture did not replay deterministically")
    advantage = advantage_equivalence_contract(fixture_first)
    family = family_aggregation_contract()
    files = _file_rows(root, (*IMPLEMENTATION_FILES, *FROZEN_DEPENDENCY_FILES))
    implementation: dict[str, object] = {
        "schema_version": 1,
        "implementation_id": "foundry-l3-grpo-signal-audit-r1-implementation-v1",
        "starting_commit": STARTING_COMMIT,
        "implementation_files": list(IMPLEMENTATION_FILES),
        "frozen_dependency_files": list(FROZEN_DEPENDENCY_FILES),
        "files": files,
        "model_loaded": False,
        "model_generation_calls": 0,
        "optimizer_steps": 0,
        "sealed_content_use": 0,
    }
    implementation["implementation_sha256"] = canonical_sha256(implementation)
    method = signal_audit_method_contract()
    contract: dict[str, object] = {
        "schema_version": 1,
        "contract_id": "foundry-l3-grpo-signal-audit-r1-v1",
        "starting_commit": STARTING_COMMIT,
        "base_model_revision": MODEL_REVISION,
        "dataset_sha256": "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31",
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
            "fixture_sha256": "ca7ea72ae288234eb769486f1a1dd0893f9c14b1c14487ae043336af30318199",
            "calibration_sha256": (
                "e0952f0034424f7817998300207ec3eecf2ce4f8443a87405899decff3fb65e7"
            ),
            "contract_sha256": "441933982c2b51b49195763440c318893cea22af947c9efc50b732d05fee7b61",
        },
        "reference_mechanism_sha256": (
            "674b368105f08b0e1eb00f54c6912f611da730ed070f60c63989de996ecb0316"
        ),
        **original,
        "prior_diagnostic_manifest_sha256": prior["prior_diagnostic_manifest_sha256"],
        "advantage_equivalence_contract_sha256": advantage["advantage_equivalence_contract_sha256"],
        "family_aggregation_contract_sha256": family["family_aggregation_contract_sha256"],
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
    return prior, advantage, family, implementation, contract


def _write_new_or_identical(path: Path, value: object) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"existing corrected signal-audit freeze differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def write_correction_freeze(
    root: Path,
    *,
    torch_module: Any,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    values = freeze_correction(root, torch_module=torch_module)
    output = root / "results/phase2_vetted_corpus"
    for name, value in zip(
        (
            PRIOR_MANIFEST_OUTPUT,
            ADVANTAGE_CONTRACT_OUTPUT,
            FAMILY_CONTRACT_OUTPUT,
            IMPLEMENTATION_OUTPUT,
            CONTRACT_OUTPUT,
        ),
        values,
        strict=True,
    ):
        _write_new_or_identical(output / name, value)
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    prior, advantage, family, implementation, contract = write_correction_freeze(
        args.root,
        torch_module=torch,
    )
    print(
        json.dumps(
            {
                "prior_diagnostic_manifest_sha256": prior["prior_diagnostic_manifest_sha256"],
                "advantage_equivalence_contract_sha256": advantage[
                    "advantage_equivalence_contract_sha256"
                ],
                "family_aggregation_contract_sha256": family["family_aggregation_contract_sha256"],
                "implementation_sha256": implementation["implementation_sha256"],
                "signal_audit_contract_sha256": contract["signal_audit_contract_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
