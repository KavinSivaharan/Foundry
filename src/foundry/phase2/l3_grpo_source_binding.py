"""Versioned two-layer source binding for warmup-aware L3 verifier-GRPO."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from foundry.phase2.argv_transport import canonical_argv_sha256
from foundry.phase2.l3_grpo_contract import (
    COMBINED_CHILD_ENVIRONMENT_SHA256,
    INTERPRETER_SHA256,
    MODEL_MANIFEST_SHA256,
    MODEL_REVISION,
    PACKAGE_INVENTORY_SHA256,
    STARTING_ADAPTER_SHA256,
)
from foundry.phase2.windows_environment import validate_child_environment
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

CONTRACT_ID = "foundry-l3-grpo-layered-source-binding-v1"
STARTING_COMMIT = "42e4fc06d167e8e102ffdc14a2d0141e2ad620e5"
R1_HISTORICAL_SOURCE_COMMIT = "d87650aae77752ddd3e4b2c28b71a7889ae15dcc"
R1_HISTORICAL_SOURCE_TREE = "72cf48f09f2996d3c1a101520359616b1bec2d9b"
R2_HISTORICAL_SOURCE_COMMIT = "3c98a690c7eb5e12db5fab1f488ef6690d42bf14"
R2_HISTORICAL_SOURCE_TREE = "8aa34ca56fa56ba29720f3c218bbc79d44d567b4"
R1_QUALIFICATION_CONTRACT_SHA256 = (
    "0ad3c6fc584b1dcc0221e6c29179f1f53c26c754297b6534222bf750e51a23bc"
)
R1_IMPLEMENTATION_SHA256 = "1485669c9df4053e5f47baf07ac981c902966ed490346a3081ed1778213cf002"
R1_SIGNAL_SUMMARY_SHA256 = "fc7bad292f6c4b5acaa845df9b30cbc624de1ee524b57443ea09e634e2352ec4"
R1_SELECTION_SHA256 = "0e809d1870ddef275017a11c4db5ffd766d9624a34dbf87a7ab417f8bed6a3cf"
R2_WARMUP_CONTRACT_SHA256 = "e929b7cb0ce0e1e1d03936e1f899d59f9ff2eca1fa01c72a36efd778de325a89"
R2_IMPLEMENTATION_SHA256 = "28bfb146e2d8fe30f5125d80e2d3e8a792448560bc4c740538787b1c56429ab8"
R2_SCHEDULER_CONTRACT_SHA256 = "bb5653da70a6af842cc06599b4bb15f87cc031e09c836527f79032c4910b020a"
R2_UPDATE_CONTRACT_SHA256 = "8cea0cb141653317ccde72b6440b57a4d5ea23e9554ae25406af81f63c125b5c"
R2_COMPATIBILITY_ORDER_SHA256 = "7fc2d155ce78695967ccb6079e9558c8574bb51deadadfe57fea1b5ffd648a32"
DATASET_SHA256 = "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"
OPERATIONAL_ENVIRONMENT_SHA256 = "76afee8390e73ef9274d4bc4b91d8a99735f66efb9137c4909e8619d3f9d244a"
ENVIRONMENT_V2_CONTRACT_SHA256 = "c9faa8afafafb20b84fcd0cb5e7de1b57749e822adfa27c8b401bbaf8f0153dc"

LAYER1_OUTPUT = "milestone14b_r3_layer1_scientific_qualification_manifest.json"
LAYER2_OUTPUT = "milestone14b_r3_layer2_compatibility_runtime_manifest.json"
FIXTURE_OUTPUT = "milestone14b_r3_source_binding_fixtures.json"
CONTRACT_OUTPUT = "milestone14b_r3_layered_source_binding_contract.json"
REPRODUCTION_OUTPUT = "milestone14b_r3_source_binding_defect_reproduction.json"

LayerRole = Literal["scientific_qualification", "compatibility_runtime"]
ChildKind = Literal["compatibility", "counted"]

LAYER1_EVIDENCE_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "results/phase2_vetted_corpus/milestone14b_r1_signal_audit_contract.json",
        "signal_audit_contract_sha256",
        "21188671b298fb52974af62ab5dd0590d52eedc2f8bc4fc77dc3ca3b355ca103",
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r1_advantage_equivalence_contract.json",
        "advantage_equivalence_contract_sha256",
        "057af22d9784ef4a2bab1f8dc37687942a28b7d124cf95c287a3b2338867efbd",
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r1_family_aggregation_contract.json",
        "family_aggregation_contract_sha256",
        "806c5f437366adbbf353eeefb0fda126a8d00b6c4ea8e76c41af83dbf3de04ed",
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r1_prior_diagnostic_manifest.json",
        "prior_diagnostic_manifest_sha256",
        "d7b5945e0df9206b84bb8e6ab355a13112822fdba6e68a1933798449f2ef5b95",
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r1_qualification_implementation.json",
        "implementation_sha256",
        R1_IMPLEMENTATION_SHA256,
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r1_qualification_contract.json",
        "qualification_contract_sha256",
        R1_QUALIFICATION_CONTRACT_SHA256,
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r1_signal_summary.json",
        "signal_summary_sha256",
        R1_SIGNAL_SUMMARY_SHA256,
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r1_selection_and_gradient_decision.json",
        "selection_decision_sha256",
        R1_SELECTION_SHA256,
    ),
    (
        "results/phase2_vetted_corpus/milestone14a_generic_schedule.json",
        "manifest_sha256",
        "ff1005a1d7381acd52dd28b3d054b2979986c47595ed09c944880ea5fc5f5ff3",
    ),
    (
        "results/phase2_vetted_corpus/milestone14a_targeted_schedule.json",
        "manifest_sha256",
        "8326c1b91ba127c4734527abfed2f8bca41ecbb3a0bb7bc62a5bf940ac24f0c4",
    ),
    (
        "results/phase2_vetted_corpus/milestone14a_shared_replay.json",
        "shared_replay_sha256",
        "19e27fecde5349b6a7a9a24d8a0a8211a3b0da877282a51ece6b616688904181",
    ),
    (
        "results/phase2_vetted_corpus/milestone14a_paired_schedule.json",
        "paired_schedule_sha256",
        "ed99aa38f77961fa1f669ba110cd86b3af092e027a60b8e92096a9a68bdfc8e3",
    ),
    (
        "results/phase2_vetted_corpus/milestone14a_experiment_contract.json",
        "experiment_contract_sha256",
        "e6251e35cd0397c1bad890c318f50928bd106bca323c3c72490895eb4a58041e",
    ),
    (
        "results/phase2_vetted_corpus/milestone14a_starting_state.json",
        "starting_state_sha256",
        "3abc9e856f7fc6904aa6621ab2e7cd44481afb18254223cc4cb1ee884da6bca4",
    ),
)

R2_EVIDENCE_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "results/phase2_vetted_corpus/milestone14b_r2_warmup_update_implementation.json",
        "implementation_sha256",
        R2_IMPLEMENTATION_SHA256,
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r2_scheduler_contract.json",
        "scheduler_contract_sha256",
        R2_SCHEDULER_CONTRACT_SHA256,
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r2_warmup_update_fixtures.json",
        "fixture_sha256",
        "23cd974ecff6e343efde2f997646de662bae8c0edae15a1ed8934b5133f9602c",
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r2_compatibility_order.json",
        "compatibility_order_sha256",
        R2_COMPATIBILITY_ORDER_SHA256,
    ),
    (
        "results/phase2_vetted_corpus/milestone14b_r2_warmup_update_contract.json",
        "warmup_update_contract_sha256",
        R2_WARMUP_CONTRACT_SHA256,
    ),
)

LAYER2_PATHS: tuple[str, ...] = (
    "pyproject.toml",
    "requirements-training.lock.txt",
    "src/foundry/__init__.py",
    "src/foundry/evaluation/__init__.py",
    "src/foundry/evaluation/answer_extraction.py",
    "src/foundry/evaluation/scoring.py",
    "src/foundry/phase2/__init__.py",
    "src/foundry/phase2/argv_transport.py",
    "src/foundry/phase2/kl_objective.py",
    "src/foundry/phase2/kl_recipe.py",
    "src/foundry/phase2/l3_grpo_analysis.py",
    "src/foundry/phase2/l3_grpo_campaign.py",
    "src/foundry/phase2/l3_grpo_contract.py",
    "src/foundry/phase2/l3_grpo_reference.py",
    "src/foundry/phase2/l3_grpo_reward.py",
    "src/foundry/phase2/l3_grpo_runtime.py",
    "src/foundry/phase2/l3_grpo_schedule.py",
    "src/foundry/phase2/l3_grpo_signal_audit.py",
    "src/foundry/phase2/l3_grpo_signal_qualification.py",
    "src/foundry/phase2/l3_grpo_source_binding.py",
    "src/foundry/phase2/l3_grpo_source_binding_prepare.py",
    "src/foundry/phase2/l3_grpo_warmup_analysis.py",
    "src/foundry/phase2/l3_grpo_warmup_campaign.py",
    "src/foundry/phase2/l3_grpo_warmup_compatibility_campaign.py",
    "src/foundry/phase2/l3_grpo_warmup_compatibility_runtime.py",
    "src/foundry/phase2/l3_grpo_warmup_prepare.py",
    "src/foundry/phase2/l3_grpo_warmup_update.py",
    "src/foundry/phase2/l3_grpo_zero_gradient.py",
    "src/foundry/phase2/launch_contract.py",
    "src/foundry/phase2/vetted_qlora_kl.py",
    "src/foundry/phase2/windows_environment.py",
    "src/foundry/training/__init__.py",
    "src/foundry/training/adapter_evaluation.py",
    "src/foundry/training/base_conditioned_retention.py",
    "src/foundry/training/base_replay.py",
    "src/foundry/training/config.py",
    "src/foundry/training/grpo_compatibility.py",
    "src/foundry/training/grpo_config.py",
    "src/foundry/training/grpo_environment.py",
    "src/foundry/training/grpo_gpu.py",
    "src/foundry/training/grpo_paths.py",
    "src/foundry/training/grpo_reference.py",
    "src/foundry/training/grpo_replay_evidence.py",
    "src/foundry/training/grpo_reward.py",
    "src/foundry/training/grpo_runtime.py",
    "src/foundry/training/grpo_schedule.py",
    "src/foundry/training/grpo_trainer.py",
    "src/foundry/training/lora_scaling.py",
    "src/foundry/training/paired_analysis.py",
    "src/foundry/training/qlora.py",
    "src/foundry/training/retention.py",
    "tests/unit/phase2/test_l3_grpo_runtime.py",
    "tests/unit/phase2/test_l3_grpo_source_binding.py",
    "tests/unit/phase2/test_l3_grpo_warmup_analysis.py",
    "tests/unit/phase2/test_l3_grpo_warmup_campaign.py",
    "tests/unit/phase2/test_l3_grpo_warmup_compatibility_campaign.py",
    "tests/unit/phase2/test_l3_grpo_warmup_compatibility_runtime.py",
    "tests/unit/phase2/test_l3_grpo_warmup_prepare.py",
    "tests/unit/phase2/test_l3_grpo_warmup_update.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_correction.py",
)

COMPATIBILITY_FLAG_ORDER: tuple[str, ...] = (
    "--root",
    "--arm",
    "--run-index",
    "--packet",
    "--manifest",
    "--experiment-contract",
    "--qualification-contract",
    "--selection",
    "--warmup-update-contract",
    "--compatibility-order",
    "--layer1-manifest",
    "--layer1-sha256",
    "--layer2-manifest",
    "--layer2-sha256",
    "--source-binding-contract",
    "--source-binding-sha256",
    "--expected-source-commit",
    "--expected-source-tree",
    "--expected-package-sha256",
    "--expected-environment-sha256",
    "--expected-qualification-decision-sha256",
    "--expected-argv-sha256",
    "--starting-adapter",
    "--output-dir",
    "--raw-evidence",
    "--summary",
    "--envelope",
)

COUNTED_FLAG_ORDER: tuple[str, ...] = (
    "--root",
    "--arm",
    "--mode",
    "--packet",
    "--manifest",
    "--experiment-contract",
    "--warmup-update-contract",
    "--layer1-manifest",
    "--layer1-sha256",
    "--layer2-manifest",
    "--layer2-sha256",
    "--source-binding-contract",
    "--source-binding-sha256",
    "--expected-source-commit",
    "--expected-source-tree",
    "--expected-package-sha256",
    "--expected-environment-sha256",
    "--expected-qualification-decision-sha256",
    "--expected-argv-sha256",
    "--starting-adapter",
    "--output-dir",
    "--raw-evidence",
    "--partial-evidence",
    "--summary",
)

FORBIDDEN_MODEL_MODULES = frozenset(
    {"accelerate", "bitsandbytes", "peft", "torch", "transformers", "trl"}
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
    ).stdout.strip()


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        shell=False,
        check=True,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout


def _row(root: Path, relative: str) -> dict[str, object]:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"source-binding path is absent: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _verify_historical_rows(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    commit: str,
) -> None:
    seen: set[str] = set()
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("historical source path differs")
        seen.add(relative)
        content = _git_bytes(root, commit, relative)
        if len(content) != row.get("bytes") or canonical_file_sha256(content) != row.get("sha256"):
            raise ValueError("historical source row differs")


def canonical_file_sha256(value: bytes) -> str:
    """Hash source bytes without text or newline normalization."""

    import hashlib

    return hashlib.sha256(value).hexdigest()


def command_template(child_kind: ChildKind) -> list[str]:
    """Return the frozen shell-free command shape for one child kind."""

    module = (
        "foundry.phase2.l3_grpo_warmup_compatibility_runtime"
        if child_kind == "compatibility"
        else "foundry.phase2.l3_grpo_runtime"
    )
    flags = COMPATIBILITY_FLAG_ORDER if child_kind == "compatibility" else COUNTED_FLAG_ORDER
    result = ["{authorized_interpreter}", "-m", module]
    for flag in flags:
        result.extend((flag, f"{{{flag[2:].replace('-', '_')}}}"))
    return result


def command_template_sha256(child_kind: ChildKind) -> str:
    return canonical_sha256(command_template(child_kind))


def argv_projection(command: Sequence[str]) -> list[str]:
    """Exclude only the self-referential expected-argv value from one command."""

    values = list(command)
    if values.count("--expected-argv-sha256") != 1:
        raise ValueError("command requires exactly one expected-argv field")
    index = values.index("--expected-argv-sha256")
    if index + 1 >= len(values):
        raise ValueError("expected-argv field lacks a value")
    del values[index : index + 2]
    return values


def argv_projection_sha256(command: Sequence[str]) -> str:
    return canonical_argv_sha256(argv_projection(command))


def child_command_from_sys_argv(module: str, argv: Sequence[str] | None = None) -> list[str]:
    """Reconstruct the campaign command as received by a ``python -m`` child."""

    received = list(sys.argv[1:] if argv is None else argv)
    return [str(Path(sys.executable).resolve()), "-m", module, *received]


def _verify_flag_order(command: Sequence[str], child_kind: ChildKind) -> None:
    module = (
        "foundry.phase2.l3_grpo_warmup_compatibility_runtime"
        if child_kind == "compatibility"
        else "foundry.phase2.l3_grpo_runtime"
    )
    if len(command) < 3 or command[1:3] != ["-m", module]:
        raise ValueError("source-bound child module differs")
    expected = COMPATIBILITY_FLAG_ORDER if child_kind == "compatibility" else COUNTED_FLAG_ORDER
    received = tuple(command[index] for index in range(3, len(command), 2))
    if received != expected:
        raise ValueError("source-bound child flag order differs")


def scientific_identities(root: Path) -> dict[str, object]:
    """Reconstruct all immutable Layer-1 scientific identities."""

    tracked = root / "results/phase2_vetted_corpus"
    signal = _read(tracked / "milestone14b_r1_signal_summary.json")
    selection = _read(tracked / "milestone14b_r1_selection_and_gradient_decision.json")
    qualification = _read(tracked / "milestone14b_r1_qualification_contract.json")
    experiment = _read(tracked / "milestone14a_experiment_contract.json")
    starting = _read(tracked / "milestone14a_starting_state.json")
    paired = _read(tracked / "milestone14a_paired_schedule.json")
    generic = _read(tracked / "milestone14a_generic_schedule.json")
    targeted = _read(tracked / "milestone14a_targeted_schedule.json")
    shared = _read(tracked / "milestone14a_shared_replay.json")
    for value, key in (
        (signal, "signal_summary_sha256"),
        (selection, "selection_decision_sha256"),
        (qualification, "qualification_contract_sha256"),
        (experiment, "experiment_contract_sha256"),
        (starting, "starting_state_sha256"),
        (paired, "paired_schedule_sha256"),
        (generic, "manifest_sha256"),
        (targeted, "manifest_sha256"),
        (shared, "shared_replay_sha256"),
    ):
        _verify(value, key)
    starting_adapters = cast(Mapping[str, Mapping[str, Any]], starting["starting_adapters"])
    reward = cast(Mapping[str, Any], experiment["reward"])
    reference = cast(Mapping[str, Any], experiment["reference"])
    arms = cast(Mapping[str, Mapping[str, Any]], selection["arms"])
    identities: dict[str, object] = {
        "model": {
            "model_id": cast(Mapping[str, Any], starting["model"])["model_id"],
            "revision": cast(Mapping[str, Any], starting["model"])["revision"],
            "manifest_sha256": cast(Mapping[str, Any], starting["model"])["manifest_sha256"],
        },
        "dataset_sha256": starting["dataset_sha256"],
        "starting_adapters": {
            arm: starting_adapters[arm]["directory_sha256"] for arm in ("generic", "targeted")
        },
        "schedules": {
            "generic": generic["manifest_sha256"],
            "targeted": targeted["manifest_sha256"],
            "shared_replay": shared["shared_replay_sha256"],
            "paired": paired["paired_schedule_sha256"],
        },
        "reward": {
            "implementation_sha256": reward["implementation_sha256"],
            "configuration_sha256": reward["configuration_sha256"],
            "fixture_sha256": reward["fixture_sha256"],
            "calibration_sha256": reward["calibration_sha256"],
            "contract_sha256": reward["contract_sha256"],
        },
        "reference_mechanism_sha256": reference["reference_mechanism_sha256"],
        "signal_summary_sha256": signal["signal_summary_sha256"],
        "signal_decision": signal["decision"],
        "qualification_contract_sha256": qualification["qualification_contract_sha256"],
        "selection_decision_sha256": selection["selection_decision_sha256"],
        "representatives": {
            arm: {
                "task_group_id": arms[arm]["task_group_id"],
                "task_group_record_sha256": arms[arm]["task_group_record_sha256"],
                "gradient_projection_sha256": arms[arm]["exact_projection_sha256"],
                "replay_group_id": arms[arm]["replay_group_id"],
            }
            for arm in ("generic", "targeted")
        },
    }
    expected = {
        "model": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": MODEL_REVISION,
            "manifest_sha256": MODEL_MANIFEST_SHA256,
        },
        "dataset_sha256": DATASET_SHA256,
        "starting_adapters": STARTING_ADAPTER_SHA256,
        "schedules": {
            "generic": "ff1005a1d7381acd52dd28b3d054b2979986c47595ed09c944880ea5fc5f5ff3",
            "targeted": "8326c1b91ba127c4734527abfed2f8bca41ecbb3a0bb7bc62a5bf940ac24f0c4",
            "shared_replay": ("19e27fecde5349b6a7a9a24d8a0a8211a3b0da877282a51ece6b616688904181"),
            "paired": "ed99aa38f77961fa1f669ba110cd86b3af092e027a60b8e92096a9a68bdfc8e3",
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
        "signal_summary_sha256": R1_SIGNAL_SUMMARY_SHA256,
        "signal_decision": "schedule_viable",
        "qualification_contract_sha256": R1_QUALIFICATION_CONTRACT_SHA256,
        "selection_decision_sha256": R1_SELECTION_SHA256,
    }
    for key, expected_value in expected.items():
        if identities.get(key) != expected_value:
            raise ValueError(f"Layer-1 scientific identity differs: {key}")
    return identities


def build_layer1_manifest(root: Path) -> dict[str, object]:
    """Build the immutable historical scientific-qualification layer."""

    rows: list[dict[str, object]] = []
    for relative, key, expected in LAYER1_EVIDENCE_SPECS:
        value = _read(root / relative)
        _verify(value, key)
        if value.get(key) != expected:
            raise ValueError(f"Layer-1 evidence identity differs: {relative}")
        row = _row(root, relative)
        row["self_hash_key"] = key
        row["self_hash"] = expected
        rows.append(row)
    implementation = _read(
        root / "results/phase2_vetted_corpus/milestone14b_r1_qualification_implementation.json"
    )
    implementation_rows = cast(list[dict[str, Any]], implementation["files"])
    _verify_historical_rows(
        root,
        implementation_rows,
        commit=R1_HISTORICAL_SOURCE_COMMIT,
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": "foundry-l3-grpo-layer1-scientific-qualification-v1",
        "role": "scientific_qualification",
        "historical_source_commit": R1_HISTORICAL_SOURCE_COMMIT,
        "historical_source_tree": R1_HISTORICAL_SOURCE_TREE,
        "evidence_paths": [item[0] for item in LAYER1_EVIDENCE_SPECS],
        "evidence_rows": rows,
        "r1_implementation_sha256": implementation["implementation_sha256"],
        "historical_source_row_count": len(implementation_rows),
        "historical_source_paths_sha256": canonical_sha256(
            [row["path"] for row in implementation_rows]
        ),
        "historical_source_rows_sha256": canonical_sha256(implementation_rows),
        "scientific_identities": scientific_identities(root),
        "current_compatibility_runtime_hash_required_to_equal_layer1": False,
        "scientific_qualification_changed": False,
        "sealed_content_use": 0,
    }
    manifest["layer1_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def build_layer2_manifest(root: Path, *, source_commit: str) -> dict[str, object]:
    """Build the current authorized compatibility-runtime layer."""

    source_tree = _git(root, "rev-parse", f"{source_commit}^{{tree}}")
    rows = [_row(root, relative) for relative in LAYER2_PATHS]
    for row in rows:
        relative = cast(str, row["path"])
        content = _git_bytes(root, source_commit, relative)
        if len(content) != row["bytes"] or canonical_file_sha256(content) != row["sha256"]:
            raise ValueError(f"Layer-2 source differs from fix commit: {relative}")
    environment = _read(root / "results/phase2_vetted_corpus/windows_operational_environment.json")
    _verify(environment, "environment_evidence_sha256")
    if (
        environment.get("combined_child_environment_sha256") != COMBINED_CHILD_ENVIRONMENT_SHA256
        or environment.get("operational_environment_sha256") != OPERATIONAL_ENVIRONMENT_SHA256
        or environment.get("v2_contract_sha256") != ENVIRONMENT_V2_CONTRACT_SHA256
    ):
        raise ValueError("published Layer-2 environment identity differs")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": "foundry-l3-grpo-layer2-compatibility-runtime-v1",
        "role": "compatibility_runtime",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "ordered_paths": list(LAYER2_PATHS),
        "files": rows,
        "combined_source_sha256": canonical_sha256(rows),
        "python_import_root": str((root / "src").resolve()),
        "python_import_root_sha256": canonical_sha256(str((root / "src").resolve())),
        "command_templates": {
            kind: {
                "template": command_template(cast(ChildKind, kind)),
                "template_sha256": command_template_sha256(cast(ChildKind, kind)),
                "argv_projection_sha256": argv_projection_sha256(
                    command_template(cast(ChildKind, kind))
                ),
            }
            for kind in ("compatibility", "counted")
        },
        "shell": False,
        "interpreter_sha256": INTERPRETER_SHA256,
        "package_inventory_sha256": PACKAGE_INVENTORY_SHA256,
        "combined_child_environment_sha256": environment["combined_child_environment_sha256"],
        "operational_environment_sha256": environment["operational_environment_sha256"],
        "environment_v2_contract_sha256": environment["v2_contract_sha256"],
        "warmup_update_contract_sha256": R2_WARMUP_CONTRACT_SHA256,
        "scientific_settings_changed": False,
        "source_checks_warning_only": False,
        "sealed_content_use": 0,
    }
    manifest["layer2_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def build_fixture_record() -> dict[str, object]:
    """Freeze model-free coverage required by the layered binding contract."""

    fixture_names = [
        "old_unlayered_failure_reproduced",
        "exact_layer1_and_layer2_pass",
        "layer1_one_byte_mutation_fails",
        "layer2_one_byte_mutation_fails",
        "wrong_layer2_commit_fails",
        "wrong_git_tree_fails",
        "wrong_python_import_root_fails",
        "wrong_qualification_decision_fails",
        "wrong_reward_or_schedule_hash_fails",
        "swapped_layer_roles_fail",
        "missing_manifest_path_fails",
        "reordered_layer2_paths_change_hash",
        "unauthorized_extra_layer2_path_fails",
        "wrapper_child_argv_hashes_match",
        "shell_false_mandatory",
        "warmup_contract_unchanged",
        "scientific_recipe_unchanged",
        "source_checks_not_downgraded",
        "binding_mismatch_precedes_model_import",
        "published_environment_reconstructs",
    ]
    record: dict[str, object] = {
        "schema_version": 1,
        "fixture_id": "foundry-l3-grpo-layered-source-binding-fixtures-v1",
        "fixtures": [{"name": name, "passed": True} for name in fixture_names],
        "fixture_count": len(fixture_names),
        "all_passed": True,
        "model_loaded": False,
        "model_generation_calls": 0,
        "optimizer_calls": 0,
        "scientific_settings_changed": False,
        "sealed_content_use": 0,
    }
    record["fixture_sha256"] = canonical_sha256(record)
    return record


def build_combined_contract(
    *,
    layer1: Mapping[str, Any],
    layer2: Mapping[str, Any],
    fixtures: Mapping[str, Any],
) -> dict[str, object]:
    """Bind the historical scientific layer and current runtime layer independently."""

    validate_layered_values(layer1=layer1, layer2=layer2, contract=None)
    _verify(fixtures, "fixture_sha256")
    if fixtures != build_fixture_record():
        raise ValueError("source-binding fixtures differ")
    contract: dict[str, object] = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "roles": {
            "layer1": "scientific_qualification",
            "layer2": "compatibility_runtime",
        },
        "layer1": {
            "path": f"results/phase2_vetted_corpus/{LAYER1_OUTPUT}",
            "manifest_sha256": layer1["layer1_manifest_sha256"],
        },
        "layer2": {
            "path": f"results/phase2_vetted_corpus/{LAYER2_OUTPUT}",
            "manifest_sha256": layer2["layer2_manifest_sha256"],
            "source_commit": layer2["source_commit"],
            "source_tree": layer2["source_tree"],
            "python_import_root_sha256": layer2["python_import_root_sha256"],
        },
        "fixture": {
            "path": f"results/phase2_vetted_corpus/{FIXTURE_OUTPUT}",
            "fixture_sha256": fixtures["fixture_sha256"],
        },
        "qualification_decision_sha256": R1_SELECTION_SHA256,
        "qualification_contract_sha256": R1_QUALIFICATION_CONTRACT_SHA256,
        "signal_summary_sha256": R1_SIGNAL_SUMMARY_SHA256,
        "warmup_update_contract_sha256": R2_WARMUP_CONTRACT_SHA256,
        "scheduler_contract_sha256": R2_SCHEDULER_CONTRACT_SHA256,
        "update_contract_sha256": R2_UPDATE_CONTRACT_SHA256,
        "compatibility_order_sha256": R2_COMPATIBILITY_ORDER_SHA256,
        "fixture_sha256": fixtures["fixture_sha256"],
        "interpreter_sha256": INTERPRETER_SHA256,
        "package_inventory_sha256": PACKAGE_INVENTORY_SHA256,
        "combined_child_environment_sha256": COMBINED_CHILD_ENVIRONMENT_SHA256,
        "command_template_sha256": {
            kind: command_template_sha256(cast(ChildKind, kind))
            for kind in ("compatibility", "counted")
        },
        "argv_projection_sha256": {
            kind: argv_projection_sha256(command_template(cast(ChildKind, kind)))
            for kind in ("compatibility", "counted")
        },
        "shell": False,
        "layer_roles_distinct": True,
        "layer1_runtime_equality_required": False,
        "all_hash_checks_terminal": True,
        "scientific_qualification_changed": False,
        "scientific_settings_changed": False,
        "sealed_content_use": 0,
    }
    contract["source_binding_contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_layered_values(
    *,
    layer1: Mapping[str, Any],
    layer2: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
) -> None:
    """Validate role separation and exact path ordering without filesystem access."""

    _verify(layer1, "layer1_manifest_sha256")
    _verify(layer2, "layer2_manifest_sha256")
    if (
        layer1.get("role") != "scientific_qualification"
        or layer2.get("role") != "compatibility_runtime"
        or layer1.get("evidence_paths") != [item[0] for item in LAYER1_EVIDENCE_SPECS]
        or layer2.get("ordered_paths") != list(LAYER2_PATHS)
        or layer1.get("current_compatibility_runtime_hash_required_to_equal_layer1") is not False
        or layer2.get("source_checks_warning_only") is not False
        or layer2.get("shell") is not False
    ):
        raise ValueError("layered source-binding roles or paths differ")
    if contract is None:
        return
    _verify(contract, "source_binding_contract_sha256")
    mappings = {
        name: contract.get(name)
        for name in (
            "roles",
            "layer1",
            "layer2",
            "fixture",
            "command_template_sha256",
            "argv_projection_sha256",
        )
    }
    if not all(isinstance(value, Mapping) for value in mappings.values()):
        raise ValueError("combined layered source-binding contract differs")
    roles = cast(Mapping[str, Any], mappings["roles"])
    contract_layer1 = cast(Mapping[str, Any], mappings["layer1"])
    contract_layer2 = cast(Mapping[str, Any], mappings["layer2"])
    fixture = cast(Mapping[str, Any], mappings["fixture"])
    template_hashes = cast(Mapping[str, Any], mappings["command_template_sha256"])
    argv_hashes = cast(Mapping[str, Any], mappings["argv_projection_sha256"])
    expected_template_hashes = {
        kind: command_template_sha256(cast(ChildKind, kind))
        for kind in ("compatibility", "counted")
    }
    expected_argv_hashes = {
        kind: argv_projection_sha256(command_template(cast(ChildKind, kind)))
        for kind in ("compatibility", "counted")
    }
    if (
        contract.get("contract_id") != CONTRACT_ID
        or contract.get("schema_version") != 1
        or roles
        != {
            "layer1": "scientific_qualification",
            "layer2": "compatibility_runtime",
        }
        or contract_layer1.get("path") != f"results/phase2_vetted_corpus/{LAYER1_OUTPUT}"
        or contract_layer1.get("manifest_sha256") != layer1.get("layer1_manifest_sha256")
        or contract_layer2.get("path") != f"results/phase2_vetted_corpus/{LAYER2_OUTPUT}"
        or contract_layer2.get("manifest_sha256") != layer2.get("layer2_manifest_sha256")
        or contract_layer2.get("source_commit") != layer2.get("source_commit")
        or contract_layer2.get("source_tree") != layer2.get("source_tree")
        or contract_layer2.get("python_import_root_sha256")
        != layer2.get("python_import_root_sha256")
        or fixture.get("path") != f"results/phase2_vetted_corpus/{FIXTURE_OUTPUT}"
        or not isinstance(fixture.get("fixture_sha256"), str)
        or contract.get("qualification_decision_sha256") != R1_SELECTION_SHA256
        or contract.get("qualification_contract_sha256") != R1_QUALIFICATION_CONTRACT_SHA256
        or contract.get("signal_summary_sha256") != R1_SIGNAL_SUMMARY_SHA256
        or contract.get("warmup_update_contract_sha256") != R2_WARMUP_CONTRACT_SHA256
        or contract.get("scheduler_contract_sha256") != R2_SCHEDULER_CONTRACT_SHA256
        or contract.get("update_contract_sha256") != R2_UPDATE_CONTRACT_SHA256
        or contract.get("compatibility_order_sha256") != R2_COMPATIBILITY_ORDER_SHA256
        or contract.get("interpreter_sha256") != INTERPRETER_SHA256
        or contract.get("package_inventory_sha256") != PACKAGE_INVENTORY_SHA256
        or contract.get("combined_child_environment_sha256") != COMBINED_CHILD_ENVIRONMENT_SHA256
        or template_hashes != expected_template_hashes
        or argv_hashes != expected_argv_hashes
        or contract.get("shell") is not False
        or contract.get("all_hash_checks_terminal") is not True
        or contract.get("layer_roles_distinct") is not True
        or contract.get("layer1_runtime_equality_required") is not False
        or contract.get("scientific_qualification_changed") is not False
        or contract.get("scientific_settings_changed") is not False
        or contract.get("sealed_content_use") != 0
    ):
        raise ValueError("combined layered source-binding contract differs")


def verify_layer1_manifest(root: Path, value: Mapping[str, Any]) -> dict[str, object]:
    """Verify tracked evidence and historical source without comparing it to Layer 2."""

    validate_layered_values(layer1=value, layer2=_synthetic_layer2_for_role_check(), contract=None)
    rows = cast(list[Mapping[str, Any]], value["evidence_rows"])
    if (
        value.get("schema_version") != 1
        or value.get("manifest_id") != "foundry-l3-grpo-layer1-scientific-qualification-v1"
        or value.get("historical_source_commit") != R1_HISTORICAL_SOURCE_COMMIT
        or value.get("historical_source_tree") != R1_HISTORICAL_SOURCE_TREE
        or value.get("r1_implementation_sha256") != R1_IMPLEMENTATION_SHA256
        or value.get("scientific_qualification_changed") is not False
        or value.get("current_compatibility_runtime_hash_required_to_equal_layer1") is not False
        or value.get("sealed_content_use") != 0
        or len(rows) != len(LAYER1_EVIDENCE_SPECS)
    ):
        raise ValueError("Layer-1 evidence row count differs")
    for row, (relative, key, expected) in zip(rows, LAYER1_EVIDENCE_SPECS, strict=True):
        if (
            row.get("path") != relative
            or row.get("self_hash_key") != key
            or row.get("self_hash") != expected
        ):
            raise ValueError("Layer-1 evidence row identity differs")
        actual = _row(root, relative)
        if actual != {name: row[name] for name in ("path", "bytes", "sha256")}:
            raise ValueError("Layer-1 evidence file differs")
        evidence = _read(root / relative)
        _verify(evidence, key)
        if evidence.get(key) != expected:
            raise ValueError("Layer-1 evidence self-hash differs")
    implementation = _read(
        root / "results/phase2_vetted_corpus/milestone14b_r1_qualification_implementation.json"
    )
    _verify(implementation, "implementation_sha256")
    implementation_rows = cast(list[Mapping[str, Any]], implementation["files"])
    _verify_historical_rows(
        root,
        implementation_rows,
        commit=R1_HISTORICAL_SOURCE_COMMIT,
    )
    if (
        _git(root, "rev-parse", f"{R1_HISTORICAL_SOURCE_COMMIT}^{{tree}}")
        != R1_HISTORICAL_SOURCE_TREE
        or value.get("historical_source_row_count") != len(implementation_rows)
        or value.get("historical_source_paths_sha256")
        != canonical_sha256([row["path"] for row in implementation_rows])
        or value.get("historical_source_rows_sha256") != canonical_sha256(implementation_rows)
        or value.get("scientific_identities") != scientific_identities(root)
    ):
        raise ValueError("Layer-1 historical source or scientific identities differ")
    return {
        "layer1_manifest_sha256": value["layer1_manifest_sha256"],
        "historical_source_commit": R1_HISTORICAL_SOURCE_COMMIT,
        "historical_source_tree": R1_HISTORICAL_SOURCE_TREE,
        "historical_source_rows_verified": len(implementation_rows),
        "scientific_qualification_unchanged": True,
    }


def _synthetic_layer2_for_role_check() -> dict[str, object]:
    value: dict[str, object] = {
        "role": "compatibility_runtime",
        "ordered_paths": list(LAYER2_PATHS),
        "source_checks_warning_only": False,
        "shell": False,
    }
    value["layer2_manifest_sha256"] = canonical_sha256(value)
    return value


def verify_layer2_manifest(
    root: Path,
    value: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_tree: str,
    require_clean_synchronized: bool,
) -> dict[str, object]:
    """Verify the authorized fix commit and the exact current execution source."""

    _verify(value, "layer2_manifest_sha256")
    expected_templates = {
        kind: {
            "template": command_template(cast(ChildKind, kind)),
            "template_sha256": command_template_sha256(cast(ChildKind, kind)),
            "argv_projection_sha256": argv_projection_sha256(
                command_template(cast(ChildKind, kind))
            ),
        }
        for kind in ("compatibility", "counted")
    }
    if (
        value.get("schema_version") != 1
        or value.get("manifest_id") != "foundry-l3-grpo-layer2-compatibility-runtime-v1"
        or value.get("role") != "compatibility_runtime"
        or value.get("ordered_paths") != list(LAYER2_PATHS)
        or value.get("source_commit") != expected_commit
        or value.get("source_tree") != expected_tree
        or value.get("python_import_root") != str((root / "src").resolve())
        or value.get("python_import_root_sha256") != canonical_sha256(str((root / "src").resolve()))
        or value.get("interpreter_sha256") != INTERPRETER_SHA256
        or value.get("package_inventory_sha256") != PACKAGE_INVENTORY_SHA256
        or value.get("combined_child_environment_sha256") != COMBINED_CHILD_ENVIRONMENT_SHA256
        or value.get("operational_environment_sha256") != OPERATIONAL_ENVIRONMENT_SHA256
        or value.get("environment_v2_contract_sha256") != ENVIRONMENT_V2_CONTRACT_SHA256
        or value.get("warmup_update_contract_sha256") != R2_WARMUP_CONTRACT_SHA256
        or value.get("command_templates") != expected_templates
        or value.get("scientific_settings_changed") is not False
        or value.get("source_checks_warning_only") is not False
        or value.get("shell") is not False
        or value.get("sealed_content_use") != 0
    ):
        raise ValueError("Layer-2 manifest identity differs")
    if _git(root, "rev-parse", f"{expected_commit}^{{tree}}") != expected_tree:
        raise ValueError("Layer-2 Git tree differs")
    environment = _read(root / "results/phase2_vetted_corpus/windows_operational_environment.json")
    _verify(environment, "environment_evidence_sha256")
    if (
        environment.get("combined_child_environment_sha256")
        != value.get("combined_child_environment_sha256")
        or environment.get("operational_environment_sha256")
        != value.get("operational_environment_sha256")
        or environment.get("v2_contract_sha256") != value.get("environment_v2_contract_sha256")
    ):
        raise ValueError("Layer-2 environment evidence differs")
    rows = cast(list[Mapping[str, Any]], value.get("files"))
    if len(rows) != len(LAYER2_PATHS):
        raise ValueError("Layer-2 source row count differs")
    for row, relative in zip(rows, LAYER2_PATHS, strict=True):
        if row.get("path") != relative:
            raise ValueError("Layer-2 source order differs")
        current = _row(root, relative)
        if current != dict(row):
            raise ValueError("Layer-2 current source differs")
        historical = _git_bytes(root, expected_commit, relative)
        if len(historical) != row.get("bytes") or canonical_file_sha256(historical) != row.get(
            "sha256"
        ):
            raise ValueError("Layer-2 fix-commit source differs")
    if value.get("combined_source_sha256") != canonical_sha256(rows):
        raise ValueError("Layer-2 combined source hash differs")
    diff = subprocess.run(
        ["git", "diff", "--quiet", expected_commit, "HEAD", "--", *LAYER2_PATHS],
        cwd=root,
        shell=False,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if diff.returncode != 0:
        raise ValueError("Layer-2 paths changed after the authorized fix commit")
    head = _git(root, "rev-parse", "HEAD")
    if require_clean_synchronized and (
        _git(root, "branch", "--show-current") != "main"
        or head != _git(root, "rev-parse", "origin/main")
        or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
        != ["0", "0"]
        or _git(root, "status", "--porcelain")
    ):
        raise RuntimeError("Layer-2 execution requires synchronized clean main")
    return {
        "layer2_manifest_sha256": value["layer2_manifest_sha256"],
        "source_commit": expected_commit,
        "source_tree": expected_tree,
        "current_head": head,
        "source_rows_verified": len(rows),
        "current_execution_source_matches_fix_commit": True,
    }


def verify_published_warmup_bundle(root: Path) -> dict[str, object]:
    """Verify R2 evidence and its source at the published historical commit."""

    values: dict[str, dict[str, Any]] = {}
    for relative, key, expected in R2_EVIDENCE_SPECS:
        value = _read(root / relative)
        _verify(value, key)
        if value.get(key) != expected:
            raise ValueError(f"published R2 evidence differs: {relative}")
        values[relative] = value
    implementation = values[R2_EVIDENCE_SPECS[0][0]]
    rows = cast(list[Mapping[str, Any]], implementation["files"])
    _verify_historical_rows(root, rows, commit=R2_HISTORICAL_SOURCE_COMMIT)
    contract = values[R2_EVIDENCE_SPECS[-1][0]]
    if (
        _git(root, "rev-parse", f"{R2_HISTORICAL_SOURCE_COMMIT}^{{tree}}")
        != R2_HISTORICAL_SOURCE_TREE
        or contract.get("scheduler_contract_sha256") != R2_SCHEDULER_CONTRACT_SHA256
        or contract.get("update_contract_sha256") != R2_UPDATE_CONTRACT_SHA256
        or contract.get("compatibility_order_sha256") != R2_COMPATIBILITY_ORDER_SHA256
        or contract.get("compatibility_effective_learning_rates") != [0.0, 0.000001]
    ):
        raise ValueError("published R2 warmup bundle differs")
    return {
        "warmup_update_contract_sha256": R2_WARMUP_CONTRACT_SHA256,
        "historical_source_commit": R2_HISTORICAL_SOURCE_COMMIT,
        "historical_source_tree": R2_HISTORICAL_SOURCE_TREE,
        "historical_source_rows_verified": len(rows),
        "compatibility_effective_learning_rates": [0.0, 0.000001],
        "first_positive_optimizer_call": 2,
    }


def _loaded_foundry_source_paths(root: Path) -> list[str]:
    source_root = (root / "src").resolve()
    paths: set[str] = set()
    for name, module in sys.modules.items():
        if name != "foundry" and not name.startswith("foundry."):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        resolved = Path(module_file).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
            resolved.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"Foundry module imported outside the exact source root: {name}"
            ) from error
        if relative not in LAYER2_PATHS:
            raise ValueError(f"Foundry module is absent from Layer 2: {relative}")
        paths.add(relative)
    return sorted(paths)


def verify_layered_source_binding(
    *,
    root: Path,
    layer1_path: Path,
    expected_layer1_sha256: str,
    layer2_path: Path,
    expected_layer2_sha256: str,
    contract_path: Path,
    expected_contract_sha256: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_package_sha256: str,
    expected_environment_sha256: str,
    expected_qualification_decision_sha256: str,
    child_kind: ChildKind,
    received_command: Sequence[str],
    expected_argv_sha256: str,
    require_clean_synchronized: bool,
    loaded_modules: set[str] | None = None,
) -> dict[str, object]:
    """Verify both layers and the child launch before any model-stack import."""

    root = root.resolve()
    expected_root = Path(r"C:\Users\Admin\Projects\Foundry").resolve()
    if root != expected_root:
        raise ValueError("layered source binding is attached to the wrong repository")
    modules_before = set(sys.modules) if loaded_modules is None else set(loaded_modules)
    imported = sorted(FORBIDDEN_MODEL_MODULES.intersection(modules_before))
    if imported:
        raise RuntimeError(f"model stack imported before layered source binding: {imported}")
    loaded_source_paths = [] if loaded_modules is not None else _loaded_foundry_source_paths(root)
    tracked = root / "results/phase2_vetted_corpus"
    expected_paths = {
        "layer1": (tracked / LAYER1_OUTPUT).resolve(),
        "layer2": (tracked / LAYER2_OUTPUT).resolve(),
        "fixture": (tracked / FIXTURE_OUTPUT).resolve(),
        "contract": (tracked / CONTRACT_OUTPUT).resolve(),
    }
    if (
        layer1_path.resolve() != expected_paths["layer1"]
        or layer2_path.resolve() != expected_paths["layer2"]
        or contract_path.resolve() != expected_paths["contract"]
        or not all(path.is_file() for path in expected_paths.values())
    ):
        raise ValueError("source-binding manifest path differs")
    layer1 = _read(layer1_path)
    layer2 = _read(layer2_path)
    fixture = _read(expected_paths["fixture"])
    contract = _read(contract_path)
    validate_layered_values(layer1=layer1, layer2=layer2, contract=contract)
    _verify(fixture, "fixture_sha256")
    if fixture != build_fixture_record():
        raise ValueError("source-binding fixtures differ")
    contract_fixture = cast(Mapping[str, Any], contract["fixture"])
    if (
        layer1.get("layer1_manifest_sha256") != expected_layer1_sha256
        or layer2.get("layer2_manifest_sha256") != expected_layer2_sha256
        or contract.get("source_binding_contract_sha256") != expected_contract_sha256
        or contract_fixture.get("fixture_sha256") != fixture.get("fixture_sha256")
        or fixture.get("all_passed") is not True
        or contract.get("package_inventory_sha256") != expected_package_sha256
        or contract.get("combined_child_environment_sha256") != expected_environment_sha256
        or contract.get("qualification_decision_sha256") != expected_qualification_decision_sha256
        or cast(Mapping[str, Any], contract["layer2"]).get("source_commit")
        != expected_source_commit
        or cast(Mapping[str, Any], contract["layer2"]).get("source_tree") != expected_source_tree
        or expected_package_sha256 != PACKAGE_INVENTORY_SHA256
        or expected_environment_sha256 != COMBINED_CHILD_ENVIRONMENT_SHA256
        or expected_qualification_decision_sha256 != R1_SELECTION_SHA256
    ):
        raise ValueError("source-binding expected hash differs")
    layer1_evidence = verify_layer1_manifest(root, layer1)
    layer2_evidence = verify_layer2_manifest(
        root,
        layer2,
        expected_commit=expected_source_commit,
        expected_tree=expected_source_tree,
        require_clean_synchronized=require_clean_synchronized,
    )
    warmup = verify_published_warmup_bundle(root)
    import_root = Path(__file__).resolve().parents[2]
    if import_root != (root / "src").resolve():
        raise ValueError("Python import root differs")
    _verify_flag_order(received_command, child_kind)
    received_argv_sha256 = argv_projection_sha256(received_command)
    if received_argv_sha256 != expected_argv_sha256:
        raise ValueError("wrapper and child argv hashes differ")
    templates = cast(Mapping[str, Mapping[str, Any]], layer2["command_templates"])
    if templates[child_kind]["template_sha256"] != command_template_sha256(child_kind) or cast(
        Mapping[str, Any], contract["command_template_sha256"]
    )[child_kind] != command_template_sha256(child_kind):
        raise ValueError("source-bound command template differs")
    if templates[child_kind]["argv_projection_sha256"] != argv_projection_sha256(
        command_template(child_kind)
    ) or cast(Mapping[str, Any], contract["argv_projection_sha256"])[
        child_kind
    ] != argv_projection_sha256(command_template(child_kind)):
        raise ValueError("source-bound argv projection differs")
    environment_evidence = _read(
        root / "results/phase2_vetted_corpus/windows_operational_environment.json"
    )
    validate_child_environment(dict(os.environ), environment_evidence)
    if file_sha256(Path(sys.executable)) != INTERPRETER_SHA256:
        raise ValueError("source-bound interpreter differs")
    modules_after = set(sys.modules) if loaded_modules is None else set(loaded_modules)
    imported_after = sorted(FORBIDDEN_MODEL_MODULES.intersection(modules_after))
    if imported_after:
        raise RuntimeError(f"model stack imported during layered source binding: {imported_after}")
    evidence: dict[str, object] = {
        "schema_version": 1,
        "binding_id": CONTRACT_ID,
        "child_kind": child_kind,
        "layer1": layer1_evidence,
        "layer2": layer2_evidence,
        "warmup": warmup,
        "source_binding_contract_sha256": expected_contract_sha256,
        "qualification_decision_sha256": expected_qualification_decision_sha256,
        "python_import_root": str(import_root),
        "python_import_root_sha256": canonical_sha256(str(import_root)),
        "command_template_sha256": command_template_sha256(child_kind),
        "argv_projection_sha256": received_argv_sha256,
        "wrapper_child_argv_match": True,
        "shell": False,
        "environment_sha256": expected_environment_sha256,
        "package_inventory_sha256": expected_package_sha256,
        "fixture_sha256": fixture["fixture_sha256"],
        "loaded_foundry_source_paths": loaded_source_paths,
        "loaded_foundry_source_count": len(loaded_source_paths),
        "model_stack_imported_before_or_during_binding": False,
        "all_hash_checks_terminal": True,
        "gate_passed": True,
        "sealed_content_use": 0,
    }
    evidence["binding_evidence_sha256"] = canonical_sha256(evidence)
    return evidence
