"""Freeze the Milestone 14B-R2 warmup-aware GRPO update contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_contract import (
    COMBINED_CHILD_ENVIRONMENT_SHA256,
    DETERMINISTIC_ENVIRONMENT,
    GRPO_RECIPE,
    GRPO_RECIPE_SHA256,
    INTERPRETER_SHA256,
    MODEL_MANIFEST_SHA256,
    MODEL_REVISION,
    STARTING_ADAPTER_SHA256,
)
from foundry.phase2.l3_grpo_warmup_update import (
    run_deterministic_fixtures,
    update_contract,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

STARTING_COMMIT = "0c0bf9cbdb0cdd7376fd02d4ddb79afe5f3479a4"
DATASET_SHA256 = "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"
OPERATIONAL_ENVIRONMENT_SHA256 = "76afee8390e73ef9274d4bc4b91d8a99735f66efb9137c4909e8619d3f9d244a"
ENVIRONMENT_V2_CONTRACT_SHA256 = "c9faa8afafafb20b84fcd0cb5e7de1b57749e822adfa27c8b401bbaf8f0153dc"
R1_SIGNAL_SUMMARY_SHA256 = "fc7bad292f6c4b5acaa845df9b30cbc624de1ee524b57443ea09e634e2352ec4"
R1_SELECTION_SHA256 = "0e809d1870ddef275017a11c4db5ffd766d9624a34dbf87a7ab417f8bed6a3cf"
R1_BLOCKER_SHA256 = "d454ec93f02e6d0d981a46735b00ca02a6120784944ef00a4016bafa3d9f407f"

IMPLEMENTATION_OUTPUT = "milestone14b_r2_warmup_update_implementation.json"
SCHEDULER_OUTPUT = "milestone14b_r2_scheduler_contract.json"
FIXTURE_OUTPUT = "milestone14b_r2_warmup_update_fixtures.json"
ORDER_OUTPUT = "milestone14b_r2_compatibility_order.json"
CONTRACT_OUTPUT = "milestone14b_r2_warmup_update_contract.json"

TRANSFORMERS_SOURCE_HASHES = {
    "optimization.py": "9554c7839c5163896a4a93c0ef259ee001ea0b83647b2f8e2986d4b35213ec8a",
    "trainer.py": "e94c52cc433a73579653238a5e4409114d13209496157ef57038ebba422598bc",
    "training_args.py": "b220831249f2c79603e48056d8a26154af1070b0d5c5205914d2cd66ac7bc76e",
}
TORCH_SOURCE_HASHES = {
    "lr_scheduler.py": "11fd39e690f8a64c6ff55aeee5da66cb82b9e47aae5a300227df64a89b2c563a",
    "optimizer.py": "d2cb89319cfb3bef9ea9cf09e727653bf07108b2fc5d74c06441486f92f090fe",
}
BITSANDBYTES_SOURCE_HASHES = {
    "optimizer.py": "a252cab404be14f29904a1d25b967ff83ad8760a8b021a145039ef2e5f3bd9bb",
    "adam.py": "fdb81e80b210464eedb62168e507133167e103c9ae5919b241600b49785acf49",
}

IMPLEMENTATION_FILES = (
    "src/foundry/phase2/l3_grpo_warmup_update.py",
    "src/foundry/phase2/l3_grpo_warmup_prepare.py",
    "src/foundry/phase2/l3_grpo_runtime.py",
    "src/foundry/phase2/l3_grpo_warmup_compatibility_runtime.py",
    "src/foundry/phase2/l3_grpo_warmup_compatibility_campaign.py",
    "src/foundry/phase2/l3_grpo_warmup_campaign.py",
    "src/foundry/phase2/l3_grpo_warmup_analysis.py",
    "tests/unit/phase2/test_l3_grpo_warmup_update.py",
    "tests/unit/phase2/test_l3_grpo_warmup_prepare.py",
    "tests/unit/phase2/test_l3_grpo_warmup_compatibility_runtime.py",
    "tests/unit/phase2/test_l3_grpo_warmup_compatibility_campaign.py",
    "tests/unit/phase2/test_l3_grpo_warmup_campaign.py",
    "tests/unit/phase2/test_l3_grpo_warmup_analysis.py",
    "tests/unit/phase2/test_l3_grpo_runtime.py",
    "tests/unit/phase2/test_l3_grpo_zero_gradient_correction.py",
)
FROZEN_DEPENDENCY_FILES = (
    "src/foundry/phase2/l3_grpo_contract.py",
    "src/foundry/phase2/l3_grpo_schedule.py",
    "src/foundry/phase2/l3_grpo_reward.py",
    "src/foundry/phase2/l3_grpo_reference.py",
    "src/foundry/phase2/l3_grpo_zero_gradient.py",
    "src/foundry/phase2/l3_grpo_analysis.py",
    "src/foundry/phase2/l3_grpo_campaign.py",
    "src/foundry/phase2/l3_grpo_signal_audit.py",
    "src/foundry/phase2/l3_grpo_signal_qualification.py",
    "src/foundry/phase2/windows_environment.py",
    "src/foundry/training/grpo_compatibility.py",
    "src/foundry/training/grpo_replay_evidence.py",
    "src/foundry/training/grpo_trainer.py",
    "src/foundry/training/retention.py",
    "src/foundry/training/base_conditioned_retention.py",
    "src/foundry/training/adapter_evaluation.py",
    "src/foundry/training/paired_analysis.py",
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
    ).stdout.strip()


def _file_rows(root: Path, paths: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"warmup-aware source is missing: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def _porcelain_paths(value: str) -> list[str]:
    paths: list[str] = []
    for line in value.splitlines():
        if len(line) >= 3 and line[2] == " ":
            relative = line[3:]
        elif len(line) >= 2 and line[1] == " ":
            relative = line[2:]
        else:
            raise RuntimeError(f"unsupported Git porcelain row: {line!r}")
        paths.append(relative.replace("\\", "/"))
    return paths


def _require_freeze_boundary(root: Path) -> None:
    if root.resolve() != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("Milestone 14B-R2 is attached to the wrong repository")
    dirty = _git(root, "status", "--porcelain")
    allowed = (
        "src/foundry/phase2/l3_grpo_runtime.py",
        "src/foundry/phase2/l3_grpo_warmup_",
        "tests/unit/phase2/test_l3_grpo_runtime.py",
        "tests/unit/phase2/test_l3_grpo_zero_gradient_correction.py",
        "tests/unit/phase2/test_l3_grpo_warmup_",
        "results/phase2_vetted_corpus/milestone14b_r2_",
        "docs/DEVLOG.md",
        "docs/VERIFIER_GRPO_RESULT.md",
    )
    dirty_paths = _porcelain_paths(dirty)
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "rev-parse", "origin/main")
    ahead_behind = _git(
        root,
        "rev-list",
        "--left-right",
        "--count",
        "main...origin/main",
    ).split()
    disallowed = [path for path in dirty_paths if not path.startswith(allowed)]
    if (
        branch != "main"
        or head != STARTING_COMMIT
        or origin != STARTING_COMMIT
        or ahead_behind != ["0", "0"]
        or disallowed
    ):
        raise RuntimeError(
            "Milestone 14B-R2 source-freeze boundary differs: "
            f"branch={branch!r}, head={head!r}, origin={origin!r}, "
            f"ahead_behind={ahead_behind!r}, disallowed={disallowed!r}"
        )
    if (root / "results/raw/phase2_vetted_corpus/milestone14b_r2").exists():
        raise RuntimeError("Milestone 14B-R2 model evidence exists before source freeze")


def _published_r1(root: Path) -> dict[str, object]:
    tracked = root / "results/phase2_vetted_corpus"
    signal = _read(tracked / "milestone14b_r1_signal_summary.json")
    selection = _read(tracked / "milestone14b_r1_selection_and_gradient_decision.json")
    blocker = _read(tracked / "milestone14b_r1_compatibility_blocker.json")
    _verify(signal, "signal_summary_sha256")
    _verify(selection, "selection_decision_sha256")
    _verify(blocker, "compatibility_blocker_sha256")
    if (
        signal.get("signal_summary_sha256") != R1_SIGNAL_SUMMARY_SHA256
        or signal.get("viability_passed") is not True
        or selection.get("selection_decision_sha256") != R1_SELECTION_SHA256
        or selection.get("both_arms_pass") is not True
        or blocker.get("compatibility_blocker_sha256") != R1_BLOCKER_SHA256
        or blocker.get("failure_classification")
        != "signal_qualified_first_warmup_step_no_policy_update"
        or blocker.get("counted_training_started") is not False
    ):
        raise ValueError("published Milestone 14B-R1 boundary differs")
    return {
        "signal_summary_sha256": signal["signal_summary_sha256"],
        "selection_decision_sha256": selection["selection_decision_sha256"],
        "compatibility_blocker_sha256": blocker["compatibility_blocker_sha256"],
        "selection": selection,
    }


def _frozen_inputs(root: Path) -> dict[str, object]:
    tracked = root / "results/phase2_vetted_corpus"
    schedules = {
        "generic": ("milestone14a_generic_schedule.json", "manifest_sha256"),
        "targeted": ("milestone14a_targeted_schedule.json", "manifest_sha256"),
        "shared_replay": ("milestone14a_shared_replay.json", "shared_replay_sha256"),
        "paired": ("milestone14a_paired_schedule.json", "paired_schedule_sha256"),
    }
    schedule_hashes: dict[str, object] = {}
    for label, (name, key) in schedules.items():
        value = _read(tracked / name)
        _verify(value, key)
        schedule_hashes[label] = value[key]
    expected_schedules = {
        "generic": "ff1005a1d7381acd52dd28b3d054b2979986c47595ed09c944880ea5fc5f5ff3",
        "targeted": "8326c1b91ba127c4734527abfed2f8bca41ecbb3a0bb7bc62a5bf940ac24f0c4",
        "shared_replay": "19e27fecde5349b6a7a9a24d8a0a8211a3b0da877282a51ece6b616688904181",
        "paired": "ed99aa38f77961fa1f669ba110cd86b3af092e027a60b8e92096a9a68bdfc8e3",
    }
    if schedule_hashes != expected_schedules:
        raise ValueError("frozen counted schedules differ")
    for arm, expected in STARTING_ADAPTER_SHA256.items():
        path = (
            root
            / f"results/raw/phase2_vetted_corpus/milestone13e/full/{arm}"
            / "training/checkpoint-64/adapter"
        )
        if directory_sha256(path) != expected:
            raise ValueError(f"{arm} starting adapter differs")
    environment = _read(tracked / "windows_operational_environment.json")
    if (
        environment.get("operational_environment_sha256") != OPERATIONAL_ENVIRONMENT_SHA256
        or environment.get("combined_child_environment_sha256") != COMBINED_CHILD_ENVIRONMENT_SHA256
        or environment.get("v2_contract_sha256") != ENVIRONMENT_V2_CONTRACT_SHA256
    ):
        raise ValueError("frozen operational environment differs")
    if any(
        file_sha256(root / environment_name / "Scripts/python.exe") != INTERPRETER_SHA256
        for environment_name in (".venv", ".venv-training")
    ):
        raise ValueError("authorized interpreter identity differs")
    return {
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "starting_adapters": STARTING_ADAPTER_SHA256,
        "schedules": schedule_hashes,
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
        "operational_environment_sha256": OPERATIONAL_ENVIRONMENT_SHA256,
        "combined_child_environment_sha256": COMBINED_CHILD_ENVIRONMENT_SHA256,
        "environment_v2_contract_sha256": ENVIRONMENT_V2_CONTRACT_SHA256,
    }


def _source_hash(path: Path, expected: str) -> dict[str, object]:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"installed scheduler source differs: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def _trajectory(
    torch_module: Any,
    optimization_module: Any,
    *,
    total_steps: int,
) -> dict[str, object]:
    warmup_steps = math.ceil(total_steps * 0.05)
    parameter = torch_module.nn.Parameter(torch_module.tensor([1.0], dtype=torch_module.float32))
    optimizer = torch_module.optim.SGD([parameter], lr=1.0e-6)
    scheduler = optimization_module.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    rows: list[dict[str, object]] = []
    for call_index in range(1, total_steps + 1):
        before = [float(group["lr"]) for group in optimizer.param_groups]
        scheduler_before = scheduler.state_dict()
        parameter.grad = torch_module.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        rows.append(
            {
                "optimizer_call_index": call_index,
                "scheduler_step_index_before": call_index - 1,
                "effective_learning_rates": before,
                "effective_learning_rate_hex": [value.hex() for value in before],
                "minimum_effective_learning_rate": min(before),
                "maximum_effective_learning_rate": max(before),
                "multiplier": max(before) / 1.0e-6,
                "multiplier_hex": (max(before) / 1.0e-6).hex(),
                "scheduler_last_epoch_before": int(scheduler_before["last_epoch"]),
                "scheduler_last_epoch_after": int(scheduler.state_dict()["last_epoch"]),
            }
        )
    first_positive = next(
        row["optimizer_call_index"]
        for row in rows
        if cast(float, row["maximum_effective_learning_rate"]) > 0.0
    )
    payload: dict[str, object] = {
        "total_steps": total_steps,
        "warmup_ratio": 0.05,
        "warmup_steps": warmup_steps,
        "base_learning_rate": 1.0e-6,
        "optimizer_then_scheduler": True,
        "first_strictly_positive_optimizer_call_index": first_positive,
        "rows": rows,
        "effective_learning_rate_trajectory": [
            row["maximum_effective_learning_rate"] for row in rows
        ],
        "effective_learning_rate_hex_trajectory": [
            cast(list[str], row["effective_learning_rate_hex"])[0] for row in rows
        ],
    }
    payload["trajectory_sha256"] = canonical_sha256(payload)
    return payload


def scheduler_contract(
    root: Path,
    *,
    torch_module: Any,
    transformers_module: Any,
) -> dict[str, object]:
    """Freeze installed source identities and exact LR trajectories."""

    training_packages = root / ".venv-training/Lib/site-packages"
    transformers_files = {
        name: _source_hash(
            training_packages / "transformers" / name,
            expected,
        )
        for name, expected in TRANSFORMERS_SOURCE_HASHES.items()
    }
    torch_files = {
        name: _source_hash(training_packages / "torch/optim" / name, expected)
        for name, expected in TORCH_SOURCE_HASHES.items()
    }
    bitsandbytes_files = {
        name: _source_hash(training_packages / "bitsandbytes/optim" / name, expected)
        for name, expected in BITSANDBYTES_SOURCE_HASHES.items()
    }
    optimization = getattr(transformers_module, "optimization", None)
    if optimization is None:
        optimization = importlib.import_module("transformers.optimization")
    training_args = transformers_module.TrainingArguments
    trainer = transformers_module.Trainer
    compatibility = _trajectory(
        torch_module,
        optimization,
        total_steps=2,
    )
    counted = _trajectory(
        torch_module,
        optimization,
        total_steps=32,
    )
    compatibility_lrs = cast(list[float], compatibility["effective_learning_rate_trajectory"])
    if compatibility_lrs[0] != 0.0 or compatibility_lrs[1] <= 0.0:
        raise RuntimeError("installed two-step scheduler sequence differs")
    payload: dict[str, object] = {
        "schema_version": 1,
        "scheduler_contract_id": "foundry-l3-grpo-frozen-cosine-trajectory-v1",
        "transformers_version": transformers_module.__version__,
        "torch_version": torch_module.__version__,
        "transformers_source_files": transformers_files,
        "torch_source_files": torch_files,
        "bitsandbytes_source_files": bitsandbytes_files,
        "callable_source_sha256": {
            "cosine_lambda": hashlib.sha256(
                inspect.getsource(optimization._get_cosine_schedule_with_warmup_lr_lambda).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "cosine_factory": hashlib.sha256(
                inspect.getsource(optimization.get_cosine_schedule_with_warmup).encode("utf-8")
            ).hexdigest(),
            "warmup_steps": hashlib.sha256(
                inspect.getsource(training_args.get_warmup_steps).encode("utf-8")
            ).hexdigest(),
            "trainer_inner_loop": hashlib.sha256(
                inspect.getsource(trainer._inner_training_loop).encode("utf-8")
            ).hexdigest(),
            "lambda_lr": hashlib.sha256(
                inspect.getsource(torch_module.optim.lr_scheduler.LambdaLR).encode("utf-8")
            ).hexdigest(),
        },
        "optimizer_then_scheduler_order": [
            "callback_on_pre_optimizer_step",
            "optimizer_step",
            "callback_on_optimizer_step",
            "scheduler_step",
            "trainer_global_step_increment",
            "callback_on_step_end",
        ],
        "compatibility": compatibility,
        "counted_training": counted,
    }
    payload["scheduler_contract_sha256"] = canonical_sha256(payload)
    return payload


def compatibility_order(root: Path, r1: Mapping[str, object]) -> dict[str, object]:
    selection = cast(Mapping[str, Any], r1["selection"])
    arms = cast(Mapping[str, Mapping[str, Any]], selection["arms"])
    rows: dict[str, object] = {}
    for arm in ("generic", "targeted"):
        selected = arms[arm]
        rows[arm] = {
            "optimizer_call_1": {
                "group_id": selected["replay_group_id"],
                "source_kind": "base_replay",
                "original_counted_schedule_position": selected["replay_schedule_position"],
                "prompt_sha256": selected["replay_prompt_sha256"],
                "required_effective_learning_rate": 0.0,
            },
            "optimizer_call_2": {
                "group_id": selected["task_group_id"],
                "source_kind": "task",
                "original_counted_schedule_position": selected["task_schedule_position"],
                "prompt_sha256": selected["task_prompt_sha256"],
                "required_effective_learning_rate_relation": "strictly_positive",
            },
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "order_id": "foundry-l3-grpo-warmup-compatibility-order-v1",
        "selection_decision_sha256": R1_SELECTION_SHA256,
        "arms": rows,
        "compatibility_only": True,
        "counted_generic_schedule_changed": False,
        "counted_targeted_schedule_changed": False,
        "task_quotas_changed": False,
        "replay_positions_changed": False,
        "group_or_completion_counts_changed": False,
    }
    payload["compatibility_order_sha256"] = canonical_sha256(payload)
    return payload


def freeze(
    root: Path,
    *,
    torch_module: Any,
    transformers_module: Any,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    root = root.resolve()
    _require_freeze_boundary(root)
    r1 = _published_r1(root)
    frozen = _frozen_inputs(root)
    scheduler = scheduler_contract(
        root,
        torch_module=torch_module,
        transformers_module=transformers_module,
    )
    fixtures = run_deterministic_fixtures(torch_module)
    order = compatibility_order(root, r1)
    files = _file_rows(root, (*IMPLEMENTATION_FILES, *FROZEN_DEPENDENCY_FILES))
    implementation: dict[str, object] = {
        "schema_version": 1,
        "implementation_id": "foundry-l3-grpo-warmup-aware-update-implementation-v1",
        "source_parent_commit": STARTING_COMMIT,
        "implementation_files": list(IMPLEMENTATION_FILES),
        "frozen_dependency_files": list(FROZEN_DEPENDENCY_FILES),
        "files": files,
        "model_loaded": False,
        "model_generation_calls": 0,
        "optimizer_steps": 0,
        "sealed_content_use": 0,
    }
    implementation["implementation_sha256"] = canonical_sha256(implementation)
    update = update_contract()
    contract: dict[str, object] = {
        "schema_version": 1,
        "contract_id": "foundry-grpo-warmup-aware-update-v1",
        "source_parent_commit": STARTING_COMMIT,
        "implementation_sha256": implementation["implementation_sha256"],
        "scheduler_contract_sha256": scheduler["scheduler_contract_sha256"],
        "fixture_sha256": fixtures["fixture_sha256"],
        "update_contract": update,
        "update_contract_sha256": update["update_contract_sha256"],
        "compatibility_order_sha256": order["compatibility_order_sha256"],
        "compatibility_effective_learning_rates": cast(
            Mapping[str, Any], scheduler["compatibility"]
        )["effective_learning_rate_trajectory"],
        "counted_effective_learning_rates": cast(Mapping[str, Any], scheduler["counted_training"])[
            "effective_learning_rate_trajectory"
        ],
        "r1_signal_summary_sha256": r1["signal_summary_sha256"],
        "r1_selection_decision_sha256": r1["selection_decision_sha256"],
        "r1_compatibility_blocker_sha256": r1["compatibility_blocker_sha256"],
        "model_revision": frozen["model_revision"],
        "model_manifest_sha256": frozen["model_manifest_sha256"],
        "dataset_sha256": frozen["dataset_sha256"],
        "starting_adapters": frozen["starting_adapters"],
        "schedules": frozen["schedules"],
        "reward": frozen["reward"],
        "reference_mechanism_sha256": frozen["reference_mechanism_sha256"],
        "recipe": GRPO_RECIPE,
        "recipe_sha256": GRPO_RECIPE_SHA256,
        "process_environment": DETERMINISTIC_ENVIRONMENT,
        "operational_environment_sha256": frozen["operational_environment_sha256"],
        "combined_child_environment_sha256": frozen["combined_child_environment_sha256"],
        "environment_v2_contract_sha256": frozen["environment_v2_contract_sha256"],
        "optimizer_changed": False,
        "scheduler_changed": False,
        "warmup_ratio_changed": False,
        "learning_rate_changed": False,
        "schedule_changed": False,
        "reward_changed": False,
        "reference_changed": False,
        "scientific_settings_changed": False,
        "counted_training_authorized_before_compatibility": False,
        "sealed_content_use": 0,
    }
    contract["warmup_update_contract_sha256"] = canonical_sha256(contract)
    return implementation, scheduler, fixtures, order, contract


def _write_new_or_identical(path: Path, value: object) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"existing warmup-aware artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def write_freeze(
    root: Path,
    *,
    torch_module: Any,
    transformers_module: Any,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    values = freeze(
        root,
        torch_module=torch_module,
        transformers_module=transformers_module,
    )
    output = root / "results/phase2_vetted_corpus"
    for name, value in zip(
        (
            IMPLEMENTATION_OUTPUT,
            SCHEDULER_OUTPUT,
            FIXTURE_OUTPUT,
            ORDER_OUTPUT,
            CONTRACT_OUTPUT,
        ),
        values,
        strict=True,
    ):
        _write_new_or_identical(output / name, value)
    return values


def verify_warmup_update_contract(
    root: Path,
    value: Mapping[str, Any],
    *,
    require_clean_synchronized: bool,
) -> None:
    """Verify a published warmup-aware contract and every bound source row."""

    expected_path = root.resolve() / "results/phase2_vetted_corpus" / CONTRACT_OUTPUT
    if not expected_path.is_file():
        raise FileNotFoundError("published warmup-aware update contract is absent")
    _verify(value, "warmup_update_contract_sha256")
    if (
        value.get("contract_id") != "foundry-grpo-warmup-aware-update-v1"
        or value.get("source_parent_commit") != STARTING_COMMIT
        or value.get("r1_signal_summary_sha256") != R1_SIGNAL_SUMMARY_SHA256
        or value.get("r1_selection_decision_sha256") != R1_SELECTION_SHA256
        or value.get("r1_compatibility_blocker_sha256") != R1_BLOCKER_SHA256
        or value.get("recipe_sha256") != GRPO_RECIPE_SHA256
        or value.get("recipe") != GRPO_RECIPE
        or value.get("starting_adapters") != STARTING_ADAPTER_SHA256
        or value.get("process_environment") != DETERMINISTIC_ENVIRONMENT
        or value.get("combined_child_environment_sha256") != COMBINED_CHILD_ENVIRONMENT_SHA256
        or value.get("operational_environment_sha256") != OPERATIONAL_ENVIRONMENT_SHA256
        or value.get("environment_v2_contract_sha256") != ENVIRONMENT_V2_CONTRACT_SHA256
        or value.get("optimizer_changed") is not False
        or value.get("scheduler_changed") is not False
        or value.get("warmup_ratio_changed") is not False
        or value.get("learning_rate_changed") is not False
        or value.get("schedule_changed") is not False
        or value.get("reward_changed") is not False
        or value.get("reference_changed") is not False
        or value.get("scientific_settings_changed") is not False
    ):
        raise ValueError("warmup-aware update contract differs")
    implementation = _read(root / "results/phase2_vetted_corpus" / IMPLEMENTATION_OUTPUT)
    _verify(implementation, "implementation_sha256")
    if implementation.get("implementation_sha256") != value.get("implementation_sha256"):
        raise ValueError("warmup-aware implementation differs")
    for row_value in cast(list[object], implementation.get("files")):
        if not isinstance(row_value, dict):
            raise ValueError("warmup-aware source row differs")
        row = cast(dict[str, object], row_value)
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ValueError("warmup-aware source path differs")
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or path.stat().st_size != row.get("bytes")
            or file_sha256(path) != row.get("sha256")
        ):
            raise ValueError("warmup-aware source hash differs")
    scheduler = _read(root / "results/phase2_vetted_corpus" / SCHEDULER_OUTPUT)
    fixtures = _read(root / "results/phase2_vetted_corpus" / FIXTURE_OUTPUT)
    order = _read(root / "results/phase2_vetted_corpus" / ORDER_OUTPUT)
    _verify(scheduler, "scheduler_contract_sha256")
    _verify(fixtures, "fixture_sha256")
    _verify(order, "compatibility_order_sha256")
    update = cast(Mapping[str, Any], value.get("update_contract"))
    _verify(update, "update_contract_sha256")
    compatibility = cast(Mapping[str, Any], scheduler.get("compatibility"))
    counted = cast(Mapping[str, Any], scheduler.get("counted_training"))
    if (
        scheduler.get("scheduler_contract_sha256") != value.get("scheduler_contract_sha256")
        or fixtures.get("fixture_sha256") != value.get("fixture_sha256")
        or order.get("compatibility_order_sha256") != value.get("compatibility_order_sha256")
        or update.get("update_contract_sha256") != value.get("update_contract_sha256")
        or compatibility.get("effective_learning_rate_trajectory")
        != value.get("compatibility_effective_learning_rates")
        or counted.get("effective_learning_rate_trajectory")
        != value.get("counted_effective_learning_rates")
        or compatibility.get("warmup_steps") != 1
        or compatibility.get("first_strictly_positive_optimizer_call_index") != 2
        or counted.get("warmup_steps") != 2
    ):
        raise ValueError("warmup-aware subsidiary contract differs")
    if require_clean_synchronized:
        head = _git(root, "rev-parse", "HEAD")
        if (
            _git(root, "branch", "--show-current") != "main"
            or head != _git(root, "rev-parse", "origin/main")
            or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
            != ["0", "0"]
            or _git(root, "status", "--porcelain")
        ):
            raise RuntimeError("warmup-aware model execution requires synchronized clean main")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, head],
            cwd=root,
            shell=False,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    import transformers  # type: ignore[import-untyped]

    implementation, scheduler, fixtures, order, contract = write_freeze(
        args.root,
        torch_module=torch,
        transformers_module=transformers,
    )
    print(
        json.dumps(
            {
                "implementation_sha256": implementation["implementation_sha256"],
                "scheduler_contract_sha256": scheduler["scheduler_contract_sha256"],
                "fixture_sha256": fixtures["fixture_sha256"],
                "compatibility_order_sha256": order["compatibility_order_sha256"],
                "warmup_update_contract_sha256": contract["warmup_update_contract_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
