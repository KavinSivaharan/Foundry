from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from foundry.phase2.l3_grpo_contract import COMBINED_CHILD_ENVIRONMENT_SHA256
from foundry.phase2.l3_grpo_signal_qualification import verify_qualification_contract
from foundry.phase2.l3_grpo_source_binding import (
    CONTRACT_OUTPUT,
    FORBIDDEN_MODEL_MODULES,
    LAYER1_OUTPUT,
    LAYER2_OUTPUT,
    LAYER2_PATHS,
    R1_QUALIFICATION_CONTRACT_SHA256,
    R1_SELECTION_SHA256,
    R2_WARMUP_CONTRACT_SHA256,
    argv_projection_sha256,
    build_combined_contract,
    build_fixture_record,
    build_layer1_manifest,
    build_layer2_manifest,
    command_template_sha256,
    validate_layered_values,
    verify_layer1_manifest,
    verify_layer2_manifest,
    verify_layered_source_binding,
    verify_published_warmup_bundle,
)
from foundry.phase2.windows_environment import (
    load_frozen_child_environment,
    validate_child_environment,
)
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
TRACKED = ROOT / "results/phase2_vetted_corpus"


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rehash(value: dict[str, Any], key: str) -> None:
    value[key] = canonical_sha256({name: item for name, item in value.items() if name != key})


def _synthetic_layer2() -> dict[str, Any]:
    value: dict[str, Any] = {
        "role": "compatibility_runtime",
        "ordered_paths": list(LAYER2_PATHS),
        "source_checks_warning_only": False,
        "shell": False,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "python_import_root_sha256": "c" * 64,
    }
    _rehash(value, "layer2_manifest_sha256")
    return value


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


def _temporary_layer2(tmp_path: Path) -> tuple[dict[str, Any], str, str]:
    for index, relative in enumerate(LAYER2_PATHS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"layer2-{index}-{relative}\n".encode())
    environment = tmp_path / "results/phase2_vetted_corpus/windows_operational_environment.json"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment_value = {
        "combined_child_environment_sha256": COMBINED_CHILD_ENVIRONMENT_SHA256,
        "operational_environment_sha256": (
            "76afee8390e73ef9274d4bc4b91d8a99735f66efb9137c4909e8619d3f9d244a"
        ),
        "v2_contract_sha256": ("c9faa8afafafb20b84fcd0cb5e7de1b57749e822adfa27c8b401bbaf8f0153dc"),
    }
    _rehash(environment_value, "environment_evidence_sha256")
    environment.write_text(
        json.dumps(environment_value, sort_keys=True),
        encoding="utf-8",
    )
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "source-binding@example.invalid")
    _git(tmp_path, "config", "user.name", "Source Binding Fixture")
    _git(tmp_path, "add", "--all")
    _git(tmp_path, "commit", "-m", "fixture")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
    return build_layer2_manifest(tmp_path, source_commit=commit), commit, tree


def test_old_unlayered_contract_reproduces_published_failure() -> None:
    qualification = _read(TRACKED / "milestone14b_r1_qualification_contract.json")
    with pytest.raises(ValueError, match="^qualification implementation source differs$"):
        verify_qualification_contract(ROOT, qualification, require_clean_synchronized=False)


def test_exact_layer1_and_exact_layer2_roles_pass(tmp_path: Path) -> None:
    layer1 = build_layer1_manifest(ROOT)
    layer2, commit, tree = _temporary_layer2(tmp_path)
    validate_layered_values(layer1=layer1, layer2=layer2, contract=None)
    result = verify_layer2_manifest(
        tmp_path,
        layer2,
        expected_commit=commit,
        expected_tree=tree,
        require_clean_synchronized=False,
    )
    assert result["current_execution_source_matches_fix_commit"] is True
    first = layer2["files"][0]
    assert set(first) == {
        "path",
        "execution_bytes",
        "execution_sha256",
        "git_blob_bytes",
        "git_blob_sha256",
    }


def test_one_byte_layer1_evidence_mutation_fails() -> None:
    layer1 = build_layer1_manifest(ROOT)
    mutated = copy.deepcopy(layer1)
    mutated["evidence_rows"][0]["sha256"] = "0" * 64
    _rehash(mutated, "layer1_manifest_sha256")
    with pytest.raises(ValueError, match="Layer-1 evidence file differs"):
        verify_layer1_manifest(ROOT, mutated)


def test_one_byte_layer2_source_mutation_fails(tmp_path: Path) -> None:
    layer2, commit, tree = _temporary_layer2(tmp_path)
    path = tmp_path / LAYER2_PATHS[0]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Layer-2 current source differs"):
        verify_layer2_manifest(
            tmp_path,
            layer2,
            expected_commit=commit,
            expected_tree=tree,
            require_clean_synchronized=False,
        )


def test_wrong_layer2_commit_fails(tmp_path: Path) -> None:
    layer2, _, tree = _temporary_layer2(tmp_path)
    with pytest.raises(ValueError, match="Layer-2 manifest identity differs"):
        verify_layer2_manifest(
            tmp_path,
            layer2,
            expected_commit="0" * 40,
            expected_tree=tree,
            require_clean_synchronized=False,
        )


def test_wrong_git_tree_fails(tmp_path: Path) -> None:
    layer2, commit, _ = _temporary_layer2(tmp_path)
    with pytest.raises(ValueError, match="Layer-2 manifest identity differs"):
        verify_layer2_manifest(
            tmp_path,
            layer2,
            expected_commit=commit,
            expected_tree="0" * 40,
            require_clean_synchronized=False,
        )


def test_wrong_python_import_root_fails(tmp_path: Path) -> None:
    layer2, commit, tree = _temporary_layer2(tmp_path)
    mutated = copy.deepcopy(layer2)
    mutated["python_import_root"] = str(tmp_path / "wrong")
    _rehash(mutated, "layer2_manifest_sha256")
    with pytest.raises(ValueError, match="Layer-2 manifest identity differs"):
        verify_layer2_manifest(
            tmp_path,
            mutated,
            expected_commit=commit,
            expected_tree=tree,
            require_clean_synchronized=False,
        )


def test_wrong_qualification_decision_hash_fails() -> None:
    layer1 = build_layer1_manifest(ROOT)
    layer2 = _synthetic_layer2()
    contract = build_combined_contract(
        layer1=layer1,
        layer2=layer2,
        fixtures=build_fixture_record(),
    )
    contract["qualification_decision_sha256"] = "0" * 64
    _rehash(contract, "source_binding_contract_sha256")
    with pytest.raises(ValueError, match="combined layered"):
        validate_layered_values(layer1=layer1, layer2=layer2, contract=contract)


def test_wrong_reward_or_schedule_hash_fails() -> None:
    layer1 = build_layer1_manifest(ROOT)
    mutated = copy.deepcopy(layer1)
    mutated["scientific_identities"]["reward"]["contract_sha256"] = "0" * 64
    _rehash(mutated, "layer1_manifest_sha256")
    with pytest.raises(ValueError, match="scientific identities"):
        verify_layer1_manifest(ROOT, mutated)


def test_swapping_layer1_and_layer2_manifests_fails() -> None:
    layer1 = build_layer1_manifest(ROOT)
    layer2 = _synthetic_layer2()
    with pytest.raises(ValueError):
        validate_layered_values(
            layer1=layer2,
            layer2=layer1,
            contract=None,
        )


def test_missing_manifest_paths_fail_before_model_import(tmp_path: Path) -> None:
    before = FORBIDDEN_MODEL_MODULES.intersection(sys.modules)
    with pytest.raises(ValueError, match="manifest path differs"):
        verify_layered_source_binding(
            root=ROOT,
            layer1_path=tmp_path / LAYER1_OUTPUT,
            expected_layer1_sha256="0" * 64,
            layer2_path=tmp_path / LAYER2_OUTPUT,
            expected_layer2_sha256="0" * 64,
            contract_path=tmp_path / CONTRACT_OUTPUT,
            expected_contract_sha256="0" * 64,
            expected_source_commit="0" * 40,
            expected_source_tree="0" * 40,
            expected_package_sha256="0" * 64,
            expected_environment_sha256="0" * 64,
            expected_qualification_decision_sha256="0" * 64,
            child_kind="compatibility",
            received_command=[],
            expected_argv_sha256="0" * 64,
            require_clean_synchronized=False,
            loaded_modules=set(),
        )
    assert FORBIDDEN_MODEL_MODULES.intersection(sys.modules) == before


def test_reordered_layer2_paths_change_manifest_hash() -> None:
    layer2 = _synthetic_layer2()
    original = layer2["layer2_manifest_sha256"]
    layer2["ordered_paths"] = list(reversed(layer2["ordered_paths"]))
    _rehash(layer2, "layer2_manifest_sha256")
    assert layer2["layer2_manifest_sha256"] != original


def test_unauthorized_extra_runtime_path_fails() -> None:
    layer1 = build_layer1_manifest(ROOT)
    layer2 = _synthetic_layer2()
    layer2["ordered_paths"].append("src/foundry/phase2/unauthorized.py")
    _rehash(layer2, "layer2_manifest_sha256")
    with pytest.raises(ValueError, match="roles or paths differ"):
        validate_layered_values(layer1=layer1, layer2=layer2, contract=None)


def test_wrapper_and_child_argv_hashes_match() -> None:
    command = [
        r"C:\Users\Admin\Projects\Foundry\.venv-training\Scripts\python.exe",
        "-m",
        "foundry.phase2.l3_grpo_warmup_compatibility_runtime",
        "--expected-argv-sha256",
        "PENDING",
    ]
    expected = argv_projection_sha256(command)
    command[-1] = expected
    assert argv_projection_sha256(command) == expected


def test_shell_false_remains_mandatory() -> None:
    layer1 = build_layer1_manifest(ROOT)
    layer2 = _synthetic_layer2()
    layer2["shell"] = True
    _rehash(layer2, "layer2_manifest_sha256")
    with pytest.raises(ValueError, match="roles or paths differ"):
        validate_layered_values(layer1=layer1, layer2=layer2, contract=None)


def test_exact_warmup_aware_contract_remains_unchanged() -> None:
    value = verify_published_warmup_bundle(ROOT)
    assert value["warmup_update_contract_sha256"] == R2_WARMUP_CONTRACT_SHA256
    assert value["compatibility_effective_learning_rates"] == [0.0, 0.000001]


def test_no_scientific_grpo_configuration_changes() -> None:
    layer1 = build_layer1_manifest(ROOT)
    assert layer1["scientific_qualification_changed"] is False
    assert layer1["scientific_identities"]["qualification_contract_sha256"] == (
        R1_QUALIFICATION_CONTRACT_SHA256
    )
    assert layer1["scientific_identities"]["selection_decision_sha256"] == R1_SELECTION_SHA256


def test_no_source_check_is_skipped_or_downgraded() -> None:
    layer1 = build_layer1_manifest(ROOT)
    layer2 = _synthetic_layer2()
    layer2["source_checks_warning_only"] = True
    _rehash(layer2, "layer2_manifest_sha256")
    with pytest.raises(ValueError, match="roles or paths differ"):
        validate_layered_values(layer1=layer1, layer2=layer2, contract=None)


def test_binding_mismatch_precedes_model_import(tmp_path: Path) -> None:
    isolated_modules: set[str] = set()
    with pytest.raises(ValueError):
        verify_layered_source_binding(
            root=ROOT,
            layer1_path=tmp_path / "missing-layer1.json",
            expected_layer1_sha256="0" * 64,
            layer2_path=tmp_path / "missing-layer2.json",
            expected_layer2_sha256="0" * 64,
            contract_path=tmp_path / "missing-contract.json",
            expected_contract_sha256="0" * 64,
            expected_source_commit="0" * 40,
            expected_source_tree="0" * 40,
            expected_package_sha256="0" * 64,
            expected_environment_sha256="0" * 64,
            expected_qualification_decision_sha256="0" * 64,
            child_kind="counted",
            received_command=[],
            expected_argv_sha256="0" * 64,
            require_clean_synchronized=False,
            loaded_modules=isolated_modules,
        )
    assert isolated_modules == set()


def test_exact_published_environment_reconstructs() -> None:
    raw = (
        ROOT
        / "results/raw/phase2_vetted_corpus/milestone13c_r1"
        / "windows_operational_environment_v2_raw.json"
    )
    tracked = TRACKED / "windows_operational_environment.json"
    evidence = _read(tracked)
    child = load_frozen_child_environment(
        raw_environment_path=raw,
        tracked_evidence_path=tracked,
    )
    validate_child_environment(child, evidence)
    assert canonical_sha256(child) == COMBINED_CHILD_ENVIRONMENT_SHA256


def test_command_templates_are_versioned_by_child_kind() -> None:
    assert command_template_sha256("compatibility") != command_template_sha256("counted")
