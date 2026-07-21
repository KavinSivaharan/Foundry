from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import foundry
from foundry.training import grpo_paths as paths
from foundry.training import grpo_runtime
from foundry.training.config import canonical_sha256

_PRIOR_ORCHESTRATION_COMMIT = "8f67e46262b7edafc57861aaf185efa345228179"
_HASH = "a" * 64


def _make_contract(
    tmp_path: Path, *, python_executable: Path | None = None
) -> paths.GrpoRuntimePaths:
    source = tmp_path / "source"
    primary = tmp_path / "primary"
    artifact = tmp_path / "artifacts"
    cache = tmp_path / "cache"
    snapshot = cache / "snapshot"
    for directory in (source / "src" / "foundry", primary, artifact, snapshot):
        directory.mkdir(parents=True, exist_ok=True)
    executable = Path(sys.executable) if python_executable is None else python_executable
    return paths.GrpoRuntimePaths(
        source_root=source.resolve(),
        primary_repository_root=primary.resolve(),
        python_executable=executable.resolve(),
        artifact_root=artifact.resolve(),
        model_cache_root=cache.resolve(),
        model_snapshot_root=snapshot.resolve(),
        source_commit="b" * 40,
        source_tree="c" * 40,
        source_tracked_file_count=10,
        source_tracked_bytes=100,
        source_tracked_manifest_sha256=_HASH,
        source_root_identity_sha256=paths._directory_identity_sha256(source),
        primary_root_identity_sha256=paths._directory_identity_sha256(primary),
        artifact_root_identity_sha256=paths._directory_identity_sha256(artifact),
        model_cache_root_identity_sha256=paths._directory_identity_sha256(cache),
        python_executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        python_executable_size_bytes=executable.stat().st_size,
        model_artifact_file_count=3,
        model_artifact_bytes=300,
        model_artifact_manifest_sha256="d" * 64,
        process_environment_sha256=canonical_sha256(
            paths._process_environment_values(source, cache, artifact)
        ),
        process_command_template_sha256=paths._command_template_sha256(executable, artifact),
    )


def _source_identity(contract: paths.GrpoRuntimePaths) -> dict[str, object]:
    return {
        "commit": contract.source_commit,
        "tree": contract.source_tree,
        "tracked_file_count": contract.source_tracked_file_count,
        "tracked_bytes": contract.source_tracked_bytes,
        "tracked_manifest_sha256": contract.source_tracked_manifest_sha256,
    }


def test_distinct_source_interpreter_and_artifact_roots_are_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    primary = tmp_path / "primary"
    artifact = tmp_path / "artifact"
    cache = tmp_path / "cache"
    snapshot = cache / "snapshot"
    executable = primary / "environment" / "python.exe"
    for directory in (source, artifact, snapshot, executable.parent):
        directory.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"configured-python")
    executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    monkeypatch.setattr(paths, "FROZEN_PYTHON_EXECUTABLE_SHA256", executable_hash)
    monkeypatch.setattr(paths, "FROZEN_PYTHON_EXECUTABLE_SIZE_BYTES", executable.stat().st_size)
    monkeypatch.setattr(paths, "_assert_git_root", lambda root, field: None)
    monkeypatch.setattr(paths, "_assert_primary_clean", lambda root: None)
    monkeypatch.setattr(
        paths,
        "_source_git_identity",
        lambda root: {
            "commit": "b" * 40,
            "tree": "c" * 40,
            "tracked_file_count": 1,
            "tracked_bytes": 1,
            "tracked_manifest_sha256": _HASH,
        },
    )
    monkeypatch.setattr(
        paths,
        "_model_artifact_identity",
        lambda cache_root, snapshot_root: {
            "file_count": 3,
            "total_bytes": 300,
            "manifest_sha256": "d" * 64,
        },
    )

    contract = paths.freeze_runtime_paths(
        source_root=source.resolve(),
        primary_repository_root=primary.resolve(),
        python_executable=executable.resolve(),
        artifact_root=artifact.resolve(),
        model_cache_root=cache.resolve(),
        model_snapshot_root=snapshot.resolve(),
        verify_current_process=False,
    )

    assert contract.source_root == source.resolve()
    assert contract.python_executable == executable.resolve()
    assert contract.artifact_root == artifact.resolve()
    assert len({str(source.resolve()), str(executable.resolve()), str(artifact.resolve())}) == 3


def test_fresh_process_uses_configured_interpreter_and_frozen_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured-python.exe"
    configured.write_bytes(Path(sys.executable).read_bytes())
    contract = _make_contract(tmp_path, python_executable=configured)
    imported = contract.source_root / "src" / "foundry" / "__init__.py"
    imported.write_text("", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"{imported}\n20260720\n1\n")

    monkeypatch.setattr(paths.subprocess, "run", fake_run)
    evidence = paths.verify_fresh_process_import(contract)

    assert calls[0][0][0] == str(configured.resolve())
    environment = calls[0][1]["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONHASHSEED"] == "20260720"
    assert environment["PYTHONPATH"] == str(contract.source_root / "src")
    assert evidence["process_environment_sha256"] == contract.process_environment_sha256


def test_external_artifact_output_is_accepted(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    output = contract.artifact_root / "runs" / "summary.json"
    assert paths.assert_artifact_path(contract, output, "summary") == output.resolve()


@pytest.mark.parametrize("root_field", ["source_root", "primary_repository_root"])
def test_repository_output_is_rejected(tmp_path: Path, root_field: str) -> None:
    contract = _make_contract(tmp_path)
    output = getattr(contract, root_field) / "generated.json"
    with pytest.raises(ValueError, match="artifact_root"):
        paths.assert_artifact_path(contract, output, "generated output")


def test_artifact_traversal_escape_is_rejected(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    output = contract.artifact_root / "nested" / ".." / ".." / "escaped.json"
    with pytest.raises(ValueError, match="artifact_root"):
        paths.assert_artifact_path(contract, output, "escaped output")


def test_running_with_a_different_interpreter_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternate = tmp_path / "primary" / "alternate-python.exe"
    alternate.parent.mkdir()
    alternate.write_bytes(b"different-interpreter")
    contract = _make_contract(tmp_path, python_executable=alternate)
    monkeypatch.setattr(paths, "_assert_git_root", lambda root, field: None)
    monkeypatch.setattr(paths, "_assert_primary_clean", lambda root: None)
    monkeypatch.setattr(paths, "_source_git_identity", lambda root: _source_identity(contract))

    with pytest.raises(RuntimeError, match="running interpreter differs"):
        paths.validate_runtime_paths(contract)


def test_relative_interpreter_contract_is_rejected(tmp_path: Path) -> None:
    value = _make_contract(tmp_path).as_dict()
    value["python_executable"] = "environment/python.exe"
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    value["contract_sha256"] = canonical_sha256(payload)
    with pytest.raises(ValueError, match="python_executable must contain an absolute path"):
        paths.runtime_paths_from_dict(value)


def test_frozen_source_import_resolution_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path)
    imported = contract.source_root / "src" / "foundry" / "__init__.py"
    imported.write_text("", encoding="utf-8")
    monkeypatch.setattr(foundry, "__file__", str(imported))
    assert paths.validate_foundry_import(contract)["foundry_import_path"] == str(imported.resolve())


def test_primary_source_import_resolution_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path)
    imported = contract.primary_repository_root / "src" / "foundry" / "__init__.py"
    imported.parent.mkdir(parents=True, exist_ok=True)
    imported.write_text("", encoding="utf-8")
    monkeypatch.setattr(foundry, "__file__", str(imported))
    with pytest.raises(RuntimeError, match="outside the immutable source worktree"):
        paths.validate_foundry_import(contract)


def test_intentional_same_repository_ignored_output_contract_remains_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    output = repository / "ignored" / "run"
    output.mkdir(parents=True)
    monkeypatch.setattr(
        grpo_runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    grpo_runtime._assert_git_ignored(repository, output, "legacy schedule output")


def test_windows_path_comparison_is_case_insensitive(tmp_path: Path) -> None:
    mixed_case = tmp_path / "MixedCase" / "Child"
    mixed_case.mkdir(parents=True)
    assert paths.same_canonical_path(mixed_case, Path(str(mixed_case).swapcase()))


def test_symlink_or_junction_escape_is_rejected(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = contract.artifact_root / "escape-link"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="artifact_root"):
        paths.assert_artifact_path(contract, link / "output.json", "linked output")


def test_precreated_artifact_root_cannot_be_silently_replaced(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    original = contract.artifact_root
    original.rename(tmp_path / "original-artifact-root")
    original.mkdir()
    with pytest.raises(RuntimeError, match="runtime roots were replaced"):
        paths.validate_runtime_paths(contract)


def test_runtime_path_contract_is_hashable_and_reconstructable(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    reconstructed = paths.runtime_paths_from_dict(
        json.loads(json.dumps(contract.as_dict(), sort_keys=True))
    )
    assert reconstructed == contract
    assert reconstructed.contract_sha256 == contract.contract_sha256


def test_fresh_process_command_and_environment_hashes_are_deterministic(
    tmp_path: Path,
) -> None:
    contract = _make_contract(tmp_path)
    first_command = paths.python_module_command(contract, "foundry.training.grpo_runtime", ["x"])
    second_command = paths.python_module_command(contract, "foundry.training.grpo_runtime", ["x"])
    assert paths.process_command_sha256(first_command) == paths.process_command_sha256(
        second_command
    )
    assert paths.frozen_environment_sha256(contract) == paths.frozen_environment_sha256(contract)
    assert paths.frozen_process_environment(contract, {}) == paths.frozen_process_environment(
        contract, {}
    )


def test_orchestration_patch_leaves_scientific_artifacts_byte_identical() -> None:
    repository = Path(__file__).resolve().parents[3]
    protected = [
        "configs/training",
        "results/training/verifier_grpo_v1_generic_schedule.json",
        "results/training/verifier_grpo_v1_targeted_schedule.json",
        "results/training/verifier_grpo_v1_schedule_summary.json",
        "results/training/verifier_grpo_v1_warning_contract.json",
        "src/foundry/training/grpo_compatibility.py",
        "src/foundry/training/grpo_config.py",
        "src/foundry/training/grpo_reference.py",
        "src/foundry/training/grpo_retention.py",
        "src/foundry/training/grpo_reward.py",
        "src/foundry/training/grpo_schedule.py",
        "src/foundry/training/grpo_trainer.py",
        "src/foundry/evaluation",
        "src/foundry/synthesis",
        "pyproject.toml",
        "uv.lock",
    ]
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--quiet",
            _PRIOR_ORCHESTRATION_COMMIT,
            "--",
            *protected,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
