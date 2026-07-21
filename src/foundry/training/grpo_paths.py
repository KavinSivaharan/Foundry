"""Typed, source-immutable path contract for verifier-GRPO execution.

The scientific GRPO runtime deliberately lives in a detached read-only worktree,
while its interpreter, model cache, and writable artifacts may live elsewhere.
This module keeps those roots explicit and independently verifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from foundry.training.config import canonical_sha256

RUNTIME_PATHS_ID = "foundry-verifier-grpo-runtime-paths-v1"
RUNTIME_PATHS_SCHEMA_VERSION = 1
FROZEN_PYTHON_IMPLEMENTATION = "CPython"
FROZEN_PYTHON_VERSION = "3.12.10"
FROZEN_PYTHON_EXECUTABLE_SHA256 = "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
FROZEN_PYTHON_EXECUTABLE_SIZE_BYTES = 274_424
FROZEN_PROCESS_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "20260720",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
PROCESS_COMMAND_TEMPLATES = {
    "generation_same_process": [
        "{python_executable}",
        "-m",
        "foundry.training.grpo_replay_runtime",
        "same-process",
        "--runtime-paths",
        "{runtime_paths}",
        "{frozen_inputs_and_outputs}",
    ],
    "generation_fresh_process": [
        "{python_executable}",
        "-m",
        "foundry.training.grpo_replay_runtime",
        "one-process",
        "--runtime-paths",
        "{runtime_paths}",
        "{frozen_inputs_and_outputs}",
    ],
    "two_step_fresh_process": [
        "{python_executable}",
        "-m",
        "foundry.training.grpo_two_step_runtime",
        "one-run",
        "--runtime-paths",
        "{runtime_paths}",
        "{frozen_inputs_and_outputs}",
    ],
    "counted_grpo": [
        "{python_executable}",
        "-m",
        "foundry.training.grpo_runtime",
        "--runtime-paths",
        "{runtime_paths}",
        "{frozen_inputs_and_outputs}",
    ],
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_absolute(path: Path, field: str, *, file: bool = False) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    resolved = path.resolve(strict=True)
    if file and not resolved.is_file():
        raise FileNotFoundError(f"{field} is not a file: {resolved}")
    if not file and not resolved.is_dir():
        raise NotADirectoryError(f"{field} is not a directory: {resolved}")
    return resolved


def _path_key(path: Path) -> str:
    return ntpath.normcase(str(path.resolve(strict=False))).rstrip("\\/")


def same_canonical_path(left: Path, right: Path) -> bool:
    """Compare canonical paths with Windows case-insensitive semantics."""

    return _path_key(left) == _path_key(right)


def path_is_within(path: Path, root: Path) -> bool:
    """Return whether a canonically resolved path is at or beneath a root."""

    candidate = _path_key(path)
    parent = _path_key(root)
    try:
        return ntpath.commonpath((candidate, parent)) == parent
    except ValueError:
        return False


def _assert_disjoint_roots(left: Path, right: Path, left_name: str, right_name: str) -> None:
    if path_is_within(left, right) or path_is_within(right, left):
        raise ValueError(f"{left_name} and {right_name} must be disjoint roots")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_identity_sha256(path: Path) -> str:
    status = path.stat()
    return canonical_sha256(
        {
            "canonical_path": _path_key(path),
            "device": int(status.st_dev),
            "inode": int(status.st_ino),
        }
    )


def _run_git(root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip()


def _assert_git_root(root: Path, field: str) -> None:
    actual = Path(_run_git(root, ("rev-parse", "--show-toplevel"))).resolve(strict=True)
    if not same_canonical_path(actual, root):
        raise ValueError(f"{field} is not the exact Git worktree root")


def _assert_primary_clean(root: Path) -> None:
    if _run_git(root, ("status", "--porcelain=v1", "--untracked-files=all")):
        raise RuntimeError("primary repository must remain clean during frozen execution")


def _source_git_identity(root: Path) -> dict[str, object]:
    _assert_git_root(root, "source_root")
    status = _run_git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise RuntimeError("immutable source worktree has tracked or untracked changes")
    ignored = _run_git(root, ("ls-files", "--others", "--ignored", "--exclude-standard"))
    if ignored:
        raise RuntimeError("immutable source worktree contains ignored generated files")
    raw_paths = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
        timeout=120,
    ).stdout
    relative_paths = [item.decode("utf-8") for item in raw_paths.split(b"\0") if item]
    entries: list[dict[str, object]] = []
    total_bytes = 0
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"tracked source path is not a file: {relative}")
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {"path": relative.replace("\\", "/"), "size": size, "sha256": _file_sha256(path)}
        )
    return {
        "commit": _run_git(root, ("rev-parse", "HEAD")),
        "tree": _run_git(root, ("rev-parse", "HEAD^{tree}")),
        "tracked_file_count": len(entries),
        "tracked_bytes": total_bytes,
        "tracked_manifest_sha256": canonical_sha256(entries),
    }


def _git_ignored(primary_root: Path, path: Path) -> bool:
    if not path_is_within(path, primary_root):
        return False
    relative = path.resolve(strict=False).relative_to(primary_root.resolve(strict=True))
    completed = subprocess.run(
        ["git", "-C", str(primary_root), "check-ignore", "--quiet", str(relative)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.returncode == 0


def _model_artifact_identity(cache_root: Path, snapshot_root: Path) -> dict[str, object]:
    if not path_is_within(snapshot_root, cache_root):
        raise ValueError("model snapshot must remain beneath model_cache_root")
    entries: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(snapshot_root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if not path_is_within(resolved, cache_root):
            raise ValueError("model-cache artifact escapes model_cache_root after resolution")
        size = resolved.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(snapshot_root).as_posix(),
                "resolved_cache_path": resolved.relative_to(cache_root).as_posix(),
                "size": size,
                "sha256": _file_sha256(resolved),
            }
        )
    names = {str(item["path"]) for item in entries}
    if "config.json" not in names or "tokenizer_config.json" not in names:
        raise FileNotFoundError("model snapshot lacks configuration or tokenizer files")
    if not any(name.endswith(".safetensors") for name in names) and (
        "model.safetensors.index.json" not in names
    ):
        raise FileNotFoundError("model snapshot lacks safetensors weights")
    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "manifest_sha256": canonical_sha256(entries),
    }


def _process_environment_values(
    source_root: Path, model_cache_root: Path, artifact_root: Path
) -> dict[str, str]:
    values = dict(FROZEN_PROCESS_ENVIRONMENT)
    values.update(
        {
            "HF_HOME": str(model_cache_root),
            "PYTHONPATH": str(source_root / "src"),
            "TEMP": str(artifact_root / "temp"),
            "TMP": str(artifact_root / "temp"),
            "TMPDIR": str(artifact_root / "temp"),
        }
    )
    return values


def _command_template_sha256(python_executable: Path, artifact_root: Path) -> str:
    return canonical_sha256(
        {
            "python_executable": str(python_executable),
            "runtime_paths": str(artifact_root / "runtime_paths.json"),
            "templates": PROCESS_COMMAND_TEMPLATES,
        }
    )


@dataclass(frozen=True)
class GrpoRuntimePaths:
    """Frozen identities for every source, executable, cache, and artifact root."""

    source_root: Path
    primary_repository_root: Path
    python_executable: Path
    artifact_root: Path
    model_cache_root: Path
    model_snapshot_root: Path
    source_commit: str
    source_tree: str
    source_tracked_file_count: int
    source_tracked_bytes: int
    source_tracked_manifest_sha256: str
    source_root_identity_sha256: str
    primary_root_identity_sha256: str
    artifact_root_identity_sha256: str
    model_cache_root_identity_sha256: str
    python_executable_sha256: str
    python_executable_size_bytes: int
    model_artifact_file_count: int
    model_artifact_bytes: int
    model_artifact_manifest_sha256: str
    process_environment_sha256: str
    process_command_template_sha256: str

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": RUNTIME_PATHS_SCHEMA_VERSION,
            "contract_id": RUNTIME_PATHS_ID,
            "source_root": str(self.source_root),
            "primary_repository_root": str(self.primary_repository_root),
            "python_executable": str(self.python_executable),
            "artifact_root": str(self.artifact_root),
            "model_cache_root": str(self.model_cache_root),
            "model_snapshot_root": str(self.model_snapshot_root),
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "source_tracked_file_count": self.source_tracked_file_count,
            "source_tracked_bytes": self.source_tracked_bytes,
            "source_tracked_manifest_sha256": self.source_tracked_manifest_sha256,
            "source_root_identity_sha256": self.source_root_identity_sha256,
            "primary_root_identity_sha256": self.primary_root_identity_sha256,
            "artifact_root_identity_sha256": self.artifact_root_identity_sha256,
            "model_cache_root_identity_sha256": self.model_cache_root_identity_sha256,
            "python_executable_sha256": self.python_executable_sha256,
            "python_executable_size_bytes": self.python_executable_size_bytes,
            "model_artifact_file_count": self.model_artifact_file_count,
            "model_artifact_bytes": self.model_artifact_bytes,
            "model_artifact_manifest_sha256": self.model_artifact_manifest_sha256,
            "process_environment_sha256": self.process_environment_sha256,
            "process_command_template_sha256": self.process_command_template_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        value = self.payload()
        value["contract_sha256"] = self.contract_sha256
        return value

    def evidence(self) -> dict[str, object]:
        return {
            "runtime_path_contract_sha256": self.contract_sha256,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "source_tracked_manifest_sha256": self.source_tracked_manifest_sha256,
            "python_executable_sha256": self.python_executable_sha256,
            "model_artifact_manifest_sha256": self.model_artifact_manifest_sha256,
            "process_environment_sha256": self.process_environment_sha256,
            "process_command_template_sha256": self.process_command_template_sha256,
        }


def freeze_runtime_paths(
    *,
    source_root: Path,
    primary_repository_root: Path,
    python_executable: Path,
    artifact_root: Path,
    model_cache_root: Path,
    model_snapshot_root: Path,
    verify_current_process: bool = True,
) -> GrpoRuntimePaths:
    """Create and verify one immutable runtime-path contract."""

    source = _require_absolute(source_root, "source_root")
    primary = _require_absolute(primary_repository_root, "primary_repository_root")
    executable = _require_absolute(python_executable, "python_executable", file=True)
    artifact = _require_absolute(artifact_root, "artifact_root")
    cache = _require_absolute(model_cache_root, "model_cache_root")
    snapshot = _require_absolute(model_snapshot_root, "model_snapshot_root")
    _assert_disjoint_roots(source, primary, "source_root", "primary_repository_root")
    _assert_disjoint_roots(source, artifact, "source_root", "artifact_root")
    _assert_disjoint_roots(primary, artifact, "primary_repository_root", "artifact_root")
    _assert_disjoint_roots(source, cache, "source_root", "model_cache_root")
    _assert_disjoint_roots(artifact, cache, "artifact_root", "model_cache_root")
    if not path_is_within(executable, primary):
        raise ValueError("python_executable must remain beneath primary_repository_root")
    _assert_git_root(primary, "primary_repository_root")
    _assert_primary_clean(primary)
    if path_is_within(cache, primary) and not _git_ignored(primary, cache):
        raise ValueError("model_cache_root beneath primary_repository_root must be Git ignored")
    source_identity = _source_git_identity(source)
    model_identity = _model_artifact_identity(cache, snapshot)
    executable_hash = _file_sha256(executable)
    executable_size = executable.stat().st_size
    if (
        executable_hash != FROZEN_PYTHON_EXECUTABLE_SHA256
        or executable_size != FROZEN_PYTHON_EXECUTABLE_SIZE_BYTES
    ):
        raise RuntimeError("configured Python executable differs from the frozen binary")
    environment_values = _process_environment_values(source, cache, artifact)
    value = GrpoRuntimePaths(
        source_root=source,
        primary_repository_root=primary,
        python_executable=executable,
        artifact_root=artifact,
        model_cache_root=cache,
        model_snapshot_root=snapshot,
        source_commit=cast(str, source_identity["commit"]),
        source_tree=cast(str, source_identity["tree"]),
        source_tracked_file_count=cast(int, source_identity["tracked_file_count"]),
        source_tracked_bytes=cast(int, source_identity["tracked_bytes"]),
        source_tracked_manifest_sha256=cast(str, source_identity["tracked_manifest_sha256"]),
        source_root_identity_sha256=_directory_identity_sha256(source),
        primary_root_identity_sha256=_directory_identity_sha256(primary),
        artifact_root_identity_sha256=_directory_identity_sha256(artifact),
        model_cache_root_identity_sha256=_directory_identity_sha256(cache),
        python_executable_sha256=executable_hash,
        python_executable_size_bytes=executable_size,
        model_artifact_file_count=cast(int, model_identity["file_count"]),
        model_artifact_bytes=cast(int, model_identity["total_bytes"]),
        model_artifact_manifest_sha256=cast(str, model_identity["manifest_sha256"]),
        process_environment_sha256=canonical_sha256(environment_values),
        process_command_template_sha256=_command_template_sha256(executable, artifact),
    )
    if verify_current_process:
        validate_runtime_paths(value)
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def runtime_paths_from_dict(value: Mapping[str, object]) -> GrpoRuntimePaths:
    payload = dict(value)
    declared = _require_sha256(payload.pop("contract_sha256", None), "contract_sha256")
    expected_fields = set(GrpoRuntimePaths.__dataclass_fields__)
    if payload.pop("schema_version", None) != RUNTIME_PATHS_SCHEMA_VERSION:
        raise ValueError("runtime-path schema version differs")
    if payload.pop("contract_id", None) != RUNTIME_PATHS_ID:
        raise ValueError("runtime-path contract ID differs")
    if set(payload) != expected_fields:
        raise ValueError("runtime-path contract fields differ from the frozen schema")
    path_fields = {
        "source_root",
        "primary_repository_root",
        "python_executable",
        "artifact_root",
        "model_cache_root",
        "model_snapshot_root",
    }
    integer_fields = {
        "source_tracked_file_count",
        "source_tracked_bytes",
        "python_executable_size_bytes",
        "model_artifact_file_count",
        "model_artifact_bytes",
    }
    hash_fields = {
        "source_tracked_manifest_sha256",
        "source_root_identity_sha256",
        "primary_root_identity_sha256",
        "artifact_root_identity_sha256",
        "model_cache_root_identity_sha256",
        "python_executable_sha256",
        "model_artifact_manifest_sha256",
        "process_environment_sha256",
        "process_command_template_sha256",
    }
    arguments: dict[str, object] = {}
    for field in expected_fields:
        item = payload[field]
        if field in path_fields:
            if not isinstance(item, str) or not Path(item).is_absolute():
                raise ValueError(f"{field} must contain an absolute path")
            arguments[field] = Path(item)
        elif field in integer_fields:
            arguments[field] = _require_int(item, field)
        elif field in hash_fields:
            arguments[field] = _require_sha256(item, field)
        elif not isinstance(item, str) or not item:
            raise ValueError(f"{field} must contain non-empty text")
        else:
            arguments[field] = item
    contract = GrpoRuntimePaths(**arguments)  # type: ignore[arg-type]
    if contract.contract_sha256 != declared:
        raise ValueError("runtime-path contract self-hash differs")
    return contract


def write_runtime_paths(path: Path, runtime_paths: GrpoRuntimePaths) -> None:
    assert_artifact_path(runtime_paths, path, "runtime-path contract")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime-path contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(runtime_paths.as_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_runtime_paths(path: Path, *, verify: bool = True) -> GrpoRuntimePaths:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("runtime-path contract is not valid JSON") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("runtime-path contract must contain a string-keyed object")
    contract = runtime_paths_from_dict(cast(dict[str, object], value))
    assert_artifact_path(contract, path, "runtime-path contract")
    if verify:
        validate_runtime_paths(contract)
    return contract


def assert_artifact_path(runtime_paths: GrpoRuntimePaths, path: Path, description: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{description} path must be absolute")
    resolved = path.resolve(strict=False)
    if not path_is_within(resolved, runtime_paths.artifact_root):
        raise ValueError(f"{description} must stay inside the configured artifact_root")
    if path_is_within(resolved, runtime_paths.source_root) or path_is_within(
        resolved, runtime_paths.primary_repository_root
    ):
        raise ValueError(f"{description} cannot be written inside a source repository")
    return resolved


def assert_source_path(runtime_paths: GrpoRuntimePaths, path: Path, description: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{description} path must be absolute")
    resolved = path.resolve(strict=True)
    if not path_is_within(resolved, runtime_paths.source_root):
        raise ValueError(f"{description} must stay inside the immutable source_root")
    return resolved


def frozen_process_environment(
    runtime_paths: GrpoRuntimePaths,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.pop("PYTHONHOME", None)
    environment.update(
        _process_environment_values(
            runtime_paths.source_root,
            runtime_paths.model_cache_root,
            runtime_paths.artifact_root,
        )
    )
    return environment


def frozen_environment_sha256(runtime_paths: GrpoRuntimePaths) -> str:
    values = _process_environment_values(
        runtime_paths.source_root,
        runtime_paths.model_cache_root,
        runtime_paths.artifact_root,
    )
    return canonical_sha256(values)


def python_module_command(
    runtime_paths: GrpoRuntimePaths, module: str, arguments: Sequence[str]
) -> list[str]:
    if not module.startswith("foundry."):
        raise ValueError("frozen Python module must belong to Foundry")
    return [str(runtime_paths.python_executable), "-m", module, *arguments]


def process_command_sha256(command: Sequence[str]) -> str:
    return canonical_sha256(list(command))


def validate_foundry_import(runtime_paths: GrpoRuntimePaths) -> dict[str, str]:
    import foundry

    raw_path = getattr(foundry, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("Foundry import does not expose a source path")
    imported = Path(raw_path).resolve(strict=True)
    expected_root = (runtime_paths.source_root / "src").resolve(strict=True)
    if not path_is_within(imported, expected_root):
        raise RuntimeError(f"Foundry imported outside the immutable source worktree: {imported}")
    return {
        "foundry_import_path": str(imported),
        "foundry_import_path_sha256": hashlib.sha256(str(imported).encode("utf-8")).hexdigest(),
    }


def _assert_process_environment(runtime_paths: GrpoRuntimePaths) -> None:
    expected = _process_environment_values(
        runtime_paths.source_root,
        runtime_paths.model_cache_root,
        runtime_paths.artifact_root,
    )
    actual = {key: os.environ.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"frozen process environment differs: {actual}")
    if frozen_environment_sha256(runtime_paths) != runtime_paths.process_environment_sha256:
        raise RuntimeError("frozen process-environment hash differs")
    if os.environ.get("PYTHONHASHSEED") != "20260720" or not bool(sys.flags.hash_randomization):
        raise RuntimeError("Python hash seed or hash randomization differs")
    if not bool(sys.dont_write_bytecode):
        raise RuntimeError("PYTHONDONTWRITEBYTECODE was not effective before interpreter launch")


def validate_runtime_paths(runtime_paths: GrpoRuntimePaths) -> dict[str, object]:
    """Reconstruct every frozen path identity and current-process invariant."""

    _assert_disjoint_roots(
        runtime_paths.source_root,
        runtime_paths.primary_repository_root,
        "source_root",
        "primary_repository_root",
    )
    _assert_disjoint_roots(
        runtime_paths.source_root, runtime_paths.artifact_root, "source_root", "artifact_root"
    )
    _assert_disjoint_roots(
        runtime_paths.primary_repository_root,
        runtime_paths.artifact_root,
        "primary_repository_root",
        "artifact_root",
    )
    _assert_disjoint_roots(
        runtime_paths.source_root,
        runtime_paths.model_cache_root,
        "source_root",
        "model_cache_root",
    )
    _assert_disjoint_roots(
        runtime_paths.artifact_root,
        runtime_paths.model_cache_root,
        "artifact_root",
        "model_cache_root",
    )
    identities = {
        "source_root_identity_sha256": _directory_identity_sha256(runtime_paths.source_root),
        "primary_root_identity_sha256": _directory_identity_sha256(
            runtime_paths.primary_repository_root
        ),
        "artifact_root_identity_sha256": _directory_identity_sha256(runtime_paths.artifact_root),
        "model_cache_root_identity_sha256": _directory_identity_sha256(
            runtime_paths.model_cache_root
        ),
    }
    expected_identities = {key: getattr(runtime_paths, key) for key in identities}
    if identities != expected_identities:
        raise RuntimeError("one or more frozen runtime roots were replaced")
    _assert_git_root(runtime_paths.primary_repository_root, "primary_repository_root")
    _assert_primary_clean(runtime_paths.primary_repository_root)
    source_identity = _source_git_identity(runtime_paths.source_root)
    expected_source = {
        "commit": runtime_paths.source_commit,
        "tree": runtime_paths.source_tree,
        "tracked_file_count": runtime_paths.source_tracked_file_count,
        "tracked_bytes": runtime_paths.source_tracked_bytes,
        "tracked_manifest_sha256": runtime_paths.source_tracked_manifest_sha256,
    }
    if source_identity != expected_source:
        raise RuntimeError("immutable source commit, tree, or tracked bytes changed")
    executable = runtime_paths.python_executable.resolve(strict=True)
    if not path_is_within(executable, runtime_paths.primary_repository_root):
        raise RuntimeError("configured Python executable escaped primary_repository_root")
    if not same_canonical_path(Path(sys.executable), executable):
        raise RuntimeError("running interpreter differs from configured python_executable")
    if platform.python_implementation() != FROZEN_PYTHON_IMPLEMENTATION or (
        platform.python_version() != FROZEN_PYTHON_VERSION
    ):
        raise RuntimeError("running Python implementation or version differs")
    if (
        runtime_paths.python_executable_sha256 != FROZEN_PYTHON_EXECUTABLE_SHA256
        or runtime_paths.python_executable_size_bytes != FROZEN_PYTHON_EXECUTABLE_SIZE_BYTES
        or _file_sha256(executable) != runtime_paths.python_executable_sha256
        or executable.stat().st_size != runtime_paths.python_executable_size_bytes
    ):
        raise RuntimeError("configured Python executable changed after contract freeze")
    if path_is_within(runtime_paths.model_cache_root, runtime_paths.primary_repository_root) and (
        not _git_ignored(runtime_paths.primary_repository_root, runtime_paths.model_cache_root)
    ):
        raise RuntimeError("configured model cache beneath primary repository is not Git ignored")
    model_identity = _model_artifact_identity(
        runtime_paths.model_cache_root, runtime_paths.model_snapshot_root
    )
    expected_model = {
        "file_count": runtime_paths.model_artifact_file_count,
        "total_bytes": runtime_paths.model_artifact_bytes,
        "manifest_sha256": runtime_paths.model_artifact_manifest_sha256,
    }
    if model_identity != expected_model:
        raise RuntimeError("read-only model artifacts changed after contract freeze")
    _assert_process_environment(runtime_paths)
    if (
        _command_template_sha256(runtime_paths.python_executable, runtime_paths.artifact_root)
        != runtime_paths.process_command_template_sha256
    ):
        raise RuntimeError("frozen process-command template hash differs")
    import_evidence = validate_foundry_import(runtime_paths)
    return {**runtime_paths.evidence(), **import_evidence, "validation_passed": True}


def verify_fresh_process_import(runtime_paths: GrpoRuntimePaths) -> dict[str, str]:
    """Launch the configured interpreter and prove its Foundry import source."""

    command = [
        str(runtime_paths.python_executable),
        "-c",
        (
            "import foundry,os,sys; "
            "print(foundry.__file__); "
            "print(os.environ['PYTHONHASHSEED']); "
            "print(sys.flags.hash_randomization)"
        ),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=runtime_paths.source_root,
        env=frozen_process_environment(runtime_paths),
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 3 or lines[1:] != ["20260720", "1"]:
        raise RuntimeError("fresh-process import probe returned unexpected environment evidence")
    imported = Path(lines[0]).resolve(strict=True)
    if not path_is_within(imported, runtime_paths.source_root / "src"):
        raise RuntimeError("fresh process imported Foundry outside immutable source_root")
    return {
        "foundry_import_path": str(imported),
        "process_command_sha256": process_command_sha256(command),
        "process_environment_sha256": frozen_environment_sha256(runtime_paths),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--source-root", type=Path, required=True)
    freeze.add_argument("--primary-repository-root", type=Path, required=True)
    freeze.add_argument("--python-executable", type=Path, required=True)
    freeze.add_argument("--artifact-root", type=Path, required=True)
    freeze.add_argument("--model-cache-root", type=Path, required=True)
    freeze.add_argument("--model-snapshot-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--runtime-paths", type=Path, required=True)
    probe = subparsers.add_parser("probe-import")
    probe.add_argument("--runtime-paths", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        runtime_paths = freeze_runtime_paths(
            source_root=args.source_root,
            primary_repository_root=args.primary_repository_root,
            python_executable=args.python_executable,
            artifact_root=args.artifact_root,
            model_cache_root=args.model_cache_root,
            model_snapshot_root=args.model_snapshot_root,
        )
        write_runtime_paths(args.output, runtime_paths)
        result: object = runtime_paths.as_dict()
    else:
        runtime_paths = load_runtime_paths(args.runtime_paths)
        result = (
            validate_runtime_paths(runtime_paths)
            if args.command == "validate"
            else verify_fresh_process_import(runtime_paths)
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
