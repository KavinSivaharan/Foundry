"""Narrow, fail-closed PyYAML metadata-exception audit for Phase 2 QLoRA."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from foundry.training.assistant_only_config import load_assistant_only_recipe
from foundry.training.config import canonical_sha256, load_qlora_recipe
from foundry.training.token_matched_config import load_token_matched_recipe

EXCEPTION_ID = "foundry-pyyaml-metadata-exception-v1"
PROJECT_REQUIRED_VERSION = "6.0.2"
TRAINING_INSTALLED_VERSION = "6.0.3"
EXPECTED_DISCREPANCY = (
    "foundry-post-training 0.1.0 has requirement PyYAML==6.0.2, but you have pyyaml 6.0.3."
)
PIP_PATTERN = re.compile(
    r"^foundry-post-training 0\.1\.0 has requirement PyYAML==6\.0\.2, "
    r"but you have pyyaml 6\.0\.3\.$",
    re.IGNORECASE,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def normalize_pip_discrepancies(stdout: str, stderr: str) -> list[str]:
    """Return non-empty normalized pip-check discrepancy lines."""

    return [
        line.strip()
        for line in (stdout + "\n" + stderr).replace("\r\n", "\n").splitlines()
        if line.strip()
    ]


def validate_narrow_exception(
    *,
    discrepancies: list[str],
    installed_version: str,
    metadata_present: bool,
    source_hashes_equal: bool,
    parsed_structures_equal: bool,
    typed_configs_equal: bool,
    replay_equal: bool,
) -> None:
    """Reject every case except the single frozen and fully equivalent mismatch."""

    if discrepancies != [EXPECTED_DISCREPANCY] or not PIP_PATTERN.fullmatch(
        discrepancies[0] if discrepancies else ""
    ):
        raise ValueError("pip check has a discrepancy outside the narrow exception")
    if installed_version != TRAINING_INSTALLED_VERSION:
        raise ValueError("installed PyYAML version is outside the narrow exception")
    gates = {
        "metadata_present": metadata_present,
        "source_hashes_equal": source_hashes_equal,
        "parsed_structures_equal": parsed_structures_equal,
        "typed_configs_equal": typed_configs_equal,
        "replay_equal": replay_equal,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError(f"PyYAML exception gates failed: {failed}")


def _package_evidence() -> dict[str, object]:
    distribution = importlib.metadata.distribution("PyYAML")
    yaml_root = Path(yaml.__file__).resolve().parent
    package_files = sorted(
        path
        for path in yaml_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".py", ".pyc", ".pyd", ".so"}
        and path.suffix.lower() != ".pyc"
    )
    files = [
        {
            "path": path.relative_to(yaml_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in package_files
    ]
    metadata = {
        key: distribution.metadata.get_all(key)
        for key in (
            "Metadata-Version",
            "Name",
            "Version",
            "Summary",
            "Requires-Python",
            "License",
            "License-File",
        )
    }
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "yaml_version": str(yaml.__version__),
        "yaml_file": str(Path(yaml.__file__).resolve()),
        "package_files": files,
        "package_files_sha256": canonical_sha256(files),
        "metadata": metadata,
        "metadata_sha256": canonical_sha256(metadata),
    }


def _typed_projection(path: Path) -> list[dict[str, object]]:
    loaders = (
        ("assistant_only", load_assistant_only_recipe),
        ("qlora", load_qlora_recipe),
        ("token_matched", load_token_matched_recipe),
    )
    projections: list[dict[str, object]] = []
    for name, loader in loaders:
        try:
            loaded = loader(path)
        except Exception as error:  # loader rejection is part of equivalence evidence
            projections.append(
                {
                    "loader": name,
                    "status": "rejected",
                    "exception": type(error).__name__,
                    "message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
                }
            )
        else:
            value = _jsonable(dataclasses.asdict(loaded))
            projections.append(
                {
                    "loader": name,
                    "status": "loaded",
                    "canonical_sha256": canonical_sha256(value),
                }
            )
    return projections


def _parser_semantics() -> dict[str, object]:
    fixture = """
anchor: &shared
  integer: 7
  float: 1.25
  boolean: true
  null_value: null
  string: "7"
alias: *shared
duplicate: first
duplicate: second
ordered:
  - alpha
  - beta
"""
    parsed = yaml.safe_load(fixture)
    canonical = _canonical(parsed)
    return {
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "duplicate_value": parsed["duplicate"],
        "alias_equal": parsed["anchor"] == parsed["alias"],
        "scalar_types": {key: type(value).__name__ for key, value in parsed["anchor"].items()},
        "ordered": parsed["ordered"],
    }


def audit_yaml_files(paths: list[Path]) -> dict[str, object]:
    """Audit source bytes, SafeLoader output, and real training loaders."""

    audited: list[dict[str, object]] = []
    for path in paths:
        source = path.read_bytes()
        try:
            parsed = yaml.safe_load(source.decode("utf-8"))
        except Exception as error:
            parse: dict[str, object] = {
                "status": "exception",
                "exception": type(error).__name__,
                "message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            }
        else:
            canonical = _canonical(parsed)
            parse = {
                "status": "parsed",
                "canonical_bytes": len(canonical),
                "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
            }
        audited.append(
            {
                "path": path.as_posix(),
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "safe_load": parse,
                "typed_loaders": _typed_projection(path),
            }
        )
    package = _package_evidence()
    result: dict[str, object] = {
        "schema_version": 1,
        "package": package,
        "parser_semantics": _parser_semantics(),
        "audited_files": audited,
        "audited_file_count": len(audited),
    }
    result["audit_sha256"] = canonical_sha256(result)
    return result


def compare_audits(
    general: dict[str, Any],
    training: dict[str, Any],
    *,
    general_replay: dict[str, Any],
    training_replay: dict[str, Any],
    pip_stdout: str,
    pip_stderr: str,
    pip_exit_code: int,
) -> dict[str, object]:
    """Compare two independently produced audits and grant only the narrow exception."""

    general_files = general["audited_files"]
    training_files = training["audited_files"]
    source_equal = [row["source_sha256"] for row in general_files] == [
        row["source_sha256"] for row in training_files
    ]
    parsed_equal = [row["safe_load"] for row in general_files] == [
        row["safe_load"] for row in training_files
    ]
    typed_equal = [row["typed_loaders"] for row in general_files] == [
        row["typed_loaders"] for row in training_files
    ]
    semantics_equal = general["parser_semantics"] == training["parser_semantics"]
    paths_equal = [row["path"] for row in general_files] == [row["path"] for row in training_files]
    discrepancies = normalize_pip_discrepancies(pip_stdout, pip_stderr)
    validate_narrow_exception(
        discrepancies=discrepancies,
        installed_version=str(training["package"]["yaml_version"]),
        metadata_present=bool(training["package"]["metadata"]),
        source_hashes_equal=source_equal and paths_equal,
        parsed_structures_equal=parsed_equal and semantics_equal,
        typed_configs_equal=typed_equal,
        replay_equal=general == general_replay and training == training_replay,
    )
    if pip_exit_code != 1:
        raise ValueError("training pip check exit code differs")
    evidence: dict[str, object] = {
        "schema_version": 1,
        "exception_id": EXCEPTION_ID,
        "project_required_version": PROJECT_REQUIRED_VERSION,
        "training_installed_version": TRAINING_INSTALLED_VERSION,
        "pip_check_exit_code": pip_exit_code,
        "pip_check_discrepancies": discrepancies,
        "pip_check_evidence_sha256": hashlib.sha256(
            (pip_stdout + pip_stderr).replace("\r\n", "\n").encode("utf-8")
        ).hexdigest(),
        "general_package": general["package"],
        "training_package": training["package"],
        "audited_file_count": len(general_files),
        "audited_files": [
            {
                "path": left["path"],
                "source_sha256": left["source_sha256"],
                "general_parsed_sha256": left["safe_load"].get("canonical_sha256"),
                "training_parsed_sha256": right["safe_load"].get("canonical_sha256"),
                "equivalent": left["safe_load"] == right["safe_load"],
            }
            for left, right in zip(general_files, training_files, strict=True)
        ],
        "parser_equivalence": parsed_equal and paths_equal and source_equal,
        "parser_semantics_equivalence": semantics_equal,
        "typed_config_equivalence": typed_equal and paths_equal,
        "general_audit_sha256": general["audit_sha256"],
        "training_audit_sha256": training["audit_sha256"],
        "deterministic_replay_equal": general == general_replay and training == training_replay,
        "exception_decision": "pass",
    }
    evidence["combined_parser_equivalence_sha256"] = canonical_sha256(evidence["audited_files"])
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def validate_evidence(evidence: dict[str, Any]) -> None:
    """Reject altered or broadened published exception evidence."""

    supplied = evidence.get("evidence_sha256")
    payload = dict(evidence)
    payload.pop("evidence_sha256", None)
    if supplied != canonical_sha256(payload):
        raise ValueError("PyYAML exception evidence hash differs")
    if (
        evidence.get("exception_id") != EXCEPTION_ID
        or evidence.get("exception_decision") != "pass"
        or evidence.get("pip_check_discrepancies") != [EXPECTED_DISCREPANCY]
    ):
        raise ValueError("published exception evidence exceeds the narrow gate")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("audit document must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--repository-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--general", type=Path, required=True)
    compare.add_argument("--general-replay", type=Path, required=True)
    compare.add_argument("--training", type=Path, required=True)
    compare.add_argument("--training-replay", type=Path, required=True)
    compare.add_argument("--pip-stdout", type=Path, required=True)
    compare.add_argument("--pip-stderr", type=Path, required=True)
    compare.add_argument("--pip-exit-code", type=int, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        listing = subprocess.run(
            ["git", "ls-files", "*.yaml", "*.yml"],
            cwd=args.repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        paths = [args.repository_root / line for line in listing.splitlines() if line]
        result = audit_yaml_files(paths)
    else:
        result = compare_audits(
            _read_json(args.general),
            _read_json(args.training),
            general_replay=_read_json(args.general_replay),
            training_replay=_read_json(args.training_replay),
            pip_stdout=args.pip_stdout.read_text(encoding="utf-8"),
            pip_stderr=args.pip_stderr.read_text(encoding="utf-8"),
            pip_exit_code=args.pip_exit_code,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
