"""Audited Windows operational environment for the Phase 2 QLoRA child."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from foundry.phase2.launch_contract import ALLOWLISTED_ENVIRONMENT
from foundry.training.config import canonical_sha256

CONTRACT_ID = "foundry-vetted-qlora-windows-operational-env-v1"
OPERATIONAL_ALLOWLIST = (
    "ALLUSERSPROFILE",
    "APPDATA",
    "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)",
    "COMMONPROGRAMW6432",
    "COMSPEC",
    "DRIVERDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL",
    "PROCESSOR_REVISION",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "PUBLIC",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERDOMAIN",
    "USERNAME",
    "USERPROFILE",
    "WINDIR",
)
REQUIRED_OPERATIONAL = ("COMSPEC", "PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP", "WINDIR")
SECRET_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def build_child_environment(parent: Mapping[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Copy only fixed operational names, then override deterministic variables."""

    by_upper: dict[str, tuple[str, str]] = {}
    for name, value in parent.items():
        upper = name.upper()
        if upper in by_upper and by_upper[upper][0] != name:
            raise ValueError(f"duplicate case-insensitive environment key: {upper}")
        by_upper[upper] = (name, value)
    operational = {name: by_upper[name][1] for name in OPERATIONAL_ALLOWLIST if name in by_upper}
    missing = sorted(set(REQUIRED_OPERATIONAL) - set(operational))
    if missing:
        raise ValueError(f"required Windows operational variables are missing: {missing}")
    child = dict(operational)
    child.update(ALLOWLISTED_ENVIRONMENT)
    unauthorized = set(child) - set(OPERATIONAL_ALLOWLIST) - set(ALLOWLISTED_ENVIRONMENT)
    if unauthorized:
        raise ValueError(f"unauthorized child variables: {sorted(unauthorized)}")
    if any(any(fragment in name for fragment in SECRET_FRAGMENTS) for name in operational):
        raise ValueError("secret-looking operational variable entered child environment")
    absent = [name for name in OPERATIONAL_ALLOWLIST if name not in operational]
    value_hashes = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in sorted(operational.items())
    }
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "ordered_allowlist": list(OPERATIONAL_ALLOWLIST),
        "present": sorted(operational),
        "absent": absent,
        "value_sha256": value_hashes,
        "operational_environment_sha256": canonical_sha256(operational),
        "deterministic_projection_sha256": canonical_sha256(ALLOWLISTED_ENVIRONMENT),
        "operational_projection_sha256": canonical_sha256(operational),
        "combined_child_environment_sha256": canonical_sha256(child),
        "parent_to_child_equal": all(
            operational[name] == by_upper[name][1] for name in operational
        ),
        "secret_name_scan": "pass",
        "unauthorized_variable_count": 0,
    }
    evidence["environment_evidence_sha256"] = canonical_sha256(evidence)
    return child, evidence


def validate_child_environment(child: Mapping[str, str], evidence: Mapping[str, Any]) -> None:
    allowed = set(OPERATIONAL_ALLOWLIST) | set(ALLOWLISTED_ENVIRONMENT)
    if set(child) - allowed:
        raise ValueError("child contains an unauthorized environment variable")
    deterministic = {name: child.get(name, "") for name in ALLOWLISTED_ENVIRONMENT}
    operational = {name: child[name] for name in OPERATIONAL_ALLOWLIST if name in child}
    if deterministic != ALLOWLISTED_ENVIRONMENT:
        raise ValueError("deterministic child environment differs")
    if canonical_sha256(operational) != evidence["operational_environment_sha256"]:
        raise ValueError("operational child environment differs")
    if canonical_sha256(dict(child)) != evidence["combined_child_environment_sha256"]:
        raise ValueError("combined child environment differs")


def tracked_evidence(parent: Mapping[str, str]) -> dict[str, Any]:
    _, evidence = build_child_environment(parent)
    return cast(dict[str, Any], json.loads(json.dumps(evidence)))


def run_import_preflight(repository_root: Path) -> subprocess.CompletedProcess[str]:
    root = repository_root.resolve()
    child, _ = build_child_environment(os.environ)
    argv = [
        str(root / ".venv-training" / "Scripts" / "python.exe"),
        "-m",
        "foundry.phase2.windows_import_preflight",
        "--environment-evidence",
        str(root / "results" / "phase2_vetted_corpus" / "windows_operational_environment.json"),
        "--output",
        str(
            root
            / "results"
            / "raw"
            / "phase2_vetted_corpus"
            / "windows_import_preflight"
            / "result.json"
        ),
    ]
    return subprocess.run(
        argv,
        shell=False,
        env=child,
        cwd=root / "src",
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_import_preflight(args.repository_root)
    print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError(result.stderr)


if __name__ == "__main__":
    main()
