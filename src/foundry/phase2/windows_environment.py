"""Audited Windows operational environment for the Phase 2 QLoRA child."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any, cast

from foundry.phase2.launch_contract import ALLOWLISTED_ENVIRONMENT
from foundry.training.config import canonical_sha256

CONTRACT_ID_V1 = "foundry-vetted-qlora-windows-operational-env-v1"
CONTRACT_ID_V2 = "foundry-vetted-qlora-windows-operational-env-v2"
CONTRACT_ID = CONTRACT_ID_V2
EXPECTED_V1_PATH_SHA256 = "fc435730951754aff2809a185479cac351afa0a4c959bad0db533dfcafcd06ec"
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
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token=|secret=|password=|credential=|://[^/;\s]+:[^/@;\s]+@)"
)


def sha256_text(value: str) -> str:
    """Hash an environment value without normalizing it."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def analyze_path(value: str) -> dict[str, Any]:
    """Return content-free evidence about one exact Windows PATH value."""

    components = value.split(";")
    normalized = [component.strip().rstrip("\\/").casefold() for component in components]
    counts: dict[str, int] = {}
    for component in normalized:
        counts[component] = counts.get(component, 0) + 1
    duplicate_hashes = sorted(sha256_text(item) for item, count in counts.items() if count > 1)
    empty_indices = [index for index, component in enumerate(components) if not component.strip()]
    relative_hashes = [
        sha256_text(component)
        for component in components
        if component.strip() and not PureWindowsPath(component.strip()).is_absolute()
    ]
    control_hashes = [
        sha256_text(component)
        for component in components
        if any(ord(character) < 32 or ord(character) == 127 for character in component)
    ]
    secret_hashes = [
        sha256_text(component)
        for component in components
        if SECRET_VALUE_PATTERN.search(component) is not None
    ]
    return {
        "path_sha256": sha256_text(value),
        "component_count": len(components),
        "ordered_component_sha256": [sha256_text(component) for component in components],
        "duplicate_component_sha256": duplicate_hashes,
        "empty_component_indices": empty_indices,
        "relative_component_sha256": relative_hashes,
        "control_character_component_sha256": control_hashes,
        "secret_like_component_sha256": secret_hashes,
        "safe": not (
            duplicate_hashes or empty_indices or relative_hashes or control_hashes or secret_hashes
        ),
    }


def validate_path(value: str) -> dict[str, Any]:
    """Reject PATH forms prohibited by the frozen Windows contract."""

    evidence = analyze_path(value)
    failures = [
        name
        for name in (
            "duplicate_component_sha256",
            "empty_component_indices",
            "relative_component_sha256",
            "control_character_component_sha256",
            "secret_like_component_sha256",
        )
        if evidence[name]
    ]
    if failures:
        raise ValueError(f"unsafe PATH evidence: {failures}")
    return evidence


def capture_operational_environment(
    parent: Mapping[str, str], *, frozen_path: str | None = None
) -> dict[str, str]:
    """Read each allowlisted value once, optionally without consulting parent PATH."""

    names_by_upper: dict[str, str] = {}
    for name in parent:
        upper = name.upper()
        if upper in names_by_upper and names_by_upper[upper] != name:
            raise ValueError(f"duplicate case-insensitive environment key: {upper}")
        names_by_upper[upper] = name
    operational: dict[str, str] = {}
    for name in OPERATIONAL_ALLOWLIST:
        if name == "PATH" and frozen_path is not None:
            operational[name] = frozen_path
        elif name in names_by_upper:
            operational[name] = parent[names_by_upper[name]]
    missing = sorted(set(REQUIRED_OPERATIONAL) - set(operational))
    if missing:
        raise ValueError(f"required Windows operational variables are missing: {missing}")
    validate_path(operational["PATH"])
    if any(any(fragment in name for fragment in SECRET_FRAGMENTS) for name in operational):
        raise ValueError("secret-looking operational variable entered child environment")
    return operational


def _build_evidence(
    operational: Mapping[str, str],
    *,
    contract_id: str,
    parent_capture_timestamp_utc: str | None = None,
    parent_to_frozen_equal: bool | None = None,
    raw_evidence_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    unauthorized_operational = set(operational) - set(OPERATIONAL_ALLOWLIST)
    if unauthorized_operational:
        raise ValueError(f"unauthorized operational variables: {sorted(unauthorized_operational)}")
    missing = sorted(set(REQUIRED_OPERATIONAL) - set(operational))
    if missing:
        raise ValueError(f"required Windows operational variables are missing: {missing}")
    path_evidence = validate_path(operational["PATH"])
    frozen = dict(operational)
    child = dict(frozen)
    child.update(ALLOWLISTED_ENVIRONMENT)
    unauthorized = set(child) - set(OPERATIONAL_ALLOWLIST) - set(ALLOWLISTED_ENVIRONMENT)
    if unauthorized:
        raise ValueError(f"unauthorized child variables: {sorted(unauthorized)}")
    absent = [name for name in OPERATIONAL_ALLOWLIST if name not in frozen]
    value_hashes = {name: sha256_text(value) for name, value in sorted(frozen.items())}
    evidence: dict[str, Any] = {
        "schema_version": 2 if contract_id == CONTRACT_ID_V2 else 1,
        "contract_id": contract_id,
        "ordered_allowlist": list(OPERATIONAL_ALLOWLIST),
        "present": sorted(frozen),
        "absent": absent,
        "value_sha256": value_hashes,
        "path": path_evidence,
        "operational_environment_sha256": canonical_sha256(frozen),
        "deterministic_value_sha256": {
            name: sha256_text(value) for name, value in sorted(ALLOWLISTED_ENVIRONMENT.items())
        },
        "deterministic_projection_sha256": canonical_sha256(ALLOWLISTED_ENVIRONMENT),
        "operational_projection_sha256": canonical_sha256(frozen),
        "combined_child_environment_sha256": canonical_sha256(child),
        "frozen_to_child_equal": all(child[name] == frozen[name] for name in frozen),
        "secret_name_scan": "pass",
        "unauthorized_variable_count": 0,
    }
    if parent_capture_timestamp_utc is not None:
        evidence["parent_capture_timestamp_utc"] = parent_capture_timestamp_utc
    if parent_to_frozen_equal is not None:
        evidence["parent_to_frozen_equal"] = parent_to_frozen_equal
    if raw_evidence_sha256 is not None:
        evidence["raw_evidence_sha256"] = raw_evidence_sha256
    if contract_id == CONTRACT_ID_V2:
        evidence["v2_contract_sha256"] = canonical_sha256(evidence)
    evidence["environment_evidence_sha256"] = canonical_sha256(evidence)
    return child, evidence


def build_child_from_frozen_operational(
    operational: Mapping[str, str],
    *,
    contract_id: str = CONTRACT_ID,
    parent_capture_timestamp_utc: str | None = None,
    parent_to_frozen_equal: bool | None = None,
    raw_evidence_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build a child only from already captured operational values."""

    return _build_evidence(
        operational,
        contract_id=contract_id,
        parent_capture_timestamp_utc=parent_capture_timestamp_utc,
        parent_to_frozen_equal=parent_to_frozen_equal,
        raw_evidence_sha256=raw_evidence_sha256,
    )


def build_child_environment(
    parent: Mapping[str, str],
    *,
    frozen_path: str | None = None,
    contract_id: str = CONTRACT_ID,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Capture allowlisted values, then override deterministic variables."""

    operational = capture_operational_environment(parent, frozen_path=frozen_path)
    return _build_evidence(
        operational,
        contract_id=contract_id,
        parent_to_frozen_equal=frozen_path is None,
    )


def build_case1_environment(
    parent: Mapping[str, str],
    *,
    frozen_path: str,
    expected_path_sha256: str = EXPECTED_V1_PATH_SHA256,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Restore a v1 PATH without consulting the current parent PATH value."""

    if sha256_text(frozen_path) != expected_path_sha256:
        raise ValueError("frozen v1 PATH hash differs")
    operational = capture_operational_environment(parent, frozen_path=frozen_path)
    return _build_evidence(
        operational,
        contract_id=CONTRACT_ID_V1,
        parent_to_frozen_equal=False,
    )


def freeze_v2_environment(
    parent: Mapping[str, str], *, parent_capture_timestamp_utc: str
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Capture Case 2 once and return raw, child, and tracked evidence."""

    operational = capture_operational_environment(parent)
    raw: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID_V2,
        "parent_capture_timestamp_utc": parent_capture_timestamp_utc,
        "operational_environment": operational,
    }
    raw_hash = canonical_sha256(raw)
    child, tracked = _build_evidence(
        operational,
        contract_id=CONTRACT_ID_V2,
        parent_capture_timestamp_utc=parent_capture_timestamp_utc,
        parent_to_frozen_equal=True,
        raw_evidence_sha256=raw_hash,
    )
    return raw, child, tracked


def reconstruct_v2_environment(
    raw: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Rebuild the content-free v2 contract from ignored raw evidence."""

    if raw.get("contract_id") != CONTRACT_ID_V2 or raw.get("schema_version") != 1:
        raise ValueError("raw v2 environment contract identity differs")
    operational_value = raw.get("operational_environment")
    if not isinstance(operational_value, dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in operational_value.items()
    ):
        raise ValueError("raw v2 operational environment is malformed")
    operational = cast(dict[str, str], operational_value)
    timestamp = raw.get("parent_capture_timestamp_utc")
    if not isinstance(timestamp, str):
        raise ValueError("raw v2 capture timestamp is missing")
    return _build_evidence(
        operational,
        contract_id=CONTRACT_ID_V2,
        parent_capture_timestamp_utc=timestamp,
        parent_to_frozen_equal=True,
        raw_evidence_sha256=canonical_sha256(dict(raw)),
    )


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
    if analyze_path(operational["PATH"]) != evidence["path"]:
        raise ValueError("child PATH evidence differs")


def load_frozen_child_environment(
    *, raw_environment_path: Path, tracked_evidence_path: Path
) -> dict[str, str]:
    """Load a v2 child without consulting the ambient parent environment."""

    raw = json.loads(raw_environment_path.read_text(encoding="utf-8"))
    tracked = json.loads(tracked_evidence_path.read_text(encoding="utf-8"))
    child, rebuilt = reconstruct_v2_environment(raw)
    if rebuilt != tracked:
        raise ValueError("tracked v2 environment differs from ignored raw reconstruction")
    validate_child_environment(child, tracked)
    return child


def tracked_evidence(parent: Mapping[str, str]) -> dict[str, Any]:
    _, evidence = build_child_environment(parent)
    return cast(dict[str, Any], json.loads(json.dumps(evidence)))


def run_import_preflight(repository_root: Path) -> subprocess.CompletedProcess[str]:
    root = repository_root.resolve()
    tracked_path = (
        root / "results" / "phase2_vetted_corpus" / "windows_operational_environment.json"
    )
    raw_path = (
        root
        / "results"
        / "raw"
        / "phase2_vetted_corpus"
        / "milestone13c_r1"
        / "windows_operational_environment_v2_raw.json"
    )
    child = load_frozen_child_environment(
        raw_environment_path=raw_path,
        tracked_evidence_path=tracked_path,
    )
    argv = [
        str(root / ".venv-training" / "Scripts" / "python.exe"),
        "-m",
        "foundry.phase2.windows_import_preflight",
        "--environment-evidence",
        str(tracked_path),
        "--output",
        str(
            root
            / "results"
            / "raw"
            / "phase2_vetted_corpus"
            / "milestone13c_r1"
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-v2")
    freeze.add_argument("--raw-output", type=Path, required=True)
    freeze.add_argument("--tracked-output", type=Path, required=True)
    preflight = subparsers.add_parser("import-preflight")
    preflight.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-v2":
        timestamp = datetime.now(UTC).isoformat()
        raw, _, tracked = freeze_v2_environment(
            os.environ,
            parent_capture_timestamp_utc=timestamp,
        )
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.tracked_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.tracked_output.write_text(
            json.dumps(tracked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(tracked, sort_keys=True))
        return
    result = run_import_preflight(args.repository_root)
    print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError(result.stderr)


if __name__ == "__main__":
    main()
