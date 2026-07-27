from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import foundry.phase2.windows_environment as windows
from foundry.phase2 import argv_transport
from foundry.phase2.launch_contract import (
    ALLOWLISTED_ENVIRONMENT,
    AUTHORIZED_INTERPRETER_SHA256,
    PACKAGE_INVENTORY_SHA256,
)
from foundry.training.qlora import file_sha256

ROOT = Path(__file__).resolve().parents[3]
DATASET_SHA256 = "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"
HOLDOUT_SHA256 = "826ccfda6714af45f2f8e0ae3926d4607a149446ae5b2f75137704e906a2d92e"
ARCHITECTURE_SHA256 = "74907ea92b2217b6f9ca39044feab6c6452600e7774a2b442e0ec9e29b6899a5"
ARGV_SHA256 = "c9d71f34956cc6a0f3a40b394ea7c8ee0e6717d0cf5aeac84582e24606a4f900"


def _parent() -> dict[str, str]:
    result = {name: f"value-{name}" for name in windows.OPERATIONAL_ALLOWLIST}
    result.update(
        {
            "COMSPEC": r"C:\Windows\System32\cmd.exe",
            "PATH": r"C:\Windows\System32;C:\Tools",
            "PATHEXT": ".COM;.EXE",
            "SYSTEMROOT": r"C:\Windows",
            "TEMP": r"C:\Temp",
            "TMP": r"C:\Temp",
            "WINDIR": r"C:\Windows",
            "UNRELATED": "excluded",
            "OPENAI_API_KEY": "excluded",
            "PYTHONHASHSEED": "wrong",
        }
    )
    return result


class _CountingMapping(Mapping[str, str]):
    def __init__(self, values: dict[str, str], *, reject_path_read: bool = False) -> None:
        self.values = values
        self.path_reads = 0
        self.reject_path_read = reject_path_read

    def __getitem__(self, key: str) -> str:
        if key.upper() == "PATH":
            self.path_reads += 1
            if self.reject_path_read:
                raise AssertionError("ambient parent PATH was consulted")
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def test_exact_allowlist_is_frozen() -> None:
    assert len(windows.OPERATIONAL_ALLOWLIST) == 31
    assert windows.OPERATIONAL_ALLOWLIST[0] == "ALLUSERSPROFILE"
    assert windows.OPERATIONAL_ALLOWLIST[-1] == "WINDIR"


def test_v1_path_reconstructs_when_raw_evidence_exists() -> None:
    frozen_path = r"C:\Frozen\One;C:\Frozen\Two"
    child, evidence = windows.build_case1_environment(
        _parent(),
        frozen_path=frozen_path,
        expected_path_sha256=windows.sha256_text(frozen_path),
    )
    assert child["PATH"] == frozen_path
    assert evidence["contract_id"] == windows.CONTRACT_ID_V1


def test_one_character_path_change_changes_hash() -> None:
    assert windows.sha256_text(r"C:\One") != windows.sha256_text(r"C:\Onf")


def test_component_reordering_changes_path_hash() -> None:
    assert windows.sha256_text(r"C:\One;C:\Two") != windows.sha256_text(r"C:\Two;C:\One")


def test_duplicate_components_are_detected() -> None:
    evidence = windows.analyze_path(r"C:\One;C:\ONE\\")
    assert len(evidence["duplicate_component_sha256"]) == 1
    with pytest.raises(ValueError, match="duplicate"):
        windows.validate_path(r"C:\One;C:\ONE\\")


def test_empty_components_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        windows.validate_path(r"C:\One;;C:\Two")


def test_relative_components_are_rejected() -> None:
    with pytest.raises(ValueError, match="relative"):
        windows.validate_path(r"C:\One;relative")


def test_control_characters_are_rejected() -> None:
    with pytest.raises(ValueError, match="control"):
        windows.validate_path("C:\\One;C:\\Bad\nPath")


def test_secret_looking_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="secret"):
        windows.validate_path(r"C:\One;C:\token=not-allowed")


def test_parent_is_not_mutated() -> None:
    parent = _parent()
    before = deepcopy(parent)
    windows.build_child_environment(parent)
    assert parent == before


def test_case1_does_not_consult_current_parent_path() -> None:
    values = _parent()
    frozen_path = r"C:\Frozen\One;C:\Frozen\Two"
    parent = _CountingMapping(values, reject_path_read=True)
    child, _ = windows.build_case1_environment(
        parent,
        frozen_path=frozen_path,
        expected_path_sha256=windows.sha256_text(frozen_path),
    )
    assert parent.path_reads == 0
    assert child["PATH"] == frozen_path


def test_case2_captures_parent_path_exactly_once() -> None:
    parent = _CountingMapping(_parent())
    raw, child, _ = windows.freeze_v2_environment(
        parent,
        parent_capture_timestamp_utc="2026-07-27T00:00:00+00:00",
    )
    assert parent.path_reads == 1
    assert raw["operational_environment"]["PATH"] == child["PATH"]


def test_case2_children_use_captured_not_later_parent_path() -> None:
    parent = _parent()
    raw, child, tracked = windows.freeze_v2_environment(
        parent,
        parent_capture_timestamp_utc="2026-07-27T00:00:00+00:00",
    )
    parent["PATH"] = r"C:\Changed"
    rebuilt_child, rebuilt_tracked = windows.reconstruct_v2_environment(raw)
    assert rebuilt_child == child
    assert rebuilt_tracked == tracked
    assert rebuilt_child["PATH"] != parent["PATH"]


def test_other_30_operational_values_remain_unchanged() -> None:
    parent = _parent()
    operational = windows.capture_operational_environment(parent)
    for name in windows.OPERATIONAL_ALLOWLIST:
        if name != "PATH":
            assert operational[name] == parent[name]


def test_deterministic_values_override_parent() -> None:
    parent = _parent()
    child, _ = windows.build_child_environment(parent)
    assert {name: child[name] for name in ALLOWLISTED_ENVIRONMENT} == (ALLOWLISTED_ENVIRONMENT)


def test_unauthorized_variables_remain_excluded() -> None:
    child, evidence = windows.build_child_environment(_parent())
    assert "UNRELATED" not in child
    assert "OPENAI_API_KEY" not in child
    assert evidence["unauthorized_variable_count"] == 0


def test_import_preflight_requires_shell_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, _, tracked = windows.freeze_v2_environment(
        _parent(),
        parent_capture_timestamp_utc="2026-07-27T00:00:00+00:00",
    )
    raw_path = (
        tmp_path
        / "results"
        / "raw"
        / "phase2_vetted_corpus"
        / "milestone13c_r1"
        / "windows_operational_environment_v2_raw.json"
    )
    tracked_path = (
        tmp_path / "results" / "phase2_vetted_corpus" / "windows_operational_environment.json"
    )
    raw_path.parent.mkdir(parents=True)
    tracked_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    tracked_path.write_text(json.dumps(tracked), encoding="utf-8")
    (tmp_path / "src").mkdir()
    called: dict[str, Any] = {}

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        called["argv"] = argv
        called.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(windows.subprocess, "run", _run)
    windows.run_import_preflight(tmp_path)
    assert called["shell"] is False


def test_argv_hash_remains_unchanged() -> None:
    assert argv_transport.build_contract(ROOT)["argv_sha256"] == ARGV_SHA256


def test_interpreter_hash_remains_unchanged() -> None:
    interpreter = ROOT / ".venv-training" / "Scripts" / "python.exe"
    assert file_sha256(interpreter) == AUTHORIZED_INTERPRETER_SHA256


def test_package_inventory_hash_remains_unchanged() -> None:
    assert PACKAGE_INVENTORY_SHA256 == (
        "2d4dbf699b73b53206d96687f1381ec22dac8a2d1575b0a43791627b9b43b2c8"
    )


def test_dataset_holdout_and_scientific_hashes_remain_unchanged() -> None:
    dataset = json.loads(
        (ROOT / "results" / "phase2_vetted_corpus" / "dataset_summary.json").read_text(
            encoding="utf-8"
        )
    )
    architecture = json.loads(
        (
            ROOT / "results" / "phase2_vetted_corpus" / "milestone13b_architecture_contract.json"
        ).read_text(encoding="utf-8")
    )
    stop = json.loads(
        (
            ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_kl_environment_stop.json"
        ).read_text(encoding="utf-8")
    )
    assert dataset["dataset_sha256"] == DATASET_SHA256
    assert architecture["architecture_decision_sha256"] == ARCHITECTURE_SHA256
    assert stop["candidate_holdout"]["suite_sha256"] == HOLDOUT_SHA256


def test_missing_required_variable_fails() -> None:
    parent = _parent()
    parent.pop("TEMP")
    with pytest.raises(ValueError, match="missing"):
        windows.build_child_environment(parent)


def test_changed_child_path_fails_validation() -> None:
    child, evidence = windows.build_child_environment(_parent())
    changed = dict(child)
    changed["PATH"] = r"C:\Changed"
    with pytest.raises(ValueError, match="differs"):
        windows.validate_child_environment(changed, evidence)


def test_added_unauthorized_child_variable_fails() -> None:
    child, evidence = windows.build_child_environment(_parent())
    child["EXTRA"] = "bad"
    with pytest.raises(ValueError, match="unauthorized"):
        windows.validate_child_environment(child, evidence)
