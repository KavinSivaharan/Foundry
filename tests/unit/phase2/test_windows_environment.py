from __future__ import annotations

from copy import deepcopy

import pytest

from foundry.phase2.launch_contract import (
    ALLOWLISTED_ENVIRONMENT,
    PACKAGE_INVENTORY_SHA256,
)
from foundry.phase2.windows_environment import (
    OPERATIONAL_ALLOWLIST,
    build_child_environment,
    validate_child_environment,
)


def _parent() -> dict[str, str]:
    return {
        "SystemRoot": r"C:\Windows",
        "windir": r"C:\Windows",
        "Path": r"C:\Windows\System32",
        "PATHEXT": ".COM;.EXE",
        "ComSpec": r"C:\Windows\System32\cmd.exe",
        "TEMP": r"C:\Temp",
        "tmp": r"C:\Temp",
        "APPDATA": r"C:\Users\fixture\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\fixture\AppData\Local",
        "USERPROFILE": r"C:\Users\fixture",
        "UNRELATED": "excluded",
        "OPENAI_API_KEY": "excluded",
        "PYTHONHASHSEED": "wrong",
    }


def test_exact_allowlist_is_frozen() -> None:
    assert len(OPERATIONAL_ALLOWLIST) == 31
    assert OPERATIONAL_ALLOWLIST[0] == "ALLUSERSPROFILE"
    assert OPERATIONAL_ALLOWLIST[-1] == "WINDIR"


def test_unknown_and_secret_variables_are_excluded() -> None:
    child, evidence = build_child_environment(_parent())
    assert "UNRELATED" not in child
    assert "OPENAI_API_KEY" not in child
    assert evidence["unauthorized_variable_count"] == 0


def test_deterministic_values_override_parent() -> None:
    child, _ = build_child_environment(_parent())
    assert {name: child[name] for name in ALLOWLISTED_ENVIRONMENT} == (ALLOWLISTED_ENVIRONMENT)


@pytest.mark.parametrize(
    "name",
    (
        "SYSTEMROOT",
        "WINDIR",
        "PATH",
        "PATHEXT",
        "COMSPEC",
        "TEMP",
        "TMP",
        "APPDATA",
        "LOCALAPPDATA",
        "USERPROFILE",
    ),
)
def test_operational_values_survive_unchanged(name: str) -> None:
    parent = _parent()
    child, _ = build_child_environment(parent)
    source = {key.upper(): value for key, value in parent.items()}
    assert child[name] == source[name]


def test_case_insensitive_names_and_duplicates() -> None:
    child, _ = build_child_environment(_parent())
    assert child["SYSTEMROOT"] == r"C:\Windows"
    duplicate = _parent()
    duplicate["SYSTEMROOT"] = r"C:\Other"
    with pytest.raises(ValueError, match="duplicate"):
        build_child_environment(duplicate)


def test_missing_required_variable_fails() -> None:
    parent = _parent()
    parent.pop("TEMP")
    with pytest.raises(ValueError, match="missing"):
        build_child_environment(parent)


def test_changed_value_changes_hash_and_validation_fails() -> None:
    child, evidence = build_child_environment(_parent())
    changed = dict(child)
    changed["PATH"] += ";changed"
    with pytest.raises(ValueError, match="differs"):
        validate_child_environment(changed, evidence)


def test_added_unauthorized_variable_fails() -> None:
    child, evidence = build_child_environment(_parent())
    child["EXTRA"] = "bad"
    with pytest.raises(ValueError, match="unauthorized"):
        validate_child_environment(child, evidence)


def test_parent_is_not_mutated() -> None:
    parent = _parent()
    before = deepcopy(parent)
    build_child_environment(parent)
    assert parent == before


def test_shell_argv_inventory_and_science_remain_frozen() -> None:
    assert PACKAGE_INVENTORY_SHA256 == (
        "2d4dbf699b73b53206d96687f1381ec22dac8a2d1575b0a43791627b9b43b2c8"
    )
    assert "learning_rate" not in OPERATIONAL_ALLOWLIST
