from __future__ import annotations

from pathlib import Path

import pytest

from foundry.phase2.launch_contract import (
    ALLOWLISTED_ENVIRONMENT,
    PACKAGE_INVENTORY_SHA256,
    command_sha256,
    validate_postimport,
    validate_preimport,
)


def _python() -> Path:
    return Path(".venv-training/Scripts/python.exe").resolve()


def _validate(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "executable": _python(),
        "modules": set(),
        "environment": dict(ALLOWLISTED_ENVIRONMENT),
    }
    values.update(changes)
    return validate_preimport(**values)  # type: ignore[arg-type]


def test_missing_cublas_fails_before_import() -> None:
    environment = dict(ALLOWLISTED_ENVIRONMENT)
    environment.pop("CUBLAS_WORKSPACE_CONFIG")
    with pytest.raises(RuntimeError):
        _validate(environment=environment)


def test_wrong_cublas_value_fails() -> None:
    environment = dict(ALLOWLISTED_ENVIRONMENT)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    with pytest.raises(RuntimeError):
        _validate(environment=environment)


def test_setting_after_torch_import_fails() -> None:
    with pytest.raises(RuntimeError, match="imported before"):
        _validate(modules={"torch"})


def test_exact_prelaunch_contract_passes() -> None:
    assert _validate()["validated_before_model_stack_import"] is True


def test_value_remains_unchanged_after_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ALLOWLISTED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "NVIDIA GeForce RTX 3080"

    class _Torch:
        cuda = _Cuda()

        @staticmethod
        def are_deterministic_algorithms_enabled() -> bool:
            return True

    module = type("_Module", (), {"__file__": __file__})()
    evidence = validate_postimport(_validate(), _Torch(), {"fixture": module})
    assert evidence["environment_unchanged"] is True


def test_authorized_interpreter_is_required(tmp_path: Path) -> None:
    other = tmp_path / "python.exe"
    other.write_bytes(b"not authorized")
    with pytest.raises(RuntimeError, match="interpreter"):
        _validate(executable=other)


def test_package_inventory_is_frozen() -> None:
    with pytest.raises(RuntimeError, match="inventory"):
        _validate(inventory_sha256="0" * 64)
    _validate(inventory_sha256=PACKAGE_INVENTORY_SHA256)


def test_command_hash_is_deterministic() -> None:
    values = {
        "interpreter": _python(),
        "source_root": Path("src"),
        "child_argv": ["qlora_environment", "--output", "evidence.json"],
    }
    assert command_sha256(**values) == command_sha256(**values)


def test_tampered_environment_fails() -> None:
    environment = dict(ALLOWLISTED_ENVIRONMENT)
    environment["HF_HUB_OFFLINE"] = "0"
    with pytest.raises(RuntimeError, match="environment"):
        _validate(environment=environment)


def test_contract_has_no_scientific_configuration() -> None:
    assert set(ALLOWLISTED_ENVIRONMENT) == {
        "CUBLAS_WORKSPACE_CONFIG",
        "HF_HUB_OFFLINE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
    }
