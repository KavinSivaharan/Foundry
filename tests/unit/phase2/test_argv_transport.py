from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import foundry.phase2.argv_transport as transport
from foundry.phase2.launch_contract import (
    ALLOWLISTED_ENVIRONMENT,
    AUTHORIZED_INTERPRETER_SHA256,
    PACKAGE_INVENTORY_SHA256,
)
from foundry.training.qlora import file_sha256

ROOT = Path(__file__).resolve().parents[3]


def _contract() -> dict[str, object]:
    return transport.build_contract(ROOT)


def _echo(values: list[str]) -> list[str]:
    python = ROOT / ".venv-training" / "Scripts" / "python.exe"
    result = subprocess.run(
        [str(python), "-m", "foundry.phase2.argv_transport", "echo", *values],
        shell=False,
        env=os.environ.copy(),
        cwd=ROOT / "src",
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_windows_paths_round_trip_without_escape_processing() -> None:
    values = [
        r"C:\base_replay_kl\replay.json",
        r"C:\temp\fixture.json",
        r"C:\path with spaces\fixture.json",
        r"C:\Unicode-Δ\fixture.json",
        'C:\\quoted-"value"\\fixture.json',
    ]
    assert _echo(values) == values
    assert "\b" not in values[0]
    assert "\t" not in values[1]


def test_no_frozen_argument_contains_control_characters() -> None:
    assert all(not transport.has_control_characters(value) for value in _contract()["argv"])


def test_shell_false_and_authorized_interpreter_are_frozen() -> None:
    contract = _contract()
    assert contract["shell"] is False
    assert file_sha256(Path(contract["argv"][0])) == AUTHORIZED_INTERPRETER_SHA256


def test_environment_and_package_inventory_are_unchanged() -> None:
    contract = _contract()
    assert contract["launch_environment"] == ALLOWLISTED_ENVIRONMENT
    assert contract["package_inventory_sha256"] == PACKAGE_INVENTORY_SHA256


def test_child_receives_exact_frozen_argument_count() -> None:
    assert len(_contract()["argv"]) == transport.EXPECTED_ARG_COUNT


def test_reordering_or_modifying_argv_changes_hash() -> None:
    argv = list(_contract()["argv"])
    reordered = list(argv)
    reordered[3], reordered[5] = reordered[5], reordered[3]
    modified = list(argv)
    modified[-1] += ".changed"
    frozen = transport.canonical_argv_sha256(argv)
    assert transport.canonical_argv_sha256(reordered) != frozen
    assert transport.canonical_argv_sha256(modified) != frozen


def test_missing_manifest_fails_before_model_import() -> None:
    contract = _contract()
    with pytest.raises(ValueError, match="argv differs"):
        transport.validate_child_paths(
            model_path=Path(contract["argv"][4]),
            replay_path=ROOT / "missing-replay.json",
            exception_path=Path(contract["argv"][8]),
            output_path=Path(contract["argv"][10]),
        )


def test_manifest_hash_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport, "EXPECTED_REPLAY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash differs"):
        transport.build_contract(ROOT)


def test_output_outside_ignored_root_fails() -> None:
    with pytest.raises(ValueError, match="outside"):
        transport._resolved_inside(  # noqa: SLF001
            ROOT / "results" / "phase2_vetted_corpus" / "outside.json",
            ROOT / "results" / "raw",
            name="output",
        )


def test_source_outside_repository_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        transport._resolved_inside(tmp_path, ROOT, name="source")  # noqa: SLF001


def test_inline_python_model_execution_is_rejected() -> None:
    contract = _contract()
    tampered = dict(contract)
    tampered["argv"] = [contract["argv"][0], "-c", "model code"]
    with pytest.raises(ValueError):
        transport.validate_contract(tampered, ROOT)


def test_contract_has_no_scientific_configuration() -> None:
    source = Path(transport.__file__).read_text(encoding="utf-8")
    for forbidden in ("learning_rate", "lora_rank", "warmup_ratio", "optimizer_steps"):
        assert forbidden not in source


def test_control_character_argument_is_rejected() -> None:
    with pytest.raises(ValueError, match="control"):
        transport.canonical_argv_sha256(["safe", "bad\bpath"])


def test_contract_reconstruction_is_exact() -> None:
    contract = _contract()
    transport.validate_contract(contract, ROOT)
    assert contract == transport.build_contract(ROOT)
