"""Shell-free argv transport for the Phase 2 native QLoRA probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from foundry.phase2.launch_contract import (
    ALLOWLISTED_ENVIRONMENT,
    AUTHORIZED_INTERPRETER_SHA256,
    PACKAGE_INVENTORY_SHA256,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

CONTRACT_ID = "foundry-vetted-qlora-argv-transport-v1"
CHILD_MODULE = "foundry.phase2.qlora_environment"
EXPECTED_ARG_COUNT = 11
EXPECTED_REPLAY_SHA256 = "a9f25258d23f05a785dfea9f8ae0e05a246b52c9798a0d10e683fdc4e01a87f6"
EXPECTED_EXCEPTION_SHA256 = "5403032fbc5f120c77f7a1dc7334f47af076f2c77ab028b05d3bbfef3b4825ee"


def has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def canonical_argv_sha256(argv: list[str]) -> str:
    if any(has_control_characters(item) for item in argv):
        raise ValueError("argv contains an ASCII control character")
    return canonical_sha256(argv)


def _resolved_inside(path: Path, root: Path, *, name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} is outside its authorized root") from error
    return resolved


def build_contract(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    source_root = _resolved_inside(root / "src", root, name="source root")
    interpreter = (root / ".venv-training" / "Scripts" / "python.exe").resolve()
    replay = _resolved_inside(
        root / "results" / "raw" / "training" / "base_replay_kl" / "replay_corpus.json",
        root,
        name="replay manifest",
    )
    model = _resolved_inside(
        root
        / "data"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen2.5-1.5B-Instruct"
        / "snapshots"
        / "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        root,
        name="model cache",
    )
    exception = _resolved_inside(
        root / "results" / "phase2_vetted_corpus" / "pyyaml_metadata_exception.json",
        root,
        name="configuration",
    )
    output_root = _resolved_inside(
        root / "results" / "raw" / "phase2_vetted_corpus" / "qlora_environment_probe",
        root / "results" / "raw",
        name="output root",
    )
    output = output_root / "training_environment.json"
    required = (interpreter, replay, model, exception)
    if any(not path.exists() for path in required):
        raise FileNotFoundError("required argv transport input is missing")
    if file_sha256(interpreter) != AUTHORIZED_INTERPRETER_SHA256:
        raise ValueError("authorized interpreter differs")
    if file_sha256(replay) != EXPECTED_REPLAY_SHA256:
        raise ValueError("replay manifest hash differs")
    if file_sha256(exception) != EXPECTED_EXCEPTION_SHA256:
        raise ValueError("PyYAML exception evidence file hash differs")
    argv = [
        str(interpreter),
        "-m",
        CHILD_MODULE,
        "--model-path",
        str(model),
        "--replay-path",
        str(replay),
        "--exception-path",
        str(exception),
        "--output",
        str(output),
    ]
    if len(argv) != EXPECTED_ARG_COUNT:
        raise RuntimeError("frozen argv count differs")
    argv_hash = canonical_argv_sha256(argv)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "argv": argv,
        "argv_canonical_json": json.dumps(argv, separators=(",", ":"), ensure_ascii=True),
        "argv_sha256": argv_hash,
        "shell": False,
        "working_directory": str(source_root),
        "working_directory_sha256": canonical_sha256(str(source_root)),
        "interpreter_sha256": file_sha256(interpreter),
        "child_module": CHILD_MODULE,
        "child_module_source_sha256": file_sha256(
            source_root / "foundry" / "phase2" / "qlora_environment.py"
        ),
        "launch_environment": ALLOWLISTED_ENVIRONMENT,
        "launch_environment_sha256": canonical_sha256(ALLOWLISTED_ENVIRONMENT),
        "package_inventory_sha256": PACKAGE_INVENTORY_SHA256,
        "replay_manifest_path_sha256": canonical_sha256(str(replay)),
        "replay_manifest_file_sha256": file_sha256(replay),
        "model_cache_path_sha256": canonical_sha256(str(model)),
        "exception_path_sha256": canonical_sha256(str(exception)),
        "exception_file_sha256": file_sha256(exception),
        "output_root_path_sha256": canonical_sha256(str(output_root)),
        "resolved_paths": {
            "repository_root": str(root),
            "source_root": str(source_root),
            "replay_manifest": str(replay),
            "model_cache": str(model),
            "configuration": str(exception),
            "output_root": str(output_root),
        },
        "control_character_check": True,
        "result_schema": {
            "argv_sha256": "sha256",
            "child_received_argv_sha256": "sha256",
            "returncode": "integer",
            "stdout_sha256": "sha256",
            "stderr_sha256": "sha256",
        },
    }
    evidence["contract_sha256"] = canonical_sha256(evidence)
    return evidence


def validate_contract(contract: dict[str, Any], repository_root: Path) -> None:
    rebuilt = build_contract(repository_root)
    if contract != rebuilt:
        raise ValueError("argv transport contract differs from reconstruction")
    argv = contract["argv"]
    if argv[1:3] != ["-m", CHILD_MODULE] or "-c" in argv:
        raise ValueError("inline Python model execution is prohibited")
    if contract["shell"] is not False:
        raise ValueError("shell=False is mandatory")


def validate_child_paths(
    *,
    model_path: Path,
    replay_path: Path,
    exception_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Validate child-received paths and hashes before model-stack imports."""

    source_root = Path(__file__).resolve().parents[2]
    root = source_root.parent
    expected = build_contract(root)
    received_argv = [
        str(Path(sys.executable).resolve()),
        "-m",
        CHILD_MODULE,
        "--model-path",
        str(model_path),
        "--replay-path",
        str(replay_path),
        "--exception-path",
        str(exception_path),
        "--output",
        str(output_path),
    ]
    if any(has_control_characters(item) for item in received_argv):
        raise ValueError("child received an ASCII control character")
    if received_argv != expected["argv"]:
        raise ValueError("child-received argv differs from frozen argv")
    if not replay_path.is_file() or file_sha256(replay_path) != EXPECTED_REPLAY_SHA256:
        raise ValueError("child replay manifest is missing or differs")
    if not exception_path.is_file() or file_sha256(exception_path) != EXPECTED_EXCEPTION_SHA256:
        raise ValueError("child exception evidence is missing or differs")
    output_root = Path(expected["resolved_paths"]["output_root"])
    _resolved_inside(output_path, output_root, name="child output")
    model_manifest = [
        {
            "name": item.relative_to(model_path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in sorted(model_path.rglob("*"))
        if item.is_file()
    ]
    if canonical_sha256(model_manifest) != (
        "02bff45c336c3650abe518a94accf4c321f0116678a99c2f56a131cf2eade34d"
    ):
        raise ValueError("child model-cache manifest differs")
    return {
        "argv_sha256": expected["argv_sha256"],
        "child_received_argv_sha256": canonical_argv_sha256(received_argv),
        "path_roundtrip_equal": True,
        "control_character_check": True,
        "replay_manifest_file_sha256": file_sha256(replay_path),
        "model_cache_manifest_sha256": canonical_sha256(model_manifest),
        "output_contained": True,
    }


def run_probe(contract: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    validate_contract(contract, repository_root)
    output = Path(contract["argv"][-1])
    output.parent.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        contract["argv"],
        shell=False,
        env=dict(ALLOWLISTED_ENVIRONMENT),
        cwd=contract["working_directory"],
        check=False,
        capture_output=True,
        text=True,
    )
    evidence = {
        "argv_sha256": contract["argv_sha256"],
        "child_received_argv_sha256": canonical_argv_sha256(contract["argv"]),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode != 0:
        raise RuntimeError(json.dumps(evidence, sort_keys=True))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--repository-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument("--contract", type=Path, required=True)
    echo = subparsers.add_parser("echo")
    echo.add_argument("values", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "echo":
        print(json.dumps(args.values, ensure_ascii=True))
        return
    if args.command == "freeze":
        value = build_contract(args.repository_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(value, sort_keys=True))
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    print(json.dumps(run_probe(contract, args.repository_root), sort_keys=True))


if __name__ == "__main__":
    main()
