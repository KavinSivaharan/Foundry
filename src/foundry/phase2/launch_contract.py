"""Deterministic prelaunch contract for Phase 2 native-Windows QLoRA."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

CONTRACT_ID = "foundry-vetted-qlora-deterministic-launch-v1"
AUTHORIZED_INTERPRETER_SHA256 = "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
PACKAGE_INVENTORY_SHA256 = "2d4dbf699b73b53206d96687f1381ec22dac8a2d1575b0a43791627b9b43b2c8"
ALLOWLISTED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "20260720",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
FORBIDDEN_PREIMPORT_MODULES = {
    "accelerate",
    "bitsandbytes",
    "peft",
    "torch",
    "transformers",
    "trl",
}
COMMAND_TEMPLATE = [
    "{authorized_interpreter}",
    "-c",
    (
        "import runpy,sys;"
        "sys.path.insert(0,{source_root!r});"
        "sys.argv={child_argv!r};"
        "runpy.run_module('foundry.phase2.qlora_environment',run_name='__main__')"
    ),
]


def launch_environment() -> dict[str, str]:
    return {name: os.environ.get(name, "") for name in ALLOWLISTED_ENVIRONMENT}


def validate_preimport(
    *,
    executable: Path | None = None,
    modules: set[str] | None = None,
    environment: dict[str, str] | None = None,
    inventory_sha256: str = PACKAGE_INVENTORY_SHA256,
) -> dict[str, object]:
    """Fail before any model-stack import when the launch contract differs."""

    actual_environment = environment if environment is not None else launch_environment()
    if actual_environment != ALLOWLISTED_ENVIRONMENT:
        raise RuntimeError("allowlisted deterministic launch environment differs")
    loaded = modules if modules is not None else set(sys.modules)
    imported = sorted(FORBIDDEN_PREIMPORT_MODULES.intersection(loaded))
    if imported:
        raise RuntimeError(f"model stack was imported before launch validation: {imported}")
    python = (executable or Path(sys.executable)).resolve()
    if file_sha256(python) != AUTHORIZED_INTERPRETER_SHA256:
        raise RuntimeError("authorized training interpreter hash differs")
    if inventory_sha256 != PACKAGE_INVENTORY_SHA256:
        raise RuntimeError("training package inventory hash differs")
    evidence: dict[str, object] = {
        "contract_id": CONTRACT_ID,
        "environment": actual_environment,
        "environment_sha256": canonical_sha256(actual_environment),
        "authorized_interpreter": str(python),
        "authorized_interpreter_sha256": AUTHORIZED_INTERPRETER_SHA256,
        "package_inventory_sha256": inventory_sha256,
        "forbidden_preimport_modules": imported,
        "validated_before_model_stack_import": True,
        "command_template": COMMAND_TEMPLATE,
        "command_template_sha256": canonical_sha256(COMMAND_TEMPLATE),
    }
    evidence["preimport_evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def validate_postimport(
    preimport: dict[str, Any], torch: Any, package_modules: dict[str, Any]
) -> dict[str, object]:
    """Require unchanged environment, deterministic CUDA, and frozen import paths."""

    if launch_environment() != preimport["environment"]:
        raise RuntimeError("launch environment changed after model-stack imports")
    if not bool(torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError("deterministic-algorithm enforcement is not active")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3080"
    ):
        raise RuntimeError("post-import CUDA device identity differs")
    paths = {
        name: str(Path(module.__file__).resolve())
        for name, module in sorted(package_modules.items())
    }
    evidence: dict[str, object] = {
        "environment": launch_environment(),
        "environment_unchanged": True,
        "deterministic_algorithms": True,
        "cuda_only": True,
        "gpu": torch.cuda.get_device_name(0),
        "imported_package_paths": paths,
    }
    evidence["postimport_evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def command_sha256(*, interpreter: Path, source_root: Path, child_argv: list[str]) -> str:
    """Hash one concrete child command without executing it."""

    command = [
        str(interpreter.resolve()),
        "-c",
        COMMAND_TEMPLATE[2].format(source_root=str(source_root.resolve()), child_argv=child_argv),
    ]
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
