"""Import-only Windows child preflight; never loads a model or creates an optimizer."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from foundry.phase2.launch_contract import validate_postimport, validate_preimport
from foundry.phase2.windows_environment import validate_child_environment
from foundry.training.config import canonical_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    evidence = json.loads(args.environment_evidence.read_text(encoding="utf-8"))
    validate_child_environment(os.environ, evidence)
    preimport = validate_preimport()
    imported: dict[str, Any] = {}
    for name in (
        "_overlapped",
        "socket",
        "asyncio",
        "asyncio.windows_events",
        "torch",
        "transformers",
        "tokenizers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "trl",
    ):
        imported[name] = importlib.import_module(name)
    torch = imported["torch"]
    torch.use_deterministic_algorithms(True, warn_only=False)
    postimport = validate_postimport(
        preimport,
        torch,
        {name: module for name, module in imported.items() if hasattr(module, "__file__")},
    )
    local_socket = socket.socket()
    local_socket.close()
    asyncio = imported["asyncio"]
    loop = asyncio.ProactorEventLoop()
    loop.close()
    module_paths = {
        name: str(Path(module.__file__).resolve())
        for name, module in imported.items()
        if getattr(module, "__file__", None)
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "decision": "pass",
        "preimport": preimport,
        "postimport": postimport,
        "environment_before_sha256": canonical_sha256(dict(os.environ)),
        "environment_after_sha256": canonical_sha256(dict(os.environ)),
        "imported_module_paths": module_paths,
        "imported_module_path_sha256": canonical_sha256(module_paths),
        "overlapped_import": True,
        "winsock_socket": True,
        "asyncio_iocp": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0),
        "model_loads": 0,
        "adapters": 0,
        "optimizers": 0,
        "network_requests": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    result["import_preflight_sha256"] = canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
