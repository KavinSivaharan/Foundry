"""Immutable pre-launch environment contract for verifier-GRPO processes.

The contract in this module is deliberately limited to process orchestration.  It
does not import Torch or Transformers at module import time, so the launch values
can be captured and checked before either package (and therefore CUDA) is loaded.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import ntpath
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from foundry.training.config import canonical_sha256

DETERMINISTIC_PROCESS_ENVIRONMENT_ID = "foundry-verifier-grpo-deterministic-process-environment-v1"
DETERMINISTIC_PROCESS_ENVIRONMENT_SCHEMA_VERSION = 1

FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT = MappingProxyType(
    {
        "ASCEND_LAUNCH_BLOCKING": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        "CUDA_LAUNCH_BLOCKING": "1",
        "FLASH_ATTENTION_DETERMINISTIC": "1",
        "HCCL_DETERMINISTIC": "1",
    }
)
FROZEN_CORE_PROCESS_ENVIRONMENT = MappingProxyType(
    {
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "20260720",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        **FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT,
    }
)

# These are the only parent fields that may be propagated to a child process.
# Authentication, cloud, telemetry, and arbitrary user variables are never copied.
INTENTIONALLY_INHERITED_ENVIRONMENT_KEYS = (
    "COMSPEC",
    "DRIVERDATA",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL",
    "PROCESSOR_REVISION",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
)

FROZEN_TRANSFORMERS_VERSION = "4.51.3"
FROZEN_TRANSFORMERS_DETERMINISM_MODULE = "transformers.trainer_utils"
FROZEN_TRANSFORMERS_DETERMINISM_QUALNAME = "enable_full_determinism"
FROZEN_TRANSFORMERS_TRAINER_UTILS_FILE_SHA256 = (
    "33561736fc04ae94729a513845b9bb900637a5eb6a768aabe018494cf631a95e"
)
FROZEN_TRANSFORMERS_DETERMINISM_FUNCTION_SHA256 = (
    "1893964197a05bfd07d1477815b58e42e883b9e64985f0e795b4562fc9f84834"
)

_PROCESS_START_CORE_ENVIRONMENT = tuple(
    sorted((key, os.environ.get(key)) for key in FROZEN_CORE_PROCESS_ENVIRONMENT)
)
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "passwd",
    "password",
    "secret",
    "session",
    "token",
)


class RuntimePathsLike(Protocol):
    """The path fields needed without importing the concrete path-contract type."""

    @property
    def source_root(self) -> Path: ...

    @property
    def python_executable(self) -> Path: ...

    @property
    def artifact_root(self) -> Path: ...

    @property
    def model_cache_root(self) -> Path: ...


def _path_key(path: Path) -> str:
    return ntpath.normcase(str(path.resolve(strict=False))).rstrip("\\/")


def _same_path(left: Path, right: Path) -> bool:
    return _path_key(left) == _path_key(right)


def deterministic_environment_values(runtime_paths: RuntimePathsLike) -> dict[str, str]:
    """Return every value that must be fixed before the interpreter starts."""

    values = dict(FROZEN_CORE_PROCESS_ENVIRONMENT)
    values.update(
        {
            "HF_HOME": str(runtime_paths.model_cache_root),
            "PYTHONPATH": str(runtime_paths.source_root / "src"),
            "TEMP": str(runtime_paths.artifact_root / "temp"),
            "TMP": str(runtime_paths.artifact_root / "temp"),
            "TMPDIR": str(runtime_paths.artifact_root / "temp"),
        }
    )
    return values


def deterministic_environment_sha256(runtime_paths: RuntimePathsLike) -> str:
    """Hash only the explicit, secret-free deterministic environment values."""

    return canonical_sha256(deterministic_environment_values(runtime_paths))


def _allowlisted_parent_environment(base_environment: Mapping[str, str]) -> dict[str, str]:
    by_upper = {key.upper(): value for key, value in base_environment.items()}
    return {
        key: by_upper[key] for key in INTENTIONALLY_INHERITED_ENVIRONMENT_KEYS if key in by_upper
    }


def build_allowlisted_launch_environment(
    runtime_paths: RuntimePathsLike,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct a child environment without copying arbitrary parent fields."""

    base = os.environ if base_environment is None else base_environment
    environment = _allowlisted_parent_environment(base)
    environment.update(deterministic_environment_values(runtime_paths))
    return environment


def current_process_command() -> tuple[str, ...]:
    """Return the exact interpreter command, including ``-m`` when used."""

    original = getattr(sys, "orig_argv", None)
    if isinstance(original, list) and original and all(isinstance(item, str) for item in original):
        # Windows venv redirects preserve the base interpreter in orig_argv[0].
        # sys.executable is the actual venv executable that was invoked and frozen.
        return (str(sys.executable), *tuple(original[1:]))
    return (str(sys.executable), *tuple(sys.argv))


@dataclass(frozen=True)
class GrpoDeterministicProcessEnvironment:
    """Typed immutable identity for one exact verifier-GRPO process launch."""

    python_executable: Path
    pythonpath: Path
    source_root: Path
    artifact_root: Path
    model_cache_root: Path
    process_command: tuple[str, ...]
    environment_items: tuple[tuple[str, str], ...]
    inherited_environment_keys: tuple[str, ...]

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.environment_items)

    @property
    def environment_sha256(self) -> str:
        return canonical_sha256(self.environment)

    @property
    def process_command_sha256(self) -> str:
        return canonical_sha256(list(self.process_command))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": DETERMINISTIC_PROCESS_ENVIRONMENT_SCHEMA_VERSION,
            "contract_id": DETERMINISTIC_PROCESS_ENVIRONMENT_ID,
            "python_executable": str(self.python_executable),
            "pythonpath": str(self.pythonpath),
            "source_root": str(self.source_root),
            "artifact_root": str(self.artifact_root),
            "model_cache_root": str(self.model_cache_root),
            "process_command": list(self.process_command),
            "process_command_sha256": self.process_command_sha256,
            "environment": self.environment,
            "environment_sha256": self.environment_sha256,
            "inherited_environment_keys": list(self.inherited_environment_keys),
            "secrets_included": False,
        }

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def evidence(self) -> dict[str, object]:
        return {**self.payload(), "contract_sha256": self.contract_sha256}


def freeze_deterministic_process_environment(
    runtime_paths: RuntimePathsLike,
    process_command: Sequence[str] | None = None,
) -> GrpoDeterministicProcessEnvironment:
    """Freeze the paths, command, and deterministic fields for one process."""

    command = tuple(current_process_command() if process_command is None else process_command)
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("GRPO process command must contain non-empty strings")
    if not _same_path(Path(command[0]), runtime_paths.python_executable):
        raise ValueError("GRPO process command uses a different Python executable")
    values = deterministic_environment_values(runtime_paths)
    return GrpoDeterministicProcessEnvironment(
        python_executable=runtime_paths.python_executable.resolve(strict=False),
        pythonpath=(runtime_paths.source_root / "src").resolve(strict=False),
        source_root=runtime_paths.source_root.resolve(strict=False),
        artifact_root=runtime_paths.artifact_root.resolve(strict=False),
        model_cache_root=runtime_paths.model_cache_root.resolve(strict=False),
        process_command=command,
        environment_items=tuple(sorted(values.items())),
        inherited_environment_keys=INTENTIONALLY_INHERITED_ENVIRONMENT_KEYS,
    )


def _process_start_environment() -> dict[str, str | None]:
    return dict(_PROCESS_START_CORE_ENVIRONMENT)


def validate_deterministic_process_environment(
    runtime_paths: RuntimePathsLike,
    stage: str,
    *,
    torch_module: Any | None = None,
    require_strict: bool = False,
    check_process_start: bool = True,
    process_command: Sequence[str] | None = None,
) -> dict[str, object]:
    """Fail closed if any deterministic launch field or runtime state differs."""

    if not stage:
        raise ValueError("deterministic-environment stage must be named")
    contract = freeze_deterministic_process_environment(runtime_paths, process_command)
    expected = contract.environment
    actual = {key: os.environ.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"deterministic process environment differs at {stage}: {actual}")
    if check_process_start:
        expected_start = {
            key: value for key, value in expected.items() if key in FROZEN_CORE_PROCESS_ENVIRONMENT
        }
        if _process_start_environment() != expected_start:
            raise RuntimeError(
                f"deterministic environment was not correct before Python launch at {stage}"
            )
    if not _same_path(Path(sys.executable), runtime_paths.python_executable):
        raise RuntimeError(f"Python executable differs at {stage}")
    if os.environ.get("PYTHONHASHSEED") != "20260720" or not bool(sys.flags.hash_randomization):
        raise RuntimeError(f"Python hash seed or randomization differs at {stage}")
    if not bool(sys.dont_write_bytecode):
        raise RuntimeError(f"PYTHONDONTWRITEBYTECODE was ineffective at {stage}")

    torch_evidence: dict[str, object] = {
        "torch_imported": torch_module is not None,
        "deterministic_algorithms": None,
        "deterministic_warn_only": None,
        "cuda_initialized": None,
    }
    if torch_module is not None:
        deterministic = bool(torch_module.are_deterministic_algorithms_enabled())
        warn_only = bool(torch_module.is_deterministic_algorithms_warn_only_enabled())
        cuda_initialized = bool(torch_module.cuda.is_initialized())
        if require_strict and (not deterministic or warn_only):
            raise RuntimeError(f"strict deterministic mode differs at {stage}")
        torch_evidence.update(
            {
                "deterministic_algorithms": deterministic,
                "deterministic_warn_only": warn_only,
                "cuda_initialized": cuda_initialized,
            }
        )
    return {
        "stage": stage,
        "environment": actual,
        "environment_sha256": contract.environment_sha256,
        "process_command_sha256": contract.process_command_sha256,
        "contract_sha256": contract.contract_sha256,
        "python_hash_randomization": True,
        **torch_evidence,
    }


class _EnvironmentWrites(ast.NodeVisitor):
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "os"
                and target.value.attr == "environ"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                self.values[target.slice.value] = node.value.value
        self.generic_visit(node)


def transformers_determinism_source_evidence(
    transformers: Any,
    *,
    expected_file_sha256: str = FROZEN_TRANSFORMERS_TRAINER_UTILS_FILE_SHA256,
    expected_function_sha256: str = FROZEN_TRANSFORMERS_DETERMINISM_FUNCTION_SHA256,
) -> dict[str, object]:
    """Verify the installed helper file, callable, and every literal env write."""

    if str(getattr(transformers, "__version__", "")) != FROZEN_TRANSFORMERS_VERSION:
        raise RuntimeError("Transformers version differs from the frozen environment")
    trainer_utils = getattr(transformers, "trainer_utils", None)
    function = getattr(trainer_utils, "enable_full_determinism", None)
    if not callable(function):
        raise RuntimeError("Transformers full-determinism function is unavailable")
    if (
        str(getattr(function, "__module__", "")) != FROZEN_TRANSFORMERS_DETERMINISM_MODULE
        or str(getattr(function, "__qualname__", "")) != FROZEN_TRANSFORMERS_DETERMINISM_QUALNAME
    ):
        raise RuntimeError("Transformers full-determinism callable identity differs")
    source_file_text = inspect.getsourcefile(function)
    if not source_file_text:
        raise RuntimeError("Transformers full-determinism source file is unavailable")
    source_file = Path(source_file_text).resolve(strict=True)
    file_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()
    source = inspect.getsource(function)
    function_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    writes = _EnvironmentWrites()
    writes.visit(ast.parse(source))
    expected_writes = dict(FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT)
    if file_sha256 != expected_file_sha256:
        raise RuntimeError("Transformers trainer_utils.py source hash differs")
    if function_sha256 != expected_function_sha256:
        raise RuntimeError("Transformers full-determinism function source hash differs")
    if writes.values != expected_writes:
        raise RuntimeError("Transformers full-determinism environment writes differ")
    return {
        "transformers_version": FROZEN_TRANSFORMERS_VERSION,
        "source_file": str(source_file),
        "source_file_sha256": file_sha256,
        "callable_module": FROZEN_TRANSFORMERS_DETERMINISM_MODULE,
        "callable_qualname": FROZEN_TRANSFORMERS_DETERMINISM_QUALNAME,
        "function_source_sha256": function_sha256,
        "environment_writes": expected_writes,
    }


def assert_idempotent_deterministic_initialization(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    """Require Transformers initialization to leave all five fields unchanged."""

    before_environment = before.get("environment")
    after_environment = after.get("environment")
    if not isinstance(before_environment, Mapping) or not isinstance(after_environment, Mapping):
        raise TypeError("deterministic initialization evidence lacks environment mappings")
    keys = tuple(FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT)
    before_values = {key: before_environment.get(key) for key in keys}
    after_values = {key: after_environment.get(key) for key in keys}
    expected = dict(FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT)
    if before_values != expected or after_values != expected or before_values != after_values:
        raise RuntimeError("Transformers deterministic initialization changed the environment")
    value: dict[str, object] = {
        "before": before_values,
        "after": after_values,
        "effective_environment_changed": False,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def allowlisted_environment_evidence(
    runtime_paths: RuntimePathsLike,
) -> tuple[str, int, int]:
    """Hash only explicitly allowed fields and count secret-named exclusions."""

    allowed = build_allowlisted_launch_environment(runtime_paths, os.environ)
    allowed_names = {key.upper() for key in allowed}
    excluded_secret_names = sum(
        key.upper() not in allowed_names
        and any(marker in key.lower() for marker in _SECRET_MARKERS)
        for key in os.environ
    )
    return canonical_sha256(allowed), len(allowed), excluded_secret_names


def make_environment_guarded_trainer(
    base_trainer_class: type[Any], validator: Callable[[str], None]
) -> type[Any]:
    """Validate the environment around generation and backward boundaries."""

    class EnvironmentGuardedTrainer(base_trainer_class):  # type: ignore[misc]
        def _generate_and_score_completions(self, inputs: Any) -> Any:
            validator("before_generation")
            try:
                return super()._generate_and_score_completions(inputs)
            finally:
                validator("after_generation")

        def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
            validator("before_backward")
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            finally:
                validator("after_backward")

    EnvironmentGuardedTrainer.__name__ = "EnvironmentGuardedTrainer"
    return EnvironmentGuardedTrainer


def make_environment_validation_callback(
    base_callback_class: type[Any], validator: Callable[[str], None]
) -> object:
    """Validate immediately before and after each stock optimizer step."""

    class EnvironmentValidationCallback(base_callback_class):  # type: ignore[misc]
        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state, kwargs
            validator("before_optimizer")
            return control

        def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state, kwargs
            validator("after_optimizer")
            return control

    EnvironmentValidationCallback.__name__ = "EnvironmentValidationCallback"
    return EnvironmentValidationCallback()
