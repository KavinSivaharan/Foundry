from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foundry.training import grpo_environment as environment
from foundry.training.config import canonical_sha256


def _runtime_paths(tmp_path: Path) -> SimpleNamespace:
    source = (tmp_path / "source").resolve()
    artifact = (tmp_path / "artifact").resolve()
    cache = (tmp_path / "cache").resolve()
    for path in (source / "src", artifact, cache):
        path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        source_root=source,
        artifact_root=artifact,
        model_cache_root=cache,
        python_executable=Path(sys.executable).resolve(),
    )


def _install_contract_environment(
    runtime_paths: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    values = environment.deterministic_environment_values(runtime_paths)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        environment,
        "_PROCESS_START_CORE_ENVIRONMENT",
        tuple(sorted((key, values[key]) for key in environment.FROZEN_CORE_PROCESS_ENVIRONMENT)),
    )
    return values


def test_launch_contract_is_typed_immutable_secret_free_and_starts_at_16_8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    values = _install_contract_environment(runtime_paths, monkeypatch)
    contract = environment.freeze_deterministic_process_environment(runtime_paths)

    assert values["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert contract.environment == values
    assert contract.python_executable == Path(sys.executable).resolve()
    assert contract.pythonpath == runtime_paths.source_root / "src"
    assert contract.environment_sha256 == canonical_sha256(values)
    assert contract.evidence()["secrets_included"] is False
    with pytest.raises((AttributeError, TypeError)):
        contract.environment_items[0] = ("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # type: ignore[index]


def test_launch_builder_uses_an_explicit_allowlist_and_ignores_parent_secrets(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    first = environment.build_allowlisted_launch_environment(
        runtime_paths,
        {
            "PATH": "stable-path",
            "SYSTEMROOT": r"C:\Windows",
            "OPENAI_API_KEY": "first-secret",
            "PYTHONHOME": "wrong-python-home",
            "ARBITRARY_PARENT_FIELD": "first",
        },
    )
    second = environment.build_allowlisted_launch_environment(
        runtime_paths,
        {
            "PATH": "stable-path",
            "SYSTEMROOT": r"C:\Windows",
            "OPENAI_API_KEY": "different-secret",
            "PYTHONHOME": "different-python-home",
            "ARBITRARY_PARENT_FIELD": "second",
        },
    )

    assert first == second
    assert first["PATH"] == "stable-path"
    assert "OPENAI_API_KEY" not in first
    assert "PYTHONHOME" not in first
    assert "ARBITRARY_PARENT_FIELD" not in first
    assert canonical_sha256(first) == canonical_sha256(second)


@pytest.mark.parametrize("bad_value", [None, ":4096:8", ":32:8"])
def test_missing_legacy_or_other_cublas_launch_values_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_value: str | None,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    _install_contract_environment(runtime_paths, monkeypatch)
    if bad_value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", bad_value)

    with pytest.raises(RuntimeError, match="deterministic process environment differs"):
        environment.validate_deterministic_process_environment(runtime_paths, "bad_launch")


@pytest.mark.parametrize(
    "name,bad_value",
    [
        ("CUDA_LAUNCH_BLOCKING", "0"),
        ("ASCEND_LAUNCH_BLOCKING", "0"),
        ("HCCL_DETERMINISTIC", "0"),
        ("FLASH_ATTENTION_DETERMINISTIC", "0"),
        ("PYTHONHASHSEED", "0"),
    ],
)
def test_runtime_mutation_of_any_deterministic_field_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    bad_value: str,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    _install_contract_environment(runtime_paths, monkeypatch)
    environment.validate_deterministic_process_environment(runtime_paths, "before_mutation")
    monkeypatch.setenv(name, bad_value)

    with pytest.raises(RuntimeError, match="deterministic process environment differs"):
        environment.validate_deterministic_process_environment(runtime_paths, "after_mutation")


def test_transformers_4513_source_and_environment_writes_are_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    training_python = repository / ".venv-training" / "Scripts" / "python.exe"
    runtime_paths = SimpleNamespace(
        source_root=repository,
        artifact_root=tmp_path.resolve(),
        model_cache_root=(repository / "data" / "huggingface").resolve(),
        python_executable=training_python.resolve(),
    )
    child_environment = environment.build_allowlisted_launch_environment(runtime_paths, os.environ)
    path_values = {
        "source_root": str(runtime_paths.source_root),
        "artifact_root": str(runtime_paths.artifact_root),
        "model_cache_root": str(runtime_paths.model_cache_root),
        "python_executable": str(runtime_paths.python_executable),
    }
    code = f"""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import transformers
from foundry.training import grpo_environment as environment
path_values = {path_values!r}
runtime_paths = SimpleNamespace(
    source_root=Path(path_values["source_root"]),
    artifact_root=Path(path_values["artifact_root"]),
    model_cache_root=Path(path_values["model_cache_root"]),
    python_executable=Path(path_values["python_executable"]),
)
source = environment.transformers_determinism_source_evidence(transformers)
before = environment.validate_deterministic_process_environment(
    runtime_paths, "before_transformers_initialization", torch_module=torch
)
transformers.trainer_utils.enable_full_determinism(20260720, warn_only=False)
after = environment.validate_deterministic_process_environment(
    runtime_paths,
    "after_transformers_initialization",
    torch_module=torch,
    require_strict=True,
)
idempotence = environment.assert_idempotent_deterministic_initialization(before, after)
print(json.dumps({{
    "source": source,
    "idempotence": idempotence,
    "cuda_initialized": torch.cuda.is_initialized(),
}}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(training_python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=child_environment,
        cwd=repository,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    source = result["source"]

    assert source["source_file_sha256"] == (
        environment.FROZEN_TRANSFORMERS_TRAINER_UTILS_FILE_SHA256
    )
    assert source["function_source_sha256"] == (
        environment.FROZEN_TRANSFORMERS_DETERMINISM_FUNCTION_SHA256
    )
    assert source["environment_writes"] == dict(
        environment.FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT
    )
    assert result["idempotence"]["effective_environment_changed"] is False
    assert result["cuda_initialized"] is False


def test_every_transformers_written_field_is_predeclared() -> None:
    assert set(environment.FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT).issubset(
        environment.FROZEN_CORE_PROCESS_ENVIRONMENT
    )
    assert environment.FROZEN_CORE_PROCESS_ENVIRONMENT["PYTHONHASHSEED"] == "20260720"
    assert sys.flags.hash_randomization == 1


def test_process_start_snapshot_rejects_a_launch_value_changed_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    _install_contract_environment(runtime_paths, monkeypatch)
    start = dict(environment._PROCESS_START_CORE_ENVIRONMENT)
    start["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    monkeypatch.setattr(
        environment, "_PROCESS_START_CORE_ENVIRONMENT", tuple(sorted(start.items()))
    )

    with pytest.raises(RuntimeError, match="before Python launch"):
        environment.validate_deterministic_process_environment(runtime_paths, "entry")


def test_exception_and_warning_boundaries_always_revalidate_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    _install_contract_environment(runtime_paths, monkeypatch)
    stages: list[str] = []

    def validate(stage: str) -> None:
        stages.append(stage)
        environment.validate_deterministic_process_environment(runtime_paths, stage)

    class WarningOnlyBase:
        def _generate_and_score_completions(self, inputs: Any) -> Any:
            return inputs

        def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
            return (model, inputs, num_items_in_batch)

    guarded = environment.make_environment_guarded_trainer(WarningOnlyBase, validate)()
    assert guarded._generate_and_score_completions("value") == "value"
    assert guarded.training_step("model", "inputs") == ("model", "inputs", None)
    assert stages == [
        "before_generation",
        "after_generation",
        "before_backward",
        "after_backward",
    ]

    class MutatingFailureBase(WarningOnlyBase):
        def _generate_and_score_completions(self, inputs: Any) -> Any:
            monkeypatch.setenv("HCCL_DETERMINISTIC", "0")
            raise ValueError("model failure")

    failing = environment.make_environment_guarded_trainer(MutatingFailureBase, validate)()
    with pytest.raises(RuntimeError, match="deterministic process environment differs"):
        failing._generate_and_score_completions("value")


def test_optimizer_callback_and_strict_state_are_validated_outside_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    _install_contract_environment(runtime_paths, monkeypatch)
    stages: list[str] = []

    class Cuda:
        @staticmethod
        def is_initialized() -> bool:
            return True

    torch = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        cuda=Cuda(),
    )

    def validate(stage: str) -> None:
        stages.append(stage)
        environment.validate_deterministic_process_environment(
            runtime_paths,
            stage,
            torch_module=torch,
            require_strict=True,
        )

    callback = environment.make_environment_validation_callback(object, validate)
    control = object()
    assert callback.on_pre_optimizer_step(None, None, control) is control
    assert callback.on_optimizer_step(None, None, control) is control
    assert stages == ["before_optimizer", "after_optimizer"]


def test_exact_process_command_is_frozen_and_a_different_interpreter_is_rejected(
    tmp_path: Path,
) -> None:
    runtime_paths = _runtime_paths(tmp_path)
    command = [str(runtime_paths.python_executable), "-m", "foundry.training.grpo_runtime"]
    contract = environment.freeze_deterministic_process_environment(runtime_paths, command)
    assert contract.process_command == tuple(command)
    assert contract.process_command_sha256 == canonical_sha256(command)

    with pytest.raises(ValueError, match="different Python executable"):
        environment.freeze_deterministic_process_environment(
            runtime_paths,
            [str(tmp_path / "other-python.exe"), "-m", "foundry.training.grpo_runtime"],
        )
