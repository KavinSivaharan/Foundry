"""Autonomous state-machine controller for the single authorized Foundry Cycle 1."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from foundry.cycle.contract import (
    CONTROLLER_ID,
    CYCLE_ID,
    RECOVERY_EXECUTION_ID,
    RECOVERY_PARENT_REJECTION_SHA256,
    RECOVERY_PARENT_RUNTIME_SHA256,
    RECOVERY_RUNTIME_ROOT,
    CycleConfig,
    CycleContractError,
    bind_cycle_execution,
    content_free_projection,
    cycle_execution_metadata,
    git_output,
    load_cycle_config,
    read_json_object,
    validate_file_identity,
    validate_frozen_source,
    validate_process_environment,
)
from foundry.cycle.generation_observability import (
    ensure_recovery_runtime,
    recovery_identity,
)
from foundry.cycle.state import StateStore
from foundry.phase2.l3_grpo_contract import _starting_adapter, model_manifest
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _verified_payload(path: Path, hash_field: str) -> dict[str, Any]:
    value = read_json_object(path)
    supplied = value.get(hash_field)
    payload = {key: item for key, item in value.items() if key != hash_field}
    if supplied != canonical_sha256(payload):
        raise CycleContractError(f"{path.name} does not reconstruct")
    return value


def _active_record(config: CycleConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry_id": "foundry-active-model-registry-v1",
        "logical_model_id": "untouched-base",
        "base_model_id": config.section("model")["model_id"],
        "base_revision": config.section("model")["revision"],
        "adapter_sha256": None,
        "checkpoint": None,
        "cycle_id": None,
    }


def copy_adapter_once(candidate: Path, destination: Path, expected_sha256: str) -> None:
    """Copy one adapter atomically and reject duplicate promotion targets."""

    temporary = destination.parent / f".{destination.name}.staging"
    if destination.exists() or temporary.exists():
        raise CycleContractError("duplicate Cycle 1 promotion is prohibited")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate, temporary)
    if directory_sha256(temporary) != expected_sha256:
        raise RuntimeError("staged promoted adapter hash differs")
    os.replace(temporary, destination)
    if directory_sha256(destination) != expected_sha256:
        raise RuntimeError("promoted registry adapter hash differs")


def assert_active_preserved(active_path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify rejection left the active-model record byte-semantically unchanged."""

    active = read_json_object(active_path)
    if canonical_sha256(active) != expected_sha256:
        raise CycleContractError("active model changed despite candidate rejection")
    return active


def evaluate_promotion_gate(
    *,
    development: dict[str, Any],
    holdout: dict[str, Any],
    benchmark: dict[str, Any],
    training: dict[str, Any],
    contract: dict[str, Any],
    identities_reconstruct: bool = True,
) -> dict[str, bool]:
    """Evaluate all eleven frozen promotion conditions without side effects."""

    return {
        "development_retention_passed": development["passed"] is True,
        "holdout_v2_retention_passed": holdout["passed"] is True,
        "candidate_at_least_522": benchmark["correct"] >= contract["minimum_candidate_correct"],
        "candidate_greater_than_base": benchmark["correct"] > contract["untouched_base_correct"],
        "candidate_greater_than_targeted_start": benchmark["correct"]
        > contract["targeted_l3_correct"],
        "extractability_at_least_91_38_percent": benchmark["extractability"]
        >= contract["minimum_extractability"],
        "zero_backend_failures": benchmark["backend_failures"] == 0,
        "untargeted_decline_at_most_2_points": benchmark["category_effects"][
            "untargeted_gate_passed"
        ]
        is True,
        "adapter_offline_reload": training["offline_reload"] is True,
        "base_parameters_unchanged": training["base_parameters_unchanged"] is True,
        "cycle_identities_reconstruct": identities_reconstruct,
    }


def _identity_payload(
    config: CycleConfig,
    source: dict[str, str],
    interpreter_sha256: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": CYCLE_ID,
        "execution": cycle_execution_metadata(config),
        "controller_id": CONTROLLER_ID,
        "config_sha256": config.sha256,
        "source": source,
        "interpreter_sha256": interpreter_sha256,
        "environment_evidence_sha256": config.section("environment")["environment_evidence_sha256"],
        "combined_child_environment_sha256": config.section("environment")[
            "combined_child_environment_sha256"
        ],
    }
    value["experiment_identity_sha256"] = canonical_sha256(value)
    return value


def ensure_runtime_identity(
    config: CycleConfig,
) -> tuple[dict[str, str], str, str]:
    """Create or verify the exact external Cycle 1 runtime root."""

    validate_process_environment(config=config)
    source = validate_frozen_source(config)
    interpreter_sha256 = file_sha256(Path(sys.executable))
    environment = config.section("environment")
    if (
        Path(sys.executable).resolve() != Path(str(environment["training_interpreter"])).resolve()
        or interpreter_sha256 != environment["interpreter_sha256"]
    ):
        raise CycleContractError("controller is not using the authorized training interpreter")
    identity = _identity_payload(config, source, interpreter_sha256)
    root = config.runtime_root
    identity_path = root / "experiment_identity.json"
    if root.exists():
        if not identity_path.is_file():
            if config.execution_id != RECOVERY_EXECUTION_ID:
                raise CycleContractError("existing Cycle 1 runtime root has no exact identity")
            ensure_recovery_runtime(
                root,
                recovery_identity(
                    config_sha256=config.sha256,
                    model_revision=str(config.section("model")["revision"]),
                    dataset_sha256=str(config.section("dataset")["identity_sha256"]),
                    starting_adapter_sha256=str(config.section("warm_start")["adapter_sha256"]),
                    prior_runtime_tree_sha256=RECOVERY_PARENT_RUNTIME_SHA256,
                    prior_rejection_sha256=RECOVERY_PARENT_REJECTION_SHA256,
                ),
            )
            _write_json(identity_path, identity, exclusive=True)
        existing = read_json_object(identity_path)
        if existing != identity:
            raise CycleContractError("existing Cycle 1 runtime root belongs to another experiment")
    else:
        root.mkdir(parents=False, exist_ok=False)
        _write_json(identity_path, identity, exclusive=True)
    registry = config.registry_root
    registry.mkdir(parents=True, exist_ok=True)
    active_path = registry / "active_model.json"
    expected_active = _active_record(config)
    if active_path.exists():
        active = read_json_object(active_path)
        if not (root / "state.json").exists() and active != expected_active:
            raise CycleContractError("unexpected preexisting active-model registry state")
    else:
        _write_json(active_path, expected_active, exclusive=True)
    initial_active_sha256 = canonical_sha256(expected_active)
    return source, interpreter_sha256, initial_active_sha256


def _run_process(
    *,
    config: CycleConfig,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("controller child-process log already exists")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("x", encoding="utf-8") as stdout,
        stderr_path.open("x", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            command,
            cwd=config.source_root / "src",
            env=dict(os.environ),
            shell=False,
            check=False,
            text=True,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Cycle 1 child process failed with {result.returncode}: {stderr_path}")


def _runtime_command(
    *,
    config: CycleConfig,
    operation: str,
    output_directory: Path,
    log_label: str,
    input_path: Path | None = None,
    training_directory: Path | None = None,
    adapter_path: Path | None = None,
    expected_adapter_sha256: str | None = None,
    smoke: bool = False,
) -> None:
    command = [
        sys.executable,
        "-m",
        "foundry.cycle.runtime",
        operation,
        "--config",
        str(config.path),
        "--output-directory",
        str(output_directory),
    ]
    if input_path is not None:
        command.extend(["--input-path", str(input_path)])
    if training_directory is not None:
        command.extend(["--training-directory", str(training_directory)])
    if adapter_path is not None:
        command.extend(["--adapter-path", str(adapter_path)])
    if expected_adapter_sha256 is not None:
        command.extend(["--expected-adapter-sha256", expected_adapter_sha256])
    if smoke:
        command.append("--smoke")
    source = validate_frozen_source(config)
    command.extend(
        [
            "--execution-id",
            config.execution_id,
            "--source-root",
            str(config.source_root),
            "--source-commit",
            source["commit"],
            "--source-tree",
            source["tree"],
        ]
    )
    _run_process(
        config=config,
        command=command,
        stdout_path=config.runtime_root / "logs" / f"{log_label}.stdout.txt",
        stderr_path=config.runtime_root / "logs" / f"{log_label}.stderr.txt",
    )


def _smoke_exact_projection(
    *,
    config: CycleConfig,
    trial: int,
    generation: dict[str, Any],
    selection: dict[str, Any],
    corpus: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    metrics = cast(list[dict[str, Any]], training["step_metrics"])
    log_root = config.runtime_root / "logs"
    stderr_sha256s = {
        operation: file_sha256(log_root / f"compatibility-trial-{trial}-{operation}.stderr.txt")
        for operation in ("generation", "selection", "corpus", "training")
    }
    fallback_records = [
        item
        for item in cast(list[dict[str, Any]], corpus["records"])
        if item["variant"] == "original"
    ]
    warning_contract = cast(dict[str, Any], generation["warning_contract_evidence"])
    return {
        "generated_token_ids_sha256": generation["generated_token_ids_sha256"],
        "decoded_completion_hashes_sha256": generation["decoded_completion_hashes_sha256"],
        "completion_token_counts_sha256": generation["completion_token_counts_sha256"],
        "warning_identity_sha256": generation["warning_identity_sha256"],
        "generation_rng_transitions_sha256": generation["cycle_rng_transitions_sha256"],
        "warning_contract_call_evidence_sha256": warning_contract["call_evidence_sha256"],
        "warning_contract_rng_transitions_sha256": warning_contract["rng_transitions_sha256"],
        "generation_launch_evidence_sha256": canonical_sha256(generation["launch_evidence"]),
        "component_decisions_sha256": selection["component_decisions_sha256"],
        "selected_trace_manifest_sha256": selection["selected_trace_manifest_sha256"],
        "fallback_identities_sha256": canonical_sha256(fallback_records),
        "corpus_sha256": corpus["corpus_sha256"],
        "training_schedule_sha256": corpus["schedule_sha256"],
        "losses": [item["loss"] for item in metrics],
        "gradient_hashes": [item["gradient_sha256"] for item in metrics],
        "optimizer_state_hashes": [item["optimizer_state_sha256"] for item in metrics],
        "scheduler_state_hashes": [item["scheduler_state_sha256"] for item in metrics],
        "training_rng_transition_sha256": training["rng_transition_sha256"],
        "training_launch_evidence_sha256": canonical_sha256(training["launch_evidence"]),
        "final_adapter_tensor_sha256": training["final_lora_tensor_sha256"],
        "final_adapter_directory_sha256": training["final_adapter_sha256"],
        "stderr_sha256s": stderr_sha256s,
        "source": generation["source"],
        "interpreter_sha256": file_sha256(Path(sys.executable)),
        "environment_sha256": config.section("environment")["combined_child_environment_sha256"],
        "execution": cycle_execution_metadata(config),
    }


def _run_compatibility_trial_processes(
    *,
    config: CycleConfig,
    root: Path,
    trial: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    _runtime_command(
        config=config,
        operation="generate",
        output_directory=root / "generation",
        log_label=f"compatibility-trial-{trial}-generation",
        smoke=True,
    )
    _runtime_command(
        config=config,
        operation="filter",
        output_directory=root / "selection",
        input_path=root / "generation" / "candidates.jsonl",
        log_label=f"compatibility-trial-{trial}-selection",
        smoke=True,
    )
    _runtime_command(
        config=config,
        operation="freeze-corpus",
        output_directory=root / "corpus",
        input_path=root / "selection" / "selected_traces.jsonl",
        log_label=f"compatibility-trial-{trial}-corpus",
        smoke=True,
    )
    _runtime_command(
        config=config,
        operation="train",
        output_directory=root / "training",
        input_path=root / "corpus",
        log_label=f"compatibility-trial-{trial}-training",
        smoke=True,
    )
    return (
        _verified_payload(
            root / "generation" / "summary.json",
            "generation_sha256",
        ),
        _verified_payload(
            root / "selection" / "summary.json",
            "selection_sha256",
        ),
        _verified_payload(root / "corpus" / "manifest.json", "corpus_sha256"),
        _verified_payload(
            root / "training" / "summary.json",
            "training_sha256",
        ),
    )


def _finalize_compatibility_rejection(
    *,
    config: CycleConfig,
    source: dict[str, str],
    interpreter_sha256: str,
    initial_active_sha256: str,
    reason: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    backend = ProductionBackend(config, source, initial_active_sha256)
    stage_evidence = backend._persist("compatibility_smoke", evidence)
    store = StateStore(
        path=config.runtime_root / "state.json",
        config=config,
        source=source,
        interpreter_sha256=interpreter_sha256,
    )
    store.reject_at_compatibility_preflight(reason, stage_evidence)
    return CycleController(config=config, store=store, backend=backend).run()


def run_compatibility_smoke(
    config_path: Path,
    trial: int,
    *,
    execution_id: str | None = None,
) -> dict[str, Any]:
    """Run one of the two predeclared fresh compatibility trials."""

    if trial not in {1, 2}:
        raise ValueError("compatibility smoke trial must be 1 or 2")
    config = bind_cycle_execution(load_cycle_config(config_path), execution_id)
    source, interpreter_sha256, initial_active_sha256 = ensure_runtime_identity(config)
    if (config.runtime_root / "state.json").exists():
        raise CycleContractError("compatibility execution is closed after a cycle decision")
    baseline_evidence = reconstruct_baseline_evidence(config, source)
    baseline_evidence_sha256 = canonical_sha256(baseline_evidence)
    root = config.runtime_root / "compatibility" / f"trial-{trial}"
    if root.exists():
        raise FileExistsError(f"compatibility trial {trial} already exists")
    if (
        trial == 2
        and not (config.runtime_root / "compatibility" / "trial-1" / "summary.json").is_file()
    ):
        raise CycleContractError("compatibility trial 2 cannot precede trial 1")
    try:
        generation, selection, corpus, training = _run_compatibility_trial_processes(
            config=config,
            root=root,
            trial=trial,
        )
    except Exception as error:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "smoke_id": "foundry-cycle1-compatibility-smoke-v1",
            "execution": cycle_execution_metadata(config),
            "trial": trial,
            "passed": False,
            "classification": "compatibility_smoke_exception",
            "error_type": type(error).__name__,
            "config_sha256": config.sha256,
            "source": source,
            "interpreter_sha256": interpreter_sha256,
            "baseline_evidence_sha256": baseline_evidence_sha256,
        }
        failure["smoke_sha256"] = canonical_sha256(failure)
        _write_json(
            config.runtime_root / "compatibility" / f"trial-{trial}-failure.json",
            failure,
            exclusive=True,
        )
        _finalize_compatibility_rejection(
            config=config,
            source=source,
            interpreter_sha256=interpreter_sha256,
            initial_active_sha256=initial_active_sha256,
            reason=f"compatibility_smoke_exception:{type(error).__name__}",
            evidence=failure,
        )
        return failure
    selection_checks = cast(dict[str, bool], selection["checks_before_overlap_audit"])
    warning_contract = cast(dict[str, Any], generation["warning_contract_evidence"])
    gates = {
        "four_generation_prompts": generation["prompts"] == 4,
        "eight_completions_per_prompt": generation["completions_per_prompt"] == 8,
        "thirty_two_generation_attempts": generation["attempted_completions"] == 32,
        "zero_generation_backend_failures": generation["backend_failures"] == 0,
        "thirty_two_output_bearing_completions": (generation["output_bearing_completions"] == 32),
        "thirty_two_successful_token_id_packets": (generation["successful_token_id_packets"] == 32),
        "empty_exception_evidence": generation["exception_evidence_failures"] == 0,
        "all_generation_warnings_whitelisted": (
            warning_contract["all_warnings_whitelisted"] is True
        ),
        "expected_top_p_warning_per_attempt": (
            warning_contract["all_expected_warnings_present"] is True
            and warning_contract["generation_calls"] == 32
        ),
        "strict_determinism_restored": (warning_contract["all_strict_restorations"] is True),
        "every_completion_scored": selection["attempted_completions"] == 32,
        "every_prompt_has_exact_attempts": (
            selection_checks["all_prompts_processed_once"]
            and selection_checks["exactly_eight_attempts_per_prompt"]
        ),
        "zero_verifier_disagreements": selection_checks["zero_verifier_disagreements"],
        "selected_or_fallback_decision_per_prompt": (
            selection["selected_prompts"] + selection["fallback_prompts"] == 4
            and len(cast(list[dict[str, Any]], corpus["records"])) == 4
        ),
        "four_unique_training_records": corpus["unique_records"] == 4,
        "two_optimizer_steps": training["optimizer_steps"] == 2,
        "finite_losses": all(
            math_isfinite(float(item["loss"]))
            for item in cast(list[dict[str, Any]], training["step_metrics"])
        ),
        "finite_gradients": all(
            math_isfinite(float(item["gradient_norm"]))
            for item in cast(list[dict[str, Any]], training["step_metrics"])
        ),
        "positive_lr_positive_gradient": training["positive_lr_positive_gradient"],
        "lora_tensors_changed": training["changed_lora_tensor_count"] > 0,
        "base_parameters_unchanged": training["base_parameters_unchanged"],
        "offline_reload": training["offline_reload"],
        "adapter_saved": bool(
            cast(dict[str, dict[str, Any]], training["checkpoints"])
            .get("2", {})
            .get("adapter_sha256")
        ),
        "base_restoration": training["base_restoration"],
        "no_cpu_offload": training["cpu_offload"] is False,
    }
    exact = _smoke_exact_projection(
        config=config,
        trial=trial,
        generation=generation,
        selection=selection,
        corpus=corpus,
        training=training,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "smoke_id": "foundry-cycle1-compatibility-smoke-v1",
        "execution": cycle_execution_metadata(config),
        "trial": trial,
        "config_sha256": config.sha256,
        "source": source,
        "interpreter_sha256": interpreter_sha256,
        "baseline_evidence_sha256": baseline_evidence_sha256,
        "gates": gates,
        "passed": all(gates.values()),
        "exact_evidence": exact,
    }
    summary["smoke_sha256"] = canonical_sha256(summary)
    _write_json(root / "summary.json", summary)
    if summary["passed"] is not True:
        _finalize_compatibility_rejection(
            config=config,
            source=source,
            interpreter_sha256=interpreter_sha256,
            initial_active_sha256=initial_active_sha256,
            reason="compatibility_smoke_gate_failed",
            evidence=summary,
        )
        return summary
    if trial == 2:
        first = _verified_payload(
            config.runtime_root / "compatibility" / "trial-1" / "summary.json",
            "smoke_sha256",
        )
        duplicate_checks = {
            "trial_1_passed": first["passed"] is True,
            "trial_2_passed": summary["passed"] is True,
            "exact_evidence_equal": first["exact_evidence"] == exact,
            "same_config": first["config_sha256"] == config.sha256,
            "same_source": first["source"] == source,
            "same_interpreter": first["interpreter_sha256"] == interpreter_sha256,
            "same_baseline_evidence": (
                first["baseline_evidence_sha256"] == baseline_evidence_sha256
            ),
            "same_execution": first["execution"] == cycle_execution_metadata(config),
        }
        duplicate: dict[str, Any] = {
            "schema_version": 1,
            "compatibility_id": "foundry-cycle1-duplicate-compatibility-v1",
            "execution": cycle_execution_metadata(config),
            "checks": duplicate_checks,
            "passed": all(duplicate_checks.values()),
            "trial_1_smoke_sha256": first["smoke_sha256"],
            "trial_2_smoke_sha256": summary["smoke_sha256"],
            "exact_evidence_sha256": canonical_sha256(exact),
        }
        duplicate["compatibility_sha256"] = canonical_sha256(duplicate)
        _write_json(
            config.runtime_root / "compatibility" / "summary.json",
            duplicate,
            exclusive=True,
        )
        if duplicate["passed"] is not True:
            _finalize_compatibility_rejection(
                config=config,
                source=source,
                interpreter_sha256=interpreter_sha256,
                initial_active_sha256=initial_active_sha256,
                reason="compatibility_smoke_mismatch",
                evidence=duplicate,
            )
    return summary


def math_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


@dataclass(frozen=True)
class StageOutcome:
    passed: bool
    evidence: dict[str, Any]
    classification: str | None = None
    terminal_decision: str | None = None


class StageBackend(Protocol):
    def execute(self, stage: str, state: dict[str, Any]) -> StageOutcome:
        """Execute one exact stage and return content-free evidence."""

    def finalize(self, state: dict[str, Any]) -> None:
        """Finalize content-free publication after the final state hash exists."""


def reconstruct_baseline_evidence(
    config: CycleConfig,
    source: dict[str, str],
) -> dict[str, Any]:
    """Reconstruct every frozen pre-model identity without running an evaluation."""

    primary = config.artifact_root
    if (
        git_output(primary, "branch", "--show-current") != "main"
        or git_output(primary, "rev-parse", "HEAD") != source["commit"]
        or git_output(primary, "rev-parse", "origin/main") != source["commit"]
        or git_output(primary, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise CycleContractError("primary repository drifted after controller publication")
    model = model_manifest(primary)
    adapters = {arm: _starting_adapter(primary, arm) for arm in ("generic", "targeted")}
    dataset = config.section("dataset")
    validate_file_identity(
        config,
        str(dataset["targeted_training_relative_path"]),
        str(dataset["targeted_training_sha256"]),
    )
    validate_file_identity(
        config,
        str(dataset["targeted_validation_relative_path"]),
        str(dataset["targeted_validation_sha256"]),
    )
    dependencies = cast(
        dict[str, dict[str, str]],
        config.payload["scientific_dependencies"],
    )
    dependency_hashes = {}
    for name, contract in dependencies.items():
        path = validate_file_identity(
            config,
            contract["relative_path"],
            contract["sha256"],
        )
        dependency_hashes[name] = file_sha256(path)
    starting = _verified_payload(
        primary / "results/phase2_vetted_corpus/milestone14a_starting_state.json",
        "starting_state_sha256",
    )
    frozen = cast(dict[str, Any], starting["frozen_gsm1k"])
    if (
        model["model_id"] != config.section("model")["model_id"]
        or model["revision"] != config.section("model")["revision"]
        or model["manifest_sha256"] != config.section("model")["snapshot_manifest_sha256"]
        or starting["dataset_sha256"] != dataset["identity_sha256"]
        or cast(dict[str, Any], starting["model"])["manifest_sha256"] != model["manifest_sha256"]
        or frozen["base"]["correct"] != 521
        or frozen["generic"]["correct"] != 517
        or frozen["targeted"]["correct"] != 519
        or adapters["generic"]["directory_sha256"]
        != config.section("warm_start")["generic_adapter_sha256"]
        or adapters["targeted"]["directory_sha256"]
        != config.section("warm_start")["adapter_sha256"]
    ):
        raise CycleContractError("frozen baseline results or adapters differ")
    return {
        "schema_version": 1,
        "execution": cycle_execution_metadata(config),
        "source": source,
        "model": model,
        "dataset_sha256": dataset["identity_sha256"],
        "targeted_training_sha256": dataset["targeted_training_sha256"],
        "targeted_validation_sha256": dataset["targeted_validation_sha256"],
        "adapters": {
            name: {
                "directory_sha256": value["directory_sha256"],
                "adapter_configuration": value["adapter_configuration"],
            }
            for name, value in adapters.items()
        },
        "baseline_results": {
            "base": 521,
            "generic": 517,
            "targeted": 519,
        },
        "holdout_v2_subset_sha256": config.section("retention")["holdout_v2"]["subset_sha256"],
        "scientific_dependency_hashes": dependency_hashes,
        "active_model_sha256": canonical_sha256(
            read_json_object(config.registry_root / "active_model.json")
        ),
        "sealed_final_accessed": False,
    }


class ProductionBackend:
    """Real model/data backend for the one authorized Cycle 1 execution."""

    def __init__(
        self,
        config: CycleConfig,
        source: dict[str, str],
        initial_active_sha256: str,
    ) -> None:
        self.config = config
        self.source = source
        self.initial_active_sha256 = initial_active_sha256
        self.evidence_root = config.runtime_root / "evidence"

    def _persist(self, stage: str, value: dict[str, Any]) -> dict[str, Any]:
        execution = cycle_execution_metadata(self.config)
        supplied_execution = value.get("execution")
        if supplied_execution is not None and supplied_execution != execution:
            raise CycleContractError(f"stage {stage} execution identity differs")
        value["execution"] = execution
        projected = content_free_projection(value)
        if projected != value:
            raise CycleContractError(f"stage {stage} attempted to persist content")
        value["stage_evidence_sha256"] = canonical_sha256(value)
        _write_json(self.evidence_root / f"{stage}.json", value)
        return value

    def execute(self, stage: str, state: dict[str, Any]) -> StageOutcome:
        method = getattr(self, f"_stage_{stage}")
        return cast(StageOutcome, method(state))

    def _stage_verify_baseline(self, state: dict[str, Any]) -> StageOutcome:
        del state
        evidence = self._persist(
            "verify_baseline",
            reconstruct_baseline_evidence(self.config, self.source),
        )
        return StageOutcome(True, evidence)

    def _stage_diagnose(self, state: dict[str, Any]) -> StageOutcome:
        del state
        benchmark = self.config.section("benchmark")
        taxonomy_path = validate_file_identity(
            self.config,
            str(benchmark["taxonomy_relative_path"]),
            str(benchmark["taxonomy_file_sha256"]),
        )
        taxonomy = read_json_object(taxonomy_path)
        evidence = self._persist(
            "diagnose",
            {
                "schema_version": 1,
                "diagnosis_id": "foundry-cycle1-published-aggregate-diagnosis-v1",
                "selected_reasoning_categories": taxonomy["selected_reasoning_categories"],
                "primary_category_counts": taxonomy["primary_category_counts"],
                "failure_kind_counts": taxonomy["failure_kind_counts"],
                "source_taxonomy_file_sha256": file_sha256(taxonomy_path),
                "content_only_aggregate_used": True,
                "gsm1k_question_content_used": False,
            },
        )
        return StageOutcome(True, evidence)

    def _stage_load_safe_warm_start(self, state: dict[str, Any]) -> StageOutcome:
        del state
        value = _starting_adapter(self.config.artifact_root, "targeted")
        evidence = self._persist(
            "load_safe_warm_start",
            {
                "schema_version": 1,
                "adapter_sha256": value["directory_sha256"],
                "adapter_configuration": value["adapter_configuration"],
                "trainable_inventory": value["trainable_inventory"],
                "development_retention": {
                    "adjudication": "183/187",
                    "anchor": "205/210",
                },
                "holdout_v2": "315/317",
                "gsm1k": "519/814",
                "authorized": True,
            },
        )
        return StageOutcome(True, evidence)

    def _stage_generate_candidates(self, state: dict[str, Any]) -> StageOutcome:
        del state
        output = self.config.runtime_root / "production" / "generation"
        _runtime_command(
            config=self.config,
            operation="generate",
            output_directory=output,
            log_label="production-generation",
        )
        summary = _verified_payload(output / "summary.json", "generation_sha256")
        evidence = self._persist(
            "generate_candidates",
            {
                key: value
                for key, value in summary.items()
                if key not in {"runtime_seconds", "launch_evidence"}
            },
        )
        passed = summary["backend_failures"] == 0
        return StageOutcome(
            passed,
            evidence,
            None if passed else "generation_backend_failure",
        )

    def _stage_verify_and_select_traces(self, state: dict[str, Any]) -> StageOutcome:
        del state
        production = self.config.runtime_root / "production"
        output = production / "selection"
        _runtime_command(
            config=self.config,
            operation="filter",
            output_directory=output,
            input_path=production / "generation" / "candidates.jsonl",
            log_label="production-selection",
        )
        summary = _verified_payload(output / "summary.json", "selection_sha256")
        evidence = self._persist("verify_and_select_traces", dict(summary))
        passed = bool(summary["passed_before_overlap_audit"])
        return StageOutcome(
            passed,
            evidence,
            None if passed else "verifier_filtered_optimizer_signal_insufficient",
        )

    def _stage_freeze_training_corpus(self, state: dict[str, Any]) -> StageOutcome:
        del state
        production = self.config.runtime_root / "production"
        output = production / "corpus_freeze"
        _runtime_command(
            config=self.config,
            operation="freeze-corpus",
            output_directory=output,
            input_path=production / "selection" / "selected_traces.jsonl",
            log_label="production-corpus-freeze",
        )
        summary = _verified_payload(
            output / "summary.json",
            "corpus_freeze_sha256",
        )
        evidence = self._persist("freeze_training_corpus", dict(summary))
        passed = bool(summary["passed"])
        return StageOutcome(
            passed,
            evidence,
            None if passed else "verifier_filtered_optimizer_signal_insufficient",
        )

    def _stage_compatibility_smoke(self, state: dict[str, Any]) -> StageOutcome:
        del state
        path = self.config.runtime_root / "compatibility" / "summary.json"
        if not path.is_file():
            raise CycleContractError("duplicate compatibility evidence is absent")
        summary = _verified_payload(path, "compatibility_sha256")
        evidence = self._persist("compatibility_smoke", dict(summary))
        return StageOutcome(
            bool(summary["passed"]),
            evidence,
            None if summary["passed"] else "compatibility_smoke_mismatch",
        )

    def _stage_train_candidate(self, state: dict[str, Any]) -> StageOutcome:
        del state
        production = self.config.runtime_root / "production"
        output = production / "training"
        _runtime_command(
            config=self.config,
            operation="train",
            output_directory=output,
            input_path=production / "corpus_freeze" / "corpus",
            log_label="production-training",
        )
        summary = _verified_payload(output / "summary.json", "training_sha256")
        evidence = self._persist(
            "train_candidate",
            {
                key: value
                for key, value in summary.items()
                if key not in {"runtime_seconds", "launch_evidence"}
            },
        )
        passed = bool(
            summary["base_parameters_unchanged"]
            and summary["offline_reload"]
            and summary["positive_lr_positive_gradient"]
            and summary["changed_lora_tensor_count"] > 0
        )
        return StageOutcome(
            passed,
            evidence,
            None if passed else "continuation_training_gate_failed",
        )

    def _stage_select_checkpoint(self, state: dict[str, Any]) -> StageOutcome:
        del state
        summary = _verified_payload(
            self.config.runtime_root / "production" / "training" / "summary.json",
            "training_sha256",
        )
        checkpoints = cast(dict[str, dict[str, Any]], summary["checkpoints"])
        if set(checkpoints) != {"8", "16", "32"}:
            raise CycleContractError("candidate checkpoint set differs")
        evidence = self._persist(
            "select_checkpoint",
            {
                "schema_version": 1,
                "selection_rule": "latest_checkpoint_passing_both_development_instruments",
                "candidate_checkpoints": {
                    step: {
                        "adapter_sha256": value["adapter_sha256"],
                        "adapter_tensor_sha256": value["adapter_tensor_sha256"],
                        "validation_ce": value["validation_ce"],
                    }
                    for step, value in checkpoints.items()
                },
                "holdout_used": False,
                "gsm1k_used": False,
            },
        )
        return StageOutcome(True, evidence)

    def _stage_development_retention(self, state: dict[str, Any]) -> StageOutcome:
        del state
        production = self.config.runtime_root / "production"
        output = production / "development_retention"
        _runtime_command(
            config=self.config,
            operation="development-retention",
            output_directory=output,
            training_directory=production / "training",
            log_label="production-development-retention",
        )
        summary = _verified_payload(
            output / "summary.json",
            "development_retention_sha256",
        )
        evidence = self._persist("development_retention", dict(summary))
        return StageOutcome(
            bool(summary["passed"]),
            evidence,
            None if summary["passed"] else "development_retention_failed",
        )

    def _selected_checkpoint(self) -> int:
        summary = _verified_payload(
            self.config.runtime_root / "production" / "development_retention" / "summary.json",
            "development_retention_sha256",
        )
        checkpoint = summary["selected_checkpoint"]
        if checkpoint not in {8, 16, 32}:
            raise CycleContractError("development selected checkpoint differs")
        return int(checkpoint)

    def _selected_adapter_identity(self) -> tuple[Path, str]:
        checkpoint = self._selected_checkpoint()
        adapter = (
            self.config.runtime_root
            / "production"
            / "training"
            / f"checkpoint-{checkpoint}"
            / "adapter"
        )
        training = _verified_payload(
            self.config.runtime_root / "production" / "training" / "summary.json",
            "training_sha256",
        )
        expected = str(
            cast(dict[str, dict[str, Any]], training["checkpoints"])[str(checkpoint)][
                "adapter_sha256"
            ]
        )
        if directory_sha256(adapter) != expected:
            raise CycleContractError("selected candidate adapter identity differs")
        return adapter, expected

    def _stage_holdout_retention(self, state: dict[str, Any]) -> StageOutcome:
        del state
        output = self.config.runtime_root / "production" / "holdout_retention"
        adapter, adapter_sha256 = self._selected_adapter_identity()
        _runtime_command(
            config=self.config,
            operation="holdout-retention",
            output_directory=output,
            adapter_path=adapter,
            expected_adapter_sha256=adapter_sha256,
            log_label="production-holdout-retention",
        )
        summary = _verified_payload(
            output / "summary.json",
            "holdout_retention_sha256",
        )
        evidence = self._persist("holdout_retention", dict(summary))
        return StageOutcome(
            bool(summary["passed"]),
            evidence,
            None if summary["passed"] else "holdout_retention_failed",
        )

    def _stage_benchmark(self, state: dict[str, Any]) -> StageOutcome:
        del state
        output = self.config.runtime_root / "production" / "benchmark"
        adapter, adapter_sha256 = self._selected_adapter_identity()
        _runtime_command(
            config=self.config,
            operation="benchmark",
            output_directory=output,
            adapter_path=adapter,
            expected_adapter_sha256=adapter_sha256,
            log_label="production-benchmark",
        )
        summary = _verified_payload(output / "summary.json", "benchmark_sha256")
        evidence = self._persist("benchmark", dict(summary))
        return StageOutcome(True, evidence)

    def _cycle_identity_chain(self, state: dict[str, Any]) -> dict[str, Any]:
        production = self.config.runtime_root / "production"
        baseline = _verified_payload(
            self.evidence_root / "verify_baseline.json",
            "stage_evidence_sha256",
        )
        baseline_payload = {
            key: value for key, value in baseline.items() if key != "stage_evidence_sha256"
        }
        if baseline_payload != reconstruct_baseline_evidence(
            self.config,
            self.source,
        ):
            raise CycleContractError("baseline evidence drifted before decision")
        compatibility = _verified_payload(
            self.config.runtime_root / "compatibility" / "summary.json",
            "compatibility_sha256",
        )
        generation = _verified_payload(
            production / "generation" / "summary.json",
            "generation_sha256",
        )
        if (
            file_sha256(production / "generation" / "candidates.jsonl")
            != generation["raw_file_sha256"]
        ):
            raise CycleContractError("generation artifact drifted before decision")
        selection = _verified_payload(
            production / "selection" / "summary.json",
            "selection_sha256",
        )
        if (
            file_sha256(production / "selection" / "component_decisions.jsonl")
            != selection["component_decisions_file_sha256"]
            or file_sha256(production / "selection" / "selected_traces.jsonl")
            != selection["selected_trace_file_sha256"]
        ):
            raise CycleContractError("selection artifact drifted before decision")
        corpus_freeze = _verified_payload(
            production / "corpus_freeze" / "summary.json",
            "corpus_freeze_sha256",
        )
        corpus = _verified_payload(
            production / "corpus_freeze" / "corpus" / "manifest.json",
            "corpus_sha256",
        )
        corpus_root = production / "corpus_freeze" / "corpus"
        if (
            corpus_freeze["corpus"] != corpus
            or file_sha256(corpus_root / "task_corpus.jsonl") != corpus["task_corpus_file_sha256"]
            or file_sha256(corpus_root / "schedule.json") != corpus["schedule_file_sha256"]
        ):
            raise CycleContractError("corpus artifact drifted before decision")
        training = _verified_payload(
            production / "training" / "summary.json",
            "training_sha256",
        )
        checkpoints = cast(dict[str, dict[str, Any]], training["checkpoints"])
        for checkpoint, item in checkpoints.items():
            adapter = production / "training" / f"checkpoint-{checkpoint}" / "adapter"
            if directory_sha256(adapter) != item["adapter_sha256"]:
                raise CycleContractError("training checkpoint drifted before decision")
        development = _verified_payload(
            production / "development_retention" / "summary.json",
            "development_retention_sha256",
        )
        trajectory = cast(dict[str, dict[str, Any]], development["trajectory"])
        for checkpoint, checkpoint_result in trajectory.items():
            suites = cast(dict[str, dict[str, Any]], checkpoint_result["suites"])
            for name, suite in suites.items():
                assessment_path = (
                    production
                    / "development_retention"
                    / f"checkpoint-{checkpoint}"
                    / name
                    / "assessment.json"
                )
                if file_sha256(assessment_path) != suite["assessment_file_sha256"]:
                    raise CycleContractError(
                        "development retention artifact drifted before decision"
                    )
        selected_checkpoint = str(development["selected_checkpoint"])
        expected_adapter = checkpoints[selected_checkpoint]["adapter_sha256"]
        holdout = _verified_payload(
            production / "holdout_retention" / "summary.json",
            "holdout_retention_sha256",
        )
        benchmark = _verified_payload(
            production / "benchmark" / "summary.json",
            "benchmark_sha256",
        )
        if (
            holdout["adapter_sha256"] != expected_adapter
            or benchmark["adapter_sha256"] != expected_adapter
            or file_sha256(production / "holdout_retention" / "evaluation" / "assessment.json")
            != holdout["assessment_file_sha256"]
            or file_sha256(production / "benchmark" / "output" / "raw" / "predictions.jsonl")
            != benchmark["candidate_predictions_file_sha256"]
            or canonical_sha256(read_json_object(self.config.registry_root / "active_model.json"))
            != self.initial_active_sha256
        ):
            raise CycleContractError(
                "final evaluation or active-model identity drifted before decision"
            )
        chain: dict[str, Any] = {
            "schema_version": 1,
            "cycle_id": CYCLE_ID,
            "execution": cycle_execution_metadata(self.config),
            "config_sha256": self.config.sha256,
            "source": self.source,
            "state_sha256": state["state_sha256"],
            "baseline_stage_sha256": baseline["stage_evidence_sha256"],
            "compatibility_sha256": compatibility["compatibility_sha256"],
            "generation_sha256": generation["generation_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "corpus_sha256": corpus["corpus_sha256"],
            "training_sha256": training["training_sha256"],
            "development_retention_sha256": development["development_retention_sha256"],
            "holdout_retention_sha256": holdout["holdout_retention_sha256"],
            "benchmark_sha256": benchmark["benchmark_sha256"],
            "selected_adapter_sha256": expected_adapter,
            "passed": True,
        }
        chain["identity_chain_sha256"] = canonical_sha256(chain)
        return chain

    def _stage_decide(self, state: dict[str, Any]) -> StageOutcome:
        if state.get("decision") == "rejected":
            evidence = self._persist(
                "decide",
                {
                    "schema_version": 1,
                    "decision_id": "foundry-cycle1-promotion-gate-v1",
                    "decision": "rejected",
                    "reason": state.get("stop_reason"),
                    "conditions": {},
                    "benchmark_reached": (
                        cast(dict[str, Any], state["stages"])["benchmark"]["status"] == "completed"
                    ),
                },
            )
            return StageOutcome(
                True,
                evidence,
                terminal_decision="rejected",
                classification=str(state.get("stop_reason")),
            )
        development = _verified_payload(
            self.config.runtime_root / "production" / "development_retention" / "summary.json",
            "development_retention_sha256",
        )
        holdout = _verified_payload(
            self.config.runtime_root / "production" / "holdout_retention" / "summary.json",
            "holdout_retention_sha256",
        )
        benchmark = _verified_payload(
            self.config.runtime_root / "production" / "benchmark" / "summary.json",
            "benchmark_sha256",
        )
        training = _verified_payload(
            self.config.runtime_root / "production" / "training" / "summary.json",
            "training_sha256",
        )
        contract = self.config.section("benchmark")
        identity_failure_type: str | None = None
        try:
            identity_chain = self._cycle_identity_chain(state)
            identities_reconstruct = True
        except Exception as error:
            identities_reconstruct = False
            identity_failure_type = type(error).__name__
            identity_chain = {
                "schema_version": 1,
                "cycle_id": CYCLE_ID,
                "passed": False,
                "failure_type": identity_failure_type,
            }
            identity_chain["identity_chain_sha256"] = canonical_sha256(identity_chain)
        conditions = evaluate_promotion_gate(
            development=development,
            holdout=holdout,
            benchmark=benchmark,
            training=training,
            contract=contract,
            identities_reconstruct=identities_reconstruct,
        )
        promoted = all(conditions.values())
        failed = [name for name, passed in conditions.items() if not passed]
        evidence = self._persist(
            "decide",
            {
                "schema_version": 1,
                "decision_id": "foundry-cycle1-promotion-gate-v1",
                "conditions": conditions,
                "failed_conditions": failed,
                "candidate_correct": benchmark["correct"],
                "candidate_total": benchmark["total"],
                "identity_chain": identity_chain,
                "identity_failure_type": identity_failure_type,
                "decision": "promoted" if promoted else "rejected",
                "reason": None if promoted else "frozen_promotion_gate_failed:" + ",".join(failed),
            },
        )
        return StageOutcome(
            True,
            evidence,
            terminal_decision="promoted" if promoted else "rejected",
            classification=cast(str | None, evidence["reason"]),
        )

    def _stage_promote_or_reject(self, state: dict[str, Any]) -> StageOutcome:
        decision = str(state["decision"])
        active_path = self.config.registry_root / "active_model.json"
        if decision == "rejected":
            rejected_active = assert_active_preserved(active_path, self.initial_active_sha256)
            record: dict[str, Any] = {
                "schema_version": 1,
                "cycle_id": CYCLE_ID,
                "execution": cycle_execution_metadata(self.config),
                "cycle_state_sha256": state["state_sha256"],
                "decision": "rejected",
                "reason": state.get("stop_reason"),
                "active_model_before": "untouched-base",
                "active_model_after": "untouched-base",
                "active_model_record_sha256": canonical_sha256(rejected_active),
                "timestamp_utc": _utc_now(),
            }
            record["rejection_record_sha256"] = canonical_sha256(record)
            _write_json(
                self.config.runtime_root / "publication" / "rejection_record.json",
                record,
            )
            evidence = self._persist("promote_or_reject", record)
            return StageOutcome(True, evidence)

        checkpoint = self._selected_checkpoint()
        candidate, candidate_hash = self._selected_adapter_identity()
        logical_id = str(self.config.section("promotion")["logical_model_id"])
        destination = self.config.registry_root / logical_id
        copy_adapter_once(candidate, destination, candidate_hash)
        prior_active = read_json_object(active_path)
        promoted_active: dict[str, Any] = {
            "schema_version": 1,
            "registry_id": "foundry-active-model-registry-v1",
            "logical_model_id": logical_id,
            "base_model_id": self.config.section("model")["model_id"],
            "base_revision": self.config.section("model")["revision"],
            "adapter_sha256": candidate_hash,
            "checkpoint": checkpoint,
            "cycle_id": CYCLE_ID,
            "execution_id": self.config.execution_id,
        }
        try:
            temporary_active = active_path.with_suffix(".tmp")
            _write_json(temporary_active, promoted_active)
            os.replace(temporary_active, active_path)
            sanity_output = self.config.runtime_root / "promotion" / "registry_sanity"
            _runtime_command(
                config=self.config,
                operation="registry-sanity",
                output_directory=sanity_output,
                adapter_path=destination,
                expected_adapter_sha256=candidate_hash,
                log_label="promotion-registry-sanity",
            )
            sanity = _verified_payload(
                sanity_output / "summary.json",
                "sanity_sha256",
            )
            if (
                read_json_object(active_path) != promoted_active
                or sanity["adapter_resolved"] is not True
            ):
                raise RuntimeError("active registry did not resolve the promoted model")
        except Exception as error:
            rollback = active_path.with_suffix(".rollback.tmp")
            _write_json(rollback, prior_active)
            os.replace(rollback, active_path)
            if read_json_object(active_path) != prior_active:
                raise RuntimeError("active-model registry rollback failed") from error
            expected_destination = (self.config.registry_root / logical_id).resolve()
            if (
                destination.exists()
                and destination.resolve() == expected_destination
                and directory_sha256(destination) == candidate_hash
            ):
                shutil.rmtree(destination)
            raise
        decision_record = _verified_payload(
            self.evidence_root / "decide.json",
            "stage_evidence_sha256",
        )
        holdout = _verified_payload(
            self.evidence_root / "holdout_retention.json",
            "stage_evidence_sha256",
        )
        benchmark = _verified_payload(
            self.evidence_root / "benchmark.json",
            "stage_evidence_sha256",
        )
        record = {
            "schema_version": 1,
            "promotion_id": "foundry-cycle1-automatic-promotion-v1",
            "cycle_id": CYCLE_ID,
            "execution": cycle_execution_metadata(self.config),
            "cycle_state_sha256": state["state_sha256"],
            "previous_active_model": "untouched-base",
            "promoted_model": logical_id,
            "base_revision": self.config.section("model")["revision"],
            "adapter_sha256": candidate_hash,
            "checkpoint": checkpoint,
            "retention_decision_sha256": holdout["stage_evidence_sha256"],
            "benchmark_decision_sha256": benchmark["stage_evidence_sha256"],
            "promotion_gate_sha256": decision_record["stage_evidence_sha256"],
            "promotion_timestamp_utc": _utc_now(),
            "active_model_record_sha256": canonical_sha256(promoted_active),
            "registry_sanity_sha256": sanity["sanity_sha256"],
        }
        record["promotion_record_sha256"] = canonical_sha256(record)
        _write_json(
            self.config.runtime_root / "publication" / "promotion_record.json",
            record,
        )
        evidence = self._persist("promote_or_reject", record)
        return StageOutcome(True, evidence)

    def _stage_publish_trace(self, state: dict[str, Any]) -> StageOutcome:
        decision = str(state["decision"])
        result = self._build_publication(state)
        publication = self.config.runtime_root / "publication"
        _write_json(publication / "cycle_result.json", result)
        diagnosis = cast(dict[str, Any] | None, result["diagnosis"])
        weakness = (
            ", ".join(cast(list[str], diagnosis["selected_reasoning_categories"]))
            if diagnosis is not None
            else "not reached"
        )
        lines = [
            "Foundry Cycle 1",
            "",
            "Baseline: 521/814",
            f"Diagnosed weakness: {weakness}",
            "Optimization: verifier-filtered-best-of-8 SFT",
            f"Decision: {decision.upper()}",
            "Active model: "
            + str(
                read_json_object(self.config.registry_root / "active_model.json")[
                    "logical_model_id"
                ]
            ),
        ]
        _write_text(publication / "terminal_trace.txt", "\n".join(lines) + "\n")
        evidence = self._persist(
            "publish_trace",
            {
                "schema_version": 1,
                "cycle_result_sha256": result["cycle_result_sha256"],
                "terminal_trace_sha256": file_sha256(publication / "terminal_trace.txt"),
                "decision": decision,
                "project_status": result["project_status"],
            },
        )
        return StageOutcome(True, evidence)

    def _build_publication(self, state: dict[str, Any]) -> dict[str, Any]:
        def optional(name: str) -> dict[str, Any] | None:
            path = self.evidence_root / f"{name}.json"
            return _verified_payload(path, "stage_evidence_sha256") if path.is_file() else None

        selection = optional("verify_and_select_traces")
        corpus = optional("freeze_training_corpus")
        training = optional("train_candidate")
        development = optional("development_retention")
        holdout = optional("holdout_retention")
        benchmark = optional("benchmark")
        promotion = optional("promote_or_reject")
        result: dict[str, Any] = {
            "schema_version": 1,
            "cycle_id": CYCLE_ID,
            "execution": cycle_execution_metadata(self.config),
            "controller_id": CONTROLLER_ID,
            "decision": state["decision"],
            "stop_reason": state.get("stop_reason"),
            "project_status": (
                "COMPLETE"
                if state["decision"] == "promoted"
                else (
                    "CYCLE_1_REJECTED_FINAL"
                    if self.config.execution_id == RECOVERY_EXECUTION_ID
                    else "CYCLE_1_REJECTED"
                )
            ),
            "starting_commit": self.config.section("starting_state")["repository_commit"],
            "controller_source": self.source,
            "config_sha256": self.config.sha256,
            "model_revision": self.config.section("model")["revision"],
            "dataset_sha256": self.config.section("dataset")["identity_sha256"],
            "targeted_training_sha256": self.config.section("dataset")["targeted_training_sha256"],
            "targeted_validation_sha256": self.config.section("dataset")[
                "targeted_validation_sha256"
            ],
            "baseline_correct": 521,
            "generic_starting_correct": 517,
            "targeted_starting_correct": 519,
            "diagnosis": optional("diagnose"),
            "generation": optional("generate_candidates"),
            "selection": selection,
            "corpus": corpus,
            "compatibility": optional("compatibility_smoke"),
            "training": training,
            "development_retention": development,
            "selected_checkpoint": (
                development.get("selected_checkpoint") if development is not None else None
            ),
            "holdout_retention": holdout,
            "benchmark": benchmark,
            "promotion_gate": optional("decide"),
            "promotion_or_rejection": promotion,
            "active_model": read_json_object(self.config.registry_root / "active_model.json"),
            "controller_state_sha256": state["state_sha256"],
            "sealed_final_accessed": False,
            "second_cycle_run": False,
            "second_seed_run": False,
            "grpo_reopened": False,
        }
        result["cycle_result_sha256"] = canonical_sha256(result)
        return result

    def finalize(self, state: dict[str, Any]) -> None:
        path = self.config.runtime_root / "publication" / "cycle_result.json"
        if not path.is_file():
            return
        result = self._build_publication(state)
        _write_json(path, result)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


class CycleController:
    """Strict ordered controller with exact resume at completed boundaries."""

    def __init__(
        self,
        *,
        config: CycleConfig,
        store: StateStore,
        backend: StageBackend,
    ) -> None:
        self.config = config
        self.store = store
        self.backend = backend

    def run(self) -> dict[str, Any]:
        state = self.store.create_or_load()
        while (stage := self.store.next_stage(state)) is not None:
            self.store.begin(state, stage)
            try:
                outcome = self.backend.execute(stage, state)
            except Exception as error:
                self.store.fail(
                    state,
                    stage,
                    f"{stage}_exception:{type(error).__name__}",
                    {"error_type": type(error).__name__, "stage": stage},
                )
                self.store.skip_until_decision(state)
                if stage == "publish_trace":
                    raise
                state = self.store.load()
                continue
            if not outcome.passed:
                reason = outcome.classification or f"{stage}_failed"
                self.store.fail(state, stage, reason, outcome.evidence)
                self.store.skip_until_decision(state)
                state = self.store.load()
                continue
            self.store.complete(
                state,
                stage,
                outcome.evidence,
                classification=outcome.classification,
            )
            if outcome.terminal_decision is not None:
                self.store.set_decision(
                    state,
                    outcome.terminal_decision,
                    outcome.classification,
                )
            state = self.store.load()
        self.backend.finalize(state)
        return state


def run_cycle(
    config_path: Path,
    *,
    resume: bool,
    execution_id: str | None = None,
) -> dict[str, Any]:
    """Run the one autonomous production cycle."""

    if not resume:
        raise CycleContractError("Cycle 1 requires the explicit --resume contract")
    config = bind_cycle_execution(load_cycle_config(config_path), execution_id)
    source, interpreter_sha256, initial_active_sha256 = ensure_runtime_identity(config)
    compatibility = config.runtime_root / "compatibility" / "summary.json"
    if not compatibility.is_file():
        raise CycleContractError("production cycle requires duplicate compatibility evidence")
    comparison = _verified_payload(compatibility, "compatibility_sha256")
    if comparison["passed"] is not True:
        raise CycleContractError("production cycle is prohibited after compatibility failure")
    store = StateStore(
        path=config.runtime_root / "state.json",
        config=config,
        source=source,
        interpreter_sha256=interpreter_sha256,
    )
    backend = ProductionBackend(config, source, initial_active_sha256)
    return CycleController(config=config, store=store, backend=backend).run()


def cycle_status(cycle_id: str, *, execution_id: str | None = None) -> dict[str, Any]:
    if cycle_id != CYCLE_ID:
        raise CycleContractError("unknown cycle ID")
    resolved_execution = CYCLE_ID if execution_id is None else execution_id
    if resolved_execution not in {CYCLE_ID, RECOVERY_EXECUTION_ID}:
        raise CycleContractError("unknown Cycle 1 execution ID")
    root = (
        RECOVERY_RUNTIME_ROOT
        if resolved_execution == RECOVERY_EXECUTION_ID
        else Path(r"C:\Users\Admin\Projects\Foundry-cycle1-runtime")
    )
    state_path = root / "state.json"
    result_path = root / "publication" / "cycle_result.json"
    if result_path.is_file():
        result = read_json_object(result_path)
        selection = result.get("selection") or {}
        development = result.get("development_retention") or {}
        holdout = result.get("holdout_retention") or {}
        benchmark = result.get("benchmark") or {}
        return {
            "cycle_id": CYCLE_ID,
            "execution_id": resolved_execution,
            "baseline": "521/814",
            "diagnosed_weakness": (
                cast(dict[str, Any], result.get("diagnosis") or {}).get(
                    "selected_reasoning_categories"
                )
            ),
            "optimization": "verifier-filtered-best-of-8 SFT",
            "verifier_approved_prompt_coverage": (
                f"{selection.get('selected_prompts')}/{selection.get('expected_prompts')}"
            ),
            "selected_checkpoint": development.get("selected_checkpoint"),
            "development_retention": ("PASS" if development.get("passed") else "NOT PASSED"),
            "holdout_v2_retention": ("PASS" if holdout.get("passed") else "NOT REACHED OR FAILED"),
            "candidate_gsm1k": (
                f"{benchmark.get('correct')}/814"
                if benchmark.get("correct") is not None
                else "NOT RUN"
            ),
            "decision": str(result["decision"]).upper(),
            "active_model": cast(dict[str, Any], result["active_model"])["logical_model_id"],
            "project_status": result["project_status"],
        }
    if state_path.is_file():
        state = read_json_object(state_path)
        return {
            "cycle_id": CYCLE_ID,
            "execution_id": resolved_execution,
            "current_stage": state.get("current_stage"),
            "decision": state.get("decision"),
            "state_sha256": state.get("state_sha256"),
        }
    compatibility = root / "compatibility" / "summary.json"
    return {
        "cycle_id": CYCLE_ID,
        "execution_id": resolved_execution,
        "status": "compatibility_complete" if compatibility.is_file() else "not_started",
    }


def active_model(*, execution_id: str | None = None) -> dict[str, Any]:
    resolved_execution = CYCLE_ID if execution_id is None else execution_id
    if resolved_execution not in {CYCLE_ID, RECOVERY_EXECUTION_ID}:
        raise CycleContractError("unknown Cycle 1 execution ID")
    root = (
        RECOVERY_RUNTIME_ROOT
        if resolved_execution == RECOVERY_EXECUTION_ID
        else Path(r"C:\Users\Admin\Projects\Foundry-cycle1-runtime")
    )
    path = root / "model_registry" / "active_model.json"
    if not path.is_file():
        return {
            "logical_model_id": "untouched-base",
            "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "adapter_sha256": None,
        }
    return read_json_object(path)
