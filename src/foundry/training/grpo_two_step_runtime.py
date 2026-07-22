"""Exact two-update compatibility replay for the frozen verifier-GRPO G1 contract.

This runtime instruments the already-audited TRL trainer without replacing its
loss, backward, clipping, optimizer, or scheduler implementations.  Heavy GPU
dependencies remain deferred until a counted ``one-run`` invocation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import weakref
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

from foundry.training import grpo_replay_runtime as replay_runtime
from foundry.training.config import canonical_sha256
from foundry.training.grpo_compatibility import (
    CONTRACT_ID,
    TopPWarningOnlyGenerationContract,
    model_adapter_state,
)
from foundry.training.grpo_config import BASE_REVISION, load_grpo_config
from foundry.training.grpo_environment import (
    assert_idempotent_deterministic_initialization,
    make_environment_guarded_trainer,
    make_environment_validation_callback,
    transformers_determinism_source_evidence,
    validate_deterministic_process_environment,
)
from foundry.training.grpo_paths import (
    GrpoRuntimePaths,
    assert_artifact_path,
    assert_source_path,
    deterministic_process_contract,
    load_runtime_paths,
    validate_runtime_paths,
)
from foundry.training.grpo_reference import assert_only_lora_trainable
from foundry.training.grpo_replay_evidence import (
    CompatibilityStepEvidence,
    GenerationEvidence,
    build_compatibility_step_evidence,
    build_two_step_packet,
    capture_base_parameter_state,
    capture_generation_evidence,
    capture_gradient_state,
    capture_lora_state,
    capture_optimizer_state,
    capture_rng_state,
    capture_scheduler_state,
    compare_fresh_process_packets,
    write_replay_packet_new,
)
from foundry.training.grpo_runtime import (
    MAX_RESERVED_VRAM_BYTES,
    VerifierRewardCallback,
    _assert_offline_model_snapshot,
    _base_reference_hash,
    _completion_token_counter,
    _finite_history_metrics,
    _load_quantized_base,
    _peak_process_ram,
    _prepare_runtime,
    _repeat_row,
    _runtime_modules,
    assert_cuda_only_model,
    assert_dropout_disabled,
    assert_frozen_base_has_no_gradients,
    assert_frozen_grpo_arguments,
    frozen_grpo_argument_values,
    load_runtime_schedule,
    save_final_adapter,
    select_compatibility_groups,
    summarize_reward_records,
)
from foundry.training.grpo_schedule import COMPLETIONS_PER_GROUP
from foundry.training.grpo_trainer import make_truncation_aware_grpo_trainer
from foundry.training.qlora import directory_sha256

TWO_STEP_RUNTIME_ID = "foundry-verifier-grpo-two-step-replay-v1"
TWO_STEP_RUNTIME_SCHEMA_VERSION = 1
FRESH_PROCESS_RUNS = 2
OPTIMIZER_STEPS = 2
GENERATION_GROUPS = 3
TOTAL_COMPLETIONS = GENERATION_GROUPS * COMPLETIONS_PER_GROUP
FROZEN_VARIANT_ID = "G1"
FROZEN_EXECUTION_SHA256 = "d7023bf6705702a39dfe8d8718db264f6b2c0e2e211753145ad71e2368f4f4c0"
_PROCESS_INSTANCE_STARTED_NS = time.time_ns()
_PROCESS_INSTANCE_SHA256 = canonical_sha256(
    {
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "process_instance_started_ns": _PROCESS_INSTANCE_STARTED_NS,
        "sys_executable": str(Path(sys.executable).resolve()),
    }
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
MAX_PRE_RELOAD_ALLOCATED_VRAM_BYTES = 256 * 1024**2
MAX_PRE_RELOAD_RESERVED_VRAM_BYTES = 512 * 1024**2


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite two-step output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_strict(torch: Any, operation: str) -> None:
    if not replay_runtime._strict_determinism(torch):
        raise RuntimeError(f"{operation} did not run under strict deterministic enforcement")


def _require_frozen_group_ids(groups: Sequence[Any]) -> None:
    actual = tuple(str(group.group_id) for group in groups)
    if actual != replay_runtime.FROZEN_REPLAY_GROUP_IDS:
        raise ValueError(
            f"two-step compatibility groups differ from the frozen G1 generic groups: {actual}"
        )
    if tuple(str(group.source_kind) for group in groups) != (
        "synthetic",
        "synthetic",
        "base_replay",
    ):
        raise ValueError("two-step compatibility source composition differs")


def _scheduler_last_epoch(scheduler: Any) -> int:
    state_dict = getattr(scheduler, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("compatibility scheduler does not expose state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("compatibility scheduler state must be a mapping")
    value = state.get("last_epoch")
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("compatibility scheduler state lacks an integer last_epoch")
    return value


def _cuda_memory_snapshot(torch: Any) -> dict[str, int]:
    """Capture synchronized allocator state around trained-model release."""

    torch.cuda.synchronize(0)
    allocated = int(torch.cuda.memory_allocated(0))
    reserved = int(torch.cuda.memory_reserved(0))
    if allocated < 0 or reserved < allocated:
        raise RuntimeError("CUDA allocator reported an invalid memory snapshot")
    return {
        "allocated_vram_bytes": allocated,
        "reserved_vram_bytes": reserved,
    }


def _validate_pre_reload_release(
    *,
    model_reference: weakref.ReferenceType[Any],
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, object]:
    """Require the trained policy to be gone before a fresh base is loaded."""

    if model_reference() is not None:
        raise RuntimeError("trained policy remains strongly referenced before offline reload")
    before_allocated = int(before["allocated_vram_bytes"])
    before_reserved = int(before["reserved_vram_bytes"])
    after_allocated = int(after["allocated_vram_bytes"])
    after_reserved = int(after["reserved_vram_bytes"])
    if after_allocated > MAX_PRE_RELOAD_ALLOCATED_VRAM_BYTES:
        raise RuntimeError("pre-reload allocated VRAM exceeds the trained-model release gate")
    if after_reserved > MAX_PRE_RELOAD_RESERVED_VRAM_BYTES:
        raise RuntimeError("pre-reload reserved VRAM exceeds the trained-model release gate")
    return {
        "trained_model_collected": True,
        "pre_release_allocated_vram_bytes": before_allocated,
        "pre_release_reserved_vram_bytes": before_reserved,
        "post_release_allocated_vram_bytes": after_allocated,
        "post_release_reserved_vram_bytes": after_reserved,
        "allocated_vram_drop_bytes": max(0, before_allocated - after_allocated),
        "reserved_vram_drop_bytes": max(0, before_reserved - after_reserved),
        "max_pre_reload_allocated_vram_bytes": MAX_PRE_RELOAD_ALLOCATED_VRAM_BYTES,
        "max_pre_reload_reserved_vram_bytes": MAX_PRE_RELOAD_RESERVED_VRAM_BYTES,
        "pre_reload_memory_gate_passed": True,
    }


class TwoStepEvidenceRecorder:
    """Collect exact evidence at stock Trainer lifecycle boundaries."""

    def __init__(
        self,
        *,
        torch_module: Any,
        numpy_random: Any,
        tokenizer: Any,
        reward_callback: VerifierRewardCallback,
        warning_contract: TopPWarningOnlyGenerationContract,
        groups: Sequence[Any],
    ) -> None:
        if len(groups) != GENERATION_GROUPS:
            raise ValueError("two-step recorder requires exactly three generation groups")
        _require_frozen_group_ids(groups)
        self.torch = torch_module
        self.numpy_random = numpy_random
        self.tokenizer = tokenizer
        self.reward_callback = reward_callback
        self.warning_contract = warning_contract
        self.groups = tuple(groups)
        self.steps: list[CompatibilityStepEvidence] = []
        self.replay_generation: GenerationEvidence | None = None
        self._generation_calls = 0
        self._record_start = 0
        self._warning_start = 0
        self._pending_result: Mapping[str, Any] | None = None
        self._pending_records: list[Any] = []
        self._pending_warning: Any | None = None
        self._pending_step: dict[str, Any] | None = None
        self._policy_capture_active = False
        self._captured_policy: Any | None = None

    def start_generation(self) -> None:
        _require_strict(self.torch, "generation entry")
        if self._pending_result is not None or self._generation_calls >= GENERATION_GROUPS:
            raise RuntimeError("generation evidence lifecycle is invalid")
        self._record_start = len(self.reward_callback.records)
        self._warning_start = len(self.warning_contract.call_records())

    def finish_generation(self, result: Mapping[str, Any]) -> None:
        _require_strict(self.torch, "generation exit")
        records = list(self.reward_callback.records[self._record_start :])
        warnings = list(self.warning_contract.call_records()[self._warning_start :])
        if len(records) != COMPLETIONS_PER_GROUP:
            raise RuntimeError("compatibility generation did not produce four reward records")
        if len(warnings) != 1:
            raise RuntimeError("compatibility generation did not produce one warning record")
        expected = self.groups[self._generation_calls]
        if {str(record.group_id) for record in records} != {str(expected.group_id)}:
            raise RuntimeError("reward records differ from the scheduled compatibility group")
        self._pending_result = result
        self._pending_records = records
        self._pending_warning = warnings[0]
        self._generation_calls += 1

    def begin_policy_capture(self) -> None:
        if self._policy_capture_active or self._captured_policy is not None:
            raise RuntimeError("nested or stale policy-logprob capture is prohibited")
        self._policy_capture_active = True

    def capture_policy(self, value: Any) -> None:
        if self._policy_capture_active:
            if self._captured_policy is not None:
                raise RuntimeError("stock GRPO loss requested policy log probabilities twice")
            self._captured_policy = value

    def end_policy_capture(self) -> Any:
        self._policy_capture_active = False
        if self._captured_policy is None:
            raise RuntimeError("stock GRPO loss produced no policy log probabilities")
        value = self._captured_policy
        self._captured_policy = None
        return value

    def abort_policy_capture(self) -> None:
        self._policy_capture_active = False
        self._captured_policy = None

    def _generation_evidence(
        self,
        *,
        reference: Any,
        policy: Any,
        per_token_kl: Any,
    ) -> GenerationEvidence:
        if self._pending_result is None or self._pending_warning is None:
            raise RuntimeError("generation tensors are unavailable for exact evidence")
        group = self.groups[self._generation_calls - 1]
        completion_ids = self._pending_result["completion_ids"]
        warning = self._pending_warning
        value = capture_generation_evidence(
            group_id=str(group.group_id),
            source_kind=cast(Any, group.source_kind),
            prompt_sha256=str(group.prompt_sha256),
            generated_token_ids=completion_ids,
            decoded_completions=[str(record.completion) for record in self._pending_records],
            completion_token_lengths=replay_runtime._token_lengths(
                completion_ids, int(self.tokenizer.eos_token_id)
            ),
            truncation_flags=[
                bool(record.reward.generation_truncated) for record in self._pending_records
            ],
            reward_components=[record.reward.as_dict() for record in self._pending_records],
            rng_before_sha256=str(warning.rng_before_sha256),
            rng_after_sha256=str(warning.rng_after_sha256),
            warning_sha256s=warning.warning_sha256s,
            reference_logprobs=reference,
            policy_logprobs=policy,
            per_token_kl=per_token_kl,
        )
        self._pending_result = None
        self._pending_records = []
        self._pending_warning = None
        return value

    def record_loss(
        self,
        *,
        loss_tensor: Any,
        reference: Any,
        policy: Any,
        per_token_kl: Any,
        mean_kl_tensor: Any,
    ) -> None:
        if self._pending_step is None or self._generation_calls not in {1, 2}:
            raise RuntimeError("loss evidence is outside an update step")
        generation = self._generation_evidence(
            reference=reference,
            policy=policy,
            per_token_kl=per_token_kl,
        )
        self._pending_step.update(
            {
                "generation": generation,
                "loss_tensor": loss_tensor.detach(),
                "loss": float(loss_tensor.detach().float().item()),
                "mean_kl_tensor": mean_kl_tensor.detach(),
                "mean_kl": float(mean_kl_tensor.detach().float().item()),
            }
        )

    def after_backward(self, model: Any) -> None:
        _require_strict(self.torch, "backward")
        if self._pending_step is None or "loss_tensor" not in self._pending_step:
            raise RuntimeError("backward evidence has no matching stock loss")
        gradients = capture_gradient_state(model)
        if gradients["frozen_gradient_count"] != 0 or gradients["lora_gradient_count"] == 0:
            raise RuntimeError("backward gradients violate the LoRA-only contract")
        self._pending_step["gradients_after_backward"] = gradients

    def step_begin(self, *, state: Any, model: Any, optimizer: Any, scheduler: Any) -> None:
        _require_strict(self.torch, "step begin")
        expected_step = len(self.steps) + 1
        if expected_step not in {1, 2} or int(state.global_step) != expected_step - 1:
            raise RuntimeError("stock Trainer step ordering differs")
        if self._pending_step is not None:
            raise RuntimeError("prior compatibility step did not finalize")
        self._pending_step = {
            "step": expected_step,
            "rng_before": capture_rng_state(self.torch, numpy_random=self.numpy_random),
            "lora_before": capture_lora_state(model),
            "optimizer_before": capture_optimizer_state(optimizer),
            "scheduler_before": capture_scheduler_state(scheduler),
            "strict_mode_evidence": {},
        }

    def pre_optimizer(self, *, model: Any) -> None:
        _require_strict(self.torch, "gradient clipping and pre-optimizer hook")
        if self._pending_step is None or "gradients_after_backward" not in self._pending_step:
            raise RuntimeError("gradient clipping preceded backward evidence")
        gradients = capture_gradient_state(model)
        if gradients["frozen_gradient_count"] != 0 or gradients["lora_gradient_count"] == 0:
            raise RuntimeError("clipped gradients violate the LoRA-only contract")
        self._pending_step["gradients_after_clipping"] = gradients
        self._pending_step["strict_mode_evidence"]["after_backward"] = True
        self._pending_step["strict_mode_evidence"]["before_optimizer"] = True

    def post_optimizer(self, *, model: Any, optimizer: Any) -> None:
        _require_strict(self.torch, "optimizer step")
        if self._pending_step is None or "gradients_after_clipping" not in self._pending_step:
            raise RuntimeError("optimizer ran without clipped-gradient evidence")
        self._pending_step["optimizer_after"] = capture_optimizer_state(optimizer)
        self._pending_step["lora_after"] = capture_lora_state(model)
        self._pending_step["strict_mode_evidence"]["after_optimizer"] = True

    def step_end(self, *, state: Any, scheduler: Any) -> None:
        _require_strict(self.torch, "scheduler step")
        pending = self._pending_step
        if pending is None or int(state.global_step) != int(pending["step"]):
            raise RuntimeError("scheduler completion differs from the stock Trainer step")
        pending["scheduler_after"] = capture_scheduler_state(scheduler)
        pending["rng_after"] = capture_rng_state(self.torch, numpy_random=self.numpy_random)
        pending["strict_mode_evidence"]["after_scheduler"] = True
        required = {
            "generation",
            "loss",
            "loss_tensor",
            "mean_kl",
            "mean_kl_tensor",
            "rng_before",
            "rng_after",
            "lora_before",
            "lora_after",
            "gradients_after_backward",
            "gradients_after_clipping",
            "optimizer_before",
            "optimizer_after",
            "scheduler_before",
            "scheduler_after",
            "strict_mode_evidence",
        }
        missing = sorted(required - set(pending))
        if missing:
            raise RuntimeError(f"compatibility step evidence is incomplete: {missing}")
        step = build_compatibility_step_evidence(
            step=int(pending["step"]),
            generation=pending["generation"],
            loss=float(pending["loss"]),
            loss_tensor=pending["loss_tensor"],
            mean_kl=float(pending["mean_kl"]),
            mean_kl_tensor=pending["mean_kl_tensor"],
            rng_before=pending["rng_before"],
            rng_after=pending["rng_after"],
            lora_before=pending["lora_before"],
            lora_after=pending["lora_after"],
            gradients_after_backward=pending["gradients_after_backward"],
            gradients_after_clipping=pending["gradients_after_clipping"],
            optimizer_before=pending["optimizer_before"],
            optimizer_after=pending["optimizer_after"],
            scheduler_before=pending["scheduler_before"],
            scheduler_after=pending["scheduler_after"],
            strict_mode_evidence=pending["strict_mode_evidence"],
        )
        self.steps.append(step)
        self._pending_step = None

    def finalize_replay(self, *, reference: Any, policy: Any, per_token_kl: Any) -> None:
        if self._generation_calls != 3 or self.replay_generation is not None:
            raise RuntimeError("separate replay generation ordering differs")
        self.replay_generation = self._generation_evidence(
            reference=reference,
            policy=policy,
            per_token_kl=per_token_kl,
        )

    def assert_complete(self) -> None:
        if (
            len(self.steps) != OPTIMIZER_STEPS
            or self.replay_generation is None
            or self._generation_calls != GENERATION_GROUPS
            or self._pending_step is not None
            or self._pending_result is not None
        ):
            raise RuntimeError("two-step exact evidence is incomplete")


def make_two_step_capture_callback(
    base_callback_class: type[Any], recorder: TwoStepEvidenceRecorder
) -> object:
    """Create a stock Trainer callback that observes clipping/optimizer boundaries."""

    class TwoStepCaptureCallback(base_callback_class):  # type: ignore[misc]
        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args
            recorder.step_begin(
                state=state,
                model=kwargs["model"],
                optimizer=kwargs["optimizer"],
                scheduler=kwargs["lr_scheduler"],
            )
            return control

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state
            recorder.pre_optimizer(model=kwargs["model"])
            return control

        def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state
            recorder.post_optimizer(model=kwargs["model"], optimizer=kwargs["optimizer"])
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args
            recorder.step_end(state=state, scheduler=kwargs["lr_scheduler"])
            return control

    TwoStepCaptureCallback.__name__ = "TwoStepCaptureCallback"
    return TwoStepCaptureCallback()


def make_two_step_evidence_trainer(
    base_trainer_class: type[Any], recorder: TwoStepEvidenceRecorder
) -> type[Any]:
    """Instrument stock GRPO math while leaving every operation delegated to TRL."""

    class ExactTwoStepGRPOTrainer(base_trainer_class):  # type: ignore[misc]
        def _generate_and_score_completions(self, inputs: Any) -> Any:
            recorder.start_generation()
            result = super()._generate_and_score_completions(inputs)
            if not isinstance(result, Mapping):
                raise TypeError("stock GRPO generation result must be a mapping")
            recorder.finish_generation(cast(Mapping[str, Any], result))
            return result

        def _get_per_token_logps(self, *args: Any, **kwargs: Any) -> Any:
            value = super()._get_per_token_logps(*args, **kwargs)
            recorder.capture_policy(value)
            return value

        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            recorder.begin_policy_capture()
            try:
                loss = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
                policy = recorder.end_policy_capture()
            except BaseException:
                recorder.abort_policy_capture()
                raise
            reference = inputs.get("ref_per_token_logps")
            completion_mask = inputs.get("completion_mask")
            if reference is None or completion_mask is None:
                raise RuntimeError("stock G1 loss lacks reference log probabilities or mask")
            delta = reference - policy
            per_token_kl = self._foundry_torch.exp(delta) - delta - 1
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            if not bool(self._foundry_torch.isfinite(loss).all().item()) or not bool(
                self._foundry_torch.isfinite(per_token_kl).all().item()
            ):
                raise RuntimeError("compatibility loss or KL is not finite")
            recorder.record_loss(
                loss_tensor=loss,
                reference=reference,
                policy=policy,
                per_token_kl=per_token_kl,
                mean_kl_tensor=mean_kl,
            )
            return loss

        def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
            loss = super().training_step(model, inputs, num_items_in_batch)
            recorder.after_backward(model)
            return loss

    ExactTwoStepGRPOTrainer.__name__ = "ExactTwoStepGRPOTrainer"
    return ExactTwoStepGRPOTrainer


def _single_two_step_run(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    trainer_output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    assert_source_path(runtime_paths, config_path, "two-step configuration")
    assert_source_path(runtime_paths, manifest_path, "two-step schedule manifest")
    assert_artifact_path(runtime_paths, packet_path, "two-step schedule packet")
    assert_artifact_path(runtime_paths, trainer_output_dir, "two-step trainer state")
    model_path = runtime_paths.model_snapshot_root
    config = load_grpo_config(config_path)
    if config.base_model.revision != BASE_REVISION:
        raise ValueError("base revision differs from the frozen Qwen checkpoint")
    if config.execution_sha256(FROZEN_VARIANT_ID) != FROZEN_EXECUTION_SHA256:
        raise ValueError("G1 execution contract differs")
    external_process = replay_runtime._external_process_evidence(
        config.grpo.seed,
        runtime_paths,
        expected_entry_cublas_workspace_config=(
            replay_runtime.FROZEN_PROCESS_START_CUBLAS_WORKSPACE_CONFIG
        ),
    )
    reward_contract = replay_runtime._assert_frozen_reward_contract()
    _assert_offline_model_snapshot(model_path, config)
    if trainer_output_dir.exists():
        raise FileExistsError("two-step trainer path must be unused")
    schedule = load_runtime_schedule(
        packet_path,
        manifest_path,
        expected_arm=replay_runtime.FROZEN_REPLAY_ARM,
    )
    update_groups, replay_group = select_compatibility_groups(schedule)
    groups = (*update_groups, replay_group)
    replay_runtime._assert_frozen_schedule_binding(
        schedule,
        arm=replay_runtime.FROZEN_REPLAY_ARM,
        group_ids=[group.group_id for group in groups],
    )
    _require_frozen_group_ids(groups)

    deterministic_stages = [
        validate_deterministic_process_environment(runtime_paths, "before_transformers_import")
    ]
    modules = _runtime_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    trl = modules["trl"]
    datasets = modules["datasets"]
    peft = modules["peft"]
    psutil = modules["psutil"]
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_transformers_import",
            torch_module=torch,
        )
    )
    transformers_source = transformers_determinism_source_evidence(transformers)
    before_full_determinism = deterministic_stages[-1]
    numpy = replay_runtime._seed_everything(modules, config.grpo.seed)
    after_full_determinism = validate_deterministic_process_environment(
        runtime_paths,
        "after_transformers_full_determinism",
        torch_module=torch,
        require_strict=True,
    )
    deterministic_stages.append(after_full_determinism)
    initialization_idempotence = assert_idempotent_deterministic_initialization(
        before_full_determinism, after_full_determinism
    )
    runtime_environment = replay_runtime._runtime_environment_evidence(runtime_paths, numpy)
    cuda = replay_runtime._validate_frozen_cuda(torch)
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_cuda_initialization",
            torch_module=torch,
            require_strict=True,
        )
    )
    replay_runtime._prepare_cuda_replay(torch)
    process = psutil.Process()
    started = time.perf_counter()
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "before_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    model, tokenizer, lora_config, model_load_seconds = _prepare_runtime(
        config, model_path, modules
    )
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    reward_callback = VerifierRewardCallback(
        groups,
        completion_token_counter=_completion_token_counter(tokenizer),
    )
    argument_values = frozen_grpo_argument_values(
        config,
        variant_id=FROZEN_VARIANT_ID,
        output_dir=trainer_output_dir,
        mode="compatibility",
    )
    arguments = trl.GRPOConfig(**argument_values)
    after_arguments = validate_deterministic_process_environment(
        runtime_paths,
        "after_grpo_config_full_determinism",
        torch_module=torch,
        require_strict=True,
    )
    deterministic_stages.append(after_arguments)
    assert_idempotent_deterministic_initialization(after_full_determinism, after_arguments)
    assert_frozen_grpo_arguments(
        arguments,
        config,
        variant_id=FROZEN_VARIANT_ID,
        output_dir=trainer_output_dir,
        mode="compatibility",
    )
    warning_contract = TopPWarningOnlyGenerationContract(
        torch_module=torch,
        generation_owner=transformers.GenerationMixin,
        top_p_call=transformers.generation.logits_process.TopPLogitsWarper.__call__,
    )
    recorder = TwoStepEvidenceRecorder(
        torch_module=torch,
        numpy_random=numpy.random,
        tokenizer=tokenizer,
        reward_callback=reward_callback,
        warning_contract=warning_contract,
        groups=groups,
    )
    boundary_counts: dict[str, int] = {}
    boundary_environment_sha256s: set[str] = set()

    def validate_boundary(stage: str) -> None:
        evidence = validate_deterministic_process_environment(
            runtime_paths,
            stage,
            torch_module=torch,
            require_strict=True,
        )
        boundary_counts[stage] = boundary_counts.get(stage, 0) + 1
        boundary_environment_sha256s.add(str(evidence["environment_sha256"]))

    audited_type = make_truncation_aware_grpo_trainer(
        trl.GRPOTrainer,
        generation_scope_factory=partial(warning_contract.install, "generation"),
    )
    guarded_type = make_environment_guarded_trainer(audited_type, validate_boundary)
    trainer_type = make_two_step_evidence_trainer(guarded_type, recorder)
    trainer_type._foundry_torch = torch
    capture_callback = make_two_step_capture_callback(transformers.TrainerCallback, recorder)
    environment_callback = make_environment_validation_callback(
        transformers.TrainerCallback, validate_boundary
    )
    train_dataset = datasets.Dataset.from_list([group.policy_row() for group in update_groups])
    trainer = trainer_type(
        model=model,
        reward_funcs=reward_callback,
        args=arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[capture_callback, environment_callback],
        peft_config=lora_config,
    )
    after_trainer = validate_deterministic_process_environment(
        runtime_paths,
        "after_grpo_trainer_construction",
        torch_module=torch,
        require_strict=True,
    )
    deterministic_stages.append(after_trainer)
    assert_idempotent_deterministic_initialization(after_full_determinism, after_trainer)
    if trainer.ref_model is not None:
        raise RuntimeError("two-step smoke unexpectedly created a second reference model")
    trained_model_reference = weakref.ref(trainer.model)
    warning_contract.bind_state_probe(partial(model_adapter_state, trainer.model))
    trainability = assert_only_lora_trainable(trainer.model)
    assert_cuda_only_model(trainer.model)
    dropout_module_count = assert_dropout_disabled(trainer.model, torch)
    _require_strict(torch, "pre-training validation")
    base_before = capture_base_parameter_state(trainer.model)
    base_output_before = _base_reference_hash(trainer.model, tokenizer, groups[0], torch)
    initial_lora = capture_lora_state(trainer.model)

    training_started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - training_started
    _require_strict(torch, "post-training validation")
    if int(trainer.state.global_step) != OPTIMIZER_STEPS:
        raise RuntimeError("two-step optimizer count differs")
    if _scheduler_last_epoch(trainer.lr_scheduler) != OPTIMIZER_STEPS:
        raise RuntimeError("two-step scheduler did not advance exactly twice")
    if len(recorder.steps) != OPTIMIZER_STEPS:
        raise RuntimeError("two-step recorder did not observe both optimizer updates")
    assert_cuda_only_model(trainer.model)
    assert_frozen_base_has_no_gradients(trainer.model)

    result = cast(
        Mapping[str, Any],
        trainer._generate_and_score_completions(_repeat_row(replay_group)),
    )
    _require_strict(torch, "separate replay scoring")
    reference, policy, per_token_kl = replay_runtime._policy_and_kl(trainer, result, torch)
    recorder.finalize_replay(
        reference=reference,
        policy=policy,
        per_token_kl=per_token_kl,
    )
    recorder.assert_complete()
    reward_summary = summarize_reward_records(
        reward_callback.records,
        groups,
        require_nonzero_variance=True,
    )
    metrics = _finite_history_metrics(
        cast(Sequence[Mapping[str, object]], trainer.state.log_history)
    )
    warning_evidence = warning_contract.evidence()
    if warning_evidence["generation_calls"] != GENERATION_GROUPS:
        raise RuntimeError("warning-only generation call count differs")
    _require_strict(torch, "adapter save")
    final_lora = capture_lora_state(trainer.model)
    base_after = capture_base_parameter_state(trainer.model)
    base_output_after = _base_reference_hash(trainer.model, tokenizer, groups[0], torch)
    if (
        base_after["base_parameter_state_sha256"] != base_before["base_parameter_state_sha256"]
        or base_output_after != base_output_before
    ):
        raise RuntimeError("frozen base changed during the two-step smoke")
    final_optimizer = capture_optimizer_state(trainer.optimizer)
    final_scheduler = capture_scheduler_state(trainer.lr_scheduler)
    if final_optimizer != recorder.steps[-1].optimizer_after:
        raise RuntimeError("final optimizer state differs from the step-2 hook")
    if final_scheduler != recorder.steps[-1].scheduler_after:
        raise RuntimeError("final scheduler state differs from the step-2 hook")
    adapter_path, adapter_artifact_hash = save_final_adapter(
        trainer_output_dir, trainer.model, tokenizer
    )
    adapter_directory_hash = directory_sha256(adapter_path)

    step_evidence = tuple(recorder.steps)
    replay_generation_evidence = cast(GenerationEvidence, recorder.replay_generation)
    pre_release_memory = _cuda_memory_snapshot(torch)
    warning_contract.release_state_probe()
    del result, reference, policy, per_token_kl
    del capture_callback, environment_callback, trainer_type, guarded_type, audited_type
    del trainer
    del model
    del recorder, warning_contract
    gc.collect()
    torch.cuda.empty_cache()
    post_release_memory = _cuda_memory_snapshot(torch)
    pre_reload_release = _validate_pre_reload_release(
        model_reference=trained_model_reference,
        before=pre_release_memory,
        after=post_release_memory,
    )
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "before_adapter_reload_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    reloaded_base_model, reloaded_tokenizer, reload_base_seconds = _load_quantized_base(
        model_path, config, modules
    )
    reloaded_base_model = peft.prepare_model_for_kbit_training(
        reloaded_base_model,
        use_gradient_checkpointing=(config.memory_and_reproducibility.gradient_checkpointing),
    )
    reload_started = time.perf_counter()
    reloaded = peft.PeftModel.from_pretrained(
        reloaded_base_model,
        str(adapter_path),
        local_files_only=True,
        is_trainable=False,
    )
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_adapter_reload_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    adapter_reload_seconds = time.perf_counter() - reload_started
    assert_cuda_only_model(reloaded)
    if any(parameter.requires_grad for parameter in reloaded.parameters()):
        raise RuntimeError("offline compatibility reload left trainable parameters")
    reloaded_lora = capture_lora_state(reloaded)
    reloaded_base = capture_base_parameter_state(reloaded)
    reloaded_base_output = _base_reference_hash(reloaded, reloaded_tokenizer, groups[0], torch)
    if reloaded_lora["lora_tensor_state_sha256"] != final_lora["lora_tensor_state_sha256"]:
        raise RuntimeError("offline adapter reload changed exact LoRA tensor bytes")
    if (
        reloaded_base["base_parameter_state_sha256"] != base_before["base_parameter_state_sha256"]
        or reloaded_base_output != base_output_before
    ):
        raise RuntimeError("adapter-disabled reload did not restore the exact base")

    environment_sha256, environment_count, excluded_environment_count = (
        replay_runtime._filtered_environment_sha256(runtime_paths)
    )
    run_contract: dict[str, object] = {
        "runtime_id": TWO_STEP_RUNTIME_ID,
        "runtime_source_sha256": _source_sha256(Path(__file__)),
        "replay_evidence_source_sha256": _source_sha256(
            Path(__file__).with_name("grpo_replay_evidence.py")
        ),
        "compatibility_contract_id": CONTRACT_ID,
        "config_sha256": config.config_sha256,
        "execution_sha256": config.execution_sha256(FROZEN_VARIANT_ID),
        "variant_id": FROZEN_VARIANT_ID,
        "arm": replay_runtime.FROZEN_REPLAY_ARM,
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "group_ids": [group.group_id for group in groups],
        "source_kinds": [group.source_kind for group in groups],
        "optimizer_steps": OPTIMIZER_STEPS,
        "generation_only_groups": 1,
        "completion_count": TOTAL_COMPLETIONS,
        "seed": config.grpo.seed,
        "beta": config.variant(FROZEN_VARIANT_ID).beta,
        "num_generations": config.grpo.num_generations,
        "max_completion_length": config.grpo.max_completion_length,
        "temperature": config.grpo.temperature,
        "top_p": config.grpo.top_p,
        "top_k": config.grpo.top_k,
        "base_revision": config.base_model.revision,
        "base_output_sha256": base_output_before,
        "reward_contract": reward_contract,
        "reward_summary": reward_summary,
        "finite_history_metric_names": sorted(metrics),
        "warning_contract": warning_evidence,
        "external_process_contract": external_process,
        "transformers_determinism_source": transformers_source,
        "deterministic_initialization_idempotence": initialization_idempotence,
        "deterministic_process_environment_sha256": (runtime_paths.process_environment_sha256),
        "runtime_paths": runtime_paths.evidence(),
        "runtime_environment": runtime_environment,
        "environment_variable_sha256": environment_sha256,
        "environment_variable_count": environment_count,
        "secret_environment_variables_excluded": excluded_environment_count,
        "only_lora_trainable": True,
        "trainable_parameters": trainability.trainable_parameters,
        "total_parameters": trainability.total_parameters,
        "second_reference_model_created": False,
        "cpu_offload": False,
        "trained_model_released_before_reload": True,
        "pre_reload_memory_gate_passed": True,
        "dropout_disabled": True,
        "dropout_module_count": dropout_module_count,
        **cuda,
    }
    packet = build_two_step_packet(
        run_contract=run_contract,
        steps=step_evidence,
        replay_generation=replay_generation_evidence,
        initial_lora=initial_lora,
        final_lora=final_lora,
        reloaded_lora=reloaded_lora,
        base_before=base_before,
        base_after=base_after,
        reloaded_base=reloaded_base,
        final_optimizer=final_optimizer,
        final_scheduler=final_scheduler,
        adapter_artifact_sha256=adapter_artifact_hash,
        adapter_directory_sha256=adapter_directory_hash,
    )
    torch.cuda.synchronize(0)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    if peak_reserved >= MAX_RESERVED_VRAM_BYTES:
        raise RuntimeError(f"two-step peak reserved VRAM exceeds the 9.6 GiB gate: {peak_reserved}")
    resource: dict[str, object] = {
        "model_load_seconds": model_load_seconds,
        "training_seconds": training_seconds,
        "reload_base_seconds": reload_base_seconds,
        "adapter_reload_seconds": adapter_reload_seconds,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "reserved_vram_gate_bytes": MAX_RESERVED_VRAM_BYTES,
        "reserved_vram_gate_passed": True,
        "peak_process_ram_bytes": _peak_process_ram(process),
        "adapter_directory_bytes": sum(
            path.stat().st_size for path in adapter_path.rglob("*") if path.is_file()
        ),
        "trained_model_release": pre_reload_release,
        "deterministic_environment_stages": deterministic_stages,
        "boundary_validation_counts": dict(sorted(boundary_counts.items())),
        "boundary_environment_sha256s": sorted(boundary_environment_sha256s),
        "environment_mutation_observed": False,
    }
    del reloaded
    del reloaded_base_model
    cleanup = replay_runtime._cleanup_cuda_replay(torch)
    resource.update(cleanup)
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "process_result_publication",
            torch_module=torch,
            require_strict=True,
        )
    )
    return packet, resource


def _run_one_fresh_process_impl(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    raw_packet_path: Path,
    trainer_output_dir: Path,
    metadata_path: Path,
) -> dict[str, object]:
    """Run one exact two-step smoke and write ignored packet/metadata evidence."""

    validate_runtime_paths(runtime_paths)
    assert_artifact_path(runtime_paths, raw_packet_path, "two-step replay packet")
    assert_artifact_path(runtime_paths, trainer_output_dir, "two-step trainer state")
    assert_artifact_path(runtime_paths, metadata_path, "two-step replay metadata")
    if raw_packet_path.exists() or metadata_path.exists():
        raise FileExistsError("two-step packet and metadata paths must start unused")
    packet, resource = _single_two_step_run(
        runtime_paths=runtime_paths,
        config_path=config_path,
        packet_path=packet_path,
        manifest_path=manifest_path,
        trainer_output_dir=trainer_output_dir,
    )
    packet_hash = write_replay_packet_new(raw_packet_path, packet, kind="two_step_compatibility")
    process_contract = deterministic_process_contract(runtime_paths)
    metadata: dict[str, object] = {
        "schema_version": TWO_STEP_RUNTIME_SCHEMA_VERSION,
        "runtime_id": TWO_STEP_RUNTIME_ID,
        "packet_sha256": packet_hash,
        "process_instance_sha256": _PROCESS_INSTANCE_SHA256,
        "process_command_sha256": process_contract.process_command_sha256,
        "deterministic_process_contract_sha256": process_contract.contract_sha256,
        "runtime_path_contract_sha256": runtime_paths.contract_sha256,
        "process_environment_sha256": runtime_paths.process_environment_sha256,
        "process_command_template_sha256": runtime_paths.process_command_template_sha256,
        "resource_measurement": resource,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    _write_json_new(metadata_path, metadata)
    validate_runtime_paths(runtime_paths)
    return metadata


def run_one_fresh_process(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    raw_packet_path: Path,
    trainer_output_dir: Path,
    metadata_path: Path,
) -> dict[str, object]:
    """Validate immutable roots on every two-step exit path."""

    validate_runtime_paths(runtime_paths)
    try:
        return _run_one_fresh_process_impl(
            runtime_paths=runtime_paths,
            config_path=config_path,
            packet_path=packet_path,
            manifest_path=manifest_path,
            raw_packet_path=raw_packet_path,
            trainer_output_dir=trainer_output_dir,
            metadata_path=metadata_path,
        )
    finally:
        validate_runtime_paths(runtime_paths)


def _load_metadata(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load two-step metadata {path}: {error}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("two-step metadata must be a string-keyed object")
    row = cast(dict[str, object], value)
    declared = row.get("metadata_sha256")
    payload = {key: item for key, item in row.items() if key != "metadata_sha256"}
    if declared != canonical_sha256(payload):
        raise ValueError("two-step metadata self-hash differs")
    if row.get("runtime_id") != TWO_STEP_RUNTIME_ID:
        raise ValueError("two-step metadata runtime ID differs")
    return row


def combine_fresh_process_runs(
    *,
    runtime_paths: GrpoRuntimePaths,
    packet_paths: Sequence[Path],
    metadata_paths: Sequence[Path],
    summary_path: Path,
) -> dict[str, object]:
    """Compare exactly two fresh-process runs and write a tracked content-free summary."""

    if len(packet_paths) != FRESH_PROCESS_RUNS or len(metadata_paths) != FRESH_PROCESS_RUNS:
        raise ValueError("two-step replay requires exactly two packets and metadata files")
    validate_runtime_paths(runtime_paths)
    for path in (*packet_paths, *metadata_paths, summary_path):
        assert_artifact_path(runtime_paths, path, "two-step replay artifact")
    if len({path.resolve() for path in packet_paths}) != FRESH_PROCESS_RUNS:
        raise ValueError("two-step packet paths must be distinct")
    if len({path.resolve() for path in metadata_paths}) != FRESH_PROCESS_RUNS:
        raise ValueError("two-step metadata paths must be distinct")
    common_hash = compare_fresh_process_packets(
        packet_paths, expected_kind="two_step_compatibility"
    )
    metadata = [_load_metadata(path) for path in metadata_paths]
    if any(row.get("packet_sha256") != common_hash for row in metadata):
        raise ValueError("two-step metadata packet hash differs")
    expected_runtime_fields = {
        "runtime_path_contract_sha256": runtime_paths.contract_sha256,
        "process_environment_sha256": runtime_paths.process_environment_sha256,
        "process_command_template_sha256": runtime_paths.process_command_template_sha256,
    }
    if any(
        row.get(key) != value for row in metadata for key, value in expected_runtime_fields.items()
    ):
        raise RuntimeError("two-step runtime path, environment, or command template differs")
    process_identities = [str(row.get("process_instance_sha256")) for row in metadata]
    if len(set(process_identities)) != FRESH_PROCESS_RUNS or any(
        len(value) != 64 or any(character not in _SHA256_CHARACTERS for character in value)
        for value in process_identities
    ):
        raise ValueError("two-step runs do not prove two distinct process identities")
    process_commands = [str(row.get("process_command_sha256")) for row in metadata]
    process_contracts = [str(row.get("deterministic_process_contract_sha256")) for row in metadata]
    if any(
        len(value) != 64 or any(character not in _SHA256_CHARACTERS for character in value)
        for value in (*process_commands, *process_contracts)
    ):
        raise ValueError("two-step process command or environment contract hash is invalid")
    if len(set(process_commands)) != FRESH_PROCESS_RUNS or len(set(process_contracts)) != (
        FRESH_PROCESS_RUNS
    ):
        raise RuntimeError("two-step runs do not bind two distinct exact process commands")
    resources = [cast(Mapping[str, object], row["resource_measurement"]) for row in metadata]
    if any(item.get("reserved_vram_gate_passed") is not True for item in resources):
        raise RuntimeError("one two-step run failed the reserved-VRAM gate")
    release_records: list[Mapping[str, object]] = []
    for item in resources:
        release = item.get("trained_model_release")
        if not isinstance(release, Mapping):
            raise RuntimeError("one two-step run lacks trained-model release evidence")
        release_records.append(cast(Mapping[str, object], release))
    if any(item.get("pre_reload_memory_gate_passed") is not True for item in release_records):
        raise RuntimeError("one two-step run failed the trained-model release gate")
    summary: dict[str, object] = {
        "schema_version": TWO_STEP_RUNTIME_SCHEMA_VERSION,
        "runtime_id": TWO_STEP_RUNTIME_ID,
        "replay_kind": "fresh_process_two_step_compatibility",
        "processes": FRESH_PROCESS_RUNS,
        "optimizer_steps_per_process": OPTIMIZER_STEPS,
        "groups_per_process": GENERATION_GROUPS,
        "completions_per_process": TOTAL_COMPLETIONS,
        "common_packet_sha256": common_hash,
        "packet_file_sha256s": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in packet_paths
        ],
        "process_identity_sha256s": process_identities,
        "process_command_sha256s": process_commands,
        "deterministic_process_contract_sha256s": process_contracts,
        "runtime_paths": runtime_paths.evidence(),
        "process_metadata_sha256s": [row["metadata_sha256"] for row in metadata],
        "resource_measurements": resources,
        "exact_replay_passed": True,
        "strict_backward_optimizer_passed": True,
        "adapter_tensor_replay_passed": True,
        "base_restoration_passed": True,
        "trained_model_release_passed": True,
        "warning_only_scope_passed": True,
        "reserved_vram_gate_passed": True,
        "prompts_completions_answers_or_tensor_values_in_summary": False,
        "compatibility_gate_passed": True,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json_new(summary_path, summary)
    validate_runtime_paths(runtime_paths)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    one = subparsers.add_parser("one-run")
    one.add_argument("--runtime-paths", type=Path, required=True)
    one.add_argument("--config", type=Path, required=True)
    one.add_argument("--packet", type=Path, required=True)
    one.add_argument("--manifest", type=Path, required=True)
    one.add_argument("--raw-packet", type=Path, required=True)
    one.add_argument("--trainer-output", type=Path, required=True)
    one.add_argument("--metadata", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--runtime-paths", type=Path, required=True)
    combine.add_argument("--packets", type=Path, nargs=FRESH_PROCESS_RUNS, required=True)
    combine.add_argument("--metadata", type=Path, nargs=FRESH_PROCESS_RUNS, required=True)
    combine.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime_paths = load_runtime_paths(args.runtime_paths)
    if args.command == "one-run":
        result = run_one_fresh_process(
            runtime_paths=runtime_paths,
            config_path=args.config,
            packet_path=args.packet,
            manifest_path=args.manifest,
            raw_packet_path=args.raw_packet,
            trainer_output_dir=args.trainer_output,
            metadata_path=args.metadata,
        )
    else:
        result = combine_fresh_process_runs(
            runtime_paths=runtime_paths,
            packet_paths=args.packets,
            metadata_paths=args.metadata,
            summary_path=args.summary,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
