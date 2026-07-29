"""Fresh-process diagnostic reproduction for the first Milestone 14A GRPO group.

The diagnostic performs generation and three separate backward projections but
never advances an optimizer or scheduler and never saves an adapter.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import l3_grpo_runtime as runtime
from foundry.phase2.l3_grpo_contract import (
    INTERPRETER_SHA256,
    MODEL_REVISION,
    STARTING_ADAPTER_SHA256,
)
from foundry.phase2.l3_grpo_reference import (
    POLICY_ADAPTER_NAME,
    REFERENCE_ADAPTER_NAME,
    SharedStartingPolicyReference,
    active_adapter_name,
    assert_policy_reference_identity,
    capture_adapter_state,
    set_policy_active,
)
from foundry.phase2.l3_grpo_schedule import COMPLETIONS_PER_GROUP
from foundry.phase2.l3_grpo_zero_gradient import (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    classification_contract,
    classify_group,
    gradient_projection,
    objective_components,
    populated_gradient_projection,
    reward_projection,
    run_deterministic_fixtures,
    tensor_graph_evidence,
)
from foundry.training.config import canonical_sha256
from foundry.training.grpo_compatibility import (
    TopPWarningOnlyGenerationContract,
    model_adapter_state,
)
from foundry.training.grpo_replay_evidence import (
    capture_base_parameter_state,
    capture_optimizer_state,
)
from foundry.training.grpo_runtime import assert_cuda_only_model, assert_dropout_disabled
from foundry.training.grpo_trainer import make_truncation_aware_grpo_trainer
from foundry.training.qlora import directory_sha256, file_sha256

DIAGNOSTIC_ID = "foundry-l3-grpo-first-group-zero-gradient-diagnostic-v1"
FREEZE_FILE = "milestone14a_r1_zero_gradient_freeze.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"diagnostic JSON must be an object: {path}")
    return cast(dict[str, Any], value)


def _write_replace(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"diagnostic evidence already exists: {path}")
    _write_replace(path, value)


def _verify_self_hash(value: Mapping[str, object], key: str) -> None:
    projected = dict(value)
    expected = projected.pop(key, None)
    if expected != canonical_sha256(projected):
        raise ValueError(f"{key} does not reconstruct")


def _parameter_partitions(
    model: Any,
) -> tuple[
    tuple[tuple[str, Any], ...],
    tuple[tuple[str, Any], ...],
    tuple[tuple[str, Any], ...],
]:
    policy: list[tuple[str, Any]] = []
    reference: list[tuple[str, Any]] = []
    base: list[tuple[str, Any]] = []
    for raw_name, parameter in sorted(model.named_parameters(), key=lambda item: str(item[0])):
        name = str(raw_name)
        if "lora_" in name and f".{POLICY_ADAPTER_NAME}." in name:
            policy.append((name, parameter))
        elif "lora_" in name and f".{REFERENCE_ADAPTER_NAME}." in name:
            reference.append((name, parameter))
        elif "lora_" not in name:
            base.append((name, parameter))
    if len(policy) != 112 or len(reference) != 112:
        raise RuntimeError("diagnostic adapter parameter inventory differs")
    if any(not bool(parameter.requires_grad) for _, parameter in policy):
        raise RuntimeError("diagnostic policy parameter is frozen")
    if any(bool(parameter.requires_grad) for _, parameter in (*reference, *base)):
        raise RuntimeError("diagnostic reference or base parameter is trainable")
    return tuple(policy), tuple(reference), tuple(base)


def _install_adapter_trace(model: Any) -> tuple[list[dict[str, object]], Any]:
    trace: list[dict[str, object]] = []
    original = model.set_adapter

    def audited(adapter_name: str, *args: Any, **kwargs: Any) -> Any:
        before = active_adapter_name(model)
        result = original(adapter_name, *args, **kwargs)
        after = active_adapter_name(model)
        trace.append(
            {
                "index": len(trace) + 1,
                "requested": adapter_name,
                "before": before,
                "after": after,
            }
        )
        return result

    model.set_adapter = audited
    return trace, original


def _tensor_list(value: Any) -> list[Any]:
    result = value.detach().cpu().tolist()
    if not isinstance(result, list):
        raise TypeError("diagnostic tensor did not produce a list")
    return result


def _finite(torch: Any, value: Any) -> bool:
    return bool(torch.isfinite(value).all().item())


def _policy_logprobs(trainer: Any, result: Mapping[str, Any], torch: Any) -> Any:
    if active_adapter_name(trainer.model) != POLICY_ADAPTER_NAME:
        raise RuntimeError("policy adapter is not active before the policy forward")
    prompt_ids = result["prompt_ids"]
    prompt_mask = result["prompt_mask"]
    completion_ids = result["completion_ids"]
    completion_mask = result["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    return trainer._get_per_token_logps(
        trainer.model,
        input_ids,
        attention_mask,
        completion_ids.size(1),
    )


def _evidence_core(
    *,
    root: Path,
    output_dir: Path,
    partial_path: Path,
) -> dict[str, object]:
    tracked = root / "results/phase2_vetted_corpus"
    freeze = _read(tracked / FREEZE_FILE)
    _verify_self_hash(freeze, "freeze_sha256")
    source_manifest = freeze.get("prediagnostic_source")
    if not isinstance(source_manifest, Mapping):
        raise TypeError("prediagnostic source manifest is absent")
    _verify_self_hash(source_manifest, "source_manifest_sha256")
    source_rows = source_manifest.get("files")
    if not isinstance(source_rows, list) or not source_rows:
        raise TypeError("prediagnostic source rows are absent")
    for row in source_rows:
        if not isinstance(row, Mapping):
            raise TypeError("prediagnostic source row is invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or file_sha256(root / relative) != row.get("sha256"):
            raise ValueError("prediagnostic source changed after its freeze")
    contract = classification_contract()
    if freeze.get("classification_contract") != contract:
        raise ValueError("diagnostic classification contract differs from the frozen packet")
    modules, launch = runtime._runtime_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    trl = modules["trl"]
    datasets = modules["datasets"]
    psutil = modules["psutil"]
    fixtures = run_deterministic_fixtures(torch)
    if freeze.get("fixture_contract") != fixtures:
        raise ValueError("diagnostic fixtures differ from the frozen packet")
    progress: dict[str, object] = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "stage": "validated_freeze",
        "classification_contract_sha256": contract["classification_contract_sha256"],
        "fixture_sha256": fixtures["fixture_sha256"],
    }
    _write_replace(partial_path, progress)

    packet_path = (
        root / "results/raw/phase2_vetted_corpus/milestone14a/schedules/generic_prompt_packet.json"
    )
    manifest_path = tracked / "milestone14a_generic_schedule.json"
    experiment_path = tracked / "milestone14a_experiment_contract.json"
    starting_adapter = (
        root
        / "results/raw/phase2_vetted_corpus/milestone13e/full/generic/training"
        / "checkpoint-64/adapter"
    )
    schedule = runtime.load_schedule(packet_path, manifest_path, "generic")
    experiment = runtime._validate_experiment_contract(
        experiment_path,
        schedule=schedule,
        arm="generic",
    )
    if directory_sha256(starting_adapter) != STARTING_ADAPTER_SHA256["generic"]:
        raise ValueError("diagnostic starting adapter differs")
    group = next(item for item in schedule.groups if item.source_kind == "task")
    runtime._strict(torch, "diagnostic before model load")
    process = psutil.Process()
    started = time.perf_counter()
    model_path = (
        root
        / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
        / f"snapshots/{MODEL_REVISION}"
    )
    model, tokenizer, initial_identity, _ = runtime._load_dual_adapter_model(
        model_path=model_path,
        starting_adapter=starting_adapter,
        expected_starting_sha256=STARTING_ADAPTER_SHA256["generic"],
        modules=modules,
    )
    runtime._strict(torch, "diagnostic after model load")
    base_before = capture_base_parameter_state(model)
    calibration = runtime._reference_perturbation_calibration(model, tokenizer, group, torch)
    initial_policy = capture_adapter_state(model, POLICY_ADAPTER_NAME)
    initial_reference = capture_adapter_state(model, REFERENCE_ADAPTER_NAME)
    progress.update(
        {
            "stage": "model_loaded_and_calibrated",
            "initial_identity_sha256": initial_identity["identity_sha256"],
            "controlled_positive_kl": calibration["controlled_positive_per_token_kl"],
        }
    )
    _write_replace(partial_path, progress)

    reward_callback = runtime.VerifierRewardCallback(
        (group,),
        completion_token_counter=runtime._completion_token_counter(tokenizer),
    )
    arguments = runtime._trainer_arguments(trl, output_dir=output_dir, max_steps=1)
    warning_contract = TopPWarningOnlyGenerationContract(
        torch_module=torch,
        generation_owner=transformers.GenerationMixin,
        top_p_call=transformers.generation.logits_process.TopPLogitsWarper.__call__,
    )
    audited_trainer = make_truncation_aware_grpo_trainer(
        trl.GRPOTrainer,
        generation_scope_factory=partial(warning_contract.install, "generation"),
    )
    dataset = datasets.Dataset.from_list([group.policy_row()])
    trainer = audited_trainer(
        model=model,
        reward_funcs=reward_callback,
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[],
        peft_config=None,
    )
    reference_proxy = SharedStartingPolicyReference(trainer.model, torch)
    trainer.ref_model = reference_proxy
    warning_contract.bind_state_probe(partial(model_adapter_state, trainer.model))
    set_policy_active(trainer.model)
    identity_at_step_start = assert_policy_reference_identity(
        trainer.model,
        require_policy_trainable=True,
    )
    assert_cuda_only_model(trainer.model)
    dropout_count = assert_dropout_disabled(trainer.model, torch)
    trainer.create_optimizer()
    ownership = runtime._optimizer_ownership(trainer)
    policy_parameters, reference_parameters, base_parameters = _parameter_partitions(trainer.model)
    adapter_trace, original_set_adapter = _install_adapter_trace(trainer.model)

    runtime._strict(torch, "diagnostic generation entry")
    record_start = len(reward_callback.records)
    warning_start = len(warning_contract.call_records())
    generation_active_adapter = active_adapter_name(trainer.model)
    rows = [copy.deepcopy(group.policy_row()) for _ in range(COMPLETIONS_PER_GROUP)]
    result = trainer._generate_and_score_completions(rows)
    if not isinstance(result, Mapping):
        raise TypeError("diagnostic generation result is not a mapping")
    result = cast(Mapping[str, Any], result)
    records = list(reward_callback.records[record_start:])
    warnings = list(warning_contract.call_records()[warning_start:])
    if len(records) != COMPLETIONS_PER_GROUP or len(warnings) != 1:
        raise RuntimeError("diagnostic generation accounting differs")
    rewards = [float(record.reward.total) for record in records]
    reward_evidence = reward_projection(torch, rewards)
    advantages = result["advantages"]
    if _tensor_list(advantages) != reward_evidence["advantages"]:
        raise RuntimeError("stock TRL advantages differ from the frozen projection")
    completion_mask = result["completion_mask"]
    valid_counts = [int(value) for value in completion_mask.sum(dim=1).detach().cpu().tolist()]
    if any(value <= 0 for value in valid_counts):
        raise RuntimeError("diagnostic completion mask is empty")
    progress.update(
        {
            "stage": "generation_and_rewards_persisted",
            "generated_token_ids": _tensor_list(result["completion_ids"]),
            "completion_hashes": [
                __import__("hashlib").sha256(record.completion.encode("utf-8")).hexdigest()
                for record in records
            ],
            "completion_lengths": runtime._token_lengths(
                result["completion_ids"],
                int(tokenizer.eos_token_id),
            ),
            "reward_vector": rewards,
            "reward_components": [record.reward.as_dict() for record in records],
            "reward_projection": reward_evidence,
            "advantages": _tensor_list(advantages),
            "valid_completion_token_counts": valid_counts,
            "truncation_mask": [bool(record.reward.generation_truncated) for record in records],
            "warning": warnings[0].as_dict(),
        }
    )
    _write_replace(partial_path, progress)

    policy_logprobs = _policy_logprobs(trainer, result, torch)
    reference_logprobs = result["ref_per_token_logps"]
    components = objective_components(
        torch,
        policy_logprobs=policy_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=advantages,
        completion_mask=completion_mask,
    )
    stock_loss = trainer.compute_loss(trainer.model, result)
    if not bool(torch.equal(stock_loss.detach(), components.combined.detach())):
        raise RuntimeError("diagnostic objective differs from stock TRL loss")
    graph_evidence = {
        "policy_objective": tensor_graph_evidence(components.policy),
        "kl_objective": tensor_graph_evidence(components.kl),
        "combined_objective": tensor_graph_evidence(components.combined),
    }
    objective_values = {
        "policy_objective": float(components.policy.detach().float().item()),
        "kl_objective": float(components.kl.detach().float().item()),
        "combined_objective": float(components.combined.detach().float().item()),
        "stock_loss": float(stock_loss.detach().float().item()),
    }
    progress.update(
        {
            "stage": "objectives_persisted",
            "policy_token_logprobs": _tensor_list(policy_logprobs),
            "reference_token_logprobs": _tensor_list(reference_logprobs),
            "policy_reference_kl": _tensor_list(components.per_token_kl),
            "objective_values": objective_values,
            "objective_graph": graph_evidence,
        }
    )
    _write_replace(partial_path, progress)

    policy_gradient = gradient_projection(
        torch,
        objective=components.policy,
        named_parameters=policy_parameters,
        retain_graph=True,
    )
    progress.update({"stage": "policy_gradient_persisted", "policy_gradient": policy_gradient})
    _write_replace(partial_path, progress)
    kl_gradient = gradient_projection(
        torch,
        objective=components.kl,
        named_parameters=policy_parameters,
        retain_graph=True,
    )
    progress.update({"stage": "kl_gradient_persisted", "kl_gradient": kl_gradient})
    _write_replace(partial_path, progress)
    combined_gradient = gradient_projection(
        torch,
        objective=components.combined,
        named_parameters=policy_parameters,
        retain_graph=False,
    )
    progress.update(
        {"stage": "combined_projection_persisted", "combined_gradient": combined_gradient}
    )
    _write_replace(partial_path, progress)

    trainer.model.zero_grad(set_to_none=True)
    populated_policy_logprobs = _policy_logprobs(trainer, result, torch)
    populated_components = objective_components(
        torch,
        policy_logprobs=populated_policy_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=advantages,
        completion_mask=completion_mask,
    )
    populated_components.combined.backward()
    populated = populated_gradient_projection(
        torch,
        named_policy_parameters=policy_parameters,
        named_reference_parameters=reference_parameters,
        named_base_parameters=base_parameters,
    )
    classification_input: dict[str, object] = {
        "rewards": reward_evidence["rewards"],
        "reward_variance": reward_evidence["reward_variance"],
        "advantages": reward_evidence["advantages"],
        "valid_completion_token_counts": valid_counts,
        "policy_logprobs_finite": _finite(torch, policy_logprobs),
        "reference_logprobs_finite": _finite(torch, reference_logprobs),
        "kl_finite": _finite(torch, components.per_token_kl),
        "adapters_identical_at_step_start": identity_at_step_start["byte_identical"],
        "controlled_live_policy_fixture_passed": True,
        "requires_grad_policy_tensor_count": len(policy_parameters),
        "optimizer_owned_tensor_count": ownership["optimizer_parameter_tensors"],
        "base_gradient_count": populated["base_gradient_count"],
        "reference_gradient_count": populated["reference_gradient_count"],
        "policy_gradient": policy_gradient,
        "kl_gradient": kl_gradient,
        "combined_gradient": combined_gradient,
    }
    classification = classify_group(classification_input)
    progress.update(
        {
            "stage": "classification_persisted",
            "populated_combined_gradient": populated,
            "classification_input": classification_input,
            "classification": classification,
        }
    )
    _write_replace(partial_path, progress)

    trainer.model.zero_grad(set_to_none=True)
    trainer.model.set_adapter = original_set_adapter
    set_policy_active(trainer.model)
    final_policy = capture_adapter_state(trainer.model, POLICY_ADAPTER_NAME)
    final_reference = capture_adapter_state(trainer.model, REFERENCE_ADAPTER_NAME)
    base_after = capture_base_parameter_state(trainer.model)
    if (
        final_policy["normalized_tensor_state_sha256"]
        != initial_policy["normalized_tensor_state_sha256"]
        or final_reference["normalized_tensor_state_sha256"]
        != initial_reference["normalized_tensor_state_sha256"]
        or base_after["base_parameter_state_sha256"] != base_before["base_parameter_state_sha256"]
    ):
        raise RuntimeError("diagnostic mutated policy, reference, or base parameters")
    reference_runtime = reference_proxy.evidence()
    if reference_runtime["call_count"] != 1:
        raise RuntimeError("diagnostic reference call accounting differs")
    if classification != EXPECTED_ZERO_ADVANTAGE_NOOP:
        progress.update(
            {
                "stage": "terminal_non_noop_classification",
                "classification": classification,
            }
        )
        _write_replace(partial_path, progress)

    generation = runtime._capture_l3_generation_evidence(
        group=group,
        generated_token_ids=result["completion_ids"],
        decoded_completions=[record.completion for record in records],
        completion_token_lengths=runtime._token_lengths(
            result["completion_ids"], int(tokenizer.eos_token_id)
        ),
        truncation_flags=[record.reward.generation_truncated for record in records],
        reward_components=[record.reward.as_dict() for record in records],
        rng_before_sha256=str(warnings[0].rng_before_sha256),
        rng_after_sha256=str(warnings[0].rng_after_sha256),
        warning_sha256s=warnings[0].warning_sha256s,
        reference_logprobs=reference_logprobs,
        policy_logprobs=policy_logprobs,
        per_token_kl=components.per_token_kl,
    )
    exact: dict[str, object] = {
        "schema_version": 1,
        "diagnostic_id": DIAGNOSTIC_ID,
        "classification_contract_sha256": contract["classification_contract_sha256"],
        "fixture_sha256": fixtures["fixture_sha256"],
        "experiment_contract_sha256": experiment["experiment_contract_sha256"],
        "generic_starting_adapter_sha256": STARTING_ADAPTER_SHA256["generic"],
        "generic_schedule_manifest_sha256": schedule.manifest_sha256,
        "group_id": group.group_id,
        "group_source_kind": group.source_kind,
        "group_prompt_sha256": group.prompt_sha256,
        "generation_active_adapter": generation_active_adapter,
        "active_adapter_before_policy_forward": POLICY_ADAPTER_NAME,
        "active_adapter_switch_trace": adapter_trace,
        "active_adapter_after_reference_forward": active_adapter_name(trainer.model),
        "generated_token_ids": _tensor_list(result["completion_ids"]),
        "generation_evidence": generation.as_dict(),
        "reward_vector": rewards,
        "reward_projection": reward_evidence,
        "advantages": _tensor_list(advantages),
        "normalized_advantages": _tensor_list(advantages),
        "valid_completion_token_counts": valid_counts,
        "completion_mask": _tensor_list(completion_mask),
        "truncation_mask": [bool(record.reward.generation_truncated) for record in records],
        "policy_token_logprobs": _tensor_list(policy_logprobs),
        "reference_token_logprobs": _tensor_list(reference_logprobs),
        "policy_reference_kl": _tensor_list(components.per_token_kl),
        "objective_values": objective_values,
        "objective_graph": graph_evidence,
        "policy_gradient": policy_gradient,
        "kl_gradient": kl_gradient,
        "combined_gradient": combined_gradient,
        "populated_combined_gradient": populated,
        "optimizer_ownership": ownership,
        "optimizer_state_without_step": capture_optimizer_state(trainer.optimizer),
        "optimizer_steps": 0,
        "scheduler_created": False,
        "scheduler_advancements": 0,
        "initial_policy_state_sha256": initial_policy["normalized_tensor_state_sha256"],
        "final_policy_state_sha256": final_policy["normalized_tensor_state_sha256"],
        "initial_reference_state_sha256": initial_reference["normalized_tensor_state_sha256"],
        "final_reference_state_sha256": final_reference["normalized_tensor_state_sha256"],
        "base_state_before_sha256": base_before["base_parameter_state_sha256"],
        "base_state_after_sha256": base_after["base_parameter_state_sha256"],
        "reference_runtime": reference_runtime,
        "dropout_modules_disabled": dropout_count,
        "warning_evidence": warnings[0].as_dict(),
        "classification_input": classification_input,
        "classification": classification,
        "adapter_or_checkpoint_saved": False,
        "sealed_content_use": 0,
    }
    exact["diagnostic_evidence_sha256"] = canonical_sha256(exact)
    runtime_seconds = time.perf_counter() - started
    resources: dict[str, object] = {
        "schema_version": 1,
        "diagnostic_id": DIAGNOSTIC_ID,
        "diagnostic_evidence_sha256": exact["diagnostic_evidence_sha256"],
        "runtime_seconds": runtime_seconds,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(0)),
        "physical_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "peak_process_rss_bytes": int(process.memory_info().rss),
        "generated_completion_tokens": sum(valid_counts),
        "evidence_disk_bytes_before_resource_file": 0,
        "launch_preimport_sha256": launch["preimport"]["preimport_evidence_sha256"],  # type: ignore[index]
        "launch_postimport_sha256": launch["postimport"]["postimport_evidence_sha256"],  # type: ignore[index]
    }
    resources["resource_evidence_sha256"] = canonical_sha256(resources)
    del stock_loss, components, populated_components, policy_logprobs, populated_policy_logprobs
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {"exact": exact, "resources": resources}


def run(*, root: Path, output_dir: Path) -> dict[str, object]:
    """Run one no-update diagnostic and persist exact and resource evidence."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"diagnostic output directory already exists: {output_dir}")
    if file_sha256(root / ".venv-training/Scripts/python.exe") != INTERPRETER_SHA256:
        raise ValueError("diagnostic training interpreter differs")
    partial_path = output_dir / "partial_evidence.json"
    try:
        result = _evidence_core(root=root, output_dir=output_dir, partial_path=partial_path)
    except BaseException as error:
        failure: dict[str, object]
        if partial_path.exists():
            failure = _read(partial_path)
        else:
            failure = {"diagnostic_id": DIAGNOSTIC_ID, "stage": "before_partial_evidence"}
        failure.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        failure["failure_sha256"] = canonical_sha256(failure)
        _write_replace(partial_path, failure)
        raise
    exact = cast(dict[str, object], result["exact"])
    resources = cast(dict[str, object], result["resources"])
    _write_new(output_dir / "evidence.json", exact)
    resources["evidence_disk_bytes_before_resource_file"] = sum(
        path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
    )
    resources.pop("resource_evidence_sha256")
    resources["resource_evidence_sha256"] = canonical_sha256(resources)
    _write_new(output_dir / "resources.json", resources)
    summary = {
        "diagnostic_evidence_sha256": exact["diagnostic_evidence_sha256"],
        "classification": exact["classification"],
        "reward_vector": exact["reward_vector"],
        "reward_variance": cast(dict[str, object], exact["reward_projection"])["reward_variance"],
        "advantages": exact["advantages"],
        "valid_completion_token_counts": exact["valid_completion_token_counts"],
        "policy_gradient_global_norm": cast(dict[str, object], exact["policy_gradient"])[
            "global_norm"
        ],
        "kl_gradient_global_norm": cast(dict[str, object], exact["kl_gradient"])["global_norm"],
        "combined_gradient_global_norm": cast(dict[str, object], exact["combined_gradient"])[
            "global_norm"
        ],
        "optimizer_steps": 0,
        "scheduler_advancements": 0,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(root=args.root, output_dir=args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
