"""Fresh-process replay and optimizer-free gradient qualification for L3 GRPO."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
import weakref
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

from foundry.phase2.l3_grpo_advantage_equivalence import (
    verify_advantage_equivalence_contract,
)
from foundry.phase2.l3_grpo_contract import (
    INTERPRETER_SHA256,
    MODEL_REVISION,
    STARTING_ADAPTER_SHA256,
)
from foundry.phase2.l3_grpo_reference import (
    POLICY_ADAPTER_NAME,
    REFERENCE_ADAPTER_NAME,
    SharedStartingPolicyReference,
    assert_policy_reference_identity,
    capture_adapter_state,
    set_policy_active,
)
from foundry.phase2.l3_grpo_runtime import (
    RuntimeGroup,
    VerifierRewardCallback,
    _completion_token_counter,
    _load_dual_adapter_model,
    _runtime_modules,
    _strict,
    _trainer_arguments,
    load_schedule,
)
from foundry.phase2.l3_grpo_signal_audit import (
    ARMS,
    COMPLETIONS_PER_GROUP,
    GROUPS_PER_ARM,
)
from foundry.phase2.l3_grpo_signal_continuity import (
    compare_fresh_group_to_prior,
    verify_prior_diagnostic_manifest,
)
from foundry.phase2.l3_grpo_signal_qualification import verify_qualification_contract
from foundry.phase2.l3_grpo_signal_runtime import (
    _group_record,
    _validate_contract,
)
from foundry.phase2.l3_grpo_zero_gradient import (
    gradient_projection,
    objective_components,
    populated_gradient_projection,
    tensor_graph_evidence,
)
from foundry.phase2.l3_grpo_zero_gradient_diagnostic import (
    _parameter_partitions,
    _policy_logprobs,
)
from foundry.training.config import canonical_sha256
from foundry.training.grpo_compatibility import (
    TopPWarningOnlyGenerationContract,
    model_adapter_state,
)
from foundry.training.grpo_replay_evidence import capture_base_parameter_state
from foundry.training.grpo_runtime import (
    _peak_process_ram,
    assert_cuda_only_model,
    assert_dropout_disabled,
)
from foundry.training.grpo_trainer import make_truncation_aware_grpo_trainer
from foundry.training.qlora import directory_sha256, file_sha256

Arm = Literal["generic", "targeted"]
RUNTIME_ID = "foundry-l3-grpo-signal-qualification-runtime-v1"


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify(value: Mapping[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite qualification evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_replace(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _selected_prefix(
    groups: Sequence[RuntimeGroup],
    selected: Mapping[str, Any],
) -> tuple[RuntimeGroup, ...]:
    position = selected.get("task_schedule_position")
    if isinstance(position, bool) or not isinstance(position, int) or position <= 0:
        raise ValueError("selected task schedule position differs")
    prefix = tuple(groups[:position])
    if (
        len(prefix) != position
        or prefix[-1].group_id != selected.get("task_group_id")
        or prefix[-1].prompt_sha256 != selected.get("task_prompt_sha256")
        or prefix[-1].source_kind != "task"
    ):
        raise ValueError("selected task does not reconstruct from the frozen schedule")
    return prefix


def _attach_continuity(
    *,
    arm: Arm,
    group: RuntimeGroup,
    group_record: dict[str, object],
    prior_manifest: Mapping[str, Any],
) -> None:
    if arm == "generic":
        continuity = compare_fresh_group_to_prior(
            prior_manifest=prior_manifest,
            fresh_group=group_record,
        )
    else:
        continuity = {
            "schema_version": 1,
            "comparison_id": "foundry-l3-grpo-pre-correction-continuity-v1",
            "schedule_position": group.position,
            "status": "not_applicable_to_targeted_arm",
            "passed": True,
            "failure_classification": None,
        }
        continuity["continuity_comparison_sha256"] = canonical_sha256(continuity)
    if continuity.get("passed") is not True:
        raise RuntimeError("scientific_replay_drift")
    group_record["prior_partial_continuity"] = continuity
    group_record["group_record_sha256"] = canonical_sha256(group_record)


def _audit_rows(
    *,
    root: Path,
    arm: Arm,
    qualification: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    path = (
        root
        / "results/raw/phase2_vetted_corpus/milestone14b_r1/signal_audit"
        / arm
        / "raw_evidence.json"
    )
    raw = _read(path)
    _verify(raw, "raw_audit_sha256")
    bound = cast(Mapping[str, Any], cast(Mapping[str, Any], qualification["audit_evidence"])[arm])
    if raw.get("raw_audit_sha256") != bound.get("raw_audit_sha256") or file_sha256(
        path
    ) != bound.get("raw_evidence_file_sha256"):
        raise ValueError("corrected audit evidence differs from the qualification freeze")
    rows = cast(list[dict[str, Any]], raw.get("groups"))
    result = {cast(int, row["schedule_position"]): row for row in rows}
    if len(rows) != 32 or len(result) != 32:
        raise ValueError("corrected audit group inventory differs")
    return result


def _finite_positive(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _gradient_gate(
    *,
    task_group: Mapping[str, Any],
    policy_gradient: Mapping[str, Any],
    combined_gradient: Mapping[str, Any],
    populated: Mapping[str, Any],
    objective_graph: Mapping[str, Mapping[str, Any]],
    stock_loss_equal: bool,
) -> None:
    advantages = cast(list[object], task_group.get("canonical_cuda_advantages"))
    valid_counts = cast(list[object], task_group.get("valid_completion_token_counts"))
    if (
        not _finite_positive(task_group.get("reward_variance"))
        or not advantages
        or not any(
            not isinstance(value, bool) and isinstance(value, int | float) and float(value) != 0.0
            for value in advantages
        )
        or not valid_counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in valid_counts
        )
        or task_group.get("policy_logprobs_finite") is not True
        or task_group.get("reference_logprobs_finite") is not True
        or task_group.get("backend_failure_count") != 0
        or objective_graph["policy_objective"].get("requires_grad") is not True
        or objective_graph["combined_objective"].get("requires_grad") is not True
        or policy_gradient.get("finite") is not True
        or policy_gradient.get("graph_connected") is not True
        or not _finite_positive(policy_gradient.get("global_norm"))
        or not isinstance(policy_gradient.get("nonzero_gradient_count"), int)
        or cast(int, policy_gradient["nonzero_gradient_count"]) <= 0
        or combined_gradient.get("finite") is not True
        or combined_gradient.get("graph_connected") is not True
        or not _finite_positive(combined_gradient.get("global_norm"))
        or populated.get("finite") is not True
        or populated.get("graph_connected") is not True
        or not _finite_positive(populated.get("global_norm"))
        or populated.get("reference_gradient_count") != 0
        or populated.get("base_gradient_count") != 0
        or stock_loss_equal is not True
    ):
        raise RuntimeError("signal-qualified gradient projection gate failed")


def run(
    *,
    root: Path,
    arm: Arm,
    run_index: int,
    packet_path: Path,
    manifest_path: Path,
    audit_contract_path: Path,
    advantage_contract_path: Path,
    prior_diagnostic_manifest_path: Path,
    qualification_contract_path: Path,
    starting_adapter: Path,
    raw_evidence_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Replay through one selected group and project its gradients without an optimizer."""

    root = root.resolve()
    if root != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("signal qualification is attached to the wrong repository")
    if arm not in ARMS or run_index not in (1, 2):
        raise ValueError("signal qualification arm or run index differs")
    partial_path = raw_evidence_path.with_name("partial_evidence.json")
    trainer_output = raw_evidence_path.parent / "trainer_state"
    if any(
        path.exists() for path in (raw_evidence_path, summary_path, partial_path, trainer_output)
    ):
        raise FileExistsError("qualification outputs must start unused")
    if file_sha256(root / ".venv-training/Scripts/python.exe") != INTERPRETER_SHA256:
        raise ValueError("authorized model interpreter differs")

    schedule = load_schedule(packet_path, manifest_path, arm)
    audit_contract, source_commit = _validate_contract(
        root,
        audit_contract_path,
        schedule=schedule,
        arm=arm,
    )
    qualification = _read(qualification_contract_path)
    verify_qualification_contract(
        root,
        qualification,
        require_clean_synchronized=True,
    )
    if qualification.get("corrected_signal_audit_contract_sha256") != audit_contract.get(
        "signal_audit_contract_sha256"
    ):
        raise ValueError("qualification is bound to a different corrected audit")
    selected = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], qualification["selected_candidates"])[arm],
    )
    prefix = _selected_prefix(schedule.groups, selected)
    audit_rows = _audit_rows(root=root, arm=arm, qualification=qualification)

    advantage_contract = _read(advantage_contract_path)
    verify_advantage_equivalence_contract(advantage_contract)
    if qualification.get("advantage_equivalence_contract_sha256") != advantage_contract.get(
        "advantage_equivalence_contract_sha256"
    ):
        raise ValueError("qualification advantage contract differs")
    prior_manifest = _read(prior_diagnostic_manifest_path)
    verify_prior_diagnostic_manifest(prior_manifest)
    if qualification.get("prior_diagnostic_manifest_sha256") != prior_manifest.get(
        "prior_diagnostic_manifest_sha256"
    ):
        raise ValueError("qualification prior-evidence manifest differs")
    if directory_sha256(starting_adapter) != STARTING_ADAPTER_SHA256[arm]:
        raise ValueError("qualification starting adapter differs")

    modules, launch = _runtime_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    trl = modules["trl"]
    datasets = modules["datasets"]
    numpy = modules["numpy"]
    psutil = modules["psutil"]
    process = psutil.Process()
    _strict(torch, "qualification before model load")
    started = time.perf_counter()
    model_path = (
        root
        / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
        / f"snapshots/{MODEL_REVISION}"
    )
    model, tokenizer, initial_identity, model_load_seconds = _load_dual_adapter_model(
        model_path=model_path,
        starting_adapter=starting_adapter,
        expected_starting_sha256=STARTING_ADAPTER_SHA256[arm],
        modules=modules,
    )
    base_before = capture_base_parameter_state(model)
    initial_policy = capture_adapter_state(model, POLICY_ADAPTER_NAME)
    initial_reference = capture_adapter_state(model, REFERENCE_ADAPTER_NAME)
    reward_callback = VerifierRewardCallback(
        prefix,
        completion_token_counter=_completion_token_counter(tokenizer),
    )
    arguments = _trainer_arguments(
        trl,
        output_dir=raw_evidence_path.parent,
        max_steps=GROUPS_PER_ARM,
    )
    warning_contract = TopPWarningOnlyGenerationContract(
        torch_module=torch,
        generation_owner=transformers.GenerationMixin,
        top_p_call=transformers.generation.logits_process.TopPLogitsWarper.__call__,
        numpy_random=numpy.random,
    )
    audited_trainer = make_truncation_aware_grpo_trainer(
        trl.GRPOTrainer,
        generation_scope_factory=partial(warning_contract.install, "generation"),
    )
    train_dataset = datasets.Dataset.from_list([group.policy_row() for group in schedule.groups])
    trainer = audited_trainer(
        model=model,
        reward_funcs=reward_callback,
        args=arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=None,
    )
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise RuntimeError("qualification constructed an optimizer or scheduler")
    reference_proxy = SharedStartingPolicyReference(trainer.model, torch)
    trainer.ref_model = reference_proxy
    warning_contract.bind_state_probe(partial(model_adapter_state, trainer.model))
    set_policy_active(trainer.model)
    identity_at_projection = assert_policy_reference_identity(
        trainer.model,
        require_policy_trainable=True,
    )
    assert_cuda_only_model(trainer.model)
    dropout_count = assert_dropout_disabled(trainer.model, torch)
    policy_parameters, reference_parameters, base_parameters = _parameter_partitions(trainer.model)
    _strict(torch, "qualification before replay")

    group_records: list[dict[str, object]] = []
    selected_result: Mapping[str, Any] | None = None
    generation_started = time.perf_counter()
    for group in prefix:
        record_start = len(reward_callback.records)
        warning_start = len(warning_contract.call_records())
        rows = [copy.deepcopy(group.policy_row()) for _ in range(COMPLETIONS_PER_GROUP)]
        generated = trainer._generate_and_score_completions(rows)
        if not isinstance(generated, Mapping):
            raise TypeError("qualification generation result is not a mapping")
        result = cast(Mapping[str, Any], generated)
        records = reward_callback.records[record_start:]
        warnings = warning_contract.call_records()[warning_start:]
        if len(records) != COMPLETIONS_PER_GROUP or len(warnings) != 1:
            raise RuntimeError("qualification generation accounting differs")
        group_record = _group_record(
            torch=torch,
            trainer=trainer,
            group=group,
            result=result,
            records=records,
            warning=warnings[0].as_dict(),
        )
        _attach_continuity(
            arm=arm,
            group=group,
            group_record=group_record,
            prior_manifest=prior_manifest,
        )
        expected = audit_rows[group.position]
        if (
            group_record.get("group_record_sha256") != expected.get("group_record_sha256")
            or group_record != expected
        ):
            raise RuntimeError("scientific_replay_drift")
        group_records.append(group_record)
        _write_replace(
            partial_path,
            {
                "schema_version": 1,
                "runtime_id": RUNTIME_ID,
                "arm": arm,
                "run_index": run_index,
                "stage": "fresh_process_replay",
                "group_record_sha256s": [row["group_record_sha256"] for row in group_records],
                "optimizer_created": False,
                "optimizer_steps": 0,
            },
        )
        if group.group_id == selected.get("task_group_id"):
            selected_result = result
        if any(parameter.grad is not None for parameter in trainer.model.parameters()):
            raise RuntimeError("qualification populated gradients before backward")
    torch.cuda.synchronize(0)
    generation_seconds = time.perf_counter() - generation_started
    if selected_result is None or len(group_records) != len(prefix):
        raise RuntimeError("selected qualification result is absent")
    task_group = group_records[-1]
    if task_group.get("group_record_sha256") != selected.get("task_group_record_sha256"):
        raise RuntimeError("selected task replay hash differs")

    policy_logprobs = _policy_logprobs(trainer, selected_result, torch)
    reference_logprobs = selected_result["ref_per_token_logps"]
    advantages = selected_result["advantages"]
    completion_mask = selected_result["completion_mask"]
    components = objective_components(
        torch,
        policy_logprobs=policy_logprobs,
        reference_logprobs=reference_logprobs,
        advantages=advantages,
        completion_mask=completion_mask,
    )
    stock_loss = trainer.compute_loss(trainer.model, selected_result)
    stock_loss_equal = bool(torch.equal(stock_loss.detach(), components.combined.detach()))
    objective_graph = {
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
    del stock_loss
    policy_gradient = gradient_projection(
        torch,
        objective=components.policy,
        named_parameters=policy_parameters,
        retain_graph=True,
    )
    kl_gradient = gradient_projection(
        torch,
        objective=components.kl,
        named_parameters=policy_parameters,
        retain_graph=True,
    )
    combined_gradient = gradient_projection(
        torch,
        objective=components.combined,
        named_parameters=policy_parameters,
        retain_graph=False,
    )
    trainer.model.zero_grad(set_to_none=True)
    populated_policy_logprobs = _policy_logprobs(trainer, selected_result, torch)
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
    _gradient_gate(
        task_group=task_group,
        policy_gradient=policy_gradient,
        combined_gradient=combined_gradient,
        populated=populated,
        objective_graph=objective_graph,
        stock_loss_equal=stock_loss_equal,
    )
    _write_replace(
        partial_path,
        {
            "schema_version": 1,
            "runtime_id": RUNTIME_ID,
            "arm": arm,
            "run_index": run_index,
            "stage": "gradient_gate_passed",
            "group_record_sha256s": [row["group_record_sha256"] for row in group_records],
            "policy_gradient_sha256": policy_gradient["gradient_projection_sha256"],
            "combined_gradient_sha256": combined_gradient["gradient_projection_sha256"],
            "populated_gradient_sha256": populated["gradient_projection_sha256"],
            "optimizer_created": False,
            "optimizer_steps": 0,
        },
    )

    trainer.model.zero_grad(set_to_none=True)
    final_policy = capture_adapter_state(trainer.model, POLICY_ADAPTER_NAME)
    final_reference = capture_adapter_state(trainer.model, REFERENCE_ADAPTER_NAME)
    base_after = capture_base_parameter_state(trainer.model)
    policy_unchanged = (
        final_policy["normalized_tensor_state_sha256"]
        == initial_policy["normalized_tensor_state_sha256"]
    )
    reference_unchanged = (
        final_reference["normalized_tensor_state_sha256"]
        == initial_reference["normalized_tensor_state_sha256"]
    )
    base_unchanged = (
        base_after["base_parameter_state_sha256"] == base_before["base_parameter_state_sha256"]
    )
    warning_evidence = warning_contract.evidence()
    reference_runtime = reference_proxy.evidence()
    if (
        not policy_unchanged
        or not reference_unchanged
        or not base_unchanged
        or warning_evidence.get("generation_calls") != len(prefix)
        or warning_evidence.get("all_warnings_whitelisted") is not True
        or warning_evidence.get("all_state_unchanged") is not True
        or reference_runtime.get("call_count") != len(prefix)
        or trainer.optimizer is not None
        or trainer.lr_scheduler is not None
        or int(trainer.state.global_step) != 0
    ):
        raise RuntimeError("qualification integrity gate failed")

    exact: dict[str, object] = {
        "schema_version": 1,
        "evidence_id": "foundry-l3-grpo-signal-gradient-exact-v1",
        "runtime_id": RUNTIME_ID,
        "arm": arm,
        "source_commit": source_commit,
        "qualification_contract_sha256": qualification["qualification_contract_sha256"],
        "signal_audit_contract_sha256": audit_contract["signal_audit_contract_sha256"],
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "starting_adapter_sha256": STARTING_ADAPTER_SHA256[arm],
        "prefix_group_ids": [group.group_id for group in prefix],
        "prefix_group_record_sha256s": [row["group_record_sha256"] for row in group_records],
        "task_group_id": selected["task_group_id"],
        "task_schedule_position": selected["task_schedule_position"],
        "task_group_record_sha256": task_group["group_record_sha256"],
        "task_reward_variance": task_group["reward_variance"],
        "task_nonzero_advantage_count": task_group["nonzero_advantage_count"],
        "task_valid_completion_token_count": task_group["valid_completion_token_count"],
        "fresh_process_replay_exact": True,
        "identity_at_projection": identity_at_projection,
        "initial_identity": initial_identity,
        "initial_policy": initial_policy,
        "initial_reference": initial_reference,
        "base_before": base_before,
        "objective_values": objective_values,
        "objective_graph": objective_graph,
        "stock_loss_equal": stock_loss_equal,
        "policy_gradient": policy_gradient,
        "kl_gradient": kl_gradient,
        "combined_gradient": combined_gradient,
        "populated_combined_gradient": populated,
        "final_policy": final_policy,
        "final_reference": final_reference,
        "base_after": base_after,
        "policy_unchanged": policy_unchanged,
        "reference_unchanged": reference_unchanged,
        "base_unchanged": base_unchanged,
        "reference_runtime": reference_runtime,
        "warning_evidence": warning_evidence,
        "optimizer_created": False,
        "scheduler_created": False,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "backward_calls": 1,
        "adapter_saved": False,
    }
    exact["exact_projection_sha256"] = canonical_sha256(exact)
    raw: dict[str, object] = {
        "schema_version": 1,
        "runtime_id": RUNTIME_ID,
        "arm": arm,
        "run_index": run_index,
        "exact_scientific_tensor_evidence": exact,
        "groups": group_records,
        "raw_prompts_or_completions_present": True,
    }
    raw["raw_projection_sha256"] = canonical_sha256(raw)
    _write_new(raw_evidence_path, raw)

    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    model_reference = weakref.ref(trainer.model)
    warning_contract.release_state_probe()
    del (
        components,
        populated_components,
        populated_policy_logprobs,
        policy_logprobs,
        reference_logprobs,
        advantages,
        completion_mask,
        selected_result,
        result,
        generated,
        policy_parameters,
        reference_parameters,
        base_parameters,
        reference_proxy,
        trainer,
        audited_trainer,
        model,
        train_dataset,
        reward_callback,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    if model_reference() is not None:
        raise RuntimeError("qualification model remained alive after release")
    _strict(torch, "qualification publication")
    runtime_seconds = time.perf_counter() - started
    summary: dict[str, object] = {
        "schema_version": 1,
        "runtime_id": RUNTIME_ID,
        "arm": arm,
        "run_index": run_index,
        "source_commit": source_commit,
        "qualification_contract_sha256": qualification["qualification_contract_sha256"],
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "starting_adapter_sha256": STARTING_ADAPTER_SHA256[arm],
        "prefix_groups": len(prefix),
        "prefix_completions": len(prefix) * COMPLETIONS_PER_GROUP,
        "task_group_id": selected["task_group_id"],
        "task_schedule_position": selected["task_schedule_position"],
        "task_group_record_sha256": task_group["group_record_sha256"],
        "fresh_process_replay_exact": True,
        "exact_projection_sha256": exact["exact_projection_sha256"],
        "policy_gradient_global_norm": policy_gradient["global_norm"],
        "combined_gradient_global_norm": combined_gradient["global_norm"],
        "policy_nonzero_gradient_tensor_count": policy_gradient["nonzero_gradient_count"],
        "reference_gradient_count": populated["reference_gradient_count"],
        "base_gradient_count": populated["base_gradient_count"],
        "policy_unchanged": policy_unchanged,
        "reference_unchanged": reference_unchanged,
        "base_unchanged": base_unchanged,
        "optimizer_created": False,
        "scheduler_created": False,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "backward_calls": 1,
        "adapter_saved": False,
        "dropout_disabled": True,
        "dropout_module_count": dropout_count,
        "launch_evidence": launch,
        "model_load_seconds": model_load_seconds,
        "generation_seconds": generation_seconds,
        "runtime_seconds": runtime_seconds,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_process_rss_bytes": _peak_process_ram(process),
        "raw_projection_file_sha256": file_sha256(raw_evidence_path),
        "partial_evidence_file_sha256": file_sha256(partial_path),
        "output_disk_bytes": sum(
            item.stat().st_size for item in raw_evidence_path.parent.rglob("*") if item.is_file()
        ),
        "prompts_completions_or_answers_present": False,
        "gate_passed": True,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_new(summary_path, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-contract", type=Path, required=True)
    parser.add_argument("--advantage-contract", type=Path, required=True)
    parser.add_argument("--prior-diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--qualification-contract", type=Path, required=True)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(
        root=args.root,
        arm=cast(Arm, args.arm),
        run_index=args.run_index,
        packet_path=args.packet,
        manifest_path=args.manifest,
        audit_contract_path=args.audit_contract,
        advantage_contract_path=args.advantage_contract,
        prior_diagnostic_manifest_path=args.prior_diagnostic_manifest,
        qualification_contract_path=args.qualification_contract,
        starting_adapter=args.starting_adapter,
        raw_evidence_path=args.raw_evidence,
        summary_path=args.summary,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
