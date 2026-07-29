"""CUDA-only, no-optimization Milestone 14B full-schedule signal audit."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import subprocess
import time
import weakref
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

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
from foundry.phase2.l3_grpo_runtime import (
    RuntimeGroup,
    RuntimeSchedule,
    VerifierRewardCallback,
    _completion_token_counter,
    _load_dual_adapter_model,
    _runtime_modules,
    _strict,
    _token_lengths,
    _trainer_arguments,
    load_schedule,
)
from foundry.phase2.l3_grpo_signal_audit import (
    ARMS,
    AUDIT_ID,
    COMPLETIONS_PER_ARM,
    COMPLETIONS_PER_GROUP,
    GROUPS_PER_ARM,
    REPLAY_GROUPS_PER_ARM,
    REWARD_COMPONENT_FIELDS,
    TASK_GROUPS_PER_ARM,
    classify_zero_variance_group,
    signal_audit_method_contract,
)
from foundry.phase2.l3_grpo_zero_gradient import reward_projection
from foundry.training.config import canonical_sha256
from foundry.training.grpo_compatibility import (
    TopPWarningOnlyGenerationContract,
    model_adapter_state,
)
from foundry.training.grpo_replay_evidence import (
    capture_base_parameter_state,
    tensor_evidence,
)
from foundry.training.grpo_runtime import (
    _peak_process_ram,
    assert_cuda_only_model,
    assert_dropout_disabled,
)
from foundry.training.grpo_trainer import make_truncation_aware_grpo_trainer
from foundry.training.qlora import directory_sha256, file_sha256

Arm = Literal["generic", "targeted"]
RUNTIME_ID = "foundry-l3-grpo-signal-audit-runtime-v1"
SOURCE_COMMIT = "d1c4edf15510128413735a19f937d5451137ae0b"
EXPECTED_SCHEDULE_SHA256 = {
    "generic": "ff1005a1d7381acd52dd28b3d054b2979986c47595ed09c944880ea5fc5f5ff3",
    "targeted": "8326c1b91ba127c4734527abfed2f8bca41ecbb3a0bb7bc62a5bf940ac24f0c4",
}


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite signal-audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_json_replace(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.strip()


def _validate_contract(
    root: Path,
    path: Path,
    *,
    schedule: RuntimeSchedule,
    arm: Arm,
) -> tuple[dict[str, Any], str]:
    contract = _read(path)
    supplied = contract.get("signal_audit_contract_sha256")
    payload = {
        name: value for name, value in contract.items() if name != "signal_audit_contract_sha256"
    }
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError("signal-audit contract does not reconstruct")
    if (
        contract.get("contract_id") != "foundry-l3-grpo-signal-audit-v1"
        or contract.get("starting_commit") != SOURCE_COMMIT
        or contract.get("method_contract") != signal_audit_method_contract()
        or cast(dict[str, str], contract["starting_adapters"])[arm] != STARTING_ADAPTER_SHA256[arm]
        or cast(dict[str, str], contract["schedules"])[arm] != EXPECTED_SCHEDULE_SHA256[arm]
        or schedule.manifest_sha256 != EXPECTED_SCHEDULE_SHA256[arm]
        or contract.get("scientific_settings_changed") is not False
        or contract.get("optimizer_creation_authorized") is not False
        or contract.get("backward_authorized") is not False
        or contract.get("adapter_save_authorized") is not False
    ):
        raise ValueError("signal-audit contract differs")
    implementation = _read(path.with_name("milestone14b_signal_audit_implementation.json"))
    implementation_sha256 = implementation.get("implementation_sha256")
    implementation_payload = {
        name: value for name, value in implementation.items() if name != "implementation_sha256"
    }
    if (
        not isinstance(implementation_sha256, str)
        or implementation_sha256 != canonical_sha256(implementation_payload)
        or implementation_sha256 != contract.get("implementation_sha256")
    ):
        raise ValueError("signal-audit implementation manifest differs")
    for row_value in cast(list[object], implementation.get("files")):
        if not isinstance(row_value, dict):
            raise ValueError("implementation file row differs")
        row = cast(dict[str, object], row_value)
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ValueError("implementation path differs")
        source = (root / relative).resolve()
        if (
            not source.is_relative_to(root.resolve())
            or file_sha256(source) != row.get("sha256")
            or source.stat().st_size != row.get("bytes")
        ):
            raise ValueError("signal-audit implementation source differs")
    head = _git(root, "rev-parse", "HEAD")
    if (
        _git(root, "branch", "--show-current") != "main"
        or head != _git(root, "rev-parse", "origin/main")
        or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
        != ["0", "0"]
        or _git(root, "status", "--porcelain")
    ):
        raise RuntimeError("signal audit requires synchronized clean main")
    return contract, head


def _component_vectors(records: Sequence[Any]) -> dict[str, list[float]]:
    return {
        name: [float(getattr(record.reward, name)) for record in records]
        for name in REWARD_COMPONENT_FIELDS
    }


def _finite_tensor(torch: Any, value: Any) -> bool:
    return bool(torch.isfinite(value).all().item())


def _group_record(
    *,
    torch: Any,
    trainer: Any,
    group: RuntimeGroup,
    result: Mapping[str, Any],
    records: Sequence[Any],
    warning: Mapping[str, object],
) -> dict[str, object]:
    completion_ids = result["completion_ids"]
    completion_mask = result["completion_mask"]
    prompt_ids = result["prompt_ids"]
    prompt_mask = result["prompt_mask"]
    reference_logprobs = result["ref_per_token_logps"]
    advantages = result["advantages"]
    if reference_logprobs is None:
        raise RuntimeError("signal audit requires frozen reference log probabilities")
    if len(records) != COMPLETIONS_PER_GROUP:
        raise RuntimeError("signal-audit reward record cardinality differs")
    if any(record.group_id != group.group_id for record in records):
        raise RuntimeError("signal-audit reward record group differs")
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    set_policy_active(trainer.model)
    with torch.no_grad():
        policy_logprobs = trainer._get_per_token_logps(
            trainer.model,
            prompt_completion_ids,
            attention_mask,
            int(completion_ids.size(1)),
            COMPLETIONS_PER_GROUP,
        )
    if active_adapter_name(trainer.model) != POLICY_ADAPTER_NAME:
        raise RuntimeError("policy adapter was not restored after log-probability audit")
    policy_finite = _finite_tensor(torch, policy_logprobs)
    reference_finite = _finite_tensor(torch, reference_logprobs)
    delta = reference_logprobs - policy_logprobs
    per_token_kl = torch.exp(delta) - delta - 1
    kl_finite = _finite_tensor(torch, per_token_kl)
    valid = completion_mask.to(dtype=torch.bool)
    valid_counts = [int(value) for value in completion_mask.sum(dim=1).detach().cpu().tolist()]
    valid_kl = per_token_kl[valid]
    mean_kl = float(valid_kl.mean().item()) if int(valid_kl.numel()) else None
    maximum_kl = float(valid_kl.max().item()) if int(valid_kl.numel()) else None

    reward_unprojected = [float(record.reward.total) for record in records]
    projected = reward_projection(torch, reward_unprojected)
    projected_advantages = torch.tensor(
        cast(list[float], projected["advantages"]),
        dtype=advantages.dtype,
        device=advantages.device,
    )
    if not bool(torch.equal(advantages, projected_advantages)):
        raise RuntimeError("stock TRL advantages differ from frozen reward projection")
    component_vectors = _component_vectors(records)
    breakdowns = [record.reward.as_dict() for record in records]
    reward_contract_consistent = all(
        math.isclose(
            float(record.reward.total),
            round(
                math.fsum(float(getattr(record.reward, name)) for name in REWARD_COMPONENT_FIELDS),
                10,
            ),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for record in records
    )
    completion_sha256s = [str(record.completion_sha256) for record in records]
    correctness = [bool(record.reward.answer_correct) for record in records]
    variance = float(cast(float, projected["reward_variance"]))
    classification = classify_zero_variance_group(
        reward_variance=variance,
        completion_sha256s=completion_sha256s,
        reward_vector=cast(list[float], projected["rewards"]),
        reward_component_vectors=component_vectors,
        correctness=correctness,
        evidence_complete=(
            policy_finite and reference_finite and kl_finite and reward_contract_consistent
        ),
    )
    lengths = _token_lengths(completion_ids, int(trainer.processing_class.eos_token_id))
    group_record: dict[str, object] = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "arm": group.arm,
        "schedule_position": group.position,
        "group_id": group.group_id,
        "source_kind": group.source_kind,
        "task_family": group.category if group.source_kind == "task" else None,
        "replay_section": group.category if group.source_kind == "base_replay" else None,
        "prompt_sha256": group.prompt_sha256,
        "completion_count": COMPLETIONS_PER_GROUP,
        "completion_sha256s": completion_sha256s,
        "completion_token_counts": lengths,
        "decoded_completion_token_counts": [int(record.completion_tokens) for record in records],
        "distinct_completion_count": len(set(completion_sha256s)),
        "generated_token_ids": completion_ids.detach().cpu().tolist(),
        "completion_mask": completion_mask.detach().cpu().tolist(),
        "valid_completion_token_counts": valid_counts,
        "valid_completion_token_count": sum(valid_counts),
        "reward_vector_unprojected": reward_unprojected,
        "reward_vector": projected["rewards"],
        "reward_component_vectors": component_vectors,
        "reward_breakdowns": breakdowns,
        "reward_mean": projected["reward_mean"],
        "reward_variance": variance,
        "advantages": projected["advantages"],
        "nonzero_advantage_count": sum(
            float(value) != 0.0 for value in cast(list[float], projected["advantages"])
        ),
        "reward_rank_count": len(set(cast(list[float], projected["rewards"]))),
        "correctness_vector": correctness,
        "correctness_count": sum(correctness),
        "extraction_count": sum(bool(record.reward.extractable) for record in records),
        "compliant_format_count": sum(bool(record.reward.exact_format) for record in records),
        "malformed_count": sum(bool(record.reward.malformed_output) for record in records),
        "prompt_echo_count": sum(bool(record.reward.prompt_echo) for record in records),
        "question_generation_count": sum(
            bool(record.reward.question_generation) for record in records
        ),
        "truncation_count": sum(bool(record.reward.generation_truncated) for record in records),
        "backend_failure_count": sum(bool(record.reward.backend_failure) for record in records),
        "reward_contract_consistent": reward_contract_consistent,
        "policy_logprobs_finite": policy_finite,
        "reference_logprobs_finite": reference_finite,
        "policy_logprobs": policy_logprobs.detach().cpu().tolist(),
        "reference_logprobs": reference_logprobs.detach().cpu().tolist(),
        "policy_logprobs_evidence": tensor_evidence(policy_logprobs).as_dict(),
        "reference_logprobs_evidence": tensor_evidence(reference_logprobs).as_dict(),
        "policy_reference_kl": {
            "finite": kl_finite,
            "valid_token_count": int(valid_kl.numel()),
            "mean": mean_kl,
            "maximum": maximum_kl,
            "per_token_values": per_token_kl.detach().cpu().tolist(),
            "tensor_evidence": tensor_evidence(per_token_kl).as_dict(),
        },
        "zero_variance_classification": classification,
        "warning_evidence": dict(warning),
    }
    group_record["group_record_sha256"] = canonical_sha256(group_record)
    return group_record


def _partial(
    *,
    arm: Arm,
    schedule: RuntimeSchedule,
    contract_sha256: str,
    source_commit: str,
    groups: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "runtime_id": RUNTIME_ID,
        "arm": arm,
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "signal_audit_contract_sha256": contract_sha256,
        "source_commit": source_commit,
        "completed_group_count": len(groups),
        "completed_completion_count": len(groups) * COMPLETIONS_PER_GROUP,
        "groups": list(groups),
        "optimizer_created": False,
        "backward_calls": 0,
        "scheduler_created": False,
        "adapter_saved": False,
    }
    value["partial_audit_sha256"] = canonical_sha256(value)
    return value


def run(
    *,
    root: Path,
    arm: Arm,
    packet_path: Path,
    manifest_path: Path,
    contract_path: Path,
    starting_adapter: Path,
    raw_evidence_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Audit one complete 32-group arm without constructing an optimizer."""

    root = root.resolve()
    if root != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("signal audit is attached to the wrong repository")
    if arm not in ARMS:
        raise ValueError("signal audit arm differs")
    partial_path = raw_evidence_path.with_name("partial_evidence.json")
    trainer_output = raw_evidence_path.parent / "trainer_state"
    if any(
        path.exists() for path in (raw_evidence_path, summary_path, partial_path, trainer_output)
    ):
        raise FileExistsError("signal-audit outputs must start unused")
    if file_sha256(root / ".venv-training/Scripts/python.exe") != INTERPRETER_SHA256:
        raise ValueError("authorized model interpreter differs")
    schedule = load_schedule(packet_path, manifest_path, arm)
    contract, source_commit = _validate_contract(root, contract_path, schedule=schedule, arm=arm)
    if directory_sha256(starting_adapter) != STARTING_ADAPTER_SHA256[arm]:
        raise ValueError("signal-audit starting adapter differs")

    modules, launch = _runtime_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    trl = modules["trl"]
    datasets = modules["datasets"]
    numpy = modules["numpy"]
    psutil = modules["psutil"]
    process = psutil.Process()
    _strict(torch, "signal audit before model load")
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
        schedule.groups,
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
        raise RuntimeError("signal audit constructed an optimizer or scheduler")
    reference_proxy = SharedStartingPolicyReference(trainer.model, torch)
    trainer.ref_model = reference_proxy
    warning_contract.bind_state_probe(partial(model_adapter_state, trainer.model))
    set_policy_active(trainer.model)
    assert_policy_reference_identity(trainer.model, require_policy_trainable=True)
    assert_cuda_only_model(trainer.model)
    dropout_count = assert_dropout_disabled(trainer.model, torch)
    _strict(torch, "signal audit before generation")

    group_records: list[dict[str, object]] = []
    generation_started = time.perf_counter()
    for group in schedule.groups:
        before_records = len(reward_callback.records)
        rows = [copy.deepcopy(group.policy_row()) for _ in range(COMPLETIONS_PER_GROUP)]
        result = cast(
            Mapping[str, Any],
            trainer._generate_and_score_completions(rows),
        )
        records = reward_callback.records[before_records:]
        warning = warning_contract.call_records()[-1].as_dict()
        group_records.append(
            _group_record(
                torch=torch,
                trainer=trainer,
                group=group,
                result=result,
                records=records,
                warning=warning,
            )
        )
        if any(parameter.grad is not None for parameter in trainer.model.parameters()):
            raise RuntimeError("signal audit populated gradients without backward")
        if (
            trainer.optimizer is not None
            or trainer.lr_scheduler is not None
            or int(trainer.state.global_step) != 0
        ):
            raise RuntimeError("signal audit advanced optimization state")
        _write_json_replace(
            partial_path,
            _partial(
                arm=arm,
                schedule=schedule,
                contract_sha256=cast(str, contract["signal_audit_contract_sha256"]),
                source_commit=source_commit,
                groups=group_records,
            ),
        )
    torch.cuda.synchronize(0)
    generation_seconds = time.perf_counter() - generation_started
    if len(group_records) != GROUPS_PER_ARM or len(reward_callback.records) != COMPLETIONS_PER_ARM:
        raise RuntimeError("signal-audit group or completion accounting differs")

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
        raise RuntimeError("signal audit mutated policy, reference, or base parameters")
    warning_evidence = warning_contract.evidence()
    if (
        warning_evidence["generation_calls"] != GROUPS_PER_ARM
        or warning_evidence["all_warnings_whitelisted"] is not True
        or warning_evidence["all_state_unchanged"] is not True
    ):
        raise RuntimeError("signal-audit generation warning contract failed")
    reference_runtime = reference_proxy.evidence()
    if reference_runtime.get("call_count") != GROUPS_PER_ARM:
        raise RuntimeError("signal-audit reference call accounting differs")

    raw: dict[str, object] = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "runtime_id": RUNTIME_ID,
        "arm": arm,
        "source_commit": source_commit,
        "signal_audit_contract_sha256": contract["signal_audit_contract_sha256"],
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "starting_adapter_sha256": STARTING_ADAPTER_SHA256[arm],
        "group_count": GROUPS_PER_ARM,
        "task_group_count": TASK_GROUPS_PER_ARM,
        "replay_group_count": REPLAY_GROUPS_PER_ARM,
        "completion_count": COMPLETIONS_PER_ARM,
        "groups": group_records,
        "initial_identity": initial_identity,
        "initial_policy": initial_policy,
        "initial_reference": initial_reference,
        "final_policy": final_policy,
        "final_reference": final_reference,
        "base_before": base_before,
        "base_after": base_after,
        "reference_runtime": reference_runtime,
        "warning_evidence": warning_evidence,
        "optimizer_created": False,
        "backward_calls": 0,
        "scheduler_created": False,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "adapter_mutated": False,
        "adapter_saved": False,
    }
    raw["raw_audit_sha256"] = canonical_sha256(raw)
    _write_json_new(raw_evidence_path, raw)

    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    runtime_seconds = time.perf_counter() - started
    summary: dict[str, object] = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "runtime_id": RUNTIME_ID,
        "arm": arm,
        "source_commit": source_commit,
        "signal_audit_contract_sha256": contract["signal_audit_contract_sha256"],
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "starting_adapter_sha256": STARTING_ADAPTER_SHA256[arm],
        "groups": GROUPS_PER_ARM,
        "task_groups": TASK_GROUPS_PER_ARM,
        "replay_groups": REPLAY_GROUPS_PER_ARM,
        "completions": COMPLETIONS_PER_ARM,
        "completion_tokens": sum(
            sum(cast(list[int], row["completion_token_counts"])) for row in group_records
        ),
        "scheduled_prompt_tokens": sum(group.prompt_tokens for group in schedule.groups),
        "generation_input_prompt_tokens": (
            sum(group.prompt_tokens for group in schedule.groups) * COMPLETIONS_PER_GROUP
        ),
        "zero_variance_groups": sum(row["reward_variance"] == 0.0 for row in group_records),
        "nonzero_variance_groups": sum(
            float(cast(float, row["reward_variance"])) > 0.0 for row in group_records
        ),
        "backend_failures": sum(cast(int, row["backend_failure_count"]) for row in group_records),
        "group_record_sha256s": [cast(str, row["group_record_sha256"]) for row in group_records],
        "initial_policy_sha256": initial_policy["normalized_tensor_state_sha256"],
        "final_policy_sha256": final_policy["normalized_tensor_state_sha256"],
        "initial_reference_sha256": initial_reference["normalized_tensor_state_sha256"],
        "final_reference_sha256": final_reference["normalized_tensor_state_sha256"],
        "base_before_sha256": base_before["base_parameter_state_sha256"],
        "base_after_sha256": base_after["base_parameter_state_sha256"],
        "policy_unchanged": True,
        "reference_unchanged": True,
        "base_unchanged": True,
        "optimizer_created": False,
        "backward_calls": 0,
        "scheduler_created": False,
        "adapter_saved": False,
        "dropout_disabled": True,
        "dropout_module_count": dropout_count,
        "launch_evidence": launch,
        "warning_evidence": warning_evidence,
        "model_load_seconds": model_load_seconds,
        "generation_seconds": generation_seconds,
        "runtime_seconds": runtime_seconds,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_process_rss_bytes": _peak_process_ram(process),
        "raw_evidence_file_sha256": file_sha256(raw_evidence_path),
        "partial_evidence_file_sha256": file_sha256(partial_path),
        "output_disk_bytes": sum(
            item.stat().st_size for item in raw_evidence_path.parent.rglob("*") if item.is_file()
        ),
        "gate_passed": True,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json_new(summary_path, summary)

    model_reference = weakref.ref(trainer.model)
    warning_contract.release_state_probe()
    del reference_proxy, trainer, audited_trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    if model_reference() is not None:
        raise RuntimeError("signal-audit model remained alive after release")
    _strict(torch, "signal audit publication")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-contract", type=Path, required=True)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(
        root=args.root,
        arm=cast(Arm, args.arm),
        packet_path=args.packet,
        manifest_path=args.manifest,
        contract_path=args.audit_contract,
        starting_adapter=args.starting_adapter,
        raw_evidence_path=args.raw_evidence,
        summary_path=args.summary,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
