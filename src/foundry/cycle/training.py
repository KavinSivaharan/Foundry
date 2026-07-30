"""Deterministic L3-targeted continuation training for Foundry Cycle 1."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from foundry.cycle.contract import (
    CycleConfig,
    validate_file_identity,
    validate_process_environment,
    verified_payload,
)
from foundry.cycle.corpus import load_cycle_records
from foundry.cycle.generation_observability import rng_state_sha256
from foundry.phase2 import vetted_qlora_kl as qlora
from foundry.phase2.layer_restricted import scope_for_label
from foundry.phase2.vetted_qlora_layer_restricted import (
    _active_trainable_parameters,
    _inventory,
    _validation_ce,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256


def _hash_state(value: object, digest: Any) -> None:
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        tensor = value.detach().contiguous().cpu()
        digest.update(b"tensor")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(cast(Any, __import__("torch")).uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_state(str(key), digest)
            _hash_state(value[key], digest)
        return
    if isinstance(value, list | tuple):
        digest.update(b"list")
        for item in value:
            _hash_state(item, digest)
        return
    if isinstance(value, float):
        digest.update(b"float")
        digest.update(struct.pack("!d", value))
        return
    digest.update(type(value).__name__.encode())
    digest.update(repr(value).encode())


def state_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _hash_state(value, digest)
    return digest.hexdigest()


def _named_tensor_sha256(named: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.view(cast(Any, __import__("torch")).uint8).numpy().tobytes())
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def _load_values(
    *,
    corpus_path: Path,
    replay_path: Path,
    validation_path: Path,
    tokenizer: Any,
) -> tuple[
    dict[tuple[str, str, str], dict[str, list[int]]],
    list[dict[str, list[int]]],
]:
    values: dict[tuple[str, str, str], dict[str, list[int]]] = {}
    for (source_id, variant), record in load_cycle_records(corpus_path).items():
        values[("vetted", source_id, variant)] = qlora._tokenize(
            {**record, "kind": "vetted"},
            tokenizer,
        )
    replay = cast(
        dict[str, Any],
        json.loads(replay_path.read_text(encoding="utf-8")),
    )
    for item in cast(list[dict[str, Any]], replay["items"]):
        values[("replay", str(item["id"]), "replay")] = qlora._tokenize(
            {**item, "kind": "replay"},
            tokenizer,
        )
    validation = [
        qlora._tokenize({**item, "kind": "vetted"}, tokenizer) for item in _jsonl(validation_path)
    ]
    return values, validation


def _save_adapter(
    *,
    model: Any,
    tokenizer: Any,
    path: Path,
    validation: list[dict[str, list[int]]],
    torch: Any,
    include_validation: bool,
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    result: dict[str, Any] = {
        "adapter_sha256": directory_sha256(path),
        "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
        "adapter_tensor_sha256": _named_tensor_sha256(
            [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name]
        ),
    }
    if include_validation:
        result["validation_ce"] = _validation_ce(model, validation, torch)
    return result


def train_candidate(
    *,
    config: CycleConfig,
    corpus_directory: Path,
    output_directory: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Continue the frozen L3 targeted adapter for two or thirty-two steps."""

    validate_process_environment(config=config)
    if output_directory.exists():
        raise FileExistsError("continuation-training output must be fresh")
    training = config.section("training")
    warm_start = config.section("warm_start")
    model_contract = config.section("model")
    dataset = config.section("dataset")
    corpus_contract = config.section("corpus")
    max_steps = 2 if smoke else int(training["optimizer_steps"])
    checkpoints = (2,) if smoke else tuple(int(item) for item in training["checkpoints"])
    corpus_manifest = verified_payload(
        corpus_directory / "manifest.json",
        "corpus_sha256",
    )
    expected_corpus_hash = corpus_manifest[
        "corpus_file_sha256" if smoke else "task_corpus_file_sha256"
    ]
    if (
        corpus_manifest["optimizer_steps"] != max_steps
        or file_sha256(corpus_directory / "task_corpus.jsonl") != expected_corpus_hash
        or file_sha256(corpus_directory / "schedule.json")
        != corpus_manifest["schedule_file_sha256"]
    ):
        raise ValueError("continuation corpus artifact identity differs")
    if not smoke:
        freeze = verified_payload(
            corpus_directory.parent / "summary.json",
            "corpus_freeze_sha256",
        )
        if (
            freeze["passed"] is not True
            or freeze["corpus"]["corpus_sha256"] != corpus_manifest["corpus_sha256"]
        ):
            raise ValueError("production corpus-freeze identity differs")
    schedule = cast(
        list[dict[str, Any]],
        json.loads((corpus_directory / "schedule.json").read_text(encoding="utf-8")),
    )
    if canonical_sha256(schedule) != corpus_manifest["schedule_sha256"]:
        raise ValueError("continuation schedule identity differs")
    if len(schedule) != max_steps:
        raise ValueError("continuation schedule step count differs")
    expected_tokens = (
        sum(int(item["loss_bearing_tokens"]) for item in schedule)
        if smoke
        else int(corpus_contract["total_assistant_tokens"])
    )
    if sum(int(item["loss_bearing_tokens"]) for item in schedule) != expected_tokens:
        raise ValueError("continuation schedule token count differs")

    modules, launch = qlora._modules()
    torch = modules["torch"]
    rng_before_sha256 = rng_state_sha256(torch)
    started = time.perf_counter()
    model_path = config.resolve_artifact(str(model_contract["snapshot_relative_path"]))
    warm_start_path = config.resolve_artifact(str(warm_start["adapter_relative_path"]))
    if directory_sha256(warm_start_path) != warm_start["adapter_sha256"]:
        raise ValueError("continuation warm-start adapter identity differs")
    base, tokenizer = qlora._load_base(model_path, modules)
    base = modules["peft"].prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=False,
    )
    if hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    model = modules["peft"].PeftModel.from_pretrained(
        base,
        str(warm_start_path),
        local_files_only=True,
        is_trainable=True,
    )
    if getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError("gradient checkpointing is unexpectedly enabled")
    model.config.use_cache = False
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("continuation training detected CPU or disk offload")
    scope = scope_for_label("L3")
    inventory = _inventory(model, scope)
    trainable = _active_trainable_parameters(model, scope)
    named_trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    initial_lora_sha256 = _named_tensor_sha256(named_trainable)
    initial_lora = {name: parameter.detach().cpu().clone() for name, parameter in named_trainable}
    optimizer = modules["bitsandbytes"].optim.PagedAdamW8bit(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in cast(list[Any], group["params"])
    }
    if optimizer_ids != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer ownership differs from L3 LoRA tensors")
    scheduler = modules["transformers"].get_scheduler(
        str(training["scheduler"]),
        optimizer=optimizer,
        num_warmup_steps=int(training["warmup_steps"]),
        num_training_steps=int(training["scheduler_horizon_steps"]),
    )
    replay_path = validate_file_identity(
        config,
        str(corpus_contract["replay_relative_source_path"]),
        str(corpus_contract["replay_file_sha256"]),
    )
    validation_path = validate_file_identity(
        config,
        str(dataset["targeted_validation_relative_path"]),
        str(dataset["targeted_validation_sha256"]),
    )
    values, validation = _load_values(
        corpus_path=corpus_directory / "task_corpus.jsonl",
        replay_path=replay_path,
        validation_path=validation_path,
        tokenizer=tokenizer,
    )
    probe_key = sorted(key for key in values if key[0] == "replay")[0]
    probe = values[probe_key]
    with model.disable_adapter():
        base_generation = qlora._generate(model, probe, torch)
    base_before = qlora._base_parameter_fingerprint(model, torch)
    model.train()
    metrics: list[dict[str, Any]] = []
    checkpoint_evidence: dict[str, Any] = {}
    total_tokens = 0
    positive_lr_positive_gradient = False
    for step in schedule:
        step_number = int(step["step"])
        step_tokens = int(step["loss_bearing_tokens"])
        optimizer.zero_grad(set_to_none=True)
        sums: defaultdict[str, float] = defaultdict(float)
        for occurrence in cast(list[dict[str, Any]], step["occurrences"]):
            kind = str(occurrence["kind"])
            record_id = str(occurrence["record_id"])
            variant = str(occurrence.get("variant", "replay"))
            value = values[(kind, record_id, variant)]
            actual = sum(label != -100 for label in value["labels"])
            if actual != int(occurrence["tokens"]):
                raise RuntimeError("scheduled assistant-token count differs")
            inputs = qlora._inputs(value, torch)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs, use_cache=False).loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite continuation loss")
            (loss * (actual / step_tokens)).backward()
            sums[f"{kind}_loss"] += float(loss.detach().float().item()) * actual
            sums[f"{kind}_tokens"] += actual
            sums["total_loss"] += float(loss.detach().float().item()) * actual
        gradients = [
            (name, parameter.grad)
            for name, parameter in named_trainable
            if parameter.grad is not None
        ]
        base_gradients = [
            name
            for name, parameter in model.named_parameters()
            if "lora_" not in name and parameter.grad is not None
        ]
        if (
            not gradients
            or base_gradients
            or any(not bool(torch.isfinite(gradient).all().item()) for _, gradient in gradients)
        ):
            raise RuntimeError("continuation gradient ownership or finiteness failed")
        gradient_norm = math.sqrt(
            sum(float(gradient.detach().float().pow(2).sum().item()) for _, gradient in gradients)
        )
        gradient_sha256 = _named_tensor_sha256([(name, gradient) for name, gradient in gradients])
        clipped_norm = float(
            torch.nn.utils.clip_grad_norm_(
                trainable,
                float(training["gradient_clip_norm"]),
            )
            .detach()
            .float()
            .item()
        )
        lr_before = float(optimizer.param_groups[0]["lr"])
        if lr_before > 0 and gradient_norm > 0:
            positive_lr_positive_gradient = True
        if lr_before > 0 and gradient_norm <= 0:
            raise RuntimeError("positive-learning-rate step has zero gradient")
        optimizer.step()
        scheduler.step()
        total_tokens += step_tokens
        record: dict[str, Any] = {
            "step": step_number,
            "loss": sums["total_loss"] / step_tokens,
            "vetted_ce": (
                sums["vetted_loss"] / sums["vetted_tokens"] if sums["vetted_tokens"] else None
            ),
            "replay_ce": (
                sums["replay_loss"] / sums["replay_tokens"] if sums["replay_tokens"] else None
            ),
            "loss_bearing_tokens": step_tokens,
            "gradient_norm": gradient_norm,
            "clipped_gradient_norm": clipped_norm,
            "gradient_sha256": gradient_sha256,
            "lr_before_optimizer": lr_before,
            "lr_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            "optimizer_state_sha256": state_sha256(optimizer.state_dict()),
            "scheduler_state_sha256": canonical_sha256(scheduler.state_dict()),
        }
        metrics.append(record)
        if step_number in checkpoints:
            checkpoint_evidence[str(step_number)] = _save_adapter(
                model=model,
                tokenizer=tokenizer,
                path=output_directory / f"checkpoint-{step_number}" / "adapter",
                validation=validation,
                torch=torch,
                include_validation=not smoke,
            )
        print(
            json.dumps(
                {
                    "cycle_training_step": step_number,
                    "cycle_training_steps_total": max_steps,
                    "smoke": smoke,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    changed = [
        name
        for name, parameter in named_trainable
        if not torch.equal(initial_lora[name], parameter.detach().cpu())
    ]
    if not changed or not positive_lr_positive_gradient:
        raise RuntimeError("continuation LoRA update gate failed")
    base_after = qlora._base_parameter_fingerprint(model, torch)
    if base_before != base_after:
        raise RuntimeError("base parameters changed during continuation")
    with model.disable_adapter():
        restored = qlora._generate(model, probe, torch)
    if restored != base_generation:
        raise RuntimeError("adapter-disabled base behavior did not restore")
    final_path = output_directory / f"checkpoint-{max_steps}" / "adapter"
    final_tensor_sha256 = _named_tensor_sha256(
        [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name]
    )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    memory = modules["psutil"].Process().memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    fresh_base, _ = qlora._load_base(model_path, modules)
    reloaded = modules["peft"].PeftModel.from_pretrained(
        fresh_base,
        str(final_path),
        local_files_only=True,
        is_trainable=False,
    )
    reload_inventory = _inventory(reloaded, scope)
    offline_reload = all(parameter.device.type == "cuda" for parameter in reloaded.parameters())
    del reloaded, fresh_base
    gc.collect()
    torch.cuda.empty_cache()
    rng_after_sha256 = rng_state_sha256(torch)
    result: dict[str, Any] = {
        "schema_version": 1,
        "training_id": (
            "foundry-cycle1-compatibility-training-v1"
            if smoke
            else "foundry-cycle1-production-training-v1"
        ),
        "smoke": smoke,
        "warm_start_adapter_sha256": warm_start["adapter_sha256"],
        "optimizer": training["optimizer"],
        "learning_rate": training["learning_rate"],
        "optimizer_steps": max_steps,
        "loss_bearing_tokens": total_tokens,
        "scheduler": training["scheduler"],
        "scheduler_horizon_steps": training["scheduler_horizon_steps"],
        "warmup_steps": training["warmup_steps"],
        "gradient_checkpointing": False,
        "cpu_offload": False,
        "step_metrics": metrics,
        "checkpoints": checkpoint_evidence,
        "initial_lora_tensor_sha256": initial_lora_sha256,
        "final_lora_tensor_sha256": final_tensor_sha256,
        "changed_lora_tensor_count": len(changed),
        "positive_lr_positive_gradient": positive_lr_positive_gradient,
        "base_parameter_fingerprint_before": base_before,
        "base_parameter_fingerprint_after": base_after,
        "base_parameters_unchanged": base_before == base_after,
        "base_restoration": True,
        "offline_reload": offline_reload,
        "trainable_inventory": inventory,
        "reload_inventory": reload_inventory,
        "rng_before_sha256": rng_before_sha256,
        "rng_after_sha256": rng_after_sha256,
        "rng_transition_sha256": canonical_sha256(
            {
                "before": rng_before_sha256,
                "after": rng_after_sha256,
            }
        ),
        "final_adapter_sha256": directory_sha256(final_path),
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_process_rss_bytes": peak_rss,
        "launch_evidence": launch,
    }
    result["training_sha256"] = canonical_sha256(result)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
