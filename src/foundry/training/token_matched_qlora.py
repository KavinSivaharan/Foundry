"""Whole-example token-budgeted QLoRA runtime for the frozen v2 protocol."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.config import QLoRARecipe
from foundry.training.qlora import (
    _format_and_tokenize,
    _load_quantized_model,
    _load_records,
    _software_versions,
    _verify_local_artifacts,
    directory_sha256,
)
from foundry.training.token_matched_config import load_token_matched_recipe
from foundry.training.token_matching import ScheduledOccurrence, ScheduledStep


def token_weighted_loss(mean_loss: Any, micro_tokens: int, step_tokens: int) -> Any:
    """Scale one mean microexample loss into the step's token-weighted mean."""

    if micro_tokens <= 0 or step_tokens <= 0 or micro_tokens > step_tokens:
        raise ValueError("invalid token-weighting counts")
    return mean_loss * (micro_tokens / step_tokens)


def _load_schedule(
    path: Path, *, expected_sha256: str, expected_steps: int
) -> tuple[ScheduledStep, ...]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("training schedule must be a list")
    payload = cast(list[object], value)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if hashlib.sha256(canonical.encode()).hexdigest() != expected_sha256:
        raise ValueError("training schedule hash differs from frozen recipe")
    steps: list[ScheduledStep] = []
    for expected_step, raw_step in enumerate(payload, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError("schedule step must be an object")
        step_map = cast(dict[str, object], raw_step)
        raw_occurrences = step_map.get("occurrences")
        if not isinstance(raw_occurrences, list):
            raise ValueError("schedule occurrences must be a list")
        occurrences: list[ScheduledOccurrence] = []
        for raw_occurrence in raw_occurrences:
            if not isinstance(raw_occurrence, dict):
                raise ValueError("scheduled occurrence must be an object")
            item = cast(dict[str, object], raw_occurrence)
            occurrences.append(
                ScheduledOccurrence(
                    synthetic_id=str(item["synthetic_id"]),
                    occurrence_index=int(cast(int, item["occurrence_index"])),
                    loss_bearing_tokens=int(cast(int, item["loss_bearing_tokens"])),
                )
            )
        step = ScheduledStep(
            step=int(cast(int, step_map["step"])),
            occurrences=tuple(occurrences),
            loss_bearing_tokens=int(cast(int, step_map["loss_bearing_tokens"])),
        )
        if step.step != expected_step or not step.occurrences:
            raise ValueError("schedule step order or membership differs")
        if step.loss_bearing_tokens != sum(item.loss_bearing_tokens for item in step.occurrences):
            raise ValueError("schedule step token total differs")
        steps.append(step)
    if len(steps) != expected_steps:
        raise ValueError("training schedule step count differs")
    return tuple(steps)


def _validate_schedule_records(
    schedule: tuple[ScheduledStep, ...],
    records: list[dict[str, Any]],
    tokenized: list[dict[str, list[int]]],
) -> tuple[dict[str, dict[str, list[int]]], dict[str, int]]:
    by_id: dict[str, dict[str, list[int]]] = {}
    loss_tokens: dict[str, int] = {}
    for record, encoded in zip(records, tokenized, strict=True):
        synthetic_id = str(record["synthetic_id"])
        if synthetic_id in by_id:
            raise ValueError("training records contain a duplicate ID")
        by_id[synthetic_id] = encoded
        loss_tokens[synthetic_id] = sum(label != -100 for label in encoded["labels"])
    occurrence_pairs: Counter[tuple[str, int]] = Counter()
    for step in schedule:
        for occurrence in step.occurrences:
            if occurrence.synthetic_id not in by_id:
                raise ValueError("schedule references an unknown training example")
            if loss_tokens[occurrence.synthetic_id] != occurrence.loss_bearing_tokens:
                raise ValueError("scheduled and tokenized loss-bearing counts differ")
            occurrence_pairs[(occurrence.synthetic_id, occurrence.occurrence_index)] += 1
    if any(count != 1 for count in occurrence_pairs.values()):
        raise ValueError("scheduled occurrence identity repeats")
    return by_id, loss_tokens


def _load_modules() -> dict[str, Any]:
    return {
        name: importlib.import_module(name)
        for name in (
            "accelerate",
            "bitsandbytes",
            "peft",
            "psutil",
            "torch",
            "transformers",
            "trl",
        )
    }


def _prepare_lora_model(model: Any, recipe: QLoRARecipe, peft: Any) -> tuple[Any, int, int]:
    model = peft.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=recipe.gradient_checkpointing
    )
    lora_config = peft.LoraConfig(
        r=recipe.rank,
        lora_alpha=recipe.alpha,
        lora_dropout=recipe.dropout,
        bias=recipe.bias,
        target_modules=list(recipe.target_modules),
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora_config)
    model.config.use_cache = False
    if not bool(getattr(model, "is_loaded_in_4bit", False)):
        raise RuntimeError("base model did not remain loaded in 4-bit mode")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in names):
        raise RuntimeError("only non-empty LoRA adapter parameters may be trainable")
    return (
        model,
        sum(int(parameter.numel()) for parameter in trainable),
        sum(int(parameter.numel()) for parameter in model.parameters()),
    )


def _validation_loss(model: Any, values: list[dict[str, list[int]]], torch: Any) -> float:
    model.eval()
    weighted = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for value in values:
            tokens = sum(label != -100 for label in value["labels"])
            inputs = {
                key: torch.tensor([item], device="cuda:0", dtype=torch.long)
                for key, item in value.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**inputs).loss
            weighted += float(loss.detach().float().item()) * tokens
            total_tokens += tokens
    model.train()
    return weighted / total_tokens


def _offline_adapter_reload(
    *,
    model_path: Path,
    adapter_path: Path,
    recipe: QLoRARecipe,
    modules: dict[str, Any],
) -> bool:
    torch = modules["torch"]
    peft = modules["peft"]
    base, _, _ = _load_quantized_model(model_path, recipe, modules)
    model = peft.PeftModel.from_pretrained(
        base, str(adapter_path), local_files_only=True, is_trainable=False
    )
    offloaded = [
        name for name, parameter in model.named_parameters() if parameter.device.type != "cuda"
    ]
    loaded = not offloaded
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return loaded


def run_token_matched_training(
    *,
    recipe_path: Path,
    model_path: Path,
    lock_path: Path,
    train_path: Path,
    validation_path: Path,
    group: str,
    output_dir: Path,
    summary_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    """Train one fresh adapter from an exact frozen variable-accumulation schedule."""

    if group not in {"generic_control", "targeted"}:
        raise ValueError("group must be generic_control or targeted")
    matched = load_token_matched_recipe(recipe_path)
    base_recipe = matched.base_recipe
    if not 1 <= max_steps <= matched.optimizer_steps:
        raise ValueError("max_steps must be between 1 and 200")
    if output_dir.exists():
        raise FileExistsError("fresh token-matched output directory already exists")
    _verify_local_artifacts(base_recipe, model_path, lock_path)
    arm = matched.arms[group]
    schedule = _load_schedule(
        arm.schedule_path,
        expected_sha256=arm.schedule_sha256,
        expected_steps=matched.optimizer_steps,
    )[:max_steps]
    records = _load_records(train_path, expected_group=group, expected_split="training")
    validation_records = _load_records(
        validation_path, expected_group=group, expected_split="synthetic_validation"
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(base_recipe.seed)
    modules = _load_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    bitsandbytes = modules["bitsandbytes"]
    process = modules["psutil"].Process()
    torch.manual_seed(base_recipe.seed)
    torch.cuda.manual_seed_all(base_recipe.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    setup_started = time.perf_counter()
    model, tokenizer, model_load_seconds = _load_quantized_model(model_path, base_recipe, modules)
    chat_hash = hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()
    if chat_hash != base_recipe.chat_template_sha256:
        raise ValueError("loaded tokenizer chat template differs")
    train_values, train_source_tokens, train_truncated = _format_and_tokenize(
        records, tokenizer, base_recipe
    )
    validation_values, validation_source_tokens, validation_truncated = _format_and_tokenize(
        validation_records, tokenizer, base_recipe
    )
    by_id, _ = _validate_schedule_records(schedule, records, train_values)
    model, trainable_parameters, total_parameters = _prepare_lora_model(model, base_recipe, peft)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = bitsandbytes.optim.PagedAdamW8bit(
        trainable,
        lr=base_recipe.learning_rate,
        weight_decay=base_recipe.weight_decay,
    )
    warmup_steps = math.ceil(base_recipe.warmup_ratio * matched.optimizer_steps)
    scheduler = transformers.get_scheduler(
        base_recipe.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=matched.optimizer_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=base_recipe.fp16)
    setup_seconds = time.perf_counter() - setup_started

    model.train()
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()
    step_losses: list[float] = []
    evaluation_losses: list[dict[str, float | int]] = []
    actual_tokens = 0
    actual_occurrences = 0
    scheduler_steps = 0
    for scheduled_step in schedule:
        step_loss = 0.0
        step_actual_tokens = 0
        for occurrence in scheduled_step.occurrences:
            value = by_id[occurrence.synthetic_id]
            micro_tokens = sum(label != -100 for label in value["labels"])
            inputs = {
                key: torch.tensor([item], device="cuda:0", dtype=torch.long)
                for key, item in value.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mean_loss = model(**inputs).loss
                weighted_loss = token_weighted_loss(
                    mean_loss, micro_tokens, scheduled_step.loss_bearing_tokens
                )
            if not bool(torch.isfinite(mean_loss).item()):
                raise RuntimeError("training loss is NaN or infinite")
            scaler.scale(weighted_loss).backward()
            step_loss += float(mean_loss.detach().float().item()) * (
                micro_tokens / scheduled_step.loss_bearing_tokens
            )
            step_actual_tokens += micro_tokens
            actual_occurrences += 1
        if step_actual_tokens != scheduled_step.loss_bearing_tokens:
            raise RuntimeError("actual optimizer-step token count differs from schedule")
        scaler.unscale_(optimizer)
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
            for parameter in trainable
        ):
            raise RuntimeError("training gradient is NaN or infinite")
        torch.nn.utils.clip_grad_norm_(trainable, base_recipe.max_gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        scheduler_steps += 1
        optimizer.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for parameter in trainable):
            raise RuntimeError("gradients were not cleared between optimizer steps")
        actual_tokens += step_actual_tokens
        step_losses.append(step_loss)
        if scheduled_step.step % base_recipe.evaluation_steps == 0:
            evaluation_losses.append(
                {
                    "step": scheduled_step.step,
                    "loss": _validation_loss(model, validation_values, torch),
                }
            )
    training_seconds = time.perf_counter() - training_started
    if scheduler_steps != max_steps or len(step_losses) != max_steps:
        raise RuntimeError("optimizer or scheduler step count differs")

    output_dir.mkdir(parents=True)
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    adapter_size = sum(item.stat().st_size for item in adapter_path.rglob("*") if item.is_file())
    adapter_sha256 = directory_sha256(adapter_path)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    peak_rss = int(process.memory_info().rss)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reload_ok = _offline_adapter_reload(
        model_path=model_path,
        adapter_path=adapter_path,
        recipe=base_recipe,
        modules=modules,
    )

    expected_tokens = sum(step.loss_bearing_tokens for step in schedule)
    occurrence_hash = hashlib.sha256(
        json.dumps(
            [
                (item.synthetic_id, item.occurrence_index)
                for step in schedule
                for item in step.occurrences
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "token_matched_parity_smoke" if max_steps == 4 else "final_adapter",
        "group": group,
        "recipe_id": matched.recipe_id,
        "recipe_sha256": matched.recipe_sha256,
        "selected_method": matched.selected_method,
        "base_recipe_sha256": matched.base_recipe_sha256,
        "base_model_id": base_recipe.base_model_id,
        "base_revision": base_recipe.base_revision,
        "requirements_lock_sha256": base_recipe.requirements_lock_sha256,
        "sft_format_sha256": base_recipe.sft_format_sha256,
        "training_schedule_sha256": arm.schedule_sha256,
        "scheduled_prefix_occurrence_sha256": occurrence_hash,
        "software_versions": _software_versions(modules),
        "cuda_runtime": str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "model_load_seconds": model_load_seconds,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "optimizer_steps": max_steps,
        "scheduler_steps": scheduler_steps,
        "examples_processed": actual_occurrences,
        "padded_training_tokens": actual_occurrences * base_recipe.max_sequence_length,
        "scheduled_loss_bearing_tokens": expected_tokens,
        "actual_loss_bearing_tokens": actual_tokens,
        "token_count_matches_schedule": actual_tokens == expected_tokens,
        "training_source_records": len(records),
        "training_source_nonpadding_tokens": train_source_tokens,
        "validation_source_records": len(validation_records),
        "validation_source_nonpadding_tokens": validation_source_tokens,
        "training_truncated_records": train_truncated,
        "validation_truncated_records": validation_truncated,
        "initial_token_weighted_step_loss": step_losses[0],
        "final_token_weighted_step_loss": step_losses[-1],
        "mean_token_weighted_step_loss": sum(step_losses) / len(step_losses),
        "losses_all_finite": all(math.isfinite(item) for item in step_losses),
        "gradients_all_finite": True,
        "evaluation_losses": evaluation_losses,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_system_rss_bytes": peak_rss,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "only_lora_trainable": True,
        "base_loaded_in_4bit": True,
        "adapter_size_bytes": adapter_size,
        "adapter_sha256": adapter_sha256,
        "adapter_offline_reload_ok": reload_ok,
        "development_benchmark_exposure_during_training": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--group", required=True, choices=("generic_control", "targeted"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--max-steps", required=True, type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_token_matched_training(
        recipe_path=args.recipe,
        model_path=args.model_path,
        lock_path=args.lock,
        train_path=args.train,
        validation_path=args.validation,
        group=args.group,
        output_dir=args.output_dir,
        summary_path=args.summary,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
