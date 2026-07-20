"""Token-matched QLoRA runtime for assistant-only SFT v3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from foundry.training.assistant_only import format_and_tokenize_assistant_only
from foundry.training.assistant_only_config import load_assistant_only_recipe
from foundry.training.qlora import (
    _load_quantized_model,
    _load_records,
    _software_versions,
    _verify_local_artifacts,
    directory_sha256,
)
from foundry.training.token_matched_qlora import (
    _load_modules,
    _load_schedule,
    _offline_adapter_reload,
    _prepare_lora_model,
    _validate_schedule_records,
    _validation_loss,
    token_weighted_loss,
)


def _save_adapter(model: Any, tokenizer: Any, path: Path) -> dict[str, Any]:
    path.mkdir(parents=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return {
        "path": path.as_posix(),
        "adapter_sha256": directory_sha256(path),
        "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
    }


def run_assistant_only_training(
    *,
    recipe_path: Path,
    learning_rate: float,
    model_path: Path,
    lock_path: Path,
    train_path: Path,
    validation_path: Path,
    group: str,
    output_dir: Path,
    summary_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    """Train one fresh v3 adapter from the untouched base and frozen schedule."""

    if group not in {"generic_control", "targeted"}:
        raise ValueError("group must be generic_control or targeted")
    recipe = load_assistant_only_recipe(recipe_path)
    execution_sha256 = recipe.execution_sha256(learning_rate)
    base_recipe = recipe.base_recipe
    if not 1 <= max_steps <= recipe.optimizer_steps:
        raise ValueError("max_steps must be between 1 and 200")
    if max_steps not in {recipe.retention_smoke_steps, recipe.optimizer_steps}:
        raise ValueError("only the 32-step smoke or complete 200-step run is approved")
    if output_dir.exists():
        raise FileExistsError("fresh assistant-only output directory already exists")
    _verify_local_artifacts(base_recipe, model_path, lock_path)
    arm = recipe.arms[group]
    complete_schedule = _load_schedule(
        arm.schedule_path,
        expected_sha256=arm.schedule_sha256,
        expected_steps=recipe.optimizer_steps,
    )
    schedule = complete_schedule[:max_steps]
    records = _load_records(train_path, expected_group=group, expected_split="training")
    validation_records = _load_records(
        validation_path,
        expected_group=group,
        expected_split="synthetic_validation",
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
        raise ValueError("loaded Qwen chat template differs")
    train_values, train_source_tokens, train_truncated, train_evidence = (
        format_and_tokenize_assistant_only(
            records, tokenizer, max_length=base_recipe.max_sequence_length
        )
    )
    validation_values, validation_source_tokens, validation_truncated, validation_evidence = (
        format_and_tokenize_assistant_only(
            validation_records,
            tokenizer,
            max_length=base_recipe.max_sequence_length,
        )
    )
    by_id, _ = _validate_schedule_records(schedule, records, train_values)
    model, trainable_parameters, total_parameters = _prepare_lora_model(model, base_recipe, peft)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = bitsandbytes.optim.PagedAdamW8bit(
        trainable,
        lr=learning_rate,
        weight_decay=base_recipe.weight_decay,
    )
    warmup_steps = math.ceil(base_recipe.warmup_ratio * recipe.optimizer_steps)
    scheduler = transformers.get_scheduler(
        base_recipe.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=recipe.optimizer_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=base_recipe.fp16)
    setup_seconds = time.perf_counter() - setup_started

    output_dir.mkdir(parents=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()
    step_losses: list[float] = []
    validation_losses: list[dict[str, float | int]] = []
    checkpoint_evidence: dict[str, dict[str, Any]] = {}
    actual_tokens = 0
    actual_occurrences = 0
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
        optimizer.zero_grad(set_to_none=True)
        actual_tokens += step_actual_tokens
        step_losses.append(step_loss)
        if scheduled_step.step % base_recipe.evaluation_steps == 0:
            validation_losses.append(
                {
                    "step": scheduled_step.step,
                    "loss": _validation_loss(model, validation_values, torch),
                }
            )
        if max_steps == recipe.optimizer_steps and scheduled_step.step in recipe.checkpoints:
            checkpoint_evidence[str(scheduled_step.step)] = _save_adapter(
                model,
                tokenizer,
                output_dir / f"checkpoint-{scheduled_step.step}" / "adapter",
            )
        if scheduled_step.step % 10 == 0 or scheduled_step.step == max_steps:
            print(
                json.dumps(
                    {
                        "assistant_only_training_group": group,
                        "completed_steps": scheduled_step.step,
                        "total_steps": max_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - training_started
    final_validation_loss = _validation_loss(model, validation_values, torch)
    if not math.isfinite(final_validation_loss):
        raise RuntimeError("final synthetic-validation loss is not finite")
    if max_steps == recipe.retention_smoke_steps:
        final_adapter_path = output_dir / "adapter"
        checkpoint_evidence[str(max_steps)] = _save_adapter(model, tokenizer, final_adapter_path)
    else:
        final_adapter_path = output_dir / f"checkpoint-{max_steps}" / "adapter"
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    peak_rss = int(process.memory_info().rss)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    offline_reload_ok = _offline_adapter_reload(
        model_path=model_path,
        adapter_path=final_adapter_path,
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
        "run_kind": "assistant_only_retention_smoke"
        if max_steps == recipe.retention_smoke_steps
        else "assistant_only_full_training",
        "group": group,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.recipe_sha256,
        "execution_sha256": execution_sha256,
        "learning_rate": learning_rate,
        "assistant_only_format_sha256": recipe.format_sha256,
        "base_recipe_sha256": recipe.base_recipe_sha256,
        "base_revision": base_recipe.base_revision,
        "requirements_lock_sha256": base_recipe.requirements_lock_sha256,
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
        "scheduler_steps": max_steps,
        "examples_processed": actual_occurrences,
        "scheduled_loss_bearing_tokens": expected_tokens,
        "actual_loss_bearing_tokens": actual_tokens,
        "token_count_matches_schedule": actual_tokens == expected_tokens,
        "training_source_records": len(records),
        "training_source_assistant_tokens": train_source_tokens,
        "validation_source_records": len(validation_records),
        "validation_source_assistant_tokens": validation_source_tokens,
        "training_truncated_records": train_truncated,
        "validation_truncated_records": validation_truncated,
        "zero_system_user_header_loss_tokens": all(
            item.system_user_header_loss_tokens == 0 for item in train_evidence
        ),
        "zero_padding_or_post_eos_loss_tokens": all(
            item.padding_loss_tokens == 0 and item.post_eos_loss_tokens == 0
            for item in (*train_evidence, *validation_evidence)
        ),
        "exact_one_final_answer_all_records": all(
            item.final_answer_line_count == 1 for item in (*train_evidence, *validation_evidence)
        ),
        "initial_token_weighted_step_loss": step_losses[0],
        "final_token_weighted_step_loss": step_losses[-1],
        "mean_token_weighted_step_loss": sum(step_losses) / len(step_losses),
        "losses_all_finite": all(math.isfinite(item) for item in step_losses),
        "gradients_all_finite": True,
        "validation_losses": validation_losses,
        "final_validation_loss": final_validation_loss,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_system_rss_bytes": peak_rss,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "only_lora_trainable": True,
        "base_loaded_in_4bit": True,
        "checkpoints": checkpoint_evidence,
        "final_adapter_sha256": directory_sha256(final_adapter_path),
        "adapter_offline_reload_ok": offline_reload_ok,
        "development_benchmark_exposure_during_training": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run an approved assistant-only smoke or complete training arm."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--group", choices=("generic_control", "targeted"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    result = run_assistant_only_training(
        recipe_path=args.recipe,
        learning_rate=args.learning_rate,
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


if __name__ == "__main__":
    main()
