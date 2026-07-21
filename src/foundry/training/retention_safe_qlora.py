"""Deterministic retention-safe ladder and full-training QLoRA runtime."""

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
from foundry.training.concise_v4 import format_and_tokenize_concise_v4
from foundry.training.config import (
    assistant_only_v3_format_contract_sha256,
    concise_assistant_v4_format_contract_sha256,
    load_qlora_recipe,
)
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

ALLOWED_LADDER = {("v3", 5e-5), ("v4", 5e-5), ("v4", 2e-5), ("v4", 1e-5)}


def _tokenize(
    records: list[dict[str, Any]], tokenizer: Any, *, format_id: str, max_length: int
) -> tuple[list[dict[str, list[int]]], int, int, tuple[Any, ...]]:
    if format_id == "v3":
        return format_and_tokenize_assistant_only(records, tokenizer, max_length=max_length)
    if format_id == "v4":
        return format_and_tokenize_concise_v4(records, tokenizer, max_length=max_length)
    raise ValueError("unknown retention-safe target format")


def _format_hash(format_id: str) -> str:
    if format_id == "v3":
        return assistant_only_v3_format_contract_sha256()
    if format_id == "v4":
        return concise_assistant_v4_format_contract_sha256()
    raise ValueError("unknown retention-safe target format")


def _save_adapter(model: Any, tokenizer: Any, path: Path) -> dict[str, Any]:
    path.mkdir(parents=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return {
        "path": path.as_posix(),
        "adapter_sha256": directory_sha256(path),
        "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
    }


def run_retention_safe_training(
    *,
    base_recipe_path: Path,
    format_id: str,
    learning_rate: float,
    model_path: Path,
    lock_path: Path,
    train_path: Path,
    validation_path: Path,
    group: str,
    schedule_path: Path,
    schedule_sha256: str,
    total_steps: int,
    checkpoints: tuple[int, ...],
    run_id: str,
    protocol_sha256: str,
    output_dir: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Train one arm from the untouched base and save every frozen checkpoint."""

    if group not in {"generic_control", "targeted"}:
        raise ValueError("unknown training arm")
    if total_steps == 32 and (format_id, learning_rate) not in ALLOWED_LADDER:
        raise ValueError("training choice is outside the predeclared adaptation ladder")
    if total_steps not in {32, 200}:
        raise ValueError("only the 32-step ladder or 200-step full run is allowed")
    expected_checkpoints = (8, 16, 24, 32) if total_steps == 32 else (25, 50, 100, 150, 200)
    if checkpoints != expected_checkpoints:
        raise ValueError("checkpoint steps differ from the frozen run kind")
    if output_dir.exists():
        raise FileExistsError("fresh retention-safe output directory already exists")

    recipe = load_qlora_recipe(base_recipe_path)
    _verify_local_artifacts(recipe, model_path, lock_path)
    schedule = _load_schedule(
        schedule_path, expected_sha256=schedule_sha256, expected_steps=total_steps
    )
    records = _load_records(train_path, expected_group=group, expected_split="training")
    validation_records = _load_records(
        validation_path, expected_group=group, expected_split="synthetic_validation"
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(recipe.seed)
    modules = _load_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    bitsandbytes = modules["bitsandbytes"]
    process = modules["psutil"].Process()
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    setup_started = time.perf_counter()
    model, tokenizer, model_load_seconds = _load_quantized_model(model_path, recipe, modules)
    chat_hash = hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()
    if chat_hash != recipe.chat_template_sha256:
        raise ValueError("loaded Qwen chat template differs")
    train_values, train_source_tokens, train_truncated, train_evidence = _tokenize(
        records,
        tokenizer,
        format_id=format_id,
        max_length=recipe.max_sequence_length,
    )
    validation_values, validation_source_tokens, validation_truncated, validation_evidence = (
        _tokenize(
            validation_records,
            tokenizer,
            format_id=format_id,
            max_length=recipe.max_sequence_length,
        )
    )
    by_id, _ = _validate_schedule_records(schedule, records, train_values)
    model, trainable_parameters, total_parameters = _prepare_lora_model(model, recipe, peft)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = bitsandbytes.optim.PagedAdamW8bit(
        trainable, lr=learning_rate, weight_decay=recipe.weight_decay
    )
    warmup_steps = math.ceil(recipe.warmup_ratio * total_steps)
    scheduler = transformers.get_scheduler(
        recipe.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=recipe.fp16)
    setup_seconds = time.perf_counter() - setup_started

    output_dir.mkdir(parents=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()
    step_losses: list[float] = []
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
        torch.nn.utils.clip_grad_norm_(trainable, recipe.max_gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        actual_tokens += step_actual_tokens
        step_losses.append(step_loss)
        if scheduled_step.step in checkpoints:
            adapter = _save_adapter(
                model,
                tokenizer,
                output_dir / f"checkpoint-{scheduled_step.step}" / "adapter",
            )
            checkpoint_evidence[str(scheduled_step.step)] = {
                **adapter,
                "cumulative_actual_loss_bearing_tokens": actual_tokens,
                "training_loss": step_loss,
                "synthetic_validation_loss": _validation_loss(model, validation_values, torch),
            }
        if scheduled_step.step % 8 == 0 or scheduled_step.step == total_steps:
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "group": group,
                        "completed_steps": scheduled_step.step,
                        "total_steps": total_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - training_started
    final_adapter_path = output_dir / f"checkpoint-{total_steps}" / "adapter"
    final_validation_loss = float(
        checkpoint_evidence[str(total_steps)]["synthetic_validation_loss"]
    )
    if not math.isfinite(final_validation_loss):
        raise RuntimeError("final synthetic-validation loss is not finite")
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    peak_rss = int(process.memory_info().rss)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    offline_reload_ok = _offline_adapter_reload(
        model_path=model_path,
        adapter_path=final_adapter_path,
        recipe=recipe,
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
        "run_id": run_id,
        "run_kind": "retention_safe_ladder" if total_steps == 32 else "retention_safe_full",
        "protocol_sha256": protocol_sha256,
        "group": group,
        "format_id": format_id,
        "format_sha256": _format_hash(format_id),
        "base_recipe_sha256": recipe.recipe_sha256,
        "base_revision": recipe.base_revision,
        "learning_rate": learning_rate,
        "seed": recipe.seed,
        "training_schedule_sha256": schedule_sha256,
        "scheduled_occurrence_sha256": occurrence_hash,
        "software_versions": _software_versions(modules),
        "cuda_runtime": str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "model_load_seconds": model_load_seconds,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "optimizer_steps": total_steps,
        "scheduler_steps": total_steps,
        "warmup_steps": warmup_steps,
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


def _parse_checkpoints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--format", choices=("v3", "v4"), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--group", choices=("generic_control", "targeted"), required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = run_retention_safe_training(
        base_recipe_path=args.base_recipe,
        format_id=args.format,
        learning_rate=args.learning_rate,
        model_path=args.model_path,
        lock_path=args.lock,
        train_path=args.train,
        validation_path=args.validation,
        group=args.group,
        schedule_path=args.schedule,
        schedule_sha256=args.schedule_sha256,
        total_steps=args.total_steps,
        checkpoints=_parse_checkpoints(args.checkpoints),
        run_id=args.run_id,
        protocol_sha256=args.protocol_sha256,
        output_dir=args.output_dir,
        summary_path=args.summary,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
