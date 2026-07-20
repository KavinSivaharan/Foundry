"""Offline, recipe-frozen QLoRA training runtime for matched signal data."""

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
from pathlib import Path
from typing import Any, cast

from foundry.training.config import QLoRARecipe, load_qlora_recipe, sft_messages


def file_sha256(path: Path) -> str:
    """Hash one file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash file names and contents in stable relative-path order."""

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def _load_records(path: Path, *, expected_group: str, expected_split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("training record must be an object")
            record = cast(dict[str, Any], value)
            if (
                record.get("group") != expected_group
                or record.get("future_split") != expected_split
                or record.get("final_decision") != "accepted"
                or record.get("verifier_agreement") is not True
            ):
                raise ValueError("training record violates group, split, or verification contract")
            records.append(record)
    expected_count = 450 if expected_split == "training" else 50
    if len(records) != expected_count:
        raise ValueError(f"{expected_group} {expected_split} requires {expected_count} records")
    if len({str(record["synthetic_id"]) for record in records}) != len(records):
        raise ValueError("training records contain duplicate synthetic IDs")
    return records


def _verify_local_artifacts(recipe: QLoRARecipe, model_path: Path, lock_path: Path) -> None:
    required = {
        "tokenizer.json": recipe.tokenizer_sha256,
        "tokenizer_config.json": recipe.tokenizer_config_sha256,
    }
    for name, expected in required.items():
        if file_sha256(model_path / name) != expected:
            raise ValueError(f"local {name} differs from the frozen recipe")
    if file_sha256(lock_path) != recipe.requirements_lock_sha256:
        raise ValueError("training dependency lock differs from the frozen recipe")


def _format_and_tokenize(
    records: list[dict[str, Any]], tokenizer: Any, recipe: QLoRARecipe
) -> tuple[list[dict[str, list[int]]], int, int]:
    tokenized: list[dict[str, list[int]]] = []
    nonpadding_total = 0
    truncated = 0
    for record in records:
        messages = sft_messages(
            str(record["rendered_question"]), str(record["training_completion"])
        )
        text = cast(
            str,
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
        )
        original = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=recipe.max_sequence_length,
            padding="max_length",
        )
        input_ids = cast(list[int], encoded["input_ids"])
        attention_mask = cast(list[int], encoded["attention_mask"])
        labels = [
            token if mask else -100 for token, mask in zip(input_ids, attention_mask, strict=True)
        ]
        nonpadding_total += sum(attention_mask)
        truncated += int(len(original) > recipe.max_sequence_length)
        tokenized.append(
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        )
    return tokenized, nonpadding_total, truncated


def _software_versions(modules: dict[str, Any]) -> dict[str, str]:
    return {name: str(module.__version__) for name, module in modules.items()}


def _make_counting_trainer(transformers: Any) -> type[Any]:
    class CountingTrainer(transformers.Trainer):  # type: ignore[misc]
        training_nonpadding_tokens = 0

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            if model.training and "attention_mask" in inputs:
                self.training_nonpadding_tokens += int(inputs["attention_mask"].sum().item())
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    return CountingTrainer


def _load_quantized_model(
    model_path: Path, recipe: QLoRARecipe, modules: dict[str, Any]
) -> tuple[Any, Any, float]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=recipe.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=recipe.load_in_4bit,
        bnb_4bit_quant_type=recipe.quantization_type,
        bnb_4bit_use_double_quant=recipe.double_quantization,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=recipe.trust_remote_code,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    offloaded = [name for name, param in model.named_parameters() if param.device.type != "cuda"]
    if offloaded:
        raise RuntimeError(f"CPU or disk offloading is prohibited: {offloaded[:3]}")
    return model, tokenizer, time.perf_counter() - started


def run_training(
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
    limit: int | None,
    compatibility_inference: bool,
) -> dict[str, Any]:
    """Run one exact training job and emit only content-free measurements."""

    if group not in {"targeted", "generic_control"}:
        raise ValueError("group must be targeted or generic_control")
    recipe = load_qlora_recipe(recipe_path)
    if not 1 <= max_steps <= recipe.max_optimizer_steps:
        raise ValueError("max_steps must be within the frozen 200-step recipe")
    _verify_local_artifacts(recipe, model_path, lock_path)
    train_records = _load_records(train_path, expected_group=group, expected_split="training")
    validation_records = _load_records(
        validation_path, expected_group=group, expected_split="synthetic_validation"
    )
    if limit is not None:
        if not 1 <= limit <= 128:
            raise ValueError("compatibility smoke limit must be between 1 and 128")
        train_records = train_records[:limit]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(recipe.seed)
    modules = {
        name: importlib.import_module(name)
        for name in (
            "accelerate",
            "bitsandbytes",
            "datasets",
            "peft",
            "torch",
            "transformers",
            "trl",
        )
    }
    torch = modules["torch"]
    datasets = modules["datasets"]
    peft = modules["peft"]
    transformers = modules["transformers"]
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    process = importlib.import_module("psutil").Process()
    setup_started = time.perf_counter()
    model, tokenizer, model_load_seconds = _load_quantized_model(model_path, recipe, modules)
    actual_chat_hash = hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest()
    if actual_chat_hash != recipe.chat_template_sha256:
        raise ValueError("loaded tokenizer chat template differs from the frozen recipe")
    train_values, train_source_tokens, train_truncated = _format_and_tokenize(
        train_records, tokenizer, recipe
    )
    validation_values, validation_source_tokens, validation_truncated = _format_and_tokenize(
        validation_records, tokenizer, recipe
    )
    train_dataset = datasets.Dataset.from_list(train_values)
    validation_dataset = datasets.Dataset.from_list(validation_values)

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
    base_loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    if not base_loaded_in_4bit:
        raise RuntimeError("base model did not remain loaded in 4-bit mode")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("only non-empty LoRA adapter parameters may be trainable")
    trainable_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(int(parameter.numel()) for parameter in model.parameters())

    arguments = transformers.TrainingArguments(
        output_dir=str(output_dir / "trainer_state"),
        per_device_train_batch_size=recipe.micro_batch_size,
        per_device_eval_batch_size=recipe.micro_batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        max_steps=max_steps,
        optim=recipe.optimizer,
        learning_rate=recipe.learning_rate,
        lr_scheduler_type=recipe.scheduler,
        warmup_ratio=recipe.warmup_ratio,
        weight_decay=recipe.weight_decay,
        max_grad_norm=recipe.max_gradient_norm,
        gradient_checkpointing=recipe.gradient_checkpointing,
        fp16=recipe.fp16,
        bf16=recipe.bf16,
        logging_steps=recipe.logging_steps,
        eval_strategy="steps",
        eval_steps=recipe.evaluation_steps,
        save_strategy="no",
        seed=recipe.seed,
        data_seed=recipe.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=False,
    )
    trainer_type = _make_counting_trainer(transformers)
    trainer = trainer_type(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )
    setup_seconds = time.perf_counter() - setup_started
    training_started = time.perf_counter()
    result = trainer.train()
    training_seconds = time.perf_counter() - training_started
    completed_steps = int(trainer.state.global_step)
    training_nonpadding_tokens = int(trainer.training_nonpadding_tokens)
    if completed_steps != max_steps:
        raise RuntimeError("training stopped before the frozen optimizer-step count")
    losses = [
        float(item["loss"])
        for item in trainer.state.log_history
        if "loss" in item and math.isfinite(float(item["loss"]))
    ]
    all_reported_losses = [
        float(value)
        for item in trainer.state.log_history
        for key, value in item.items()
        if key in {"loss", "eval_loss"} and isinstance(value, int | float)
    ]
    if not losses or not all(math.isfinite(value) for value in all_reported_losses):
        raise RuntimeError("training loss is missing, NaN, or infinite")
    evaluation_losses = [
        float(item["eval_loss"])
        for item in trainer.state.log_history
        if "eval_loss" in item and math.isfinite(float(item["eval_loss"]))
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    adapter_size = sum(item.stat().st_size for item in adapter_path.rglob("*") if item.is_file())
    adapter_sha256 = directory_sha256(adapter_path)

    inference_ok = False
    inference_text_sha256: str | None = None
    if compatibility_inference:
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded_base, reloaded_tokenizer, _ = _load_quantized_model(model_path, recipe, modules)
        reloaded = peft.PeftModel.from_pretrained(
            reloaded_base, str(adapter_path), local_files_only=True, is_trainable=False
        )
        reloaded.eval()
        prompt = reloaded_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "State one friendly greeting."},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda:0")
        with torch.inference_mode():
            generated = reloaded.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=False,
                max_new_tokens=24,
                pad_token_id=reloaded_tokenizer.pad_token_id,
                eos_token_id=reloaded_tokenizer.eos_token_id,
            )
        decoded = cast(
            str,
            reloaded_tokenizer.decode(generated[0, prompt.shape[-1] :], skip_special_tokens=True),
        )
        inference_ok = bool(decoded.strip())
        inference_text_sha256 = hashlib.sha256(decoded.encode("utf-8")).hexdigest()

    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "qlora_compatibility_smoke" if compatibility_inference else "final_adapter",
        "group": group,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.recipe_sha256,
        "base_model_id": recipe.base_model_id,
        "base_revision": recipe.base_revision,
        "requirements_lock_sha256": recipe.requirements_lock_sha256,
        "sft_format_sha256": recipe.sft_format_sha256,
        "software_versions": _software_versions(modules),
        "cuda_runtime": str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "model_load_seconds": model_load_seconds,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "optimizer_steps": completed_steps,
        "examples_processed": max_steps * recipe.effective_batch_size,
        "padded_training_tokens": max_steps
        * recipe.effective_batch_size
        * recipe.max_sequence_length,
        "nonpadding_training_tokens": training_nonpadding_tokens,
        "training_source_records": len(train_records),
        "training_source_nonpadding_tokens": train_source_tokens,
        "validation_source_records": len(validation_records),
        "validation_source_nonpadding_tokens": validation_source_tokens,
        "training_truncated_records": train_truncated,
        "validation_truncated_records": validation_truncated,
        "initial_logged_loss": losses[0],
        "final_logged_loss": losses[-1],
        "train_runtime_reported": float(result.metrics.get("train_runtime", training_seconds)),
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_system_rss_bytes": int(process.memory_info().rss),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "only_lora_trainable": True,
        "base_loaded_in_4bit": base_loaded_in_4bit,
        "forward_backward_optimizer_succeeded": completed_steps == max_steps,
        "losses_all_finite": all(math.isfinite(value) for value in all_reported_losses),
        "final_evaluation_loss": evaluation_losses[-1] if evaluation_losses else None,
        "adapter_size_bytes": adapter_size,
        "adapter_sha256": adapter_sha256,
        "adapter_reload_inference_ok": inference_ok if compatibility_inference else None,
        "adapter_reload_inference_text_sha256": inference_text_sha256,
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
    parser.add_argument("--group", required=True, choices=("targeted", "generic_control"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compatibility-inference", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_training(
        recipe_path=args.recipe,
        model_path=args.model_path,
        lock_path=args.lock,
        train_path=args.train,
        validation_path=args.validation,
        group=args.group,
        output_dir=args.output_dir,
        summary_path=args.summary,
        max_steps=args.max_steps,
        limit=args.limit,
        compatibility_inference=args.compatibility_inference,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
