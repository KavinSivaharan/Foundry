"""Execute frozen vetted-corpus QLoRA smoke and training schedules."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import random
import time
from pathlib import Path
from typing import Any, cast

from foundry.phase2.launch_contract import validate_preimport
from foundry.training.config import assistant_only_v3_messages, canonical_sha256
from foundry.training.qlora import directory_sha256

SEED = 20260720
CHECKPOINTS = (16, 32, 64)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle]


def _schedule(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    value = cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))
    if canonical_sha256(value) != expected_sha256 or len(value) != 64:
        raise ValueError("frozen schedule differs")
    return value


def _messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if record["kind"] == "vetted":
        return assistant_only_v3_messages(
            str(record["question"]), str(record["assistant_completion"])
        )
    return [
        {"role": "system", "content": str(record["system_prompt"])},
        {"role": "user", "content": str(record["prompt"])},
        {"role": "assistant", "content": str(record["assistant_response"])},
    ]


def _tokenize(record: dict[str, Any], tokenizer: Any) -> dict[str, list[int]]:
    messages = _messages(record)
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    eos_positions = [
        index for index, value in enumerate(full_ids) if value == tokenizer.eos_token_id
    ]
    if not eos_positions:
        raise ValueError("assistant rendering lacks final EOS")
    eos = eos_positions[-1]
    if eos >= 512:
        raise ValueError("record exceeds maximum sequence length")
    input_ids = [*full_ids[: eos + 1], *([tokenizer.pad_token_id] * (511 - eos))]
    attention_mask = [*([1] * (eos + 1)), *([0] * (511 - eos))]
    labels = [-100] * 512
    labels[len(prefix_ids) : eos + 1] = input_ids[len(prefix_ids) : eos + 1]
    if any(value != -100 for value in labels[: len(prefix_ids)]):
        raise RuntimeError("system/user/header token became loss-bearing")
    if any(value != -100 for value in labels[eos + 1 :]):
        raise RuntimeError("padding or post-EOS token became loss-bearing")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _records(vetted_path: Path, replay_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in _jsonl(vetted_path):
        value = {**item, "kind": "vetted"}
        result[("vetted", str(item["source_id"]))] = value
    replay = cast(dict[str, Any], json.loads(replay_path.read_text(encoding="utf-8")))
    for item in cast(list[dict[str, Any]], replay["items"]):
        value = {**item, "kind": "replay"}
        result[("replay", str(item["id"]))] = value
    return result


def _load_model(model_path: Path, modules: dict[str, Any]) -> tuple[Any, Any]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("CPU or disk offload detected")
    return model, tokenizer


def _prepare(model: Any, peft: Any) -> Any:
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any("lora_" not in name for name in names):
        raise RuntimeError("trainable parameter set is not LoRA-only")
    return model


def _generate(model: Any, value: dict[str, list[int]], torch: Any) -> str:
    prompt_length = next(index for index, label in enumerate(value["labels"]) if label != -100)
    input_ids = torch.tensor([value["input_ids"][:prompt_length]], device="cuda")
    attention = torch.ones_like(input_ids)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            do_sample=False,
            max_new_tokens=24,
            pad_token_id=model.config.eos_token_id,
        )
    model.train()
    return hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest()


def _validation_loss(model: Any, values: list[dict[str, list[int]]], torch: Any) -> float:
    weighted = 0.0
    tokens = 0
    model.eval()
    with torch.inference_mode():
        for value in values:
            count = sum(label != -100 for label in value["labels"])
            inputs = {
                key: torch.tensor([items], device="cuda", dtype=torch.long)
                for key, items in value.items()
            }
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs).loss
            weighted += float(loss.detach().float().item()) * count
            tokens += count
    model.train()
    return weighted / tokens


def _save(model: Any, tokenizer: Any, path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return {
        "adapter_sha256": directory_sha256(path),
        "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
    }


def run(
    *,
    arm: str,
    max_steps: int,
    model_path: Path,
    vetted_path: Path,
    validation_path: Path,
    replay_path: Path,
    schedule_path: Path,
    schedule_sha256: str,
    output_directory: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Run a four-step smoke or complete 64-step frozen arm."""

    validate_preimport()
    modules = {
        name: importlib.import_module(name)
        for name in ("bitsandbytes", "peft", "psutil", "torch", "transformers")
    }
    torch = modules["torch"]
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if arm not in {"generic", "targeted"} or max_steps not in {4, 64}:
        raise ValueError("arm or max_steps is not authorized")
    if output_directory.exists():
        raise FileExistsError("output directory must be fresh")
    schedule = _schedule(schedule_path, schedule_sha256)[:max_steps]
    records = _records(vetted_path, replay_path)
    validation_records = [{**item, "kind": "vetted"} for item in _jsonl(validation_path)]
    started = time.perf_counter()
    model, tokenizer = _load_model(model_path, modules)
    values = {key: _tokenize(record, tokenizer) for key, record in records.items()}
    validation = [_tokenize(record, tokenizer) for record in validation_records]
    model = _prepare(model, modules["peft"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = modules["bitsandbytes"].optim.PagedAdamW8bit(trainable, lr=1e-5, weight_decay=0.0)
    scheduler = modules["transformers"].get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=4, num_training_steps=64
    )
    probe = values[("replay", sorted(key[1] for key in values if key[0] == "replay")[0])]
    base_generation_sha256 = _generate(model, probe, torch)
    model.train()
    losses: list[float] = []
    learning_rates: list[dict[str, float | int]] = []
    checkpoints: dict[str, Any] = {}
    total_tokens = 0
    for step in schedule:
        step_tokens = int(step["loss_bearing_tokens"])
        weighted_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for occurrence in cast(list[dict[str, Any]], step["occurrences"]):
            value = values[(str(occurrence["kind"]), str(occurrence["record_id"]))]
            actual = sum(label != -100 for label in value["labels"])
            if actual != int(occurrence["tokens"]):
                raise RuntimeError("scheduled assistant-token count differs")
            inputs = {
                key: torch.tensor([items], device="cuda", dtype=torch.long)
                for key, items in value.items()
            }
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs).loss
                scaled = loss * (actual / step_tokens)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite training loss")
            scaled.backward()
            weighted_loss += float(loss.detach().float().item()) * actual / step_tokens
        gradients = [item.grad for item in trainable if item.grad is not None]
        if not gradients or any(not bool(torch.isfinite(item).all().item()) for item in gradients):
            raise RuntimeError("missing or non-finite LoRA gradients")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        lr_before = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        learning_rates.append(
            {
                "step": int(step["step"]),
                "before_optimizer": lr_before,
                "after_scheduler": float(optimizer.param_groups[0]["lr"]),
            }
        )
        losses.append(weighted_loss)
        total_tokens += step_tokens
        if max_steps == 64 and int(step["step"]) in CHECKPOINTS:
            checkpoint = output_directory / f"checkpoint-{step['step']}" / "adapter"
            checkpoints[str(step["step"])] = {
                **_save(model, tokenizer, checkpoint),
                "validation_loss": _validation_loss(model, validation, torch),
            }
        print(json.dumps({"arm": arm, "completed_step": step["step"]}), flush=True)
    torch.cuda.synchronize()
    if max_steps == 4:
        final_path = output_directory / "adapter"
        checkpoints["4"] = _save(model, tokenizer, final_path)
    else:
        final_path = output_directory / "checkpoint-64" / "adapter"
    with model.disable_adapter():
        restored_generation_sha256 = _generate(model, probe, torch)
    if restored_generation_sha256 != base_generation_sha256:
        raise RuntimeError("adapter-disabled base generation did not restore exactly")
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_rss = int(modules["psutil"].Process().memory_info().rss)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    base, _ = _load_model(model_path, modules)
    reloaded = modules["peft"].PeftModel.from_pretrained(
        base, str(final_path), local_files_only=True, is_trainable=False
    )
    offline_reload = all(parameter.device.type == "cuda" for parameter in reloaded.parameters())
    del reloaded, base
    gc.collect()
    torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "schema_version": 1,
        "arm": arm,
        "run_kind": "compatibility_smoke" if max_steps == 4 else "complete_training",
        "optimizer_steps": max_steps,
        "loss_bearing_tokens": total_tokens,
        "losses": losses,
        "learning_rates": learning_rates,
        "checkpoints": checkpoints,
        "offline_reload": offline_reload,
        "base_restoration": True,
        "base_generation_sha256": base_generation_sha256,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_process_rss_bytes": peak_rss,
        "runtime_seconds": time.perf_counter() - started,
        "schedule_sha256": schedule_sha256,
        "sealed_final_accessed": False,
        "gsm1k_runs": 0,
    }
    result["result_sha256"] = canonical_sha256(result)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vetted-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--schedule-path", type=Path, required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
