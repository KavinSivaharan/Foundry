"""Run V1-equivalent replay cross-entropy plus token-level KL training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_recipe
from foundry.phase2.kl_objective import (
    assistant_shift_mask,
    forward_token_kl,
    replay_total_loss,
)
from foundry.phase2.launch_contract import validate_postimport, validate_preimport
from foundry.training.config import assistant_only_v3_messages, canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

SEED = 20260720
COEFFICIENTS = (0.01, 0.03, 0.10, 0.30)
CHECKPOINTS = (16, 32, 64)
MODEL_REVISION = kl_recipe.MODEL_REVISION
RECIPE_SHA256 = "3bc9fbcdb44dc53b12149d3832153a7fce90d0c7839868b5ec6c3b10939e7862"
RECIPE_DECISION_SHA256 = "b03dfc9d6f66843f2a83e9b4ff5b82e133fb9c6a4a27ab1783d64991a0f7d118"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle]


def _read_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _validate_recipe(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    supplied = value.get("final_recipe_decision_sha256")
    payload = {name: item for name, item in value.items() if name != "final_recipe_decision_sha256"}
    if (
        supplied != canonical_sha256(payload)
        or supplied != RECIPE_DECISION_SHA256
        or value["canonical_lora_configuration"]["canonical_kl_lora_configuration_sha256"]
        != RECIPE_SHA256
    ):
        raise ValueError("canonical V1-equivalent KL recipe differs")
    return value


def _schedule(path: Path, expected_sha256: str, max_steps: int) -> list[dict[str, Any]]:
    value = cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))
    if canonical_sha256(value) != expected_sha256 or len(value) != 64:
        raise ValueError("frozen V1 schedule differs")
    prefix = value[:max_steps]
    expected_tokens = 16_000 if max_steps == 16 else 64_000
    if sum(int(step["loss_bearing_tokens"]) for step in prefix) != expected_tokens:
        raise ValueError("frozen V1 schedule-prefix token count differs")
    return prefix


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
        raise RuntimeError("system, user, or assistant-header token became supervised")
    if any(value != -100 for value in labels[eos + 1 :]):
        raise RuntimeError("padding or post-EOS token became supervised")
    if labels[eos] != tokenizer.eos_token_id:
        raise RuntimeError("final assistant EOS is not supervised")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _records(vetted_path: Path, replay_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in _jsonl(vetted_path):
        result[("vetted", str(item["source_id"]))] = {**item, "kind": "vetted"}
    replay = _read_object(replay_path)
    for item in cast(list[dict[str, Any]], replay["items"]):
        result[("replay", str(item["id"]))] = {**item, "kind": "replay"}
    return result


def _load_base(model_path: Path, modules: dict[str, Any]) -> tuple[Any, Any]:
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
    return model


def _normalized_lora_inventory(model: Any) -> dict[str, Any]:
    shapes = [
        {
            "name": name.replace(".default", ""),
            "shape": list(parameter.shape),
        }
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    shapes.sort(key=lambda value: str(value["name"]))
    result = {
        "trainable_tensor_count": len(shapes),
        "trainable_parameter_count": sum(
            math.prod(cast(list[int], value["shape"])) for value in shapes
        ),
        "tensor_inventory_sha256": canonical_sha256(shapes),
        "only_lora_tensors": all("lora_" in str(value["name"]) for value in shapes),
    }
    expected = {
        "trainable_tensor_count": kl_recipe.EXPECTED_TRAINABLE_TENSORS,
        "trainable_parameter_count": kl_recipe.EXPECTED_TRAINABLE_PARAMETERS,
        "tensor_inventory_sha256": (
            "d3edea65d6d09226eb743182474ea51b2af1c0f94b163812ce67913ffc865e78"
        ),
        "only_lora_tensors": True,
    }
    if result != expected:
        raise RuntimeError("normalized LoRA inventory differs from historical V1")
    return result


def _active_trainable_parameters(model: Any) -> list[Any]:
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if (
        len(named) != kl_recipe.EXPECTED_TRAINABLE_TENSORS
        or any("lora_" not in name for name, _ in named)
        or sum(parameter.numel() for _, parameter in named)
        != kl_recipe.EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("active trainable parameters differ from historical V1 LoRA")
    return [parameter for _, parameter in named]


def _base_parameter_fingerprint(model: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            continue
        tensor = parameter.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("base parameter fingerprint selected no tensors")
    return digest.hexdigest()


def _inputs(value: dict[str, list[int]], torch: Any) -> dict[str, Any]:
    return {
        key: torch.tensor([items], device="cuda", dtype=torch.long) for key, items in value.items()
    }


def _reference_logits(model: Any, inputs: dict[str, Any], torch: Any) -> Any:
    mask = assistant_shift_mask(inputs["labels"])
    with model.disable_adapter(), torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
    return outputs.logits[:, :-1, :].detach(), mask


def _loss(
    model: Any,
    value: dict[str, list[int]],
    kind: str,
    coefficient: float,
    torch: Any,
) -> tuple[Any, Any, Any]:
    inputs = _inputs(value, torch)
    reference_logits = None
    mask = None
    if kind == "replay":
        reference_logits, mask = _reference_logits(model, inputs, torch)
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(**inputs, use_cache=False)
        cross_entropy = outputs.loss
    if kind == "replay":
        if reference_logits is None or mask is None:
            raise RuntimeError("replay reference logits were not produced")
        token_kl = forward_token_kl(
            reference_logits,
            outputs.logits[:, :-1, :],
            mask,
            torch,
        )
        total = replay_total_loss(cross_entropy, token_kl, coefficient)
    else:
        token_kl = torch.zeros((), device="cuda", dtype=torch.float32)
        total = cross_entropy
    return cross_entropy, token_kl, total


def _generate(model: Any, value: dict[str, list[int]], torch: Any) -> str:
    prompt_length = next(index for index, label in enumerate(value["labels"]) if label != -100)
    input_ids = torch.tensor([value["input_ids"][:prompt_length]], device="cuda")
    attention = torch.ones_like(input_ids)
    prior = model.training
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            do_sample=False,
            max_new_tokens=24,
            pad_token_id=model.config.eos_token_id,
        )
    model.train(prior)
    return hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest()


def _measure(
    model: Any,
    schedule: list[dict[str, Any]],
    values: dict[tuple[str, str], dict[str, list[int]]],
    validation: list[dict[str, list[int]]],
    torch: Any,
) -> dict[str, Any]:
    prior = model.training
    model.eval()
    ce_sums: defaultdict[str, float] = defaultdict(float)
    token_sums: Counter[str] = Counter()
    replay_kl_sum = 0.0
    with torch.inference_mode():
        for step in schedule:
            for occurrence in cast(list[dict[str, Any]], step["occurrences"]):
                kind = str(occurrence["kind"])
                tokens = int(occurrence["tokens"])
                value = values[(kind, str(occurrence["record_id"]))]
                inputs = _inputs(value, torch)
                reference_logits = None
                mask = None
                if kind == "replay":
                    reference_logits, mask = _reference_logits(model, inputs, torch)
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(**inputs, use_cache=False)
                ce = float(outputs.loss.detach().float().item())
                ce_sums[kind] += ce * tokens
                token_sums[kind] += tokens
                if kind == "replay":
                    if reference_logits is None or mask is None:
                        raise RuntimeError("measurement reference logits are absent")
                    kl = forward_token_kl(
                        reference_logits,
                        outputs.logits[:, :-1, :],
                        mask,
                        torch,
                    )
                    replay_kl_sum += float(kl.detach().item()) * tokens
        validation_sum = 0.0
        validation_tokens = 0
        for value in validation:
            tokens = sum(label != -100 for label in value["labels"])
            inputs = _inputs(value, torch)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs, use_cache=False).loss
            validation_sum += float(loss.detach().float().item()) * tokens
            validation_tokens += tokens
    model.train(prior)
    result: dict[str, Any] = {
        "vetted_ce": ce_sums["vetted"] / token_sums["vetted"],
        "replay_ce": ce_sums["replay"] / token_sums["replay"],
        "replay_token_kl": replay_kl_sum / token_sums["replay"],
        "vetted_validation_ce": validation_sum / validation_tokens,
        "vetted_tokens": token_sums["vetted"],
        "replay_tokens": token_sums["replay"],
        "total_tokens": token_sums["vetted"] + token_sums["replay"],
        "validation_tokens": validation_tokens,
        "finite": all(
            math.isfinite(value)
            for value in (
                ce_sums["vetted"],
                ce_sums["replay"],
                replay_kl_sum,
                validation_sum,
            )
        ),
    }
    result["measurement_sha256"] = canonical_sha256(result)
    return result


def _modules() -> tuple[dict[str, Any], dict[str, Any]]:
    preimport = validate_preimport()
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
    postimport = validate_postimport(preimport, torch, modules)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return modules, {"preimport": preimport, "postimport": postimport}


def _common_inputs(
    *,
    arm: str,
    max_steps: int,
    model_path: Path,
    vetted_path: Path,
    validation_path: Path,
    replay_path: Path,
    schedule_path: Path,
    schedule_sha256: str,
    recipe_path: Path,
    modules: dict[str, Any],
) -> tuple[
    Any,
    Any,
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, list[int]]],
    list[dict[str, list[int]]],
]:
    if arm not in kl_recipe.ARMS or max_steps not in {16, 64}:
        raise ValueError("arm or step count is not authorized")
    _validate_recipe(recipe_path)
    schedule = _schedule(schedule_path, schedule_sha256, max_steps)
    records = _records(vetted_path, replay_path)
    validation_records = [{**item, "kind": "vetted"} for item in _jsonl(validation_path)]
    model, tokenizer = _load_base(model_path, modules)
    values = {key: _tokenize(record, tokenizer) for key, record in records.items()}
    validation = [_tokenize(record, tokenizer) for record in validation_records]
    return model, tokenizer, schedule, values, validation


def measure_historical(
    *,
    arm: str,
    model_path: Path,
    adapter_path: Path,
    vetted_path: Path,
    validation_path: Path,
    replay_path: Path,
    schedule_path: Path,
    schedule_sha256: str,
    recipe_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Measure one immutable historical V1 step-16 adapter without updating it."""

    if output_path.exists():
        raise FileExistsError("historical measurement output already exists")
    modules, launch = _modules()
    torch = modules["torch"]
    started = time.perf_counter()
    model, tokenizer, schedule, values, validation = _common_inputs(
        arm=arm,
        max_steps=16,
        model_path=model_path,
        vetted_path=vetted_path,
        validation_path=validation_path,
        replay_path=replay_path,
        schedule_path=schedule_path,
        schedule_sha256=schedule_sha256,
        recipe_path=recipe_path,
        modules=modules,
    )
    probe = values[("replay", sorted(key[1] for key in values if key[0] == "replay")[0])]
    base_generation = _generate(model, probe, torch)
    expected_adapter = kl_recipe.EXPECTED_ADAPTER_HASHES[arm]["16"]
    if directory_sha256(adapter_path) != expected_adapter:
        raise ValueError("historical step-16 adapter directory differs")
    model = modules["peft"].PeftModel.from_pretrained(
        model,
        str(adapter_path),
        local_files_only=True,
        is_trainable=False,
    )
    inventory = _normalized_lora_inventory(model)
    measurement = _measure(model, schedule, values, validation, torch)
    with model.disable_adapter():
        restored_generation = _generate(model, probe, torch)
    if restored_generation != base_generation:
        raise RuntimeError("historical adapter-disabled base did not restore")
    result: dict[str, Any] = {
        "schema_version": 1,
        "measurement_id": "foundry-historical-v1-step16-lambda-zero-measurement-v1",
        "arm": arm,
        "lambda_kl": 0,
        "optimizer_steps": 16,
        "adapter_sha256": expected_adapter,
        "adapter_config_file_sha256": file_sha256(adapter_path / "adapter_config.json"),
        "schedule_sha256": schedule_sha256,
        "schedule_prefix_sha256": canonical_sha256(schedule),
        "measurement": measurement,
        "trainable_inventory": inventory,
        "base_restoration": True,
        "model_update_performed": False,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_process_rss_bytes": int(modules["psutil"].Process().memory_info().rss),
        "launch_evidence": launch,
    }
    result["result_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train(
    *,
    arm: str,
    coefficient: float,
    max_steps: int,
    model_path: Path,
    vetted_path: Path,
    validation_path: Path,
    replay_path: Path,
    schedule_path: Path,
    schedule_sha256: str,
    recipe_path: Path,
    output_directory: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Train one fresh V1-equivalent KL calibration or complete arm."""

    if coefficient not in COEFFICIENTS:
        raise ValueError("KL coefficient is not predeclared")
    if output_directory.exists() or summary_path.exists():
        raise FileExistsError("KL training outputs must be fresh")
    modules, launch = _modules()
    torch = modules["torch"]
    started = time.perf_counter()
    model, tokenizer, schedule, values, validation = _common_inputs(
        arm=arm,
        max_steps=max_steps,
        model_path=model_path,
        vetted_path=vetted_path,
        validation_path=validation_path,
        replay_path=replay_path,
        schedule_path=schedule_path,
        schedule_sha256=schedule_sha256,
        recipe_path=recipe_path,
        modules=modules,
    )
    probe = values[("replay", sorted(key[1] for key in values if key[0] == "replay")[0])]
    base_generation = _generate(model, probe, torch)
    model = _prepare(model, modules["peft"])
    inventory = _normalized_lora_inventory(model)
    trainable = _active_trainable_parameters(model)
    trainable_ids = {id(parameter) for parameter in trainable}
    optimizer = modules["bitsandbytes"].optim.PagedAdamW8bit(trainable, lr=1e-5, weight_decay=0.0)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in cast(list[Any], group["params"])
    }
    if optimizer_ids != trainable_ids:
        raise RuntimeError("optimizer ownership differs from LoRA-only trainable set")
    scheduler = modules["transformers"].get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=4,
        num_training_steps=64,
    )
    base_before = _base_parameter_fingerprint(model, torch)
    initial_lora = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model.train()
    step_metrics: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    finite_gradients = True
    total_tokens = 0
    for step in schedule:
        step_tokens = int(step["loss_bearing_tokens"])
        optimizer.zero_grad(set_to_none=True)
        sums: defaultdict[str, float] = defaultdict(float)
        for occurrence in cast(list[dict[str, Any]], step["occurrences"]):
            kind = str(occurrence["kind"])
            value = values[(kind, str(occurrence["record_id"]))]
            actual = sum(label != -100 for label in value["labels"])
            if actual != int(occurrence["tokens"]):
                raise RuntimeError("scheduled assistant-token count differs")
            ce, kl, total = _loss(model, value, kind, coefficient, torch)
            if not all(bool(torch.isfinite(item).item()) for item in (ce, kl, total)):
                raise RuntimeError("non-finite CE, KL, or total loss")
            (total * (actual / step_tokens)).backward()
            sums[f"{kind}_ce"] += float(ce.detach().float().item()) * actual
            sums[f"{kind}_tokens"] += actual
            sums["replay_kl"] += float(kl.detach().float().item()) * actual
            sums["total"] += float(total.detach().float().item()) * actual
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        base_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" not in name and parameter.grad is not None
        ]
        if (
            not gradients
            or base_gradients
            or any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
        ):
            finite_gradients = False
            raise RuntimeError("gradient ownership or finiteness gate failed")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        lr_before = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        lr_after = float(optimizer.param_groups[0]["lr"])
        total_tokens += step_tokens
        step_metrics.append(
            {
                "step": int(step["step"]),
                "vetted_ce": sums["vetted_ce"] / sums["vetted_tokens"],
                "replay_ce": sums["replay_ce"] / sums["replay_tokens"],
                "replay_kl": sums["replay_kl"] / sums["replay_tokens"],
                "total_loss": sums["total"] / step_tokens,
                "loss_bearing_tokens": step_tokens,
                "lr_before_optimizer": lr_before,
                "lr_after_scheduler": lr_after,
            }
        )
        checkpoint_step = int(step["step"])
        if checkpoint_step in CHECKPOINTS and (max_steps == 64 or checkpoint_step == max_steps):
            checkpoint_path = output_directory / f"checkpoint-{checkpoint_step}" / "adapter"
            checkpoint_path.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(checkpoint_path, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_path)
            checkpoints[str(checkpoint_step)] = {
                "adapter_sha256": directory_sha256(checkpoint_path),
                "bytes": sum(
                    item.stat().st_size for item in checkpoint_path.rglob("*") if item.is_file()
                ),
            }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "coefficient": coefficient,
                    "completed_step": step["step"],
                }
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    lora_changed = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not torch.equal(initial_lora[name], parameter.detach().cpu())
    ]
    if not lora_changed:
        raise RuntimeError("LoRA parameters did not update")
    base_after = _base_parameter_fingerprint(model, torch)
    if base_after != base_before:
        raise RuntimeError("base parameter fingerprint changed")
    measurement = _measure(model, schedule, values, validation, torch)
    with model.disable_adapter():
        restored_generation = _generate(model, probe, torch)
    if restored_generation != base_generation:
        raise RuntimeError("adapter-disabled base generation did not restore")
    final_path = output_directory / f"checkpoint-{max_steps}" / "adapter"
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_rss = int(modules["psutil"].Process().memory_info().rss)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    base, _ = _load_base(model_path, modules)
    reloaded = modules["peft"].PeftModel.from_pretrained(
        base,
        str(final_path),
        local_files_only=True,
        is_trainable=False,
    )
    offline_reload = all(parameter.device.type == "cuda" for parameter in reloaded.parameters())
    reload_inventory = _normalized_lora_inventory(reloaded)
    del reloaded, base
    gc.collect()
    torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "foundry-replay-ce-token-kl-v1",
        "run_kind": "calibration" if max_steps == 16 else "complete_training",
        "arm": arm,
        "coefficient": coefficient,
        "optimizer_steps": max_steps,
        "loss_bearing_tokens": total_tokens,
        "schedule_sha256": schedule_sha256,
        "schedule_prefix_sha256": canonical_sha256(schedule),
        "recipe_sha256": RECIPE_SHA256,
        "recipe_decision_sha256": RECIPE_DECISION_SHA256,
        "step_metrics": step_metrics,
        "final_measurement": measurement,
        "checkpoints": checkpoints,
        "trainable_inventory": inventory,
        "reload_inventory": reload_inventory,
        "optimizer_owned_only_lora": optimizer_ids == trainable_ids,
        "finite_gradients": finite_gradients,
        "lora_updated": True,
        "changed_lora_tensor_count": len(lora_changed),
        "base_parameter_fingerprint_before": base_before,
        "base_parameter_fingerprint_after": base_after,
        "base_parameters_unchanged": base_before == base_after,
        "base_restoration": True,
        "offline_reload": offline_reload,
        "cuda_only": True,
        "cpu_offload": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_process_rss_bytes": peak_rss,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "launch_evidence": launch,
    }
    result["result_sha256"] = canonical_sha256(result)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def runtime_fixture(
    *,
    model_path: Path,
    replay_path: Path,
    recipe_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run a bounded two-update full-model fixture without saving an adapter."""

    if output_path.exists():
        raise FileExistsError("runtime fixture output already exists")
    modules, launch = _modules()
    torch = modules["torch"]
    started = time.perf_counter()
    _validate_recipe(recipe_path)
    replay = _read_object(replay_path)
    raw = cast(list[dict[str, Any]], replay["items"])[0]
    record = {**raw, "kind": "replay"}
    model, tokenizer = _load_base(model_path, modules)
    value = _tokenize(record, tokenizer)
    base_generation = _generate(model, value, torch)
    model = _prepare(model, modules["peft"])
    inventory = _normalized_lora_inventory(model)
    model.eval()
    inputs = _inputs(value, torch)
    reference, mask = _reference_logits(model, inputs, torch)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        policy = model(**inputs, use_cache=False).logits[:, :-1, :]
    identical_kl = float(forward_token_kl(reference, policy, mask, torch).item())
    if abs(identical_kl) > 1e-7:
        raise RuntimeError("zero-KL full-model fixture differs")
    trainable = _active_trainable_parameters(model)
    optimizer = modules["bitsandbytes"].optim.PagedAdamW8bit(trainable, lr=1e-5, weight_decay=0.0)
    trainable_ids = {id(parameter) for parameter in trainable}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in cast(list[Any], group["params"])
    }
    if optimizer_ids != trainable_ids:
        raise RuntimeError("runtime fixture optimizer ownership differs")
    scheduler = modules["transformers"].get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=4, num_training_steps=64
    )
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    base_before = _base_parameter_fingerprint(model, torch)
    model.train()
    losses: list[dict[str, float]] = []
    finite_gradients = True
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        ce, kl, total = _loss(model, value, "replay", 0.01, torch)
        total.backward()
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        base_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" not in name and parameter.grad is not None
        ]
        if (
            not gradients
            or base_gradients
            or any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
        ):
            finite_gradients = False
            raise RuntimeError("runtime fixture gradient gate failed")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(
            {
                "ce": float(ce.detach().float().item()),
                "kl": float(kl.detach().float().item()),
                "total": float(total.detach().float().item()),
            }
        )
    changed = sum(
        not torch.equal(initial[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    base_after = _base_parameter_fingerprint(model, torch)
    model.eval()
    with torch.no_grad():
        _, post_update_kl, _ = _loss(model, value, "replay", 0.01, torch)
    post_update_replay_kl = float(post_update_kl.detach().float().item())
    if not math.isfinite(post_update_replay_kl) or post_update_replay_kl <= 0.0:
        raise RuntimeError("runtime fixture post-update KL is not finite and positive")
    with model.disable_adapter():
        restored_generation = _generate(model, value, torch)
    if not changed or base_before != base_after or restored_generation != base_generation:
        raise RuntimeError("runtime fixture update, base, or restoration gate failed")
    result: dict[str, Any] = {
        "schema_version": 1,
        "fixture_id": "foundry-replay-ce-token-kl-v1-runtime-fixture-v1",
        "identical_logits_kl": identical_kl,
        "two_updates": True,
        "losses": losses,
        "finite_losses": all(math.isfinite(value) for row in losses for value in row.values()),
        "finite_gradients": finite_gradients,
        "post_update_replay_kl": post_update_replay_kl,
        "reference_no_grad": True,
        "optimizer_owned_only_lora": optimizer_ids == trainable_ids,
        "changed_lora_tensor_count": changed,
        "base_parameter_fingerprint_before": base_before,
        "base_parameter_fingerprint_after": base_after,
        "base_parameters_unchanged": base_before == base_after,
        "base_restoration": True,
        "trainable_inventory": inventory,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_process_rss_bytes": int(modules["psutil"].Process().memory_info().rss),
        "adapter_saved": False,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "launch_evidence": launch,
    }
    result["fixture_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("runtime-fixture")
    fixture.add_argument("--model-path", type=Path, required=True)
    fixture.add_argument("--replay-path", type=Path, required=True)
    fixture.add_argument("--recipe-path", type=Path, required=True)
    fixture.add_argument("--output-path", type=Path, required=True)
    measure = subparsers.add_parser("measure-historical")
    measure.add_argument("--arm", required=True)
    measure.add_argument("--model-path", type=Path, required=True)
    measure.add_argument("--adapter-path", type=Path, required=True)
    measure.add_argument("--vetted-path", type=Path, required=True)
    measure.add_argument("--validation-path", type=Path, required=True)
    measure.add_argument("--replay-path", type=Path, required=True)
    measure.add_argument("--schedule-path", type=Path, required=True)
    measure.add_argument("--schedule-sha256", required=True)
    measure.add_argument("--recipe-path", type=Path, required=True)
    measure.add_argument("--output-path", type=Path, required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--arm", required=True)
    training.add_argument("--coefficient", type=float, required=True)
    training.add_argument("--max-steps", type=int, required=True)
    training.add_argument("--model-path", type=Path, required=True)
    training.add_argument("--vetted-path", type=Path, required=True)
    training.add_argument("--validation-path", type=Path, required=True)
    training.add_argument("--replay-path", type=Path, required=True)
    training.add_argument("--schedule-path", type=Path, required=True)
    training.add_argument("--schedule-sha256", required=True)
    training.add_argument("--recipe-path", type=Path, required=True)
    training.add_argument("--output-directory", type=Path, required=True)
    training.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()
    command = str(args.command)
    values = vars(args)
    values.pop("command")
    if command == "runtime-fixture":
        result = runtime_fixture(**values)
    elif command == "measure-historical":
        result = measure_historical(**values)
    else:
        result = train(**values)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
