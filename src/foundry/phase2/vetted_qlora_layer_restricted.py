"""Train one frozen CE-only V1 LoRA arm with a predeclared layer restriction."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import vetted_qlora_kl as kl_runner
from foundry.phase2.layer_restricted import (
    MODEL_REVISION,
    V1_LORA_SHA256,
    LayerScope,
    scope_for_label,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256

CHECKPOINTS = (16, 32, 64)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _prepare(model: Any, peft: Any, scope: LayerScope) -> Any:
    model = peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=list(PROJECTIONS),
            layers_to_transform=list(scope.layer_indices),
            layers_pattern="layers",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    return model


def _inventory(model: Any, scope: LayerScope) -> dict[str, Any]:
    shapes = [
        {
            "name": name.replace(".default", ""),
            "shape": list(parameter.shape),
        }
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    shapes.sort(key=lambda value: str(value["name"]))
    names = [str(value["name"]) for value in shapes]
    layer_indices: set[int] = set()
    projection_counts: dict[str, int] = {name: 0 for name in PROJECTIONS}
    for name in names:
        match = LAYER_PATTERN.search(name)
        if match is None:
            raise RuntimeError("LoRA tensor lacks a transformer-layer index")
        layer_indices.add(int(match.group(1)))
        matches = [projection for projection in PROJECTIONS if f".{projection}." in name]
        if len(matches) != 1:
            raise RuntimeError("LoRA tensor projection is outside the frozen target set")
        projection_counts[matches[0]] += 1
    result = {
        "scope_label": scope.label,
        "layer_indices": sorted(layer_indices),
        "adapted_layer_count": len(layer_indices),
        "adapted_module_count": len(names) // 2,
        "projection_tensor_counts": projection_counts,
        "trainable_tensor_count": len(shapes),
        "trainable_parameter_count": sum(
            math.prod(cast(list[int], value["shape"])) for value in shapes
        ),
        "tensor_inventory_sha256": canonical_sha256(shapes),
        "only_lora_tensors": all("lora_" in name for name in names),
    }
    expected = {
        "scope_label": scope.label,
        "layer_indices": list(scope.layer_indices),
        "adapted_layer_count": scope.top_layer_count,
        "adapted_module_count": scope.adapted_module_count,
        "projection_tensor_counts": {
            projection: scope.top_layer_count * 2 for projection in PROJECTIONS
        },
        "trainable_tensor_count": scope.trainable_tensor_count,
        "trainable_parameter_count": scope.trainable_parameter_count,
        "only_lora_tensors": True,
    }
    for key, value in expected.items():
        if result[key] != value:
            raise RuntimeError(f"layer-restricted LoRA inventory differs: {key}")
    return result


def _active_trainable_parameters(model: Any, scope: LayerScope) -> list[Any]:
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if (
        len(named) != scope.trainable_tensor_count
        or any("lora_" not in name for name, _ in named)
        or sum(parameter.numel() for _, parameter in named) != scope.trainable_parameter_count
    ):
        raise RuntimeError("active parameters differ from the frozen layer scope")
    return [parameter for _, parameter in named]


def _validation_ce(
    model: Any,
    validation: list[dict[str, list[int]]],
    torch: Any,
) -> float:
    prior = model.training
    model.eval()
    weighted = 0.0
    tokens = 0
    with torch.inference_mode():
        for value in validation:
            count = sum(label != -100 for label in value["labels"])
            inputs = kl_runner._inputs(value, torch)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs, use_cache=False).loss
            weighted += float(loss.detach().float().item()) * count
            tokens += count
    model.train(prior)
    return weighted / tokens


def _save(
    model: Any,
    tokenizer: Any,
    path: Path,
    validation: list[dict[str, list[int]]],
    torch: Any,
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return {
        "adapter_sha256": directory_sha256(path),
        "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
        "validation_ce": _validation_ce(model, validation, torch),
    }


def train(
    *,
    arm: str,
    scope_label: str,
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
    """Run one fresh 16-step calibration or 64-step complete training arm."""

    scope = scope_for_label(scope_label)
    if output_directory.exists() or summary_path.exists():
        raise FileExistsError("layer-restricted training outputs must be fresh")
    modules, launch = kl_runner._modules()
    torch = modules["torch"]
    started = time.perf_counter()
    model, tokenizer, schedule, values, validation = kl_runner._common_inputs(
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
    if getattr(model.config, "_name_or_path", "") and MODEL_REVISION not in str(model_path):
        raise ValueError("base-model revision path differs")
    probe = values[("replay", sorted(key[1] for key in values if key[0] == "replay")[0])]
    base_generation = kl_runner._generate(model, probe, torch)
    model = _prepare(model, modules["peft"], scope)
    inventory = _inventory(model, scope)
    trainable = _active_trainable_parameters(model, scope)
    trainable_ids = {id(parameter) for parameter in trainable}
    optimizer = modules["bitsandbytes"].optim.PagedAdamW8bit(
        trainable,
        lr=1e-5,
        weight_decay=0.0,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in cast(list[Any], group["params"])
    }
    if optimizer_ids != trainable_ids:
        raise RuntimeError("optimizer ownership differs from selected LoRA tensors")
    scheduler = modules["transformers"].get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=4,
        num_training_steps=64,
    )
    base_before = kl_runner._base_parameter_fingerprint(model, torch)
    initial_lora = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model.train()
    step_metrics: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    finite_gradients = True
    finite_losses = True
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
            inputs = kl_runner._inputs(value, torch)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs, use_cache=False).loss
            if not bool(torch.isfinite(loss).item()):
                finite_losses = False
                raise RuntimeError("non-finite cross-entropy loss")
            (loss * (actual / step_tokens)).backward()
            sums[f"{kind}_ce"] += float(loss.detach().float().item()) * actual
            sums[f"{kind}_tokens"] += actual
            sums["total"] += float(loss.detach().float().item()) * actual
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
        total_tokens += step_tokens
        step_number = int(step["step"])
        step_metrics.append(
            {
                "step": step_number,
                "vetted_ce": sums["vetted_ce"] / sums["vetted_tokens"],
                "replay_ce": sums["replay_ce"] / sums["replay_tokens"],
                "total_loss": sums["total"] / step_tokens,
                "loss_bearing_tokens": step_tokens,
                "lr_before_optimizer": lr_before,
                "lr_after_scheduler": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if step_number in CHECKPOINTS and (max_steps == 64 or step_number == max_steps):
            checkpoint_path = output_directory / f"checkpoint-{step_number}" / "adapter"
            checkpoints[str(step_number)] = _save(
                model,
                tokenizer,
                checkpoint_path,
                validation,
                torch,
            )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "scope": scope.label,
                    "completed_step": step_number,
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
    base_after = kl_runner._base_parameter_fingerprint(model, torch)
    if base_after != base_before:
        raise RuntimeError("base parameter fingerprint changed")
    measurement = kl_runner._measure(model, schedule, values, validation, torch)
    with model.disable_adapter():
        restored_generation = kl_runner._generate(model, probe, torch)
    if restored_generation != base_generation:
        raise RuntimeError("adapter-disabled base generation did not restore")
    final_path = output_directory / f"checkpoint-{max_steps}" / "adapter"
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_rss = int(modules["psutil"].Process().memory_info().rss)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    base, _ = kl_runner._load_base(model_path, modules)
    reloaded = modules["peft"].PeftModel.from_pretrained(
        base,
        str(final_path),
        local_files_only=True,
        is_trainable=False,
    )
    offline_reload = all(parameter.device.type == "cuda" for parameter in reloaded.parameters())
    reload_inventory = _inventory(reloaded, scope)
    del reloaded, base
    gc.collect()
    torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "foundry-layer-restricted-v1-ce-only",
        "run_kind": "calibration" if max_steps == 16 else "complete_training",
        "arm": arm,
        "scope_label": scope.label,
        "top_layer_count": scope.top_layer_count,
        "layer_indices": list(scope.layer_indices),
        "sole_scientific_intervention": "layers_to_transform",
        "training_objective": {
            "vetted": "assistant_only_cross_entropy",
            "replay": "assistant_only_cross_entropy",
            "replay_kl": "post_training_diagnostic_only",
        },
        "optimizer_steps": max_steps,
        "loss_bearing_tokens": total_tokens,
        "schedule_sha256": schedule_sha256,
        "schedule_prefix_sha256": canonical_sha256(schedule),
        "v1_lora_configuration_sha256": V1_LORA_SHA256,
        "step_metrics": step_metrics,
        "final_measurement": measurement,
        "checkpoints": checkpoints,
        "trainable_inventory": inventory,
        "reload_inventory": reload_inventory,
        "optimizer_owned_only_selected_lora": optimizer_ids == trainable_ids,
        "finite_gradients": finite_gradients,
        "finite_losses": finite_losses,
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
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--scope-label", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vetted-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--schedule-path", type=Path, required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--recipe-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
