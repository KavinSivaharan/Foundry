"""Run gradient-calibrated replay-ce-token-kl-v1 smoke or calibration training."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_gradient_runtime, kl_recipe, vetted_qlora_kl
from foundry.phase2.update_detection import (
    detect_updates,
    snapshot_trainable,
    validate_optimizer_ownership,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256

HISTORICAL_REPLAY_KL = {
    "generic": 0.00013135224548594236,
    "targeted": 0.00012543418646708914,
}
GRADIENT_MEASUREMENT_STEPS = (2, 8, 16)


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify_self_hash(value: dict[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _validate_ladder(path: Path, rho: str, coefficient: str) -> dict[str, Any]:
    ladder = _read(path)
    _verify_self_hash(ladder, "ladder_sha256")
    rows = [row for row in cast(list[dict[str, Any]], ladder["ladder"]) if row["rho_exact"] == rho]
    if (
        len(rows) != 1
        or rows[0]["lambda_common_exact"] != coefficient
        or ladder.get("frozen_before_coefficient_execution") is not True
    ):
        raise ValueError("rho or coefficient differs from the frozen common ladder")
    return ladder


def _finite_step_metrics(rows: list[dict[str, Any]]) -> bool:
    return all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("vetted_ce", "replay_ce", "replay_kl", "total_loss")
    )


def train(
    *,
    root: Path,
    arm: str,
    rho: str,
    coefficient: str,
    max_steps: int,
    model_path: Path,
    vetted_path: Path,
    validation_path: Path,
    replay_path: Path,
    schedule_path: Path,
    schedule_sha256: str,
    recipe_path: Path,
    ladder_path: Path,
    output_directory: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Train one fresh rank-8 adapter under one frozen derived coefficient."""

    if arm not in kl_recipe.ARMS or max_steps not in {2, 16}:
        raise ValueError("arm or bounded step count is not authorized")
    if output_directory.exists() or summary_path.exists():
        raise FileExistsError("gradient-calibrated training outputs must be fresh")
    ladder = _validate_ladder(ladder_path, rho, coefficient)
    coefficient_float = float(coefficient)
    if (
        not math.isfinite(coefficient_float)
        or coefficient_float <= 0.0
        or coefficient_float > 1_000_000.0
    ):
        raise ValueError("gradient-calibrated coefficient is not finite and bounded")
    modules, launch = vetted_qlora_kl._modules()
    torch = modules["torch"]
    started = time.perf_counter()
    model, tokenizer, schedule_16, values, validation = vetted_qlora_kl._common_inputs(
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
    schedule = schedule_16[:max_steps]
    if len(schedule) != max_steps or not any(
        occurrence["kind"] == "replay"
        for step in schedule
        for occurrence in cast(list[dict[str, Any]], step["occurrences"])
    ):
        raise RuntimeError("bounded schedule is not the first replay-containing segment")
    total_expected_tokens = sum(int(step["loss_bearing_tokens"]) for step in schedule)
    replay_measurement, replay_values, manifest = kl_gradient_runtime.prepare_replay_measurement(
        root=root, tokenizer=tokenizer
    )
    probe = values[("replay", sorted(key[1] for key in values if key[0] == "replay")[0])]
    base_generation = vetted_qlora_kl._generate(model, probe, torch)
    model = vetted_qlora_kl._prepare(model, modules["peft"])
    inventory = vetted_qlora_kl._normalized_lora_inventory(model)
    named_trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    optimizer = modules["bitsandbytes"].optim.PagedAdamW8bit(
        trainable,
        lr=1e-5,
        weight_decay=0.0,
    )
    validate_optimizer_ownership(named_trainable, optimizer)
    scheduler = modules["transformers"].get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=4,
        num_training_steps=64,
    )
    base_before = vetted_qlora_kl._base_parameter_fingerprint(model, torch)
    initial_lora = {name: parameter.detach().cpu().clone() for name, parameter in named_trainable}
    model.train()
    step_metrics: list[dict[str, Any]] = []
    gradient_measurements: dict[str, Any] = {}
    smoke_measurements: list[dict[str, Any]] = []
    finite_gradients = True
    total_tokens = 0
    first_positive_lr_update = False
    for step in schedule:
        step_number = int(step["step"])
        if step_number in GRADIENT_MEASUREMENT_STEPS:
            gradient = kl_gradient_runtime.measure_gradient_components(
                model=model,
                replay=replay_measurement,
                values=replay_values,
                torch=torch,
                require_nonzero_kl=False,
            )
            gradient_measurements[str(step_number)] = {
                "measurement_position": "before_optimizer_step",
                "ce_global_l2_norm": gradient["ce_global_l2_norm"],
                "kl_global_l2_norm": gradient["kl_global_l2_norm"],
                "unweighted_kl_to_ce_gradient_ratio": gradient["kl_to_ce_gradient_norm_ratio"],
                "weighted_kl_to_ce_gradient_ratio": (
                    coefficient_float * float(gradient["kl_to_ce_gradient_norm_ratio"])
                ),
                "cosine_similarity": gradient["cosine_similarity"],
                "dot_product": gradient["dot_product"],
                "finite_gradients": gradient["finite_gradients"],
                "base_gradient_count": gradient["base_gradient_count"],
                "reference_gradient_count": gradient["reference_gradient_count"],
                "gradient_summary_sha256": gradient["gradient_summary_sha256"],
            }
        step_tokens = int(step["loss_bearing_tokens"])
        optimizer.zero_grad(set_to_none=True)
        sums: defaultdict[str, float] = defaultdict(float)
        for occurrence in cast(list[dict[str, Any]], step["occurrences"]):
            kind = str(occurrence["kind"])
            value = values[(kind, str(occurrence["record_id"]))]
            actual = sum(label != -100 for label in value["labels"])
            if actual != int(occurrence["tokens"]):
                raise RuntimeError("scheduled assistant-token count differs")
            ce, kl, total = vetted_qlora_kl._loss(
                model,
                value,
                kind,
                coefficient_float,
                torch,
            )
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
        snapshots = snapshot_trainable(named_trainable)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        lr_before = float(optimizer.param_groups[0]["lr"])
        scheduler_epoch_before = int(scheduler.last_epoch)
        optimizer.step()
        update = detect_updates(snapshots, named_trainable)
        scheduler.step()
        lr_after = float(optimizer.param_groups[0]["lr"])
        if step_number == 1 and (lr_before != 0.0 or int(update["tensors_changed"]) != 0):
            raise RuntimeError("zero-LR warmup step behavior differs")
        if lr_before > 0.0 and not first_positive_lr_update:
            if int(update["tensors_changed"]) <= 0 or float(update["global_delta_norm"]) <= 0.0:
                raise RuntimeError("first positive-LR step did not update LoRA")
            first_positive_lr_update = True
        total_tokens += step_tokens
        step_metrics.append(
            {
                "step": step_number,
                "vetted_ce": sums["vetted_ce"] / sums["vetted_tokens"],
                "replay_ce": sums["replay_ce"] / sums["replay_tokens"],
                "replay_kl": sums["replay_kl"] / sums["replay_tokens"],
                "total_loss": sums["total"] / step_tokens,
                "loss_bearing_tokens": step_tokens,
                "lr_before_optimizer": lr_before,
                "lr_after_scheduler": lr_after,
                "scheduler_last_epoch_before": scheduler_epoch_before,
                "scheduler_last_epoch_after": int(scheduler.last_epoch),
                "update": {
                    "tensors_changed": update["tensors_changed"],
                    "total_changed_elements": update["total_changed_elements"],
                    "global_delta_norm": update["global_delta_norm"],
                    "evidence_sha256": update["evidence_sha256"],
                },
            }
        )
        if max_steps == 2:
            smoke_measurements.append(
                vetted_qlora_kl._measure(model, schedule, values, validation, torch)
            )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "rho": rho,
                    "coefficient": coefficient,
                    "completed_step": step_number,
                }
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    changed = [
        name
        for name, parameter in named_trainable
        if not torch.equal(initial_lora[name], parameter.detach().cpu())
    ]
    if not changed or not first_positive_lr_update:
        raise RuntimeError("LoRA parameters did not update at positive learning rate")
    base_after = vetted_qlora_kl._base_parameter_fingerprint(model, torch)
    if base_after != base_before:
        raise RuntimeError("base parameter fingerprint changed")
    final_measurement = vetted_qlora_kl._measure(
        model,
        schedule,
        values,
        validation,
        torch,
    )
    with model.disable_adapter():
        restored_generation = vetted_qlora_kl._generate(model, probe, torch)
    if restored_generation != base_generation:
        raise RuntimeError("adapter-disabled base generation did not restore")
    adapter_path = output_directory / f"checkpoint-{max_steps}" / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    adapter_sha256 = directory_sha256(adapter_path)
    checkpoint_bytes = sum(
        item.stat().st_size for item in adapter_path.rglob("*") if item.is_file()
    )
    smoke_ceiling = 10.0 * HISTORICAL_REPLAY_KL[arm]
    smoke_safe = max_steps != 2 or (
        bool(smoke_measurements)
        and all(
            bool(row["finite"])
            and math.isfinite(float(row["replay_token_kl"]))
            and float(row["replay_token_kl"]) <= smoke_ceiling
            for row in smoke_measurements
        )
        and _finite_step_metrics(step_metrics)
    )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_rss = int(modules["psutil"].Process().memory_info().rss)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    base, _ = vetted_qlora_kl._load_base(model_path, modules)
    reloaded = modules["peft"].PeftModel.from_pretrained(
        base,
        str(adapter_path),
        local_files_only=True,
        is_trainable=False,
    )
    offline_reload = all(parameter.device.type == "cuda" for parameter in reloaded.parameters())
    reload_inventory = vetted_qlora_kl._normalized_lora_inventory(reloaded)
    del reloaded, base
    gc.collect()
    torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "foundry-gradient-calibrated-replay-ce-token-kl-v1",
        "run_kind": "compatibility_smoke" if max_steps == 2 else "calibration",
        "arm": arm,
        "rho_exact": rho,
        "coefficient_exact": coefficient,
        "coefficient_float": coefficient_float,
        "ladder_sha256": ladder["ladder_sha256"],
        "optimizer_steps": max_steps,
        "loss_bearing_tokens": total_tokens,
        "expected_loss_bearing_tokens": total_expected_tokens,
        "schedule_sha256": schedule_sha256,
        "schedule_prefix_sha256": canonical_sha256(schedule),
        "gradient_measurement_manifest_projection_sha256": manifest[
            "token_identity_projection_sha256"
        ],
        "recipe_sha256": vetted_qlora_kl.RECIPE_SHA256,
        "recipe_decision_sha256": vetted_qlora_kl.RECIPE_DECISION_SHA256,
        "step_metrics": step_metrics,
        "gradient_measurements": gradient_measurements,
        "smoke_post_step_measurements": smoke_measurements,
        "smoke_replay_kl_ceiling": smoke_ceiling if max_steps == 2 else None,
        "smoke_replay_kl_within_10x_historical": smoke_safe if max_steps == 2 else None,
        "final_measurement": final_measurement,
        "checkpoint": {
            "step": max_steps,
            "adapter_sha256": adapter_sha256,
            "bytes": checkpoint_bytes,
        },
        "trainable_inventory": inventory,
        "reload_inventory": reload_inventory,
        "optimizer_owned_only_lora": True,
        "finite_gradients": finite_gradients,
        "finite_losses": _finite_step_metrics(step_metrics),
        "first_positive_lr_step_updated_lora": first_positive_lr_update,
        "lora_updated": True,
        "changed_lora_tensor_count": len(changed),
        "base_parameter_fingerprint_before": base_before,
        "base_parameter_fingerprint_after": base_after,
        "base_parameters_unchanged": base_before == base_after,
        "base_restoration": True,
        "offline_reload": offline_reload,
        "cuda_only": True,
        "cpu_offload": False,
        "overflow_or_nan": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_process_rss_bytes": peak_rss,
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--rho", required=True)
    parser.add_argument("--coefficient", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vetted-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--schedule-path", type=Path, required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--recipe-path", type=Path, required=True)
    parser.add_argument("--ladder-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    args = parser.parse_args()
    result = train(
        root=args.root.resolve(),
        arm=args.arm,
        rho=args.rho,
        coefficient=args.coefficient,
        max_steps=args.max_steps,
        model_path=args.model_path.resolve(),
        vetted_path=args.vetted_path.resolve(),
        validation_path=args.validation_path.resolve(),
        replay_path=args.replay_path.resolve(),
        schedule_path=args.schedule_path.resolve(),
        schedule_sha256=args.schedule_sha256,
        recipe_path=args.recipe_path.resolve(),
        ladder_path=args.ladder_path.resolve(),
        output_directory=args.output_directory.resolve(),
        summary_path=args.summary_path.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
