"""Measure historical replay CE and KL gradients without optimizer updates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, cast

from foundry.phase2 import kl_recipe, vetted_qlora_kl
from foundry.phase2.kl_objective import forward_token_kl
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

SCHEDULES = {
    "generic": "4bc00d29d5cf308c12c77111d7943567521cc533b13440dc06c3d8b39c74e9df",
    "targeted": "88c5378cac7efe927b29d3f421d97777cd6d917187c71c8388b60bbe7b57e259",
}
ADAPTER_CONFIG_SHA256 = "2cf0fb6637747b0aa31525f08ba8b412cc4f1986689ef8b9f555cd4b299039e2"
LAYER_PROJECTION_PATTERN = re.compile(
    r"\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\."
)


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _tensor_sha256(tensor: Any) -> str:
    digest = hashlib.sha256()
    value = tensor.detach().contiguous()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.view(-1).cpu().numpy().tobytes())
    return digest.hexdigest()


def _lora_fingerprint(model: Any) -> str:
    rows = [
        {
            "name": name,
            "tensor_sha256": _tensor_sha256(parameter),
        }
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    if len(rows) != kl_recipe.EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError("LoRA fingerprint inventory differs")
    return canonical_sha256(rows)


def replay_projection(schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the replay stream independently of arm-specific vetted placement."""

    result: list[dict[str, Any]] = []
    for step in schedule:
        for occurrence in cast(list[dict[str, Any]], step["occurrences"]):
            if occurrence["kind"] != "replay":
                continue
            result.append(
                {
                    "replay_occurrence_position": len(result),
                    "record_id": occurrence["record_id"],
                    "occurrence_index": occurrence["occurrence_index"],
                    "tokens": occurrence["tokens"],
                }
            )
    return result


def _identity_projection(
    replay: list[dict[str, Any]],
    values: dict[str, dict[str, list[int]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for occurrence in replay:
        record_id = str(occurrence["record_id"])
        value = values[record_id]
        labels = value["labels"]
        positions = [index for index, label in enumerate(labels) if label != -100]
        target_ids = [value["input_ids"][index] for index in positions]
        prompt_ids = value["input_ids"][: positions[0]]
        if len(target_ids) != int(occurrence["tokens"]):
            raise RuntimeError("replay occurrence token identity differs from its schedule")
        rows.append(
            {
                **occurrence,
                "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
                "target_token_ids_sha256": canonical_sha256(target_ids),
                "assistant_positions_sha256": canonical_sha256(positions),
            }
        )
    return rows


def prepare_replay_measurement(
    *,
    root: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[int]]], dict[str, Any]]:
    """Freeze the shared replay stream and content-free token identities."""

    schedule_root = root / "results/raw/phase2_vetted_corpus/v1_replay25_schedules"
    schedules = {
        arm: vetted_qlora_kl._schedule(
            schedule_root / f"{arm}_schedule.json",
            SCHEDULES[arm],
            16,
        )
        for arm in kl_recipe.ARMS
    }
    projections = {arm: replay_projection(schedules[arm]) for arm in kl_recipe.ARMS}
    if projections["generic"] != projections["targeted"]:
        raise RuntimeError("generic and targeted replay occurrence streams differ")
    replay_object = _read(root / "results/raw/training/base_replay_kl/replay_corpus.json")
    raw_items = cast(list[dict[str, Any]], replay_object["items"])
    values = {
        str(item["id"]): vetted_qlora_kl._tokenize({**item, "kind": "replay"}, tokenizer)
        for item in raw_items
    }
    identities = _identity_projection(projections["generic"], values)
    token_count = sum(int(row["tokens"]) for row in identities)
    if token_count != 4_000 or len(identities) != 213:
        raise RuntimeError("shared replay measurement stream count differs")
    manifest: dict[str, Any] = {
        "source_schedule_prefix_assistant_tokens": 16_000,
        "replay_assistant_token_count": token_count,
        "replay_occurrence_count": len(identities),
        "replay_record_count": len({str(row["record_id"]) for row in identities}),
        "generic_targeted_identical": True,
        "generic_source_schedule_sha256": SCHEDULES["generic"],
        "targeted_source_schedule_sha256": SCHEDULES["targeted"],
        "generic_source_schedule_prefix_sha256": canonical_sha256(schedules["generic"]),
        "targeted_source_schedule_prefix_sha256": canonical_sha256(schedules["targeted"]),
        "shared_replay_schedule_prefix_sha256": canonical_sha256(projections["generic"]),
        "token_identity_projection": identities,
        "token_identity_projection_sha256": canonical_sha256(identities),
    }
    return projections["generic"], values, manifest


def _named_lora(model: Any) -> list[tuple[str, Any]]:
    result = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if (
        len(result) != kl_recipe.EXPECTED_TRAINABLE_TENSORS
        or any("lora_" not in name for name, _ in result)
        or sum(parameter.numel() for _, parameter in result)
        != kl_recipe.EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("trainable LoRA gradient inventory differs")
    return result


def _clone_gradients(named_lora: list[tuple[str, Any]], torch: Any) -> dict[str, Any]:
    gradients: dict[str, Any] = {}
    for name, parameter in named_lora:
        if parameter.grad is None:
            gradients[name] = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        else:
            gradients[name] = parameter.grad.detach().float().cpu().clone()
    return gradients


def _base_gradient_count(model: Any) -> int:
    return sum(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )


def _backward_component(
    *,
    component: str,
    model: Any,
    replay: list[dict[str, Any]],
    values: dict[str, dict[str, list[int]]],
    named_lora: list[tuple[str, Any]],
    torch: Any,
) -> tuple[dict[str, Any], float, int]:
    model.zero_grad(set_to_none=True)
    total_tokens = sum(int(row["tokens"]) for row in replay)
    weighted_loss = 0.0
    for occurrence in replay:
        value = values[str(occurrence["record_id"])]
        actual = int(occurrence["tokens"])
        inputs = vetted_qlora_kl._inputs(value, torch)
        if component == "ce":
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs, use_cache=False).loss
        elif component == "kl":
            reference, mask = vetted_qlora_kl._reference_logits(model, inputs, torch)
            with torch.autocast("cuda", dtype=torch.float16):
                policy = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                ).logits[:, :-1, :]
            loss = forward_token_kl(reference, policy, mask, torch)
        else:
            raise ValueError("gradient component must be ce or kl")
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"{component} loss is nonfinite")
        (loss * (actual / total_tokens)).backward()
        weighted_loss += float(loss.detach().float().item()) * actual
    base_gradients = _base_gradient_count(model)
    gradients = _clone_gradients(named_lora, torch)
    return gradients, weighted_loss / total_tokens, base_gradients


def summarize_gradients(
    ce_gradients: dict[str, Any],
    kl_gradients: dict[str, Any],
    torch: Any,
    *,
    require_nonzero_kl: bool = True,
) -> dict[str, Any]:
    """Calculate global, tensor, layer, and projection gradient relationships."""

    if set(ce_gradients) != set(kl_gradients):
        raise ValueError("CE and KL gradient inventories differ")
    ce_sq = 0.0
    kl_sq = 0.0
    dot = 0.0
    nonzero_ce = 0
    nonzero_kl = 0
    opposing = 0
    ratios: list[float] = []
    layer_ce: defaultdict[str, float] = defaultdict(float)
    layer_kl: defaultdict[str, float] = defaultdict(float)
    projection_ce: defaultdict[str, float] = defaultdict(float)
    projection_kl: defaultdict[str, float] = defaultdict(float)
    all_finite = True
    for name in sorted(ce_gradients):
        ce = ce_gradients[name].double()
        kl = kl_gradients[name].double()
        all_finite = all_finite and bool(torch.isfinite(ce).all().item())
        all_finite = all_finite and bool(torch.isfinite(kl).all().item())
        ce_tensor_sq = float(torch.sum(ce * ce).item())
        kl_tensor_sq = float(torch.sum(kl * kl).item())
        tensor_dot = float(torch.sum(ce * kl).item())
        ce_sq += ce_tensor_sq
        kl_sq += kl_tensor_sq
        dot += tensor_dot
        nonzero_ce += ce_tensor_sq > 0.0
        nonzero_kl += kl_tensor_sq > 0.0
        opposing += tensor_dot < 0.0
        if ce_tensor_sq > 0.0:
            ratios.append(math.sqrt(kl_tensor_sq / ce_tensor_sq))
        match = LAYER_PROJECTION_PATTERN.search(name)
        if match is None:
            raise RuntimeError(f"LoRA gradient name lacks layer/projection: {name}")
        layer, projection = match.groups()
        layer_ce[layer] += ce_tensor_sq
        layer_kl[layer] += kl_tensor_sq
        projection_ce[projection] += ce_tensor_sq
        projection_kl[projection] += kl_tensor_sq
    ce_norm = math.sqrt(ce_sq)
    kl_norm = math.sqrt(kl_sq)
    if ce_norm <= 0.0 or (require_nonzero_kl and kl_norm <= 0.0):
        raise RuntimeError("required CE or KL global LoRA-gradient norm is zero")
    cosine = dot / (ce_norm * kl_norm) if kl_norm > 0.0 else 0.0
    result: dict[str, Any] = {
        "ce_global_l2_norm": ce_norm,
        "kl_global_l2_norm": kl_norm,
        "ce_to_kl_gradient_norm_ratio": (ce_norm / kl_norm if kl_norm > 0.0 else None),
        "kl_to_ce_gradient_norm_ratio": kl_norm / ce_norm,
        "cosine_similarity": cosine,
        "dot_product": dot,
        "nonzero_ce_lora_tensor_count": nonzero_ce,
        "nonzero_kl_lora_tensor_count": nonzero_kl,
        "opposing_gradient_sign_tensor_count": opposing,
        "per_layer_ce_l2_norm": {
            layer: math.sqrt(value) for layer, value in sorted(layer_ce.items())
        },
        "per_layer_kl_l2_norm": {
            layer: math.sqrt(value) for layer, value in sorted(layer_kl.items())
        },
        "per_projection_ce_l2_norm": {
            projection: math.sqrt(value) for projection, value in sorted(projection_ce.items())
        },
        "per_projection_kl_l2_norm": {
            projection: math.sqrt(value) for projection, value in sorted(projection_kl.items())
        },
        "maximum_per_tensor_kl_to_ce_norm_ratio": max(ratios),
        "median_per_tensor_kl_to_ce_norm_ratio": median(ratios),
        "finite_gradients": all_finite
        and all(math.isfinite(value) for value in (ce_norm, kl_norm, dot, cosine, *ratios)),
    }
    result["gradient_summary_sha256"] = canonical_sha256(result)
    return result


def measure_gradient_components(
    *,
    model: Any,
    replay: list[dict[str, Any]],
    values: dict[str, dict[str, list[int]]],
    torch: Any,
    require_nonzero_kl: bool = True,
) -> dict[str, Any]:
    """Run independent CE and KL backward passes on one unchanged model state."""

    prior = model.training
    model.eval()
    named_lora = _named_lora(model)
    ce_gradients, ce_loss, ce_base_count = _backward_component(
        component="ce",
        model=model,
        replay=replay,
        values=values,
        named_lora=named_lora,
        torch=torch,
    )
    model.zero_grad(set_to_none=True)
    kl_gradients, kl_loss, kl_base_count = _backward_component(
        component="kl",
        model=model,
        replay=replay,
        values=values,
        named_lora=named_lora,
        torch=torch,
    )
    summary = summarize_gradients(
        ce_gradients,
        kl_gradients,
        torch,
        require_nonzero_kl=require_nonzero_kl,
    )
    model.zero_grad(set_to_none=True)
    model.train(prior)
    summary.update(
        {
            "replay_ce": ce_loss,
            "replay_token_kl": kl_loss,
            "base_gradient_count": max(ce_base_count, kl_base_count),
            "reference_gradient_count": 0,
            "independent_backward_passes": True,
            "gradients_cloned_before_clear": True,
            "optimizer_step_performed": False,
            "scheduler_state_altered": False,
        }
    )
    summary["gradient_summary_sha256"] = canonical_sha256(
        {name: value for name, value in summary.items() if name != "gradient_summary_sha256"}
    )
    return summary


def audit_historical(
    *,
    root: Path,
    arm: str,
    model_path: Path,
    adapter_path: Path,
    recipe_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Measure one immutable historical step-16 adapter twice."""

    if arm not in kl_recipe.ARMS:
        raise ValueError("arm is not authorized")
    if output_path.exists():
        raise FileExistsError("historical gradient output already exists")
    expected_adapter = kl_recipe.EXPECTED_ADAPTER_HASHES[arm]["16"]
    adapter_before = directory_sha256(adapter_path)
    if (
        adapter_before != expected_adapter
        or file_sha256(adapter_path / "adapter_config.json") != ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("historical adapter identity differs")
    vetted_qlora_kl._validate_recipe(recipe_path)
    modules, launch = vetted_qlora_kl._modules()
    torch = modules["torch"]
    started = time.perf_counter()
    model, tokenizer = vetted_qlora_kl._load_base(model_path, modules)
    model = modules["peft"].prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = modules["peft"].PeftModel.from_pretrained(
        model,
        str(adapter_path),
        local_files_only=True,
        is_trainable=True,
    )
    model.config.use_cache = False
    inventory = vetted_qlora_kl._normalized_lora_inventory(model)
    replay, values, manifest = prepare_replay_measurement(root=root, tokenizer=tokenizer)
    base_before = vetted_qlora_kl._base_parameter_fingerprint(model, torch)
    lora_before = _lora_fingerprint(model)
    measurements = [
        measure_gradient_components(
            model=model,
            replay=replay,
            values=values,
            torch=torch,
        )
        for _ in range(2)
    ]
    base_after = vetted_qlora_kl._base_parameter_fingerprint(model, torch)
    lora_after = _lora_fingerprint(model)
    adapter_after = directory_sha256(adapter_path)
    if measurements[0] != measurements[1]:
        raise RuntimeError("duplicate historical gradient measurement differs")
    if base_before != base_after or lora_before != lora_after or adapter_before != adapter_after:
        raise RuntimeError("historical adapter or model state changed")
    measurement = measurements[0]
    if (
        measurement["finite_gradients"] is not True
        or measurement["ce_global_l2_norm"] <= 0.0
        or measurement["kl_global_l2_norm"] <= 0.0
        or measurement["base_gradient_count"] != 0
        or measurement["reference_gradient_count"] != 0
    ):
        raise RuntimeError("historical gradient validity gate failed")
    torch.cuda.synchronize()
    result: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-milestone13d-historical-gradient-runtime-v1",
        "arm": arm,
        "adapter_sha256": expected_adapter,
        "adapter_config_file_sha256": ADAPTER_CONFIG_SHA256,
        "recipe_sha256": vetted_qlora_kl.RECIPE_SHA256,
        "measurement_manifest": manifest,
        "measurements": measurements,
        "duplicate_measurement_identical": True,
        "base_parameter_fingerprint_before": base_before,
        "base_parameter_fingerprint_after": base_after,
        "lora_fingerprint_before": lora_before,
        "lora_fingerprint_after": lora_after,
        "model_state_unchanged": True,
        "adapter_directory_sha256_before": adapter_before,
        "adapter_directory_sha256_after": adapter_after,
        "adapter_unchanged": True,
        "trainable_inventory": inventory,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "peak_process_rss_bytes": int(modules["psutil"].Process().memory_info().rss),
        "holdout_v2_use": False,
        "gsm1k_use": False,
        "sealed_paths_accessed": False,
        "launch_evidence": launch,
    }
    result["result_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--recipe-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    result = audit_historical(
        root=args.root.resolve(),
        arm=args.arm,
        model_path=args.model_path.resolve(),
        adapter_path=args.adapter_path.resolve(),
        recipe_path=args.recipe_path.resolve(),
        output_path=args.output_path.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
