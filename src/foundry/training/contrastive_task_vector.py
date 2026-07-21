"""Construct and verify the exact targeted-minus-generic LoRA task vector."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from foundry.training.adapter_sanity import (
    DIAGNOSTIC_PROMPTS,
    DIAGNOSTIC_SYSTEM,
    _generate,
    _load_base,
    _output_hash,
)
from foundry.training.config import canonical_sha256
from foundry.training.lora_scaling import (
    adapter_state_sha256,
    base_parameter_signature_sha256,
    scaled_lora_adapter,
)
from foundry.training.qlora import directory_sha256, file_sha256

GENERIC_NAME = "generic_source"
TARGETED_NAME = "targeted_source"
CONTRASTIVE_NAME = "targeted_minus_generic_v1"
LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])\.weight$")
LAYER_KEY = re.compile(r"\.layers\.(?P<layer>\d+)\.")


def _object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def load_protocol(path: Path) -> dict[str, Any]:
    """Load and validate the frozen task-vector protocol."""

    config = _object(path)
    expected = config.get("config_sha256")
    payload = {key: value for key, value in config.items() if key != "config_sha256"}
    if not isinstance(expected, str) or expected != canonical_sha256(payload):
        raise ValueError("contrastive task-vector configuration hash differs")
    if config.get("protocol_id") != "foundry-targeted-minus-generic-task-vector-v1":
        raise ValueError("contrastive task-vector protocol identity differs")
    composition = cast(dict[str, Any], config.get("composition"))
    if (
        composition.get("adapter_name") != CONTRASTIVE_NAME
        or composition.get("adapters") != [TARGETED_NAME, GENERIC_NAME]
        or composition.get("weights") != [1.0, -1.0]
        or composition.get("combination_type") != "cat"
        or composition.get("factor_dtype") != "float32"
        or composition.get("merge_allowed") is not False
        or composition.get("compression_allowed") is not False
    ):
        raise ValueError("contrastive composition definition differs")
    return config


def _tensor_inventory(adapter_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    safetensors = importlib.import_module("safetensors")
    weights_path = adapter_path / "adapter_model.safetensors"
    records: list[dict[str, object]] = []
    with safetensors.safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            records.append({"name": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return records, keys


def _paired_modules(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    modules: dict[str, dict[str, object]] = {}
    for record in records:
        key = str(record["name"])
        match = LORA_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"unexpected adapter tensor key: {key}")
        module = match.group("module")
        side = match.group("side")
        if side in modules.setdefault(module, {}):
            raise ValueError(f"duplicated LoRA {side} tensor: {module}")
        modules[module][side] = record
    if any(set(sides) != {"A", "B"} for sides in modules.values()):
        raise ValueError("every LoRA module requires exactly one A and one B tensor")
    return modules


def inspect_adapter(adapter_path: Path) -> dict[str, Any]:
    """Return strict, content-free adapter metadata without loading a base model."""

    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    if not adapter_path.is_dir() or not config_path.is_file() or not weights_path.is_file():
        raise ValueError(f"adapter is incomplete: {adapter_path}")
    config = _object(config_path)
    records, _ = _tensor_inventory(adapter_path)
    modules = _paired_modules(records)
    return {
        "directory_sha256": directory_sha256(adapter_path),
        "config_file_sha256": file_sha256(config_path),
        "weight_file_sha256": file_sha256(weights_path),
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "dropout": config.get("lora_dropout"),
        "bias": config.get("bias"),
        "modules_to_save": config.get("modules_to_save"),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "target_modules": sorted(cast(list[str], config.get("target_modules"))),
        "use_dora": config.get("use_dora"),
        "use_rslora": config.get("use_rslora"),
        "fan_in_fan_out": config.get("fan_in_fan_out"),
        "saved_tensor_count": len(records),
        "lora_module_count": len(modules),
        "tensor_inventory_sha256": canonical_sha256(records),
        "tensor_inventory": records,
    }


def validate_source_compatibility(
    *, generic: dict[str, Any], targeted: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed unless both frozen source adapters have one compatible contract."""

    source_contract = cast(dict[str, Any], protocol["source_contract"])
    source_adapters = cast(dict[str, dict[str, Any]], protocol["source_adapters"])
    base_revision = cast(dict[str, Any], protocol["base_model"])["revision"]
    checked = {
        "generic_directory_hash": generic["directory_sha256"]
        == source_adapters[GENERIC_NAME]["directory_sha256"],
        "targeted_directory_hash": targeted["directory_sha256"]
        == source_adapters[TARGETED_NAME]["directory_sha256"],
        "same_inventory": generic["tensor_inventory"] == targeted["tensor_inventory"],
        "same_rank": generic["rank"] == targeted["rank"] == source_contract["rank"],
        "same_alpha": generic["alpha"] == targeted["alpha"] == source_contract["alpha"],
        "same_dropout": generic["dropout"] == targeted["dropout"] == source_contract["dropout"],
        "same_bias": generic["bias"] == targeted["bias"] == source_contract["bias"],
        "no_modules_to_save": generic["modules_to_save"] is None
        and targeted["modules_to_save"] is None
        and source_contract["modules_to_save"] is None,
        "same_target_modules": generic["target_modules"]
        == targeted["target_modules"]
        == sorted(source_contract["target_modules"]),
        "same_base_revision": str(generic["base_model_name_or_path"]).endswith(base_revision)
        and str(targeted["base_model_name_or_path"]).endswith(base_revision),
        "expected_tensor_count": generic["saved_tensor_count"]
        == targeted["saved_tensor_count"]
        == source_contract["saved_tensor_count"],
        "expected_module_count": generic["lora_module_count"]
        == targeted["lora_module_count"]
        == source_contract["lora_module_count"],
        "supported_standard_lora": all(
            item is False
            for item in (
                generic["use_dora"],
                generic["use_rslora"],
                generic["fan_in_fan_out"],
                targeted["use_dora"],
                targeted["use_rslora"],
                targeted["fan_in_fan_out"],
            )
        ),
    }
    if not all(checked.values()):
        failed = [name for name, passed in checked.items() if not passed]
        raise ValueError(f"source adapter compatibility failed: {failed}")
    inventory_hash = str(generic["tensor_inventory_sha256"])
    configured_inventory = str(source_contract["inventory_sha256"])
    if inventory_hash != configured_inventory:
        raise ValueError("source tensor inventory hash differs from frozen protocol")
    result: dict[str, Any] = {
        "checks": checked,
        "source_inventory_sha256": inventory_hash,
        "generic": {key: value for key, value in generic.items() if key != "tensor_inventory"},
        "targeted": {key: value for key, value in targeted.items() if key != "tensor_inventory"},
        "gate_passed": True,
    }
    result["compatibility_sha256"] = canonical_sha256(result)
    return result


def _module_fields(module: str) -> tuple[int, str, str]:
    layer_match = LAYER_KEY.search(module)
    if layer_match is None:
        raise ValueError(f"LoRA module lacks a model layer: {module}")
    projection = module.rsplit(".", 1)[-1]
    module_type = "attention" if ".self_attn." in module else "feed_forward"
    return int(layer_match.group("layer")), projection, module_type


def _empty_aggregate() -> dict[str, float | int]:
    return {
        "module_count": 0,
        "generic_squared_norm": 0.0,
        "targeted_squared_norm": 0.0,
        "contrastive_squared_norm": 0.0,
        "generic_targeted_dot": 0.0,
        "maximum_absolute_contrastive_value": 0.0,
    }


def _update_aggregate(
    aggregate: dict[str, float | int],
    *,
    generic_squared: float,
    targeted_squared: float,
    contrastive_squared: float,
    dot: float,
    maximum: float,
) -> None:
    aggregate["module_count"] = int(aggregate["module_count"]) + 1
    aggregate["generic_squared_norm"] = float(aggregate["generic_squared_norm"]) + generic_squared
    aggregate["targeted_squared_norm"] = (
        float(aggregate["targeted_squared_norm"]) + targeted_squared
    )
    aggregate["contrastive_squared_norm"] = (
        float(aggregate["contrastive_squared_norm"]) + contrastive_squared
    )
    aggregate["generic_targeted_dot"] = float(aggregate["generic_targeted_dot"]) + dot
    aggregate["maximum_absolute_contrastive_value"] = max(
        float(aggregate["maximum_absolute_contrastive_value"]), maximum
    )


def _finalize_aggregate(aggregate: dict[str, float | int]) -> dict[str, float | int]:
    generic = math.sqrt(float(aggregate["generic_squared_norm"]))
    targeted = math.sqrt(float(aggregate["targeted_squared_norm"]))
    contrastive = math.sqrt(float(aggregate["contrastive_squared_norm"]))
    denominator = generic * targeted
    return {
        "module_count": int(aggregate["module_count"]),
        "generic_update_frobenius_norm": generic,
        "targeted_update_frobenius_norm": targeted,
        "contrastive_update_frobenius_norm": contrastive,
        "generic_targeted_cosine_similarity": (
            float(aggregate["generic_targeted_dot"]) / denominator if denominator else 0.0
        ),
        "contrastive_to_targeted_norm_ratio": contrastive / targeted if targeted else 0.0,
        "contrastive_to_generic_norm_ratio": contrastive / generic if generic else 0.0,
        "maximum_absolute_contrastive_value": float(
            aggregate["maximum_absolute_contrastive_value"]
        ),
    }


def analyze_dense_updates(
    *, generic_path: Path, targeted_path: Path, protocol_path: Path
) -> dict[str, Any]:
    """Stream all dense FP32 source updates and summarize their exact difference."""

    protocol = load_protocol(protocol_path)
    generic_info = inspect_adapter(generic_path)
    targeted_info = inspect_adapter(targeted_path)
    compatibility = validate_source_compatibility(
        generic=generic_info, targeted=targeted_info, protocol=protocol
    )
    torch = importlib.import_module("torch")
    safetensors = importlib.import_module("safetensors")
    source = cast(dict[str, Any], protocol["source_contract"])
    scaling = float(source["alpha"]) / float(source["rank"])
    records = cast(list[dict[str, object]], generic_info["tensor_inventory"])
    modules = sorted(_paired_modules(records))
    aggregates: dict[str, dict[str, dict[str, float | int]]] = {
        "by_model_layer": defaultdict(_empty_aggregate),
        "by_projection_type": defaultdict(_empty_aggregate),
        "by_module_type": defaultdict(_empty_aggregate),
    }
    global_aggregate = _empty_aggregate()
    per_module: list[dict[str, Any]] = []
    generic_weights = generic_path / "adapter_model.safetensors"
    targeted_weights = targeted_path / "adapter_model.safetensors"
    with (
        safetensors.safe_open(str(generic_weights), framework="pt", device="cpu") as generic_handle,
        safetensors.safe_open(
            str(targeted_weights), framework="pt", device="cpu"
        ) as targeted_handle,
    ):
        for module in modules:
            generic_a = generic_handle.get_tensor(f"{module}.lora_A.weight").float()
            generic_b = generic_handle.get_tensor(f"{module}.lora_B.weight").float()
            targeted_a = targeted_handle.get_tensor(f"{module}.lora_A.weight").float()
            targeted_b = targeted_handle.get_tensor(f"{module}.lora_B.weight").float()
            generic_delta = (generic_b @ generic_a) * scaling
            targeted_delta = (targeted_b @ targeted_a) * scaling
            contrastive_delta = targeted_delta - generic_delta
            finite = bool(
                torch.isfinite(generic_delta).all()
                and torch.isfinite(targeted_delta).all()
                and torch.isfinite(contrastive_delta).all()
            )
            if not finite:
                raise ValueError(f"non-finite dense update: {module}")
            generic_squared = float(torch.sum(generic_delta.double().square()).item())
            targeted_squared = float(torch.sum(targeted_delta.double().square()).item())
            contrastive_squared = float(torch.sum(contrastive_delta.double().square()).item())
            dot = float(torch.sum(generic_delta.double() * targeted_delta.double()).item())
            maximum = float(contrastive_delta.abs().max().item())
            generic_norm = math.sqrt(generic_squared)
            targeted_norm = math.sqrt(targeted_squared)
            contrastive_norm = math.sqrt(contrastive_squared)
            layer, projection, module_type = _module_fields(module)
            per_module.append(
                {
                    "module": module,
                    "model_layer": layer,
                    "projection_type": projection,
                    "module_type": module_type,
                    "generic_update_frobenius_norm": generic_norm,
                    "targeted_update_frobenius_norm": targeted_norm,
                    "contrastive_update_frobenius_norm": contrastive_norm,
                    "generic_targeted_cosine_similarity": (
                        dot / (generic_norm * targeted_norm)
                        if generic_norm and targeted_norm
                        else 0.0
                    ),
                    "contrastive_to_targeted_norm_ratio": (
                        contrastive_norm / targeted_norm if targeted_norm else 0.0
                    ),
                    "contrastive_to_generic_norm_ratio": (
                        contrastive_norm / generic_norm if generic_norm else 0.0
                    ),
                    "maximum_absolute_contrastive_value": maximum,
                    "finite": finite,
                }
            )
            for aggregate in (
                global_aggregate,
                aggregates["by_model_layer"][str(layer)],
                aggregates["by_projection_type"][projection],
                aggregates["by_module_type"][module_type],
            ):
                _update_aggregate(
                    aggregate,
                    generic_squared=generic_squared,
                    targeted_squared=targeted_squared,
                    contrastive_squared=contrastive_squared,
                    dot=dot,
                    maximum=maximum,
                )
            del generic_delta, targeted_delta, contrastive_delta
    aggregate_result = _finalize_aggregate(global_aggregate)
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": "foundry-contrastive-dense-delta-analysis-v1",
        "protocol_sha256": protocol["config_sha256"],
        "compatibility": compatibility,
        "contrastive_definition": {
            "formula": "delta_targeted_minus_delta_generic",
            "targeted_weight": 1.0,
            "generic_weight": -1.0,
            "source_scaling": scaling,
        },
        "contrastive_definition_sha256": canonical_sha256(
            {
                "formula": "delta_targeted_minus_delta_generic",
                "targeted_weight": 1.0,
                "generic_weight": -1.0,
                "source_scaling": scaling,
            }
        ),
        "aggregate": aggregate_result,
        "targeted_magnitude_represented_by_contrastive_percent": (
            float(aggregate_result["contrastive_to_targeted_norm_ratio"]) * 100.0
        ),
        "by_model_layer": {
            key: _finalize_aggregate(value)
            for key, value in sorted(
                aggregates["by_model_layer"].items(), key=lambda item: int(item[0])
            )
        },
        "by_projection_type": {
            key: _finalize_aggregate(value)
            for key, value in sorted(aggregates["by_projection_type"].items())
        },
        "by_module_type": {
            key: _finalize_aggregate(value)
            for key, value in sorted(aggregates["by_module_type"].items())
        },
        "per_module": per_module,
        "all_values_finite": True,
        "sealed_final_accessed": False,
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _validate_write_paths(
    *,
    write_targets: dict[str, Path],
    protected_trees: dict[str, Path],
    protected_files: dict[str, Path],
) -> None:
    """Fail closed when any output could overwrite or contain a frozen input."""

    resolved_writes = {name: path.resolve() for name, path in write_targets.items()}
    for write_name, write_path in resolved_writes.items():
        for protected_name, protected_path in protected_trees.items():
            if _paths_overlap(write_path, protected_path):
                raise ValueError(
                    f"{write_name} overlaps protected {protected_name}: {protected_path}"
                )
        for protected_name, protected_path in protected_files.items():
            if _paths_overlap(write_path, protected_path):
                raise ValueError(
                    f"{write_name} overlaps protected {protected_name}: {protected_path}"
                )
    items = list(resolved_writes.items())
    for index, (left_name, left_path) in enumerate(items):
        for right_name, right_path in items[index + 1 :]:
            if _paths_overlap(left_path, right_path):
                raise ValueError(f"write targets overlap: {left_name} and {right_name}")


def _lora_modules(model: Any, adapter_names: tuple[str, ...]) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        scaling = getattr(module, "scaling", None)
        if lora_a is None or lora_b is None or not isinstance(scaling, dict):
            continue
        if all(
            adapter in lora_a and adapter in lora_b and adapter in scaling
            for adapter in adapter_names
        ):
            if bool(getattr(module, "merged", False)):
                raise ValueError(f"merged LoRA module is prohibited: {name}")
            result.append((name, module))
    return sorted(result)


def _materialize_exact_fp32_cat(
    model: Any,
    torch: Any,
    *,
    generic_path: Path,
    targeted_path: Path,
) -> list[tuple[str, Any]]:
    """Replace PEFT's cat factors with exact FP32 factors from the frozen files.

    PEFT may cast already-loaded source factors to the base-model dtype while it
    constructs a weighted adapter.  The immutable source safetensors files are
    therefore the only authoritative input for exact adapter arithmetic.
    """

    safetensors = importlib.import_module("safetensors")
    modules = _lora_modules(model, (GENERIC_NAME, TARGETED_NAME, CONTRASTIVE_NAME))
    with (
        safetensors.safe_open(
            str(generic_path / "adapter_model.safetensors"), framework="pt", device="cpu"
        ) as generic,
        safetensors.safe_open(
            str(targeted_path / "adapter_model.safetensors"), framework="pt", device="cpu"
        ) as targeted,
        torch.no_grad(),
    ):
        for name, module in modules:
            target_a = module.lora_A[CONTRASTIVE_NAME].to(dtype=torch.float32)
            target_b = module.lora_B[CONTRASTIVE_NAME].to(dtype=torch.float32)
            device = target_a.weight.device
            generic_a = generic.get_tensor(f"{name}.lora_A.weight").to(
                device=device, dtype=torch.float32
            )
            generic_b = generic.get_tensor(f"{name}.lora_B.weight").to(
                device=device, dtype=torch.float32
            )
            targeted_a = targeted.get_tensor(f"{name}.lora_A.weight").to(
                device=device, dtype=torch.float32
            )
            targeted_b = targeted.get_tensor(f"{name}.lora_B.weight").to(
                device=device, dtype=torch.float32
            )
            exact_a = torch.cat(
                [
                    targeted_a * float(module.scaling[TARGETED_NAME]),
                    -generic_a * float(module.scaling[GENERIC_NAME]),
                ],
                dim=0,
            )
            exact_b = torch.cat([targeted_b, generic_b], dim=1)
            target_a.weight.copy_(exact_a)
            target_b.weight.copy_(exact_b)
    return modules


def _functional_logits(model: Any, tokenizer: Any, torch: Any, prompt_count: int) -> Any:
    prompts = DIAGNOSTIC_PROMPTS[:prompt_count]
    tokenizer.padding_side = "right"
    rendered = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DIAGNOSTIC_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    encoded = tokenizer(rendered, add_special_tokens=False, padding=True, return_tensors="pt").to(
        "cuda:0"
    )
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits
    lengths = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, lengths].float().cpu()


def _direct_dense_reference_logits(
    model: Any, tokenizer: Any, torch: Any, prompt_count: int
) -> Any:
    functional = importlib.import_module("torch.nn.functional")
    modules = _lora_modules(model, (CONTRASTIVE_NAME,))
    handles = []

    def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        value = inputs[0]
        factor_a = module.lora_A[CONTRASTIVE_NAME].weight.float()
        factor_b = module.lora_B[CONTRASTIVE_NAME].weight.float()
        source_rank = factor_a.shape[0] // 2
        targeted_delta = factor_b[:, :source_rank] @ factor_a[:source_rank, :]
        negative_generic_delta = factor_b[:, source_rank:] @ factor_a[source_rank:, :]
        direct_delta = (targeted_delta + negative_generic_delta) * float(
            module.scaling[CONTRASTIVE_NAME]
        )
        contribution = functional.linear(value.float(), direct_delta)
        return (output + contribution).to(output.dtype)

    try:
        handles = [module.register_forward_hook(hook) for _, module in modules]
        with model.disable_adapter():
            return _functional_logits(model, tokenizer, torch, prompt_count)
    finally:
        for handle in handles:
            handle.remove()


def _canonicalize_saved_config(adapter_path: Path) -> None:
    config_path = adapter_path / "adapter_config.json"
    config = _object(config_path)
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list):
        raise ValueError("saved contrastive adapter lacks target modules")
    config["target_modules"] = sorted(str(item) for item in target_modules)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_saved_dense_equivalence(
    *,
    generic_path: Path,
    targeted_path: Path,
    contrastive_path: Path,
    maximum_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    safetensors = importlib.import_module("safetensors")
    generic_info = inspect_adapter(generic_path)
    contrastive_info = inspect_adapter(contrastive_path)
    modules = sorted(
        _paired_modules(cast(list[dict[str, object]], generic_info["tensor_inventory"]))
    )
    maximum_error = 0.0
    error_squared = 0.0
    reference_squared = 0.0
    per_module: list[dict[str, Any]] = []
    with (
        safetensors.safe_open(
            str(generic_path / "adapter_model.safetensors"), framework="pt", device="cpu"
        ) as generic,
        safetensors.safe_open(
            str(targeted_path / "adapter_model.safetensors"), framework="pt", device="cpu"
        ) as targeted,
        safetensors.safe_open(
            str(contrastive_path / "adapter_model.safetensors"),
            framework="pt",
            device="cpu",
        ) as contrastive,
    ):
        for module in modules:
            generic_dense = (
                generic.get_tensor(f"{module}.lora_B.weight").float()
                @ generic.get_tensor(f"{module}.lora_A.weight").float()
            ) * 2.0
            targeted_dense = (
                targeted.get_tensor(f"{module}.lora_B.weight").float()
                @ targeted.get_tensor(f"{module}.lora_A.weight").float()
            ) * 2.0
            contrastive_dense = (
                contrastive.get_tensor(f"{module}.lora_B.weight").float()
                @ contrastive.get_tensor(f"{module}.lora_A.weight").float()
            )
            reference = targeted_dense - generic_dense
            error = contrastive_dense - reference
            module_maximum = float(error.abs().max().item())
            module_error_squared = float(torch.sum(error.double().square()).item())
            module_reference_squared = float(torch.sum(reference.double().square()).item())
            module_relative = math.sqrt(module_error_squared) / max(
                math.sqrt(module_reference_squared), 1e-30
            )
            finite = bool(torch.isfinite(contrastive_dense).all() and torch.isfinite(error).all())
            passed = (
                finite
                and module_maximum <= maximum_tolerance
                and module_relative <= relative_tolerance
            )
            per_module.append(
                {
                    "module": module,
                    "maximum_absolute_error": module_maximum,
                    "relative_frobenius_error": module_relative,
                    "finite": finite,
                    "passed": passed,
                }
            )
            maximum_error = max(maximum_error, module_maximum)
            error_squared += module_error_squared
            reference_squared += module_reference_squared
    relative_error = math.sqrt(error_squared) / max(math.sqrt(reference_squared), 1e-30)
    gate_passed = (
        contrastive_info["rank"] == 32
        and contrastive_info["alpha"] == 32
        and contrastive_info["lora_module_count"] == len(modules) == 196
        and contrastive_info["saved_tensor_count"] == 392
        and all(item["passed"] for item in per_module)
        and maximum_error <= maximum_tolerance
        and relative_error <= relative_tolerance
    )
    return {
        "module_count": len(modules),
        "contrastive_rank": contrastive_info["rank"],
        "contrastive_alpha": contrastive_info["alpha"],
        "contrastive_inventory_sha256": contrastive_info["tensor_inventory_sha256"],
        "maximum_absolute_error": maximum_error,
        "relative_frobenius_error": relative_error,
        "maximum_absolute_tolerance": maximum_tolerance,
        "relative_frobenius_tolerance": relative_tolerance,
        "per_module": per_module,
        "gate_passed": gate_passed,
    }


def construct_task_vector(
    *,
    generic_path: Path,
    targeted_path: Path,
    model_path: Path,
    protocol_path: Path,
    output_parent: Path,
    raw_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    """Compose, save, reload-ready verify, and state-audit the exact task vector."""

    protocol = load_protocol(protocol_path)
    generic_info = inspect_adapter(generic_path)
    targeted_info = inspect_adapter(targeted_path)
    compatibility = validate_source_compatibility(
        generic=generic_info, targeted=targeted_info, protocol=protocol
    )
    _validate_write_paths(
        write_targets={
            "contrastive adapter output": output_parent,
            "raw diagnostic output": raw_path,
            "construction evidence output": evidence_path,
        },
        protected_trees={
            "generic source adapter": generic_path,
            "targeted source adapter": targeted_path,
            "base-model snapshot": model_path,
        },
        protected_files={"frozen protocol": protocol_path},
    )
    if output_parent.exists():
        raise FileExistsError("contrastive output path already exists")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    psutil = importlib.import_module("psutil")
    random.seed(20260720)
    torch.manual_seed(20260720)
    torch.cuda.manual_seed_all(20260720)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_model, tokenizer, base_load_seconds = _load_base(model_path, torch, transformers)
    base_snapshot_hash = directory_sha256(model_path)
    expected_base = cast(dict[str, Any], protocol["base_model"])
    if base_snapshot_hash != expected_base["snapshot_sha256"]:
        raise ValueError("base snapshot hash differs")
    base_outputs = _generate(base_model, tokenizer, torch)
    base_output_hash = _output_hash(base_outputs)
    if base_output_hash != expected_base["untouched_diagnostic_output_sha256"]:
        raise ValueError("untouched-base diagnostic output differs")

    model = peft.PeftModel.from_pretrained(
        base_model,
        str(generic_path),
        adapter_name=GENERIC_NAME,
        is_trainable=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.load_adapter(
        str(targeted_path),
        adapter_name=TARGETED_NAME,
        is_trainable=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    # PEFT 0.15.2 re-enables the previously active adapter when a second
    # inference-only adapter is loaded. Freeze all parameters explicitly;
    # this changes only autograd flags, never tensor values.
    model.requires_grad_(False)
    model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("source-adapter model was offloaded from CUDA")
    if any(bool(getattr(module, "merged", False)) for module in model.modules()):
        raise RuntimeError("source adapter was unexpectedly merged")
    source_state_before = {
        GENERIC_NAME: adapter_state_sha256(model, GENERIC_NAME),
        TARGETED_NAME: adapter_state_sha256(model, TARGETED_NAME),
    }
    base_signature_before = base_parameter_signature_sha256(model)
    model.set_adapter(GENERIC_NAME)
    generic_before = _generate(model, tokenizer, torch)
    model.set_adapter(TARGETED_NAME)
    targeted_before = _generate(model, tokenizer, torch)
    with model.disable_adapter():
        disabled_before = _generate(model, tokenizer, torch)
    source_config = cast(dict[str, dict[str, Any]], protocol["source_adapters"])
    if (
        _output_hash(generic_before) != source_config[GENERIC_NAME]["diagnostic_output_sha256"]
        or _output_hash(targeted_before) != source_config[TARGETED_NAME]["diagnostic_output_sha256"]
        or disabled_before != base_outputs
    ):
        raise ValueError("source-adapter diagnostic compatibility failed")

    cast(Any, model).add_weighted_adapter(
        adapters=[TARGETED_NAME, GENERIC_NAME],
        weights=[1.0, -1.0],
        adapter_name=CONTRASTIVE_NAME,
        combination_type="cat",
    )
    modules = _materialize_exact_fp32_cat(
        model,
        torch,
        generic_path=generic_path,
        targeted_path=targeted_path,
    )
    model.set_adapter(CONTRASTIVE_NAME)
    model.requires_grad_(False)
    composition = cast(dict[str, Any], protocol["composition"])
    rank_and_scaling = all(
        module.lora_A[CONTRASTIVE_NAME].weight.shape[0] == composition["expected_rank"]
        and module.lora_B[CONTRASTIVE_NAME].weight.shape[1] == composition["expected_rank"]
        and float(module.scaling[CONTRASTIVE_NAME]) == composition["expected_scaling"]
        for _, module in modules
    )
    if len(modules) != 196 or not rank_and_scaling:
        raise ValueError("contrastive in-memory rank, scaling, or module inventory differs")
    lora_module_count = len(modules)
    merged_module_count = sum(bool(getattr(module, "merged", False)) for _, module in modules)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("contrastive inference model unexpectedly has trainable parameters")

    contrastive_outputs = _generate(model, tokenizer, torch)
    with model.disable_adapter():
        disabled_contrastive = _generate(model, tokenizer, torch)
    with scaled_lora_adapter(model, 0.0) as zero_evidence:
        zero_outputs = _generate(model, tokenizer, torch)
    with scaled_lora_adapter(model, 1.0) as one_evidence:
        one_outputs = _generate(model, tokenizer, torch)
    if not (
        disabled_contrastive == base_outputs
        and zero_outputs == base_outputs
        and one_outputs == contrastive_outputs
        and zero_evidence.original_scaling_restored
        and zero_evidence.adapter_state_unchanged
        and zero_evidence.base_parameter_signature_unchanged
        and one_evidence.original_scaling_restored
        and one_evidence.adapter_state_unchanged
        and one_evidence.base_parameter_signature_unchanged
    ):
        raise ValueError("contrastive scale-zero or scale-one sanity failed")

    model.set_adapter(GENERIC_NAME)
    generic_after = _generate(model, tokenizer, torch)
    model.set_adapter(TARGETED_NAME)
    targeted_after = _generate(model, tokenizer, torch)
    source_state_after = {
        GENERIC_NAME: adapter_state_sha256(model, GENERIC_NAME),
        TARGETED_NAME: adapter_state_sha256(model, TARGETED_NAME),
    }
    base_signature_after = base_parameter_signature_sha256(model)
    source_directories_after = {
        GENERIC_NAME: directory_sha256(generic_path),
        TARGETED_NAME: directory_sha256(targeted_path),
    }
    if (
        source_state_after != source_state_before
        or base_signature_after != base_signature_before
        or generic_after != generic_before
        or targeted_after != targeted_before
        or source_directories_after[GENERIC_NAME] != generic_info["directory_sha256"]
        or source_directories_after[TARGETED_NAME] != targeted_info["directory_sha256"]
    ):
        raise ValueError("source adapter or base state changed during composition")

    output_parent.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        str(output_parent), selected_adapters=[CONTRASTIVE_NAME], safe_serialization=True
    )
    contrastive_path = output_parent / CONTRASTIVE_NAME
    _canonicalize_saved_config(contrastive_path)
    saved_hash = directory_sha256(contrastive_path)
    equivalence_config = cast(dict[str, Any], protocol["equivalence"])
    dense_equivalence = _verify_saved_dense_equivalence(
        generic_path=generic_path,
        targeted_path=targeted_path,
        contrastive_path=contrastive_path,
        maximum_tolerance=float(equivalence_config["dense_maximum_absolute_error"]),
        relative_tolerance=float(equivalence_config["dense_relative_frobenius_error"]),
    )
    # Functional equivalence is a numerical proof, not the production inference
    # path.  Verify it with an FP32 base so low-rank versus dense GEMM association
    # error is not amplified by FP16 residual-stream rounding.  The saved adapter
    # remains the same exact FP32, unmerged rank-32 artifact used at inference.
    if equivalence_config.get("functional_verification_base_dtype") != "float32":
        raise ValueError("functional verification base dtype differs")
    del modules, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    verification_load_started = time.perf_counter()
    verification_base = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    verification_model = peft.PeftModel.from_pretrained(
        verification_base,
        str(contrastive_path),
        adapter_name=CONTRASTIVE_NAME,
        is_trainable=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    verification_model.requires_grad_(False)
    verification_model.eval()
    functional_verification_load_seconds = time.perf_counter() - verification_load_started
    if any(parameter.device.type != "cuda" for parameter in verification_model.parameters()):
        raise RuntimeError("functional verification model was offloaded from CUDA")
    if any(bool(getattr(module, "merged", False)) for module in verification_model.modules()):
        raise RuntimeError("functional verification adapter was unexpectedly merged")
    verification_adapter_state_before = adapter_state_sha256(verification_model, CONTRASTIVE_NAME)
    verification_base_state_before = base_parameter_signature_sha256(verification_model)
    composed_logits = _functional_logits(
        verification_model,
        tokenizer,
        torch,
        int(equivalence_config["diagnostic_prompt_count"]),
    )
    direct_logits = _direct_dense_reference_logits(
        verification_model,
        tokenizer,
        torch,
        int(equivalence_config["diagnostic_prompt_count"]),
    )
    verification_adapter_state_after = adapter_state_sha256(verification_model, CONTRASTIVE_NAME)
    verification_base_state_after = base_parameter_signature_sha256(verification_model)
    if (
        verification_adapter_state_after != verification_adapter_state_before
        or verification_base_state_after != verification_base_state_before
        or directory_sha256(contrastive_path) != saved_hash
    ):
        raise ValueError("functional verification changed adapter or base state")
    logit_error = composed_logits - direct_logits
    functional_maximum = float(logit_error.abs().max().item())
    functional_relative = float(logit_error.double().norm().item()) / max(
        float(direct_logits.double().norm().item()), 1e-30
    )
    functional_passed = functional_maximum <= float(
        equivalence_config["functional_maximum_absolute_logit_error"]
    ) and functional_relative <= float(equivalence_config["functional_relative_frobenius_error"])
    functional = {
        "prompt_count": int(equivalence_config["diagnostic_prompt_count"]),
        "prompt_sha256": canonical_sha256(
            {
                "system": DIAGNOSTIC_SYSTEM,
                "prompts": list(
                    DIAGNOSTIC_PROMPTS[: int(equivalence_config["diagnostic_prompt_count"])]
                ),
            }
        ),
        "composed_logits_sha256": canonical_sha256(composed_logits.tolist()),
        "direct_dense_logits_sha256": canonical_sha256(direct_logits.tolist()),
        "maximum_absolute_logit_error": functional_maximum,
        "relative_frobenius_error": functional_relative,
        "maximum_absolute_tolerance": equivalence_config["functional_maximum_absolute_logit_error"],
        "relative_frobenius_tolerance": equivalence_config["functional_relative_frobenius_error"],
        "verification_base_dtype": "float32",
        "verification_model_load_seconds": functional_verification_load_seconds,
        "adapter_state_unchanged": (
            verification_adapter_state_before == verification_adapter_state_after
        ),
        "base_state_unchanged": verification_base_state_before == verification_base_state_after,
        "gate_passed": functional_passed,
    }
    raw = {
        "base_outputs": base_outputs,
        "generic_source_outputs": generic_before,
        "targeted_source_outputs": targeted_before,
        "contrastive_outputs": contrastive_outputs,
        "scale_zero_outputs": zero_outputs,
        "scale_one_outputs": one_outputs,
    }
    _write_json(raw_path, raw)
    if directory_sha256(contrastive_path) != saved_hash:
        raise ValueError("contrastive adapter hash changed after writing diagnostic evidence")
    process = psutil.Process()
    memory = process.memory_info()
    result: dict[str, Any] = {
        "schema_version": 1,
        "construction_id": "foundry-targeted-minus-generic-task-vector-construction-v1",
        "protocol_sha256": protocol["config_sha256"],
        "compatibility_sha256": compatibility["compatibility_sha256"],
        "source_adapter_sha256s": {
            GENERIC_NAME: generic_info["directory_sha256"],
            TARGETED_NAME: targeted_info["directory_sha256"],
        },
        "source_state_sha256_before": source_state_before,
        "source_state_sha256_after": source_state_after,
        "source_directories_unchanged": True,
        "base_snapshot_sha256": base_snapshot_hash,
        "base_parameter_signature_before": base_signature_before,
        "base_parameter_signature_after": base_signature_after,
        "base_state_unchanged": base_signature_before == base_signature_after,
        "composition": composition,
        "contrastive_adapter_relative_path": CONTRASTIVE_NAME,
        "contrastive_adapter_sha256": saved_hash,
        "contrastive_rank": 32,
        "contrastive_alpha": 32,
        "contrastive_scaling": 1.0,
        "lora_module_count": lora_module_count,
        "merged_module_count": merged_module_count,
        "dense_equivalence": dense_equivalence,
        "dense_equivalence_sha256": canonical_sha256(dense_equivalence),
        "functional_logit_equivalence": functional,
        "functional_equivalence_sha256": canonical_sha256(functional),
        "scale_zero_sanity": {
            "matches_untouched_base": zero_outputs == base_outputs,
            "output_sha256": _output_hash(zero_outputs),
            "state": zero_evidence.as_dict(),
        },
        "scale_one_sanity": {
            "matches_unscaled_contrastive": one_outputs == contrastive_outputs,
            "output_sha256": _output_hash(one_outputs),
            "state": one_evidence.as_dict(),
        },
        "diagnostic_output_sha256s": {
            "base": base_output_hash,
            GENERIC_NAME: _output_hash(generic_after),
            TARGETED_NAME: _output_hash(targeted_after),
            CONTRASTIVE_NAME: _output_hash(contrastive_outputs),
        },
        "raw_packet_sha256": file_sha256(raw_path),
        "base_load_seconds": base_load_seconds,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "process_working_set_bytes": int(memory.rss),
        "process_peak_working_set_bytes": int(getattr(memory, "peak_wset", memory.rss)),
        "offline_only": True,
        "unmerged_and_reversible": True,
        "sealed_final_accessed": False,
        "gate_passed": bool(dense_equivalence["gate_passed"] and functional_passed),
    }
    result["equivalence_evidence_sha256"] = canonical_sha256(
        {
            "dense": dense_equivalence,
            "functional": functional,
            "scale_zero": result["scale_zero_sanity"],
            "scale_one": result["scale_one_sanity"],
            "source_state_unchanged": source_state_before == source_state_after,
            "base_state_unchanged": base_signature_before == base_signature_after,
        }
    )
    result["summary_sha256"] = canonical_sha256(result)
    del verification_model, verification_base
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--generic-adapter", required=True, type=Path)
    analyze.add_argument("--targeted-adapter", required=True, type=Path)
    analyze.add_argument("--protocol", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    construct = subparsers.add_parser("construct")
    construct.add_argument("--generic-adapter", required=True, type=Path)
    construct.add_argument("--targeted-adapter", required=True, type=Path)
    construct.add_argument("--model-path", required=True, type=Path)
    construct.add_argument("--protocol", required=True, type=Path)
    construct.add_argument("--output-parent", required=True, type=Path)
    construct.add_argument("--raw-path", required=True, type=Path)
    construct.add_argument("--evidence-output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "analyze":
        _validate_write_paths(
            write_targets={"dense-analysis output": args.output},
            protected_trees={
                "generic source adapter": args.generic_adapter,
                "targeted source adapter": args.targeted_adapter,
            },
            protected_files={"frozen protocol": args.protocol},
        )
        result = analyze_dense_updates(
            generic_path=args.generic_adapter,
            targeted_path=args.targeted_adapter,
            protocol_path=args.protocol,
        )
        _write_json(args.output, result)
    else:
        result = construct_task_vector(
            generic_path=args.generic_adapter,
            targeted_path=args.targeted_adapter,
            model_path=args.model_path,
            protocol_path=args.protocol,
            output_parent=args.output_parent,
            raw_path=args.raw_path,
            evidence_path=args.evidence_output,
        )
        _write_json(args.evidence_output, result)
        contrastive_path = args.output_parent / CONTRASTIVE_NAME
        if directory_sha256(contrastive_path) != result["contrastive_adapter_sha256"]:
            raise ValueError("contrastive adapter hash changed after writing construction evidence")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
