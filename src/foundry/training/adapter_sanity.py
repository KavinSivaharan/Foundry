"""Offline enabled/disabled sanity checks for preserved collapsed LoRA adapters."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from foundry.training.config import TARGET_MODULES, canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256

DIAGNOSTIC_SYSTEM = (
    "Follow the user's deterministic instruction. Respond directly and do not invent a new task."
)
DIAGNOSTIC_PROMPTS = (
    "Add 17 and 25. End with exactly: Final answer: 42",
    "Subtract 19 from 64. End with exactly: Final answer: 45",
    "Return exactly this one word and nothing else: cobalt",
    'Return exactly this JSON object and nothing else: {"ready":true}',
    "Write the three letters A, B, and C separated by commas, with no other text.",
    "State the integer immediately after 98 and provide no explanation.",
)
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def diagnostic_prompt_sha256() -> str:
    """Hash the original non-benchmark diagnostic prompt set."""

    return canonical_sha256({"system": DIAGNOSTIC_SYSTEM, "prompts": list(DIAGNOSTIC_PROMPTS)})


def _load_base(model_path: Path, torch: Any, transformers: Any) -> tuple[Any, Any, float]:
    started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("base model was offloaded from CUDA")
    model.eval()
    return model, tokenizer, time.perf_counter() - started


def _generate(model: Any, tokenizer: Any, torch: Any) -> list[str]:
    outputs: list[str] = []
    for prompt in DIAGNOSTIC_PROMPTS:
        messages = [
            {"role": "system", "content": DIAGNOSTIC_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda:0")
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=96,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response_ids = generated[0, input_ids.shape[-1] :]
        outputs.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    return outputs


def _output_hash(outputs: list[str]) -> str:
    return canonical_sha256(outputs)


def _adapter_inventory(adapter_path: Path, safetensors: Any) -> dict[str, Any]:
    weights_path = adapter_path / "adapter_model.safetensors"
    tensors: dict[str, Any] = {}
    with safetensors.safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        for key in keys:
            tensors[key] = handle.get_tensor(key)
    unexpected = [
        key
        for key in keys
        if not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"))
    ]
    layer_norms: dict[str, list[float]] = {}
    for key, tensor in tensors.items():
        match = LAYER_PATTERN.search(key)
        if match is None:
            unexpected.append(key)
            continue
        layer_norms.setdefault(match.group(1), []).append(float(tensor.float().norm().item()))
    expected = {
        (
            f"base_model.model.model.layers.{layer}."
            f"{'self_attn' if module in {'q_proj', 'k_proj', 'v_proj', 'o_proj'} else 'mlp'}."
            f"{module}.lora_{side}.weight"
        )
        for layer in range(28)
        for module in TARGET_MODULES
        for side in ("A", "B")
    }
    missing = sorted(expected - set(keys))
    unexpected = sorted(set(unexpected) | (set(keys) - expected))
    norm_summary = {
        layer: {
            "tensor_count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "mean": sum(values) / len(values),
        }
        for layer, values in sorted(layer_norms.items(), key=lambda item: int(item[0]))
    }
    return {
        "weight_file_sha256": file_sha256(weights_path),
        "saved_tensor_count": len(keys),
        "missing_expected_keys": missing,
        "unexpected_saved_keys": unexpected,
        "layer_norms": norm_summary,
    }


def _cleanup(model: Any, torch: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def audit_adapter(
    *,
    name: str,
    model_path: Path,
    adapter_path: Path,
    base_outputs: list[str],
    modules: dict[str, Any],
    raw_path: Path,
) -> dict[str, Any]:
    """Check one adapter's state inventory and enabled/disabled generation."""

    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    model, tokenizer, load_seconds = _load_base(model_path, torch, transformers)
    model = peft.PeftModel.from_pretrained(
        model,
        str(adapter_path),
        adapter_name="default",
        is_trainable=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError(f"{name} adapter caused CPU or disk offload")
    active = list(model.active_adapters)
    enabled_outputs = _generate(model, tokenizer, torch)
    with model.disable_adapter():
        disabled_outputs = _generate(model, tokenizer, torch)
    reenabled_outputs = _generate(model, tokenizer, torch)

    config = model.peft_config["default"]
    lora_modules = [module for module in model.modules() if hasattr(module, "lora_A")]
    adapter_names = sorted(
        {adapter_name for module in lora_modules for adapter_name in list(module.lora_A.keys())}
    )
    merged_modules = sum(bool(getattr(module, "merged", False)) for module in lora_modules)
    inventory = _adapter_inventory(adapter_path, modules["safetensors"])
    state = peft.get_peft_model_state_dict(model, adapter_name="default")
    loaded_state_count_matches = len(state) == inventory["saved_tensor_count"]
    scaling_values = sorted({float(module.scaling["default"]) for module in lora_modules})
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "name": name,
        "base_outputs": base_outputs,
        "enabled_outputs": enabled_outputs,
        "disabled_outputs": disabled_outputs,
        "reenabled_outputs": reenabled_outputs,
    }
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "adapter": name,
        "adapter_directory_sha256": directory_sha256(adapter_path),
        "active_adapters": active,
        "adapter_names_in_modules": adapter_names,
        "lora_module_count": len(lora_modules),
        "merged_lora_modules": merged_modules,
        "rank": int(config.r),
        "alpha": int(config.lora_alpha),
        "dropout": float(config.lora_dropout),
        "scaling_values": scaling_values,
        "expected_scaling": 2.0,
        "enabled_output_sha256": _output_hash(enabled_outputs),
        "disabled_output_sha256": _output_hash(disabled_outputs),
        "reenabled_output_sha256": _output_hash(reenabled_outputs),
        "disabled_matches_untouched_base": disabled_outputs == base_outputs,
        "reenabled_matches_enabled": reenabled_outputs == enabled_outputs,
        "enabled_differs_from_base_count": sum(
            enabled != base for enabled, base in zip(enabled_outputs, base_outputs, strict=True)
        ),
        "loaded_state_count_matches_saved": loaded_state_count_matches,
        "raw_packet_sha256": file_sha256(raw_path),
        "base_plus_adapter_load_seconds": load_seconds,
        **inventory,
    }
    result["gate_passed"] = bool(
        active == ["default"]
        and adapter_names == ["default"]
        and merged_modules == 0
        and not inventory["missing_expected_keys"]
        and not inventory["unexpected_saved_keys"]
        and loaded_state_count_matches
        and scaling_values == [2.0]
        and disabled_outputs == base_outputs
        and reenabled_outputs == enabled_outputs
        and result["enabled_differs_from_base_count"] > 0
    )
    result["aggregate_sha256"] = canonical_sha256(result)
    _cleanup(model, torch)
    return result


def run_audit(
    *,
    model_path: Path,
    generic_adapter: Path,
    targeted_adapter: Path,
    raw_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run untouched-base and two collapsed-adapter sanity checks."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    modules = {
        "torch": importlib.import_module("torch"),
        "transformers": importlib.import_module("transformers"),
        "peft": importlib.import_module("peft"),
        "safetensors": importlib.import_module("safetensors"),
    }
    torch = modules["torch"]
    torch.manual_seed(20260720)
    torch.cuda.manual_seed_all(20260720)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    base, tokenizer, base_load_seconds = _load_base(model_path, torch, modules["transformers"])
    base_outputs = _generate(base, tokenizer, torch)
    raw_directory.mkdir(parents=True, exist_ok=True)
    base_raw = raw_directory / "untouched_base.json"
    base_raw.write_text(json.dumps({"outputs": base_outputs}, indent=2) + "\n", encoding="utf-8")
    base_output_hash = _output_hash(base_outputs)
    _cleanup(base, torch)

    arms = {
        "generic_control": audit_adapter(
            name="generic_control",
            model_path=model_path,
            adapter_path=generic_adapter,
            base_outputs=base_outputs,
            modules=modules,
            raw_path=raw_directory / "generic_control.json",
        ),
        "targeted": audit_adapter(
            name="targeted",
            model_path=model_path,
            adapter_path=targeted_adapter,
            base_outputs=base_outputs,
            modules=modules,
            raw_path=raw_directory / "targeted.json",
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-collapsed-adapter-application-sanity-v1",
        "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "base_snapshot_sha256": directory_sha256(model_path),
        "diagnostic_prompt_sha256": diagnostic_prompt_sha256(),
        "diagnostic_prompt_count": len(DIAGNOSTIC_PROMPTS),
        "base_output_sha256": base_output_hash,
        "base_raw_packet_sha256": file_sha256(base_raw),
        "base_load_seconds": base_load_seconds,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "arms": arms,
        "gate_passed": all(bool(arm["gate_passed"]) for arm in arms.values()),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run the adapter sanity CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--generic-adapter", type=Path, required=True)
    parser.add_argument("--targeted-adapter", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    summary = run_audit(
        model_path=args.model_path,
        generic_adapter=args.generic_adapter,
        targeted_adapter=args.targeted_adapter,
        raw_directory=args.raw_directory,
        output_path=args.output_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
