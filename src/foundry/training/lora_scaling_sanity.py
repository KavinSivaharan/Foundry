"""Offline scale-zero and scale-one identity audit for frozen LoRA adapters."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
from typing import Any

from foundry.training.adapter_sanity import _generate, _load_base, _output_hash
from foundry.training.config import canonical_sha256
from foundry.training.lora_scaling import scaled_lora_adapter
from foundry.training.qlora import directory_sha256, file_sha256


def _load_config(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("common scaling configuration must be an object")
    config = dict(value)
    expected = config.pop("config_sha256", None)
    if expected != canonical_sha256(config):
        raise ValueError("common scaling configuration hash differs")
    return dict(value)


def _audit_arm(
    *,
    name: str,
    model_path: Path,
    adapter_path: Path,
    raw_path: Path,
    modules: dict[str, Any],
) -> dict[str, Any]:
    torch = modules["torch"]
    model, tokenizer, load_seconds = _load_base(model_path, torch, modules["transformers"])
    model = modules["peft"].PeftModel.from_pretrained(
        model,
        str(adapter_path),
        adapter_name="default",
        is_trainable=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("scaled sanity model was offloaded from CUDA")
    active_before = list(model.active_adapters)
    directory_before = directory_sha256(adapter_path)
    unscaled_outputs = _generate(model, tokenizer, torch)
    with model.disable_adapter():
        base_outputs = _generate(model, tokenizer, torch)
    with scaled_lora_adapter(model, 0.0) as zero_evidence:
        zero_outputs = _generate(model, tokenizer, torch)
    with scaled_lora_adapter(model, 1.0) as one_evidence:
        one_outputs = _generate(model, tokenizer, torch)
    active_after = list(model.active_adapters)
    directory_after = directory_sha256(adapter_path)
    raw = {
        "arm": name,
        "base_outputs": base_outputs,
        "unscaled_outputs": unscaled_outputs,
        "scale_zero_outputs": zero_outputs,
        "scale_one_outputs": one_outputs,
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "arm": name,
        "adapter_sha256_before": directory_before,
        "adapter_sha256_after": directory_after,
        "active_adapters_before": active_before,
        "active_adapters_after": active_after,
        "unscaled_output_sha256": _output_hash(unscaled_outputs),
        "untouched_base_output_sha256": _output_hash(base_outputs),
        "scale_zero_output_sha256": _output_hash(zero_outputs),
        "scale_one_output_sha256": _output_hash(one_outputs),
        "scale_zero_matches_untouched_base": zero_outputs == base_outputs,
        "scale_one_matches_unscaled_adapter": one_outputs == unscaled_outputs,
        "scale_zero_state": zero_evidence.as_dict(),
        "scale_one_state": one_evidence.as_dict(),
        "raw_packet_sha256": file_sha256(raw_path),
        "load_seconds": load_seconds,
    }
    result["gate_passed"] = bool(
        active_before == ["default"]
        and active_after == active_before
        and directory_after == directory_before
        and zero_outputs == base_outputs
        and one_outputs == unscaled_outputs
        and zero_evidence.original_scaling_restored
        and zero_evidence.adapter_state_unchanged
        and zero_evidence.base_parameter_signature_unchanged
        and one_evidence.original_scaling_restored
        and one_evidence.adapter_state_unchanged
        and one_evidence.base_parameter_signature_unchanged
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_sanity(
    *,
    model_path: Path,
    generic_adapter: Path,
    targeted_adapter: Path,
    config_path: Path,
    raw_directory: Path,
) -> dict[str, Any]:
    """Run the complete two-arm scale identity and restoration audit."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    modules = {
        "torch": importlib.import_module("torch"),
        "transformers": importlib.import_module("transformers"),
        "peft": importlib.import_module("peft"),
    }
    torch = modules["torch"]
    torch.manual_seed(20260720)
    torch.cuda.manual_seed_all(20260720)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = _load_config(config_path)
    arms = {
        "generic_control": _audit_arm(
            name="generic_control",
            model_path=model_path,
            adapter_path=generic_adapter,
            raw_path=raw_directory / "generic_control.json",
            modules=modules,
        ),
        "targeted": _audit_arm(
            name="targeted",
            model_path=model_path,
            adapter_path=targeted_adapter,
            raw_path=raw_directory / "targeted.json",
            modules=modules,
        ),
    }
    source_path = Path(__file__).with_name("lora_scaling.py")
    result: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-common-lora-runtime-scaling-sanity-v1",
        "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "base_snapshot_sha256": directory_sha256(model_path),
        "scaling_source_sha256": file_sha256(source_path),
        "scale_config_sha256": config["config_sha256"],
        "arms": arms,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gate_passed": all(bool(arm["gate_passed"]) for arm in arms.values()),
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--generic-adapter", required=True, type=Path)
    parser.add_argument("--targeted-adapter", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_sanity(
        model_path=args.model_path,
        generic_adapter=args.generic_adapter,
        targeted_adapter=args.targeted_adapter,
        config_path=args.config,
        raw_directory=args.raw_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
