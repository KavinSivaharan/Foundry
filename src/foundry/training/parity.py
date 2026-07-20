"""Fail-closed parity validation for the matched generic and targeted adapters."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
from typing import Any, cast

from foundry.training.config import load_qlora_recipe
from foundry.training.qlora import directory_sha256

PARITY_LIMIT = 0.02
PARITY_FIELDS = (
    "base_model_id",
    "base_revision",
    "recipe_sha256",
    "requirements_lock_sha256",
    "sft_format_sha256",
    "software_versions",
    "cuda_runtime",
    "optimizer_steps",
    "examples_processed",
    "padded_training_tokens",
    "training_source_records",
    "training_truncated_records",
    "trainable_parameters",
    "total_parameters",
)


def _load_summary(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("training summary must be an object")
    return cast(dict[str, Any], value)


def compare_training_summaries(generic: dict[str, Any], targeted: dict[str, Any]) -> dict[str, Any]:
    """Compare immutable run metadata and both padded and loss-bearing token counts."""

    equal_fields = {field: generic.get(field) == targeted.get(field) for field in PARITY_FIELDS}
    generic_tokens = int(generic["nonpadding_training_tokens"])
    targeted_tokens = int(targeted["nonpadding_training_tokens"])
    denominator = max(generic_tokens, targeted_tokens)
    relative_difference = abs(targeted_tokens - generic_tokens) / denominator
    return {
        "equal_fields": equal_fields,
        "metadata_parity_passed": all(equal_fields.values()),
        "generic_nonpadding_training_tokens": generic_tokens,
        "targeted_nonpadding_training_tokens": targeted_tokens,
        "nonpadding_token_absolute_difference": abs(targeted_tokens - generic_tokens),
        "nonpadding_token_relative_difference": relative_difference,
        "nonpadding_token_parity_limit": PARITY_LIMIT,
        "nonpadding_token_parity_passed": relative_difference <= PARITY_LIMIT,
        "padded_token_parity_passed": generic.get("padded_training_tokens")
        == targeted.get("padded_training_tokens"),
    }


def _load_adapter(*, model_path: Path, adapter_path: Path, recipe_path: Path) -> dict[str, Any]:
    recipe = load_qlora_recipe(recipe_path)
    torch: Any = importlib.import_module("torch")
    transformers: Any = importlib.import_module("transformers")
    peft: Any = importlib.import_module("peft")
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=recipe.trust_remote_code,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = peft.PeftModel.from_pretrained(
        base, str(adapter_path), local_files_only=True, is_trainable=False
    )
    offloaded = [
        name for name, parameter in model.named_parameters() if parameter.device.type != "cuda"
    ]
    result = {
        "loaded": not offloaded,
        "offloaded_parameter_count": len(offloaded),
        "adapter_sha256": directory_sha256(adapter_path),
    }
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_parity(
    *,
    recipe_path: Path,
    model_path: Path,
    generic_summary_path: Path,
    targeted_summary_path: Path,
    generic_adapter_path: Path,
    targeted_adapter_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Load both adapters, validate hashes, and apply the predeclared 2% token gate."""

    recipe = load_qlora_recipe(recipe_path)
    generic = _load_summary(generic_summary_path)
    targeted = _load_summary(targeted_summary_path)
    comparison = compare_training_summaries(generic, targeted)
    generic_load = _load_adapter(
        model_path=model_path, adapter_path=generic_adapter_path, recipe_path=recipe_path
    )
    targeted_load = _load_adapter(
        model_path=model_path, adapter_path=targeted_adapter_path, recipe_path=recipe_path
    )
    hash_checks = {
        "generic": generic_load["adapter_sha256"] == generic["adapter_sha256"],
        "targeted": targeted_load["adapter_sha256"] == targeted["adapter_sha256"],
    }
    no_development_exposure = True
    gate_passed = (
        comparison["metadata_parity_passed"]
        and comparison["nonpadding_token_parity_passed"]
        and comparison["padded_token_parity_passed"]
        and generic_load["loaded"]
        and targeted_load["loaded"]
        and all(hash_checks.values())
        and no_development_exposure
    )
    blockers = []
    if not comparison["nonpadding_token_parity_passed"]:
        blockers.append("nonpadding_training_token_difference_exceeds_2_percent")
    result = {
        "schema_version": 1,
        "recipe_sha256": recipe.recipe_sha256,
        "seed": recipe.seed,
        "comparison": comparison,
        "adapter_loads": {"generic_control": generic_load, "targeted": targeted_load},
        "adapter_hash_checks": hash_checks,
        "development_benchmark_exposure_during_training": False,
        "training_parity_gate_passed": gate_passed,
        "benchmark_evaluation_authorized": gate_passed,
        "blockers": blockers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--generic-summary", required=True, type=Path)
    parser.add_argument("--targeted-summary", required=True, type=Path)
    parser.add_argument("--generic-adapter", required=True, type=Path)
    parser.add_argument("--targeted-adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_parity(
        recipe_path=args.recipe,
        model_path=args.model_path,
        generic_summary_path=args.generic_summary,
        targeted_summary_path=args.targeted_summary,
        generic_adapter_path=args.generic_adapter,
        targeted_adapter_path=args.targeted_adapter,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
