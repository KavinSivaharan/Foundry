"""Complete the frozen Phase 2 native-Windows QLoRA environment gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, cast

from foundry.phase2.pyyaml_exception import validate_evidence
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

SEED = 20260720
EXPECTED_VERSIONS = {
    "accelerate": "1.7.0",
    "bitsandbytes": "0.49.2",
    "peft": "0.15.2",
    "tokenizers": "0.21.4",
    "torch": "2.5.1+cu121",
    "transformers": "4.51.3",
    "trl": "0.17.0",
}
RECIPE = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "quantization": "nf4",
    "double_quantization": True,
    "compute_dtype": "float16",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 1e-5,
    "optimizer": "paged_adamw_8bit",
    "scheduler": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.0,
    "max_gradient_norm": 1.0,
    "max_sequence_length": 512,
    "gradient_checkpointing": True,
    "optimizer_steps": 64,
    "checkpoints": [16, 32, 64],
    "seed": SEED,
}


def _assistant_labels(tokenizer: Any, fixture: dict[str, Any]) -> tuple[Any, Any, int]:
    messages = [
        {"role": "system", "content": fixture["system_prompt"]},
        {"role": "user", "content": fixture["prompt"]},
        {"role": "assistant", "content": fixture["assistant_response"]},
    ]
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prefix_length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    eos_positions = (input_ids[0] == tokenizer.eos_token_id).nonzero().flatten()
    if not len(eos_positions):
        raise RuntimeError("official Qwen rendering has no assistant EOS")
    eos = int(eos_positions[-1].item())
    labels = input_ids.clone()
    labels[:, :prefix_length] = -100
    labels[:, eos + 1 :] = -100
    if labels[0, eos].item() != tokenizer.eos_token_id:
        raise RuntimeError("assistant EOS is not loss-bearing")
    return input_ids, labels, eos


def _probe(model_path: Path, replay_path: Path) -> dict[str, object]:
    modules = {
        name: importlib.import_module(name)
        for name in ("bitsandbytes", "peft", "psutil", "torch", "transformers")
    }
    bitsandbytes = modules["bitsandbytes"]
    peft = modules["peft"]
    torch = modules["torch"]
    transformers = modules["transformers"]
    process = modules["psutil"].Process()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fixture = cast(dict[str, Any], json.loads(replay_path.read_text(encoding="utf-8")))["items"][0]
    started = time.perf_counter()
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
    input_ids, labels, assistant_eos = _assistant_labels(tokenizer, fixture)
    input_ids = input_ids.to("cuda")
    labels = labels.to("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = bitsandbytes.optim.PagedAdamW8bit(trainable, lr=1e-5, weight_decay=0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
        ).loss
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("forward loss is nonfinite")
    loss.backward()
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if not gradients or not all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    ):
        raise RuntimeError("LoRA gradients are missing or nonfinite")
    before = [parameter.detach().clone() for parameter in trainable]
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    if not optimizer.state:
        raise RuntimeError("paged AdamW 8-bit optimizer state was not created")
    if not any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, trainable, strict=True)
    ):
        raise RuntimeError("paged AdamW 8-bit did not update LoRA parameters")
    states: list[Any] = []
    for module in model.modules():
        state = getattr(getattr(module, "weight", None), "quant_state", None)
        if state is not None:
            states.append(state)
    if {str(state.quant_type) for state in states} != {"nf4"}:
        raise RuntimeError("loaded quantization type is not uniformly NF4")
    if {bool(state.nested) for state in states} != {True}:
        raise RuntimeError("double quantization is not uniformly enabled")
    return {
        "assistant_eos_index": assistant_eos,
        "backward_finite": True,
        "compute_dtype": "float16",
        "cuda_synchronized": True,
        "double_quantization": True,
        "fixture_sha256": canonical_sha256(
            {
                "id": fixture["id"],
                "response_sha256": fixture["assistant_response_sha256"],
                "prompt_sha256": hashlib.sha256(str(fixture["prompt"]).encode("utf-8")).hexdigest(),
            }
        ),
        "forward_finite": True,
        "gradient_finite": True,
        "loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "loss_bearing_tokens": int((labels != -100).sum().item()),
        "no_cpu_offload": True,
        "optimizer": "PagedAdamW8bit",
        "optimizer_state_created": True,
        "optimizer_update": True,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "process_rss_bytes": int(process.memory_info().rss),
        "quantization": "nf4",
        "runtime_seconds": time.perf_counter() - started,
    }


def complete_gate(
    *,
    model_path: Path,
    replay_path: Path,
    exception_path: Path,
    output_path: Path,
) -> dict[str, object]:
    exception = json.loads(exception_path.read_text(encoding="utf-8"))
    validate_evidence(exception)
    versions = {name: str(importlib.import_module(name).__version__) for name in EXPECTED_VERSIONS}
    torch = importlib.import_module("torch")
    environment = {
        name: os.environ.get(name)
        for name in (
            "HF_HUB_OFFLINE",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONHASHSEED",
            "TOKENIZERS_PARALLELISM",
            "TRANSFORMERS_OFFLINE",
        )
    }
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError("frozen package versions differ")
    if platform.python_version() != "3.12.10":
        raise RuntimeError("CPython version differs")
    if (
        torch.__version__ != "2.5.1+cu121"
        or torch.version.cuda != "12.1"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3080"
        or list(torch.cuda.get_device_capability(0)) != [8, 6]
    ):
        raise RuntimeError("CUDA or GPU identity differs")
    if (
        environment
        != {
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "20260720",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
        or sys.flags.hash_randomization != 1
    ):
        raise RuntimeError("deterministic process environment differs")
    evidence: dict[str, object] = {
        "schema_version": 1,
        "decision": "pass",
        "interpreter": str(Path(sys.executable).resolve()),
        "interpreter_sha256": file_sha256(Path(sys.executable)),
        "python_version": platform.python_version(),
        "versions": versions,
        "cuda_runtime": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_device_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "process_environment": environment,
        "pyyaml_exception_evidence_sha256": exception["evidence_sha256"],
        "recipe": RECIPE,
        "recipe_sha256": canonical_sha256(RECIPE),
        "probe": _probe(model_path, replay_path),
    }
    evidence["environment_evidence_sha256"] = canonical_sha256(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--exception-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            complete_gate(
                model_path=args.model_path,
                replay_path=args.replay_path,
                exception_path=args.exception_path,
                output_path=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
