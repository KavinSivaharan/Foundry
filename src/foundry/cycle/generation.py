"""Deterministic best-of-eight generation from the frozen L3 targeted adapter."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

from foundry.cycle.contract import (
    CYCLE_ID,
    CycleConfig,
    normalized_completion,
    prompt_subseed,
    text_sha256,
    validate_file_identity,
    validate_process_environment,
)
from foundry.phase2 import vetted_qlora_kl as qlora
from foundry.training.config import assistant_only_v3_messages, canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def select_smoke_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the frozen four compatibility prompts without reading model results."""

    return sorted(
        records,
        key=lambda item: (
            hashlib.sha256(f"{CYCLE_ID}:{item['source_id']}".encode()).hexdigest(),
            str(item["source_id"]),
        ),
    )[:4]


def generate_candidates(
    *,
    config: CycleConfig,
    output_directory: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Generate exactly eight candidates for every authorized prompt."""

    validate_process_environment(config=config)
    if output_directory.exists():
        raise FileExistsError("generation output must be fresh")
    generation = config.section("generation")
    dataset = config.section("dataset")
    warm_start = config.section("warm_start")
    model_contract = config.section("model")
    records = _jsonl(
        validate_file_identity(
            config,
            str(dataset["targeted_training_relative_path"]),
            str(dataset["targeted_training_sha256"]),
        )
    )
    if len(records) != int(dataset["training_records"]):
        raise ValueError("targeted training record count differs")
    selected_records = (
        select_smoke_records(records)
        if smoke
        else sorted(records, key=lambda item: str(item["source_id"]))
    )
    expected_prompts = 4 if smoke else 180
    if len(selected_records) != expected_prompts:
        raise ValueError("generation prompt count differs")

    modules, launch = qlora._modules()
    torch = modules["torch"]
    started = time.perf_counter()
    model_path = config.resolve_artifact(str(model_contract["snapshot_relative_path"]))
    adapter_path = config.resolve_artifact(str(warm_start["adapter_relative_path"]))
    if directory_sha256(adapter_path) != warm_start["adapter_sha256"]:
        raise ValueError("generation warm-start adapter identity differs")
    model, tokenizer = qlora._load_base(model_path, modules)
    model = modules["peft"].PeftModel.from_pretrained(
        model,
        str(adapter_path),
        local_files_only=True,
        is_trainable=False,
    )
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("generation detected CPU or disk offload")
    model.eval()
    rows: list[dict[str, Any]] = []
    backend_failures = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for prompt_number, record in enumerate(selected_records, start=1):
        source_id = str(record["source_id"])
        messages = assistant_only_v3_messages(
            str(record["question"]),
            str(record["assistant_completion"]),
        )
        input_ids = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda:0")
        attention_mask = torch.ones_like(input_ids)
        prompt_tokens = int(input_ids.shape[-1])
        total_input_tokens += prompt_tokens * int(generation["completions_per_prompt"])
        for completion_index in range(int(generation["completions_per_prompt"])):
            subseed = prompt_subseed(CYCLE_ID, source_id, completion_index)
            torch.manual_seed(subseed)
            torch.cuda.manual_seed_all(subseed)
            backend_error_type: str | None = None
            token_ids: list[int] = []
            completion = ""
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=float(generation["temperature"]),
                        top_p=float(generation["top_p"]),
                        top_k=int(generation["top_k"]),
                        max_new_tokens=int(generation["max_new_tokens"]),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                generated = outputs[0, prompt_tokens:]
                token_ids = [int(value) for value in generated.detach().cpu().tolist()]
                completion = cast(
                    str,
                    tokenizer.decode(token_ids, skip_special_tokens=True),
                )
            except Exception as error:  # fail-closed evidence, never a retry
                backend_error_type = type(error).__name__
                backend_failures += 1
            completion_tokens = len(token_ids)
            total_output_tokens += completion_tokens
            ended_with_eos = bool(token_ids and token_ids[-1] == int(tokenizer.eos_token_id))
            truncated = (
                completion_tokens == int(generation["max_new_tokens"]) and not ended_with_eos
            )
            rows.append(
                {
                    "schema_version": 1,
                    "cycle_id": CYCLE_ID,
                    "source_id": source_id,
                    "family": str(record["family"]),
                    "question": str(record["question"]),
                    "question_sha256": str(record["question_sha256"]),
                    "canonical_answer": str(record["canonical_answer"]),
                    "original_completion_sha256": str(record["assistant_completion_sha256"]),
                    "completion_index": completion_index,
                    "subseed": subseed,
                    "token_ids": token_ids,
                    "token_ids_sha256": canonical_sha256(token_ids),
                    "completion_tokens": completion_tokens,
                    "completion": completion,
                    "raw_completion_sha256": text_sha256(completion),
                    "normalized_completion_sha256": text_sha256(normalized_completion(completion)),
                    "ended_with_eos": ended_with_eos,
                    "truncated": truncated,
                    "backend_error_type": backend_error_type,
                }
            )
        print(
            json.dumps(
                {
                    "cycle_id": CYCLE_ID,
                    "generation_prompt": prompt_number,
                    "generation_prompts_total": expected_prompts,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    memory = modules["psutil"].Process().memory_info()
    peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    output_directory.mkdir(parents=True, exist_ok=False)
    raw_path = output_directory / "candidates.jsonl"
    _write_jsonl(raw_path, rows)
    deterministic_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"question", "canonical_answer", "completion"}
        }
        for row in rows
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generation_id": (
            "foundry-cycle1-compatibility-generation-v1"
            if smoke
            else "foundry-cycle1-production-generation-v1"
        ),
        "cycle_id": CYCLE_ID,
        "smoke": smoke,
        "prompts": expected_prompts,
        "completions_per_prompt": int(generation["completions_per_prompt"]),
        "attempted_completions": len(rows),
        "backend_failures": backend_failures,
        "generated_token_ids_sha256": canonical_sha256([row["token_ids"] for row in rows]),
        "deterministic_rows_sha256": canonical_sha256(deterministic_rows),
        "raw_file_sha256": file_sha256(raw_path),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "peak_process_rss_bytes": peak_rss,
        "adapter_sha256": warm_start["adapter_sha256"],
        "model_revision": model_contract["revision"],
        "cpu_offload": False,
        "vllm": False,
        "launch_evidence": launch,
    }
    summary["generation_sha256"] = canonical_sha256(summary)
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_registry_sanity(
    *,
    config: CycleConfig,
    adapter_path: Path,
    expected_adapter_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Reload one promoted adapter and run a nonbenchmark registry fixture."""

    validate_process_environment(config=config)
    if output_directory.exists():
        raise FileExistsError("registry sanity output must be fresh")
    if directory_sha256(adapter_path) != expected_adapter_sha256:
        raise ValueError("registry sanity adapter identity differs")
    modules, launch = qlora._modules()
    torch = modules["torch"]
    model_path = config.resolve_artifact(str(config.section("model")["snapshot_relative_path"]))
    model, tokenizer = qlora._load_base(model_path, modules)
    model = modules["peft"].PeftModel.from_pretrained(
        model,
        str(adapter_path),
        local_files_only=True,
        is_trainable=False,
    )
    model.eval()
    messages = assistant_only_v3_messages(
        "A box has two blue markers and three red markers. How many markers are in the box?",
        "Calculation: 2 + 3 = 5\nFinal answer: 5",
    )
    input_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda:0")
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0, int(input_ids.shape[-1]) :]
    token_ids = [int(value) for value in generated.detach().cpu().tolist()]
    response = cast(str, tokenizer.decode(token_ids, skip_special_tokens=True))
    if not token_ids or not response.strip():
        raise RuntimeError("promoted registry sanity fixture produced no response")
    lora_module_count = sum(1 for name, _ in model.named_modules() if "lora" in name.casefold())
    if lora_module_count == 0:
        raise RuntimeError("registry sanity did not load LoRA modules")
    result: dict[str, Any] = {
        "schema_version": 1,
        "sanity_id": "registry-nonbenchmark-arithmetic-v1",
        "adapter_sha256": expected_adapter_sha256,
        "adapter_resolved": True,
        "lora_module_count": lora_module_count,
        "response_token_count": len(token_ids),
        "response_token_ids_sha256": canonical_sha256(token_ids),
        "response_sha256": text_sha256(response),
        "backend_failures": 0,
        "cpu_offload": False,
        "launch_evidence": launch,
    }
    result["sanity_sha256"] = canonical_sha256(result)
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
