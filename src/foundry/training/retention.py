"""Frozen original retention suite and deterministic local evaluator."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import re
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

from foundry.evaluation.answer_extraction import (
    CanonicalExtractionError,
    extract_canonical_number,
)
from foundry.training.config import canonical_sha256
from foundry.training.lora_scaling import ScalingEvidence, scaled_lora_adapter
from foundry.training.qlora import directory_sha256, file_sha256

Section = Literal["arithmetic", "format", "instruction"]
Kind = Literal["numeric_terminal", "exact_text", "json_exact"]
SUITE_LAYOUTS = {
    "foundry-original-retention-suite-v1": {"arithmetic": 30, "format": 15, "instruction": 15},
    "foundry-retention-validation-v1": {"arithmetic": 45, "format": 20, "instruction": 25},
    "foundry-retention-final-holdout-v1": {"arithmetic": 45, "format": 20, "instruction": 25},
    "foundry-retention-adjudication-v2": {
        "arithmetic": 100,
        "format": 100,
        "instruction": 100,
    },
    "foundry-retention-anchor-holdout-v1": {
        "arithmetic": 100,
        "format": 100,
        "instruction": 100,
    },
    "foundry-retention-scale-final-holdout-v1": {
        "arithmetic": 150,
        "format": 150,
        "instruction": 150,
    },
    "foundry-retention-replay-final-holdout-v1": {
        "arithmetic": 150,
        "format": 150,
        "instruction": 150,
    },
}
EVALUATION_IDS = {
    "foundry-original-retention-suite-v1": "foundry-original-retention-evaluation-v1",
    "foundry-retention-validation-v1": "foundry-retention-validation-evaluation-v1",
    "foundry-retention-final-holdout-v1": "foundry-retention-final-holdout-evaluation-v1",
    "foundry-retention-adjudication-v2": "foundry-retention-adjudication-evaluation-v2",
    "foundry-retention-anchor-holdout-v1": ("foundry-retention-anchor-holdout-evaluation-v1"),
    "foundry-retention-scale-final-holdout-v1": (
        "foundry-retention-scale-final-holdout-evaluation-v1"
    ),
    "foundry-retention-replay-final-holdout-v1": (
        "foundry-retention-replay-final-holdout-evaluation-v1"
    ),
}


@dataclass(frozen=True)
class RetentionItem:
    """One original prompt with an objective expected result."""

    item_id: str
    section: Section
    skill: str
    kind: Kind
    prompt: str
    expected: str


@dataclass(frozen=True)
class RetentionSuite:
    """Complete frozen retention suite and generation contract."""

    suite_id: str
    system_prompt: str
    do_sample: bool
    max_new_tokens: int
    seed: int
    items: tuple[RetentionItem, ...]
    suite_sha256: str
    prompt_sha256: str
    generation_sha256: str


@dataclass(frozen=True)
class BaseConditionedSubset:
    """Content-free ordered IDs selected from untouched-base successes."""

    instrument_id: str
    subset_id: str
    suite_sha256: str
    base_summary_sha256: str
    items: tuple[tuple[str, Section, str], ...]
    subset_sha256: str


def load_suite(path: Path) -> RetentionSuite:
    """Load and strictly validate one recognized frozen retention suite."""

    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("retention suite must be an object")
    root = cast(dict[str, Any], value)
    suite_id = str(root.get("suite_id"))
    if root.get("schema_version") != 1 or suite_id not in SUITE_LAYOUTS:
        raise ValueError("retention suite identity differs")
    generation = root.get("generation")
    raw_items = root.get("items")
    if not isinstance(generation, dict) or not isinstance(raw_items, list):
        raise ValueError("retention generation and items are required")
    items: list[RetentionItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("retention item must be an object")
        item = RetentionItem(
            item_id=str(raw["id"]),
            section=cast(Section, raw["section"]),
            skill=str(raw["skill"]),
            kind=cast(Kind, raw["kind"]),
            prompt=str(raw["prompt"]),
            expected=str(raw["expected"]),
        )
        if item.section not in {"arithmetic", "format", "instruction"}:
            raise ValueError("unknown retention section")
        if item.kind not in {"numeric_terminal", "exact_text", "json_exact"}:
            raise ValueError("unknown retention scoring kind")
        if not all((item.item_id, item.skill, item.prompt, item.expected)):
            raise ValueError("retention item fields must be non-empty")
        items.append(item)
    counts = Counter(item.section for item in items)
    expected_counts = SUITE_LAYOUTS[suite_id]
    if counts != expected_counts:
        raise ValueError(f"retention suite requires {expected_counts}")
    if len({item.item_id for item in items}) != sum(expected_counts.values()):
        raise ValueError("retention IDs must be unique")
    payload = {
        "suite_id": root["suite_id"],
        "system_prompt": root["system_prompt"],
        "generation": generation,
        "items": raw_items,
    }
    return RetentionSuite(
        suite_id=str(root["suite_id"]),
        system_prompt=str(root["system_prompt"]),
        do_sample=bool(generation["do_sample"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
        items=tuple(items),
        suite_sha256=canonical_sha256(payload),
        prompt_sha256=canonical_sha256(
            {"system": root["system_prompt"], "prompts": [item.prompt for item in items]}
        ),
        generation_sha256=canonical_sha256(generation),
    )


def load_base_conditioned_subset(path: Path, suite: RetentionSuite) -> BaseConditionedSubset:
    """Load one frozen base-correct ID manifest without prompt or answer content."""

    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("base-conditioned subset must be an object")
    root = cast(dict[str, Any], value)
    expected_hash = root.get("subset_sha256")
    payload = {key: item for key, item in root.items() if key != "subset_sha256"}
    if (
        root.get("schema_version") != 1
        or root.get("instrument_id") != "foundry-base-conditioned-retention-v1"
        or not isinstance(expected_hash, str)
        or expected_hash != canonical_sha256(payload)
    ):
        raise ValueError("base-conditioned subset identity or hash differs")
    if root.get("suite_sha256") != suite.suite_sha256:
        raise ValueError("base-conditioned subset suite differs")
    raw_items = root.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("base-conditioned subset requires ordered items")
    suite_index = {item.item_id: item for item in suite.items}
    items: list[tuple[str, Section, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("base-conditioned subset item must be an object")
        item_id = str(raw.get("id"))
        suite_item = suite_index.get(item_id)
        if suite_item is None:
            raise ValueError("base-conditioned subset contains an unknown ID")
        if raw.get("section") != suite_item.section or raw.get("skill") != suite_item.skill:
            raise ValueError("base-conditioned category label differs from suite")
        items.append((item_id, suite_item.section, suite_item.skill))
    if len({item[0] for item in items}) != len(items):
        raise ValueError("base-conditioned subset IDs must be unique")
    if root.get("total") != len(items) or not str(root.get("subset_id", "")):
        raise ValueError("base-conditioned subset total or ID differs")
    counts = Counter(item[1] for item in items)
    if root.get("section_counts") != {
        section: counts[cast(Section, section)]
        for section in ("arithmetic", "format", "instruction")
    }:
        raise ValueError("base-conditioned subset counts differ")
    base_summary_sha256 = root.get("base_summary_sha256")
    if not isinstance(base_summary_sha256, str) or len(base_summary_sha256) != 64:
        raise ValueError("base-conditioned base summary hash is required")
    return BaseConditionedSubset(
        instrument_id=str(root["instrument_id"]),
        subset_id=str(root["subset_id"]),
        suite_sha256=suite.suite_sha256,
        base_summary_sha256=base_summary_sha256,
        items=tuple(items),
        subset_sha256=expected_hash,
    )


def score_response(item: RetentionItem, response: str) -> dict[str, Any]:
    """Score one response with no subjective judgment."""

    stripped = response.strip()
    correct = False
    extractable = False
    malformed = False
    extracted: str | None = None
    if item.kind == "numeric_terminal":
        try:
            value = extract_canonical_number(response)
            extracted = str(value)
            extractable = True
            correct = value == Fraction(item.expected)
        except (CanonicalExtractionError, ValueError, ZeroDivisionError):
            malformed = True
    elif item.kind == "json_exact":
        try:
            actual_json = json.loads(stripped)
            expected_json = json.loads(item.expected)
            extractable = True
            correct = actual_json == expected_json and isinstance(actual_json, dict)
            extracted = canonical_sha256(actual_json)
        except (json.JSONDecodeError, TypeError):
            malformed = True
    else:
        extractable = bool(stripped)
        correct = stripped == item.expected
        extracted = stripped if len(stripped) <= 160 else None
        malformed = not extractable
    normalized_prompt = " ".join(item.prompt.lower().split())
    normalized_response = " ".join(response.lower().split())
    prompt_echo = len(normalized_prompt) >= 24 and normalized_prompt in normalized_response
    question_generation = "?" in response or bool(
        re.search(r"(?:^|\n)\s*(?:question|problem)\s*:", response, re.IGNORECASE)
    )
    exact_format = (
        correct
        if item.kind != "numeric_terminal"
        else bool(
            re.search(
                rf"(?:^|\n)Final answer:\s*{re.escape(item.expected)}\s*$",
                response,
                re.IGNORECASE,
            )
        )
    )
    return {
        "correct": correct,
        "extractable": extractable,
        "malformed": malformed,
        "prompt_echo": prompt_echo,
        "question_generation": question_generation,
        "exact_format": exact_format,
        "extracted_hash": None
        if extracted is None
        else hashlib.sha256(extracted.encode("utf-8")).hexdigest(),
    }


def _load_model(
    *, model_path: Path, adapter_path: Path | None, torch: Any, transformers: Any
) -> tuple[Any, Any, float, str | None]:
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
    adapter_sha256: str | None = None
    if adapter_path is not None:
        peft = importlib.import_module("peft")
        adapter_sha256 = directory_sha256(adapter_path)
        model = peft.PeftModel.from_pretrained(
            model,
            str(adapter_path),
            local_files_only=True,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("retention model or adapter was offloaded")
    model.eval()
    return model, tokenizer, time.perf_counter() - started, adapter_sha256


def evaluate_suite(
    *,
    suite_path: Path,
    model_path: Path,
    adapter_path: Path | None,
    raw_path: Path,
    output_path: Path,
    subset_manifest_path: Path | None = None,
    adapter_scale: float | None = None,
) -> dict[str, Any]:
    """Evaluate one base or adapter on a frozen suite or base-correct subset."""

    suite = load_suite(suite_path)
    subset = (
        None
        if subset_manifest_path is None
        else load_base_conditioned_subset(subset_manifest_path, suite)
    )
    if subset is None:
        evaluated_items = suite.items
    else:
        suite_index = {item.item_id: item for item in suite.items}
        evaluated_items = tuple(suite_index[item_id] for item_id, _, _ in subset.items)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    random.seed(suite.seed)
    torch.manual_seed(suite.seed)
    torch.cuda.manual_seed_all(suite.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, load_seconds, adapter_sha256 = _load_model(
        model_path=model_path,
        adapter_path=adapter_path,
        torch=torch,
        transformers=transformers,
    )
    if adapter_scale is not None and adapter_sha256 is None:
        raise ValueError("runtime LoRA scaling requires an adapter")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    backend_failures = 0
    input_tokens = 0
    output_tokens = 0
    scale_context = (
        scaled_lora_adapter(model, adapter_scale)
        if adapter_scale is not None
        else nullcontext(None)
    )
    scale_evidence: ScalingEvidence | None
    with scale_context as scale_evidence:
        for item in evaluated_items:
            response = ""
            try:
                input_ids = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": suite.system_prompt},
                        {"role": "user", "content": item.prompt},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda:0")
                attention_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=suite.do_sample,
                        max_new_tokens=suite.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated_ids = generated[0, input_ids.shape[-1] :]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                input_tokens += int(input_ids.shape[-1])
                output_tokens += int(generated_ids.shape[-1])
                score = score_response(item, response)
            except Exception as error:  # pragma: no cover - hardware/backend guard
                backend_failures += 1
                score = {
                    "correct": False,
                    "extractable": False,
                    "malformed": True,
                    "prompt_echo": False,
                    "question_generation": False,
                    "exact_format": False,
                    "extracted_hash": None,
                    "backend_error_type": type(error).__name__,
                }
            rows.append(
                {
                    "id": item.item_id,
                    "section": item.section,
                    "skill": item.skill,
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                    "score": score,
                }
            )
    runtime = time.perf_counter() - started
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    section_metrics: dict[str, dict[str, int | float]] = {}
    for section in ("arithmetic", "format", "instruction"):
        selected = [row for row in rows if row["section"] == section]
        correct = sum(bool(row["score"]["correct"]) for row in selected)
        section_metrics[section] = {
            "total": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected),
        }
    total = len(rows)
    extractable = sum(bool(row["score"]["extractable"]) for row in rows)
    prompt_echo = sum(bool(row["score"]["prompt_echo"]) for row in rows)
    malformed = sum(bool(row["score"]["malformed"]) for row in rows)
    question_generation = sum(bool(row["score"]["question_generation"]) for row in rows)
    exact_format = sum(bool(row["score"]["exact_format"]) for row in rows)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": EVALUATION_IDS[suite.suite_id],
        "suite_sha256": suite.suite_sha256,
        "prompt_sha256": suite.prompt_sha256,
        "generation_sha256": suite.generation_sha256,
        "suite_file_sha256": file_sha256(suite_path),
        "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "adapter_sha256": adapter_sha256,
        "adapter_scale": adapter_scale,
        "adapter_scale_evidence": None if scale_evidence is None else scale_evidence.as_dict(),
        "section_metrics": section_metrics,
        "total": total,
        "extractable": extractable,
        "extractability": extractable / total,
        "exact_format": exact_format,
        "exact_format_rate": exact_format / total,
        "prompt_echo": prompt_echo,
        "prompt_echo_rate": prompt_echo / total,
        "malformed_outputs": malformed,
        "question_generation": question_generation,
        "backend_failures": backend_failures,
        "runtime_seconds": runtime,
        "examples_per_second": total / runtime,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "load_seconds": load_seconds,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "raw_packet_sha256": file_sha256(raw_path),
        "base_conditioned_instrument_id": None if subset is None else subset.instrument_id,
        "base_conditioned_subset_id": None if subset is None else subset.subset_id,
        "base_conditioned_subset_sha256": None if subset is None else subset.subset_sha256,
        "base_conditioned_subset_file_sha256": None
        if subset_manifest_path is None
        else file_sha256(subset_manifest_path),
        "per_item_decision_sha256": canonical_sha256(
            [
                {
                    "id": row["id"],
                    "response_sha256": row["response_sha256"],
                    "score": row["score"],
                }
                for row in rows
            ]
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    """Run the frozen retention evaluator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path)
    parser.add_argument("--adapter-scale", type=float)
    args = parser.parse_args()
    result = evaluate_suite(
        suite_path=args.suite,
        model_path=args.model_path,
        adapter_path=args.adapter,
        raw_path=args.raw_path,
        output_path=args.output_path,
        subset_manifest_path=args.subset_manifest,
        adapter_scale=args.adapter_scale,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
