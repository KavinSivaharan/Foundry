"""Freeze correct untouched-base behavior from the original shared replay anchor."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256
from foundry.training.retention import RetentionItem, score_response

Section = Literal["arithmetic", "format", "instruction"]
Kind = Literal["numeric_terminal", "exact_text", "json_exact"]

ANCHOR_ID = "foundry-shared-retention-anchor-v1"
ANCHOR_SHA256 = "a15df37c7318432576878ff86e567a0f0bac050cc62e2af61081a937f2c1740c"
ANCHOR_PROMPT_SHA256 = "aca5eef33cc48aee74080d6408eff40b06df60108933a70c5739214773394fb9"
ANCHOR_GOLD_RESPONSE_SHA256 = "9634f8cbf5490ef3ca237ce07ff5f142f118dbf891c9b448a9eda214ebde13a6"
ANCHOR_ANSWER_SHA256 = "b748ed3f321ae195f4044c28113296f2f55757577b8e1ce92126a221fd21c9a1"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CHAT_TEMPLATE_ID = "official-pinned-qwen-chat-template"
MASKING_ID = "assistant-content-plus-final-eos-only"
EVALUATION_ID = "foundry-base-replay-anchor-evaluation-v1"
REPLAY_CORPUS_ID = "foundry-base-correct-replay-corpus-v1"
REPLAY_FORMAT_ID = "foundry-shared-base-replay-v1"
SECTION_ORDER: tuple[Section, ...] = ("arithmetic", "format", "instruction")
EXPECTED_SECTION_COUNTS = {section: 40 for section in SECTION_ORDER}
MINIMUM_REPLAY_SECTION_COUNTS = {section: 20 for section in SECTION_ORDER}
MINIMUM_REPLAY_TOTAL = 75

GENERATION_CONFIG: dict[str, bool | int] = {
    "do_sample": False,
    "max_new_tokens": 384,
    "seed": 20260720,
}
GENERATION_CONFIG_SHA256 = "f628fe7faafe94040de3df696a63d98494525cd1a57487b718fd9a86b5292093"


@dataclass(frozen=True)
class ReplayAnchorItem:
    """One objectively scored prompt in the original shared replay anchor."""

    item_id: str
    section: Section
    skill: str
    kind: Kind
    prompt: str
    expected: str
    gold_response: str

    def as_retention_item(self) -> RetentionItem:
        """Adapt this item to the already-frozen objective scorer."""

        return RetentionItem(
            item_id=self.item_id,
            section=self.section,
            skill=self.skill,
            kind=self.kind,
            prompt=self.prompt,
            expected=self.expected,
        )


@dataclass(frozen=True)
class ReplayAnchor:
    """Strictly identified original anchor and its ordered prompts."""

    anchor_id: str
    system_prompt: str
    chat_template: str
    masking: str
    items: tuple[ReplayAnchorItem, ...]
    anchor_sha256: str
    prompt_sha256: str
    gold_response_sha256: str
    answer_sha256: str


@dataclass(frozen=True)
class BaseGenerationResult:
    """One untouched-base generation or one typed backend failure."""

    item_id: str
    response: str
    backend_error_type: str | None = None


@dataclass(frozen=True)
class _AnchorIdentity:
    anchor_sha256: str
    prompt_sha256: str
    gold_response_sha256: str
    answer_sha256: str


_FROZEN_IDENTITY = _AnchorIdentity(
    anchor_sha256=ANCHOR_SHA256,
    prompt_sha256=ANCHOR_PROMPT_SHA256,
    gold_response_sha256=ANCHOR_GOLD_RESPONSE_SHA256,
    answer_sha256=ANCHOR_ANSWER_SHA256,
)


def replay_format_sha256() -> str:
    """Hash the frozen assistant-only replay formatting contract."""

    return canonical_sha256(
        {
            "format_id": REPLAY_FORMAT_ID,
            "chat_template": CHAT_TEMPLATE_ID,
            "messages": ["system", "user", "assistant"],
            "target": "actual_deterministic_untouched_base_assistant_output",
            "loss_bearing": ["assistant_response_content", "final_eos"],
            "masked": [
                "system",
                "user",
                "assistant_header",
                "padding",
                "post_eos",
            ],
            "final_answer_line_added": False,
        }
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return cast(dict[str, Any], value)


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain an array of objects")
    return cast(list[dict[str, Any]], value)


def _parse_anchor(root: dict[str, Any], identity: _AnchorIdentity) -> ReplayAnchor:
    expected_root_keys = {
        "anchor_id",
        "chat_template",
        "items",
        "masking",
        "schema_version",
        "system_prompt",
    }
    if set(root) != expected_root_keys:
        raise ValueError("replay anchor fields differ from the frozen schema")
    if (
        root.get("schema_version") != 1
        or root.get("anchor_id") != ANCHOR_ID
        or root.get("chat_template") != CHAT_TEMPLATE_ID
        or root.get("masking") != MASKING_ID
    ):
        raise ValueError("replay anchor identity or formatting contract differs")
    system_prompt = root.get("system_prompt")
    raw_items = root.get("items")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("replay anchor requires a system prompt")
    if not isinstance(raw_items, list):
        raise ValueError("replay anchor items must be an array")

    items: list[ReplayAnchorItem] = []
    expected_item_keys = {
        "expected",
        "gold_response",
        "id",
        "kind",
        "prompt",
        "section",
        "skill",
    }
    for raw in raw_items:
        if not isinstance(raw, dict) or set(raw) != expected_item_keys:
            raise ValueError("replay anchor item fields differ from the frozen schema")
        section = raw.get("section")
        kind = raw.get("kind")
        if section not in SECTION_ORDER:
            raise ValueError("replay anchor contains an unknown section")
        if kind not in {"numeric_terminal", "exact_text", "json_exact"}:
            raise ValueError("replay anchor contains an unknown scorer kind")
        item = ReplayAnchorItem(
            item_id=str(raw["id"]),
            section=cast(Section, section),
            skill=str(raw["skill"]),
            kind=cast(Kind, kind),
            prompt=str(raw["prompt"]),
            expected=str(raw["expected"]),
            gold_response=str(raw["gold_response"]),
        )
        if not all(
            (
                item.item_id,
                item.skill,
                item.prompt,
                item.expected,
                item.gold_response,
            )
        ):
            raise ValueError("replay anchor item fields must be non-empty")
        if score_response(item.as_retention_item(), item.gold_response)["correct"] is not True:
            raise ValueError("replay anchor gold response does not pass its frozen scorer")
        items.append(item)

    counts = Counter(item.section for item in items)
    if len(items) != 120 or counts != EXPECTED_SECTION_COUNTS:
        raise ValueError("replay anchor requires exactly 40 items per section")
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("replay anchor IDs must be unique")

    hashes = _AnchorIdentity(
        anchor_sha256=canonical_sha256(root),
        prompt_sha256=canonical_sha256(
            [{"id": item.item_id, "prompt": item.prompt} for item in items]
        ),
        gold_response_sha256=canonical_sha256(
            [{"id": item.item_id, "gold_response": item.gold_response} for item in items]
        ),
        answer_sha256=canonical_sha256(
            [{"id": item.item_id, "expected": item.expected} for item in items]
        ),
    )
    if hashes != identity:
        raise ValueError("replay anchor, prompt, gold-response, or answer hash differs")
    return ReplayAnchor(
        anchor_id=ANCHOR_ID,
        system_prompt=system_prompt,
        chat_template=CHAT_TEMPLATE_ID,
        masking=MASKING_ID,
        items=tuple(items),
        anchor_sha256=hashes.anchor_sha256,
        prompt_sha256=hashes.prompt_sha256,
        gold_response_sha256=hashes.gold_response_sha256,
        answer_sha256=hashes.answer_sha256,
    )


def load_frozen_replay_anchor(path: Path) -> ReplayAnchor:
    """Load the exact frozen 120-item nonbenchmark replay source."""

    if canonical_sha256(GENERATION_CONFIG) != GENERATION_CONFIG_SHA256:
        raise RuntimeError("base-replay generation contract hash differs")
    return _parse_anchor(_load_json_object(path), _FROZEN_IDENTITY)


def _failure_score() -> dict[str, bool | str | None]:
    return {
        "correct": False,
        "extractable": False,
        "malformed": True,
        "prompt_echo": False,
        "question_generation": False,
        "exact_format": False,
        "extracted_hash": None,
    }


def _serialize_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_base_anchor_evaluation(
    *,
    anchor: ReplayAnchor,
    generations: Sequence[BaseGenerationResult],
    raw_path: Path,
    summary_path: Path,
    runtime_metrics: dict[str, int | float | str] | None = None,
) -> dict[str, Any]:
    """Score and persist one complete untouched-base anchor evaluation."""

    if len(generations) != len(anchor.items):
        raise ValueError("base evaluation must cover all 120 replay-anchor items")
    if [generation.item_id for generation in generations] != [
        item.item_id for item in anchor.items
    ]:
        raise ValueError("base evaluation order or IDs differ from the replay anchor")

    rows: list[dict[str, Any]] = []
    for item, generation in zip(anchor.items, generations, strict=True):
        if generation.backend_error_type is not None:
            if generation.response:
                raise ValueError("a backend failure cannot carry an assistant response")
            score: dict[str, Any] = _failure_score()
            backend_status = "failed"
        else:
            score = score_response(item.as_retention_item(), generation.response)
            backend_status = "ok"
        rows.append(
            {
                "id": item.item_id,
                "section": item.section,
                "skill": item.skill,
                "response": generation.response,
                "response_sha256": hashlib.sha256(generation.response.encode("utf-8")).hexdigest(),
                "score": score,
                "backend_status": backend_status,
                "backend_error_type": generation.backend_error_type,
            }
        )

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(_serialize_json(rows), encoding="utf-8")
    section_metrics: dict[str, dict[str, int | float]] = {}
    for section in SECTION_ORDER:
        section_rows = [row for row in rows if row["section"] == section]
        correct = sum(row["score"]["correct"] is True for row in section_rows)
        section_metrics[section] = {
            "total": len(section_rows),
            "correct": correct,
            "accuracy": correct / len(section_rows),
        }
    total = len(rows)
    evidence = [
        {
            "id": row["id"],
            "response_sha256": row["response_sha256"],
            "score": row["score"],
            "backend_status": row["backend_status"],
            "backend_error_type": row["backend_error_type"],
        }
        for row in rows
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "anchor_id": anchor.anchor_id,
        "anchor_sha256": anchor.anchor_sha256,
        "prompt_sha256": anchor.prompt_sha256,
        "gold_response_sha256": anchor.gold_response_sha256,
        "answer_sha256": anchor.answer_sha256,
        "generation_config": GENERATION_CONFIG,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "adapter_loaded": False,
        "adapter_sha256": None,
        "total": total,
        "section_metrics": section_metrics,
        "correct": sum(row["score"]["correct"] is True for row in rows),
        "extractable": sum(row["score"]["extractable"] is True for row in rows),
        "prompt_echo": sum(row["score"]["prompt_echo"] is True for row in rows),
        "question_generation": sum(row["score"]["question_generation"] is True for row in rows),
        "malformed_outputs": sum(row["score"]["malformed"] is True for row in rows),
        "backend_failures": sum(row["backend_status"] == "failed" for row in rows),
        "confirmed_prompt_or_scorer_defects": 0,
        "raw_packet_sha256": file_sha256(raw_path),
        "base_result_sha256": canonical_sha256(
            {
                "anchor_sha256": anchor.anchor_sha256,
                "generation_config_sha256": GENERATION_CONFIG_SHA256,
                "rows": evidence,
            }
        ),
        "per_item_evidence_sha256": canonical_sha256(evidence),
        "runtime_metrics": {} if runtime_metrics is None else runtime_metrics,
        "sealed_final_accessed": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_serialize_json(summary), encoding="utf-8")
    return summary


def evaluate_untouched_base_anchor(
    *,
    anchor_path: Path,
    model_path: Path,
    raw_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Run the frozen anchor on the untouched base; this API cannot accept an adapter."""

    anchor = load_frozen_replay_anchor(anchor_path)
    if (model_path / "adapter_config.json").exists():
        raise ValueError("base replay evaluation refuses an adapter directory")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    seed = int(GENERATION_CONFIG["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
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
        raise RuntimeError("untouched base model was offloaded")
    model.eval()
    load_seconds = time.perf_counter() - load_started

    started = time.perf_counter()
    input_tokens = 0
    output_tokens = 0
    generations: list[BaseGenerationResult] = []
    for item in anchor.items:
        try:
            input_ids = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": anchor.system_prompt},
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
                    do_sample=False,
                    max_new_tokens=384,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = generated[0, input_ids.shape[-1] :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            input_tokens += int(input_ids.shape[-1])
            output_tokens += int(generated_ids.shape[-1])
            generations.append(BaseGenerationResult(item.item_id, response))
        except Exception as error:  # pragma: no cover - hardware/backend guard
            generations.append(BaseGenerationResult(item.item_id, "", type(error).__name__))
    runtime_seconds = time.perf_counter() - started
    metrics: dict[str, int | float | str] = {
        "load_seconds": load_seconds,
        "runtime_seconds": runtime_seconds,
        "examples_per_second": len(anchor.items) / runtime_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    summary = write_base_anchor_evaluation(
        anchor=anchor,
        generations=generations,
        raw_path=raw_path,
        summary_path=summary_path,
        runtime_metrics=metrics,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _validated_base_result(
    *, anchor: ReplayAnchor, summary_path: Path, raw_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_json_object(summary_path)
    expected_summary_hash = summary.get("summary_sha256")
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if not isinstance(expected_summary_hash, str) or expected_summary_hash != canonical_sha256(
        payload
    ):
        raise ValueError("base replay evaluation summary hash differs")
    required = {
        "evaluation_id": EVALUATION_ID,
        "anchor_sha256": anchor.anchor_sha256,
        "prompt_sha256": anchor.prompt_sha256,
        "gold_response_sha256": anchor.gold_response_sha256,
        "answer_sha256": anchor.answer_sha256,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "adapter_loaded": False,
        "adapter_sha256": None,
        "total": 120,
    }
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError("base replay evaluation identity, model, or adapter state differs")
    if summary.get("generation_config") != GENERATION_CONFIG:
        raise ValueError("base replay evaluation generation config differs")
    if summary.get("raw_packet_sha256") != file_sha256(raw_path):
        raise ValueError("base replay raw packet hash differs")

    rows = _load_json_rows(raw_path)
    if len(rows) != len(anchor.items) or [row.get("id") for row in rows] != [
        item.item_id for item in anchor.items
    ]:
        raise ValueError("base replay raw result order or IDs differ")
    evidence: list[dict[str, Any]] = []
    for item, row in zip(anchor.items, rows, strict=True):
        if row.get("section") != item.section or row.get("skill") != item.skill:
            raise ValueError("base replay raw category differs from the anchor")
        response = row.get("response")
        if (
            not isinstance(response, str)
            or row.get("response_sha256") != hashlib.sha256(response.encode("utf-8")).hexdigest()
        ):
            raise ValueError("base replay response hash differs")
        status = row.get("backend_status")
        error_type = row.get("backend_error_type")
        if status == "ok" and error_type is None:
            expected_score: dict[str, Any] = score_response(item.as_retention_item(), response)
        elif status == "failed" and isinstance(error_type, str) and not response:
            expected_score = _failure_score()
        else:
            raise ValueError("base replay backend status is inconsistent")
        if row.get("score") != expected_score:
            raise ValueError("base replay score differs from the frozen scorer")
        evidence.append(
            {
                "id": item.item_id,
                "response_sha256": row["response_sha256"],
                "score": expected_score,
                "backend_status": status,
                "backend_error_type": error_type,
            }
        )
    base_result_sha256 = canonical_sha256(
        {
            "anchor_sha256": anchor.anchor_sha256,
            "generation_config_sha256": GENERATION_CONFIG_SHA256,
            "rows": evidence,
        }
    )
    if summary.get("base_result_sha256") != base_result_sha256 or summary.get(
        "per_item_evidence_sha256"
    ) != canonical_sha256(evidence):
        raise ValueError("base replay per-item result hash differs")

    section_metrics: dict[str, dict[str, int | float]] = {}
    for section in SECTION_ORDER:
        selected = [row for row in rows if row["section"] == section]
        correct = sum(cast(dict[str, Any], row["score"])["correct"] is True for row in selected)
        section_metrics[section] = {
            "total": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected),
        }
    aggregate = {
        "section_metrics": section_metrics,
        "correct": sum(cast(dict[str, Any], row["score"])["correct"] is True for row in rows),
        "extractable": sum(
            cast(dict[str, Any], row["score"])["extractable"] is True for row in rows
        ),
        "prompt_echo": sum(
            cast(dict[str, Any], row["score"])["prompt_echo"] is True for row in rows
        ),
        "question_generation": sum(
            cast(dict[str, Any], row["score"])["question_generation"] is True for row in rows
        ),
        "malformed_outputs": sum(
            cast(dict[str, Any], row["score"])["malformed"] is True for row in rows
        ),
        "backend_failures": sum(row["backend_status"] == "failed" for row in rows),
    }
    if any(summary.get(key) != value for key, value in aggregate.items()):
        raise ValueError("base replay aggregate metrics differ from the raw packet")
    return summary, rows


def freeze_base_correct_replay_corpus(
    *,
    anchor_path: Path,
    base_summary_path: Path,
    base_raw_path: Path,
    replay_raw_path: Path,
    manifest_path: Path,
    objectively_defective_item_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Freeze scorer-correct base outputs as replay targets after the strict gate passes."""

    anchor = load_frozen_replay_anchor(anchor_path)
    summary, rows = _validated_base_result(
        anchor=anchor, summary_path=base_summary_path, raw_path=base_raw_path
    )
    defective = tuple(objectively_defective_item_ids)
    if len(set(defective)) != len(defective):
        raise ValueError("objective-defect IDs must be unique")
    if any(item_id not in {item.item_id for item in anchor.items} for item_id in defective):
        raise ValueError("objective-defect audit contains an unknown anchor ID")

    selected_pairs = [
        (item, row)
        for item, row in zip(anchor.items, rows, strict=True)
        if cast(dict[str, Any], row["score"])["correct"] is True
    ]
    section_counts = Counter(item.section for item, _ in selected_pairs)
    gate_checks = {
        "arithmetic_at_least_20": section_counts["arithmetic"] >= 20,
        "format_at_least_20": section_counts["format"] >= 20,
        "instruction_at_least_20": section_counts["instruction"] >= 20,
        "overall_at_least_75": len(selected_pairs) >= MINIMUM_REPLAY_TOTAL,
        "zero_backend_failures": summary.get("backend_failures") == 0,
        "zero_objective_prompt_or_scorer_defects": not defective
        and summary.get("confirmed_prompt_or_scorer_defects") == 0,
    }
    if not all(gate_checks.values()):
        failed = ", ".join(key for key, passed in gate_checks.items() if not passed)
        raise ValueError(f"base replay corpus gate failed: {failed}")

    raw_items = [
        {
            "id": item.item_id,
            "section": item.section,
            "skill": item.skill,
            "system_prompt": anchor.system_prompt,
            "prompt": item.prompt,
            "assistant_response": row["response"],
            "assistant_response_sha256": row["response_sha256"],
        }
        for item, row in selected_pairs
    ]
    raw_replay: dict[str, Any] = {
        "schema_version": 1,
        "replay_corpus_id": REPLAY_CORPUS_ID,
        "replay_format_id": REPLAY_FORMAT_ID,
        "source_anchor_sha256": anchor.anchor_sha256,
        "base_result_sha256": summary["base_result_sha256"],
        "items": raw_items,
    }
    replay_corpus_sha256 = canonical_sha256(raw_replay)
    replay_raw_path.parent.mkdir(parents=True, exist_ok=True)
    replay_raw_path.write_text(_serialize_json(raw_replay), encoding="utf-8")

    manifest_items = [
        {
            "id": item.item_id,
            "section": item.section,
            "skill": item.skill,
            "base_output_sha256": row["response_sha256"],
        }
        for item, row in selected_pairs
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "replay_corpus_id": REPLAY_CORPUS_ID,
        "replay_format_id": REPLAY_FORMAT_ID,
        "replay_format_sha256": replay_format_sha256(),
        "anchor_id": anchor.anchor_id,
        "anchor_sha256": anchor.anchor_sha256,
        "prompt_sha256": anchor.prompt_sha256,
        "gold_response_sha256": anchor.gold_response_sha256,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "base_result_sha256": summary["base_result_sha256"],
        "base_summary_sha256": summary["summary_sha256"],
        "replay_corpus_sha256": replay_corpus_sha256,
        "raw_replay_packet_sha256": file_sha256(replay_raw_path),
        "section_counts": {section: section_counts[section] for section in SECTION_ORDER},
        "total": len(manifest_items),
        "items": manifest_items,
        "gate_checks": gate_checks,
        "gate_passed": True,
        "selection": "frozen_scorer_correct_on_untouched_base",
        "target_source": "actual_deterministic_untouched_base_output",
        "predefined_gold_used_as_replay_target": False,
        "prompts_or_outputs_in_manifest": False,
        "adapter_loaded": False,
        "sealed_final_accessed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_serialize_json(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate-base")
    evaluate.add_argument("--anchor", required=True, type=Path)
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--raw-output", required=True, type=Path)
    evaluate.add_argument("--summary-output", required=True, type=Path)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--anchor", required=True, type=Path)
    freeze.add_argument("--base-summary", required=True, type=Path)
    freeze.add_argument("--base-raw", required=True, type=Path)
    freeze.add_argument("--replay-raw-output", required=True, type=Path)
    freeze.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "evaluate-base":
        result = evaluate_untouched_base_anchor(
            anchor_path=args.anchor,
            model_path=args.model,
            raw_path=args.raw_output,
            summary_path=args.summary_output,
        )
    else:
        result = freeze_base_correct_replay_corpus(
            anchor_path=args.anchor,
            base_summary_path=args.base_summary,
            base_raw_path=args.base_raw,
            replay_raw_path=args.replay_raw_output,
            manifest_path=args.manifest_output,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
