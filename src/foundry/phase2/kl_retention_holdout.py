"""Freeze and evaluate the Milestone 13C independent KL-retention instrument."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.base_conditioned_retention import wilson_lower_bound
from foundry.training.config import canonical_sha256
from foundry.training.qlora import directory_sha256, file_sha256
from foundry.training.retention import Kind, RetentionItem, RetentionSuite, Section, score_response

INSTRUMENT_ID = "foundry-kl-independent-retention-v1"
SUITE_ID = INSTRUMENT_ID
EVALUATION_ID = "foundry-kl-independent-retention-evaluation-v1"
SUBSET_ID = "foundry-kl-independent-retention-base-correct-v1"
SEED = 20260720
SECTION_ORDER = ("arithmetic", "format", "instruction")
EXPECTED_COUNTS = {"arithmetic": 120, "format": 120, "instruction": 120}
BASE_CORRECT_MINIMUMS = {"arithmetic": 60, "format": 60, "instruction": 50}
BASE_CORRECT_TOTAL_MINIMUM = 170
SEALED_BOUNDARY_STATUS = "metadata_accessed_example_content_unseen"


def _read_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain row objects")
    return cast(list[dict[str, Any]], value)


def _validate_frozen(value: dict[str, Any], hash_key: str) -> None:
    expected = value.get(hash_key)
    payload = {key: item for key, item in value.items() if key != hash_key}
    if not isinstance(expected, str) or expected != canonical_sha256(payload):
        raise ValueError(f"{hash_key} does not reconstruct")


def load_suite(path: Path) -> RetentionSuite:
    """Load the exact 360-item KL-independent candidate suite."""

    root = _read_object(path)
    generation = root.get("generation")
    raw_items = root.get("items")
    if (
        root.get("schema_version") != 1
        or root.get("suite_id") != SUITE_ID
        or not isinstance(generation, dict)
        or not isinstance(raw_items, list)
        or generation
        != {
            "do_sample": False,
            "max_new_tokens": 96,
            "seed": SEED,
        }
    ):
        raise ValueError("KL-independent suite identity or generation contract differs")
    items: list[RetentionItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("KL-independent suite item must be an object")
        item = RetentionItem(
            item_id=str(raw.get("id")),
            section=cast(Section, raw.get("section")),
            skill=str(raw.get("skill")),
            kind=cast(Kind, raw.get("kind")),
            prompt=str(raw.get("prompt")),
            expected=str(raw.get("expected")),
        )
        if (
            item.section not in EXPECTED_COUNTS
            or item.kind not in {"numeric_terminal", "exact_text", "json_exact"}
            or not all((item.item_id, item.skill, item.prompt, item.expected))
        ):
            raise ValueError("KL-independent suite item fields differ")
        items.append(item)
    counts = Counter(item.section for item in items)
    if counts != EXPECTED_COUNTS or len(items) != 360:
        raise ValueError("KL-independent suite requires exactly 120 items per section")
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("KL-independent suite IDs must be unique")
    if len({" ".join(item.prompt.casefold().split()) for item in items}) != len(items):
        raise ValueError("KL-independent suite prompts must be normalized-unique")
    payload = {
        "suite_id": root["suite_id"],
        "system_prompt": root["system_prompt"],
        "generation": generation,
        "items": raw_items,
    }
    return RetentionSuite(
        suite_id=SUITE_ID,
        system_prompt=str(root["system_prompt"]),
        do_sample=False,
        max_new_tokens=96,
        seed=SEED,
        items=tuple(items),
        suite_sha256=canonical_sha256(payload),
        prompt_sha256=canonical_sha256(
            {
                "system": root["system_prompt"],
                "prompts": [item.prompt for item in items],
            }
        ),
        generation_sha256=canonical_sha256(generation),
    )


def validate_integrity(path: Path, suite: RetentionSuite) -> dict[str, Any]:
    """Validate the pre-exposure prompt, scorer, and overlap audit."""

    audit = _read_object(path)
    _validate_frozen(audit, "integrity_audit_sha256")
    overlaps = audit.get("overlap_sources")
    prompt_hashes = audit.get("candidate_prompt_hashes")
    scorer_hashes = audit.get("candidate_scorer_hashes")
    if (
        audit.get("audit_id") != "foundry-kl-independent-retention-integrity-v1"
        or audit.get("suite_sha256") != suite.suite_sha256
        or audit.get("candidate_count") != 360
        or audit.get("category_counts") != EXPECTED_COUNTS
        or audit.get("candidate_exact_duplicates") != 0
        or audit.get("candidate_normalized_duplicates") != 0
        or audit.get("reference_self_score_failures") != 0
        or audit.get("ambiguous_or_subjective_scorers") != 0
        or audit.get("llm_judge_used") is not False
        or audit.get("sealed_paths_accessed") is not False
        or not isinstance(overlaps, dict)
        or not isinstance(prompt_hashes, list)
        or not isinstance(scorer_hashes, list)
        or len(prompt_hashes) != 360
        or len(scorer_hashes) != 360
    ):
        raise ValueError("KL-independent integrity audit differs")
    required_sources = {
        "vetted_curriculum_400",
        "replay_prompts_83",
        "gsm1k_development_904",
        "previous_retention_prompts",
        "prior_calibration_prompts",
    }
    if set(overlaps) != required_sources:
        raise ValueError("KL-independent overlap source set differs")
    for source in overlaps.values():
        if not isinstance(source, dict) or any(
            source.get(key) != 0
            for key in ("exact_overlap", "normalized_exact_overlap", "contiguous_12_token_overlap")
        ):
            raise ValueError("KL-independent overlap gate failed")
    expected_prompt_hashes = [
        {
            "id": item.item_id,
            "sha256": hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
        }
        for item in suite.items
    ]
    expected_scorer_hashes = [
        {
            "id": item.item_id,
            "sha256": canonical_sha256(
                {
                    "kind": item.kind,
                    "expected": item.expected,
                    "scorer": "foundry.training.retention.score_response",
                }
            ),
        }
        for item in suite.items
    ]
    if prompt_hashes != expected_prompt_hashes or scorer_hashes != expected_scorer_hashes:
        raise ValueError("KL-independent prompt or scorer hashes differ")
    return audit


def _load_model(
    *,
    model_path: Path,
    adapter_path: Path | None,
    torch: Any,
    transformers: Any,
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
        raise RuntimeError("KL-independent evaluation model was offloaded")
    model.eval()
    return model, tokenizer, time.perf_counter() - started, adapter_sha256


def _load_subset(path: Path, suite: RetentionSuite) -> dict[str, Any]:
    subset = _read_object(path)
    _validate_frozen(subset, "subset_sha256")
    raw_items = subset.get("items")
    if (
        subset.get("instrument_id") != INSTRUMENT_ID
        or subset.get("subset_id") != SUBSET_ID
        or subset.get("suite_sha256") != suite.suite_sha256
        or not isinstance(raw_items, list)
        or subset.get("total") != len(raw_items)
    ):
        raise ValueError("KL-independent subset identity differs")
    suite_index = {item.item_id: item for item in suite.items}
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("KL-independent subset item must be an object")
        item = suite_index.get(str(raw.get("id")))
        if item is None or raw.get("section") != item.section or raw.get("skill") != item.skill:
            raise ValueError("KL-independent subset item differs from suite")
    if len({str(raw["id"]) for raw in cast(list[dict[str, Any]], raw_items)}) != len(raw_items):
        raise ValueError("KL-independent subset IDs are duplicated")
    return subset


def evaluate(
    *,
    suite_path: Path,
    model_path: Path,
    raw_path: Path,
    summary_path: Path,
    adapter_path: Path | None = None,
    subset_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the untouched base or one selected adapter deterministically."""

    if raw_path.exists() or summary_path.exists():
        raise FileExistsError("KL-independent evaluation output path already exists")
    suite = load_suite(suite_path)
    subset = None if subset_path is None else _load_subset(subset_path, suite)
    if subset is None:
        evaluated_items = suite.items
    else:
        index = {item.item_id: item for item in suite.items}
        evaluated_items = tuple(index[str(item["id"])] for item in subset["items"])
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    psutil = importlib.import_module("psutil")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    rows: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    backend_failures = 0
    started = time.perf_counter()
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
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=False,
                    max_new_tokens=suite.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = generated[0, input_ids.shape[-1] :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            input_tokens += int(input_ids.shape[-1])
            output_tokens += int(generated_ids.shape[-1])
            score = score_response(item, response)
        except Exception as error:  # pragma: no cover - hardware failure guard
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
        peak_rss = max(peak_rss, int(process.memory_info().rss))
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
    for section in SECTION_ORDER:
        selected = [row for row in rows if row["section"] == section]
        correct = sum(bool(cast(dict[str, Any], row["score"])["correct"]) for row in selected)
        section_metrics[section] = {
            "total": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected),
        }
    total = len(rows)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "suite_sha256": suite.suite_sha256,
        "prompt_sha256": suite.prompt_sha256,
        "generation_sha256": suite.generation_sha256,
        "suite_file_sha256": file_sha256(suite_path),
        "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "adapter_sha256": adapter_sha256,
        "subset_sha256": None if subset is None else subset["subset_sha256"],
        "section_metrics": section_metrics,
        "total": total,
        "correct": sum(bool(cast(dict[str, Any], row["score"])["correct"]) for row in rows),
        "extractable": sum(bool(cast(dict[str, Any], row["score"])["extractable"]) for row in rows),
        "exact_format": sum(
            bool(cast(dict[str, Any], row["score"])["exact_format"]) for row in rows
        ),
        "prompt_echo": sum(bool(cast(dict[str, Any], row["score"])["prompt_echo"]) for row in rows),
        "question_generation": sum(
            bool(cast(dict[str, Any], row["score"])["question_generation"]) for row in rows
        ),
        "malformed_outputs": sum(
            bool(cast(dict[str, Any], row["score"])["malformed"]) for row in rows
        ),
        "backend_failures": backend_failures,
        "runtime_seconds": runtime,
        "load_seconds": load_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "peak_process_rss_bytes": peak_rss,
        "raw_packet_sha256": file_sha256(raw_path),
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def freeze_base_record(
    *,
    suite_path: Path,
    integrity_path: Path,
    base_raw_path: Path,
    base_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the content-free base-correct subset and publication record."""

    suite = load_suite(suite_path)
    integrity = validate_integrity(integrity_path, suite)
    summary = _read_object(base_summary_path)
    _validate_frozen(summary, "summary_sha256")
    rows = _read_rows(base_raw_path)
    if (
        summary.get("adapter_sha256") is not None
        or summary.get("suite_sha256") != suite.suite_sha256
        or summary.get("raw_packet_sha256") != file_sha256(base_raw_path)
        or summary.get("total") != 360
        or len(rows) != 360
        or [row.get("id") for row in rows] != [item.item_id for item in suite.items]
    ):
        raise ValueError("KL-independent untouched-base result differs")
    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for item, row in zip(suite.items, rows, strict=True):
        score = row.get("score")
        if not isinstance(score, dict) or not isinstance(score.get("correct"), bool):
            raise ValueError("KL-independent base row lacks objective decision")
        if score["correct"]:
            selected.append({"id": item.item_id, "section": item.section, "skill": item.skill})
            counts[item.section] += 1
    subset: dict[str, Any] = {
        "schema_version": 1,
        "instrument_id": INSTRUMENT_ID,
        "subset_id": SUBSET_ID,
        "definition": "frozen_scorer_correct_on_untouched_base_before_adapter_exposure",
        "suite_sha256": suite.suite_sha256,
        "base_summary_sha256": summary["summary_sha256"],
        "section_counts": {section: counts[section] for section in SECTION_ORDER},
        "total": len(selected),
        "items": selected,
        "adapter_exposure_before_freeze": False,
        "prompts_or_references_in_manifest": False,
    }
    subset["subset_sha256"] = canonical_sha256(subset)
    gate_checks = {
        f"{section}_at_least_{BASE_CORRECT_MINIMUMS[section]}": (
            counts[section] >= BASE_CORRECT_MINIMUMS[section]
        )
        for section in SECTION_ORDER
    }
    gate_checks[f"total_at_least_{BASE_CORRECT_TOTAL_MINIMUM}"] = (
        len(selected) >= BASE_CORRECT_TOTAL_MINIMUM
    )
    gate_checks["zero_backend_failures"] = summary.get("backend_failures") == 0
    gate_checks["zero_reference_or_scorer_defects"] = (
        integrity.get("reference_self_score_failures") == 0
        and integrity.get("ambiguous_or_subjective_scorers") == 0
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "instrument_id": INSTRUMENT_ID,
        "decision": (
            "holdout_frozen_before_adapter_exposure"
            if all(gate_checks.values())
            else "base_usability_blocker"
        ),
        "candidate_suite": {
            "candidate_count": 360,
            "category_counts": EXPECTED_COUNTS,
            "suite_sha256": suite.suite_sha256,
            "suite_file_sha256": file_sha256(suite_path),
            "prompt_sha256": suite.prompt_sha256,
            "generation_sha256": suite.generation_sha256,
            "candidate_prompt_hashes": integrity["candidate_prompt_hashes"],
            "candidate_scorer_hashes": integrity["candidate_scorer_hashes"],
        },
        "integrity": {
            "integrity_audit_sha256": integrity["integrity_audit_sha256"],
            "overlap_sources": integrity["overlap_sources"],
            "reference_self_score_failures": integrity["reference_self_score_failures"],
            "ambiguous_or_subjective_scorers": integrity["ambiguous_or_subjective_scorers"],
            "llm_judge_used": False,
            "sealed_paths_accessed": False,
        },
        "untouched_base": {
            "summary_sha256": summary["summary_sha256"],
            "summary_file_sha256": file_sha256(base_summary_path),
            "raw_packet_sha256": summary["raw_packet_sha256"],
            "section_metrics": summary["section_metrics"],
            "correct": summary["correct"],
            "total": summary["total"],
            "extractable": summary["extractable"],
            "exact_format": summary["exact_format"],
            "prompt_echo": summary["prompt_echo"],
            "question_generation": summary["question_generation"],
            "malformed_outputs": summary["malformed_outputs"],
            "backend_failures": summary["backend_failures"],
            "runtime_seconds": summary["runtime_seconds"],
            "load_seconds": summary["load_seconds"],
            "input_tokens": summary["input_tokens"],
            "output_tokens": summary["output_tokens"],
            "peak_vram_allocated_bytes": summary["peak_vram_allocated_bytes"],
            "peak_vram_reserved_bytes": summary["peak_vram_reserved_bytes"],
            "peak_process_rss_bytes": summary["peak_process_rss_bytes"],
        },
        "base_correct_subset": subset,
        "gate_checks": gate_checks,
        "adapter_exposure_before_freeze": False,
        "previous_independent_subset_used": False,
        "sealed_boundary_status": SEALED_BOUNDARY_STATUS,
    }
    record["holdout_record_sha256"] = canonical_sha256(record)
    return subset, record


def assess_adapter(
    *,
    suite_path: Path,
    subset_path: Path,
    raw_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Apply the unchanged retention gate to one selected adapter result."""

    suite = load_suite(suite_path)
    subset = _load_subset(subset_path, suite)
    summary = _read_object(summary_path)
    _validate_frozen(summary, "summary_sha256")
    rows = _read_rows(raw_path)
    raw_items = cast(list[dict[str, Any]], subset["items"])
    if (
        not isinstance(summary.get("adapter_sha256"), str)
        or summary.get("suite_sha256") != suite.suite_sha256
        or summary.get("subset_sha256") != subset["subset_sha256"]
        or summary.get("raw_packet_sha256") != file_sha256(raw_path)
        or len(rows) != len(raw_items)
        or [row.get("id") for row in rows] != [item["id"] for item in raw_items]
    ):
        raise ValueError("KL-independent adapter result differs")
    correct: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    broken_ids: list[str] = []
    for item, row in zip(raw_items, rows, strict=True):
        section = str(item["section"])
        totals[section] += 1
        score = row.get("score")
        if not isinstance(score, dict) or not isinstance(score.get("correct"), bool):
            raise ValueError("KL-independent adapter row lacks objective decision")
        if score["correct"]:
            correct[section] += 1
        else:
            broken_ids.append(str(item["id"]))
            failures[f"{section}/{item['skill']}"] += 1
    preserved = sum(correct.values())
    total = len(rows)
    section_preservation = {
        section: {
            "preserved": correct[section],
            "total": totals[section],
            "rate": correct[section] / totals[section],
        }
        for section in SECTION_ORDER
    }
    prompt_echo = int(summary["prompt_echo"])
    gate_checks = {
        "overall_preservation_at_least_90_percent": preserved / total >= 0.90,
        **{
            f"{section}_preservation_at_least_90_percent": (
                section_preservation[section]["rate"] >= 0.90
            )
            for section in SECTION_ORDER
        },
        "overall_wilson_lower_bound_at_least_85_percent": (
            wilson_lower_bound(preserved, total) >= 0.85
        ),
        "prompt_echo_at_most_2_percent": prompt_echo / total <= 0.02,
        "zero_question_generation": summary.get("question_generation") == 0,
        "zero_backend_failures": summary.get("backend_failures") == 0,
        "maximum_failure_family_at_most_3": max(failures.values(), default=0) <= 3,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "assessment_id": "foundry-kl-independent-retention-assessment-v1",
        "instrument_id": INSTRUMENT_ID,
        "suite_sha256": suite.suite_sha256,
        "subset_sha256": subset["subset_sha256"],
        "adapter_sha256": summary["adapter_sha256"],
        "evaluation_summary_sha256": summary["summary_sha256"],
        "raw_packet_sha256": summary["raw_packet_sha256"],
        "total": total,
        "preserved": preserved,
        "broken": total - preserved,
        "overall_preservation": preserved / total,
        "overall_wilson_95_lower_bound": wilson_lower_bound(preserved, total),
        "section_preservation": section_preservation,
        "prompt_echo": prompt_echo,
        "question_generation": summary["question_generation"],
        "backend_failures": summary["backend_failures"],
        "broken_item_ids": broken_ids,
        "failure_families": dict(sorted(failures.items())),
        "maximum_failure_family": max(failures.values(), default=0),
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }
    result["assessment_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--suite", type=Path, required=True)
    validate.add_argument("--integrity", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--suite", type=Path, required=True)
    evaluate_parser.add_argument("--model-path", type=Path, required=True)
    evaluate_parser.add_argument("--adapter", type=Path)
    evaluate_parser.add_argument("--subset", type=Path)
    evaluate_parser.add_argument("--raw", type=Path, required=True)
    evaluate_parser.add_argument("--summary", type=Path, required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--suite", type=Path, required=True)
    freeze.add_argument("--integrity", type=Path, required=True)
    freeze.add_argument("--base-raw", type=Path, required=True)
    freeze.add_argument("--base-summary", type=Path, required=True)
    freeze.add_argument("--subset-output", type=Path, required=True)
    freeze.add_argument("--record-output", type=Path, required=True)
    assess = subparsers.add_parser("assess")
    assess.add_argument("--suite", type=Path, required=True)
    assess.add_argument("--subset", type=Path, required=True)
    assess.add_argument("--raw", type=Path, required=True)
    assess.add_argument("--summary", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        suite = load_suite(args.suite)
        integrity = validate_integrity(args.integrity, suite)
        result = {
            "suite_sha256": suite.suite_sha256,
            "integrity_audit_sha256": integrity["integrity_audit_sha256"],
            "decision": "pass",
        }
    elif args.command == "evaluate":
        result = evaluate(
            suite_path=args.suite,
            model_path=args.model_path,
            adapter_path=args.adapter,
            subset_path=args.subset,
            raw_path=args.raw,
            summary_path=args.summary,
        )
    elif args.command == "freeze":
        subset, result = freeze_base_record(
            suite_path=args.suite,
            integrity_path=args.integrity,
            base_raw_path=args.base_raw,
            base_summary_path=args.base_summary,
        )
        args.subset_output.write_text(
            json.dumps(subset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.record_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        result = assess_adapter(
            suite_path=args.suite,
            subset_path=args.subset,
            raw_path=args.raw,
            summary_path=args.summary,
        )
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
