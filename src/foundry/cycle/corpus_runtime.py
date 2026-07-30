"""Tokenizer-bound corpus freezing and GSM1K contamination audit."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

from foundry.config import load_config
from foundry.cycle.contract import (
    CycleConfig,
    validate_file_identity,
    validate_process_environment,
    verified_payload,
)
from foundry.cycle.corpus import build_continuation_corpus, build_smoke_corpus
from foundry.cycle.generation import select_smoke_records
from foundry.cycle.selection import audit_gsm1k_overlap
from foundry.evaluation.benchmark import load_huggingface_examples
from foundry.evaluation.calibration import load_development_subset
from foundry.evaluation.manifests import load_manifest
from foundry.evaluation.validation import (
    FINAL_MAIN_DEVELOPMENT_BASELINE,
    as_frozen_baseline_manifest,
    assert_final_evaluator_config,
    load_answer_validation_manifest,
    load_final_evaluator_manifest,
)
from foundry.phase2.launch_contract import validate_preimport
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def _gsm1k_questions(config: CycleConfig) -> list[str]:
    root = config.artifact_root
    base_config = load_config(root / "configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml")
    evaluator_config = load_config(root / "configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml")
    assert_final_evaluator_config(base_config, evaluator_config)
    development = load_manifest(
        root / "configs/eval/manifests/gsm1k_development.json",
        base_config,
    )
    source_pool = load_development_subset(
        root / "configs/eval/manifests/gsm1k_development_baseline.json",
        development,
    )
    source_baseline = load_answer_validation_manifest(
        root / "configs/eval/manifests/gsm1k_development_baseline_844.json",
        source_pool,
        development,
    )
    baseline = load_final_evaluator_manifest(
        root / "configs/eval/manifests/gsm1k_development_baseline_814.json",
        source_baseline,
        development,
    )
    if baseline.purpose != FINAL_MAIN_DEVELOPMENT_BASELINE or len(baseline.entries) != 814:
        raise ValueError("contamination audit requires the frozen 814 GSM1K IDs")
    manifest = as_frozen_baseline_manifest(
        baseline,
        source_baseline,
        development,
        evaluator_config,
    )
    return [item.question for item in load_huggingface_examples(evaluator_config, manifest)]


def freeze_production_corpus(
    *,
    config: CycleConfig,
    selected_trace_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Audit overlap and freeze the exact 32k-token production corpus."""

    validate_process_environment(config=config)
    validate_preimport()
    dataset = config.section("dataset")
    corpus = config.section("corpus")
    model = config.section("model")
    selection = verified_payload(
        selected_trace_path.parent / "summary.json",
        "selection_sha256",
    )
    if (
        file_sha256(selected_trace_path) != selection["selected_trace_file_sha256"]
        or selection["passed_before_overlap_audit"] is not True
    ):
        raise ValueError("selected-trace artifact identity or coverage gate differs")
    targeted_path = validate_file_identity(
        config,
        str(dataset["targeted_training_relative_path"]),
        str(dataset["targeted_training_sha256"]),
    )
    records = _jsonl(targeted_path)
    selected = _jsonl(selected_trace_path)
    prompts = {str(item["source_id"]): str(item["question"]) for item in records}
    traces = {str(item["source_id"]): str(item["completion"]) for item in selected}
    overlap = audit_gsm1k_overlap(
        training_prompts=prompts,
        selected_traces=traces,
        gsm1k_questions=_gsm1k_questions(config),
        window_tokens=12,
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "overlap_audit.json").write_text(
        json.dumps(overlap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not overlap["passed"]:
        result = {
            "schema_version": 1,
            "corpus_freeze_id": "foundry-cycle1-corpus-freeze-v1",
            "passed": False,
            "overlap_audit": overlap,
            "classification": "verifier_filtered_optimizer_signal_insufficient",
        }
        result["corpus_freeze_sha256"] = canonical_sha256(result)
        (output_directory / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(config.resolve_artifact(str(model["snapshot_relative_path"]))),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    manifest = build_continuation_corpus(
        targeted_training_path=targeted_path,
        selected_trace_path=selected_trace_path,
        source_schedule_path=validate_file_identity(
            config,
            str(corpus["source_schedule_relative_path"]),
            str(corpus["source_schedule_file_sha256"]),
        ),
        expected_source_schedule_sha256=str(corpus["source_schedule_sha256"]),
        replay_path=config.resolve_artifact(str(corpus["replay_relative_source_path"])),
        expected_replay_file_sha256=str(corpus["replay_file_sha256"]),
        tokenizer=tokenizer,
        output_directory=output_directory / "corpus",
    )
    if (
        manifest["source_schedule_prefix_sha256"] != corpus["source_schedule_32_step_prefix_sha256"]
        or manifest["task_tokens"] != corpus["task_assistant_tokens"]
        or manifest["replay_tokens"] != corpus["replay_assistant_tokens"]
        or manifest["total_tokens"] != corpus["total_assistant_tokens"]
        or manifest["all_prompt_mixtures_exact"] is not True
        or manifest["no_holdout_or_gsm_records"] is not True
    ):
        raise ValueError("frozen production corpus reconstruction differs")
    result = {
        "schema_version": 1,
        "corpus_freeze_id": "foundry-cycle1-corpus-freeze-v1",
        "passed": True,
        "overlap_audit": overlap,
        "corpus": manifest,
    }
    result["corpus_freeze_sha256"] = canonical_sha256(result)
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def freeze_smoke_corpus(
    *,
    config: CycleConfig,
    selected_trace_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Freeze the exact four-record compatibility corpus."""

    validate_process_environment(config=config)
    validate_preimport()
    dataset = config.section("dataset")
    model = config.section("model")
    selection = verified_payload(
        selected_trace_path.parent / "summary.json",
        "selection_sha256",
    )
    if file_sha256(selected_trace_path) != selection["selected_trace_file_sha256"]:
        raise ValueError("compatibility selected-trace artifact identity differs")
    records = select_smoke_records(
        _jsonl(
            validate_file_identity(
                config,
                str(dataset["targeted_training_relative_path"]),
                str(dataset["targeted_training_sha256"]),
            )
        )
    )
    selected = {str(item["source_id"]): item for item in _jsonl(selected_trace_path)}
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(config.resolve_artifact(str(model["snapshot_relative_path"]))),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return build_smoke_corpus(
        records=records,
        selected_traces=selected,
        tokenizer=tokenizer,
        output_directory=output_directory,
    )
