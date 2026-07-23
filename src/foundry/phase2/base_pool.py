from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

from foundry.config import (
    DatasetConfig,
    EvaluationConfig,
    GenerationConfig,
    ModelConfig,
    PartitionConfig,
    PromptConfig,
)
from foundry.evaluation.answer_extraction import (
    CANONICAL_EXTRACTOR_ID,
    CanonicalExtractionError,
    canonical_extractor_sha256,
    extract_canonical_number,
    serialize_canonical_number,
)
from foundry.evaluation.backends import HuggingFaceCudaBackend, ModelBackend
from foundry.evaluation.prompting import ChatMessage
from foundry.phase2.asdiv import canonical_sha256, file_sha256
from foundry.phase2.capacity import question_token_length

MODEL_ID: Final = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION: Final = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ASDIV_COMMIT: Final = "883f90a9a65bf00304ba8f37423910fe743abc47"
MATHQA_COMMIT: Final = "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6"
SEED: Final = 20260720
SYSTEM_INSTRUCTION: Final = (
    "Solve the math problem carefully. Show concise reasoning and end with exactly one line in "
    "this form: Final answer: <canonical-number>"
)
REPLAY_SEED: Final = "foundry-phase2-asdiv-base-replay-v1"
MATHQA_REPLAY_SEED: Final = "foundry-phase2-mathqa-base-replay-v1"
REPLAY_SIZE: Final = 30
_EXACT_FINAL_LINE: Final = re.compile(
    r"^Final answer:\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?$"
)


@dataclass(frozen=True)
class PoolEvaluationConfig:
    schema_version: int
    candidate_corpus: str
    candidate_source_commit: str
    model_id: str
    model_revision: str
    seed: int
    system_instruction: str
    canonical_extractor_id: str
    replay_sample_seed: str
    replay_sample_size: int
    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int


@dataclass(frozen=True)
class PoolPrediction:
    source_id: str
    family: str
    grade: str
    operation_count: int
    formula_depth: int
    answer_type: str
    question_token_count: int
    response: str | None
    output_sha256: str | None
    predicted_answer: int | str | None
    extractable: bool
    correct: bool
    exact_format_compliant: bool
    extraction_failure_category: str | None
    generation_truncated: bool
    input_tokens: int | None
    output_tokens: int | None
    generation_seconds: float
    backend_error: str | None


def _load_mapping(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain an object")
    return cast(dict[str, object], raw)


def _string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def _integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"field {key!r} must be an integer")
    return value


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"field {key!r} must be numerical")
    return float(value)


def load_pool_config(path: Path) -> PoolEvaluationConfig:
    root = _load_mapping(path)
    decoding_raw = root.get("decoding")
    if not isinstance(decoding_raw, dict):
        raise ValueError("decoding must be an object")
    decoding = cast(dict[str, object], decoding_raw)
    config = PoolEvaluationConfig(
        schema_version=_integer(root, "schema_version"),
        candidate_corpus=_string(root, "candidate_corpus"),
        candidate_source_commit=_string(root, "candidate_source_commit"),
        model_id=_string(root, "model_id"),
        model_revision=_string(root, "model_revision"),
        seed=_integer(root, "seed"),
        system_instruction=_string(root, "system_instruction"),
        canonical_extractor_id=_string(root, "canonical_extractor_id"),
        replay_sample_seed=_string(root, "replay_sample_seed"),
        replay_sample_size=_integer(root, "replay_sample_size"),
        do_sample=bool(decoding.get("do_sample")),
        temperature=_number(decoding, "temperature"),
        top_p=_number(decoding, "top_p"),
        max_new_tokens=_integer(decoding, "max_new_tokens"),
    )
    expected_asdiv = PoolEvaluationConfig(
        schema_version=1,
        candidate_corpus="ASDiv V1.0 verified uncontaminated pool",
        candidate_source_commit=ASDIV_COMMIT,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        seed=SEED,
        system_instruction=SYSTEM_INSTRUCTION,
        canonical_extractor_id=CANONICAL_EXTRACTOR_ID,
        replay_sample_seed=REPLAY_SEED,
        replay_sample_size=REPLAY_SIZE,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=512,
    )
    expected_mathqa = PoolEvaluationConfig(
        schema_version=1,
        candidate_corpus="MathQA verified uncontaminated train subset",
        candidate_source_commit=MATHQA_COMMIT,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        seed=SEED,
        system_instruction=SYSTEM_INSTRUCTION,
        canonical_extractor_id=CANONICAL_EXTRACTOR_ID,
        replay_sample_seed=MATHQA_REPLAY_SEED,
        replay_sample_size=REPLAY_SIZE,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=512,
    )
    if config not in {expected_asdiv, expected_mathqa}:
        raise ValueError("base-pool evaluation config differs from the frozen contract")
    return config


def load_candidates(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: object = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            row = cast(dict[str, object], raw)
            _string(row, "source_id")
            _string(row, "combined_question")
            _string(row, "canonical_answer")
            rows.append(row)
    rows.sort(key=lambda row: _string(row, "source_id"))
    source_ids = [_string(row, "source_id") for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("candidate source IDs are not unique")
    return rows


def _backend_config(config: PoolEvaluationConfig, candidate_count: int) -> EvaluationConfig:
    return EvaluationConfig(
        schema_version=1,
        model=ModelConfig(
            repo_id=config.model_id,
            revision=config.model_revision,
            dtype="float16",
            device="cuda",
        ),
        dataset=DatasetConfig(
            repo_id="MathQA" if config.candidate_source_commit == MATHQA_COMMIT else "ASDiv",
            revision=config.candidate_source_commit,
            config_name="verified_uncontaminated",
            source_split="external_pool",
            expected_examples=candidate_count,
        ),
        partition=PartitionConfig(
            seed="phase2-external-pool-only",
            sealed_final_size=0,
            development_manifest="not_used",
            sealed_final_manifest="not_used",
        ),
        prompt=PromptConfig(system=config.system_instruction, user_template="{question}"),
        generation=GenerationConfig(
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
        ),
    )


def initialize_determinism(seed: int) -> None:
    random.seed(seed)
    torch: Any = importlib.import_module("torch")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _messages(question: str, system: str) -> tuple[ChatMessage, ChatMessage]:
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    )


def _exact_format(response: str) -> bool:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    markers = [line for line in lines if line.startswith("Final answer:")]
    return (
        len(markers) == 1
        and lines[-1] == markers[0]
        and bool(_EXACT_FINAL_LINE.fullmatch(lines[-1]))
    )


def _prediction_content(prediction: PoolPrediction) -> dict[str, object]:
    payload = asdict(prediction)
    payload.pop("generation_seconds")
    return payload


def _evaluate_one(
    row: dict[str, object],
    config: PoolEvaluationConfig,
    generation: GenerationConfig,
    backend: ModelBackend,
) -> PoolPrediction:
    source_id = _string(row, "source_id")
    question = _string(row, "combined_question")
    expected = Fraction(_string(row, "canonical_answer"))
    started = time.perf_counter()
    try:
        result = backend.generate(
            source_id, _messages(question, config.system_instruction), generation
        )
        elapsed = time.perf_counter() - started
        truncated = (
            result.output_tokens is not None and result.output_tokens >= config.max_new_tokens
        )
        predicted: Fraction | None = None
        failure_category: str | None = None
        try:
            predicted = extract_canonical_number(result.text, generation_truncated=truncated)
        except CanonicalExtractionError as error:
            failure_category = error.category
        return PoolPrediction(
            source_id=source_id,
            family=_string(row, "family"),
            grade=_string(row, "grade"),
            operation_count=_integer(row, "operation_count"),
            formula_depth=_integer(row, "formula_depth"),
            answer_type=_string(row, "answer_type"),
            question_token_count=question_token_length(question),
            response=result.text,
            output_sha256=hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
            predicted_answer=(
                serialize_canonical_number(predicted) if predicted is not None else None
            ),
            extractable=predicted is not None,
            correct=predicted == expected,
            exact_format_compliant=_exact_format(result.text),
            extraction_failure_category=failure_category,
            generation_truncated=truncated,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            generation_seconds=elapsed,
            backend_error=None,
        )
    except Exception as error:  # noqa: BLE001 - persist every backend failure
        return PoolPrediction(
            source_id=source_id,
            family=_string(row, "family"),
            grade=_string(row, "grade"),
            operation_count=_integer(row, "operation_count"),
            formula_depth=_integer(row, "formula_depth"),
            answer_type=_string(row, "answer_type"),
            question_token_count=question_token_length(question),
            response=None,
            output_sha256=None,
            predicted_answer=None,
            extractable=False,
            correct=False,
            exact_format_compliant=False,
            extraction_failure_category="backend_failure",
            generation_truncated=False,
            input_tokens=None,
            output_tokens=None,
            generation_seconds=time.perf_counter() - started,
            backend_error=f"{type(error).__name__}: {error}",
        )


def replay_sample_ids(rows: Sequence[dict[str, object]], seed: str, size: int) -> tuple[str, ...]:
    ranked = sorted(
        (_string(row, "source_id") for row in rows),
        key=lambda source_id: hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest(),
    )
    if size <= 0 or size > len(ranked):
        raise ValueError("replay sample size is invalid")
    return tuple(ranked[:size])


def run_base_pool_evaluation(
    *,
    config_path: Path,
    candidates_path: Path,
    output_dir: Path,
    backend_factory: Callable[[EvaluationConfig], ModelBackend] = HuggingFaceCudaBackend,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict[str, object]:
    config = load_pool_config(config_path)
    rows = load_candidates(candidates_path)
    backend_config = _backend_config(config, len(rows))
    initialize_determinism(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    if prediction_path.exists() and prediction_path.stat().st_size:
        raise FileExistsError(f"refusing to overwrite existing predictions: {prediction_path}")
    backend = backend_factory(backend_config)
    started = time.perf_counter()
    predictions: list[PoolPrediction] = []
    progress_points = {
        max(1, (len(rows) * percent + 99) // 100): percent for percent in (25, 50, 75, 100)
    }
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for completed, row in enumerate(rows, start=1):
            prediction = _evaluate_one(row, config, backend_config.generation, backend)
            predictions.append(prediction)
            handle.write(json.dumps(asdict(prediction), sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            percent = progress_points.get(completed)
            if percent is not None and progress_callback is not None:
                progress_callback(completed, len(rows), percent)
    evaluation_seconds = time.perf_counter() - started

    by_id = {_string(row, "source_id"): row for row in rows}
    first_by_id = {prediction.source_id: prediction for prediction in predictions}
    replay_ids = replay_sample_ids(rows, config.replay_sample_seed, config.replay_sample_size)
    replay_evidence: list[dict[str, object]] = []
    for source_id in replay_ids:
        replay = _evaluate_one(by_id[source_id], config, backend_config.generation, backend)
        original = first_by_id[source_id]
        identical = _prediction_content(replay) == _prediction_content(original)
        replay_evidence.append(
            {
                "source_id": source_id,
                "original_output_sha256": original.output_sha256,
                "replay_output_sha256": replay.output_sha256,
                "identical": identical,
            }
        )
    if not all(bool(item["identical"]) for item in replay_evidence):
        raise RuntimeError("fixed 30-example base-pool replay is not exact")

    processed = len(predictions)
    backend_failures = sum(item.backend_error is not None for item in predictions)
    correct = sum(item.correct for item in predictions)
    extractable = sum(item.extractable for item in predictions)
    exact_format = sum(item.exact_format_compliant for item in predictions)
    truncated = sum(item.generation_truncated for item in predictions)
    input_tokens = sum(item.input_tokens or 0 for item in predictions)
    output_tokens = sum(item.output_tokens or 0 for item in predictions)
    failure_counts = Counter(
        item.extraction_failure_category
        for item in predictions
        if item.extraction_failure_category is not None
    )
    family_counts = Counter(item.family for item in predictions)
    family_failures = Counter(item.family for item in predictions if not item.correct)
    output_content = [_prediction_content(item) for item in predictions]
    metrics = backend.metrics()
    summary: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "config_sha256": canonical_sha256(asdict(config)),
        "model_backend": backend.name,
        "canonical_extractor_sha256": canonical_extractor_sha256(),
        "candidate_manifest_sha256": file_sha256(candidates_path),
        "candidate_count": len(rows),
        "processed": processed,
        "ids_accounted_for": len(first_by_id) == len(rows),
        "backend_failures": backend_failures,
        "correct": correct,
        "accuracy": correct / processed,
        "extractable": extractable,
        "extractability": extractable / processed,
        "exact_format_compliant": exact_format,
        "exact_format_compliance": exact_format / processed,
        "incorrect": processed - correct,
        "unextractable": processed - extractable,
        "truncated": truncated,
        "failure_counts": dict(sorted(failure_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "base_failed_family_counts": dict(sorted(family_failures.items())),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "evaluation_seconds": evaluation_seconds,
        "examples_per_second": processed / evaluation_seconds,
        "generated_tokens_per_second": output_tokens / evaluation_seconds,
        "backend_metrics": metrics,
        "replay_sample_ids": replay_ids,
        "replay_evidence_sha256": canonical_sha256(replay_evidence),
        "replay_exact": True,
        "prediction_content_sha256": canonical_sha256(output_content),
        "raw_prediction_file_sha256": file_sha256(prediction_path),
        "labels_derived_from_model_output": False,
        "gsm1k_exposed_to_selection": False,
        "sealed_final_accessed": False,
    }
    summary["aggregate_result_sha256"] = canonical_sha256(summary)
    (output_dir / "replay_evidence.json").write_text(
        json.dumps(replay_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def _progress(completed: int, total: int, percent: int) -> None:
    print(
        json.dumps(
            {
                "base_candidate_evaluation_completed": completed,
                "base_candidate_evaluation_percent": percent,
                "base_candidate_evaluation_total": total,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen on the clean Phase 2 candidate pool"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_base_pool_evaluation(
        config_path=args.config,
        candidates_path=args.candidates,
        output_dir=args.output_dir,
        progress_callback=_progress,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
