"""Content-free characterization of the preserved collapsed-adapter outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.synthesis.contamination import DevelopmentQuestion, normalize_text
from foundry.training.config import canonical_sha256

_FINAL_ANSWER = re.compile(r"(?im)^\s*Final answer\s*:")
_RAW_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:[.,/]\d+)*")
_QUESTION_PHRASES = (
    "how many",
    "how much",
    "what is",
    "what was",
    "find the",
    "calculate the",
)
_REASONING_TOKENS = frozenset(
    {
        "because",
        "calculate",
        "first",
        "multiply",
        "divide",
        "subtract",
        "add",
        "remaining",
        "therefore",
        "total",
        "so",
        "then",
    }
)
_ROLE_MARKERS = frozenset({"system", "user", "assistant", "solve"})
_TRACE_OPERATION_STARTS = frozenset(
    {"add", "convert", "divide", "multiply", "represent", "subtract", "treat"}
)
_SAFE_TRACE_PREFIXES = frozenset(
    {
        "add the",
        "add up",
        "divide <num>",
        "exact equal",
        "multiply the",
        "represent <num>",
        "the exact",
        "the typed",
        "treat all",
    }
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"prediction row {line_number} is not an object")
        row = cast(dict[str, Any], value)
        if not isinstance(row.get("stable_id"), str) or not isinstance(row.get("response"), str):
            raise ValueError(f"prediction row {line_number} lacks stable ID or response")
        rows.append(row)
    if len(rows) != 814 or len({str(row["stable_id"]) for row in rows}) != 814:
        raise ValueError("collapse diagnosis requires 814 unique prediction rows")
    return rows


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _has_repetition(tokens: list[str]) -> bool:
    if any(left == right for left, right in zip(tokens, tokens[1:], strict=False)):
        return True
    if len(tokens) < 9:
        return False
    trigrams = Counter(tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2))
    return max(trigrams.values(), default=0) >= 3


def _question_overlap(response_tokens: list[str], question_tokens: list[str]) -> tuple[bool, float]:
    if not response_tokens or not question_tokens:
        return False, 0.0
    prefix_size = min(8, len(question_tokens))
    prefix_echo = (
        prefix_size >= 5 and response_tokens[:prefix_size] == question_tokens[:prefix_size]
    )
    left = set(response_tokens)
    right = set(question_tokens)
    jaccard = len(left & right) / len(left | right)
    return prefix_echo, jaccard


def _first_pattern(tokens: list[str]) -> str:
    if not tokens:
        return "empty"
    first = tokens[0]
    pair = " ".join(tokens[:2])
    if pair == "final answer":
        return "final_answer"
    if pair in {"to find", "we need", "the answer"} or first in {
        "first",
        "because",
        "since",
        "therefore",
    }:
        return "reasoning_connective"
    if pair in _QUESTION_PHRASES or first in {"how", "what", "which"}:
        return "question_phrase"
    if first in _ROLE_MARKERS:
        return "role_or_prompt_marker"
    if pair in {"the exact", "the typed", "exact equal"}:
        return "synthetic_trace_metadata_phrase"
    if first in _TRACE_OPERATION_STARTS:
        return "synthetic_trace_operation"
    if first == "num" or first.isdigit():
        return "numeric"
    return "other_lexical"


def _classify(row: dict[str, Any], question: DevelopmentQuestion) -> tuple[str, dict[str, object]]:
    response = str(row["response"])
    normalized_response = normalize_text(response, replace_numbers=True)
    response_tokens = normalized_response.split()
    question_tokens = normalize_text(question.question, replace_numbers=True).split()
    final_count = len(_FINAL_ANSWER.findall(response))
    prefix_echo, question_jaccard = _question_overlap(response_tokens, question_tokens)
    question_phrase = any(phrase in normalized_response for phrase in _QUESTION_PHRASES)
    role_marker = bool(response_tokens and response_tokens[0] in _ROLE_MARKERS)
    safe_trace_prefix = " ".join(response_tokens[:2])
    training_trace_prefix = safe_trace_prefix in _SAFE_TRACE_PREFIXES or bool(
        response_tokens and response_tokens[0] in _TRACE_OPERATION_STARTS
    )
    repetition = _has_repetition(response_tokens)
    reasoning_like = bool(set(response_tokens) & _REASONING_TOKENS) and bool(
        _RAW_NUMBER.search(response)
    )
    truncated = bool(row.get("generation_truncated"))
    failure = row.get("extraction_failure_category")
    exact_format = bool(row.get("exact_format_compliant"))
    predicted = row.get("predicted_answer") is not None

    if truncated:
        category = "token_limit_truncation"
    elif len(response_tokens) <= 3:
        category = "near_empty_or_premature_eos"
    elif prefix_echo or question_jaccard >= 0.72:
        category = "prompt_or_question_echo"
    elif role_marker or (question_phrase and "?" in response):
        category = "question_generation_or_transcript_continuation"
    elif repetition:
        category = "repetitive_output"
    elif exact_format and final_count == 1:
        category = "terminal_answer_contract_present"
    elif failure in {
        "ambiguous_terminal_answer",
        "conflicting_answers",
        "malformed_terminal_answer",
    }:
        category = "ambiguous_or_malformed_answer"
    elif predicted:
        category = "answer_in_unsupported_form"
    elif failure == "no_terminal_answer" and reasoning_like:
        category = "reasoning_like_without_terminal_answer"
    else:
        category = "unrelated_or_unclassifiable_prose"

    appears_to_answer = category in {
        "terminal_answer_contract_present",
        "answer_in_unsupported_form",
        "ambiguous_or_malformed_answer",
        "reasoning_like_without_terminal_answer",
    }
    appears_to_continue = category in {
        "prompt_or_question_echo",
        "question_generation_or_transcript_continuation",
    }
    return category, {
        "stable_id": row["stable_id"],
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "category": category,
        "output_tokens": int(row["output_tokens"]),
        "inferred_eos_before_limit": int(row["output_tokens"]) < 768 and not truncated,
        "final_answer_prefix_count": final_count,
        "prefix_echo": prefix_echo,
        "question_token_jaccard": question_jaccard,
        "question_phrase": question_phrase,
        "role_marker": role_marker,
        "training_trace_prefix": training_trace_prefix,
        "safe_trace_prefix": safe_trace_prefix
        if safe_trace_prefix in _SAFE_TRACE_PREFIXES
        else None,
        "repetition": repetition,
        "reasoning_like": reasoning_like,
        "first_pattern": _first_pattern(response_tokens),
        "appears_to_answer_user": appears_to_answer,
        "appears_to_continue_transcript": appears_to_continue,
    }


def characterize_outputs(
    *,
    predictions_path: Path,
    questions: tuple[DevelopmentQuestion, ...],
    arm: str,
    raw_output_path: Path,
) -> dict[str, object]:
    """Classify every existing output and return content-free aggregate evidence."""

    question_by_id = {item.stable_id: item for item in questions}
    rows = _load_predictions(predictions_path)
    details: list[dict[str, object]] = []
    for row in rows:
        stable_id = str(row["stable_id"])
        if stable_id not in question_by_id:
            raise ValueError(f"prediction stable ID is absent from development: {stable_id[:12]}")
        _, detail = _classify(row, question_by_id[stable_id])
        details.append(detail)

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in details),
        encoding="utf-8",
    )
    category_counts = Counter(str(item["category"]) for item in details)
    first_patterns = Counter(str(item["first_pattern"]) for item in details)
    safe_trace_prefixes = Counter(
        str(item["safe_trace_prefix"]) for item in details if item["safe_trace_prefix"] is not None
    )
    output_tokens = [_integer(item["output_tokens"], "output_tokens") for item in details]
    result: dict[str, object] = {
        "arm": arm,
        "predictions_sha256": _file_sha256(predictions_path),
        "processed_outputs": len(details),
        "category_counts": dict(sorted(category_counts.items())),
        "first_pattern_counts": dict(sorted(first_patterns.items())),
        "safe_trace_prefix_counts": dict(sorted(safe_trace_prefixes.items())),
        "average_output_tokens": statistics.fmean(output_tokens),
        "median_output_tokens": statistics.median(output_tokens),
        "minimum_output_tokens": min(output_tokens),
        "maximum_output_tokens": max(output_tokens),
        "inferred_eos_before_limit_count": sum(
            bool(item["inferred_eos_before_limit"]) for item in details
        ),
        "token_limit_truncation_count": sum(
            item["category"] == "token_limit_truncation" for item in details
        ),
        "final_answer_prefix_any_count": sum(
            _integer(item["final_answer_prefix_count"], "final_answer_prefix_count") >= 1
            for item in details
        ),
        "final_answer_prefix_exactly_once_count": sum(
            _integer(item["final_answer_prefix_count"], "final_answer_prefix_count") == 1
            for item in details
        ),
        "repetition_detected_count": sum(bool(item["repetition"]) for item in details),
        "prompt_or_question_echo_count": sum(bool(item["prefix_echo"]) for item in details),
        "appears_to_answer_user_count": sum(
            bool(item["appears_to_answer_user"]) for item in details
        ),
        "appears_to_continue_transcript_count": sum(
            bool(item["appears_to_continue_transcript"]) for item in details
        ),
        "training_trace_prefix_count": sum(bool(item["training_trace_prefix"]) for item in details),
        "raw_diagnostics_sha256": _file_sha256(raw_output_path),
    }
    result["arm_summary_sha256"] = canonical_sha256(result)
    return result


def run_characterization(
    *,
    generic_predictions_path: Path,
    targeted_predictions_path: Path,
    questions: tuple[DevelopmentQuestion, ...],
    raw_directory: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Characterize both preserved collapsed adapters without new inference."""

    arms = {
        "generic_control": characterize_outputs(
            predictions_path=generic_predictions_path,
            questions=questions,
            arm="generic_control",
            raw_output_path=raw_directory / "generic_control.jsonl",
        ),
        "targeted": characterize_outputs(
            predictions_path=targeted_predictions_path,
            questions=questions,
            arm="targeted",
            raw_output_path=raw_directory / "targeted.jsonl",
        ),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "classification_policy": "deterministic-collapsed-output-taxonomy-v1",
        "arms": arms,
        "new_benchmark_inference": False,
        "raw_predictions_committed": False,
        "sealed_final_accessed": False,
    }
    result["summary_sha256"] = canonical_sha256(result)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-predictions", required=True, type=Path)
    parser.add_argument("--targeted-predictions", required=True, type=Path)
    parser.add_argument("--evaluation-config", required=True, type=Path)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--raw-directory", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser


def main() -> int:
    from foundry.synthesis.contamination import load_development_questions_for_contamination

    args = _parser().parse_args()
    questions = load_development_questions_for_contamination(
        evaluation_config_path=args.evaluation_config,
        development_manifest_path=args.development_manifest,
    )
    result = run_characterization(
        generic_predictions_path=args.generic_predictions,
        targeted_predictions_path=args.targeted_predictions,
        questions=questions,
        raw_directory=args.raw_directory,
        summary_path=args.summary,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
