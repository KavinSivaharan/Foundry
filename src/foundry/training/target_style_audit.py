"""Deterministic content-free audit of frozen assistant-only v3 targets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.assistant_only import normalized_assistant_completion
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

INTERNAL_TERMS = re.compile(
    r"\b(?:typed ledger|update \d+|program step|state node|ledger operation|"
    r"semantic frame|template id|generator|operation node|internal node)\b",
    re.IGNORECASE,
)
PROCEDURAL_PREFIX = re.compile(
    r"^(?:add|subtract|multiply|divide|partition|represent|treat|update|"
    r"the typed|the exact|exact equality|the resource difference|"
    r"the remaining|combine)",
    re.IGNORECASE,
)
EQUATION = re.compile(r"(?:\d+(?:\.\d+)?\s*){1,3}[+\-*/×÷]\s*\d+(?:\.\d+)?\s*=")
FINAL_LINE = re.compile(r"^Final answer:\s*[-+]?\d+(?:/\d+|\.\d+)?$", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _question_restatement(question: str, completion: str) -> bool:
    question_tokens = _tokens(question)
    completion_tokens = _tokens(completion)
    if len(question_tokens) < 8:
        return False
    question_ngrams = {
        tuple(question_tokens[index : index + 8]) for index in range(len(question_tokens) - 7)
    }
    return any(
        tuple(completion_tokens[index : index + 8]) in question_ngrams
        for index in range(max(0, len(completion_tokens) - 7))
    )


def analyze_target(record: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Return content-free metrics and style categories for one completion."""

    completion = normalized_assistant_completion(record)
    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    reasoning = lines[:-1]
    sentence_count = sum(len(re.findall(r"[.!?](?:\s|$)", line)) for line in reasoning)
    equation_count = sum(len(EQUATION.findall(line)) for line in reasoning)
    internal_term_count = sum(len(INTERNAL_TERMS.findall(line)) for line in reasoning)
    procedural_lines = sum(bool(PROCEDURAL_PREFIX.search(line)) for line in reasoning)
    procedural_ratio = procedural_lines / len(reasoning) if reasoning else 0.0
    final_count = sum(bool(FINAL_LINE.fullmatch(line)) for line in lines)
    answer = str(record["canonical_final_answer"])
    answer_before_final = any(
        re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", line) for line in reasoning
    )
    prefixes = [" ".join(_tokens(line)[:3]) for line in reasoning if _tokens(line)]
    repeated_structural_phrases = sum(
        count - 1 for count in Counter(prefixes).values() if count > 1
    )
    assistant_tokens = len(
        cast(
            list[int],
            tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"],
        )
    )
    categories: list[str] = []
    if procedural_ratio >= 0.5:
        categories.append("procedural_or_program_trace_style")
    elif equation_count and len(reasoning) <= 4 and assistant_tokens <= 128:
        categories.append("concise_equation_based_reasoning")
    elif assistant_tokens <= 64 and len(reasoning) <= 4:
        categories.append("concise_natural_reasoning")
    else:
        categories.append("verbose_natural_reasoning")
    if internal_term_count:
        categories.append("internal_operation_terminology")
    if _question_restatement(str(record["rendered_question"]), completion):
        categories.append("question_restatement")
    if repeated_structural_phrases:
        categories.append("repeated_structural_phrase")
    if final_count > 1:
        categories.append("multiple_final_answer_lines")
    if answer_before_final:
        categories.append("answer_before_final_line")
    if final_count != 1 or not FINAL_LINE.fullmatch(lines[-1]):
        categories.append("malformed_final_line")
    return {
        "synthetic_id": str(record["synthetic_id"]),
        "group": str(record["group"]),
        "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
        "assistant_tokens": assistant_tokens,
        "sentence_count": sentence_count,
        "line_count": len(lines),
        "reasoning_line_count": len(reasoning),
        "equation_count": equation_count,
        "internal_term_count": internal_term_count,
        "procedural_line_count": procedural_lines,
        "procedural_ratio": procedural_ratio,
        "repeated_structural_phrases": repeated_structural_phrases,
        "final_answer_line_count": final_count,
        "final_answer_is_last": bool(lines and FINAL_LINE.fullmatch(lines[-1])),
        "answer_before_final_line": answer_before_final,
        "categories": categories,
        "completion": completion,
    }


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    values = [float(row[field]) for row in rows]
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "total": sum(values),
    }


def _load_records(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("target-style source row must be an object")
            rows.append(cast(dict[str, Any], value))
    if len(rows) != 1000 or len({str(row["synthetic_id"]) for row in rows}) != 1000:
        raise ValueError("target-style audit requires 1,000 unique records")
    return rows


def run_audit(
    *,
    model_path: Path,
    source_paths: tuple[Path, ...],
    raw_directory: Path,
    output_path: Path,
    codex_inspection_path: Path | None = None,
) -> dict[str, Any]:
    """Audit all targets and write a blind deterministic inspection packet."""

    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    records = _load_records(source_paths)
    analyzed = [analyze_target(record, tokenizer) for record in records]
    raw_directory.mkdir(parents=True, exist_ok=True)
    packet_hashes: dict[str, str] = {}
    arms: dict[str, Any] = {}
    for arm in ("generic_control", "targeted"):
        arm_rows = [row for row in analyzed if row["group"] == arm]
        if len(arm_rows) != 500:
            raise ValueError(f"{arm} target-style rows differ")
        selected = sorted(
            arm_rows,
            key=lambda row: (
                hashlib.sha256(f"20260720:{row['synthetic_id']}:style".encode()).hexdigest(),
                row["synthetic_id"],
            ),
        )[:50]
        packet = raw_directory / f"{arm}_blind_sample.jsonl"
        packet.write_text(
            "".join(
                json.dumps(
                    {
                        "synthetic_id": row["synthetic_id"],
                        "completion": row["completion"],
                    },
                    sort_keys=True,
                )
                + "\n"
                for row in selected
            ),
            encoding="utf-8",
        )
        packet_hashes[arm] = file_sha256(packet)
        categories = Counter(
            category for row in arm_rows for category in cast(list[str], row["categories"])
        )
        arms[arm] = {
            "records": len(arm_rows),
            "category_counts": dict(sorted(categories.items())),
            "assistant_tokens": _distribution(arm_rows, "assistant_tokens"),
            "sentence_count": _distribution(arm_rows, "sentence_count"),
            "line_count": _distribution(arm_rows, "line_count"),
            "equation_count": _distribution(arm_rows, "equation_count"),
            "internal_term_count": _distribution(arm_rows, "internal_term_count"),
            "procedural_ratio": _distribution(arm_rows, "procedural_ratio"),
            "repeated_structural_phrases": _distribution(arm_rows, "repeated_structural_phrases"),
            "final_answer_line_count": _distribution(arm_rows, "final_answer_line_count"),
            "final_answer_is_last_records": sum(row["final_answer_is_last"] for row in arm_rows),
            "blind_sample_records": len(selected),
            "blind_sample_sha256": packet_hashes[arm],
            "aggregate_sha256": canonical_sha256(
                [
                    {key: value for key, value in row.items() if key != "completion"}
                    for row in arm_rows
                ]
            ),
        }
    codex_inspection: dict[str, Any] | None = None
    if codex_inspection_path is not None:
        raw_inspection: object = json.loads(codex_inspection_path.read_text(encoding="utf-8"))
        if not isinstance(raw_inspection, dict):
            raise ValueError("Codex inspection must be an object")
        codex_inspection = cast(dict[str, Any], raw_inspection)
        inspection_arms = codex_inspection.get("arms")
        if not isinstance(inspection_arms, dict):
            raise ValueError("Codex inspection arms are required")
        for arm in ("generic_control", "targeted"):
            value = inspection_arms.get(arm)
            if not isinstance(value, dict):
                raise ValueError(f"Codex inspection lacks {arm}")
            if value.get("records") != 50 or value.get("sample_sha256") != packet_hashes[arm]:
                raise ValueError("Codex inspection differs from the frozen blind sample")
        codex_inspection = {
            "inspection_id": codex_inspection["inspection_id"],
            "ai_assisted_not_human_review": bool(codex_inspection["ai_assisted_not_human_review"]),
            "arms": inspection_arms,
            "inspection_sha256": file_sha256(codex_inspection_path),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-assistant-target-style-audit-v1",
        "records": len(analyzed),
        "classification_is_deterministic_and_content_free": True,
        "codex_blind_inspection_status": (
            "complete_ai_assisted_not_human_review" if codex_inspection else "pending"
        ),
        "codex_blind_inspection": codex_inspection,
        "arms": arms,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--codex-inspection-path", type=Path)
    args = parser.parse_args()
    summary = run_audit(
        model_path=args.model_path,
        source_paths=tuple(args.source),
        raw_directory=args.raw_directory,
        output_path=args.output_path,
        codex_inspection_path=args.codex_inspection_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
