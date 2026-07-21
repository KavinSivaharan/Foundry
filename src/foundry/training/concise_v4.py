"""Deterministic concise equation-grounded assistant targets for SFT v4."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import statistics
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from foundry.training.assistant_only import tokenize_assistant_completion
from foundry.training.config import (
    canonical_sha256,
    concise_assistant_v4_format_contract_sha256,
)
from foundry.training.qlora import file_sha256

V4_ID = "foundry-concise-assistant-sft-v4"
MAX_ASSISTANT_TOKENS = 128
FORBIDDEN = re.compile(
    r"\b(?:update \d+|program step|state node|ledger operation|typed ledger|"
    r"semantic frame|template|generator|metadata|internal node|operator node)\b",
    re.IGNORECASE,
)
FINAL = re.compile(r"^Final answer:\s*[-+]?\d+(?:/\d+|\.\d+)?$")


def _match(pattern: str, text: str) -> re.Match[str]:
    value = re.search(pattern, text, re.IGNORECASE)
    if value is None:
        raise ValueError(f"unsupported deterministic trace form: {pattern}")
    return value


def _canonical(value: Fraction) -> str:
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def _expected(record: dict[str, Any]) -> Fraction:
    return Fraction(str(record["canonical_final_answer"]))


def _signed_expression(initial: int, updates: list[int]) -> str:
    pieces = [str(initial)]
    for update in updates:
        pieces.extend(("+" if update >= 0 else "-", str(abs(update))))
    return " ".join(pieces)


def concise_reasoning(record: dict[str, Any]) -> tuple[tuple[str, ...], Fraction]:
    """Compile the frozen deterministic trace into concise equations and replay its answer."""

    raw_trace = record.get("deterministic_solution_trace")
    if not isinstance(raw_trace, list) or not all(isinstance(item, str) for item in raw_trace):
        raise ValueError("deterministic trace must be a string list")
    trace = cast(list[str], raw_trace)
    mode = str(record["mode"])
    lines: list[str]
    answer: Fraction
    if mode in {"inventory", "grouping"}:
        initial = int(_match(r"starts with (\d+)", trace[0]).group(1))
        update_lines = trace[1:] if mode == "inventory" else trace[1:-1]
        updates: list[int] = []
        state = initial
        for line in update_lines:
            match = _match(r"by ([+-]\d+), giving (\d+)", line)
            update, declared = int(match.group(1)), int(match.group(2))
            state += update
            if state != declared:
                raise ValueError("inventory trace state does not replay")
            updates.append(update)
        expression = _signed_expression(initial, updates)
        if mode == "inventory":
            answer = Fraction(state)
            lines = [f"{expression} = {_canonical(answer)}."]
        else:
            match = _match(r"balance of (\d+) into groups of (\d+); this gives (\d+)", trace[-1])
            declared_state, divisor, declared_answer = map(int, match.groups())
            if state != declared_state:
                raise ValueError("grouping trace closing state differs")
            answer = Fraction(state // divisor)
            if state % divisor or answer != declared_answer:
                raise ValueError("grouping trace is not an exact complete-group result")
            lines = [f"({expression}) / {divisor} = {_canonical(answer)}."]
    elif mode == "combined_rate":
        rate = int(_match(r"obtain (\d+) per interval", trace[0]).group(1))
        intervals = int(_match(r"by (\d+) intervals", trace[1]).group(1))
        answer = Fraction(rate * intervals)
        lines = [f"The combined rate is {rate} per interval; {rate} × {intervals} = {answer}."]
    elif mode == "complete_packages":
        total, size, declared = map(
            int,
            _match(r"division: (\d+) divided by (\d+) gives (\d+)", trace[0]).groups(),
        )
        quotient, remainder = divmod(total, size)
        if quotient != declared:
            raise ValueError("complete-package quotient differs")
        answer = Fraction(quotient)
        lines = [f"{total} = {size} × {quotient} + {remainder}."]
    elif mode == "dual_capacity":
        values = [int(value) for value in re.findall(r"permit (\d+) complete", " ".join(trace))]
        if len(values) != 2:
            raise ValueError("dual-capacity trace requires two capacities")
        answer = Fraction(min(values))
        lines = [f"min({values[0]}, {values[1]}) = {answer}."]
    elif mode == "equal_distribution":
        total, divisor, declared = map(int, _match(r"gives (\d+)/(\d+) = (\d+)", trace[0]).groups())
        answer = Fraction(total, divisor)
        if answer != declared:
            raise ValueError("equal-distribution trace differs")
        lines = [f"{total} / {divisor} = {_canonical(answer)}."]
    elif mode == "percentage":
        percent = int(_match(r"Represent (\d+)%", trace[0]).group(1))
        total = int(_match(r"Multiply (\d+) by", trace[1]).group(1))
        answer = Fraction(total * percent, 100)
        lines = [f"{percent}/100 × {total} = {_canonical(answer)}."]
    elif mode == "rate_total":
        rate, cycles = map(int, _match(r"rate (\d+) by (\d+) cycles", trace[0]).groups())
        answer = Fraction(rate * cycles)
        lines = [f"{rate} × {cycles} = {answer}."]
    elif mode == "ratio_scale":
        total, first, scale = map(
            int,
            _match(
                r"Divide (\d+) by the first ratio part (\d+) to obtain scale (\d+)", trace[0]
            ).groups(),
        )
        scale_again, second = map(
            int, _match(r"Multiply scale (\d+) by the second part (\d+)", trace[1]).groups()
        )
        if Fraction(total, first) != scale or scale_again != scale:
            raise ValueError("ratio-scale trace differs")
        answer = Fraction(scale * second)
        lines = [f"{total} / {first} = {scale}.", f"{scale} × {second} = {answer}."]
    elif mode == "two_type_allocation":
        all_b = int(_match(r"would use (\d+) parts", trace[0]).group(1))
        difference = int(_match(r"resource difference is (\d+)", trace[1]).group(1))
        per_difference = int(_match(r"per-design difference (\d+)", trace[2]).group(1))
        actual = all_b - difference
        answer = Fraction(difference, per_difference)
        lines = [
            f"{all_b} - {actual} = {difference}.",
            f"{difference} / {per_difference} = {_canonical(answer)}.",
        ]
    elif mode == "weighted_average":
        weighted = Fraction(_match(r"weighted total is ([\d/]+)", trace[0]).group(1))
        weight = int(_match(r"total weight (\d+)", trace[1]).group(1))
        answer = weighted / weight
        lines = [f"{_canonical(weighted)} / {weight} = {_canonical(answer)}."]
    else:
        raise ValueError(f"unsupported v4 mode: {mode}")
    if not 1 <= len(lines) <= 4 or answer != _expected(record):
        raise ValueError("concise reasoning does not replay the canonical answer")
    return tuple(lines), answer


def concise_completion(record: dict[str, Any]) -> str:
    """Return one validated concise completion with exactly one terminal line."""

    lines, answer = concise_reasoning(record)
    terminal = f"Final answer: {_canonical(answer)}"
    completion = "\n".join((*lines, terminal))
    if (
        FORBIDDEN.search(completion)
        or "```" in completion
        or "{" in completion
        or sum(bool(FINAL.fullmatch(line)) for line in completion.splitlines()) != 1
        or completion.splitlines()[-1] != terminal
    ):
        raise ValueError("concise completion violates the v4 surface contract")
    if not all(bool(re.search(r"(?:=|×|/|min\()", line)) for line in completion.splitlines()[:-1]):
        raise ValueError("every concise reasoning line must be equation-grounded")
    if not all(
        bool(record.get(key))
        for key in (
            "primary_verifier_success",
            "independent_verifier_success",
            "verifier_agreement",
        )
    ):
        raise ValueError("frozen mathematical verifier evidence does not pass")
    return completion


def validate_and_tokenize_v4(
    record: dict[str, Any], tokenizer: Any, *, max_length: int
) -> tuple[dict[str, list[int]], Any, str]:
    """Validate, tokenize, and enforce the 128-token assistant-target bound."""

    completion = concise_completion(record)
    tokenized, evidence = tokenize_assistant_completion(
        record, completion, tokenizer, max_length=max_length
    )
    if evidence.assistant_loss_tokens > MAX_ASSISTANT_TOKENS:
        raise ValueError("concise assistant target exceeds 128 loss-bearing tokens")
    return tokenized, evidence, completion


def format_and_tokenize_concise_v4(
    records: list[dict[str, Any]], tokenizer: Any, *, max_length: int
) -> tuple[list[dict[str, list[int]]], int, int, tuple[Any, ...]]:
    """Tokenize a stable record sequence under the concise-v4 contract."""

    values: list[dict[str, list[int]]] = []
    evidence: list[Any] = []
    for record in records:
        value, item, _ = validate_and_tokenize_v4(record, tokenizer, max_length=max_length)
        values.append(value)
        evidence.append(item)
    return (
        values,
        sum(item.assistant_loss_tokens for item in evidence),
        sum(item.truncated_tokens > 0 for item in evidence),
        tuple(evidence),
    )


def _distribution(values: list[int]) -> dict[str, int | float]:
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "total": sum(values),
    }


def _load(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("concise-v4 source row must be an object")
            rows.append(cast(dict[str, Any], value))
    if len(rows) != 1000 or len({str(row["synthetic_id"]) for row in rows}) != 1000:
        raise ValueError("concise-v4 reconstruction requires 1,000 unique rows")
    return rows


def run_reconstruction(
    *,
    model_path: Path,
    source_paths: tuple[Path, ...],
    raw_path: Path,
    output_path: Path,
    max_length: int = 512,
) -> dict[str, Any]:
    """Reconstruct all v4 targets twice and freeze content-free evidence."""

    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = _load(source_paths)
    results: list[dict[str, Any]] = []
    for record in rows:
        _, evidence, completion = validate_and_tokenize_v4(record, tokenizer, max_length=max_length)
        replay = concise_completion(record)
        if replay != completion:
            raise ValueError("concise-v4 reconstruction differs")
        results.append(
            {
                "synthetic_id": record["synthetic_id"],
                "group": record["group"],
                "future_split": record["future_split"],
                "family": record["family"],
                "mode": record["mode"],
                "difficulty": record["difficulty"],
                "output_contract_enabled": record["output_contract_enabled"],
                "assistant_loss_tokens": evidence.assistant_loss_tokens,
                "reasoning_lines": len(completion.splitlines()) - 1,
                "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                "formatted_text_sha256": evidence.formatted_text_sha256,
                "completion": completion,
            }
        )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    arms: dict[str, Any] = {}
    for arm in ("generic_control", "targeted"):
        arm_rows = [row for row in results if row["group"] == arm]
        training_rows = [row for row in arm_rows if row["future_split"] == "training"]
        validation_rows = [row for row in arm_rows if row["future_split"] == "synthetic_validation"]
        arms[arm] = {
            "records": len(arm_rows),
            "training_records": len(training_rows),
            "validation_records": len(validation_rows),
            "assistant_tokens": _distribution(
                [int(row["assistant_loss_tokens"]) for row in arm_rows]
            ),
            "training_assistant_tokens": _distribution(
                [int(row["assistant_loss_tokens"]) for row in training_rows]
            ),
            "validation_assistant_tokens": _distribution(
                [int(row["assistant_loss_tokens"]) for row in validation_rows]
            ),
            "reasoning_line_counts": dict(
                sorted(Counter(str(row["reasoning_lines"]) for row in arm_rows).items())
            ),
            "mode_counts": dict(sorted(Counter(str(row["mode"]) for row in arm_rows).items())),
            "aggregate_sha256": canonical_sha256(
                [
                    {key: value for key, value in row.items() if key != "completion"}
                    for row in arm_rows
                ]
            ),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "format_id": V4_ID,
        "format_sha256": concise_assistant_v4_format_contract_sha256(),
        "maximum_assistant_loss_tokens": MAX_ASSISTANT_TOKENS,
        "records": len(results),
        "rejected_records": 0,
        "deterministic_reconstruction_match": True,
        "primary_and_independent_verification_passed": True,
        "exact_canonical_answer_replay_passed": True,
        "assistant_only_masking_passed": True,
        "raw_reconstruction_sha256": file_sha256(raw_path),
        "reconstruction_sha256": canonical_sha256(
            [{key: value for key, value in row.items() if key != "completion"} for row in results]
        ),
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
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    summary = run_reconstruction(
        model_path=args.model_path,
        source_paths=tuple(args.source),
        raw_path=args.raw_path,
        output_path=args.output_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
