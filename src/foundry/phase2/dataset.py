from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

from foundry.phase2.asdiv import canonical_sha256, execute_formula, file_sha256, serialize_fraction
from foundry.phase2.mathqa import execute_program
from foundry.synthesis.contamination import normalized_text_sha256

DATASET_CONTRACT: Final = "foundry-vetted-curriculum-dataset-v1"
TARGET_FORMAT_ID: Final = "foundry-vetted-formula-target-v1"
SPLIT_SEED: Final = "foundry-phase2-vetted-180-20-v1"
MODEL_REVISION: Final = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ARM_SIZE: Final = 200
TRAIN_SIZE: Final = 180
VALIDATION_SIZE: Final = 20
MAX_ASSISTANT_TOKENS: Final = 128


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: object = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(cast(dict[str, object], raw))
    return rows


def _string(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def _integer(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"field {key!r} must be an integer")
    return value


def _target_format_contract() -> dict[str, object]:
    return {
        "format_id": TARGET_FORMAT_ID,
        "calculation_lines": {"minimum": 1, "maximum": 3},
        "terminal": "Final answer: <canonical-number>",
        "terminal_count": 1,
        "question_restatement": False,
        "source_metadata": False,
        "generator_terminology": False,
        "code_block": False,
        "assistant_token_maximum_including_eos": MAX_ASSISTANT_TOKENS,
        "terminal_eos_count": 1,
    }


def _replay_formula(row: dict[str, object]) -> str:
    formula = _string(row, "formula")
    expected = Fraction(_string(row, "canonical_answer"))
    source = _string(row, "source_corpus")
    if source == "asdiv_v1_0":
        first = execute_formula(formula)
        second = execute_formula(formula)
        actual = first.value
        first_hash = first.program_sha256
        second_hash = second.program_sha256
    elif source == "mathqa_train":
        first_program = execute_program(formula)
        second_program = execute_program(formula)
        actual = first_program.value
        first_hash = first_program.program_sha256
        second_hash = second_program.program_sha256
    else:
        raise ValueError(f"unsupported Phase 2 source: {source}")
    if actual != expected:
        raise ValueError("formula/program replay differs from the canonical answer")
    if first_hash != second_hash or first_hash != _string(row, "program_sha256"):
        raise ValueError("formula/program replay identity differs")
    return serialize_fraction(expected)


def construct_completion(row: dict[str, object]) -> str:
    answer = _replay_formula(row)
    formula = _string(row, "formula")
    operation_count = _integer(row, "operation_count")
    if operation_count <= 6 and len(formula) <= 120:
        calculation = f"Calculation: {formula}"
    else:
        calculation = f"Calculation: {operation_count} operations evaluate to {answer}."
    terminal = f"Final answer: {answer}"
    completion = f"{calculation}\n{terminal}"
    lines = completion.splitlines()
    if not 1 <= len(lines) - 1 <= 3:
        raise ValueError("target requires one to three calculation lines")
    if sum(line.startswith("Final answer: ") for line in lines) != 1 or lines[-1] != terminal:
        raise ValueError("target terminal-answer contract failed")
    if "```" in completion or "generator" in completion.casefold() or "source_" in completion:
        raise ValueError("target contains forbidden surface material")
    question = _string(row, "combined_question")
    if len(question) >= 20 and question.casefold() in completion.casefold():
        raise ValueError("target copied the source question")
    if Fraction(terminal.removeprefix("Final answer: ")) != Fraction(
        _string(row, "canonical_answer")
    ):
        raise ValueError("independent terminal-answer verification failed")
    return completion


def _stratum(row: dict[str, object]) -> tuple[str, ...]:
    return (
        _string(row, "family"),
        _string(row, "source_corpus"),
        _string(row, "grade"),
        _string(row, "answer_type"),
        str(_integer(row, "operation_count")),
    )


def _stable_rank(source_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{source_id}".encode()).hexdigest()


def deterministic_split(rows: Sequence[dict[str, object]]) -> tuple[list[str], list[str]]:
    if len(rows) != ARM_SIZE:
        raise ValueError("dataset split requires exactly 200 rows")
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_stratum(row)].append(row)
    for values in grouped.values():
        values.sort(
            key=lambda row: (_stable_rank(_string(row, "source_id")), _string(row, "source_id"))
        )
    raw_quotas = {key: len(values) * VALIDATION_SIZE / ARM_SIZE for key, values in grouped.items()}
    quotas = {key: math.floor(value) for key, value in raw_quotas.items()}
    remaining = VALIDATION_SIZE - sum(quotas.values())
    order = sorted(
        grouped,
        key=lambda key: (
            -(raw_quotas[key] - quotas[key]),
            canonical_sha256(key),
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    validation = sorted(
        _string(row, "source_id")
        for key, values in grouped.items()
        for row in values[: quotas[key]]
    )
    all_ids = {_string(row, "source_id") for row in rows}
    training = sorted(all_ids - set(validation))
    if len(training) != TRAIN_SIZE or len(validation) != VALIDATION_SIZE:
        raise RuntimeError("deterministic split sizes differ from 180/20")
    return training, validation


def _write_jsonl(path: Path, rows: Iterable[object]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def _arm_dataset(
    rows: list[dict[str, object]], tokenizer: Any, arm: str, output_root: Path
) -> dict[str, object]:
    if len(rows) != ARM_SIZE:
        raise ValueError(f"{arm} repaired manifest does not contain 200 rows")
    training_ids, validation_ids = deterministic_split(rows)
    split_by_id = {source_id: "training" for source_id in training_ids}
    split_by_id.update({source_id: "validation" for source_id in validation_ids})
    built: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: _string(item, "source_id")):
        source_id = _string(row, "source_id")
        completion = construct_completion(row)
        replay = construct_completion(row)
        if completion != replay:
            raise ValueError("target reconstruction differs")
        assistant_tokens = len(tokenizer.encode(completion, add_special_tokens=False)) + 1
        if assistant_tokens > MAX_ASSISTANT_TOKENS:
            raise ValueError(f"assistant target exceeds 128 tokens: {source_id}")
        built.append(
            {
                "source_id": source_id,
                "arm": arm,
                "split": split_by_id[source_id],
                "question": _string(row, "combined_question"),
                "question_sha256": _string(row, "question_sha256"),
                "question_normalized_sha256": normalized_text_sha256(
                    _string(row, "combined_question")
                ),
                "program_sha256": _string(row, "program_sha256"),
                "family": _string(row, "family"),
                "source_corpus": _string(row, "source_corpus"),
                "difficulty": _string(row, "grade"),
                "answer_type": _string(row, "answer_type"),
                "operation_count": _integer(row, "operation_count"),
                "canonical_answer": _string(row, "canonical_answer"),
                "assistant_completion": completion,
                "assistant_completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                "assistant_tokens_including_eos": assistant_tokens,
                "append_exactly_one_eos": True,
            }
        )
    training_rows = [row for row in built if row["split"] == "training"]
    validation_rows = [row for row in built if row["split"] == "validation"]
    training_hash = _write_jsonl(output_root / f"{arm}_training.jsonl", training_rows)
    validation_hash = _write_jsonl(output_root / f"{arm}_validation.jsonl", validation_rows)
    manifest_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"question", "assistant_completion"}
        }
        for row in built
    ]
    manifest_hash = _write_jsonl(output_root / f"{arm}_manifest.jsonl", manifest_rows)
    train_normalized = {str(row["question_normalized_sha256"]) for row in training_rows}
    validation_normalized = {str(row["question_normalized_sha256"]) for row in validation_rows}
    train_programs = {str(row["program_sha256"]) for row in training_rows}
    validation_programs = {str(row["program_sha256"]) for row in validation_rows}
    if train_normalized & validation_normalized or train_programs & validation_programs:
        raise ValueError("training and validation splits overlap by normalized question or program")
    return {
        "records": len(built),
        "training_records": len(training_rows),
        "validation_records": len(validation_rows),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in built).items())),
        "source_counts": dict(sorted(Counter(str(row["source_corpus"]) for row in built).items())),
        "training_sha256": training_hash,
        "validation_sha256": validation_hash,
        "manifest_sha256": manifest_hash,
        "training_ids_sha256": canonical_sha256(training_ids),
        "validation_ids_sha256": canonical_sha256(validation_ids),
        "assistant_tokens_total": sum(
            cast(int, row["assistant_tokens_including_eos"]) for row in built
        ),
        "assistant_tokens_maximum": max(
            cast(int, row["assistant_tokens_including_eos"]) for row in built
        ),
        "target_replay_exact": True,
        "formula_program_replay_passed": True,
        "canonical_answer_equality_passed": True,
        "terminal_contract_passed": True,
        "split_overlap_count": 0,
    }


def build_datasets(
    *,
    targeted_path: Path,
    generic_path: Path,
    repair_summary_path: Path,
    model_path: Path,
    output_root: Path,
) -> dict[str, object]:
    repair = json.loads(repair_summary_path.read_text(encoding="utf-8"))
    if not isinstance(repair, dict) or repair.get("matching_gate_passed") is not True:
        raise ValueError("matching repair did not pass")
    output_root.mkdir(parents=True, exist_ok=True)
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("frozen tokenizer has no EOS token")
    targeted_rows = _load_jsonl(targeted_path)
    generic_rows = _load_jsonl(generic_path)
    if {_string(row, "source_id") for row in targeted_rows} & {
        _string(row, "source_id") for row in generic_rows
    }:
        raise ValueError("repaired arms share source IDs")
    if {_string(row, "question_sha256") for row in targeted_rows} & {
        _string(row, "question_sha256") for row in generic_rows
    }:
        raise ValueError("repaired arms share exact questions")
    if {normalized_text_sha256(_string(row, "combined_question")) for row in targeted_rows} & {
        normalized_text_sha256(_string(row, "combined_question")) for row in generic_rows
    }:
        raise ValueError("repaired arms share normalized questions")
    if {_string(row, "program_sha256") for row in targeted_rows} & {
        _string(row, "program_sha256") for row in generic_rows
    }:
        raise ValueError("repaired arms share latent programs")
    targeted = _arm_dataset(targeted_rows, tokenizer, "targeted", output_root)
    generic = _arm_dataset(generic_rows, tokenizer, "generic", output_root)
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_contract": DATASET_CONTRACT,
        "target_format": _target_format_contract(),
        "target_format_sha256": canonical_sha256(_target_format_contract()),
        "split_seed": SPLIT_SEED,
        "model_revision": MODEL_REVISION,
        "repair_summary_sha256": file_sha256(repair_summary_path),
        "matching_evidence_sha256": repair["matching_evidence_sha256"],
        "targeted": targeted,
        "generic": generic,
        "cross_arm_overlap_count": 0,
        "source_acquisition_rerun": False,
        "source_verification_rerun": False,
        "contamination_rerun": False,
        "base_evaluation_rerun": False,
        "sealed_final_accessed": False,
    }
    summary["dataset_sha256"] = canonical_sha256(summary)
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen vetted-corpus targets and splits")
    parser.add_argument("--targeted", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--repair-summary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_datasets(
        targeted_path=args.targeted,
        generic_path=args.generic,
        repair_summary_path=args.repair_summary,
        model_path=args.model_path,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
