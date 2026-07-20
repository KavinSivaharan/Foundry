"""Audit synthetic completions and their alignment with the frozen evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.config import load_config
from foundry.evaluation.prompting import prompt_sha256
from foundry.training.config import (
    SFT_SYSTEM_PROMPT,
    SFT_USER_PREFIX,
    SFT_USER_SUFFIX,
    canonical_sha256,
    sft_format_contract_sha256,
)

FINAL_PATTERN = re.compile(r"^Final answer:\s*(\S+)\s*$")
OTHER_TERMINAL_PATTERN = re.compile(
    r"^(?:answer|result|therefore|thus|total)\s*(?::|is\b)", re.IGNORECASE
)
TRACE_PREFIX_PATTERNS = (
    re.compile(r"^the typed ledger\b", re.IGNORECASE),
    re.compile(r"^use exact integer\b", re.IGNORECASE),
    re.compile(r"^treat all\b", re.IGNORECASE),
    re.compile(r"^exact equal distribution\b", re.IGNORECASE),
    re.compile(r"^amber parts permit\b", re.IGNORECASE),
    re.compile(r"^divide\b", re.IGNORECASE),
    re.compile(r"^represent\b", re.IGNORECASE),
    re.compile(r"^multiply\b", re.IGNORECASE),
    re.compile(r"^the exact weighted\b", re.IGNORECASE),
    re.compile(r"^add the two\b", re.IGNORECASE),
)


def _answer_occurs(text: str, answer: str) -> bool:
    pattern = re.compile(rf"(?<![\w.]){re.escape(answer)}(?!\w)")
    return pattern.search(text) is not None


def classify_completion(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one completion without changing or interpreting its mathematics."""

    completion = str(record["training_completion"])
    answer = str(record["canonical_final_answer"])
    lines = completion.strip().splitlines()
    final_matches = [FINAL_PATTERN.fullmatch(line.strip()) for line in lines]
    explicit_final_lines = [match for match in final_matches if match is not None]
    expected_terminal = f"Final answer: {answer}"
    another_terminal = [
        line for line in lines if OTHER_TERMINAL_PATTERN.match(line.strip()) is not None
    ]
    answer_before_end = _answer_occurs("\n".join(lines[:-1]), answer)
    first_line = lines[0].strip() if lines else ""
    trace_like = any(pattern.match(first_line) is not None for pattern in TRACE_PREFIX_PATTERNS)
    worksheet_like = "?" in completion or any(
        phrase in completion.lower()
        for phrase in ("write a question", "create a problem", "worksheet")
    )
    return {
        "exactly_one_clear_final_answer": len(explicit_final_lines) == 1
        and explicit_final_lines[0].group(1) == answer,
        "ends_with_exact_terminal": bool(lines) and lines[-1].strip() == expected_terminal,
        "uses_another_terminal_form": bool(another_terminal),
        "answer_occurs_before_final_line": answer_before_end,
        "contains_multiple_answer_markers": len(explicit_final_lines) + len(another_terminal) > 1,
        "looks_like_internal_program_trace": trace_like,
        "looks_like_worksheet_generation": worksheet_like,
        "output_contract_enabled": bool(record["output_contract_enabled"]),
        "trace_preserved_exactly": completion
        in {
            "\n".join(cast(list[str], record["deterministic_solution_trace"])),
            "\n".join(
                [
                    *cast(list[str], record["deterministic_solution_trace"]),
                    expected_terminal,
                ]
            ),
        },
        "explicit_final_line_count": len(explicit_final_lines),
        "other_terminal_line_count": len(another_terminal),
    }


def _load_arm(paths: tuple[Path, Path], expected_group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value: object = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("synthetic record must be an object")
                row = cast(dict[str, Any], value)
                if row.get("group") != expected_group or row.get("final_decision") != "accepted":
                    raise ValueError("synthetic record violates group or acceptance contract")
                rows.append(row)
    if len(rows) != 500:
        raise ValueError(f"{expected_group} requires exactly 500 rows")
    return rows


def audit_arm(*, group: str, paths: tuple[Path, Path], packet_path: Path) -> dict[str, Any]:
    """Audit all completions in one arm and write an ignored 30-row sample."""

    records = _load_arm(paths, group)
    classifications = [classify_completion(record) for record in records]
    keys = (
        "exactly_one_clear_final_answer",
        "ends_with_exact_terminal",
        "uses_another_terminal_form",
        "answer_occurs_before_final_line",
        "contains_multiple_answer_markers",
        "looks_like_internal_program_trace",
        "looks_like_worksheet_generation",
        "output_contract_enabled",
        "trace_preserved_exactly",
    )
    counts = {key: sum(bool(item[key]) for item in classifications) for key in keys}
    final_line_distribution = Counter(
        int(item["explicit_final_line_count"]) for item in classifications
    )
    selected_indices = sorted(
        range(len(records)),
        key=lambda index: hashlib.sha256(
            str(records[index]["synthetic_id"]).encode("utf-8")
        ).hexdigest(),
    )[:30]
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    with packet_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in selected_indices:
            record = records[index]
            payload = {
                "inspection_kind": "AI-assisted completion-format inspection",
                "synthetic_id": record["synthetic_id"],
                "completion": record["training_completion"],
                "canonical_answer": record["canonical_final_answer"],
                "classification": classifications[index],
            }
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    result: dict[str, Any] = {
        "group": group,
        "records": len(records),
        "counts": counts,
        "explicit_final_line_count_distribution": {
            str(key): value for key, value in sorted(final_line_distribution.items())
        },
        "packet_records": len(selected_indices),
        "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
    }
    result["aggregate_sha256"] = canonical_sha256(result)
    return result


def run_audit(
    *,
    targeted_training: Path,
    targeted_validation: Path,
    generic_training: Path,
    generic_validation: Path,
    evaluation_config: Path,
    raw_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the full 1,000-completion and prompt-alignment audit."""

    evaluation = load_config(evaluation_config)
    arms = {
        "generic_control": audit_arm(
            group="generic_control",
            paths=(generic_training, generic_validation),
            packet_path=raw_directory / "generic_control_completion_audit.jsonl",
        ),
        "targeted": audit_arm(
            group="targeted",
            paths=(targeted_training, targeted_validation),
            packet_path=raw_directory / "targeted_completion_audit.jsonl",
        ),
    }
    training_user_template = f"{SFT_USER_PREFIX}{{question}}{SFT_USER_SUFFIX}"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-sft-evaluation-format-alignment-v1",
        "arms": arms,
        "alignment": {
            "training_system_sha256": hashlib.sha256(SFT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "evaluation_system_sha256": hashlib.sha256(
                evaluation.prompt.system.encode("utf-8")
            ).hexdigest(),
            "system_text_exact_match": SFT_SYSTEM_PROMPT == evaluation.prompt.system,
            "training_user_template_sha256": hashlib.sha256(
                training_user_template.encode("utf-8")
            ).hexdigest(),
            "evaluation_user_template_sha256": hashlib.sha256(
                evaluation.prompt.user_template.encode("utf-8")
            ).hexdigest(),
            "user_text_exact_match": training_user_template == evaluation.prompt.user_template,
            "training_prompt_contract_sha256": sft_format_contract_sha256(),
            "evaluation_prompt_sha256": prompt_sha256(evaluation.prompt),
            "training_roles": ["system", "user", "assistant"],
            "evaluation_roles": ["system", "user", "assistant_generation_prompt"],
            "role_order_aligned": True,
            "training_add_generation_prompt": False,
            "evaluation_add_generation_prompt": True,
            "training_terminal_placeholder": "<canonical-number>",
            "evaluation_terminal_placeholder": "<integer>",
            "terminal_line_wording_aligned": False,
            "training_response_style": (
                "deterministic procedural trace; terminal line only on output-contract track"
            ),
            "evaluation_response_style": (
                "model-generated concise reasoning; exactly one required terminal line"
            ),
            "assistant_style_aligned": False,
            "training_eos_behavior": (
                "full assistant message appends Qwen im_end and newline; all were loss-bearing"
            ),
            "evaluation_eos_behavior": (
                "generation stops on tokenizer EOS and decoded special tokens are removed"
            ),
            "same_chat_template_and_special_tokens": True,
        },
    }
    summary["completion_contract_defect_found"] = any(
        arm["counts"]["ends_with_exact_terminal"] != arm["records"] for arm in arms.values()
    )
    summary["summary_sha256"] = canonical_sha256(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run the completion and alignment audit CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-training", type=Path, required=True)
    parser.add_argument("--targeted-validation", type=Path, required=True)
    parser.add_argument("--generic-training", type=Path, required=True)
    parser.add_argument("--generic-validation", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    summary = run_audit(
        targeted_training=args.targeted_training,
        targeted_validation=args.targeted_validation,
        generic_training=args.generic_training,
        generic_validation=args.generic_validation,
        evaluation_config=args.evaluation_config,
        raw_directory=args.raw_directory,
        output_path=args.output_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
