"""Deterministic 32-step assistant-token schedules for the retention ladder."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from foundry.training.assistant_only import format_and_tokenize_assistant_only
from foundry.training.concise_v4 import format_and_tokenize_concise_v4
from foundry.training.config import (
    assistant_only_v3_format_contract_sha256,
    canonical_sha256,
    concise_assistant_v4_format_contract_sha256,
    load_qlora_recipe,
)
from foundry.training.qlora import _load_records, file_sha256
from foundry.training.token_matching import (
    ScheduledOccurrence,
    ScheduledStep,
    TokenCensusEntry,
)

SEED = 20260720
ARMS = ("generic_control", "targeted")
FORMATS = ("v3", "v4")
VARIANTS = {
    "a": ("v3", 5e-5),
    "b": ("v4", 5e-5),
    "c": ("v4", 2e-5),
    "d": ("v4", 1e-5),
}
CHECKPOINTS = (8, 16, 24, 32)
BLOCK_TOKENS = 3_600
STEPS_PER_BLOCK = 8


def _stable_key(*values: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, values)).encode()).hexdigest()


def _format_hash(format_id: str) -> str:
    if format_id == "v3":
        return assistant_only_v3_format_contract_sha256()
    if format_id == "v4":
        return concise_assistant_v4_format_contract_sha256()
    raise ValueError("unknown ladder format")


def _build_entries(
    records: list[dict[str, Any]], values: list[dict[str, list[int]]], evidence: tuple[Any, ...]
) -> tuple[TokenCensusEntry, ...]:
    entries: list[TokenCensusEntry] = []
    for record, value, item in zip(records, values, evidence, strict=True):
        loss_tokens = sum(label != -100 for label in value["labels"])
        entries.append(
            TokenCensusEntry(
                synthetic_id=str(record["synthetic_id"]),
                dataset_arm=str(record["group"]),
                family=str(record["family"]),
                submode=str(record["mode"]),
                difficulty=str(record["difficulty"]),
                output_contract_enabled=bool(record["output_contract_enabled"]),
                formatted_input_tokens=int(item.formatted_tokens),
                loss_bearing_tokens=loss_tokens,
                truncated_tokens=int(item.truncated_tokens),
                labels_entirely_masked=loss_tokens == 0,
                formatted_example_sha256=str(item.formatted_text_sha256),
            )
        )
    return tuple(entries)


def _balanced_order(
    entries: tuple[TokenCensusEntry, ...], *, format_id: str, arm: str, block: int
) -> tuple[TokenCensusEntry, ...]:
    grouped: dict[tuple[str, str, bool], list[TokenCensusEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.stratum].append(entry)
    for stratum in grouped:
        grouped[stratum].sort(
            key=lambda item: (
                _stable_key(SEED, format_id, arm, block, item.synthetic_id),
                item.synthetic_id,
            )
        )
    strata = sorted(grouped)
    ordered: list[TokenCensusEntry] = []
    offset = 0
    while True:
        added = False
        for stratum in strata:
            bucket = grouped[stratum]
            if offset < len(bucket):
                ordered.append(bucket[offset])
                added = True
        if not added:
            break
        offset += 1
    return tuple(ordered)


def select_exact_block(
    entries: tuple[TokenCensusEntry, ...], *, format_id: str, arm: str, block: int
) -> tuple[ScheduledOccurrence, ...]:
    """Select one deterministic, stratum-interleaved exact-token subset."""

    ordered = _balanced_order(entries, format_id=format_id, arm=arm, block=block)
    predecessor: list[tuple[int, int] | None] = [None] * (BLOCK_TOKENS + 1)
    predecessor[0] = (-1, -1)
    for index, entry in enumerate(ordered):
        tokens = entry.loss_bearing_tokens
        for total in range(BLOCK_TOKENS, tokens - 1, -1):
            if predecessor[total] is None and predecessor[total - tokens] is not None:
                predecessor[total] = (total - tokens, index)
    if predecessor[BLOCK_TOKENS] is None:
        raise ValueError(f"{format_id}/{arm}/block-{block} has no exact token subset")
    selected: list[ScheduledOccurrence] = []
    total = BLOCK_TOKENS
    while total:
        previous = predecessor[total]
        if previous is None or previous[1] < 0:
            raise RuntimeError("ladder subset reconstruction failed")
        previous_total, index = previous
        entry = ordered[index]
        selected.append(
            ScheduledOccurrence(
                synthetic_id=entry.synthetic_id,
                occurrence_index=block,
                loss_bearing_tokens=entry.loss_bearing_tokens,
            )
        )
        total = previous_total
    selected.sort(key=lambda item: (item.synthetic_id, item.occurrence_index))
    return tuple(selected)


def _steps_for_block(
    occurrences: tuple[ScheduledOccurrence, ...], *, first_step: int
) -> tuple[ScheduledStep, ...]:
    buckets: list[list[ScheduledOccurrence]] = [[] for _ in range(STEPS_PER_BLOCK)]
    totals = [0] * STEPS_PER_BLOCK
    for occurrence in sorted(
        occurrences,
        key=lambda item: (-item.loss_bearing_tokens, item.synthetic_id, item.occurrence_index),
    ):
        target = min(range(STEPS_PER_BLOCK), key=lambda index: (totals[index], index))
        buckets[target].append(occurrence)
        totals[target] += occurrence.loss_bearing_tokens
    return tuple(
        ScheduledStep(
            step=first_step + index,
            occurrences=tuple(
                sorted(bucket, key=lambda item: (item.synthetic_id, item.occurrence_index))
            ),
            loss_bearing_tokens=totals[index],
        )
        for index, bucket in enumerate(buckets)
    )


def build_ladder_schedule(
    entries: tuple[TokenCensusEntry, ...], *, format_id: str, arm: str
) -> tuple[ScheduledStep, ...]:
    """Build four exact 3,600-token blocks and 32 deterministic steps."""

    schedule: list[ScheduledStep] = []
    for block in range(4):
        occurrences = select_exact_block(entries, format_id=format_id, arm=arm, block=block)
        schedule.extend(_steps_for_block(occurrences, first_step=block * STEPS_PER_BLOCK + 1))
    result = tuple(schedule)
    if len(result) != 32 or any(not step.occurrences for step in result):
        raise RuntimeError("ladder schedule shape differs")
    if any(
        sum(step.loss_bearing_tokens for step in result[:checkpoint])
        != BLOCK_TOKENS * (checkpoint // STEPS_PER_BLOCK)
        for checkpoint in CHECKPOINTS
    ):
        raise RuntimeError("ladder checkpoint token totals differ")
    pairs = [
        (item.synthetic_id, item.occurrence_index) for step in result for item in step.occurrences
    ]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("ladder occurrence identity repeats")
    return result


def _schedule_evidence(schedule: tuple[ScheduledStep, ...], path: Path) -> dict[str, Any]:
    payload = [asdict(step) for step in schedule]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoints = {
        str(checkpoint): {
            "cumulative_loss_bearing_tokens": sum(
                step.loss_bearing_tokens for step in schedule[:checkpoint]
            ),
            "prefix_sha256": canonical_sha256(payload[:checkpoint]),
        }
        for checkpoint in CHECKPOINTS
    }
    return {
        "schedule_sha256": canonical_sha256(payload),
        "schedule_file_sha256": file_sha256(path),
        "optimizer_steps": len(schedule),
        "occurrences": sum(len(step.occurrences) for step in schedule),
        "loss_bearing_tokens": sum(step.loss_bearing_tokens for step in schedule),
        "minimum_step_tokens": min(step.loss_bearing_tokens for step in schedule),
        "maximum_step_tokens": max(step.loss_bearing_tokens for step in schedule),
        "mean_step_tokens": statistics.fmean(step.loss_bearing_tokens for step in schedule),
        "checkpoints": checkpoints,
    }


def run_ladder_schedule_build(
    *,
    base_recipe_path: Path,
    model_path: Path,
    train_paths: dict[str, Path],
    raw_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Measure both formats and freeze all four predeclared variant schedules."""

    transformers: Any = importlib.import_module("transformers")
    recipe = load_qlora_recipe(base_recipe_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = {
        arm: _load_records(path, expected_group=arm, expected_split="training")
        for arm, path in train_paths.items()
    }
    entries: dict[str, dict[str, tuple[TokenCensusEntry, ...]]] = defaultdict(dict)
    census: dict[str, Any] = {}
    for format_id in FORMATS:
        census[format_id] = {}
        for arm in ARMS:
            if format_id == "v3":
                values, _, truncated, evidence = format_and_tokenize_assistant_only(
                    records[arm], tokenizer, max_length=recipe.max_sequence_length
                )
            else:
                values, _, truncated, evidence = format_and_tokenize_concise_v4(
                    records[arm], tokenizer, max_length=recipe.max_sequence_length
                )
            arm_entries = _build_entries(records[arm], values, evidence)
            if truncated or any(item.labels_entirely_masked for item in arm_entries):
                raise ValueError("ladder census contains invalid tokenized records")
            entries[format_id][arm] = arm_entries
            raw_census = raw_directory / "census" / f"{format_id}_{arm}.jsonl"
            raw_census.parent.mkdir(parents=True, exist_ok=True)
            raw_census.write_text(
                "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in arm_entries),
                encoding="utf-8",
            )
            counts = [item.loss_bearing_tokens for item in arm_entries]
            census[format_id][arm] = {
                "records": len(arm_entries),
                "census_sha256": canonical_sha256([asdict(item) for item in arm_entries]),
                "census_file_sha256": file_sha256(raw_census),
                "source_loss_bearing_tokens": sum(counts),
                "minimum": min(counts),
                "maximum": max(counts),
                "mean": statistics.fmean(counts),
            }
    schedules_by_format = {
        format_id: {
            arm: build_ladder_schedule(entries[format_id][arm], format_id=format_id, arm=arm)
            for arm in ARMS
        }
        for format_id in FORMATS
    }
    variants: dict[str, Any] = {}
    for variant, (format_id, learning_rate) in VARIANTS.items():
        variant_arms = {
            arm: _schedule_evidence(
                schedules_by_format[format_id][arm],
                raw_directory / "schedules" / f"variant_{variant}_{arm}.json",
            )
            for arm in ARMS
        }
        checkpoint_parity: dict[str, Any] = {}
        for checkpoint in CHECKPOINTS:
            generic = variant_arms["generic_control"]["checkpoints"][str(checkpoint)][
                "cumulative_loss_bearing_tokens"
            ]
            targeted = variant_arms["targeted"]["checkpoints"][str(checkpoint)][
                "cumulative_loss_bearing_tokens"
            ]
            relative = abs(generic - targeted) / max(generic, targeted)
            checkpoint_parity[str(checkpoint)] = {
                "generic_control": generic,
                "targeted": targeted,
                "relative_difference": relative,
                "within_0_5_percent": relative <= 0.005,
            }
        variants[variant] = {
            "format_id": format_id,
            "format_sha256": _format_hash(format_id),
            "learning_rate": learning_rate,
            "arms": variant_arms,
            "checkpoint_token_parity": checkpoint_parity,
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "schedule_id": "foundry-retention-safe-ladder-schedules-v1",
        "seed": SEED,
        "optimizer_steps": 32,
        "checkpoints": list(CHECKPOINTS),
        "whole_example_accumulation": True,
        "block_loss_bearing_tokens": BLOCK_TOKENS,
        "census": census,
        "variants": variants,
        "deterministic_reconstruction": all(
            schedules_by_format[format_id][arm]
            == build_ladder_schedule(entries[format_id][arm], format_id=format_id, arm=arm)
            for format_id in FORMATS
            for arm in ARMS
        ),
        "all_checkpoint_parity_within_0_5_percent": all(
            value["within_0_5_percent"]
            for variant in variants.values()
            for value in variant["checkpoint_token_parity"].values()
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-recipe", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--generic-train", type=Path, required=True)
    parser.add_argument("--targeted-train", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_ladder_schedule_build(
        base_recipe_path=args.base_recipe,
        model_path=args.model_path,
        train_paths={"generic_control": args.generic_train, "targeted": args.targeted_train},
        raw_directory=args.raw_directory,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
