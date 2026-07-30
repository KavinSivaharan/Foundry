"""Deterministic 32k-token continuation corpus construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import vetted_qlora_kl as qlora
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256


@dataclass(frozen=True)
class VariantOccurrence:
    source_id: str
    variant: str
    tokens: int


@dataclass(frozen=True)
class ScheduleUnit:
    source_id: str
    occurrences: tuple[VariantOccurrence, ...]

    @property
    def tokens(self) -> int:
        return sum(item.tokens for item in self.occurrences)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _assistant_tokens(record: dict[str, Any], tokenizer: Any) -> int:
    tokenized = qlora._tokenize({**record, "kind": "vetted"}, tokenizer)
    return sum(label != -100 for label in tokenized["labels"])


def _bounded_exact_fill(
    units: tuple[ScheduleUnit, ...],
    budget: int,
) -> tuple[ScheduleUnit, ...]:
    """Fill exactly while keeping added repeats deterministically bounded and balanced."""

    if budget == 0:
        return ()
    ordered = tuple(sorted(units, key=lambda item: (item.source_id, item.tokens)))
    for maximum_repeats in range(1, 9):
        candidates = tuple(unit for repeat in range(maximum_repeats) for unit in ordered)
        reachable: dict[int, tuple[int, int] | None] = {0: None}
        for index, unit in enumerate(candidates):
            for total in sorted(tuple(reachable), reverse=True):
                nxt = total + unit.tokens
                if nxt <= budget and nxt not in reachable:
                    reachable[nxt] = (total, index)
        if budget not in reachable:
            continue
        selected: list[ScheduleUnit] = []
        cursor = budget
        while cursor:
            predecessor = reachable[cursor]
            if predecessor is None:
                raise RuntimeError("invalid exact-fill predecessor")
            cursor, index = predecessor
            selected.append(candidates[index])
        return tuple(reversed(selected))
    raise ValueError(f"cannot fill exact task-token residual {budget}")


def _flatten_units(units: list[ScheduleUnit]) -> list[VariantOccurrence]:
    ranked = sorted(
        enumerate(units),
        key=lambda pair: hashlib.sha256(
            f"foundry-cycle1-corpus:{pair[1].source_id}:{pair[0]}".encode()
        ).hexdigest(),
    )
    return [occurrence for _, unit in ranked for occurrence in unit.occurrences]


def _task_bins(
    occurrences: list[VariantOccurrence],
    replay_by_step: list[list[dict[str, Any]]],
) -> list[list[VariantOccurrence]]:
    bins: list[list[VariantOccurrence]] = [[] for _ in replay_by_step]
    totals = [sum(int(item["tokens"]) for item in replay) for replay in replay_by_step]
    ordered = sorted(
        enumerate(occurrences),
        key=lambda pair: (
            -pair[1].tokens,
            hashlib.sha256(
                (f"foundry-cycle1-bin:{pair[1].source_id}:{pair[1].variant}:{pair[0]}").encode()
            ).hexdigest(),
        ),
    )
    for _, occurrence in ordered:
        index = min(range(len(bins)), key=lambda item: (totals[item], item))
        bins[index].append(occurrence)
        totals[index] += occurrence.tokens
    for values in bins:
        values.sort(key=lambda item: (item.source_id, item.variant, item.tokens))
    return bins


def build_continuation_corpus(
    *,
    targeted_training_path: Path,
    selected_trace_path: Path,
    source_schedule_path: Path,
    expected_source_schedule_sha256: str,
    replay_path: Path,
    expected_replay_file_sha256: str,
    tokenizer: Any,
    output_directory: Path,
) -> dict[str, Any]:
    """Freeze the exact 24k task plus 8k replay Cycle 1 schedule."""

    if output_directory.exists():
        raise FileExistsError("Cycle 1 corpus output must be fresh")
    if file_sha256(replay_path) != expected_replay_file_sha256:
        raise ValueError("replay source file identity differs")
    source_schedule = cast(
        list[dict[str, Any]],
        json.loads(source_schedule_path.read_text(encoding="utf-8")),
    )
    if canonical_sha256(source_schedule) != expected_source_schedule_sha256:
        raise ValueError("source L3 targeted schedule differs")
    prefix = source_schedule[:32]
    if len(prefix) != 32 or sum(int(step["loss_bearing_tokens"]) for step in prefix) != 32_000:
        raise ValueError("source 32-step schedule prefix differs")
    replay_by_step = [
        [
            dict(item)
            for item in cast(list[dict[str, Any]], step["occurrences"])
            if item["kind"] == "replay"
        ]
        for step in prefix
    ]
    replay_tokens = sum(int(item["tokens"]) for values in replay_by_step for item in values)
    if replay_tokens != 8_000:
        raise ValueError("source replay projection is not exactly 8,000 tokens")

    originals = _jsonl(targeted_training_path)
    if len(originals) != 180:
        raise ValueError("Cycle 1 requires exactly 180 targeted training records")
    original_by_id = {str(item["source_id"]): item for item in originals}
    if len(original_by_id) != 180:
        raise ValueError("targeted training source IDs are not unique")
    trace_rows = _jsonl(selected_trace_path)
    trace_by_id = {str(item["source_id"]): item for item in trace_rows}
    if len(trace_by_id) != len(trace_rows):
        raise ValueError("selected trace source IDs are not unique")

    variant_records: dict[tuple[str, str], dict[str, Any]] = {}
    units: list[ScheduleUnit] = []
    variant_tokens: dict[tuple[str, str], int] = {}
    for source_id, original in sorted(original_by_id.items()):
        original_row = dict(original)
        original_row["cycle_variant"] = "original"
        original_tokens = _assistant_tokens(original_row, tokenizer)
        if original_tokens != int(original["assistant_tokens_including_eos"]):
            raise ValueError("original targeted assistant-token identity differs")
        original_row["cycle_assistant_tokens_including_eos"] = original_tokens
        variant_records[(source_id, "original")] = original_row
        variant_tokens[(source_id, "original")] = original_tokens
        if source_id not in trace_by_id:
            units.append(
                ScheduleUnit(
                    source_id,
                    (VariantOccurrence(source_id, "original", original_tokens),),
                )
            )
            continue
        trace = trace_by_id[source_id]
        trace_text = str(trace["completion"])
        trace_row = {
            **original,
            "assistant_completion": trace_text,
            "assistant_completion_sha256": hashlib.sha256(trace_text.encode()).hexdigest(),
            "cycle_variant": "selected_trace",
            "selected_completion_index": int(trace["completion_index"]),
        }
        trace_tokens = _assistant_tokens(trace_row, tokenizer)
        if not 1 <= trace_tokens <= 256:
            raise ValueError("selected trace training-token count is outside the frozen bound")
        trace_row["cycle_assistant_tokens_including_eos"] = trace_tokens
        variant_records[(source_id, "selected_trace")] = trace_row
        variant_tokens[(source_id, "selected_trace")] = trace_tokens
        units.append(
            ScheduleUnit(
                source_id,
                (
                    VariantOccurrence(source_id, "original", original_tokens),
                    VariantOccurrence(source_id, "selected_trace", trace_tokens),
                ),
            )
        )

    baseline_tokens = sum(unit.tokens for unit in units)
    if baseline_tokens > 24_000:
        raise ValueError("one deterministic coverage unit per prompt exceeds the task budget")
    added = _bounded_exact_fill(tuple(units), 24_000 - baseline_tokens)
    all_units = [*units, *added]
    task_occurrences = _flatten_units(all_units)
    if sum(item.tokens for item in task_occurrences) != 24_000:
        raise RuntimeError("task schedule is not exactly 24,000 tokens")
    task_bins = _task_bins(task_occurrences, replay_by_step)

    counters: Counter[tuple[str, str]] = Counter()
    schedule: list[dict[str, Any]] = []
    for index, (task_values, replay_values) in enumerate(
        zip(task_bins, replay_by_step, strict=True),
        start=1,
    ):
        task_rows: list[dict[str, Any]] = []
        for item in task_values:
            key = (item.source_id, item.variant)
            counters[key] += 1
            task_rows.append(
                {
                    "kind": "vetted",
                    "record_id": item.source_id,
                    "variant": item.variant,
                    "occurrence_index": counters[key],
                    "tokens": item.tokens,
                }
            )
        occurrences = [*task_rows, *replay_values]
        schedule.append(
            {
                "step": index,
                "loss_bearing_tokens": sum(int(item["tokens"]) for item in occurrences),
                "occurrences": occurrences,
            }
        )
    if sum(int(item["loss_bearing_tokens"]) for item in schedule) != 32_000:
        raise RuntimeError("complete Cycle 1 schedule is not exactly 32,000 tokens")

    counts: Counter[tuple[str, str]] = Counter(
        (item.source_id, item.variant) for item in task_occurrences
    )
    mixture_checks: dict[str, bool] = {}
    for source_id in sorted(original_by_id):
        original_count = counts[(source_id, "original")]
        selected_count = counts[(source_id, "selected_trace")]
        mixture_checks[source_id] = (
            original_count == selected_count and original_count > 0
            if source_id in trace_by_id
            else original_count > 0 and selected_count == 0
        )
    if not all(mixture_checks.values()):
        raise RuntimeError("per-prompt original/selected mixture differs")

    output_directory.mkdir(parents=True, exist_ok=False)
    corpus_path = output_directory / "task_corpus.jsonl"
    selected_keys = {(item.source_id, item.variant) for item in task_occurrences}
    corpus_rows = [variant_records[key] for key in sorted(selected_keys)]
    _write_jsonl(corpus_path, corpus_rows)
    schedule_path = output_directory / "schedule.json"
    _write_json(schedule_path, schedule)
    content_free_records = [
        {
            "source_id": source_id,
            "variant": variant,
            "assistant_tokens": variant_tokens[(source_id, variant)],
            "completion_sha256": str(
                variant_records[(source_id, variant)]["assistant_completion_sha256"]
            ),
            "family": str(variant_records[(source_id, variant)]["family"]),
        }
        for source_id, variant in sorted(selected_keys)
    ]
    projection = [
        [
            {
                "kind": item["kind"],
                "record_id": item["record_id"],
                "occurrence_index": item["occurrence_index"],
                "tokens": item["tokens"],
            }
            for item in values
        ]
        for values in replay_by_step
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "corpus_id": "foundry-cycle1-32k-verifier-filtered-corpus-v1",
        "task_tokens": 24_000,
        "replay_tokens": 8_000,
        "total_tokens": 32_000,
        "optimizer_steps": 32,
        "selected_trace_prompts": len(trace_by_id),
        "fallback_prompts": len(original_by_id) - len(trace_by_id),
        "task_occurrences": len(task_occurrences),
        "replay_occurrences": sum(len(value) for value in replay_by_step),
        "variant_records": content_free_records,
        "variant_record_manifest_sha256": canonical_sha256(content_free_records),
        "task_corpus_file_sha256": file_sha256(corpus_path),
        "schedule_sha256": canonical_sha256(schedule),
        "schedule_file_sha256": file_sha256(schedule_path),
        "source_schedule_prefix_sha256": canonical_sha256(prefix),
        "replay_source_file_sha256": file_sha256(replay_path),
        "replay_projection_sha256": canonical_sha256(projection),
        "mixture_counts": {
            source_id: {
                "original": counts[(source_id, "original")],
                "selected_trace": counts[(source_id, "selected_trace")],
            }
            for source_id in sorted(original_by_id)
        },
        "all_prompt_mixtures_exact": all(mixture_checks.values()),
        "no_holdout_or_gsm_records": True,
        "deterministic_reconstruction": True,
    }
    result["corpus_sha256"] = canonical_sha256(result)
    _write_json(output_directory / "manifest.json", result)
    return result


def build_smoke_corpus(
    *,
    records: list[dict[str, Any]],
    selected_traces: dict[str, dict[str, Any]],
    tokenizer: Any,
    output_directory: Path,
) -> dict[str, Any]:
    """Create the exact four-record, two-step compatibility corpus."""

    if len(records) != 4 or output_directory.exists():
        raise ValueError("compatibility corpus requires four fresh records")
    rows: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["source_id"])):
        source_id = str(record["source_id"])
        if source_id in selected_traces:
            completion = str(selected_traces[source_id]["completion"])
            variant = "selected_trace"
        else:
            completion = str(record["assistant_completion"])
            variant = "original"
        row = {
            **record,
            "assistant_completion": completion,
            "assistant_completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
            "cycle_variant": variant,
        }
        tokens = _assistant_tokens(row, tokenizer)
        row["cycle_assistant_tokens_including_eos"] = tokens
        rows.append(row)
        occurrences.append(
            {
                "kind": "vetted",
                "record_id": source_id,
                "variant": variant,
                "tokens": tokens,
            }
        )
    schedule = []
    for step in (1, 2):
        values = [{**item, "occurrence_index": step} for item in occurrences]
        schedule.append(
            {
                "step": step,
                "loss_bearing_tokens": sum(int(item["tokens"]) for item in values),
                "occurrences": values,
            }
        )
    output_directory.mkdir(parents=True, exist_ok=False)
    corpus_path = output_directory / "task_corpus.jsonl"
    schedule_path = output_directory / "schedule.json"
    _write_jsonl(corpus_path, rows)
    _write_json(schedule_path, schedule)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "corpus_id": "foundry-cycle1-compatibility-corpus-v1",
        "unique_records": 4,
        "optimizer_steps": 2,
        "corpus_file_sha256": file_sha256(corpus_path),
        "schedule_sha256": canonical_sha256(schedule),
        "schedule_file_sha256": file_sha256(schedule_path),
        "records": [
            {
                "source_id": str(row["source_id"]),
                "variant": str(row["cycle_variant"]),
                "completion_sha256": str(row["assistant_completion_sha256"]),
                "assistant_tokens": int(row["cycle_assistant_tokens_including_eos"]),
            }
            for row in rows
        ],
    }
    manifest["corpus_sha256"] = canonical_sha256(manifest)
    _write_json(output_directory / "manifest.json", manifest)
    return manifest


def load_cycle_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(item["source_id"]), str(item["cycle_variant"])): item for item in _jsonl(path)}


def occurrence_dict(value: VariantOccurrence) -> dict[str, Any]:
    return asdict(value)
