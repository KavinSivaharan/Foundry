"""Deterministic whole-example schedules for vetted-corpus QLoRA."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from foundry.phase2.launch_contract import validate_preimport
from foundry.training.config import canonical_sha256

SEED = 20260720
STEPS = 64
CHECKPOINTS = (16, 32, 64)
V1_VETTED_TOKENS = 48_000
V1_REPLAY_TOKENS = 16_000
V2_VETTED_TOKENS = 38_400
V2_REPLAY_TOKENS = 25_600
SEGMENTS = 4


@dataclass(frozen=True)
class Entry:
    record_id: str
    tokens: int
    kind: str
    family: str
    difficulty: str


@dataclass(frozen=True)
class Occurrence:
    record_id: str
    occurrence_index: int
    tokens: int
    kind: str


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle]


def _replay_entries(path: Path, tokenizer: Any) -> tuple[Entry, ...]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    items = cast(list[dict[str, Any]], payload["items"])
    result: list[Entry] = []
    for item in items:
        messages = [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["assistant_response"]},
        ]
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        eos = max(index for index, value in enumerate(full_ids) if value == tokenizer.eos_token_id)
        tokens = eos - len(prefix_ids) + 1
        if tokens <= 0 or eos != len(full_ids) - 2:
            raise ValueError("replay assistant-only EOS boundary differs")
        result.append(
            Entry(str(item["id"]), tokens, "replay", str(item["section"]), str(item["skill"]))
        )
    if len(result) != 83 or len({item.record_id for item in result}) != 83:
        raise ValueError("replay corpus must contain 83 unique records")
    return tuple(result)


def _vetted_entries(path: Path) -> tuple[Entry, ...]:
    records = _jsonl(path)
    entries = tuple(
        Entry(
            str(item["source_id"]),
            int(item["assistant_tokens_including_eos"]),
            "vetted",
            str(item["family"]),
            str(item["difficulty"]),
        )
        for item in records
    )
    if len(entries) != 180 or len({item.record_id for item in entries}) != 180:
        raise ValueError("vetted training split must contain 180 unique records")
    return entries


def _exact_fill(entries: tuple[Entry, ...], budget: int) -> tuple[Entry, ...]:
    """Fill an exact residual with deterministic unbounded dynamic programming."""

    reachable: dict[int, tuple[int, int] | None] = {0: None}
    ordered = sorted(entries, key=lambda item: (item.tokens, item.record_id))
    for total in range(budget + 1):
        if total not in reachable:
            continue
        for index, entry in enumerate(ordered):
            nxt = total + entry.tokens
            if nxt <= budget and nxt not in reachable:
                reachable[nxt] = (total, index)
    if budget not in reachable:
        raise ValueError(f"cannot fill exact assistant-token residual {budget}")
    selected: list[Entry] = []
    cursor = budget
    while cursor:
        previous = reachable[cursor]
        if previous is None:
            raise RuntimeError("invalid exact-fill predecessor")
        cursor, index = previous
        selected.append(ordered[index])
    return tuple(reversed(selected))


def _bounded_fill(
    entries: tuple[Entry, ...], budget: int, *, maximum_repeats: int
) -> tuple[Entry, ...]:
    """Fill an exact residual while bounding each record's added repeats."""

    candidates = tuple(
        item
        for item in sorted(entries, key=lambda value: value.record_id)
        for _ in range(maximum_repeats)
    )
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for index, entry in enumerate(candidates):
        for total in sorted(tuple(reachable), reverse=True):
            nxt = total + entry.tokens
            if nxt <= budget and nxt not in reachable:
                reachable[nxt] = (total, index)
    if budget not in reachable:
        raise ValueError(f"cannot bounded-fill assistant-token residual {budget}")
    selected: list[Entry] = []
    cursor = budget
    while cursor:
        previous = reachable[cursor]
        if previous is None:
            raise RuntimeError("invalid bounded-fill predecessor")
        cursor, index = previous
        selected.append(candidates[index])
    return tuple(reversed(selected))


def _segment_occurrences(
    entries: tuple[Entry, ...], budget: int, segment: int, counters: Counter[str]
) -> tuple[Occurrence, ...]:
    # Each vetted row is assigned to exactly one baseline segment. Replay rows use
    # the same rule, giving all 83 demonstrated behaviors baseline coverage.
    baseline = tuple(
        item
        for index, item in enumerate(sorted(entries, key=lambda value: value.record_id))
        if index % SEGMENTS == segment
    )
    selected = list(baseline)
    remaining = budget - sum(item.tokens for item in selected)
    if remaining < 0:
        raise ValueError("segment baseline exceeds its token budget")

    # Add balanced complete cycles while leaving a small exactly-solvable tail.
    cycle = tuple(sorted(entries, key=lambda item: (counters[item.record_id], item.record_id)))
    cycle_tokens = sum(item.tokens for item in cycle)
    while remaining >= cycle_tokens:
        selected.extend(cycle)
        remaining -= cycle_tokens
    try:
        selected.extend(_bounded_fill(entries, remaining, maximum_repeats=1))
    except ValueError:
        if not all(item in selected for item in cycle):
            raise
        for item in cycle:
            selected.remove(item)
        remaining += cycle_tokens
        selected.extend(_bounded_fill(entries, remaining, maximum_repeats=2))

    result: list[Occurrence] = []
    for entry in selected:
        counters[entry.record_id] += 1
        result.append(
            Occurrence(entry.record_id, counters[entry.record_id], entry.tokens, entry.kind)
        )
    if sum(item.tokens for item in result) != budget:
        raise RuntimeError("segment token budget differs")
    return tuple(result)


def _bins(occurrences: tuple[Occurrence, ...]) -> tuple[tuple[Occurrence, ...], ...]:
    bins: list[list[Occurrence]] = [[] for _ in range(16)]
    totals = [0] * 16
    for item in sorted(
        occurrences, key=lambda value: (-value.tokens, value.record_id, value.occurrence_index)
    ):
        target = min(range(16), key=lambda index: (totals[index], index))
        bins[target].append(item)
        totals[target] += item.tokens
    normalized = [
        tuple(sorted(values, key=lambda item: (item.kind, item.record_id, item.occurrence_index)))
        for values in bins
    ]
    return tuple(
        normalized[index]
        for index in sorted(
            range(16),
            key=lambda index: (
                totals[index],
                tuple(
                    (item.kind, item.record_id, item.occurrence_index) for item in normalized[index]
                ),
            ),
        )
    )


def _schedule(
    vetted: tuple[Entry, ...],
    replay_segments: tuple[tuple[Occurrence, ...], ...],
    vetted_tokens: int,
) -> tuple[dict[str, Any], ...]:
    counters: Counter[str] = Counter()
    result: list[dict[str, Any]] = []
    for segment in range(SEGMENTS):
        vetted_bins = _bins(
            _segment_occurrences(vetted, vetted_tokens // SEGMENTS, segment, counters)
        )
        replay_bins = _bins(replay_segments[segment])
        for offset, (vetted_bin, replay_bin) in enumerate(
            zip(vetted_bins, replay_bins, strict=True)
        ):
            occurrences = (*vetted_bin, *replay_bin)
            result.append(
                {
                    "step": segment * 16 + offset + 1,
                    "loss_bearing_tokens": sum(item.tokens for item in occurrences),
                    "occurrences": [asdict(item) for item in occurrences],
                }
            )
    if len(result) != STEPS:
        raise RuntimeError("schedule must contain 64 steps")
    return tuple(result)


def _hash(value: object) -> str:
    return canonical_sha256(value)


def _prefix_hash(schedule: tuple[dict[str, Any], ...], step: int) -> str:
    return _hash(schedule[:step])


def _summary(
    generic: tuple[dict[str, Any], ...],
    targeted: tuple[dict[str, Any], ...],
    replay_segments: tuple[tuple[Occurrence, ...], ...],
    generic_entries: tuple[Entry, ...],
    targeted_entries: tuple[Entry, ...],
    *,
    variant: str,
    vetted_tokens: int,
    replay_tokens: int,
) -> dict[str, Any]:
    def arm(value: tuple[dict[str, Any], ...], entries: tuple[Entry, ...]) -> dict[str, Any]:
        occurrences = [
            item for step in value for item in cast(list[dict[str, Any]], step["occurrences"])
        ]
        vetted = [item for item in occurrences if item["kind"] == "vetted"]
        replay = [item for item in occurrences if item["kind"] == "replay"]
        counts = Counter(str(item["record_id"]) for item in vetted)
        metadata = {item.record_id: item for item in entries}
        by_family: Counter[str] = Counter()
        by_difficulty: Counter[str] = Counter()
        for record_id, count in counts.items():
            by_family[metadata[record_id].family] += count
            by_difficulty[metadata[record_id].difficulty] += count
        return {
            "schedule_sha256": _hash(value),
            "checkpoint_prefix_sha256": {
                str(step): _prefix_hash(value, step) for step in CHECKPOINTS
            },
            "vetted_tokens": sum(int(item["tokens"]) for item in vetted),
            "replay_tokens": sum(int(item["tokens"]) for item in replay),
            "total_tokens": sum(int(step["loss_bearing_tokens"]) for step in value),
            "vetted_occurrences": len(vetted),
            "replay_occurrences": len(replay),
            "unique_vetted_records": len(counts),
            "minimum_vetted_occurrences": min(counts.values()),
            "maximum_vetted_occurrences": max(counts.values()),
            "occurrences_by_family": dict(sorted(by_family.items())),
            "occurrences_by_difficulty": dict(sorted(by_difficulty.items())),
            "prefix_tokens": {
                str(step): sum(int(item["loss_bearing_tokens"]) for item in value[:step])
                for step in CHECKPOINTS
            },
            "smoke_prefix_tokens": sum(int(item["loss_bearing_tokens"]) for item in value[:4]),
        }

    generic_summary = arm(generic, generic_entries)
    targeted_summary = arm(targeted, targeted_entries)
    replay_payload = [[asdict(item) for item in segment] for segment in replay_segments]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "schedule_id": f"foundry-vetted-corpus-{variant}",
        "seed": SEED,
        "optimizer_steps": STEPS,
        "budgets": {"vetted": vetted_tokens, "replay": replay_tokens, "total": 64_000},
        "recipe_sha256": _hash(
            {
                "schedule_id": f"foundry-vetted-corpus-{variant}",
                "seed": SEED,
                "optimizer_steps": STEPS,
                "checkpoints": CHECKPOINTS,
                "vetted_tokens": vetted_tokens,
                "replay_tokens": replay_tokens,
                "whole_examples": True,
            }
        ),
        "replay_schedule_sha256": _hash(replay_payload),
        "generic": generic_summary,
        "targeted": targeted_summary,
    }
    for step in CHECKPOINTS:
        left = int(cast(dict[str, int], generic_summary["prefix_tokens"])[str(step)])
        right = int(cast(dict[str, int], targeted_summary["prefix_tokens"])[str(step)])
        if abs(left - right) / max(left, right) > 0.005:
            raise ValueError(f"checkpoint {step} token parity exceeds 0.5%")
    smoke_left = int(generic_summary["smoke_prefix_tokens"])
    smoke_right = int(targeted_summary["smoke_prefix_tokens"])
    if abs(smoke_left - smoke_right) / max(smoke_left, smoke_right) > 0.005:
        raise ValueError(
            f"four-step smoke token parity exceeds 0.5%: {smoke_left} versus {smoke_right}"
        )
    generic_replay = [
        item
        for step in generic
        for item in cast(list[dict[str, Any]], step["occurrences"])
        if item["kind"] == "replay"
    ]
    targeted_replay = [
        item
        for step in targeted
        for item in cast(list[dict[str, Any]], step["occurrences"])
        if item["kind"] == "replay"
    ]
    if generic_replay != targeted_replay:
        raise ValueError("generic and targeted replay ordering differs")
    summary["replay_identical_between_arms"] = True
    summary["summary_sha256"] = _hash(summary)
    return summary


def run(
    *,
    model_path: Path,
    generic_path: Path,
    targeted_path: Path,
    replay_path: Path,
    raw_directory: Path,
    summary_path: Path,
    variant: str = "replay25-v1",
) -> dict[str, Any]:
    """Tokenize replay records and freeze deterministic V1 or V2 schedules."""

    validate_preimport()
    transformers = importlib.import_module("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if variant == "replay25-v1":
        vetted_tokens, replay_tokens = V1_VETTED_TOKENS, V1_REPLAY_TOKENS
    elif variant == "replay40-v2":
        vetted_tokens, replay_tokens = V2_VETTED_TOKENS, V2_REPLAY_TOKENS
    else:
        raise ValueError("unknown vetted-corpus schedule variant")
    replay = _replay_entries(replay_path, tokenizer)
    replay_counters: Counter[str] = Counter()
    replay_segments = tuple(
        _segment_occurrences(replay, replay_tokens // SEGMENTS, segment, replay_counters)
        for segment in range(SEGMENTS)
    )
    generic_entries = _vetted_entries(generic_path)
    targeted_entries = _vetted_entries(targeted_path)
    generic = _schedule(generic_entries, replay_segments, vetted_tokens)
    targeted = _schedule(targeted_entries, replay_segments, vetted_tokens)
    summary = _summary(
        generic,
        targeted,
        replay_segments,
        generic_entries,
        targeted_entries,
        variant=variant,
        vetted_tokens=vetted_tokens,
        replay_tokens=replay_tokens,
    )
    raw_directory.mkdir(parents=True, exist_ok=False)
    for name, value in (("generic", generic), ("targeted", targeted)):
        (raw_directory / f"{name}_schedule.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--generic-path", type=Path, required=True)
    parser.add_argument("--targeted-path", type=Path, required=True)
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--variant", choices=("replay25-v1", "replay40-v2"), default="replay25-v1")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
