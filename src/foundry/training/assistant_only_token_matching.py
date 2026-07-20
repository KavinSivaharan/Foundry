"""Assistant-only token census and token-matched Method B schedules."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from foundry.training.assistant_only import assistant_only_tokenize
from foundry.training.config import (
    assistant_only_v3_format_contract_sha256,
    canonical_sha256,
    load_qlora_recipe,
)
from foundry.training.qlora import file_sha256
from foundry.training.token_matching import (
    ARM_ORDER,
    METHOD_B_PARITY_LIMIT,
    METHOD_B_SELECTION_SEED,
    OPTIMIZER_STEPS,
    ScheduledOccurrence,
    ScheduledStep,
    TokenCensusEntry,
    _balanced_variable_steps,
    _distribution,
    _grouped_distributions,
    _stratum_repeat_balance,
    load_census,
)

ASSISTANT_ONLY_NOMINAL_TOKENS = 90_000
ASSISTANT_ONLY_MAX_OCCURRENCES = 1_800
RETENTION_SMOKE_STEPS = 32


def _load_records(path: Path, arm: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("assistant-only source row must be an object")
            record = cast(dict[str, Any], value)
            if (
                record.get("group") != arm
                or record.get("future_split") != "training"
                or record.get("final_decision") != "accepted"
            ):
                raise ValueError("assistant-only source row violates the frozen split")
            records.append(record)
    if len(records) != 450:
        raise ValueError(f"{arm} requires exactly 450 training rows")
    return records


def _load_tokenizer(model_path: Path, base_recipe_path: Path) -> tuple[Any, Any]:
    recipe = load_qlora_recipe(base_recipe_path)
    if file_sha256(model_path / "tokenizer.json") != recipe.tokenizer_sha256:
        raise ValueError("local tokenizer differs from frozen recipe")
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    chat_hash = hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()
    if chat_hash != recipe.chat_template_sha256:
        raise ValueError("Qwen chat template differs from frozen recipe")
    return tokenizer, recipe


def _entry(record: dict[str, Any], tokenizer: Any, max_length: int) -> TokenCensusEntry:
    _, evidence = assistant_only_tokenize(record, tokenizer, max_length=max_length)
    return TokenCensusEntry(
        synthetic_id=str(record["synthetic_id"]),
        dataset_arm=str(record["group"]),
        family=str(record["family"]),
        submode=str(record["mode"]),
        difficulty=str(record["difficulty"]),
        output_contract_enabled=bool(record["output_contract_enabled"]),
        formatted_input_tokens=evidence.formatted_tokens,
        loss_bearing_tokens=evidence.assistant_loss_tokens,
        truncated_tokens=evidence.truncated_tokens,
        labels_entirely_masked=evidence.assistant_loss_tokens == 0,
        formatted_example_sha256=evidence.formatted_text_sha256,
    )


def run_census(
    *,
    base_recipe_path: Path,
    model_path: Path,
    generic_path: Path,
    targeted_path: Path,
    raw_directory: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Build the deterministic assistant-only census for both training arms."""

    tokenizer, recipe = _load_tokenizer(model_path, base_recipe_path)
    sources = {"generic_control": generic_path, "targeted": targeted_path}
    by_arm: dict[str, tuple[TokenCensusEntry, ...]] = {}
    raw_directory.mkdir(parents=True, exist_ok=True)
    arm_summaries: dict[str, Any] = {}
    for arm in ARM_ORDER:
        records = _load_records(sources[arm], arm)
        entries = tuple(_entry(record, tokenizer, recipe.max_sequence_length) for record in records)
        replay = tuple(_entry(record, tokenizer, recipe.max_sequence_length) for record in records)
        if entries != replay or any(entry.labels_entirely_masked for entry in entries):
            raise ValueError("assistant-only census reconstruction or labels differ")
        by_arm[arm] = entries
        path = raw_directory / f"{arm}_census.jsonl"
        path.write_text(
            "".join(json.dumps(asdict(entry), sort_keys=True) + "\n" for entry in entries),
            encoding="utf-8",
        )
        arm_summaries[arm] = {
            "census_sha256": canonical_sha256([asdict(entry) for entry in entries]),
            "census_file_sha256": file_sha256(path),
            "loss_bearing_tokens": _distribution(entries),
            "by_family": _grouped_distributions(entries, "family"),
            "by_difficulty": _grouped_distributions(entries, "difficulty"),
            "by_output_contract": _grouped_distributions(entries, "output_contract_enabled"),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "census_id": "foundry-assistant-only-token-census-v3",
        "base_revision": recipe.base_revision,
        "chat_template_sha256": recipe.chat_template_sha256,
        "assistant_only_format_sha256": assistant_only_v3_format_contract_sha256(),
        "label_construction": "assistant_completion_content_and_final_eos_only",
        "max_sequence_length": recipe.max_sequence_length,
        "arms": arm_summaries,
        "total_training_examples": 900,
        "all_examples_have_loss_bearing_tokens": True,
        "deterministic_reconstruction_match": True,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _stable_occurrences(
    entries: tuple[TokenCensusEntry, ...], occurrence_count: int, arm: str
) -> tuple[ScheduledOccurrence, ...]:
    full_cycles, remainder = divmod(occurrence_count, len(entries))
    grouped: dict[tuple[str, str, bool], list[TokenCensusEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.stratum].append(entry)
    exact = {key: remainder * len(values) / len(entries) for key, values in grouped.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    left = remainder - sum(quotas.values())
    order = sorted(grouped, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:left]:
        quotas[key] += 1
    extras: set[str] = set()
    for stratum in sorted(grouped):
        ordered = sorted(
            grouped[stratum],
            key=lambda entry: (
                hashlib.sha256(
                    f"{METHOD_B_SELECTION_SEED}:{arm}:{entry.synthetic_id}:v3-extra".encode()
                ).hexdigest(),
                entry.synthetic_id,
            ),
        )
        extras.update(entry.synthetic_id for entry in ordered[: quotas[stratum]])
    occurrences: list[ScheduledOccurrence] = []
    for entry in sorted(entries, key=lambda item: item.synthetic_id):
        repeats = full_cycles + int(entry.synthetic_id in extras)
        for index in range(1, repeats + 1):
            occurrences.append(
                ScheduledOccurrence(entry.synthetic_id, index, entry.loss_bearing_tokens)
            )
    return tuple(occurrences)


def _candidate_occurrences(
    entries: tuple[TokenCensusEntry, ...], arm: str
) -> list[tuple[int, int, tuple[ScheduledOccurrence, ...]]]:
    candidates: list[tuple[int, int, tuple[ScheduledOccurrence, ...]]] = []
    for count in range(450, ASSISTANT_ONLY_MAX_OCCURRENCES + 1):
        occurrences = _stable_occurrences(entries, count, arm)
        total = sum(item.loss_bearing_tokens for item in occurrences)
        if abs(total - ASSISTANT_ONLY_NOMINAL_TOKENS) / ASSISTANT_ONLY_NOMINAL_TOKENS <= 0.005:
            candidates.append((total, count, occurrences))
    if not candidates:
        raise ValueError(f"{arm} has no assistant-only occurrence schedule near nominal tokens")
    return candidates


def _select_pair(
    entries: dict[str, tuple[TokenCensusEntry, ...]],
) -> dict[str, tuple[ScheduledOccurrence, ...]]:
    generic = _candidate_occurrences(entries["generic_control"], "generic_control")
    targeted = _candidate_occurrences(entries["targeted"], "targeted")
    feasible = [
        (generic_item, targeted_item)
        for generic_item in generic
        for targeted_item in targeted
        if abs(generic_item[0] - targeted_item[0]) / max(generic_item[0], targeted_item[0])
        <= METHOD_B_PARITY_LIMIT
    ]
    if not feasible:
        raise ValueError("assistant-only arms have no token-parity-feasible pair")
    selected = min(
        feasible,
        key=lambda pair: (
            abs(pair[0][0] - ASSISTANT_ONLY_NOMINAL_TOKENS)
            + abs(pair[1][0] - ASSISTANT_ONLY_NOMINAL_TOKENS),
            abs(pair[0][0] - pair[1][0]),
            pair[0][1],
            pair[1][1],
        ),
    )
    return {"generic_control": selected[0][2], "targeted": selected[1][2]}


def _reorder_smoke_prefix(
    schedules: dict[str, tuple[ScheduledStep, ...]],
) -> tuple[dict[str, tuple[ScheduledStep, ...]], dict[str, int]]:
    window = RETENTION_SMOKE_STEPS
    totals = {
        arm: [
            sum(step.loss_bearing_tokens for step in schedule[start : start + window])
            for start in range(OPTIMIZER_STEPS - window + 1)
        ]
        for arm, schedule in schedules.items()
    }
    starts = min(
        (
            {"generic_control": generic, "targeted": targeted}
            for generic in range(len(totals["generic_control"]))
            for targeted in range(len(totals["targeted"]))
        ),
        key=lambda pair: (
            abs(
                totals["generic_control"][pair["generic_control"]]
                - totals["targeted"][pair["targeted"]]
            ),
            pair["generic_control"],
            pair["targeted"],
        ),
    )
    result: dict[str, tuple[ScheduledStep, ...]] = {}
    for arm in ARM_ORDER:
        start = starts[arm]
        schedule = schedules[arm]
        ordered = schedule[start : start + window] + schedule[:start] + schedule[start + window :]
        result[arm] = tuple(
            ScheduledStep(index + 1, step.occurrences, step.loss_bearing_tokens)
            for index, step in enumerate(ordered)
        )
    return result, starts


def _build_schedules(
    entries: dict[str, tuple[TokenCensusEntry, ...]],
) -> tuple[dict[str, tuple[ScheduledStep, ...]], dict[str, int]]:
    occurrences = _select_pair(entries)
    schedules = {arm: _balanced_variable_steps(occurrences[arm]) for arm in ARM_ORDER}
    return _reorder_smoke_prefix(schedules)


def run_method_b(
    *,
    generic_census_path: Path,
    targeted_census_path: Path,
    raw_directory: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Freeze assistant-only whole-example schedules and 32-step smoke parity."""

    entries = {
        "generic_control": load_census(generic_census_path),
        "targeted": load_census(targeted_census_path),
    }
    schedules, starts = _build_schedules(entries)
    raw_directory.mkdir(parents=True, exist_ok=True)
    arms: dict[str, Any] = {}
    totals: dict[str, int] = {}
    smoke_totals: dict[str, int] = {}
    for arm in ARM_ORDER:
        schedule = schedules[arm]
        payload = [asdict(step) for step in schedule]
        path = raw_directory / f"method_b_{arm}_schedule.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        counts = Counter(
            occurrence.synthetic_id for step in schedule for occurrence in step.occurrences
        )
        balanced, balance = _stratum_repeat_balance(entries[arm], schedule)
        totals[arm] = sum(step.loss_bearing_tokens for step in schedule)
        smoke_totals[arm] = sum(
            step.loss_bearing_tokens for step in schedule[:RETENTION_SMOKE_STEPS]
        )
        arms[arm] = {
            "schedule_sha256": canonical_sha256(payload),
            "schedule_file_sha256": file_sha256(path),
            "occurrences": sum(len(step.occurrences) for step in schedule),
            "optimizer_steps": len(schedule),
            "unique_examples": len(counts),
            "minimum_repeat_count": min(counts.values()),
            "maximum_repeat_count": max(counts.values()),
            "loss_bearing_tokens": totals[arm],
            "minimum_step_tokens": min(step.loss_bearing_tokens for step in schedule),
            "maximum_step_tokens": max(step.loss_bearing_tokens for step in schedule),
            "mean_step_tokens": statistics.fmean(step.loss_bearing_tokens for step in schedule),
            "first_32_step_tokens": smoke_totals[arm],
            "smoke_prefix_source_start": starts[arm] + 1,
            "stratum_repeat_counts_balanced": balanced,
            "stratum_repeat_balance": balance,
        }
    pair_relative = abs(totals["generic_control"] - totals["targeted"]) / max(totals.values())
    smoke_relative = abs(smoke_totals["generic_control"] - smoke_totals["targeted"]) / max(
        smoke_totals.values()
    )
    reconstruction = _build_schedules(entries)[0]
    gate_checks = {
        "optimizer_steps_exact": all(arms[arm]["optimizer_steps"] == 200 for arm in ARM_ORDER),
        "nominal_tokens_within_0_5_percent": all(
            abs(totals[arm] - ASSISTANT_ONLY_NOMINAL_TOKENS) / ASSISTANT_ONLY_NOMINAL_TOKENS
            <= METHOD_B_PARITY_LIMIT
            for arm in ARM_ORDER
        ),
        "full_pairwise_parity_within_0_5_percent": pair_relative <= METHOD_B_PARITY_LIMIT,
        "smoke_pairwise_parity_within_0_5_percent": smoke_relative <= METHOD_B_PARITY_LIMIT,
        "every_example_appears": all(arms[arm]["unique_examples"] == 450 for arm in ARM_ORDER),
        "stratum_repeat_counts_balanced": all(
            arms[arm]["stratum_repeat_counts_balanced"] for arm in ARM_ORDER
        ),
        "deterministic_reconstruction": schedules == reconstruction,
        "whole_examples_only": True,
        "no_data_content_changed": True,
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "method": "assistant_only_whole_example_token_budgeted_gradient_accumulation",
        "nominal_loss_bearing_tokens": ASSISTANT_ONLY_NOMINAL_TOKENS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "retention_smoke_steps": RETENTION_SMOKE_STEPS,
        "selection_seed": METHOD_B_SELECTION_SEED,
        "assistant_only_format_sha256": assistant_only_v3_format_contract_sha256(),
        "arms": arms,
        "pairwise_absolute_token_difference": abs(totals["generic_control"] - totals["targeted"]),
        "pairwise_relative_token_difference": pair_relative,
        "first_32_step_tokens": smoke_totals,
        "first_32_step_relative_difference": smoke_relative,
        "parity_limit": METHOD_B_PARITY_LIMIT,
        "gate_checks": gate_checks,
        "method_b_schedule_gate_passed": all(gate_checks.values()),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run assistant-only census or schedule construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    census = commands.add_parser("census")
    census.add_argument("--base-recipe", type=Path, required=True)
    census.add_argument("--model-path", type=Path, required=True)
    census.add_argument("--generic", type=Path, required=True)
    census.add_argument("--targeted", type=Path, required=True)
    census.add_argument("--raw-directory", type=Path, required=True)
    census.add_argument("--summary", type=Path, required=True)
    method = commands.add_parser("method-b")
    method.add_argument("--generic-census", type=Path, required=True)
    method.add_argument("--targeted-census", type=Path, required=True)
    method.add_argument("--raw-directory", type=Path, required=True)
    method.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "census":
        result = run_census(
            base_recipe_path=args.base_recipe,
            model_path=args.model_path,
            generic_path=args.generic,
            targeted_path=args.targeted,
            raw_directory=args.raw_directory,
            summary_path=args.summary,
        )
    else:
        result = run_method_b(
            generic_census_path=args.generic_census,
            targeted_census_path=args.targeted_census,
            raw_directory=args.raw_directory,
            summary_path=args.summary,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
