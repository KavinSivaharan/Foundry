"""Deterministic token census and token-matched schedule construction."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256, load_qlora_recipe, sft_messages
from foundry.training.qlora import file_sha256

ARM_ORDER = ("generic_control", "targeted")
EXTRA_REPEAT_COUNT = 250
METHOD_A_OCCURRENCES = 1600
OPTIMIZER_STEPS = 200
METHOD_A_EXAMPLES_PER_STEP = 8
METHOD_A_PARITY_LIMIT = 0.005
PREVIOUS_SAFE_PADDED_TOKENS_PER_STEP = 4096
METHOD_B_NOMINAL_TOKENS = 271200
METHOD_B_PARITY_LIMIT = 0.005
METHOD_B_SELECTION_SEED = 20260720


@dataclass(frozen=True)
class TokenCensusEntry:
    """Content-free token measurements for one frozen training example."""

    synthetic_id: str
    dataset_arm: str
    family: str
    submode: str
    difficulty: str
    output_contract_enabled: bool
    formatted_input_tokens: int
    loss_bearing_tokens: int
    truncated_tokens: int
    labels_entirely_masked: bool
    formatted_example_sha256: str

    @property
    def stratum(self) -> tuple[str, str, bool]:
        """Return the frozen extra-repeat stratum."""

        return (self.family, self.difficulty, self.output_contract_enabled)


@dataclass(frozen=True)
class ScheduledOccurrence:
    """One immutable example occurrence in a token-matched schedule."""

    synthetic_id: str
    occurrence_index: int
    loss_bearing_tokens: int


@dataclass(frozen=True)
class ScheduledStep:
    """One fixed optimizer step."""

    step: int
    occurrences: tuple[ScheduledOccurrence, ...]
    loss_bearing_tokens: int


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL record must be an object")
            records.append(cast(dict[str, Any], value))
    return records


def _load_tokenizer(model_path: Path, recipe_path: Path) -> tuple[Any, Any]:
    recipe = load_qlora_recipe(recipe_path)
    if file_sha256(model_path / "tokenizer.json") != recipe.tokenizer_sha256:
        raise ValueError("local tokenizer.json differs from frozen recipe")
    if file_sha256(model_path / "tokenizer_config.json") != recipe.tokenizer_config_sha256:
        raise ValueError("local tokenizer_config.json differs from frozen recipe")
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    chat_hash = hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest()
    if chat_hash != recipe.chat_template_sha256:
        raise ValueError("loaded tokenizer chat template differs from frozen recipe")
    return tokenizer, recipe


def measure_record(
    record: dict[str, Any], *, tokenizer: Any, max_sequence_length: int
) -> TokenCensusEntry:
    """Measure one record through the exact frozen chat and label construction."""

    messages = sft_messages(str(record["rendered_question"]), str(record["training_completion"]))
    text = cast(
        str,
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
    )
    original = cast(
        list[int], tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
        padding="max_length",
    )
    input_ids = cast(list[int], encoded["input_ids"])
    attention_mask = cast(list[int], encoded["attention_mask"])
    labels = [
        token if mask else -100 for token, mask in zip(input_ids, attention_mask, strict=True)
    ]
    loss_tokens = sum(label != -100 for label in labels)
    return TokenCensusEntry(
        synthetic_id=str(record["synthetic_id"]),
        dataset_arm=str(record["group"]),
        family=str(record["family"]),
        submode=str(record["mode"]),
        difficulty=str(record["difficulty"]),
        output_contract_enabled=bool(record["output_contract_enabled"]),
        formatted_input_tokens=len(original),
        loss_bearing_tokens=loss_tokens,
        truncated_tokens=max(0, len(original) - max_sequence_length),
        labels_entirely_masked=loss_tokens == 0,
        formatted_example_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def build_census_entries(
    records: list[dict[str, Any]], *, tokenizer: Any, max_sequence_length: int
) -> tuple[TokenCensusEntry, ...]:
    """Measure and validate one arm in stable source order."""

    entries = tuple(
        measure_record(record, tokenizer=tokenizer, max_sequence_length=max_sequence_length)
        for record in records
    )
    if len(entries) != 450 or len({entry.synthetic_id for entry in entries}) != 450:
        raise ValueError("each training arm must contain 450 unique examples")
    if any(entry.labels_entirely_masked for entry in entries):
        raise ValueError("every training example must contain a loss-bearing label")
    return entries


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _distribution(entries: tuple[TokenCensusEntry, ...]) -> dict[str, object]:
    values = [entry.loss_bearing_tokens for entry in entries]
    return {
        "count": len(values),
        "total": sum(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "percentiles": {str(item): _percentile(values, item) for item in (25, 50, 75, 90, 95, 99)},
        "truncated_examples": sum(entry.truncated_tokens > 0 for entry in entries),
        "truncated_tokens": sum(entry.truncated_tokens for entry in entries),
        "entirely_masked_examples": sum(entry.labels_entirely_masked for entry in entries),
    }


def _grouped_distributions(
    entries: tuple[TokenCensusEntry, ...], key_name: str
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[TokenCensusEntry]] = defaultdict(list)
    for entry in entries:
        value = getattr(entry, key_name)
        grouped[str(value)].append(entry)
    return {key: _distribution(tuple(values)) for key, values in sorted(grouped.items())}


def run_census(
    *,
    recipe_path: Path,
    model_path: Path,
    generic_path: Path,
    targeted_path: Path,
    raw_directory: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Build both deterministic censuses and content-free aggregate evidence."""

    tokenizer, recipe = _load_tokenizer(model_path, recipe_path)
    paths = {"generic_control": generic_path, "targeted": targeted_path}
    entries_by_arm: dict[str, tuple[TokenCensusEntry, ...]] = {}
    raw_directory.mkdir(parents=True, exist_ok=True)
    for arm in ARM_ORDER:
        records = _load_jsonl(paths[arm])
        entries = build_census_entries(
            records, tokenizer=tokenizer, max_sequence_length=recipe.max_sequence_length
        )
        if any(entry.dataset_arm != arm for entry in entries):
            raise ValueError("census arm differs from source record")
        replay = build_census_entries(
            records, tokenizer=tokenizer, max_sequence_length=recipe.max_sequence_length
        )
        if entries != replay:
            raise ValueError("token census replay differs")
        entries_by_arm[arm] = entries
        (raw_directory / f"{arm}_census.jsonl").write_text(
            "".join(json.dumps(asdict(entry), sort_keys=True) + "\n" for entry in entries),
            encoding="utf-8",
        )
    arm_summaries: dict[str, object] = {}
    for arm in ARM_ORDER:
        entries = entries_by_arm[arm]
        arm_summaries[arm] = {
            "census_sha256": canonical_sha256([asdict(entry) for entry in entries]),
            "loss_bearing_tokens": _distribution(entries),
            "by_family": _grouped_distributions(entries, "family"),
            "by_difficulty": _grouped_distributions(entries, "difficulty"),
            "by_output_contract": _grouped_distributions(entries, "output_contract_enabled"),
        }
    summary: dict[str, object] = {
        "schema_version": 1,
        "census_id": "foundry-token-census-v1",
        "base_model_id": recipe.base_model_id,
        "base_revision": recipe.base_revision,
        "tokenizer_sha256": recipe.tokenizer_sha256,
        "tokenizer_config_sha256": recipe.tokenizer_config_sha256,
        "chat_template_sha256": recipe.chat_template_sha256,
        "sft_format_sha256": recipe.sft_format_sha256,
        "max_sequence_length": recipe.max_sequence_length,
        "label_construction": "input_token_when_attention_mask_else_minus_100",
        "arms": arm_summaries,
        "total_training_examples": sum(len(value) for value in entries_by_arm.values()),
        "all_examples_have_loss_bearing_tokens": all(
            not entry.labels_entirely_masked
            for entries in entries_by_arm.values()
            for entry in entries
        ),
        "deterministic_reconstruction_match": True,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_census(path: Path) -> tuple[TokenCensusEntry, ...]:
    """Load and strictly reconstruct ignored census records."""

    entries = tuple(TokenCensusEntry(**record) for record in _load_jsonl(path))
    if len(entries) != 450 or len({entry.synthetic_id for entry in entries}) != 450:
        raise ValueError("census must contain 450 unique examples")
    return entries


def largest_remainder_quotas(
    total: int, counts: dict[tuple[str, str, bool], int]
) -> dict[tuple[str, str, bool], int]:
    """Allocate an exact integer total proportionally with stable tie-breaking."""

    denominator = sum(counts.values())
    if total < 0 or denominator <= 0 or total > denominator:
        raise ValueError("invalid largest-remainder inputs")
    exact = {key: total * value / denominator for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def _select_method_a_extras(
    generic: tuple[TokenCensusEntry, ...], targeted: tuple[TokenCensusEntry, ...]
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, int]], str]:
    by_arm: dict[str, dict[tuple[str, str, bool], list[TokenCensusEntry]]] = {}
    quotas_by_arm: dict[str, dict[tuple[str, str, bool], int]] = {}
    for arm, entries in (("generic_control", generic), ("targeted", targeted)):
        grouped: dict[tuple[str, str, bool], list[TokenCensusEntry]] = defaultdict(list)
        for entry in entries:
            grouped[entry.stratum].append(entry)
        by_arm[arm] = grouped
        quotas_by_arm[arm] = largest_remainder_quotas(
            EXTRA_REPEAT_COUNT, {key: len(value) for key, value in grouped.items()}
        )

    selected: dict[str, tuple[str, ...]] = {}
    selected_entries: dict[str, list[TokenCensusEntry]] = {}
    # Targeted has the higher three-cycle baseline. Maximizing generic extras and
    # minimizing targeted extras gives the exact lower bound on the pairwise gap.
    for arm in ARM_ORDER:
        values: list[TokenCensusEntry] = []
        for stratum in sorted(by_arm[arm]):
            stratum_entries = by_arm[arm][stratum]
            quota = quotas_by_arm[arm][stratum]
            if arm == "generic_control":
                ordered = sorted(
                    stratum_entries,
                    key=lambda item: (-item.loss_bearing_tokens, item.synthetic_id),
                )
            else:
                ordered = sorted(
                    stratum_entries,
                    key=lambda item: (item.loss_bearing_tokens, item.synthetic_id),
                )
            values.extend(ordered[:quota])
        selected_entries[arm] = values
        selected[arm] = tuple(sorted(item.synthetic_id for item in values))

    generic_total = 3 * sum(item.loss_bearing_tokens for item in generic) + sum(
        item.loss_bearing_tokens for item in selected_entries["generic_control"]
    )
    targeted_total = 3 * sum(item.loss_bearing_tokens for item in targeted) + sum(
        item.loss_bearing_tokens for item in selected_entries["targeted"]
    )
    if targeted_total < generic_total:
        raise ValueError("Method A boundary proof no longer establishes the exact optimum")
    quota_evidence = {
        arm: {
            "|".join((family, difficulty, str(output).lower())): quota
            for (family, difficulty, output), quota in sorted(quotas.items())
        }
        for arm, quotas in quotas_by_arm.items()
    }
    return selected, quota_evidence, "target_minimum_minus_generic_maximum_exact_lower_bound"


def _balanced_fixed_steps(
    entries: tuple[TokenCensusEntry, ...], extras: tuple[str, ...]
) -> tuple[ScheduledStep, ...]:
    extra_ids = set(extras)
    occurrences: list[ScheduledOccurrence] = []
    for entry in sorted(entries, key=lambda item: item.synthetic_id):
        repeats = 4 if entry.synthetic_id in extra_ids else 3
        for occurrence_index in range(1, repeats + 1):
            occurrences.append(
                ScheduledOccurrence(
                    synthetic_id=entry.synthetic_id,
                    occurrence_index=occurrence_index,
                    loss_bearing_tokens=entry.loss_bearing_tokens,
                )
            )
    if len(occurrences) != METHOD_A_OCCURRENCES:
        raise ValueError("Method A must contain exactly 1,600 occurrences")
    bins: list[list[ScheduledOccurrence]] = [[] for _ in range(OPTIMIZER_STEPS)]
    totals = [0] * OPTIMIZER_STEPS
    ordered = sorted(
        occurrences,
        key=lambda item: (-item.loss_bearing_tokens, item.synthetic_id, item.occurrence_index),
    )
    for occurrence in ordered:
        eligible = [index for index, values in enumerate(bins) if len(values) < 8]
        target = min(eligible, key=lambda index: (totals[index], index))
        bins[target].append(occurrence)
        totals[target] += occurrence.loss_bearing_tokens
    return tuple(
        ScheduledStep(
            step=index + 1,
            occurrences=tuple(
                sorted(values, key=lambda item: (item.synthetic_id, item.occurrence_index))
            ),
            loss_bearing_tokens=totals[index],
        )
        for index, values in enumerate(bins)
    )


def run_method_a(
    *,
    generic_census_path: Path,
    targeted_census_path: Path,
    raw_directory: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Construct and evaluate the predeclared fixed-occurrence Method A."""

    generic = load_census(generic_census_path)
    targeted = load_census(targeted_census_path)
    selected, quotas, optimality = _select_method_a_extras(generic, targeted)
    entries = {"generic_control": generic, "targeted": targeted}
    schedules = {arm: _balanced_fixed_steps(entries[arm], selected[arm]) for arm in ARM_ORDER}
    raw_directory.mkdir(parents=True, exist_ok=True)
    arms: dict[str, object] = {}
    totals: dict[str, int] = {}
    for arm in ARM_ORDER:
        schedule = schedules[arm]
        payload = [asdict(step) for step in schedule]
        (raw_directory / f"method_a_{arm}_schedule.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        counts = Counter(
            occurrence.synthetic_id for step in schedule for occurrence in step.occurrences
        )
        totals[arm] = sum(step.loss_bearing_tokens for step in schedule)
        arms[arm] = {
            "schedule_sha256": canonical_sha256(payload),
            "occurrences": sum(len(step.occurrences) for step in schedule),
            "optimizer_steps": len(schedule),
            "examples_repeated_three_times": sum(value == 3 for value in counts.values()),
            "examples_repeated_four_times": sum(value == 4 for value in counts.values()),
            "minimum_repeat_count": min(counts.values()),
            "maximum_repeat_count": max(counts.values()),
            "loss_bearing_tokens": totals[arm],
            "minimum_step_tokens": min(step.loss_bearing_tokens for step in schedule),
            "maximum_step_tokens": max(step.loss_bearing_tokens for step in schedule),
            "mean_step_tokens": statistics.fmean(step.loss_bearing_tokens for step in schedule),
            "step_token_population_variance": statistics.pvariance(
                step.loss_bearing_tokens for step in schedule
            ),
            "extra_repeat_quotas": quotas[arm],
        }
    difference = abs(totals["targeted"] - totals["generic_control"])
    relative = difference / max(totals.values())
    gate_checks = {
        "occurrences_exact": all(
            cast(dict[str, Any], arms[arm])["occurrences"] == METHOD_A_OCCURRENCES
            for arm in ARM_ORDER
        ),
        "optimizer_steps_exact": all(
            cast(dict[str, Any], arms[arm])["optimizer_steps"] == OPTIMIZER_STEPS
            for arm in ARM_ORDER
        ),
        "every_example_three_or_four": all(
            cast(dict[str, Any], arms[arm])["minimum_repeat_count"] == 3
            and cast(dict[str, Any], arms[arm])["maximum_repeat_count"] == 4
            for arm in ARM_ORDER
        ),
        "exactly_250_fourth_occurrences": all(
            cast(dict[str, Any], arms[arm])["examples_repeated_four_times"] == EXTRA_REPEAT_COUNT
            for arm in ARM_ORDER
        ),
        "token_parity_within_0_5_percent": relative <= METHOD_A_PARITY_LIMIT,
        "step_safety_within_120_percent": all(
            cast(dict[str, Any], arms[arm])["maximum_step_tokens"]
            <= PREVIOUS_SAFE_PADDED_TOKENS_PER_STEP * 1.2
            for arm in ARM_ORDER
        ),
        "deterministic_reconstruction": all(
            schedules[arm] == _balanced_fixed_steps(entries[arm], selected[arm])
            for arm in ARM_ORDER
        ),
        "no_data_content_changed": True,
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "method": "balanced_fixed_occurrence_token_matching",
        "selection_optimality": optimality,
        "arms": arms,
        "absolute_token_difference": difference,
        "relative_token_difference": relative,
        "parity_limit": METHOD_A_PARITY_LIMIT,
        "previous_safe_padded_token_envelope_per_step": PREVIOUS_SAFE_PADDED_TOKENS_PER_STEP,
        "gate_checks": gate_checks,
        "method_a_gate_passed": all(gate_checks.values()),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _stable_extra_selection(
    entries: tuple[TokenCensusEntry, ...], remainder: int, *, arm: str
) -> tuple[str, ...]:
    grouped: dict[tuple[str, str, bool], list[TokenCensusEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.stratum].append(entry)
    quotas = largest_remainder_quotas(
        remainder, {key: len(values) for key, values in grouped.items()}
    )
    selected: list[str] = []
    for stratum in sorted(grouped):
        ordered = sorted(
            grouped[stratum],
            key=lambda entry: (
                hashlib.sha256(
                    (
                        f"{METHOD_B_SELECTION_SEED}:{arm}:{entry.synthetic_id}:method-b-extra"
                    ).encode()
                ).hexdigest(),
                entry.synthetic_id,
            ),
        )
        selected.extend(entry.synthetic_id for entry in ordered[: quotas[stratum]])
    return tuple(sorted(selected))


def _method_b_occurrences(
    entries: tuple[TokenCensusEntry, ...], *, arm: str
) -> tuple[ScheduledOccurrence, ...]:
    """Choose the closest balanced-cycle whole-example budget for one arm."""

    best: tuple[int, int, tuple[ScheduledOccurrence, ...]] | None = None
    for occurrence_count in range(450, 1801):
        full_cycles, remainder = divmod(occurrence_count, len(entries))
        extra_ids = set(_stable_extra_selection(entries, remainder, arm=arm))
        occurrences: list[ScheduledOccurrence] = []
        for entry in sorted(entries, key=lambda item: item.synthetic_id):
            repeats = full_cycles + int(entry.synthetic_id in extra_ids)
            for occurrence_index in range(1, repeats + 1):
                occurrences.append(
                    ScheduledOccurrence(
                        synthetic_id=entry.synthetic_id,
                        occurrence_index=occurrence_index,
                        loss_bearing_tokens=entry.loss_bearing_tokens,
                    )
                )
        total = sum(item.loss_bearing_tokens for item in occurrences)
        candidate = (abs(total - METHOD_B_NOMINAL_TOKENS), occurrence_count, tuple(occurrences))
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("Method B occurrence search produced no candidate")
    return best[2]


def _balanced_variable_steps(
    occurrences: tuple[ScheduledOccurrence, ...],
) -> tuple[ScheduledStep, ...]:
    if len(occurrences) < OPTIMIZER_STEPS:
        raise ValueError("Method B needs at least one occurrence per optimizer step")
    bins: list[list[ScheduledOccurrence]] = [[] for _ in range(OPTIMIZER_STEPS)]
    totals = [0] * OPTIMIZER_STEPS
    ordered = sorted(
        occurrences,
        key=lambda item: (-item.loss_bearing_tokens, item.synthetic_id, item.occurrence_index),
    )
    for occurrence in ordered:
        target = min(range(OPTIMIZER_STEPS), key=lambda index: (totals[index], index))
        bins[target].append(occurrence)
        totals[target] += occurrence.loss_bearing_tokens
    steps = tuple(
        ScheduledStep(
            step=index + 1,
            occurrences=tuple(
                sorted(values, key=lambda item: (item.synthetic_id, item.occurrence_index))
            ),
            loss_bearing_tokens=totals[index],
        )
        for index, values in enumerate(bins)
    )
    if any(not step.occurrences for step in steps):
        raise ValueError("Method B created an empty optimizer step")
    return steps


def _stratum_repeat_balance(
    entries: tuple[TokenCensusEntry, ...], schedule: tuple[ScheduledStep, ...]
) -> tuple[bool, dict[str, dict[str, int]]]:
    counts = Counter(
        occurrence.synthetic_id for step in schedule for occurrence in step.occurrences
    )
    grouped: dict[tuple[str, str, bool], list[int]] = defaultdict(list)
    for entry in entries:
        grouped[entry.stratum].append(counts[entry.synthetic_id])
    evidence = {
        "|".join((family, difficulty, str(output).lower())): {
            "examples": len(values),
            "minimum_repeats": min(values),
            "maximum_repeats": max(values),
            "total_occurrences": sum(values),
        }
        for (family, difficulty, output), values in sorted(grouped.items())
    }
    return all(max(values) - min(values) <= 1 for values in grouped.values()), evidence


def _reorder_for_four_step_smoke(
    schedules: dict[str, tuple[ScheduledStep, ...]],
) -> tuple[dict[str, tuple[ScheduledStep, ...]], dict[str, int]]:
    """Move the closest real four-step window in each arm to its prefix."""

    windows = {
        arm: [
            sum(schedule[start + offset].loss_bearing_tokens for offset in range(4))
            for start in range(OPTIMIZER_STEPS - 3)
        ]
        for arm, schedule in schedules.items()
    }
    generic_start, targeted_start = min(
        (
            (generic_start, targeted_start)
            for generic_start in range(OPTIMIZER_STEPS - 3)
            for targeted_start in range(OPTIMIZER_STEPS - 3)
        ),
        key=lambda pair: (
            abs(windows["generic_control"][pair[0]] - windows["targeted"][pair[1]]),
            pair,
        ),
    )
    starts = {"generic_control": generic_start, "targeted": targeted_start}
    reordered: dict[str, tuple[ScheduledStep, ...]] = {}
    for arm in ARM_ORDER:
        start = starts[arm]
        schedule = schedules[arm]
        selected = schedule[start : start + 4]
        remaining = schedule[:start] + schedule[start + 4 :]
        reordered[arm] = tuple(
            ScheduledStep(
                step=index + 1,
                occurrences=step.occurrences,
                loss_bearing_tokens=step.loss_bearing_tokens,
            )
            for index, step in enumerate(selected + remaining)
        )
    return reordered, starts


def _build_method_b_schedules(
    entries: dict[str, tuple[TokenCensusEntry, ...]],
) -> tuple[dict[str, tuple[ScheduledStep, ...]], dict[str, int]]:
    occurrences = {arm: _method_b_occurrences(entries[arm], arm=arm) for arm in ARM_ORDER}
    schedules = {arm: _balanced_variable_steps(occurrences[arm]) for arm in ARM_ORDER}
    return _reorder_for_four_step_smoke(schedules)


def run_method_b(
    *,
    generic_census_path: Path,
    targeted_census_path: Path,
    raw_directory: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Construct and evaluate the preapproved whole-example Method B schedules."""

    entries = {
        "generic_control": load_census(generic_census_path),
        "targeted": load_census(targeted_census_path),
    }
    schedules, smoke_prefix_starts = _build_method_b_schedules(entries)

    raw_directory.mkdir(parents=True, exist_ok=True)
    arms: dict[str, object] = {}
    totals: dict[str, int] = {}
    balance_checks: dict[str, bool] = {}
    for arm in ARM_ORDER:
        schedule = schedules[arm]
        payload = [asdict(step) for step in schedule]
        path = raw_directory / f"method_b_{arm}_schedule.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        counts = Counter(
            occurrence.synthetic_id for step in schedule for occurrence in step.occurrences
        )
        balanced, stratum_evidence = _stratum_repeat_balance(entries[arm], schedule)
        balance_checks[arm] = balanced
        totals[arm] = sum(step.loss_bearing_tokens for step in schedule)
        arms[arm] = {
            "schedule_sha256": canonical_sha256(payload),
            "schedule_file_sha256": file_sha256(path),
            "occurrences": sum(len(step.occurrences) for step in schedule),
            "optimizer_steps": len(schedule),
            "unique_examples": len(counts),
            "minimum_repeat_count": min(counts.values()),
            "maximum_repeat_count": max(counts.values()),
            "loss_bearing_tokens": totals[arm],
            "nominal_token_difference": abs(totals[arm] - METHOD_B_NOMINAL_TOKENS),
            "nominal_token_relative_difference": abs(totals[arm] - METHOD_B_NOMINAL_TOKENS)
            / METHOD_B_NOMINAL_TOKENS,
            "minimum_step_tokens": min(step.loss_bearing_tokens for step in schedule),
            "maximum_step_tokens": max(step.loss_bearing_tokens for step in schedule),
            "mean_step_tokens": statistics.fmean(step.loss_bearing_tokens for step in schedule),
            "step_token_population_variance": statistics.pvariance(
                step.loss_bearing_tokens for step in schedule
            ),
            "first_four_step_tokens": sum(step.loss_bearing_tokens for step in schedule[:4]),
            "smoke_prefix_source_start": smoke_prefix_starts[arm] + 1,
            "stratum_repeat_balance": stratum_evidence,
        }
    pair_difference = abs(totals["targeted"] - totals["generic_control"])
    pair_relative = pair_difference / max(totals.values())
    smoke_tokens = {
        arm: cast(dict[str, Any], arms[arm])["first_four_step_tokens"] for arm in ARM_ORDER
    }
    smoke_relative = abs(smoke_tokens["targeted"] - smoke_tokens["generic_control"]) / max(
        smoke_tokens.values()
    )
    gate_checks = {
        "optimizer_steps_exact": all(
            cast(dict[str, Any], arms[arm])["optimizer_steps"] == OPTIMIZER_STEPS
            for arm in ARM_ORDER
        ),
        "totals_within_0_5_percent_of_nominal": all(
            cast(dict[str, Any], arms[arm])["nominal_token_relative_difference"]
            <= METHOD_B_PARITY_LIMIT
            for arm in ARM_ORDER
        ),
        "pairwise_parity_within_0_5_percent": pair_relative <= METHOD_B_PARITY_LIMIT,
        "every_example_appears": all(
            cast(dict[str, Any], arms[arm])["unique_examples"] == 450
            and cast(dict[str, Any], arms[arm])["minimum_repeat_count"] >= 1
            for arm in ARM_ORDER
        ),
        "stratum_repeat_counts_balanced": all(balance_checks.values()),
        "step_safety_within_observed_padded_envelope": all(
            cast(dict[str, Any], arms[arm])["maximum_step_tokens"]
            <= PREVIOUS_SAFE_PADDED_TOKENS_PER_STEP
            for arm in ARM_ORDER
        ),
        "four_step_smoke_parity_within_0_5_percent": smoke_relative <= METHOD_B_PARITY_LIMIT,
        "deterministic_reconstruction": schedules == _build_method_b_schedules(entries)[0],
        "whole_examples_only": True,
        "no_data_content_changed": True,
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "method": "whole_example_token_budgeted_gradient_accumulation",
        "nominal_loss_bearing_tokens": METHOD_B_NOMINAL_TOKENS,
        "nominal_tokens_per_step": METHOD_B_NOMINAL_TOKENS // OPTIMIZER_STEPS,
        "selection_seed": METHOD_B_SELECTION_SEED,
        "arms": arms,
        "pairwise_absolute_token_difference": pair_difference,
        "pairwise_relative_token_difference": pair_relative,
        "four_step_smoke_tokens": smoke_tokens,
        "four_step_smoke_relative_difference": smoke_relative,
        "parity_limit": METHOD_B_PARITY_LIMIT,
        "safety_envelope_loss_tokens": PREVIOUS_SAFE_PADDED_TOKENS_PER_STEP,
        "gate_checks": gate_checks,
        "method_b_schedule_gate_passed": all(gate_checks.values()),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    census = subparsers.add_parser("census")
    census.add_argument("--recipe", required=True, type=Path)
    census.add_argument("--model-path", required=True, type=Path)
    census.add_argument("--generic", required=True, type=Path)
    census.add_argument("--targeted", required=True, type=Path)
    census.add_argument("--raw-directory", required=True, type=Path)
    census.add_argument("--summary", required=True, type=Path)
    method_a = subparsers.add_parser("method-a")
    method_a.add_argument("--generic-census", required=True, type=Path)
    method_a.add_argument("--targeted-census", required=True, type=Path)
    method_a.add_argument("--raw-directory", required=True, type=Path)
    method_a.add_argument("--summary", required=True, type=Path)
    method_b = subparsers.add_parser("method-b")
    method_b.add_argument("--generic-census", required=True, type=Path)
    method_b.add_argument("--targeted-census", required=True, type=Path)
    method_b.add_argument("--raw-directory", required=True, type=Path)
    method_b.add_argument("--summary", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "census":
        result = run_census(
            recipe_path=args.recipe,
            model_path=args.model_path,
            generic_path=args.generic,
            targeted_path=args.targeted,
            raw_directory=args.raw_directory,
            summary_path=args.summary,
        )
    elif args.command == "method-a":
        result = run_method_a(
            generic_census_path=args.generic_census,
            targeted_census_path=args.targeted_census,
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
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
