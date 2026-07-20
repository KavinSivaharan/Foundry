"""Deterministic 500x2 matched-template dataset generation and validation."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import psutil  # type: ignore[import-untyped]
import torch
import yaml

from foundry.synthesis.contamination import (
    ContaminationOutcome,
    canonical_number_neutral_identity,
    load_development_questions_for_contamination,
    normalized_text_sha256,
)
from foundry.synthesis.deduplication import DeduplicationIndex
from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.pipeline import _verify
from foundry.synthesis.quality import validate_rendered_candidate
from foundry.synthesis.realization import validate_realization
from foundry.synthesis.semantic import PinnedSentenceEncoder, load_semantic_artifact_config
from foundry.synthesis.template_bank.bank import build_template_bank
from foundry.synthesis.template_bank.composition import audit_surface_provenance
from foundry.synthesis.template_bank.contracts import SentencePlanSpec, TemplateSpec
from foundry.synthesis.template_bank.matched_policy import MatchedTemplateCaps, derive_caps
from foundry.synthesis.template_bank.renderer import render_with_template
from foundry.synthesis.template_bank.reuse import load_contract
from foundry.synthesis.template_bank.signal_allocator import (
    LatentCandidate,
    _generate,
    _latent_hash,
    _mode,
    _pool_for_mode,
)
from foundry.synthesis.template_bank.signal_pilot import (
    CATEGORY_ORDER,
    DIFFICULTY_ORDER,
    GROUP_ORDER,
    MODE_ORDER,
    SPLIT_ORDER,
    balanced_counts,
    canonical_sha256,
)
from foundry.synthesis.template_bank.submode_policy import (
    balanced_matrix,
    constrained_attempt_split,
    load_policy_config,
    water_fill,
)
from foundry.synthesis.verification import validate_final_answer_contract

DATASET_ID = "foundry-matched-template-signal-500x2-v1"
SCHEDULE_VERSION = "foundry-matched-template-schedule-v1"
FAMILY_ACCEPTED = {
    "targeted": dict(zip(CATEGORY_ORDER, (275, 117, 108), strict=True)),
    "generic_control": dict(zip(CATEGORY_ORDER, (167, 167, 166), strict=True)),
}
FAMILY_ATTEMPTS = {
    "targeted": dict(zip(CATEGORY_ORDER, (302, 129, 119), strict=True)),
    "generic_control": dict(zip(CATEGORY_ORDER, (184, 184, 182), strict=True)),
}
SCENARIO_INVENTORY = dict(zip(CATEGORY_ORDER, (24, 5, 20), strict=True))


@dataclass(frozen=True)
class MatchedDatasetConfig:
    """Strict, hashable generation contract."""

    config_sha256: str
    master_seed: str
    policy_config: Path
    submode_policy_config: Path
    template_reuse_config: Path
    semantic_config: Path
    development_manifest: Path
    evaluation_config: Path
    schedule_manifest: Path
    dataset_manifest: Path
    summary_path: Path
    raw_directory: Path
    review_directory: Path


@dataclass(frozen=True)
class MatchedSlot:
    """One fixed candidate position before latent and surface assignment."""

    group: str
    family: str
    mode: str
    difficulty: str
    output_contract_enabled: bool
    future_split: str
    quota_cell_id: str
    quota_primary: bool


@dataclass(frozen=True)
class MatchedScheduleRecord:
    """Content-free reconstruction data for one fixed candidate."""

    attempt_index: int
    slot_id: str
    synthetic_id: str
    group: str
    family: str
    mode: str
    difficulty: str
    output_contract_enabled: bool
    future_split: str
    quota_cell_id: str
    quota_primary: bool
    quota_cell_target: int
    latent_seed: int
    generator_variant: int
    latent_program_sha256: str
    semantic_ir_sha256: str
    template_id: str
    sentence_plan_id: str
    semantic_frame: str
    scenario_domain: str
    lexical_family: str
    rendered_text_sha256: str
    exact_text_sha256: str
    render_signature_sha256: str
    number_neutral_sha256: str


@dataclass(frozen=True)
class MatchedAttemptRecord:
    """Ignored content-bearing validation result for one candidate."""

    attempt_index: int
    synthetic_id: str
    group: str
    family: str
    mode: str
    difficulty: str
    output_contract_enabled: bool
    future_split: str
    quota_cell_id: str
    template_id: str
    sentence_plan_id: str
    semantic_frame: str
    scenario_domain: str
    rendered_question: str
    deterministic_solution_trace: tuple[str, ...]
    canonical_final_answer: str
    training_completion: str
    latent_program_sha256: str
    semantic_ir_sha256: str
    rendered_text_sha256: str
    exact_text_sha256: str
    render_signature_sha256: str
    number_neutral_sha256: str
    surface_provenance_sha256: str
    primary_evidence_sha256: str
    independent_evidence_sha256: str
    primary_verifier_success: bool
    independent_verifier_success: bool
    verifier_agreement: bool
    deterministic_language_reasons: tuple[str, ...]
    benchmark_lexical_reason: str | None
    benchmark_ngram_maximum: float
    benchmark_semantic_outcome: str
    benchmark_semantic_maximum: float
    final_decision: str
    rejection_reason: str | None


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    denominator = sum(weights.values())
    values = {key: total * value // denominator for key, value in weights.items()}
    missing = total - sum(values.values())
    order = {key: index for index, key in enumerate(weights)}
    ranked = sorted(
        weights,
        key=lambda key: (-((total * weights[key]) % denominator), order[key]),
    )
    for key in ranked[:missing]:
        values[key] += 1
    return values


def _stable_labels(counts: dict[str, int], material: str) -> list[str]:
    expanded = [
        (label, occurrence) for label, quantity in counts.items() for occurrence in range(quantity)
    ]
    expanded.sort(
        key=lambda item: hashlib.sha256(f"{material}:{item[0]}:{item[1]}".encode()).hexdigest()
    )
    return [label for label, _ in expanded]


def load_matched_dataset_config(path: Path) -> MatchedDatasetConfig:
    """Load the exact 1,100-attempt contract and fail on any quota drift."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "matched dataset config")
    if root.get("schema_version") != 1 or root.get("dataset_id") != DATASET_ID:
        raise ValueError("matched dataset identity differs")
    if root.get("candidate_attempts_total") != 1100:
        raise ValueError("matched candidate pool must contain exactly 1,100 attempts")
    datasets = _mapping(root.get("datasets"), "datasets")
    if tuple(datasets) != GROUP_ORDER:
        raise ValueError("matched dataset group order differs")
    for group in GROUP_ORDER:
        current = _mapping(datasets[group], group)
        if (
            current.get("accepted_total") != 500
            or current.get("attempts_total") != 550
            or current.get("training") != 450
            or current.get("synthetic_validation") != 50
            or current.get("output_contract_enabled") != 100
        ):
            raise ValueError(f"{group} matched dataset totals differ")
        families = _mapping(current.get("families"), f"{group}.families")
        if tuple(families) != CATEGORY_ORDER:
            raise ValueError(f"{group} family order differs")
        accepted: dict[str, int] = {}
        attempts: dict[str, int] = {}
        for family in CATEGORY_ORDER:
            item = _mapping(families[family], f"{group}.{family}")
            accepted[family] = cast(int, item.get("accepted"))
            attempts[family] = cast(int, item.get("attempts"))
        if accepted != FAMILY_ACCEPTED[group] or attempts != FAMILY_ATTEMPTS[group]:
            raise ValueError(f"{group} family quotas differ")
    safety = _mapping(root.get("safety"), "safety")
    if safety != {
        "benchmark_answers_allowed": False,
        "sealed_final_allowed": False,
        "llm_generation_allowed": False,
        "replacement_pool_allowed": False,
        "training_allowed_before_dataset_gate": False,
    }:
        raise ValueError("matched dataset safety boundary differs")
    paths = _mapping(root.get("paths"), "paths")
    canonical = json.loads(json.dumps(root, sort_keys=True))
    return MatchedDatasetConfig(
        config_sha256=canonical_sha256(canonical),
        master_seed=cast(str, root.get("master_seed")),
        policy_config=Path(cast(str, root.get("policy_config"))),
        submode_policy_config=Path(cast(str, root.get("submode_policy_config"))),
        template_reuse_config=Path(cast(str, root.get("template_reuse_config"))),
        semantic_config=Path(cast(str, root.get("semantic_config"))),
        development_manifest=Path(cast(str, root.get("development_manifest"))),
        evaluation_config=Path(cast(str, root.get("evaluation_config"))),
        schedule_manifest=Path(cast(str, paths.get("schedule_manifest"))),
        dataset_manifest=Path(cast(str, paths.get("dataset_manifest"))),
        summary_path=Path(cast(str, paths.get("summary"))),
        raw_directory=Path(cast(str, paths.get("raw_directory"))),
        review_directory=Path(cast(str, paths.get("review_directory"))),
    )


def _mode_margins(config: MatchedDatasetConfig) -> tuple[dict[str, object], dict[str, object]]:
    policy = load_policy_config(config.submode_policy_config)
    accepted: dict[str, dict[str, dict[str, int]]] = {group: {} for group in GROUP_ORDER}
    attempts: dict[str, dict[str, dict[str, int]]] = {group: {} for group in GROUP_ORDER}
    for family in CATEGORY_ORDER:
        global_accepted = water_fill(
            sum(FAMILY_ACCEPTED[group][family] for group in GROUP_ORDER),
            policy.capacities[family],
        )
        global_attempts = water_fill(
            sum(FAMILY_ATTEMPTS[group][family] for group in GROUP_ORDER),
            policy.capacities[family],
        )
        first_attempts, second_attempts, first_accepted, second_accepted = (
            constrained_attempt_split(
                global_attempts,
                global_accepted,
                first_attempt_total=FAMILY_ATTEMPTS[GROUP_ORDER[0]][family],
                first_accepted_total=FAMILY_ACCEPTED[GROUP_ORDER[0]][family],
            )
        )
        attempts[GROUP_ORDER[0]][family] = first_attempts
        attempts[GROUP_ORDER[1]][family] = second_attempts
        accepted[GROUP_ORDER[0]][family] = first_accepted
        accepted[GROUP_ORDER[1]][family] = second_accepted
    return cast(dict[str, object], accepted), cast(dict[str, object], attempts)


def _family_targets(group: str, family: str) -> tuple[int, int]:
    training = _largest_remainder(450, FAMILY_ACCEPTED[group])[family]
    output = _largest_remainder(100, FAMILY_ACCEPTED[group])[family]
    return training, output


def build_quota_contract(config_path: Path) -> dict[str, object]:
    """Freeze exact mode/cell margins for accepted examples and fixed attempts."""

    config = load_matched_dataset_config(config_path)
    accepted_modes_raw, attempt_modes_raw = _mode_margins(config)
    accepted_modes = cast(dict[str, dict[str, dict[str, int]]], accepted_modes_raw)
    attempt_modes = cast(dict[str, dict[str, dict[str, int]]], attempt_modes_raw)
    groups: dict[str, object] = {}
    for group in GROUP_ORDER:
        families: dict[str, object] = {}
        for family in CATEGORY_ORDER:
            accepted_total = FAMILY_ACCEPTED[group][family]
            training, output = _family_targets(group, family)
            difficulty_matrix = balanced_matrix(
                accepted_modes[group][family],
                balanced_counts(accepted_total, DIFFICULTY_ORDER),
            )
            output_matrix = balanced_matrix(
                accepted_modes[group][family],
                {"enabled": output, "disabled": accepted_total - output},
            )
            split_matrix = balanced_matrix(
                accepted_modes[group][family],
                {"training": training, "synthetic_validation": accepted_total - training},
            )
            cell_counts: Counter[str] = Counter()
            cell_values: dict[str, dict[str, object]] = {}
            for mode in MODE_ORDER[family]:
                count = accepted_modes[group][family][mode]
                difficulties = _stable_labels(
                    difficulty_matrix[mode], f"{config.master_seed}:{group}:{family}:{mode}:d"
                )
                outputs = _stable_labels(
                    output_matrix[mode], f"{config.master_seed}:{group}:{family}:{mode}:o"
                )
                splits = _stable_labels(
                    split_matrix[mode], f"{config.master_seed}:{group}:{family}:{mode}:s"
                )
                for index in range(count):
                    values = {
                        "mode": mode,
                        "difficulty": difficulties[index],
                        "output_contract_enabled": outputs[index] == "enabled",
                        "future_split": splits[index],
                    }
                    cell_id = canonical_sha256({"group": group, "family": family, **values})[:20]
                    cell_counts[cell_id] += 1
                    cell_values[cell_id] = values
            attempt_cells = dict(cell_counts)
            for mode in MODE_ORDER[family]:
                reserve = attempt_modes[group][family][mode] - accepted_modes[group][family][mode]
                mode_cells = {
                    cell_id: count
                    for cell_id, count in cell_counts.items()
                    if cell_values[cell_id]["mode"] == mode
                }
                extras = _largest_remainder(reserve, mode_cells) if reserve else {}
                for cell_id, quantity in extras.items():
                    attempt_cells[cell_id] += quantity
            families[family] = {
                "accepted": accepted_total,
                "attempts": FAMILY_ATTEMPTS[group][family],
                "accepted_modes": accepted_modes[group][family],
                "attempt_modes": attempt_modes[group][family],
                "accepted_difficulty": dict(
                    sorted(
                        Counter(
                            cast(str, cell_values[cell_id]["difficulty"])
                            for cell_id, count in cell_counts.items()
                            for _ in range(count)
                        ).items()
                    )
                ),
                "accepted_output_contract": output,
                "accepted_training": training,
                "cells": [
                    {
                        "cell_id": cell_id,
                        **cell_values[cell_id],
                        "accepted": cell_counts[cell_id],
                        "attempts": attempt_cells[cell_id],
                    }
                    for cell_id in sorted(cell_counts)
                ],
            }
        groups[group] = {"families": families}
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "master_seed": config.master_seed,
        "groups": groups,
        "accepted_total": 1000,
        "attempt_total": 1100,
    }
    payload["quota_sha256"] = canonical_sha256(payload)
    return payload


def _slots(config_path: Path) -> tuple[MatchedSlot, ...]:
    quota = build_quota_contract(config_path)
    slots: list[MatchedSlot] = []
    groups = cast(dict[str, dict[str, object]], quota["groups"])
    for group in GROUP_ORDER:
        families = cast(dict[str, dict[str, object]], groups[group]["families"])
        primaries: list[MatchedSlot] = []
        reserves: list[MatchedSlot] = []
        for family in CATEGORY_ORDER:
            for raw_cell in cast(list[dict[str, object]], families[family]["cells"]):
                accepted = cast(int, raw_cell["accepted"])
                attempts = cast(int, raw_cell["attempts"])
                common = (
                    group,
                    family,
                    cast(str, raw_cell["mode"]),
                    cast(str, raw_cell["difficulty"]),
                    cast(bool, raw_cell["output_contract_enabled"]),
                    cast(str, raw_cell["future_split"]),
                    cast(str, raw_cell["cell_id"]),
                )
                primaries.extend(MatchedSlot(*common, quota_primary=True) for _ in range(accepted))
                reserves.extend(
                    MatchedSlot(*common, quota_primary=False) for _ in range(attempts - accepted)
                )
        primaries.sort(
            key=lambda item: canonical_sha256({"seed": quota["master_seed"], **asdict(item)})
        )
        reserves.sort(
            key=lambda item: canonical_sha256({"seed": quota["master_seed"], **asdict(item)})
        )
        slots.extend(primaries + reserves)
    if len(slots) != 1100:
        raise ValueError("matched schedule does not contain exactly 1,100 slots")
    return tuple(slots)


def _compatible_bank(
    bank: tuple[TemplateSpec, ...], family: str, mode: str
) -> tuple[TemplateSpec, ...]:
    values = [item for item in bank if str(item.reasoning_category) == family]
    if family != CATEGORY_ORDER[0]:
        values = [item for item in values if item.semantic_frame.startswith(mode + ".")]
    if any(item.review_status != "human_review_pending" for item in values):
        raise ValueError("matched schedule includes an ineligible template")
    return tuple(values)


def _find_plan(
    bank: tuple[TemplateSpec, ...], template_id: str, plan_id: str
) -> tuple[TemplateSpec, SentencePlanSpec]:
    for template in bank:
        if template.template_id == template_id:
            for plan in template.sentence_plan_variants:
                if plan.plan_id == plan_id:
                    return template, plan
    raise ValueError("matched scheduled template or plan is missing")


def build_schedule(
    config_path: Path, *, progress: bool = False
) -> tuple[MatchedScheduleRecord, ...]:
    """Construct all 1,100 unique latent/exact candidates without persistence."""

    config = load_matched_dataset_config(config_path)
    quota = build_quota_contract(config_path)
    cell_targets = {
        cast(str, cell["cell_id"]): cast(int, cell["accepted"])
        for group in cast(dict[str, dict[str, object]], quota["groups"]).values()
        for family in cast(dict[str, dict[str, object]], group["families"]).values()
        for cell in cast(list[dict[str, object]], family["cells"])
    }
    slots = _slots(config_path)
    bank = build_template_bank()
    latent_by_position: dict[int, LatentCandidate] = {}
    used_latent: set[str] = set()
    scenario_use: Counter[tuple[str, str, str]] = Counter()
    lexical_use: Counter[tuple[str, str, str]] = Counter()
    for family in CATEGORY_ORDER:
        for mode in MODE_ORDER[family]:
            positions = [
                index
                for index, slot in enumerate(slots)
                if slot.family == family and slot.mode == mode
            ]
            pool = _pool_for_mode(
                family=family,
                mode=mode,
                master_seed=f"{config.master_seed}:latent",
                pool_size=1024,
            )
            for position in positions:
                slot = slots[position]
                choices = [
                    by_difficulty[slot.difficulty]
                    for latent_hash, by_difficulty in sorted(pool.items())
                    if latent_hash not in used_latent and slot.difficulty in by_difficulty
                ]
                if not choices:
                    raise ValueError(f"unique latent capacity exhausted for {family}/{mode}")
                selected = min(
                    choices,
                    key=lambda item: (
                        scenario_use[(slot.group, family, item.scenario_domain)],
                        lexical_use[(slot.group, family, item.lexical_family)],
                        item.latent_sha256,
                    ),
                )
                latent_by_position[position] = selected
                used_latent.add(selected.latent_sha256)
                scenario_use[(slot.group, family, selected.scenario_domain)] += 1
                lexical_use[(slot.group, family, selected.lexical_family)] += 1
            if progress:
                print(f"matched schedule latent {family}/{mode}: {len(positions)}", flush=True)
    if len(used_latent) != 1100:
        raise ValueError("matched schedule latent programs are not globally unique")

    plan_use: Counter[tuple[str, str, str]] = Counter()
    frame_use: Counter[tuple[str, str, str]] = Counter()
    number_neutral_use: Counter[tuple[str, str, str]] = Counter()
    exact_hashes: set[str] = set()
    records: list[MatchedScheduleRecord] = []
    for position, slot in enumerate(slots):
        latent = latent_by_position[position]
        draft = _generate(
            slot.family,
            seed=latent.seed,
            difficulty=slot.difficulty,
            variant=latent.variant,
            output_contract_enabled=slot.output_contract_enabled,
        )
        if _latent_hash(draft) != latent.latent_sha256 or _mode(draft) != slot.mode:
            raise ValueError("matched latent reconstruction differs")
        options: list[tuple[tuple[object, ...], TemplateSpec, SentencePlanSpec, str, str, str]] = []
        for template in _compatible_bank(bank, slot.family, slot.mode):
            for plan in template.sentence_plan_variants:
                rendered = render_with_template(draft, template, plan)
                normalized_hash = normalized_text_sha256(rendered.rendered_question)
                if normalized_hash in exact_hashes:
                    continue
                exact_hash = hashlib.sha256(rendered.rendered_question.encode("utf-8")).hexdigest()
                number_neutral = canonical_number_neutral_identity(
                    rendered.rendered_question
                ).sha256
                plan_key = f"{template.template_id}:{plan.plan_id}"
                score = (
                    plan_use[(slot.group, slot.family, plan_key)],
                    frame_use[(slot.group, slot.family, template.template_id)],
                    scenario_use[(slot.group, slot.family, latent.scenario_domain)],
                    number_neutral_use[(slot.group, slot.family, number_neutral)],
                    canonical_sha256(
                        {
                            "seed": config.master_seed,
                            "position": position,
                            "template": template.template_id,
                            "plan": plan.plan_id,
                        }
                    ),
                )
                options.append((score, template, plan, normalized_hash, exact_hash, number_neutral))
        if not options:
            raise ValueError("matched exact rendered-question capacity exhausted")
        _, template, plan, normalized_hash, exact_hash, number_neutral = min(
            options, key=lambda item: item[0]
        )
        plan_key = f"{template.template_id}:{plan.plan_id}"
        plan_use[(slot.group, slot.family, plan_key)] += 1
        frame_use[(slot.group, slot.family, template.template_id)] += 1
        number_neutral_use[(slot.group, slot.family, number_neutral)] += 1
        exact_hashes.add(normalized_hash)
        slot_material = {"position": position + 1, **asdict(slot)}
        slot_id = f"matched-slot-{position + 1:04d}-{canonical_sha256(slot_material)[:12]}"
        synthetic_material = {"slot": slot_id, "latent": latent.latent_sha256}
        synthetic_id = f"matched-syn-{canonical_sha256(synthetic_material)[:24]}"
        records.append(
            MatchedScheduleRecord(
                attempt_index=position + 1,
                slot_id=slot_id,
                synthetic_id=synthetic_id,
                group=slot.group,
                family=slot.family,
                mode=slot.mode,
                difficulty=slot.difficulty,
                output_contract_enabled=slot.output_contract_enabled,
                future_split=slot.future_split,
                quota_cell_id=slot.quota_cell_id,
                quota_primary=slot.quota_primary,
                quota_cell_target=cell_targets[slot.quota_cell_id],
                latent_seed=latent.seed,
                generator_variant=latent.variant,
                latent_program_sha256=latent.latent_sha256,
                semantic_ir_sha256=draft.semantic_ir_sha256,
                template_id=template.template_id,
                sentence_plan_id=plan.plan_id,
                semantic_frame=template.semantic_frame,
                scenario_domain=latent.scenario_domain,
                lexical_family=latent.lexical_family,
                rendered_text_sha256=normalized_hash,
                exact_text_sha256=exact_hash,
                render_signature_sha256=template.render_signature_hash(plan),
                number_neutral_sha256=number_neutral,
            )
        )
    if len({item.synthetic_id for item in records}) != 1100:
        raise ValueError("matched synthetic IDs are not globally unique")
    if len({item.rendered_text_sha256 for item in records}) != 1100:
        raise ValueError("matched normalized questions are not globally unique")
    return tuple(records)


def write_schedule(config_path: Path, *, progress: bool = False) -> dict[str, object]:
    """Write the tracked content-free fixed schedule and hash."""

    config = load_matched_dataset_config(config_path)
    quota = build_quota_contract(config_path)
    records = build_schedule(config_path, progress=progress)
    payload: dict[str, object] = {
        "schema_version": 1,
        "schedule_version": SCHEDULE_VERSION,
        "dataset_id": DATASET_ID,
        "config_sha256": config.config_sha256,
        "quota_contract": quota,
        "record_count": len(records),
        "records": [asdict(item) for item in records],
    }
    payload["schedule_sha256"] = canonical_sha256(payload)
    config.schedule_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.schedule_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def load_schedule(config: MatchedDatasetConfig) -> tuple[MatchedScheduleRecord, ...]:
    """Load and hash-check the fixed content-free schedule."""

    root = json.loads(config.schedule_manifest.read_text(encoding="utf-8"))
    expected = root.pop("schedule_sha256", None)
    if expected != canonical_sha256(root):
        raise ValueError("matched schedule hash differs")
    records = tuple(MatchedScheduleRecord(**item) for item in root["records"])
    if len(records) != 1100 or len({item.slot_id for item in records}) != 1100:
        raise ValueError("matched schedule count or identity differs")
    return records


@dataclass(frozen=True)
class _PreparedCandidate:
    scheduled: MatchedScheduleRecord
    draft: CandidateDraft
    surface_provenance_sha256: str
    primary_evidence_sha256: str
    independent_evidence_sha256: str
    primary_success: bool
    independent_success: bool
    agreement: bool
    language_reasons: tuple[str, ...]
    lexical_reason: str | None
    lexical_maximum: float


def _prepare_candidates(
    repository_root: Path,
    config: MatchedDatasetConfig,
    schedule: tuple[MatchedScheduleRecord, ...],
    development_index: DeduplicationIndex,
) -> tuple[_PreparedCandidate, ...]:
    bank = build_template_bank()
    prepared: list[_PreparedCandidate] = []
    for scheduled in schedule:
        original = _generate(
            scheduled.family,
            seed=scheduled.latent_seed,
            difficulty=scheduled.difficulty,
            variant=scheduled.generator_variant,
            output_contract_enabled=scheduled.output_contract_enabled,
        )
        template, plan = _find_plan(bank, scheduled.template_id, scheduled.sentence_plan_id)
        draft = render_with_template(original, template, plan)
        if (
            _latent_hash(draft) != scheduled.latent_program_sha256
            or draft.semantic_ir_sha256 != scheduled.semantic_ir_sha256
            or _mode(draft) != scheduled.mode
            or normalized_text_sha256(draft.rendered_question) != scheduled.rendered_text_sha256
            or hashlib.sha256(draft.rendered_question.encode("utf-8")).hexdigest()
            != scheduled.exact_text_sha256
            or template.render_signature_hash(plan) != scheduled.render_signature_sha256
            or canonical_number_neutral_identity(draft.rendered_question).sha256
            != scheduled.number_neutral_sha256
        ):
            raise ValueError("matched schedule/runtime reconstruction differs")
        provenance = audit_surface_provenance(draft.problem_ir, draft.realization, template)
        language_reasons = tuple(
            dict.fromkeys(
                validate_realization(
                    problem=draft.problem_ir,
                    realization=draft.realization,
                    answer=draft.canonical_final_answer,
                )
                + validate_rendered_candidate(
                    question=draft.rendered_question,
                    completion=draft.training_completion,
                    answer=draft.canonical_final_answer,
                    output_contract_enabled=draft.output_contract_enabled,
                    metadata=draft.quality_metadata,
                )
                + provenance.reasons
            )
        )
        primary, independent, generator_reasons = _verify(draft)
        if generator_reasons:
            language_reasons = tuple(dict.fromkeys(language_reasons + generator_reasons))
        agreement = (
            primary.success
            and independent.success
            and primary.answer == independent.answer == draft.canonical_final_answer
            and primary.verifier_id != independent.verifier_id
            and primary.method_family != independent.method_family
        )
        if draft.output_contract_enabled and not validate_final_answer_contract(
            draft.training_completion, draft.canonical_final_answer
        ):
            language_reasons = tuple(dict.fromkeys(language_reasons + ("output_contract_failure",)))
        lexical = development_index.screen(draft)
        prepared.append(
            _PreparedCandidate(
                scheduled=scheduled,
                draft=draft,
                surface_provenance_sha256=provenance.provenance_sha256,
                primary_evidence_sha256=primary.evidence_sha256,
                independent_evidence_sha256=independent.evidence_sha256,
                primary_success=primary.success,
                independent_success=independent.success,
                agreement=agreement,
                language_reasons=language_reasons,
                lexical_reason=lexical.rejection_reason,
                lexical_maximum=lexical.maximum_ngram_jaccard,
            )
        )
    return tuple(prepared)


def _accepted_caps(
    config: MatchedDatasetConfig,
) -> dict[str, dict[str, MatchedTemplateCaps]]:
    reuse = load_contract(config.template_reuse_config)
    caps: dict[str, dict[str, MatchedTemplateCaps]] = {group: {} for group in GROUP_ORDER}
    for group in GROUP_ORDER:
        for family in CATEGORY_ORDER:
            inventory = reuse.identity_inventory[family]
            caps[group][family] = derive_caps(
                FAMILY_ACCEPTED[group][family],
                compatible_sentence_plans=inventory["sentence_plans"],
                compatible_frames=inventory["semantic_frames"],
                compatible_scenarios=SCENARIO_INVENTORY[family],
            )
    return caps


def _deterministic_attempt(record: MatchedAttemptRecord) -> dict[str, object]:
    return asdict(record)


def _process_prepared(
    prepared: tuple[_PreparedCandidate, ...],
    semantic_scores: tuple[float, ...],
    semantic_outcomes: tuple[str, ...],
    caps: dict[str, dict[str, MatchedTemplateCaps]],
    *,
    progress: bool,
) -> tuple[MatchedAttemptRecord, ...]:
    cell_acceptance: Counter[str] = Counter()
    plan_use: Counter[tuple[str, str, str]] = Counter()
    frame_use: Counter[tuple[str, str, str]] = Counter()
    scenario_use: Counter[tuple[str, str, str]] = Counter()
    number_neutral_use: Counter[tuple[str, str, str]] = Counter()
    results: list[MatchedAttemptRecord] = []
    for index, item in enumerate(prepared):
        scheduled = item.scheduled
        draft = item.draft
        reason: str | None = None
        if item.language_reasons:
            reason = item.language_reasons[0]
        elif not item.primary_success:
            reason = "primary_verifier_failure"
        elif not item.independent_success:
            reason = "independent_verifier_failure"
        elif not item.agreement:
            reason = "verifier_disagreement"
        elif item.lexical_reason is not None:
            reason = f"development_{item.lexical_reason}"
        elif semantic_outcomes[index] == str(ContaminationOutcome.REJECT):
            reason = "development_semantic_rejection"
        elif semantic_outcomes[index] == str(ContaminationOutcome.MANUAL_REVIEW):
            reason = "development_semantic_review_rejected_conservatively"

        cap = caps[scheduled.group][scheduled.family]
        plan_key = f"{scheduled.template_id}:{scheduled.sentence_plan_id}"
        plan_counter = (scheduled.group, scheduled.family, plan_key)
        frame_counter = (scheduled.group, scheduled.family, scheduled.template_id)
        scenario_counter = (scheduled.group, scheduled.family, scheduled.scenario_domain)
        neutral_counter = (scheduled.group, scheduled.family, scheduled.number_neutral_sha256)
        if reason is None and cell_acceptance[scheduled.quota_cell_id] >= (
            scheduled.quota_cell_target
        ):
            reason = "quota_cell_filled"
        if reason is None and (
            plan_use[plan_counter] >= cap.max_sentence_plan_usage
            or frame_use[frame_counter] >= cap.max_frame_usage
            or scenario_use[scenario_counter] >= cap.max_scenario_usage
            or number_neutral_use[neutral_counter] >= cap.max_number_neutral_usage
        ):
            reason = "matched_template_concentration_cap"
        if reason is None:
            cell_acceptance[scheduled.quota_cell_id] += 1
            plan_use[plan_counter] += 1
            frame_use[frame_counter] += 1
            scenario_use[scenario_counter] += 1
            number_neutral_use[neutral_counter] += 1
        results.append(
            MatchedAttemptRecord(
                attempt_index=scheduled.attempt_index,
                synthetic_id=scheduled.synthetic_id,
                group=scheduled.group,
                family=scheduled.family,
                mode=scheduled.mode,
                difficulty=scheduled.difficulty,
                output_contract_enabled=scheduled.output_contract_enabled,
                future_split=scheduled.future_split,
                quota_cell_id=scheduled.quota_cell_id,
                template_id=scheduled.template_id,
                sentence_plan_id=scheduled.sentence_plan_id,
                semantic_frame=scheduled.semantic_frame,
                scenario_domain=scheduled.scenario_domain,
                rendered_question=draft.rendered_question,
                deterministic_solution_trace=draft.deterministic_solution_trace,
                canonical_final_answer=draft.canonical_final_answer.render(),
                training_completion=draft.training_completion,
                latent_program_sha256=scheduled.latent_program_sha256,
                semantic_ir_sha256=scheduled.semantic_ir_sha256,
                rendered_text_sha256=scheduled.rendered_text_sha256,
                exact_text_sha256=scheduled.exact_text_sha256,
                render_signature_sha256=scheduled.render_signature_sha256,
                number_neutral_sha256=scheduled.number_neutral_sha256,
                surface_provenance_sha256=item.surface_provenance_sha256,
                primary_evidence_sha256=item.primary_evidence_sha256,
                independent_evidence_sha256=item.independent_evidence_sha256,
                primary_verifier_success=item.primary_success,
                independent_verifier_success=item.independent_success,
                verifier_agreement=item.agreement,
                deterministic_language_reasons=item.language_reasons,
                benchmark_lexical_reason=item.lexical_reason,
                benchmark_ngram_maximum=item.lexical_maximum,
                benchmark_semantic_outcome=semantic_outcomes[index],
                benchmark_semantic_maximum=semantic_scores[index],
                final_decision="accepted" if reason is None else "rejected",
                rejection_reason=reason,
            )
        )
        processed = index + 1
        if progress and processed in {275, 550, 825, 1100}:
            accepted = sum(record.final_decision == "accepted" for record in results)
            print(
                f"matched dataset progress {processed}/1100: "
                f"accepted={accepted} rejected={processed - accepted}",
                flush=True,
            )
    return tuple(results)


def _counts(records: tuple[MatchedAttemptRecord, ...], *fields: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter["/".join(str(getattr(record, field)) for field in fields)] += 1
    return dict(sorted(counter.items()))


def _dataset_hash(records: tuple[MatchedAttemptRecord, ...]) -> str:
    return canonical_sha256([_deterministic_attempt(item) for item in records])


def run_generation(
    repository_root: Path,
    config_path: Path,
    *,
    progress: bool = False,
) -> dict[str, object]:
    """Run exactly 1,100 scheduled attempts, replay, split, and persist ignored data."""

    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    config = load_matched_dataset_config(config_path)
    schedule_root = json.loads(config.schedule_manifest.read_text(encoding="utf-8"))
    schedule_hash = cast(str, schedule_root["schedule_sha256"])
    schedule = load_schedule(config)
    development = load_development_questions_for_contamination(
        evaluation_config_path=repository_root / config.evaluation_config,
        development_manifest_path=repository_root / config.development_manifest,
    )
    development_index = DeduplicationIndex(development)
    prepared = _prepare_candidates(repository_root, config, schedule, development_index)
    semantic_config = load_semantic_artifact_config(repository_root / config.semantic_config)
    encoder = PinnedSentenceEncoder(semantic_config, repository_root)
    development_embeddings = encoder.encode([item.question for item in development])
    candidate_embeddings = encoder.encode([item.draft.rendered_question for item in prepared])
    similarity = encoder.cosine_matrix(candidate_embeddings, development_embeddings)
    semantic_scores = tuple(float(value) for value in torch.max(similarity, dim=1).values.tolist())
    semantic_outcomes = tuple(
        str(semantic_config.thresholds.classify(value)) for value in semantic_scores
    )
    caps = _accepted_caps(config)
    counted = _process_prepared(
        prepared,
        semantic_scores,
        semantic_outcomes,
        caps,
        progress=progress,
    )
    replay_prepared = _prepare_candidates(repository_root, config, schedule, development_index)
    replay_embeddings = encoder.encode([item.draft.rendered_question for item in replay_prepared])
    replay_similarity = encoder.cosine_matrix(replay_embeddings, development_embeddings)
    replay_scores = tuple(
        float(value) for value in torch.max(replay_similarity, dim=1).values.tolist()
    )
    replay_outcomes = tuple(
        str(semantic_config.thresholds.classify(value)) for value in replay_scores
    )
    replay = _process_prepared(
        replay_prepared,
        replay_scores,
        replay_outcomes,
        caps,
        progress=False,
    )
    counted_hash = _dataset_hash(counted)
    replay_hash = _dataset_hash(replay)
    if counted_hash != replay_hash:
        raise ValueError("matched dataset deterministic reconstruction differs")
    accepted = tuple(item for item in counted if item.final_decision == "accepted")
    accepted_by_group = {
        group: tuple(item for item in accepted if item.group == group) for group in GROUP_ORDER
    }
    gate = (
        len(counted) == 1100
        and all(len(accepted_by_group[group]) == 500 for group in GROUP_ORDER)
        and all(
            sum(item.family == family for item in accepted_by_group[group])
            == FAMILY_ACCEPTED[group][family]
            for group in GROUP_ORDER
            for family in CATEGORY_ORDER
        )
        and all(
            sum(item.output_contract_enabled for item in accepted_by_group[group]) == 100
            for group in GROUP_ORDER
        )
    )
    raw = repository_root / config.raw_directory
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "attempts.jsonl").write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in counted),
        encoding="utf-8",
    )
    (raw / "replay.jsonl").write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in replay),
        encoding="utf-8",
    )
    split_hashes: dict[str, dict[str, str]] = {}
    dataset_hashes: dict[str, str] = {}
    manifest_records: list[dict[str, object]] = []
    for group in GROUP_ORDER:
        group_records = accepted_by_group[group]
        dataset_hashes[group] = _dataset_hash(group_records)
        split_hashes[group] = {}
        for split in SPLIT_ORDER:
            split_records = tuple(item for item in group_records if item.future_split == split)
            expected_count = 450 if split == "training" else 50
            if len(split_records) != expected_count:
                raise ValueError(f"{group}/{split} accepted split differs")
            path = raw / f"{group}_{split}.jsonl"
            path.write_text(
                "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in split_records),
                encoding="utf-8",
            )
            split_hashes[group][split] = _dataset_hash(split_records)
        for item in group_records:
            manifest_records.append(
                {
                    "synthetic_id": item.synthetic_id,
                    "group": item.group,
                    "family": item.family,
                    "mode": item.mode,
                    "difficulty": item.difficulty,
                    "output_contract_enabled": item.output_contract_enabled,
                    "future_split": item.future_split,
                    "rendered_text_sha256": item.rendered_text_sha256,
                    "exact_text_sha256": item.exact_text_sha256,
                    "latent_program_sha256": item.latent_program_sha256,
                    "semantic_ir_sha256": item.semantic_ir_sha256,
                    "template_id": item.template_id,
                    "sentence_plan_id": item.sentence_plan_id,
                    "semantic_frame": item.semantic_frame,
                    "scenario_domain": item.scenario_domain,
                    "number_neutral_sha256": item.number_neutral_sha256,
                    "primary_evidence_sha256": item.primary_evidence_sha256,
                    "independent_evidence_sha256": item.independent_evidence_sha256,
                }
            )
    if len({item["rendered_text_sha256"] for item in manifest_records}) != 1000:
        raise ValueError("accepted normalized questions are not globally unique")
    if len({item["latent_program_sha256"] for item in manifest_records}) != 1000:
        raise ValueError("accepted latent programs are not globally unique")
    if len({item["synthetic_id"] for item in manifest_records}) != 1000:
        raise ValueError("accepted synthetic IDs are not globally unique")
    peak_rss = max(peak_rss, process.memory_info().rss)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "config_sha256": config.config_sha256,
        "schedule_sha256": schedule_hash,
        "dataset_hashes": dataset_hashes,
        "split_hashes": split_hashes,
        "records": manifest_records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    config.dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.dataset_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rejection_reasons = Counter(
        item.rejection_reason for item in counted if item.rejection_reason is not None
    )
    accepted_plan = Counter(
        (item.group, item.family, item.template_id, item.sentence_plan_id) for item in accepted
    )
    accepted_frame = Counter((item.group, item.family, item.template_id) for item in accepted)
    accepted_scenario = Counter(
        (item.group, item.family, item.scenario_domain) for item in accepted
    )
    accepted_neutral = Counter(
        (item.group, item.family, item.number_neutral_sha256) for item in accepted
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "config_sha256": config.config_sha256,
        "schedule_sha256": schedule_hash,
        "attempted": len(counted),
        "accepted": len(accepted),
        "rejected": len(counted) - len(accepted),
        "attempts_by_group_family": _counts(counted, "group", "family"),
        "accepted_by_group_family": _counts(accepted, "group", "family"),
        "accepted_by_group_difficulty": _counts(accepted, "group", "difficulty"),
        "accepted_by_group_output": _counts(accepted, "group", "output_contract_enabled"),
        "accepted_by_group_split": _counts(accepted, "group", "future_split"),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "false_labels": 0,
        "primary_verifier_failures": sum(not item.primary_verifier_success for item in counted),
        "independent_verifier_failures": sum(
            not item.independent_verifier_success for item in counted
        ),
        "verifier_disagreements": sum(not item.verifier_agreement for item in counted),
        "target_mismatches": sum(
            any("target" in reason for reason in item.deterministic_language_reasons)
            for item in counted
        ),
        "deterministic_language_defects": sum(
            bool(item.deterministic_language_reasons) for item in counted
        ),
        "accepted_exact_duplicates": len(accepted)
        - len({item.exact_text_sha256 for item in accepted}),
        "accepted_normalized_duplicates": len(accepted)
        - len({item.rendered_text_sha256 for item in accepted}),
        "accepted_latent_duplicates": len(accepted)
        - len({item.latent_program_sha256 for item in accepted}),
        "cross_dataset_exact_overlap": len(
            {item.rendered_text_sha256 for item in accepted_by_group[GROUP_ORDER[0]]}
            & {item.rendered_text_sha256 for item in accepted_by_group[GROUP_ORDER[1]]}
        ),
        "cross_dataset_latent_overlap": len(
            {item.latent_program_sha256 for item in accepted_by_group[GROUP_ORDER[0]]}
            & {item.latent_program_sha256 for item in accepted_by_group[GROUP_ORDER[1]]}
        ),
        "train_validation_exact_overlap": len(
            {item.rendered_text_sha256 for item in accepted if item.future_split == "training"}
            & {
                item.rendered_text_sha256
                for item in accepted
                if item.future_split == "synthetic_validation"
            }
        ),
        "train_validation_latent_overlap": len(
            {item.latent_program_sha256 for item in accepted if item.future_split == "training"}
            & {
                item.latent_program_sha256
                for item in accepted
                if item.future_split == "synthetic_validation"
            }
        ),
        "benchmark_lexical_rejections": sum(
            item.benchmark_lexical_reason is not None for item in counted
        ),
        "benchmark_semantic_rejections": sum(
            item.benchmark_semantic_outcome == str(ContaminationOutcome.REJECT) for item in counted
        ),
        "benchmark_semantic_manual_rejections": sum(
            item.benchmark_semantic_outcome == str(ContaminationOutcome.MANUAL_REVIEW)
            for item in counted
        ),
        "unresolved_contamination_cases": 0,
        "maximum_sentence_plan_usage": max(accepted_plan.values()),
        "maximum_frame_usage": max(accepted_frame.values()),
        "maximum_scenario_usage": max(accepted_scenario.values()),
        "maximum_number_neutral_usage": max(accepted_neutral.values()),
        "unique_number_neutral_identities": len({item.number_neutral_sha256 for item in accepted}),
        "dataset_hashes": dataset_hashes,
        "split_hashes": split_hashes,
        "manifest_sha256": manifest["manifest_sha256"],
        "counted_decision_sha256": counted_hash,
        "replay_decision_sha256": replay_hash,
        "deterministic_reconstruction_match": counted_hash == replay_hash,
        "runtime_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "raw_artifact_bytes": sum(path.stat().st_size for path in raw.rglob("*") if path.is_file()),
        "dataset_generation_gate_passed": gate,
        "sealed_final_accessed": False,
    }
    summary["summary_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "runtime_seconds"}
    )
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
