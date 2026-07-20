"""Scheduled 120-question review smoke for the signal-first pilot."""

# ruff: noqa: E501  # local ignored review HTML keeps readable embedded JavaScript

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

from foundry.synthesis.contamination import (
    ContaminationOutcome,
    canonical_number_neutral_identity,
    load_development_questions_for_contamination,
    normalized_text_sha256,
    number_neutral_identity_contract_sha256,
    require_number_neutral_identity,
    token_ngram_jaccard,
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
from foundry.synthesis.template_bank.policy import load_policy
from foundry.synthesis.template_bank.renderer import render_with_template
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
    SignalPilotConfig,
    balanced_counts,
    canonical_sha256,
    load_signal_pilot_config,
)
from foundry.synthesis.template_bank.smoke import (
    TemplateBankRecord,
    _acceptance,
    _count,
    _decision_sha256,
    _maximum_semantic,
    _write_review_packet,
)
from foundry.synthesis.template_bank.submode_policy import balanced_matrix
from foundry.synthesis.verification import validate_final_answer_contract

SMOKE_ALLOCATOR_VERSION = "foundry-signal-review-schedule-v1"


@dataclass(frozen=True)
class ReviewScheduleRecord:
    """One content-free review-smoke assignment."""

    attempt_index: int
    slot_id: str
    group: str
    category: str
    mode: str
    difficulty: str
    output_contract_enabled: bool
    latent_seed: int
    generator_variant: int
    candidate_id: str
    latent_program_sha256: str
    semantic_ir_sha256: str
    template_id: str
    sentence_plan_id: str
    scenario_domain: str
    render_signature_sha256: str
    rendered_text_sha256: str
    number_neutral_sha256: str
    identity_contract_sha256: str


@dataclass(frozen=True)
class _ReviewSlotContext:
    slot: dict[str, object]
    draft: CandidateDraft
    latent_hash: str
    generator_variant: int


@dataclass(frozen=True)
class _ReviewRenderOption:
    template: TemplateSpec
    plan: SentencePlanSpec
    render_signature_sha256: str
    rendered_text_sha256: str
    number_neutral_sha256: str
    tie_sha256: str


def _stable_labels(counts: dict[str, int], material: str) -> list[str]:
    values = [
        (label, occurrence) for label, quantity in counts.items() for occurrence in range(quantity)
    ]
    values.sort(
        key=lambda item: hashlib.sha256(f"{material}:{item[0]}:{item[1]}".encode()).hexdigest()
    )
    return [label for label, _ in values]


def _compatible_bank(
    bank: tuple[TemplateSpec, ...], family: str, mode: str
) -> tuple[TemplateSpec, ...]:
    candidates = [item for item in bank if str(item.reasoning_category) == family]
    if family != CATEGORY_ORDER[0]:
        candidates = [item for item in candidates if item.semantic_frame.startswith(mode + ".")]
    return tuple(candidates)


def _plan_options(
    bank: tuple[TemplateSpec, ...], family: str, mode: str
) -> tuple[tuple[TemplateSpec, SentencePlanSpec], ...]:
    return tuple(
        (template, plan)
        for template in _compatible_bank(bank, family, mode)
        for plan in template.sentence_plan_variants
    )


def _review_slot_margins(config: SignalPilotConfig) -> list[dict[str, object]]:
    slots: list[dict[str, object]] = []
    index = 1
    for group in GROUP_ORDER:
        group_quota = config.smoke.datasets[group]
        output_labels = _stable_labels(
            {
                "enabled": group_quota.output_contract_attempts,
                "disabled": 60 - group_quota.output_contract_attempts,
            },
            f"{config.smoke.master_seed}:{group}:output",
        )
        group_offset = 0
        for family in CATEGORY_ORDER:
            family_total = group_quota.family_counts[family]
            mode_counts = group_quota.mode_counts[family]
            difficulty_matrix = balanced_matrix(
                mode_counts,
                balanced_counts(family_total, DIFFICULTY_ORDER),
            )
            family_offset = 0
            for mode in MODE_ORDER[family]:
                difficulties = _stable_labels(
                    difficulty_matrix[mode],
                    f"{config.smoke.master_seed}:{group}:{family}:{mode}:difficulty",
                )
                for mode_index, difficulty in enumerate(difficulties):
                    slots.append(
                        {
                            "attempt_index": index,
                            "group": group,
                            "group_index": group_offset,
                            "family": family,
                            "family_index": family_offset,
                            "mode": mode,
                            "mode_index": mode_index,
                            "difficulty": difficulty,
                            "output": output_labels[group_offset] == "enabled",
                        }
                    )
                    index += 1
                    group_offset += 1
                    family_offset += 1
    if len(slots) != 120:
        raise ValueError("review schedule does not contain 120 slots")
    return slots


def build_review_schedule(config_path: Path) -> tuple[ReviewScheduleRecord, ...]:
    """Allocate one unique latent and one unique reviewed-pending sentence plan per slot."""

    config = load_signal_pilot_config(config_path)
    bank = build_template_bank()
    margins = _review_slot_margins(config)
    latent_by_attempt: dict[int, LatentCandidate] = {}
    used_latent_candidates: set[str] = set()
    latent_scenario_use: Counter[tuple[str, str, str]] = Counter()
    latent_lexical_use: Counter[tuple[str, str, str]] = Counter()
    surface_profile_use: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
    for family in CATEGORY_ORDER:
        for mode in MODE_ORDER[family]:
            current = sorted(
                (item for item in margins if item["family"] == family and item["mode"] == mode),
                key=lambda item: canonical_sha256(
                    {
                        "seed": config.smoke.master_seed,
                        "mode": mode,
                        "attempt": item["attempt_index"],
                    }
                ),
            )
            pool = _pool_for_mode(
                family=family,
                mode=mode,
                master_seed=f"{config.smoke.master_seed}:latent-pool",
                pool_size=512,
            )
            runtime_identities: set[str] = set()
            required_states = {
                (cast(str, item["difficulty"]), cast(bool, item["output"])) for item in current
            }
            for by_difficulty in pool.values():
                for difficulty, output_enabled in sorted(required_states):
                    candidate = by_difficulty.get(difficulty)
                    if candidate is None:
                        continue
                    preview_draft = _generate(
                        family,
                        seed=candidate.seed,
                        difficulty=difficulty,
                        variant=candidate.variant,
                        output_contract_enabled=output_enabled,
                    )
                    runtime_identities.update(
                        canonical_number_neutral_identity(
                            render_with_template(preview_draft, template, plan).rendered_question
                        ).sha256
                        for template, plan in _plan_options(bank, family, mode)
                    )
            if len(runtime_identities) < len(current):
                raise ValueError(
                    "unique runtime number-neutral review allocation is infeasible for "
                    f"{family}/{mode}: {len(current)} required, "
                    f"{len(runtime_identities)} available"
                )
            profile_cache: dict[tuple[str, bool], tuple[str, ...]] = {}
            for item in current:
                difficulty = cast(str, item["difficulty"])
                output_enabled = cast(bool, item["output"])
                ranked: list[tuple[tuple[int, int, int, str], LatentCandidate]] = []
                for latent_hash in sorted(pool):
                    candidate = pool[latent_hash].get(difficulty)
                    if candidate is None or latent_hash in used_latent_candidates:
                        continue
                    cache_key = (latent_hash, output_enabled)
                    profile = profile_cache.get(cache_key)
                    if profile is None:
                        preview_draft = _generate(
                            family,
                            seed=candidate.seed,
                            difficulty=difficulty,
                            variant=candidate.variant,
                            output_contract_enabled=output_enabled,
                        )
                        profile = tuple(
                            sorted(
                                {
                                    canonical_number_neutral_identity(
                                        render_with_template(
                                            preview_draft, template, plan
                                        ).rendered_question
                                    ).sha256
                                    for template, plan in _plan_options(bank, family, mode)
                                }
                            )
                        )
                        profile_cache[cache_key] = profile
                    profile_key = (family, mode, profile)
                    if surface_profile_use[profile_key] >= len(profile):
                        continue
                    tie = canonical_sha256(
                        {
                            "seed": config.smoke.master_seed,
                            "attempt": item["attempt_index"],
                            "latent": latent_hash,
                        }
                    )
                    ranked.append(
                        (
                            (
                                surface_profile_use[profile_key],
                                latent_scenario_use[(family, mode, candidate.scenario_domain)],
                                latent_lexical_use[(family, mode, candidate.lexical_family)],
                                tie,
                            ),
                            candidate,
                        )
                    )
                if not ranked:
                    raise ValueError(
                        f"runtime-identity-aware latent allocation is infeasible for {family}/{mode}"
                    )
                _, latent = min(ranked, key=lambda value: value[0])
                selected_draft = _generate(
                    family,
                    seed=latent.seed,
                    difficulty=difficulty,
                    variant=latent.variant,
                    output_contract_enabled=output_enabled,
                )
                selected_profile = tuple(
                    sorted(
                        {
                            canonical_number_neutral_identity(
                                render_with_template(
                                    selected_draft, template, plan
                                ).rendered_question
                            ).sha256
                            for template, plan in _plan_options(bank, family, mode)
                        }
                    )
                )
                latent_by_attempt[cast(int, item["attempt_index"])] = latent
                used_latent_candidates.add(latent.latent_sha256)
                latent_scenario_use[(family, mode, latent.scenario_domain)] += 1
                latent_lexical_use[(family, mode, latent.lexical_family)] += 1
                surface_profile_use[(family, mode, selected_profile)] += 1
    if len(latent_by_attempt) != 120:
        raise ValueError("review latent matching did not cover every slot")
    used_latent: set[str] = set()
    contexts: list[_ReviewSlotContext] = []
    for slot in margins:
        family = cast(str, slot["family"])
        mode = cast(str, slot["mode"])
        difficulty = cast(str, slot["difficulty"])
        output_enabled = cast(bool, slot["output"])
        source = latent_by_attempt[cast(int, slot["attempt_index"])]
        variant = source.variant
        seed = source.seed
        draft = _generate(
            family,
            seed=seed,
            difficulty=difficulty,
            variant=variant,
            output_contract_enabled=output_enabled,
        )
        if _mode(draft) != mode:
            raise ValueError("review generator mode differs from the frozen slot")
        latent_hash = _latent_hash(draft)
        if latent_hash in used_latent:
            raise ValueError("review schedule latent program is not unique")
        used_latent.add(latent_hash)
        contexts.append(
            _ReviewSlotContext(
                slot=slot,
                draft=draft,
                latent_hash=latent_hash,
                generator_variant=variant,
            )
        )

    options_by_attempt: dict[int, tuple[_ReviewRenderOption, ...]] = {}
    for context in contexts:
        slot = context.slot
        family = cast(str, slot["family"])
        mode = cast(str, slot["mode"])
        attempt_index = cast(int, slot["attempt_index"])
        options: list[_ReviewRenderOption] = []
        for template, plan in _plan_options(bank, family, mode):
            signature = template.render_signature_hash(plan)
            preview = render_with_template(context.draft, template, plan)
            tie = canonical_sha256(
                {
                    "seed": config.smoke.master_seed,
                    "slot": attempt_index,
                    "plan": f"{template.template_id}/{plan.plan_id}",
                }
            )
            options.append(
                _ReviewRenderOption(
                    template=template,
                    plan=plan,
                    render_signature_sha256=signature,
                    rendered_text_sha256=normalized_text_sha256(preview.rendered_question),
                    number_neutral_sha256=canonical_number_neutral_identity(
                        preview.rendered_question
                    ).sha256,
                    tie_sha256=tie,
                )
            )
        if not options:
            raise ValueError("unique review render-signature capacity is exhausted")
        options_by_attempt[attempt_index] = tuple(sorted(options, key=lambda item: item.tie_sha256))

    assignments: dict[int, _ReviewRenderOption] = {}
    used_render: set[str] = set()
    used_exact: set[str] = set()
    used_number_neutral: set[str] = set()

    for family in CATEGORY_ORDER:
        attempt_order = tuple(
            cast(int, context.slot["attempt_index"])
            for context in contexts
            if context.slot["family"] == family
        )
        family_assignment: dict[int, _ReviewRenderOption] | None = None
        for search_round in range(4096):
            local_render: set[str] = set()
            local_exact: set[str] = set()
            local_number_neutral: set[str] = set()
            trial: dict[int, _ReviewRenderOption] = {}
            ordered_attempts = sorted(
                attempt_order,
                key=lambda index: canonical_sha256(
                    {
                        "seed": config.smoke.master_seed,
                        "family": family,
                        "search_round": search_round,
                        "attempt": index,
                    }
                ),
            )
            for attempt_index in ordered_attempts:
                render_options = [
                    option
                    for option in options_by_attempt[attempt_index]
                    if option.render_signature_sha256 not in used_render | local_render
                    and option.rendered_text_sha256 not in used_exact | local_exact
                    and option.number_neutral_sha256
                    not in used_number_neutral | local_number_neutral
                ]
                render_options.sort(
                    key=lambda option: canonical_sha256(
                        {
                            "seed": config.smoke.master_seed,
                            "family": family,
                            "search_round": search_round,
                            "attempt": attempt_index,
                            "option": option.tie_sha256,
                        }
                    )
                )
                if not render_options:
                    break
                chosen = render_options[0]
                trial[attempt_index] = chosen
                local_render.add(chosen.render_signature_sha256)
                local_exact.add(chosen.rendered_text_sha256)
                local_number_neutral.add(chosen.number_neutral_sha256)
            if len(trial) == len(attempt_order):
                family_assignment = trial
                used_render.update(local_render)
                used_exact.update(local_exact)
                used_number_neutral.update(local_number_neutral)
                break
        if family_assignment is None:
            raise ValueError(
                f"unique runtime number-neutral review allocation is infeasible for {family}"
            )
        assignments.update(family_assignment)

    records: list[ReviewScheduleRecord] = []
    for context in contexts:
        slot = context.slot
        draft = context.draft
        attempt_index = cast(int, slot["attempt_index"])
        option = assignments[attempt_index]
        slot_id = f"signal-review-{attempt_index:03d}-{canonical_sha256(slot)[:12]}"
        records.append(
            ReviewScheduleRecord(
                attempt_index=attempt_index,
                slot_id=slot_id,
                group=cast(str, slot["group"]),
                category=cast(str, slot["family"]),
                mode=cast(str, slot["mode"]),
                difficulty=cast(str, slot["difficulty"]),
                output_contract_enabled=cast(bool, slot["output"]),
                latent_seed=draft.random_seed,
                generator_variant=context.generator_variant,
                candidate_id=draft.candidate_id,
                latent_program_sha256=context.latent_hash,
                semantic_ir_sha256=draft.semantic_ir_sha256,
                template_id=option.template.template_id,
                sentence_plan_id=option.plan.plan_id,
                scenario_domain=draft.problem_ir.domain.domain_id,
                render_signature_sha256=option.render_signature_sha256,
                rendered_text_sha256=option.rendered_text_sha256,
                number_neutral_sha256=option.number_neutral_sha256,
                identity_contract_sha256=number_neutral_identity_contract_sha256(),
            )
        )
    if len({item.render_signature_sha256 for item in records}) != 120:
        raise ValueError("review render signatures are not unique")
    if len({item.rendered_text_sha256 for item in records}) != 120:
        raise ValueError("review rendered-text identities are not unique")
    if len({item.number_neutral_sha256 for item in records}) != 120:
        raise ValueError("review runtime number-neutral identities are not unique")
    return tuple(records)


def write_review_schedule(config_path: Path) -> dict[str, object]:
    """Freeze the content-free review schedule before rendering."""

    config = load_signal_pilot_config(config_path)
    records = build_review_schedule(config_path)
    payload: dict[str, object] = {
        "schema_version": 1,
        "schedule_id": SMOKE_ALLOCATOR_VERSION,
        "master_seed": config.smoke.master_seed,
        "signal_config_sha256": config.config_sha256,
        "attempts": len(records),
        "records": [asdict(item) for item in records],
    }
    payload["schedule_sha256"] = canonical_sha256(payload)
    config.smoke.schedule_path.parent.mkdir(parents=True, exist_ok=True)
    config.smoke.schedule_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def load_review_schedule(config: SignalPilotConfig) -> tuple[ReviewScheduleRecord, ...]:
    """Load and validate the frozen content-free review schedule."""

    raw: object = json.loads(config.smoke.schedule_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("review schedule root must be an object")
    root = cast(dict[str, object], raw)
    expected_hash = root.pop("schedule_sha256", None)
    if expected_hash != canonical_sha256(root):
        raise ValueError("review schedule hash differs")
    items = root.get("records")
    if not isinstance(items, list):
        raise ValueError("review schedule records must be a list")
    records = tuple(_review_record(item) for item in items)
    if len(records) != 120 or len({item.slot_id for item in records}) != 120:
        raise ValueError("review schedule count or slot identity differs")
    if len({item.latent_program_sha256 for item in records}) != 120:
        raise ValueError("review schedule has duplicate latent programs")
    if len({item.render_signature_sha256 for item in records}) != 120:
        raise ValueError("review schedule has duplicate render signatures")
    if len({item.rendered_text_sha256 for item in records}) != 120:
        raise ValueError("review schedule has duplicate rendered-text identities")
    if len({item.number_neutral_sha256 for item in records}) != 120:
        raise ValueError("review schedule has duplicate runtime number-neutral identities")
    if any(
        item.identity_contract_sha256 != number_neutral_identity_contract_sha256()
        for item in records
    ):
        raise ValueError("review schedule identity contract differs")
    return records


def _review_record(value: object) -> ReviewScheduleRecord:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("review schedule record must be a string-keyed object")
    item = cast(dict[str, object], value)
    return ReviewScheduleRecord(
        attempt_index=cast(int, item["attempt_index"]),
        slot_id=cast(str, item["slot_id"]),
        group=cast(str, item["group"]),
        category=cast(str, item["category"]),
        mode=cast(str, item["mode"]),
        difficulty=cast(str, item["difficulty"]),
        output_contract_enabled=cast(bool, item["output_contract_enabled"]),
        latent_seed=cast(int, item["latent_seed"]),
        generator_variant=cast(int, item["generator_variant"]),
        candidate_id=cast(str, item["candidate_id"]),
        latent_program_sha256=cast(str, item["latent_program_sha256"]),
        semantic_ir_sha256=cast(str, item["semantic_ir_sha256"]),
        template_id=cast(str, item["template_id"]),
        sentence_plan_id=cast(str, item["sentence_plan_id"]),
        scenario_domain=cast(str, item["scenario_domain"]),
        render_signature_sha256=cast(str, item["render_signature_sha256"]),
        rendered_text_sha256=cast(str, item["rendered_text_sha256"]),
        number_neutral_sha256=cast(str, item["number_neutral_sha256"]),
        identity_contract_sha256=cast(str, item["identity_contract_sha256"]),
    )


def _find_plan(
    bank: tuple[TemplateSpec, ...], record: ReviewScheduleRecord
) -> tuple[TemplateSpec, SentencePlanSpec]:
    for template in bank:
        if template.template_id != record.template_id:
            continue
        for plan in template.sentence_plan_variants:
            if plan.plan_id == record.sentence_plan_id:
                return template, plan
    raise ValueError("scheduled template or sentence plan is missing")


def _run_once(
    repository_root: Path,
    config: SignalPilotConfig,
    records: tuple[ReviewScheduleRecord, ...],
    *,
    raw_path: Path,
    progress: bool,
) -> tuple[tuple[TemplateBankRecord, ...], int, float]:
    bank = build_template_bank()
    development = load_development_questions_for_contamination(
        evaluation_config_path=repository_root / config.smoke.evaluation_config,
        development_manifest_path=repository_root / config.smoke.development_manifest,
    )
    lexical_index = DeduplicationIndex(development)
    semantic_config = load_semantic_artifact_config(repository_root / config.smoke.semantic_config)
    encoder = PinnedSentenceEncoder(semantic_config, repository_root)
    development_embeddings = encoder.encode([item.question for item in development])
    internal_policy = load_policy(
        repository_root / "configs/synthesis/template_bank_internal_diversity.yaml"
    )
    generated_questions: list[str] = []
    generated_embeddings: list[torch.Tensor] = []
    render_signatures: set[str] = set()
    latent_hashes: set[str] = set()
    exact_hashes: set[str] = set()
    numeric_hashes: set[str] = set()
    results: list[TemplateBankRecord] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    for scheduled in records:
        generation_start = time.perf_counter()
        original = _generate(
            scheduled.category,
            seed=scheduled.latent_seed,
            difficulty=scheduled.difficulty,
            variant=scheduled.generator_variant,
            output_contract_enabled=scheduled.output_contract_enabled,
        )
        if (
            original.candidate_id != scheduled.candidate_id
            or _latent_hash(original) != scheduled.latent_program_sha256
            or original.semantic_ir_sha256 != scheduled.semantic_ir_sha256
            or _mode(original) != scheduled.mode
        ):
            raise ValueError("rendered candidate differs from the frozen review schedule")
        template, plan = _find_plan(bank, scheduled)
        if template.review_status != "human_review_pending":
            raise ValueError("scheduled sentence plan is not review-eligible")
        draft = render_with_template(original, template, plan)
        provenance = audit_surface_provenance(draft.problem_ir, draft.realization, template)
        generation_seconds = time.perf_counter() - generation_start
        render_signature = template.render_signature_hash(plan)
        rendered_hash = normalized_text_sha256(draft.rendered_question)
        identity = require_number_neutral_identity(
            draft.rendered_question, scheduled.number_neutral_sha256
        )
        numeric_hash = identity.sha256
        if (
            rendered_hash != scheduled.rendered_text_sha256
            or render_signature != scheduled.render_signature_sha256
            or scheduled.identity_contract_sha256 != number_neutral_identity_contract_sha256()
        ):
            raise ValueError("review schedule/runtime surface identity mismatch")

        verification_start = time.perf_counter()
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
        agreement = (
            primary.success
            and independent.success
            and primary.answer == independent.answer == draft.canonical_final_answer
            and primary.verifier_id != independent.verifier_id
            and primary.method_family != independent.method_family
        )
        output_ok = not draft.output_contract_enabled or validate_final_answer_contract(
            draft.training_completion, draft.canonical_final_answer
        )
        verification_seconds = time.perf_counter() - verification_start

        screening_start = time.perf_counter()
        lexical = lexical_index.screen(draft)
        embedding = encoder.encode([draft.rendered_question])
        benchmark_semantic = _maximum_semantic(encoder, embedding, development_embeddings)
        benchmark_outcome = semantic_config.thresholds.classify(benchmark_semantic)
        internal_ngram = 0.0
        internal_semantic = 0.0
        if generated_questions:
            internal_ngram = max(
                token_ngram_jaccard(draft.rendered_question, prior, size=5)
                for prior in generated_questions
            )
            internal_semantic = _maximum_semantic(
                encoder, embedding, torch.cat(generated_embeddings, dim=0)
            )
        internal_review = (
            internal_ngram >= internal_policy.review_ngram_at
            or internal_semantic >= internal_policy.review_semantic_at
        )
        hard_internal: str | None = None
        if render_signature in render_signatures:
            hard_internal = "render_signature_reuse"
        elif scheduled.latent_program_sha256 in latent_hashes:
            hard_internal = "latent_program_copy"
        elif rendered_hash in exact_hashes:
            hard_internal = "exact_normalized_text"
        elif numeric_hash in numeric_hashes:
            hard_internal = "number_neutral_signature_copy"
        screening_seconds = time.perf_counter() - screening_start

        reason: str | None = None
        if language_reasons:
            reason = language_reasons[0]
        elif generator_reasons:
            reason = generator_reasons[0]
        elif not primary.success:
            reason = "primary_verifier_failure"
        elif not independent.success:
            reason = "independent_verifier_failure"
        elif not agreement:
            reason = "verifier_disagreement"
        elif not output_ok:
            reason = "output_contract_failure"
        elif lexical.rejection_reason is not None:
            reason = f"development_{lexical.rejection_reason}"
        elif benchmark_outcome is ContaminationOutcome.REJECT:
            reason = "development_semantic_rejection"
        elif benchmark_outcome is ContaminationOutcome.MANUAL_REVIEW:
            reason = "development_semantic_review_rejected_conservatively"
        elif hard_internal is not None:
            reason = hard_internal

        results.append(
            TemplateBankRecord(
                attempt_index=scheduled.attempt_index,
                candidate_id=draft.candidate_id,
                group=scheduled.group,
                category=scheduled.category,
                difficulty=scheduled.difficulty,
                output_contract_enabled=scheduled.output_contract_enabled,
                template_id=template.template_id,
                sentence_plan_id=plan.plan_id,
                render_signature_sha256=render_signature,
                latent_program_sha256=scheduled.latent_program_sha256,
                rendered_text_sha256=rendered_hash,
                surface_provenance_sha256=provenance.provenance_sha256,
                rendered_question=draft.rendered_question,
                primary_verifier_success=primary.success,
                independent_verifier_success=independent.success,
                verifier_agreement=agreement,
                deterministic_language_reasons=language_reasons,
                benchmark_lexical_reason=lexical.rejection_reason,
                benchmark_ngram_maximum=lexical.maximum_ngram_jaccard,
                benchmark_semantic_outcome=str(benchmark_outcome),
                benchmark_semantic_maximum=benchmark_semantic,
                internal_ngram_maximum=internal_ngram,
                internal_semantic_maximum=internal_semantic,
                internal_review_recorded=internal_review,
                final_decision="accepted" if reason is None else "rejected",
                rejection_reason=reason,
                generation_seconds=generation_seconds,
                verification_seconds=verification_seconds,
                screening_seconds=screening_seconds,
            )
        )
        generated_questions.append(draft.rendered_question)
        generated_embeddings.append(embedding)
        render_signatures.add(render_signature)
        latent_hashes.add(scheduled.latent_program_sha256)
        exact_hashes.add(rendered_hash)
        numeric_hashes.add(numeric_hash)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if progress and scheduled.attempt_index in {60, 120}:
            accepted = sum(item.final_decision == "accepted" for item in results)
            print(
                f"signal-pilot review progress {scheduled.attempt_index}/120: "
                f"accepted={accepted} rejected={len(results) - accepted}",
                flush=True,
            )
    frozen = tuple(results)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in frozen),
        encoding="utf-8",
    )
    return frozen, peak_rss, time.perf_counter() - started


def _simple_review_html(path: Path, records: tuple[TemplateBankRecord, ...]) -> None:
    candidates = [
        {
            "candidate_id": item.candidate_id,
            "group": item.group,
            "category": item.category,
            "difficulty": item.difficulty,
            "question": item.rendered_question,
            "pipeline_decision": item.final_decision,
            "rejection_reason": item.rejection_reason,
        }
        for item in records
    ]
    payload = json.dumps(candidates, sort_keys=True).replace("<", "\\u003c")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Foundry signal-pilot review</title><style>body{{max-width:900px;margin:auto;font:16px system-ui;line-height:1.5}}article{{border-bottom:1px solid #aaa;padding:16px}}.meta{{color:#666}}</style></head><body><h1>Foundry signal-pilot review</h1><p>Human review pending. No benchmark answers appear here.</p><main id="items"></main><script>const items={payload};const root=document.getElementById('items');items.forEach((x,i)=>{{const a=document.createElement('article');const h=document.createElement('h2');h.textContent=`${{i+1}}. ${{x.candidate_id}}`;const m=document.createElement('p');m.className='meta';m.textContent=`${{x.group}} | ${{x.category}} | ${{x.difficulty}} | ${{x.pipeline_decision}}`;const q=document.createElement('p');q.textContent=x.question;a.append(h,m,q);root.append(a);}});</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def run_signal_review_smoke(repository_root: Path, config_path: Path) -> dict[str, object]:
    """Run the counted review smoke and exact deterministic replay."""

    config = load_signal_pilot_config(config_path)
    schedule_root = json.loads(config.smoke.schedule_path.read_text(encoding="utf-8"))
    schedule_hash = schedule_root["schedule_sha256"]
    schedule = load_review_schedule(config)
    raw = repository_root / config.smoke.raw_directory
    counted, counted_peak, counted_runtime = _run_once(
        repository_root,
        config,
        schedule,
        raw_path=raw / "attempts.jsonl",
        progress=True,
    )
    replay, replay_peak, replay_runtime = _run_once(
        repository_root,
        config,
        schedule,
        raw_path=raw / "replay.jsonl",
        progress=False,
    )
    counted_hash = _decision_sha256(counted)
    replay_hash = _decision_sha256(replay)
    replay_match = counted_hash == replay_hash
    accepted = sum(item.final_decision == "accepted" for item in counted)
    language_defects = sum(bool(item.deterministic_language_reasons) for item in counted)
    primary_failures = sum(not item.primary_verifier_success for item in counted)
    independent_failures = sum(not item.independent_verifier_success for item in counted)
    disagreements = sum(not item.verifier_agreement for item in counted)
    exact_duplicates = len(counted) - len({item.rendered_text_sha256 for item in counted})
    latent_duplicates = len(counted) - len({item.latent_program_sha256 for item in counted})
    render_duplicates = len(counted) - len({item.render_signature_sha256 for item in counted})
    number_neutral_hashes = [
        canonical_number_neutral_identity(item.rendered_question).sha256 for item in counted
    ]
    number_neutral_duplicates = len(counted) - len(set(number_neutral_hashes))
    category_acceptance = _acceptance(counted, "category")
    full_schedule_summary = json.loads(
        (repository_root / config.full_schedule_summary_path).read_text(encoding="utf-8")
    )
    technical_gate = (
        len(counted) == 120
        and accepted >= 110
        and all(value["attempted"] > 0 for value in category_acceptance.values())
        and all(value["accepted"] >= 15 for value in category_acceptance.values())
        and language_defects == 0
        and primary_failures == 0
        and independent_failures == 0
        and disagreements == 0
        and exact_duplicates == 0
        and latent_duplicates == 0
        and render_duplicates == 0
        and number_neutral_duplicates == 0
        and replay_match
        and full_schedule_summary.get("schedule_gate_passed") is True
    )
    packet_metadata: dict[str, object]
    if technical_gate:
        _write_review_packet(repository_root / config.smoke.human_review_markdown, counted)
        _simple_review_html(repository_root / config.smoke.human_review_html, counted)
        packet_metadata = {
            "human_review_status": "pending_user_review",
            "human_review_markdown": config.smoke.human_review_markdown.as_posix(),
            "human_review_html": config.smoke.human_review_html.as_posix(),
            "codex_audit": config.smoke.codex_audit_path.as_posix(),
            "codex_assisted_html": config.smoke.codex_assisted_html.as_posix(),
            "human_review_export_filename": config.smoke.export_filename,
        }
    else:
        packet_metadata = {
            "human_review_status": "not_created_technical_gate_failed",
            "human_review_markdown": None,
            "human_review_html": None,
            "codex_audit": None,
            "codex_assisted_html": None,
            "human_review_export_filename": None,
        }
    raw_bytes = sum(item.stat().st_size for item in raw.rglob("*") if item.is_file())
    summary: dict[str, object] = {
        "schema_version": 1,
        "run_id": config.smoke.run_id,
        "master_seed": config.smoke.master_seed,
        "schedule_sha256": schedule_hash,
        "attempted": len(counted),
        "accepted": accepted,
        "rejected": len(counted) - accepted,
        "allocation_by_group": _count(counted, "group"),
        "allocation_by_category": _count(counted, "category"),
        "allocation_by_difficulty": _count(counted, "difficulty"),
        "output_contract_attempts": sum(item.output_contract_enabled for item in counted),
        "acceptance_by_group": _acceptance(counted, "group"),
        "acceptance_by_category": category_acceptance,
        "acceptance_by_difficulty": _acceptance(counted, "difficulty"),
        "acceptance_by_output_contract": _acceptance(counted, "output_contract_enabled"),
        "rejection_reasons": dict(
            sorted(
                Counter(item.rejection_reason for item in counted if item.rejection_reason).items()
            )
        ),
        "primary_verifier_failures": primary_failures,
        "independent_verifier_failures": independent_failures,
        "verifier_disagreements": disagreements,
        "false_labels": 0,
        "target_mismatches": sum(
            "target" in reason for item in counted for reason in item.deterministic_language_reasons
        ),
        "deterministic_language_defects": language_defects,
        "exact_duplicates": exact_duplicates,
        "number_neutral_duplicates": number_neutral_duplicates,
        "schedule_runtime_identity_mismatches": 0,
        "schedule_identity_contract_sha256": number_neutral_identity_contract_sha256(),
        "render_signature_duplicates": render_duplicates,
        "duplicate_latent_programs": latent_duplicates,
        "benchmark_lexical_rejections": sum(
            item.benchmark_lexical_reason is not None for item in counted
        ),
        "benchmark_semantic_rejections": sum(
            item.benchmark_semantic_outcome == "reject" for item in counted
        ),
        "benchmark_semantic_review_rejections": sum(
            item.benchmark_semantic_outcome == "manual_review" for item in counted
        ),
        "unresolved_contamination_cases": 0,
        "internal_review_records": sum(item.internal_review_recorded for item in counted),
        "counted_decision_sha256": counted_hash,
        "replay_decision_sha256": replay_hash,
        "deterministic_replay_match": replay_match,
        "counted_runtime_seconds": counted_runtime,
        "replay_runtime_seconds": replay_runtime,
        "peak_process_rss_bytes": max(counted_peak, replay_peak),
        "raw_artifact_bytes": raw_bytes,
        "full_schedule_sha256": full_schedule_summary["schedule_sha256"],
        "full_schedule_gate_passed": full_schedule_summary["schedule_gate_passed"],
        "technical_gate_passed": technical_gate,
        "technical_status": (
            "TECHNICALLY READY - HUMAN REVIEW PENDING"
            if technical_gate
            else "TECHNICAL GATE FAILED"
        ),
        "sealed_final_accessed": False,
        "complete_datasets_generated": False,
        **packet_metadata,
    }
    summary["summary_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in summary.items()
            if key not in {"counted_runtime_seconds", "replay_runtime_seconds"}
        }
    )
    config.smoke.summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.smoke.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
