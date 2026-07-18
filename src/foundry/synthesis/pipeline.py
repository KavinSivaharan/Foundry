"""Bounded deterministic synthesis smoke pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import psutil  # type: ignore[import-untyped]
import torch
import yaml

from foundry.synthesis.contamination import (
    ContaminationOutcome,
    DevelopmentQuestion,
    load_development_questions_for_contamination,
    normalized_text_sha256,
)
from foundry.synthesis.deduplication import DeduplicationIndex, LexicalScreenResult
from foundry.synthesis.generators import CandidateDraft, GeneratorVerification
from foundry.synthesis.generators.bookkeeping import (
    generate_bookkeeping,
    validate_bookkeeping_constraints,
    verify_bookkeeping_dag,
    verify_bookkeeping_ledger,
)
from foundry.synthesis.generators.discrete import (
    generate_discrete,
    validate_discrete_constraints,
    verify_discrete_constructive,
    verify_discrete_enumeration,
)
from foundry.synthesis.generators.rates import (
    generate_rates,
    validate_rate_constraints,
    verify_rate_equation,
    verify_rate_inverse,
)
from foundry.synthesis.realization import validate_realization
from foundry.synthesis.schema import (
    DifficultyLevel,
    ProvenanceMetadata,
    ReviewStatus,
    SyntheticExample,
    ValidityStatus,
    VerificationEvidence,
)
from foundry.synthesis.semantic import (
    PinnedSentenceEncoder,
    SemanticArtifactConfig,
    load_semantic_artifact_config,
)
from foundry.synthesis.taxonomy import FailureCategory, taxonomy_contract_sha256
from foundry.synthesis.verification import validate_final_answer_contract

_CATEGORY_ORDER = (
    FailureCategory.MULTI_STEP_BOOKKEEPING,
    FailureCategory.RATE_RATIO_PERCENTAGE,
    FailureCategory.CONSTRAINT_DISCRETE,
)


class SynthesisSmokeError(RuntimeError):
    """Raised when the bounded smoke contract is invalid or cannot complete."""


class GroupName(StrEnum):
    TARGETED = "targeted"
    GENERIC_CONTROL = "generic_control"


@dataclass(frozen=True)
class GroupContract:
    attempts: int
    category_counts: dict[str, int]
    output_contract_counts: dict[str, int]


@dataclass(frozen=True)
class SynthesisSmokeConfig:
    run_id: str
    master_seed: str
    attempts: int
    targeted: GroupContract
    generic_control: GroupContract
    difficulty_cycle: tuple[DifficultyLevel, ...]
    output_contract_id: str
    semantic_config_path: Path
    development_manifest_path: Path
    evaluation_config_path: Path
    raw_directory: Path
    summary_path: Path
    manual_audit_path: Path
    config_sha256: str


@dataclass(frozen=True)
class AttemptPlan:
    attempt_index: int
    group: GroupName
    group_index: int
    category: FailureCategory
    category_variant: int
    difficulty: DifficultyLevel
    output_contract_enabled: bool
    random_seed: int


@dataclass(frozen=True)
class SemanticScreenResult:
    outcome: ContaminationOutcome
    maximum_similarity: float | None
    matched_scope: str | None
    matched_identifier_prefix: str | None
    failure_reason: str | None


@dataclass(frozen=True)
class AttemptTimings:
    generation_seconds: float
    schema_seconds: float
    verification_seconds: float
    deduplication_seconds: float
    semantic_seconds: float


@dataclass(frozen=True)
class AttemptRecord:
    plan: AttemptPlan
    draft: CandidateDraft | None
    schema_valid: bool
    primary_verification: GeneratorVerification | None
    independent_verification: GeneratorVerification | None
    verifier_agreement: bool
    constraint_rejections: tuple[str, ...]
    lexical_screen: LexicalScreenResult | None
    semantic_screen: SemanticScreenResult | None
    manual_review_resolution: bool | None
    final_decision: str
    rejection_reason: str | None
    frozen_example: SyntheticExample | None
    timings: AttemptTimings


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SynthesisSmokeError(f"{location} must be a mapping")
    return cast(dict[str, object], value)


def _string(mapping: dict[str, object], key: str, location: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SynthesisSmokeError(f"{location}.{key} must be a non-empty string")
    return value


def _integer(mapping: dict[str, object], key: str, location: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SynthesisSmokeError(f"{location}.{key} must be an integer")
    return value


def _group_contract(raw: object, location: str) -> GroupContract:
    mapping = _mapping(raw, location)
    category_raw = _mapping(mapping.get("category_counts"), f"{location}.category_counts")
    output_raw = _mapping(
        mapping.get("output_contract_counts"), f"{location}.output_contract_counts"
    )
    expected = {str(category) for category in _CATEGORY_ORDER}
    if category_raw.keys() != expected or output_raw.keys() != expected:
        raise SynthesisSmokeError(f"{location} category keys differ from the frozen taxonomy")
    category_counts = {
        key: _integer(category_raw, key, f"{location}.category_counts") for key in expected
    }
    output_counts = {
        key: _integer(output_raw, key, f"{location}.output_contract_counts") for key in expected
    }
    return GroupContract(
        attempts=_integer(mapping, "attempts", location),
        category_counts=category_counts,
        output_contract_counts=output_counts,
    )


def load_smoke_config(path: Path) -> SynthesisSmokeConfig:
    """Load and enforce the exact 120-attempt curriculum contract."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SynthesisSmokeError(f"could not load synthesis smoke config: {error}") from error
    root = _mapping(raw, "synthesis smoke config")
    groups = _mapping(root.get("groups"), "groups")
    targeted = _group_contract(groups.get("targeted"), "groups.targeted")
    generic = _group_contract(groups.get("generic_control"), "groups.generic_control")
    difficulty_raw = root.get("difficulty_cycle")
    if not isinstance(difficulty_raw, list):
        raise SynthesisSmokeError("difficulty_cycle must be a list")
    try:
        difficulty = tuple(DifficultyLevel(item) for item in difficulty_raw)
    except (TypeError, ValueError) as error:
        raise SynthesisSmokeError("difficulty_cycle contains an invalid level") from error
    output = _mapping(root.get("output_contract"), "output_contract")
    rules = _mapping(root.get("rules"), "rules")
    config = SynthesisSmokeConfig(
        run_id=_string(root, "run_id", "root"),
        master_seed=_string(root, "master_seed", "root"),
        attempts=_integer(root, "attempts", "root"),
        targeted=targeted,
        generic_control=generic,
        difficulty_cycle=difficulty,
        output_contract_id=_string(output, "id", "output_contract"),
        semantic_config_path=Path(_string(root, "semantic_config", "root")),
        development_manifest_path=Path(_string(root, "development_manifest", "root")),
        evaluation_config_path=Path(_string(root, "evaluation_config", "root")),
        raw_directory=Path(_string(root, "raw_directory", "root")),
        summary_path=Path(_string(root, "summary_path", "root")),
        manual_audit_path=Path(_string(root, "manual_audit_path", "root")),
        config_sha256=hashlib.sha256(
            json.dumps(root, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
    if config.attempts != 120 or targeted.attempts != 60 or generic.attempts != 60:
        raise SynthesisSmokeError("smoke must contain exactly 60 targeted and 60 generic attempts")
    if sum(targeted.category_counts.values()) != 60 or sum(generic.category_counts.values()) != 60:
        raise SynthesisSmokeError("group category counts must sum to 60")
    if (
        sum(targeted.output_contract_counts.values()) != 12
        or sum(generic.output_contract_counts.values()) != 12
    ):
        raise SynthesisSmokeError("each group must contain exactly 12 output-contract attempts")
    if any(
        contract.output_contract_counts[key] > contract.category_counts[key]
        for contract in (targeted, generic)
        for key in contract.category_counts
    ):
        raise SynthesisSmokeError("output-contract allocation exceeds a category allocation")
    if difficulty != (
        DifficultyLevel.EASY,
        DifficultyLevel.MEDIUM,
        DifficultyLevel.HARD,
    ):
        raise SynthesisSmokeError("difficulty cycle differs from easy/medium/hard")
    if config.output_contract_id != "terminal-final-answer-contract-v1":
        raise SynthesisSmokeError("output-contract ID differs from the frozen design")
    required_rules = {
        "replace_rejected_candidates": False,
        "benchmark_answers_allowed": False,
        "sealed_final_allowed": False,
        "llm_generation_allowed": False,
        "llm_judge_allowed": False,
    }
    if rules != required_rules:
        raise SynthesisSmokeError("smoke safety rules differ from the approved boundary")
    return config


def _seed(master_seed: str, group: GroupName, group_index: int, category: str) -> int:
    material = f"{master_seed}:{group}:{group_index}:{category}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def build_attempt_plan(config: SynthesisSmokeConfig) -> tuple[AttemptPlan, ...]:
    """Build stable targeted/generic schedules with matched seed and difficulty methods."""

    preliminary: list[tuple[GroupName, int, FailureCategory]] = []
    for group, contract in (
        (GroupName.TARGETED, config.targeted),
        (GroupName.GENERIC_CONTROL, config.generic_control),
    ):
        group_index = 0
        for category in _CATEGORY_ORDER:
            for _ in range(contract.category_counts[str(category)]):
                preliminary.append((group, group_index, category))
                group_index += 1

    output_indices: set[tuple[GroupName, int]] = set()
    for group, contract in (
        (GroupName.TARGETED, config.targeted),
        (GroupName.GENERIC_CONTROL, config.generic_control),
    ):
        for category in _CATEGORY_ORDER:
            entries = [
                group_index
                for entry_group, group_index, entry_category in preliminary
                if entry_group is group and entry_category is category
            ]
            ranked = sorted(
                entries,
                key=lambda index: hashlib.sha256(
                    f"{config.master_seed}:output:{group}:{category}:{index}".encode()
                ).hexdigest(),
            )
            for index in ranked[: contract.output_contract_counts[str(category)]]:
                output_indices.add((group, index))

    variants: Counter[str] = Counter()
    plans: list[AttemptPlan] = []
    for attempt_index, (group, group_index, category) in enumerate(preliminary, start=1):
        variant = variants[str(category)]
        variants[str(category)] += 1
        plans.append(
            AttemptPlan(
                attempt_index=attempt_index,
                group=group,
                group_index=group_index,
                category=category,
                category_variant=variant,
                difficulty=config.difficulty_cycle[group_index % len(config.difficulty_cycle)],
                output_contract_enabled=(group, group_index) in output_indices,
                random_seed=_seed(config.master_seed, group, group_index, str(category)),
            )
        )
    if len(plans) != 120 or len({plan.random_seed for plan in plans}) != 120:
        raise SynthesisSmokeError("attempt plan is incomplete or has duplicate seeds")
    return tuple(plans)


def _generate(plan: AttemptPlan) -> CandidateDraft:
    if plan.category is FailureCategory.MULTI_STEP_BOOKKEEPING:
        return generate_bookkeeping(
            seed=plan.random_seed,
            difficulty=plan.difficulty,
            variant=plan.category_variant,
            output_contract_enabled=plan.output_contract_enabled,
        )
    if plan.category is FailureCategory.RATE_RATIO_PERCENTAGE:
        return generate_rates(
            seed=plan.random_seed,
            difficulty=plan.difficulty,
            variant=plan.category_variant,
            output_contract_enabled=plan.output_contract_enabled,
        )
    if plan.category is FailureCategory.CONSTRAINT_DISCRETE:
        return generate_discrete(
            seed=plan.random_seed,
            difficulty=plan.difficulty,
            variant=plan.category_variant,
            output_contract_enabled=plan.output_contract_enabled,
        )
    raise SynthesisSmokeError("attempt plan contains an unapproved generator category")


def _verify(
    draft: CandidateDraft,
) -> tuple[GeneratorVerification, GeneratorVerification, tuple[str, ...]]:
    if draft.target_failure_category == FailureCategory.MULTI_STEP_BOOKKEEPING:
        return (
            verify_bookkeeping_dag(draft),
            verify_bookkeeping_ledger(draft),
            validate_bookkeeping_constraints(draft),
        )
    if draft.target_failure_category == FailureCategory.RATE_RATIO_PERCENTAGE:
        return (
            verify_rate_equation(draft),
            verify_rate_inverse(draft),
            validate_rate_constraints(draft),
        )
    if draft.target_failure_category == FailureCategory.CONSTRAINT_DISCRETE:
        return (
            verify_discrete_constructive(draft),
            verify_discrete_enumeration(draft),
            validate_discrete_constraints(draft),
        )
    raise SynthesisSmokeError("draft contains an unapproved generator category")


class _SemanticIndex:
    def __init__(
        self,
        encoder: PinnedSentenceEncoder,
        development: tuple[DevelopmentQuestion, ...],
    ) -> None:
        self.encoder = encoder
        self.development_ids = tuple(question.stable_id for question in development)
        self.development_embeddings = encoder.encode(
            [question.question for question in development]
        )
        self.generated_ids: list[str] = []
        self.generated_embeddings: list[torch.Tensor] = []

    def encode(self, question: str) -> torch.Tensor:
        return self.encoder.encode([question])

    def screen(self, embedding: torch.Tensor) -> SemanticScreenResult:
        try:
            development_scores = self.encoder.cosine_matrix(embedding, self.development_embeddings)[
                0
            ]
            development_value, development_index = torch.max(development_scores, dim=0)
            maximum = float(development_value.item())
            scope = "development"
            identifier = self.development_ids[int(development_index.item())]
            if self.generated_embeddings:
                generated_matrix = torch.cat(self.generated_embeddings, dim=0)
                generated_scores = self.encoder.cosine_matrix(embedding, generated_matrix)[0]
                generated_value, generated_index = torch.max(generated_scores, dim=0)
                if float(generated_value.item()) > maximum:
                    maximum = float(generated_value.item())
                    scope = "generated"
                    identifier = self.generated_ids[int(generated_index.item())]
            return SemanticScreenResult(
                outcome=self.encoder.config.thresholds.classify(maximum),
                maximum_similarity=maximum,
                matched_scope=scope,
                matched_identifier_prefix=identifier[:12],
                failure_reason=None,
            )
        except (RuntimeError, ValueError) as error:
            return SemanticScreenResult(
                outcome=ContaminationOutcome.REJECT,
                maximum_similarity=None,
                matched_scope=None,
                matched_identifier_prefix=None,
                failure_reason=f"semantic_inference_failure:{type(error).__name__}",
            )

    def add(self, candidate_id: str, embedding: torch.Tensor) -> None:
        self.generated_ids.append(candidate_id)
        self.generated_embeddings.append(embedding)


def _verification_evidence(result: GeneratorVerification) -> VerificationEvidence:
    if not result.success or result.answer is None:
        raise SynthesisSmokeError("cannot create schema evidence from a failed verifier")
    return VerificationEvidence(
        verifier_id=result.verifier_id,
        verifier_version=result.verifier_version,
        method_family=result.method_family,
        computed_answer=result.answer,
        evidence_sha256=result.evidence_sha256,
    )


def _build_frozen_example(
    *,
    draft: CandidateDraft,
    primary: GeneratorVerification,
    independent: GeneratorVerification,
    config_sha256: str,
    accepted: bool,
    rejection_reason: str | None,
    lexical: LexicalScreenResult | None,
    semantic: SemanticScreenResult | None,
) -> SyntheticExample:
    dedup_status = ReviewStatus.NOT_RUN
    if lexical is not None:
        dedup_status = ReviewStatus.REJECTED if lexical.rejection_reason else ReviewStatus.PASSED
    contamination_status = ReviewStatus.NOT_RUN
    if semantic is not None:
        contamination_status = {
            ContaminationOutcome.PASS: ReviewStatus.PASSED,
            ContaminationOutcome.REJECT: ReviewStatus.REJECTED,
            ContaminationOutcome.MANUAL_REVIEW: ReviewStatus.MANUAL_REVIEW,
        }[semantic.outcome]
    tags = draft.secondary_skill_tags + (
        ("terminal-final-answer-contract-v1",) if draft.output_contract_enabled else ()
    )
    return SyntheticExample(
        synthetic_example_id=draft.candidate_id,
        generator_version=draft.generator_version,
        random_seed=draft.random_seed,
        target_failure_category=draft.target_failure_category,
        secondary_skill_tags=tags,
        difficulty_level=draft.difficulty_level,
        latent_program=draft.latent_program,
        rendered_question=draft.rendered_question,
        deterministic_solution_trace=draft.deterministic_solution_trace,
        canonical_final_answer=draft.canonical_final_answer,
        required_final_answer_format="Final answer: <canonical-number>",
        primary_verification_evidence=_verification_evidence(primary),
        independent_verification_evidence=_verification_evidence(independent),
        validity_status=ValidityStatus.ACCEPTED if accepted else ValidityStatus.REJECTED,
        rejection_reason=rejection_reason,
        normalized_text_hash=normalized_text_sha256(draft.rendered_question),
        latent_program_hash=draft.structure_sha256,
        provenance=ProvenanceMetadata(
            source_kind="independent_procedural",
            generator_config_sha256=config_sha256,
            taxonomy_sha256=taxonomy_contract_sha256(),
            benchmark_content_used_as_generator_input=False,
        ),
        contamination_check_status=contamination_status,
        deduplication_status=dedup_status,
    )


def _empty_timings(*, generation: float) -> AttemptTimings:
    return AttemptTimings(generation, 0.0, 0.0, 0.0, 0.0)


def _process_attempt(
    *,
    plan: AttemptPlan,
    config: SynthesisSmokeConfig,
    deduplication: DeduplicationIndex,
    semantic: _SemanticIndex,
    manual_decisions: Mapping[str, bool],
) -> AttemptRecord:
    generation_start = time.perf_counter()
    try:
        draft = _generate(plan)
    except Exception as error:
        return AttemptRecord(
            plan=plan,
            draft=None,
            schema_valid=False,
            primary_verification=None,
            independent_verification=None,
            verifier_agreement=False,
            constraint_rejections=(),
            lexical_screen=None,
            semantic_screen=None,
            manual_review_resolution=None,
            final_decision="rejected",
            rejection_reason=f"generator_exception:{type(error).__name__}",
            frozen_example=None,
            timings=_empty_timings(generation=time.perf_counter() - generation_start),
        )
    generation_seconds = time.perf_counter() - generation_start

    schema_start = time.perf_counter()
    schema_valid = (
        draft.random_seed == plan.random_seed
        and draft.difficulty_level is plan.difficulty
        and draft.output_contract_enabled is plan.output_contract_enabled
        and draft.target_failure_category == plan.category
        and bool(draft.structure_sha256)
    )
    schema_seconds = time.perf_counter() - schema_start

    verification_start = time.perf_counter()
    typed_rejections = validate_realization(
        problem=draft.problem_ir,
        realization=draft.realization,
        answer=draft.canonical_final_answer,
    )
    primary, independent, generator_rejections = _verify(draft)
    constraint_rejections = typed_rejections + generator_rejections
    agreement = (
        primary.success
        and independent.success
        and primary.answer is not None
        and primary.answer == independent.answer == draft.canonical_final_answer
        and primary.method_family != independent.method_family
        and primary.verifier_id != independent.verifier_id
    )
    output_valid = True
    if draft.output_contract_enabled:
        output_valid = validate_final_answer_contract(
            draft.training_completion, draft.canonical_final_answer
        ) and draft.training_completion.strip().splitlines()[-1] == (
            f"Final answer: {draft.canonical_final_answer.render()}"
        )
    verification_seconds = time.perf_counter() - verification_start

    rejection_reason: str | None = None
    if not schema_valid:
        rejection_reason = "schema_validation_failure"
    elif typed_rejections:
        rejection_reason = typed_rejections[0]
    elif not primary.success:
        rejection_reason = f"primary_verifier_failure:{primary.failure_reason}"
    elif not independent.success:
        rejection_reason = f"independent_verifier_failure:{independent.failure_reason}"
    elif not agreement:
        rejection_reason = "verifier_disagreement"
    elif generator_rejections:
        rejection_reason = generator_rejections[0]
    elif not output_valid:
        rejection_reason = "output_contract_failure"

    lexical: LexicalScreenResult | None = None
    deduplication_start = time.perf_counter()
    if rejection_reason is None:
        lexical = deduplication.screen(draft)
        if lexical.rejection_reason is not None:
            rejection_reason = lexical.rejection_reason
    deduplication_seconds = time.perf_counter() - deduplication_start

    semantic_start = time.perf_counter()
    semantic_result: SemanticScreenResult | None = None
    manual_resolution: bool | None = None
    try:
        embedding = semantic.encode(draft.rendered_question)
    except (RuntimeError, ValueError) as error:
        embedding = None
        if rejection_reason is None:
            rejection_reason = f"semantic_inference_failure:{type(error).__name__}"
    if rejection_reason is None and embedding is not None:
        semantic_result = semantic.screen(embedding)
        if semantic_result.failure_reason is not None:
            rejection_reason = semantic_result.failure_reason
        elif semantic_result.outcome is ContaminationOutcome.REJECT:
            rejection_reason = "semantic_similarity"
        elif semantic_result.outcome is ContaminationOutcome.MANUAL_REVIEW:
            manual_resolution = manual_decisions.get(draft.candidate_id)
            if manual_resolution is None:
                rejection_reason = "unresolved_contamination"
            elif not manual_resolution:
                rejection_reason = "manual_contamination_rejection"
    if embedding is not None:
        semantic.add(draft.candidate_id, embedding)
    semantic_seconds = time.perf_counter() - semantic_start
    deduplication.add_candidate(draft)

    accepted = rejection_reason is None
    frozen_example: SyntheticExample | None = None
    if primary.success and independent.success and agreement:
        frozen_example = _build_frozen_example(
            draft=draft,
            primary=primary,
            independent=independent,
            config_sha256=config.config_sha256,
            accepted=accepted,
            rejection_reason=rejection_reason,
            lexical=lexical,
            semantic=semantic_result,
        )
    return AttemptRecord(
        plan=plan,
        draft=draft,
        schema_valid=schema_valid,
        primary_verification=primary,
        independent_verification=independent,
        verifier_agreement=agreement,
        constraint_rejections=constraint_rejections,
        lexical_screen=lexical,
        semantic_screen=semantic_result,
        manual_review_resolution=manual_resolution,
        final_decision="accepted" if accepted else "rejected",
        rejection_reason=rejection_reason,
        frozen_example=frozen_example,
        timings=AttemptTimings(
            generation_seconds,
            schema_seconds,
            verification_seconds,
            deduplication_seconds,
            semantic_seconds,
        ),
    )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, StrEnum):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _record_payload(record: AttemptRecord) -> dict[str, object]:
    return cast(dict[str, object], asdict(record))


def write_raw_attempts(path: Path, records: tuple[AttemptRecord, ...]) -> None:
    """Write complete content-bearing attempts only to the ignored raw directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(
        json.dumps(_record_payload(record), sort_keys=True, default=_json_default)
        for record in records
    )
    path.write_text(rendered + ("\n" if records else ""), encoding="utf-8")


def _deterministic_record(record: AttemptRecord) -> dict[str, object]:
    draft = record.draft
    primary = record.primary_verification
    independent = record.independent_verification
    lexical = record.lexical_screen
    semantic = record.semantic_screen
    return {
        "plan": asdict(record.plan),
        "candidate_id": None if draft is None else draft.candidate_id,
        "semantic_ir_sha256": None if draft is None else draft.semantic_ir_sha256,
        "render_signature_sha256": None if draft is None else draft.render_signature_sha256,
        "text_sha256": None if draft is None else normalized_text_sha256(draft.rendered_question),
        "latent_program_sha256": None if draft is None else draft.structure_sha256,
        "primary": None
        if primary is None
        else {
            "success": primary.success,
            "answer": None if primary.answer is None else primary.answer.render(),
            "evidence_sha256": primary.evidence_sha256,
            "failure_reason": primary.failure_reason,
        },
        "independent": None
        if independent is None
        else {
            "success": independent.success,
            "answer": None if independent.answer is None else independent.answer.render(),
            "evidence_sha256": independent.evidence_sha256,
            "failure_reason": independent.failure_reason,
        },
        "verifier_agreement": record.verifier_agreement,
        "constraints": list(record.constraint_rejections),
        "lexical": None
        if lexical is None
        else {
            "reason": lexical.rejection_reason,
            "scope": lexical.matched_scope,
            "identifier": lexical.matched_identifier_prefix,
            "ngram": format(lexical.maximum_ngram_jaccard, ".8f"),
        },
        "semantic": None
        if semantic is None
        else {
            "outcome": semantic.outcome,
            "similarity": None
            if semantic.maximum_similarity is None
            else format(semantic.maximum_similarity, ".8f"),
            "scope": semantic.matched_scope,
            "identifier": semantic.matched_identifier_prefix,
            "failure_reason": semantic.failure_reason,
        },
        "manual_review_resolution": record.manual_review_resolution,
        "final_decision": record.final_decision,
        "rejection_reason": record.rejection_reason,
    }


def deterministic_decision_sha256(records: tuple[AttemptRecord, ...]) -> str:
    """Hash all deterministic candidate, verification, and decision fields."""

    payload = [_deterministic_record(record) for record in records]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode(
            "utf-8"
        )
    ).hexdigest()


def deterministic_aggregate_sha256(
    *, config: SynthesisSmokeConfig, records: tuple[AttemptRecord, ...]
) -> str:
    """Hash only deterministic aggregate fields, excluding runtime and resource telemetry."""

    accepted = [record for record in records if record.final_decision == "accepted"]
    rejected = [record for record in records if record.final_decision == "rejected"]
    payload = {
        "run_id": config.run_id,
        "master_seed": config.master_seed,
        "config_sha256": config.config_sha256,
        "attempted": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "counts_by_group": _counter(records, "group"),
        "counts_by_category": _counter(records, "category"),
        "counts_by_difficulty": _counter(records, "difficulty"),
        "output_contract_total": sum(record.plan.output_contract_enabled for record in records),
        "rejection_reasons": dict(
            sorted(
                Counter(
                    record.rejection_reason
                    for record in rejected
                    if record.rejection_reason is not None
                ).items()
            )
        ),
        "decision_sha256": deterministic_decision_sha256(records),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _counter(records: tuple[AttemptRecord, ...], field: str) -> dict[str, int]:
    if field == "group":
        values = [str(record.plan.group) for record in records]
    elif field == "category":
        values = [str(record.plan.category) for record in records]
    elif field == "difficulty":
        values = [str(record.plan.difficulty) for record in records]
    else:
        raise ValueError("unsupported summary counter")
    return dict(sorted(Counter(values).items()))


def build_content_free_summary(
    *,
    config: SynthesisSmokeConfig,
    records: tuple[AttemptRecord, ...],
    runtime_seconds: float,
    initialization_seconds: dict[str, float],
    peak_system_ram_bytes: int,
    semantic_artifact_disk_bytes: int,
    raw_artifact_disk_bytes: int,
    gpu_peak_allocated_bytes: int,
) -> dict[str, object]:
    """Build aggregate-only smoke evidence safe for version control."""

    rejected = [record for record in records if record.final_decision == "rejected"]
    accepted = [record for record in records if record.final_decision == "accepted"]
    reasons = Counter(record.rejection_reason for record in rejected if record.rejection_reason)
    semantic_reviews = sum(
        record.semantic_screen is not None
        and record.semantic_screen.outcome is ContaminationOutcome.MANUAL_REVIEW
        for record in records
    )
    semantic_rejections = sum(
        record.rejection_reason == "semantic_similarity" for record in records
    )
    timings = {
        "generation_seconds": sum(record.timings.generation_seconds for record in records),
        "schema_seconds": sum(record.timings.schema_seconds for record in records),
        "verification_seconds": sum(record.timings.verification_seconds for record in records),
        "deduplication_seconds": initialization_seconds["deduplication"]
        + sum(record.timings.deduplication_seconds for record in records),
        "semantic_screening_seconds": initialization_seconds["semantic"]
        + sum(record.timings.semantic_seconds for record in records),
        "manual_audit_seconds": 0.0,
        "total_runtime_seconds": runtime_seconds,
    }
    family_accepts = Counter(str(record.plan.category) for record in accepted)
    output_by_group = Counter(
        str(record.plan.group) for record in records if record.plan.output_contract_enabled
    )
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "master_seed": config.master_seed,
        "config_sha256": config.config_sha256,
        "attempted": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "counts_by_group": _counter(records, "group"),
        "counts_by_category": _counter(records, "category"),
        "accepted_by_category": dict(sorted(family_accepts.items())),
        "counts_by_difficulty": _counter(records, "difficulty"),
        "output_contract_total": sum(record.plan.output_contract_enabled for record in records),
        "output_contract_by_group": dict(sorted(output_by_group.items())),
        "primary_verifier_failures": sum(
            record.primary_verification is not None and not record.primary_verification.success
            for record in records
        ),
        "independent_verifier_failures": sum(
            record.independent_verification is not None
            and not record.independent_verification.success
            for record in records
        ),
        "verifier_disagreements": sum(
            record.primary_verification is not None
            and record.independent_verification is not None
            and record.primary_verification.success
            and record.independent_verification.success
            and not record.verifier_agreement
            for record in records
        ),
        "ambiguous_target_rejections": reasons["ambiguous_target"],
        "exact_duplicate_rejections": reasons["exact_normalized_text"],
        "numeric_template_rejections": reasons["numeric_template_copy"],
        "latent_structure_rejections": reasons["latent_structure_copy"],
        "ngram_rejections": reasons["token_ngram_overlap"],
        "semantic_review_cases": semantic_reviews,
        "semantic_rejections": semantic_rejections,
        "unresolved_contamination_cases": reasons["unresolved_contamination"],
        "generator_exceptions": sum(
            bool(
                record.rejection_reason
                and record.rejection_reason.startswith("generator_exception")
            )
            for record in records
        ),
        "rejection_reasons": dict(sorted((str(key), value) for key, value in reasons.items())),
        "deterministic_decision_sha256": deterministic_decision_sha256(records),
        "deterministic_aggregate_sha256": deterministic_aggregate_sha256(
            config=config, records=records
        ),
        "deterministic_replay_passed": False,
        "manual_audit": {
            "completed": False,
            "reviewed": 0,
            "false_labels": None,
            "invalid_acceptances": None,
            "incorrect_rejections": None,
            "unresolved_contamination": None,
            "systematic_weaknesses": [],
        },
        "readiness_gate": {"passed": False, "reasons": ["manual_audit_pending"]},
        "runtime": {key: round(value, 6) for key, value in timings.items()},
        "resources": {
            "peak_system_ram_bytes": peak_system_ram_bytes,
            "gpu_used": gpu_peak_allocated_bytes > 0,
            "gpu_peak_allocated_bytes": gpu_peak_allocated_bytes,
            "semantic_artifact_disk_bytes": semantic_artifact_disk_bytes,
            "raw_artifact_disk_bytes": raw_artifact_disk_bytes,
        },
        "benchmark_boundary": {
            "development_questions_compared": 904,
            "benchmark_answers_loaded": False,
            "sealed_final_accessed": False,
            "benchmark_content_in_generator_inputs": False,
        },
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke(
    *,
    repository_root: Path,
    config: SynthesisSmokeConfig,
    manual_decisions: Mapping[str, bool],
    pause_at: int | None,
) -> tuple[tuple[AttemptRecord, ...], dict[str, object]]:
    """Run exactly the configured attempts, optionally pausing after a progress checkpoint."""

    total_start = time.perf_counter()
    process = psutil.Process()
    initial_gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    deduplication_start = time.perf_counter()
    development = load_development_questions_for_contamination(
        evaluation_config_path=repository_root / config.evaluation_config_path,
        development_manifest_path=repository_root / config.development_manifest_path,
    )
    deduplication = DeduplicationIndex(development)
    deduplication_initialization = time.perf_counter() - deduplication_start

    semantic_start = time.perf_counter()
    semantic_config: SemanticArtifactConfig = load_semantic_artifact_config(
        repository_root / config.semantic_config_path
    )
    encoder = PinnedSentenceEncoder(semantic_config, repository_root)
    semantic = _SemanticIndex(encoder, development)
    semantic_initialization = time.perf_counter() - semantic_start
    plans = build_attempt_plan(config)
    records: list[AttemptRecord] = []
    raw_path = repository_root / config.raw_directory / "attempts.jsonl"
    for plan in plans:
        records.append(
            _process_attempt(
                plan=plan,
                config=config,
                deduplication=deduplication,
                semantic=semantic,
                manual_decisions=manual_decisions,
            )
        )
        if pause_at is not None and len(records) == pause_at:
            write_raw_attempts(raw_path, tuple(records))
            accepted = sum(record.final_decision == "accepted" for record in records)
            print(
                f"PROGRESS attempts={len(records)} accepted={accepted} "
                f"rejected={len(records) - accepted}",
                flush=True,
            )
            input("PAUSED: press Enter to continue the same bounded smoke process... ")
    records_tuple = tuple(records)
    write_raw_attempts(raw_path, records_tuple)
    raw_bytes = sum(
        path.stat().st_size
        for path in (repository_root / config.raw_directory).rglob("*")
        if path.is_file()
    )
    artifact_bytes = sum(
        path.stat().st_size
        for path in semantic_config.snapshot_path(repository_root).rglob("*")
        if path.is_file()
    )
    memory = process.memory_info()
    peak_ram = int(getattr(memory, "peak_wset", memory.rss))
    gpu_peak = 0
    if torch.cuda.is_available():
        gpu_peak = max(0, torch.cuda.max_memory_allocated() - initial_gpu)
    summary = build_content_free_summary(
        config=config,
        records=records_tuple,
        runtime_seconds=time.perf_counter() - total_start,
        initialization_seconds={
            "deduplication": deduplication_initialization,
            "semantic": semantic_initialization,
        },
        peak_system_ram_bytes=peak_ram,
        semantic_artifact_disk_bytes=artifact_bytes,
        raw_artifact_disk_bytes=raw_bytes,
        gpu_peak_allocated_bytes=gpu_peak,
    )
    write_summary(repository_root / config.summary_path, summary)
    print(
        f"COMPLETE attempts={summary['attempted']} accepted={summary['accepted']} "
        f"rejected={summary['rejected']} "
        f"decision_sha256={summary['deterministic_decision_sha256']}",
        flush=True,
    )
    return records_tuple, summary
