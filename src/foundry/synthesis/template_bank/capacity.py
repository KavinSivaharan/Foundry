"""Content-free runtime-collision diagnosis and full-generation capacity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from foundry.synthesis.contamination import numeric_template_sha256
from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.generators.bookkeeping import generate_bookkeeping
from foundry.synthesis.generators.discrete import generate_discrete
from foundry.synthesis.generators.rates import generate_rates
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.taxonomy import FailureCategory
from foundry.synthesis.template_bank.bank import build_template_bank
from foundry.synthesis.template_bank.renderer import render_with_template
from foundry.synthesis.template_bank.smoke import _compatible_templates

CAPACITY_AUDIT_ID = "foundry-template-bank-runtime-capacity-v1"
ATTEMPT_MULTIPLIER_NUMERATOR = 5
ATTEMPT_MULTIPLIER_DENOMINATOR = 4

_BOOKKEEPING = str(FailureCategory.MULTI_STEP_BOOKKEEPING)
_RATES = str(FailureCategory.RATE_RATIO_PERCENTAGE)
_DISCRETE = str(FailureCategory.CONSTRAINT_DISCRETE)

_PLANNED_ACCEPTED_QUOTAS: dict[str, dict[str, int]] = {
    "targeted": {_BOOKKEEPING: 2200, _RATES: 933, _DISCRETE: 867},
    "generic_control": {_BOOKKEEPING: 1334, _RATES: 1333, _DISCRETE: 1333},
}

_STRUCTURAL_VARIANT_PERIODS = {
    _BOOKKEEPING: 240,
    _RATES: 25,
    _DISCRETE: 80,
}

_GENERATORS: dict[str, Callable[..., CandidateDraft]] = {
    _BOOKKEEPING: generate_bookkeeping,
    _RATES: generate_rates,
    _DISCRETE: generate_discrete,
}


@dataclass(frozen=True)
class CollisionRecord:
    """Content-free identity of one rejected runtime collision."""

    candidate_id_prefix: str
    attempt_index: int
    group: str
    category: str
    difficulty: str
    output_contract_enabled: bool
    semantic_frame_id: str
    template_id: str
    sentence_plan_id: str
    render_signature_sha256: str
    number_neutral_sha256: str
    latent_program_sha256: str
    earlier_candidate_id_prefix: str
    earlier_attempt_index: int
    earlier_template_id: str
    earlier_sentence_plan_id: str
    collision_stage: str
    cause: str


@dataclass
class _CapacitySets:
    plan_render_signatures: set[str]
    domain_render_signatures: set[str]
    number_neutral_signatures: set[str]
    semantic_frame_signatures: set[str]

    @classmethod
    def empty(cls) -> _CapacitySets:
        return cls(set(), set(), set(), set())

    def update(self, other: _CapacitySets) -> None:
        self.plan_render_signatures.update(other.plan_render_signatures)
        self.domain_render_signatures.update(other.domain_render_signatures)
        self.number_neutral_signatures.update(other.number_neutral_signatures)
        self.semantic_frame_signatures.update(other.semantic_frame_signatures)

    def counts(self) -> dict[str, int]:
        return {
            "active_plan_render_signatures": len(self.plan_render_signatures),
            "domain_aware_render_signatures": len(self.domain_render_signatures),
            "number_neutral_signatures": len(self.number_neutral_signatures),
            "semantic_frame_signatures": len(self.semantic_frame_signatures),
            "limiting_unique_capacity": min(
                len(self.plan_render_signatures),
                len(self.domain_render_signatures),
                len(self.number_neutral_signatures),
            ),
        }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _attempt_pool(accepted: int) -> int:
    numerator = accepted * ATTEMPT_MULTIPLIER_NUMERATOR
    return (numerator + ATTEMPT_MULTIPLIER_DENOMINATOR - 1) // ATTEMPT_MULTIPLIER_DENOMINATOR


def _probe_seed(category: str, variant: int, difficulty: DifficultyLevel) -> int:
    material = f"{CAPACITY_AUDIT_ID}:{category}:{variant}:{difficulty}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _structure_key(draft: CandidateDraft) -> tuple[object, ...]:
    problem = draft.problem_ir
    changes = getattr(problem, "changes", ())
    return (
        problem.domain.domain_id,
        tuple(str(change.kind) for change in changes),
        str(problem.target.kind),
        problem.context_node_id,
        str(getattr(problem, "relation_kind", "bookkeeping")),
        len(getattr(problem, "scalars", ())),
    )


def _structural_drafts(
    category: str, variant: int, difficulty: DifficultyLevel
) -> tuple[CandidateDraft, ...]:
    generator = _GENERATORS[category]
    seeds = [_probe_seed(category, variant, difficulty)]
    # Only a four-step all-subtraction bookkeeping program can hit the positive-state guard.
    # These two fixed witnesses cover the reachable guarded branch for ordinary and grouping modes.
    if category == _BOOKKEEPING and difficulty is DifficultyLevel.HARD and variant % 16 == 15:
        seeds.append(591177 if variant % 5 == 4 else 53676)
    drafts: dict[tuple[object, ...], CandidateDraft] = {}
    for seed in seeds:
        draft = generator(
            seed=seed,
            difficulty=difficulty,
            variant=variant,
            output_contract_enabled=False,
        )
        drafts.setdefault(_structure_key(draft), draft)
    return tuple(drafts.values())


def _enumerate_category(category: str) -> tuple[_CapacitySets, dict[str, dict[str, int]]]:
    bank = build_template_bank()
    combined = _CapacitySets.empty()
    by_difficulty: dict[str, dict[str, int]] = {}
    for difficulty in DifficultyLevel:
        current = _CapacitySets.empty()
        for variant in range(_STRUCTURAL_VARIANT_PERIODS[category]):
            for draft in _structural_drafts(category, variant, difficulty):
                for template in _compatible_templates(draft, bank):
                    for plan in template.sentence_plan_variants:
                        rendered = render_with_template(draft, template, plan)
                        current.plan_render_signatures.add(template.render_signature_hash(plan))
                        current.domain_render_signatures.add(rendered.realization.signature.sha256)
                        current.number_neutral_signatures.add(
                            numeric_template_sha256(rendered.rendered_question)
                        )
                        current.semantic_frame_signatures.add(
                            _canonical_sha256(
                                {
                                    "semantic_frame": template.semantic_frame,
                                    "domain_id": rendered.problem_ir.domain.domain_id,
                                    "sentence_plan_id": plan.plan_id,
                                    "target_kind": str(rendered.problem_ir.target.kind),
                                }
                            )
                        )
        combined.update(current)
        by_difficulty[str(difficulty)] = current.counts()
    return combined, by_difficulty


def _load_attempts(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"attempt record {line_number} is not an object")
        records.append(cast(dict[str, object], value))
    return tuple(records)


def _required_string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"attempt record has invalid {key}")
    return value


def _required_int(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"attempt record has invalid {key}")
    return value


def diagnose_collisions(attempts_path: Path) -> tuple[CollisionRecord, ...]:
    """Resolve every conservative duplicate rejection to its first earlier partner."""

    frames = {item.template_id: item.semantic_frame for item in build_template_bank()}
    seen_number_neutral: dict[str, dict[str, object]] = {}
    seen_latent: dict[str, dict[str, object]] = {}
    diagnoses: list[CollisionRecord] = []
    for record in _load_attempts(attempts_path):
        rendered_question = _required_string(record, "rendered_question")
        number_neutral = numeric_template_sha256(rendered_question)
        latent_hash = _required_string(record, "latent_program_sha256")
        reason = record.get("rejection_reason")
        if reason in {"numeric_template_copy", "latent_program_copy"}:
            prior = (
                seen_number_neutral.get(number_neutral)
                if reason == "numeric_template_copy"
                else seen_latent.get(latent_hash)
            )
            if prior is None:
                raise ValueError("duplicate rejection has no earlier collision partner")
            template_id = _required_string(record, "template_id")
            plan_id = _required_string(record, "sentence_plan_id")
            prior_plan = _required_string(prior, "sentence_plan_id")
            cause = (
                "same_plan_and_lexical_realization_across_semantic_frames"
                if reason == "numeric_template_copy" and plan_id == prior_plan
                else "different_plans_same_number_neutral_surface"
                if reason == "numeric_template_copy"
                else "latent_seed_collision"
            )
            candidate_id = _required_string(record, "candidate_id")
            prior_candidate_id = _required_string(prior, "candidate_id")
            output_enabled = record.get("output_contract_enabled")
            if not isinstance(output_enabled, bool):
                raise ValueError("attempt record has invalid output_contract_enabled")
            diagnoses.append(
                CollisionRecord(
                    candidate_id_prefix=candidate_id[:12],
                    attempt_index=_required_int(record, "attempt_index"),
                    group=_required_string(record, "group"),
                    category=_required_string(record, "category"),
                    difficulty=_required_string(record, "difficulty"),
                    output_contract_enabled=output_enabled,
                    semantic_frame_id=frames[template_id],
                    template_id=template_id,
                    sentence_plan_id=plan_id,
                    render_signature_sha256=_required_string(record, "render_signature_sha256"),
                    number_neutral_sha256=number_neutral,
                    latent_program_sha256=latent_hash,
                    earlier_candidate_id_prefix=prior_candidate_id[:12],
                    earlier_attempt_index=_required_int(prior, "attempt_index"),
                    earlier_template_id=_required_string(prior, "template_id"),
                    earlier_sentence_plan_id=prior_plan,
                    collision_stage=str(reason),
                    cause=cause,
                )
            )
        seen_number_neutral.setdefault(number_neutral, record)
        seen_latent.setdefault(latent_hash, record)
    counts = Counter(item.collision_stage for item in diagnoses)
    if counts != {"numeric_template_copy": 15, "latent_program_copy": 1}:
        raise ValueError("source smoke does not contain the expected 15/1 collision inventory")
    return tuple(diagnoses)


def _cycle_counts(total: int) -> dict[str, int]:
    labels = tuple(str(item) for item in DifficultyLevel)
    base, remainder = divmod(total, len(labels))
    return {label: base + (index < remainder) for index, label in enumerate(labels)}


def build_capacity_audit(attempts_path: Path) -> dict[str, object]:
    """Build the deterministic, content-free Milestone 6D capacity decision."""

    diagnoses = diagnose_collisions(attempts_path)
    capacity_by_category: dict[str, object] = {}
    capacity_gate = True
    limiting_strata: list[dict[str, object]] = []
    total_required_attempts = 0
    total_limiting_capacity = 0
    for category in (_BOOKKEEPING, _RATES, _DISCRETE):
        combined, by_difficulty = _enumerate_category(category)
        counts = combined.counts()
        required_by_group = {
            group: {
                "accepted_quota": quotas[category],
                "required_125_percent_attempt_pool": _attempt_pool(quotas[category]),
            }
            for group, quotas in _PLANNED_ACCEPTED_QUOTAS.items()
        }
        combined_accepted = sum(quotas[category] for quotas in _PLANNED_ACCEPTED_QUOTAS.values())
        combined_attempts = sum(
            _attempt_pool(quotas[category]) for quotas in _PLANNED_ACCEPTED_QUOTAS.values()
        )
        difficulty_requirements = _cycle_counts(combined_attempts)
        output_enabled_required = (combined_attempts + 4) // 5
        output_disabled_required = combined_attempts - output_enabled_required
        available = counts["limiting_unique_capacity"]
        category_passed = available >= combined_attempts
        capacity_gate = capacity_gate and category_passed
        total_required_attempts += combined_attempts
        total_limiting_capacity += available
        if not category_passed:
            limiting_strata.append(
                {
                    "category": category,
                    "required_unique_signatures": combined_attempts,
                    "available_under_all_current_controls": available,
                    "shortfall": combined_attempts - available,
                }
            )
        capacity_by_category[category] = {
            **counts,
            "by_difficulty": {
                difficulty: {
                    **difficulty_counts,
                    "required_attempt_pool_if_cycled": difficulty_requirements[difficulty],
                    "stratum_passed": (
                        difficulty_counts["limiting_unique_capacity"]
                        >= difficulty_requirements[difficulty]
                    ),
                }
                for difficulty, difficulty_counts in by_difficulty.items()
            },
            "by_output_contract": {
                "enabled": {
                    **counts,
                    "required_attempt_pool_at_20_percent": output_enabled_required,
                    "stratum_passed": available >= output_enabled_required,
                },
                "disabled": {
                    **counts,
                    "required_attempt_pool_at_80_percent": output_disabled_required,
                    "stratum_passed": available >= output_disabled_required,
                },
                "signature_pool_shared_across_statuses": True,
            },
            "planned_quotas": required_by_group,
            "combined_cross_dataset": {
                "accepted_quota": combined_accepted,
                "required_125_percent_attempt_pool": combined_attempts,
                "available_under_all_current_controls": available,
                "capacity_ratio": available / combined_attempts,
                "expected_average_plan_reuse_without_uniqueness": (
                    combined_attempts / counts["active_plan_render_signatures"]
                ),
                "gate_passed": category_passed,
            },
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "audit_id": CAPACITY_AUDIT_ID,
        "source_attempts_sha256": hashlib.sha256(attempts_path.read_bytes()).hexdigest(),
        "collision_inventory": {
            "attempted_source_candidates": 120,
            "number_neutral_collisions": 15,
            "latent_program_collisions": 1,
            "diagnoses": [asdict(item) for item in diagnoses],
            "root_cause_counts": dict(sorted(Counter(item.cause for item in diagnoses).items())),
        },
        "methodology": {
            "structural_variant_periods": _STRUCTURAL_VARIANT_PERIODS,
            "bookkeeping_positive_guard_witness_seeds": {
                "inventory": 53676,
                "grouping": 591177,
            },
            "active_render_signature_definition": "template_id:template_version:sentence_plan_hash",
            "domain_aware_signature_definition": (
                "existing RenderSignature with template, plan, and domain"
            ),
            "number_neutral_definition": "existing numeric_template_sha256",
            "output_contract_changes_question_surface": False,
            "persisted_question_corpus_created": False,
            "duplicate_thresholds_changed": False,
        },
        "planned_accepted_total": 8000,
        "required_125_percent_attempt_pool_total": total_required_attempts,
        "available_unique_total_under_all_current_controls": total_limiting_capacity,
        "overall_capacity_ratio": total_limiting_capacity / total_required_attempts,
        "capacity_by_category": capacity_by_category,
        "limiting_strata": limiting_strata,
        "capacity_gate_passed": capacity_gate,
        "existing_controls_make_8000_examples_feasible": capacity_gate,
        "allocator_implemented": False,
        "latent_schedule_implemented": False,
        "candidate_schedule_created": False,
        "fresh_smoke_run": False,
        "review_packet_created": False,
        "stop_reason": (
            None
            if capacity_gate
            else "current cross-dataset uniqueness capacity is below every 125% category pool"
        ),
        "scope_exclusions": [
            "no_template_wording_changes",
            "no_allocator_after_failed_capacity_gate",
            "no_fresh_smoke_after_failed_capacity_gate",
            "no_review_packet_after_failed_capacity_gate",
            "no_full_dataset_generation",
            "no_training",
            "no_benchmark_evaluation",
            "no_sealed_final_access",
        ],
    }
    payload["audit_sha256"] = _canonical_sha256(payload)
    return payload


def run_capacity_audit(attempts_path: Path, output_path: Path) -> dict[str, object]:
    """Write only the content-free aggregate audit."""

    payload = build_capacity_audit(attempts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_capacity_audit(args.attempts.resolve(), args.output.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
