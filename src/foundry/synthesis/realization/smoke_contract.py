"""Frozen configuration and exact 120-IR plan for the Qwen3 realization smoke."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.generators.bookkeeping import generate_bookkeeping
from foundry.synthesis.generators.discrete import generate_discrete
from foundry.synthesis.generators.rates import generate_rates
from foundry.synthesis.realization.model_contracts import SYSTEM_PROMPT_SHA256
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.taxonomy import FailureCategory

_CATEGORY_ORDER = (
    FailureCategory.MULTI_STEP_BOOKKEEPING,
    FailureCategory.RATE_RATIO_PERCENTAGE,
    FailureCategory.CONSTRAINT_DISCRETE,
)


class RealizationGroup(StrEnum):
    """The only two curricula in the matched smoke."""

    TARGETED = "targeted"
    GENERIC_CONTROL = "generic_control"


@dataclass(frozen=True)
class GroupAllocation:
    attempts: int
    category_counts: dict[str, int]
    output_contract_counts: dict[str, int]


@dataclass(frozen=True)
class GenerationContract:
    seed: int
    enable_thinking: bool
    do_sample: bool
    num_beams: int
    num_return_sequences: int
    max_new_tokens: int
    timeout_seconds_per_ir: int


@dataclass(frozen=True)
class ModelContract:
    repo_id: str
    revision: str
    license_id: str
    trust_remote_code: bool
    dtype: str
    local_files_only: bool
    cache_root: Path


@dataclass(frozen=True)
class RealizationSmokeConfig:
    run_id: str
    ir_master_seed: str
    ir_attempts: int
    model: ModelContract
    generation: GenerationContract
    targeted: GroupAllocation
    generic_control: GroupAllocation
    difficulty_cycle: tuple[DifficultyLevel, ...]
    output_contract_id: str
    system_prompt_sha256: str
    realization_design_sha256: str
    chat_template_sha256: str
    internal_diversity_policy: Path
    semantic_artifact: Path
    development_question_export: Path
    raw_directory: Path
    summary_path: Path
    config_sha256: str


@dataclass(frozen=True)
class RealizationAttemptPlan:
    attempt_index: int
    group: RealizationGroup
    group_index: int
    category: FailureCategory
    category_variant: int
    difficulty: DifficultyLevel
    output_contract_enabled: bool
    random_seed: int
    style_variant: int


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _string(data: dict[str, object], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a nonempty string")
    return value


def _integer(data: dict[str, object], key: str, location: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}.{key} must be an integer")
    return value


def _boolean(data: dict[str, object], key: str, location: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{location}.{key} must be a boolean")
    return value


def _group(value: object, location: str) -> GroupAllocation:
    data = _mapping(value, location)
    category = _mapping(data.get("category_counts"), f"{location}.category_counts")
    output = _mapping(data.get("output_contract_counts"), f"{location}.output_contract_counts")
    expected = {str(item) for item in _CATEGORY_ORDER}
    if set(category) != expected or set(output) != expected:
        raise ValueError(f"{location} category keys differ from the frozen taxonomy")
    return GroupAllocation(
        attempts=_integer(data, "attempts", location),
        category_counts={key: _integer(category, key, location) for key in expected},
        output_contract_counts={key: _integer(output, key, location) for key in expected},
    )


def load_realization_smoke_config(path: Path) -> RealizationSmokeConfig:
    """Load and enforce every bounded realization policy before inference."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "realization smoke")
    if root.get("schema_version") != 1:
        raise ValueError("realization smoke schema version changed")
    model = _mapping(root.get("model"), "model")
    generation = _mapping(root.get("generation"), "generation")
    groups = _mapping(root.get("groups"), "groups")
    rules = _mapping(root.get("rules"), "rules")
    difficulty_raw = root.get("difficulty_cycle")
    if not isinstance(difficulty_raw, list):
        raise ValueError("difficulty_cycle must be a list")
    difficulty = tuple(DifficultyLevel(item) for item in difficulty_raw)
    targeted = _group(groups.get("targeted"), "groups.targeted")
    generic = _group(groups.get("generic_control"), "groups.generic_control")
    config = RealizationSmokeConfig(
        run_id=_string(root, "run_id", "root"),
        ir_master_seed=_string(root, "ir_master_seed", "root"),
        ir_attempts=_integer(root, "ir_attempts", "root"),
        model=ModelContract(
            repo_id=_string(model, "repo_id", "model"),
            revision=_string(model, "revision", "model"),
            license_id=_string(model, "license_id", "model"),
            trust_remote_code=_boolean(model, "trust_remote_code", "model"),
            dtype=_string(model, "dtype", "model"),
            local_files_only=_boolean(model, "local_files_only", "model"),
            cache_root=Path(_string(model, "cache_root", "model")),
        ),
        generation=GenerationContract(
            seed=_integer(generation, "seed", "generation"),
            enable_thinking=_boolean(generation, "enable_thinking", "generation"),
            do_sample=_boolean(generation, "do_sample", "generation"),
            num_beams=_integer(generation, "num_beams", "generation"),
            num_return_sequences=_integer(generation, "num_return_sequences", "generation"),
            max_new_tokens=_integer(generation, "max_new_tokens", "generation"),
            timeout_seconds_per_ir=_integer(generation, "timeout_seconds_per_ir", "generation"),
        ),
        targeted=targeted,
        generic_control=generic,
        difficulty_cycle=difficulty,
        output_contract_id=_string(root, "output_contract_id", "root"),
        system_prompt_sha256=_string(root, "system_prompt_sha256", "root"),
        realization_design_sha256=_string(root, "realization_design_sha256", "root"),
        chat_template_sha256=_string(root, "chat_template_sha256", "root"),
        internal_diversity_policy=Path(_string(root, "internal_diversity_policy", "root")),
        semantic_artifact=Path(_string(root, "semantic_artifact", "root")),
        development_question_export=Path(_string(root, "development_question_export", "root")),
        raw_directory=Path(_string(root, "raw_directory", "root")),
        summary_path=Path(_string(root, "summary_path", "root")),
        config_sha256=hashlib.sha256(
            json.dumps(root, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
    _validate_config(config, rules)
    return config


def _validate_config(config: RealizationSmokeConfig, rules: dict[str, object]) -> None:
    if config.ir_attempts != 120 or config.targeted.attempts != 60:
        raise ValueError("realization smoke must contain 60 targeted IRs")
    if config.generic_control.attempts != 60:
        raise ValueError("realization smoke must contain 60 generic-control IRs")
    for allocation in (config.targeted, config.generic_control):
        if sum(allocation.category_counts.values()) != 60:
            raise ValueError("each group category allocation must sum to 60")
        if sum(allocation.output_contract_counts.values()) != 12:
            raise ValueError("each group must contain exactly 12 output-contract IRs")
    expected_targeted = {
        str(FailureCategory.MULTI_STEP_BOOKKEEPING): 33,
        str(FailureCategory.RATE_RATIO_PERCENTAGE): 14,
        str(FailureCategory.CONSTRAINT_DISCRETE): 13,
    }
    expected_generic = {key: 20 for key in expected_targeted}
    if config.targeted.category_counts != expected_targeted:
        raise ValueError("targeted allocation differs from the approved curriculum")
    if config.generic_control.category_counts != expected_generic:
        raise ValueError("generic allocation differs from the approved uniform control")
    generation = config.generation
    if generation != GenerationContract(5172026, False, False, 3, 3, 256, 90):
        raise ValueError("generation settings differ from the frozen Qwen3 contract")
    if config.model != ModelContract(
        "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "Apache-2.0",
        False,
        "float16",
        True,
        Path("data/huggingface"),
    ):
        raise ValueError("model contract differs from the approved Qwen3 pin")
    if config.difficulty_cycle != (
        DifficultyLevel.EASY,
        DifficultyLevel.MEDIUM,
        DifficultyLevel.HARD,
    ):
        raise ValueError("difficulty cycle changed")
    if config.output_contract_id != "terminal-final-answer-contract-v1":
        raise ValueError("output-contract track changed")
    if config.system_prompt_sha256 != SYSTEM_PROMPT_SHA256:
        raise ValueError("system-prompt hash differs from the frozen design")
    if config.realization_design_sha256 != (
        "d6e6ca82681b702e07c71a9732a8c81159ea7a9bca78c73193228f72ca4ec3a5"
    ):
        raise ValueError("realization-design hash changed")
    if config.chat_template_sha256 != (
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    ):
        raise ValueError("chat-template hash changed")
    expected_rules = {
        "benchmark_content_is_model_input": False,
        "benchmark_answers_allowed": False,
        "sealed_final_allowed": False,
        "replace_rejected_irs": False,
        "retry_until_success": False,
        "llm_math_verifier": False,
        "llm_judge": False,
    }
    if rules != expected_rules:
        raise ValueError("realization smoke safety rules changed")


def _seed(master_seed: str, group: RealizationGroup, index: int, category: str) -> int:
    material = f"{master_seed}:{group}:{index}:{category}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def build_realization_attempt_plan(
    config: RealizationSmokeConfig,
) -> tuple[RealizationAttemptPlan, ...]:
    """Build a stable 120-IR plan with no rejected-IR replacement path."""

    preliminary: list[tuple[RealizationGroup, int, FailureCategory]] = []
    for group, allocation in (
        (RealizationGroup.TARGETED, config.targeted),
        (RealizationGroup.GENERIC_CONTROL, config.generic_control),
    ):
        group_index = 0
        for category in _CATEGORY_ORDER:
            for _ in range(allocation.category_counts[str(category)]):
                preliminary.append((group, group_index, category))
                group_index += 1
    output_indices: set[tuple[RealizationGroup, int]] = set()
    for group, allocation in (
        (RealizationGroup.TARGETED, config.targeted),
        (RealizationGroup.GENERIC_CONTROL, config.generic_control),
    ):
        for category in _CATEGORY_ORDER:
            indexes = [
                index
                for candidate_group, index, candidate_category in preliminary
                if candidate_group is group and candidate_category is category
            ]
            ranked = sorted(
                indexes,
                key=lambda value: hashlib.sha256(
                    f"{config.ir_master_seed}:output:{group}:{category}:{value}".encode()
                ).hexdigest(),
            )
            count_for_category = allocation.output_contract_counts[str(category)]
            output_indices.update((group, index) for index in ranked[:count_for_category])
    variants: Counter[str] = Counter()
    plans: list[RealizationAttemptPlan] = []
    for attempt_index, (group, group_index, category) in enumerate(preliminary, start=1):
        variant = variants[str(category)]
        variants[str(category)] += 1
        plans.append(
            RealizationAttemptPlan(
                attempt_index=attempt_index,
                group=group,
                group_index=group_index,
                category=category,
                category_variant=variant,
                difficulty=config.difficulty_cycle[group_index % len(config.difficulty_cycle)],
                output_contract_enabled=(group, group_index) in output_indices,
                random_seed=_seed(config.ir_master_seed, group, group_index, str(category)),
                style_variant=int(
                    hashlib.sha256(
                        f"{config.ir_master_seed}:style:{group}:{group_index}".encode()
                    ).hexdigest()[:8],
                    16,
                ),
            )
        )
    if len(plans) != 120 or len({plan.random_seed for plan in plans}) != 120:
        raise ValueError("realization attempt plan is incomplete or has duplicate seeds")
    return tuple(plans)


def generate_procedural_ir(plan: RealizationAttemptPlan) -> CandidateDraft:
    """Generate one fresh procedural semantic IR through an approved family."""

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
    raise ValueError("attempt plan contains an unapproved category")


__all__ = [
    "RealizationAttemptPlan",
    "RealizationGroup",
    "RealizationSmokeConfig",
    "build_realization_attempt_plan",
    "generate_procedural_ir",
    "load_realization_smoke_config",
]
