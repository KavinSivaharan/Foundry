"""Exact 30-IR plan and frozen configuration for the compact micro-smoke."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.realization.compact_contracts import COMPACT_SYSTEM_PROMPT_SHA256
from foundry.synthesis.realization.compact_prompting import (
    COMPACT_COMBINED_PROTOCOL_SHA256,
    COMPACT_USER_PROTOCOL_SHA256,
)
from foundry.synthesis.realization.smoke_contract import (
    GroupAllocation,
    ModelContract,
    RealizationAttemptPlan,
    RealizationGroup,
    generate_procedural_ir,
)
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.taxonomy import FailureCategory

_CATEGORY_ORDER = (
    FailureCategory.MULTI_STEP_BOOKKEEPING,
    FailureCategory.RATE_RATIO_PERCENTAGE,
    FailureCategory.CONSTRAINT_DISCRETE,
)


@dataclass(frozen=True)
class CompactGenerationContract:
    seed: int
    enable_thinking: bool
    do_sample: bool
    num_beams: int
    num_return_sequences: int
    max_new_tokens: int
    timeout_seconds_per_ir: int
    stopping_tag: str


@dataclass(frozen=True)
class CompactSmokeConfig:
    run_id: str
    ir_master_seed: str
    ir_attempts: int
    model: ModelContract
    generation: CompactGenerationContract
    targeted: GroupAllocation
    generic_control: GroupAllocation
    difficulty_cycle: tuple[DifficultyLevel, ...]
    output_contract_id: str
    compact_system_prompt_sha256: str
    compact_user_protocol_sha256: str
    compact_combined_protocol_sha256: str
    chat_template_sha256: str
    internal_diversity_policy: Path
    semantic_artifact: Path
    development_question_export: Path
    raw_directory: Path
    summary_path: Path
    config_sha256: str


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
        raise ValueError(f"{location}.{key} must be boolean")
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


def load_compact_smoke_config(path: Path) -> CompactSmokeConfig:
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "compact smoke")
    if root.get("schema_version") != 1:
        raise ValueError("compact smoke schema version changed")
    model = _mapping(root.get("model"), "model")
    generation = _mapping(root.get("generation"), "generation")
    groups = _mapping(root.get("groups"), "groups")
    difficulty_raw = root.get("difficulty_cycle")
    if not isinstance(difficulty_raw, list):
        raise ValueError("difficulty_cycle must be a list")
    config = CompactSmokeConfig(
        run_id=_string(root, "run_id", "root"),
        ir_master_seed=_string(root, "ir_master_seed", "root"),
        ir_attempts=_integer(root, "ir_attempts", "root"),
        model=ModelContract(
            _string(model, "repo_id", "model"),
            _string(model, "revision", "model"),
            _string(model, "license_id", "model"),
            _boolean(model, "trust_remote_code", "model"),
            _string(model, "dtype", "model"),
            _boolean(model, "local_files_only", "model"),
            Path(_string(model, "cache_root", "model")),
        ),
        generation=CompactGenerationContract(
            _integer(generation, "seed", "generation"),
            _boolean(generation, "enable_thinking", "generation"),
            _boolean(generation, "do_sample", "generation"),
            _integer(generation, "num_beams", "generation"),
            _integer(generation, "num_return_sequences", "generation"),
            _integer(generation, "max_new_tokens", "generation"),
            _integer(generation, "timeout_seconds_per_ir", "generation"),
            _string(generation, "stopping_tag", "generation"),
        ),
        targeted=_group(groups.get("targeted"), "groups.targeted"),
        generic_control=_group(groups.get("generic_control"), "groups.generic_control"),
        difficulty_cycle=tuple(DifficultyLevel(item) for item in difficulty_raw),
        output_contract_id=_string(root, "output_contract_id", "root"),
        compact_system_prompt_sha256=_string(root, "compact_system_prompt_sha256", "root"),
        compact_user_protocol_sha256=_string(root, "compact_user_protocol_sha256", "root"),
        compact_combined_protocol_sha256=_string(root, "compact_combined_protocol_sha256", "root"),
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
    _validate_config(config, _mapping(root.get("rules"), "rules"))
    return config


def _validate_config(config: CompactSmokeConfig, rules: dict[str, object]) -> None:
    expected_targeted = {
        str(FailureCategory.MULTI_STEP_BOOKKEEPING): 8,
        str(FailureCategory.RATE_RATIO_PERCENTAGE): 4,
        str(FailureCategory.CONSTRAINT_DISCRETE): 3,
    }
    expected_generic = {
        str(FailureCategory.MULTI_STEP_BOOKKEEPING): 5,
        str(FailureCategory.RATE_RATIO_PERCENTAGE): 5,
        str(FailureCategory.CONSTRAINT_DISCRETE): 5,
    }
    if config.ir_attempts != 30 or config.targeted.attempts != 15:
        raise ValueError("compact smoke must contain 15 targeted IRs")
    if config.generic_control.attempts != 15:
        raise ValueError("compact smoke must contain 15 generic-control IRs")
    if config.targeted.category_counts != expected_targeted:
        raise ValueError("targeted compact allocation changed")
    if config.generic_control.category_counts != expected_generic:
        raise ValueError("generic compact allocation changed")
    for allocation in (config.targeted, config.generic_control):
        if sum(allocation.category_counts.values()) != 15:
            raise ValueError("compact group category allocation must sum to 15")
        if sum(allocation.output_contract_counts.values()) != 3:
            raise ValueError("each compact group requires three output-contract IRs")
    if config.generation != CompactGenerationContract(5172026, False, False, 3, 3, 384, 90, "</Q>"):
        raise ValueError("compact generation settings changed")
    if config.model != ModelContract(
        "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "Apache-2.0",
        False,
        "float16",
        True,
        Path("data/huggingface"),
    ):
        raise ValueError("compact model contract changed")
    if config.difficulty_cycle != (
        DifficultyLevel.EASY,
        DifficultyLevel.MEDIUM,
        DifficultyLevel.HARD,
    ):
        raise ValueError("compact difficulty cycle changed")
    if config.output_contract_id != "terminal-final-answer-contract-v1":
        raise ValueError("output-contract track changed")
    if config.compact_system_prompt_sha256 != COMPACT_SYSTEM_PROMPT_SHA256:
        raise ValueError("compact system-prompt hash changed")
    if config.compact_user_protocol_sha256 != COMPACT_USER_PROTOCOL_SHA256:
        raise ValueError("compact user-protocol hash changed")
    if config.compact_combined_protocol_sha256 != COMPACT_COMBINED_PROTOCOL_SHA256:
        raise ValueError("compact combined-protocol hash changed")
    if (
        config.chat_template_sha256
        != "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    ):
        raise ValueError("Qwen3 chat-template hash changed")
    expected_rules = {
        "benchmark_content_is_model_input": False,
        "benchmark_answers_allowed": False,
        "sealed_final_allowed": False,
        "replace_rejected_irs": False,
        "retry_until_success": False,
        "llm_math_verifier": False,
        "llm_judge": False,
        "repair_model_output": False,
    }
    if rules != expected_rules:
        raise ValueError("compact smoke safety rules changed")


def _seed(master_seed: str, group: RealizationGroup, index: int, category: str) -> int:
    material = f"{master_seed}:{group}:{index}:{category}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def build_compact_attempt_plan(
    config: CompactSmokeConfig,
) -> tuple[RealizationAttemptPlan, ...]:
    """Build the immutable fresh 30-IR micro-smoke plan."""

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
            output_indices.update(
                (group, index)
                for index in ranked[: allocation.output_contract_counts[str(category)]]
            )
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
                difficulty=config.difficulty_cycle[group_index % 3],
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
    if len(plans) != 30 or len({plan.random_seed for plan in plans}) != 30:
        raise ValueError("compact plan is incomplete or has duplicate seeds")
    return tuple(plans)


__all__ = [
    "CompactGenerationContract",
    "CompactSmokeConfig",
    "build_compact_attempt_plan",
    "generate_procedural_ir",
    "load_compact_smoke_config",
]
