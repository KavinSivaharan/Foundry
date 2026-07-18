"""Frozen content-free failure taxonomy and pilot priorities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum


class FailureCategory(StrEnum):
    """Primary categories assigned to every development failure."""

    MULTI_STEP_BOOKKEEPING = "multi_step_bookkeeping_or_omission"
    TARGET_INTERPRETATION = "target_or_language_interpretation"
    RATE_RATIO_PERCENTAGE = "rate_ratio_percentage_or_average"
    CONSTRAINT_DISCRETE = "constraint_distribution_or_discrete_reasoning"
    TIME_UNIT_SEQUENCE = "time_unit_or_sequence_reasoning"
    ARITHMETIC_EXECUTION = "arithmetic_execution"
    OUTPUT_FORMAT = "output_format_or_answer_extraction"
    BENCHMARK_RISK = "benchmark_ambiguity_or_annotation_risk"


@dataclass(frozen=True)
class CategoryDefinition:
    """Content-free meaning and synthesis eligibility for one category."""

    category: FailureCategory
    observed_count: int
    definition: str
    automatically_targetable: bool
    independently_generatable: bool
    deterministically_verifiable: bool
    selected_for_pilot: bool
    selection_note: str


CATEGORY_DEFINITIONS: tuple[CategoryDefinition, ...] = (
    CategoryDefinition(
        FailureCategory.MULTI_STEP_BOOKKEEPING,
        68,
        "A required state update, term, branch, or final aggregation is omitted, duplicated, "
        "or applied in the wrong order.",
        True,
        True,
        True,
        True,
        "Most prevalent reasoning failure and naturally represented by arithmetic DAGs.",
    ),
    CategoryDefinition(
        FailureCategory.TARGET_INTERPRETATION,
        53,
        "The computation answers a related quantity rather than the quantity explicitly requested.",
        True,
        True,
        True,
        False,
        "Deferred because subtle wording and ambiguity make independent rendering riskier.",
    ),
    CategoryDefinition(
        FailureCategory.RATE_RATIO_PERCENTAGE,
        28,
        "A rate, ratio, percentage, fraction, weighted average, or denominator is modeled "
        "incorrectly.",
        True,
        True,
        True,
        True,
        "High exact-verification quality and strong diversity across controlled templates.",
    ),
    CategoryDefinition(
        FailureCategory.CONSTRAINT_DISCRETE,
        27,
        "A bounded, integral, allocation, capacity, remainder, or distribution constraint is "
        "violated or rounded incorrectly.",
        True,
        True,
        True,
        True,
        "Supports constructive generation plus independent bounded enumeration.",
    ),
    CategoryDefinition(
        FailureCategory.TIME_UNIT_SEQUENCE,
        24,
        "A unit conversion, elapsed-time boundary, recurrence, or ordered event is mishandled.",
        True,
        True,
        True,
        False,
        "Retained as a secondary tag and candidate for a later curriculum expansion.",
    ),
    CategoryDefinition(
        FailureCategory.ARITHMETIC_EXECUTION,
        22,
        "The mathematical plan is appropriate but an exact elementary operation is wrong.",
        True,
        True,
        True,
        False,
        "Generic control data already exercises this skill; defer targeted emphasis.",
    ),
    CategoryDefinition(
        FailureCategory.OUTPUT_FORMAT,
        69,
        "The terminal answer is missing, conflicting, truncated, malformed, or extracted from "
        "a value other than the model's clear intent.",
        True,
        True,
        True,
        False,
        "Handled through a separate output-contract track shared by both curricula.",
    ),
    CategoryDefinition(
        FailureCategory.BENCHMARK_RISK,
        2,
        "The prompt or reference appears ambiguous, internally inconsistent, or annotation-risky.",
        False,
        False,
        False,
        False,
        "Excluded from synthesis because uncertainty cannot be repaired by generated labels.",
    ),
)

SELECTED_REASONING_CATEGORIES: tuple[FailureCategory, ...] = (
    FailureCategory.MULTI_STEP_BOOKKEEPING,
    FailureCategory.RATE_RATIO_PERCENTAGE,
    FailureCategory.CONSTRAINT_DISCRETE,
)
OUTPUT_CONTRACT_TRACK_ID = "terminal-final-answer-contract-v1"


def taxonomy_contract_sha256() -> str:
    """Hash the category definitions and frozen pilot selection."""

    payload = {
        "categories": [asdict(definition) for definition in CATEGORY_DEFINITIONS],
        "selected_reasoning_categories": list(SELECTED_REASONING_CATEGORIES),
        "output_contract_track": OUTPUT_CONTRACT_TRACK_ID,
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
