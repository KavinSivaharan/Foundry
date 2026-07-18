"""Auditable invariants for typed semantic IR and compiled English."""

# ruff: noqa: E501  # invariant expressions are intentionally kept together

from __future__ import annotations

import re
from collections import Counter

from foundry.synthesis.realization.ir import (
    BookkeepingProblemIR,
    CompiledRealization,
    DiscreteProblemIR,
    ProblemIR,
    RateProblemIR,
    RateRelationKind,
    TargetKind,
)
from foundry.synthesis.realization.morphology import attributive_form, noun_form
from foundry.synthesis.schema import ExactValue

_BAD_PUNCTUATION = re.compile(r"(?:\.\.|;;|,,|\s+[?.!,;])")


def _expected_target(problem: ProblemIR) -> TargetKind:
    if isinstance(problem, BookkeepingProblemIR):
        return (
            TargetKind.GROUP_COUNT
            if problem.group_size is not None
            else TargetKind.REMAINING_QUANTITY
        )
    if isinstance(problem, RateProblemIR):
        return {
            RateRelationKind.RATE_TOTAL: TargetKind.TOTAL_QUANTITY,
            RateRelationKind.RATIO_SCALE: TargetKind.RATIO,
            RateRelationKind.PERCENTAGE: TargetKind.PERCENTAGE,
            RateRelationKind.WEIGHTED_MEAN: TargetKind.WEIGHTED_MEAN,
            RateRelationKind.COMBINED_RATE: TargetKind.TOTAL_QUANTITY,
        }[problem.relation_kind]
    if isinstance(problem, DiscreteProblemIR):
        return {
            "two_type_allocation": TargetKind.COUNT,
            "complete_packages": TargetKind.GROUP_COUNT,
            "equal_distribution": TargetKind.COUNT,
            "dual_capacity": TargetKind.CAPACITY,
        }[str(problem.relation_kind)]
    raise TypeError("unsupported problem IR")


def validate_realization(
    *, problem: ProblemIR, realization: CompiledRealization, answer: ExactValue
) -> tuple[str, ...]:
    """Return stable pre-contamination rejection reasons in architectural order."""

    reasons: list[str] = []
    if problem.target.kind is not _expected_target(problem):
        reasons.append("target_type_mismatch")
    if realization.rendered_target_kind is not problem.target.kind:
        reasons.append("rendered_target_type_mismatch")
    integral_targets = {
        TargetKind.COUNT,
        TargetKind.TOTAL_QUANTITY,
        TargetKind.REMAINING_QUANTITY,
        TargetKind.VALID_ASSIGNMENT_COUNT,
        TargetKind.CAPACITY,
        TargetKind.GROUP_COUNT,
    }
    if problem.target.kind in integral_targets and answer.denominator != 1:
        reasons.append("answer_type_mismatch")
    required = Counter(problem.required_node_ids)
    rendered = Counter(entry.node_id for entry in realization.coverage)
    if required != rendered:
        if any(rendered[node] == 0 for node in required):
            reasons.append("missing_semantic_node")
        if any(rendered[node] > 1 for node in required):
            reasons.append("duplicated_semantic_node")
        if any(node not in required for node in rendered):
            reasons.append("unexplained_semantic_node")
    if any(
        not (0 <= entry.clause_index < len(realization.clauses)) for entry in realization.coverage
    ):
        reasons.append("invalid_render_coverage_index")
    if (
        isinstance(problem, RateProblemIR)
        and problem.relation_kind is RateRelationKind.WEIGHTED_MEAN
    ):
        if len({(group.weight, group.value) for group in problem.groups}) != len(problem.groups):
            reasons.append("duplicated_semantic_slot")
    for use in realization.morphology_uses:
        try:
            expected = (
                attributive_form(use.lexeme)[0]
                if use.grammatical_role == "attributive"
                else noun_form(use.lexeme, use.quantity if use.quantity is not None else 2)[0]
            )
        except ValueError:
            reasons.append("unsupported_morphology")
            continue
        if expected != use.rendered:
            reasons.append("morphology_or_agreement_failure")
    if isinstance(problem, RateProblemIR) and problem.relation_kind in {
        RateRelationKind.RATE_TOTAL,
        RateRelationKind.COMBINED_RATE,
    }:
        expected_rate_units = 1 if problem.relation_kind is RateRelationKind.RATE_TOTAL else 2
        explicit = sum(use.denominator_rendered == "interval" for use in realization.unit_uses)
        if explicit < expected_rate_units:
            reasons.append("missing_rate_denominator")
    normalized_clauses = [" ".join(clause.lower().split()) for clause in realization.clauses]
    if not realization.grammar_complete or any(not clause for clause in normalized_clauses):
        reasons.append("grammar_metadata_failure")
    if len(normalized_clauses) != len(set(normalized_clauses)):
        reasons.append("duplicated_clause")
    if (
        not realization.question_clause.endswith("?")
        or realization.question_clause not in realization.text
    ):
        reasons.append("question_or_conclusion_failure")
    if _BAD_PUNCTUATION.search(realization.text):
        reasons.append("malformed_punctuation")
    if realization.licensed_elisions:
        reasons.append("illegal_noun_elision")
    return tuple(dict.fromkeys(reasons))
