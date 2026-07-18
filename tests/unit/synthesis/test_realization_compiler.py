"""Architectural regression tests for the typed realization compiler."""

from __future__ import annotations

from dataclasses import replace

import pytest

from foundry.synthesis.generators.bookkeeping import generate_bookkeeping
from foundry.synthesis.generators.discrete import generate_discrete
from foundry.synthesis.generators.rates import generate_rates
from foundry.synthesis.realization import compile_problem, select_plan, validate_realization
from foundry.synthesis.realization.ir import (
    CompiledRealization,
    MorphologyUse,
    RateProblemIR,
    RenderedUnitUse,
    TargetKind,
)
from foundry.synthesis.realization.morphology import BOX, PERSON, SHELF, noun_form
from foundry.synthesis.schema import DifficultyLevel


@pytest.mark.parametrize(
    ("lexeme", "singular", "plural"),
    ((SHELF, "shelf", "shelves"), (PERSON, "person", "people"), (BOX, "box", "boxes")),
)
def test_explicit_irregular_morphology(lexeme: object, singular: str, plural: str) -> None:
    assert noun_form(lexeme, 1)[0] == singular  # type: ignore[arg-type]
    assert noun_form(lexeme, 2)[0] == plural  # type: ignore[arg-type]


@pytest.mark.parametrize("generator", (generate_bookkeeping, generate_rates, generate_discrete))
def test_compiler_sweep_has_complete_typed_realizations(generator: object) -> None:
    for variant in range(36):
        draft = generator(  # type: ignore[operator]
            seed=44_000 + variant,
            difficulty=(DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD)[
                variant % 3
            ],
            variant=variant,
            output_contract_enabled=variant % 5 == 0,
        )
        assert draft.rendered_question == draft.realization.text
        assert not validate_realization(
            problem=draft.problem_ir,
            realization=draft.realization,
            answer=draft.canonical_final_answer,
        )


def _replace_realization(
    realization: CompiledRealization, **changes: object
) -> CompiledRealization:
    return replace(realization, **changes)


@pytest.mark.parametrize(
    ("defect_id", "expected"),
    (
        ("attributive_plural_a", "morphology_or_agreement_failure"),
        ("attributive_plural_b", "morphology_or_agreement_failure"),
        ("attributive_plural_c", "morphology_or_agreement_failure"),
        ("grouping_noun_a", "morphology_or_agreement_failure"),
        ("grouping_noun_b", "morphology_or_agreement_failure"),
        ("illegal_elision", "illegal_noun_elision"),
        ("irregular_plural", "morphology_or_agreement_failure"),
    ),
)
def test_sanitized_surface_defects_are_rejected(defect_id: str, expected: str) -> None:
    draft = generate_bookkeeping(
        seed=91,
        difficulty=DifficultyLevel.MEDIUM,
        variant=4,
        output_contract_enabled=False,
    )
    realization = draft.realization
    if defect_id == "illegal_elision":
        damaged = _replace_realization(realization, licensed_elisions=("missing-object",))
    else:
        damaged = _replace_realization(
            realization,
            morphology_uses=(MorphologyUse(SHELF, "head", 2, "shelfs"),),
        )
    assert expected in validate_realization(
        problem=draft.problem_ir,
        realization=damaged,
        answer=draft.canonical_final_answer,
    )


def test_sanitized_weighted_group_duplication_is_rejected() -> None:
    draft = generate_rates(
        seed=191,
        difficulty=DifficultyLevel.HARD,
        variant=3,
        output_contract_enabled=False,
    )
    problem = draft.problem_ir
    assert isinstance(problem, RateProblemIR)
    duplicated = replace(problem, groups=(problem.groups[0], problem.groups[0]))
    with pytest.raises(ValueError, match="semantically unique"):
        compile_problem(duplicated, select_plan(seed=191, variant=3, family="rates"))


def test_sanitized_weighted_target_mismatch_is_rejected() -> None:
    draft = generate_rates(
        seed=192,
        difficulty=DifficultyLevel.MEDIUM,
        variant=3,
        output_contract_enabled=False,
    )
    problem = draft.problem_ir
    assert isinstance(problem, RateProblemIR)
    wrong = replace(problem, target=replace(problem.target, kind=TargetKind.COUNT))
    assert "target_type_mismatch" in validate_realization(
        problem=wrong,
        realization=draft.realization,
        answer=draft.canonical_final_answer,
    )


def test_sanitized_missing_rate_denominator_is_rejected() -> None:
    draft = generate_rates(
        seed=193,
        difficulty=DifficultyLevel.EASY,
        variant=4,
        output_contract_enabled=False,
    )
    damaged = _replace_realization(
        draft.realization,
        unit_uses=(RenderedUnitUse("rate", "items", None),),
    )
    assert "missing_rate_denominator" in validate_realization(
        problem=draft.problem_ir,
        realization=damaged,
        answer=draft.canonical_final_answer,
    )


def test_sanitized_capacity_target_mismatch_is_rejected() -> None:
    draft = generate_discrete(
        seed=194,
        difficulty=DifficultyLevel.HARD,
        variant=3,
        output_contract_enabled=False,
    )
    problem = draft.problem_ir
    wrong = replace(problem, target=replace(problem.target, kind=TargetKind.COUNT))
    assert "target_type_mismatch" in validate_realization(
        problem=wrong,
        realization=draft.realization,
        answer=draft.canonical_final_answer,
    )


def test_missing_and_duplicated_semantic_nodes_are_rejected() -> None:
    draft = generate_bookkeeping(
        seed=195,
        difficulty=DifficultyLevel.EASY,
        variant=1,
        output_contract_enabled=False,
    )
    missing = _replace_realization(draft.realization, coverage=draft.realization.coverage[1:])
    duplicated = _replace_realization(
        draft.realization,
        coverage=(*draft.realization.coverage, draft.realization.coverage[0]),
    )
    assert "missing_semantic_node" in validate_realization(
        problem=draft.problem_ir, realization=missing, answer=draft.canonical_final_answer
    )
    assert "duplicated_semantic_node" in validate_realization(
        problem=draft.problem_ir, realization=duplicated, answer=draft.canonical_final_answer
    )
