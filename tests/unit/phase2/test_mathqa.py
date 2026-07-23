from fractions import Fraction

import pytest

from foundry.phase2.asdiv import VerificationError
from foundry.phase2.mathqa import (
    FAMILY_BOOKKEEPING,
    FAMILY_DISCRETE,
    FAMILY_RATE,
    _verify_row,
    classify_family,
    execute_program,
    extract_option_value,
    parse_options,
)


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("add(2, multiply(3, 4))", Fraction(14)),
        ("divide(const_1, const_4)", Fraction(1, 4)),
        ("power(3, 4)", Fraction(81)),
        ("choose(10, 2)", Fraction(45)),
        ("rectangle_perimeter(3, 5)", Fraction(16)),
        ("triangle_area(6, 7)", Fraction(21)),
        ("cube_edge_by_volume(125)", Fraction(5)),
        ("speed(120, 3)", Fraction(40)),
        ("negate_prob(divide(1, 4))", Fraction(3, 4)),
    ],
)
def test_mathqa_executor_is_exact_and_deterministic(program: str, expected: Fraction) -> None:
    execution = execute_program(program)

    assert execution.value == expected
    assert execution == execute_program(program)
    assert len(execution.program_sha256) == 64
    assert len(execution.program_structure_sha256) == 64


@pytest.mark.parametrize(
    ("program", "reason"),
    [
        ("__import__(1)", "mathqa_unsupported_operation"),
        ("divide(1, 0)", "mathqa_division_by_zero"),
        ("sqrt(2)", "mathqa_sqrt_not_exact"),
        ("power(2, 11)", "mathqa_power_out_of_bounds"),
        ("circle_area(3)", "mathqa_unsupported_operation"),
        ("add(1, 2); multiply(3, 4)", "mathqa_unknown_token"),
        ("const_pi", "mathqa_unknown_constant"),
    ],
)
def test_mathqa_executor_fails_closed(program: str, reason: str) -> None:
    with pytest.raises(VerificationError, match=reason):
        execute_program(program)


def test_options_are_parsed_without_using_rationale() -> None:
    options = parse_options("a ) rs . 10 , b ) 12.5 % , c ) 3 / 4 , d ) none , e ) $ 20")

    assert [item.label for item in options] == list("abcde")
    assert [item.value for item in options] == [
        Fraction(10),
        Fraction(25, 2),
        Fraction(3, 4),
        None,
        Fraction(20),
    ]
    assert extract_option_value("2 : 3") is None
    assert extract_option_value("3 hours 30 minutes") is None


def test_family_classifier_uses_only_category_and_operations() -> None:
    assert classify_family("gain", ["add"]) == FAMILY_RATE
    assert classify_family("physics", ["divide"]) == FAMILY_RATE
    assert classify_family("probability", ["divide"]) == FAMILY_DISCRETE
    assert classify_family("general", ["choose"]) == FAMILY_DISCRETE
    assert classify_family("general", ["add"]) == FAMILY_BOOKKEEPING


def test_verified_row_retains_original_question_and_exact_label() -> None:
    raw = {
        "Problem": "what is two plus three ?",
        "options": "a ) 4 , b ) 5 , c ) 6 , d ) 7 , e ) 8",
        "correct": "b",
        "annotated_formula": "add(2, 3)",
        "linear_formula": "add(n0,n1)|",
        "category": "general",
    }

    verified = _verify_row(raw, 7, "a" * 64)

    assert verified["source_id"] == "mathqa-train-00007"
    assert verified["combined_question"] == raw["Problem"]
    assert verified["canonical_answer"] == "5"
    assert verified["rationale_loaded"] is False
    assert verified["formula_replay_verified"] is True


def test_option_dependent_question_retains_original_options() -> None:
    raw = {
        "Problem": "which of the following equals five ?",
        "options": "a ) 4 , b ) 5 , c ) 6 , d ) 7 , e ) 8",
        "correct": "b",
        "annotated_formula": "add(2, 3)",
        "linear_formula": "add(n0,n1)|",
        "category": "general",
    }

    verified = _verify_row(raw, 8, "b" * 64)

    assert verified["options_included"] is True
    assert verified["combined_question"] == f"{raw['Problem']}\n{raw['options']}"


def test_program_option_disagreement_is_rejected() -> None:
    raw = {
        "Problem": "what is two plus three ?",
        "options": "a ) 4 , b ) 6 , c ) 7 , d ) 8 , e ) 9",
        "correct": "b",
        "annotated_formula": "add(2, 3)",
        "linear_formula": "add(n0,n1)|",
        "category": "general",
    }

    with pytest.raises(VerificationError, match="mathqa_program_option_disagreement"):
        _verify_row(raw, 9, "c" * 64)


def test_ambiguous_correct_option_is_rejected() -> None:
    raw = {
        "Problem": "what is two plus three ?",
        "options": "a ) 5 , b ) 5 , c ) 6 , d ) 7 , e ) 8",
        "correct": "b",
        "annotated_formula": "add(2, 3)",
        "linear_formula": "add(n0,n1)|",
        "category": "general",
    }

    with pytest.raises(VerificationError, match="mathqa_ambiguous_correct_value"):
        _verify_row(raw, 10, "d" * 64)
