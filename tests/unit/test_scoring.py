import pytest

from foundry.evaluation.scoring import (
    AnswerExtractionError,
    extract_final_integer,
    score_response,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Reasoning.\nFinal answer: 42", 42),
        ("Final answer: -7", -7),
        ("Final answer: +7", 7),
        ("Final answer: 1,234", 1234),
        ("Final answer: 42.0", 42),
        (r"Final answer: \boxed{42}", 42),
        ("Work used 3, 12, and 99.\n" r"Final answer: \boxed{-1,234.00}", -1234),
    ],
)
def test_normal_integer_formats(response: str, expected: int) -> None:
    assert extract_final_integer(response) == expected


def test_distracting_numbers_do_not_override_final_line() -> None:
    response = "First I considered 100, then 27, and rejected 999.\nFinal answer: 8"

    assert extract_final_integer(response) == 8


@pytest.mark.parametrize(
    "response",
    [
        "",
        "The answer is 42.",
        "Final answer: 42 apples",
        "Final answer: 12,34",
        "Final answer: 42.5",
        r"Final answer: \boxed{42",
        "Final answer: 1\nFinal answer: 2",
        "Final answer: 42\nAdditional explanation.",
        "Reasoning contains 42 but no required marker.",
    ],
)
def test_malformed_or_ambiguous_answers_are_rejected(response: str) -> None:
    with pytest.raises(AnswerExtractionError):
        extract_final_integer(response)


def test_score_reports_invalid_output_without_guessing() -> None:
    score = score_response("I saw 42 and 17.", "42")

    assert score.predicted is None
    assert score.expected == 42
    assert score.correct is False
    assert score.error is not None


def test_score_uses_exact_integer_equality() -> None:
    assert score_response("Final answer: 42.0", "42").correct is True
    assert score_response("Final answer: -42", "42").correct is False
