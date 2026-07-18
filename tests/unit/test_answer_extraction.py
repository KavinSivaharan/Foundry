from fractions import Fraction

import pytest

from foundry.evaluation.answer_extraction import (
    CANONICAL_EXTRACTOR_ID,
    CanonicalExtractionError,
    canonical_extractor_sha256,
    extract_canonical_number,
    score_canonical_answer,
    serialize_canonical_number,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Reasoning used 10 and 32.\nFinal answer: 42", 42),
        ("Reasoning.\n" r"\[\boxed{42}\]", 42),
        ("Reasoning.\n**42**", 42),
        ("The answer is 42.", 42),
        ("The total is $42.", 42),
        ("The distance is 42 miles.", 42),
        ("Therefore, the final amount is **$1,234**.", 1234),
        (r"Thus, the result is \(-42.0\).", -42),
        ("Consequently, there are +42 books.", 42),
        ("Therefore, Jimmy needs to earn **330** more tickets to buy the prize.", 330),
        ("So, Amy earns a total of $126 after six days.", 126),
        ("Therefore, Launa made $2,790 in 8 weeks.", 2790),
        ("**Final Answer:** 12", 12),
        ("**Final Answer:** Freddy's sister is 2 years old.", 2),
        (r"\[\text{Final answer: } 4 \text{ gallons}\]", 4),
        ("100", 100),
        ("x = 42.000", 42),
        ("Final answer: -1,234", -1234),
        ("Final answer: 42.5", Fraction(85, 2)),
        ("The answer is 3/2.", Fraction(3, 2)),
        (r"\[\boxed{\frac{3}{2}}\]", Fraction(3, 2)),
        (r"Thus, the result is \(-1.25\).", Fraction(-5, 4)),
        ("Final answer: 12.5%", Fraction(25, 2)),
        ("The total is $1,234.50.", Fraction(2469, 2)),
        ("20 + 22 = 42", 42),
        (
            "Therefore, Billy must complete **12** Pomodoro cycles in order to "
            "finish his 4-hour workday.",
            12,
        ),
        ("Therefore, Sarah left her house at 1 AM.", 1),
        (
            "Therefore, 6 more dogs her size will fit in the wagon without breaking "
            "the 200-pound limit.",
            6,
        ),
    ],
)
def test_extracts_clear_terminal_number_formats(
    response: str,
    expected: int | Fraction,
) -> None:
    assert extract_canonical_number(response) == expected


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("", "empty_response"),
        ("The answer is 42.\nFinal answer: 43", "conflicting_answers"),
        ("Reasoning uses 10, 20, and 42 without a conclusion.", "no_terminal_answer"),
        (r"Therefore, the answer is \boxed{42", "malformed_terminal_answer"),
        ("The answer might be 42 or 43.", "conflicting_answers"),
        ("Final answer: 12,34", "malformed_terminal_answer"),
        ("Reasoning ends here.\n100", "no_terminal_answer"),
        ("The final calculation is still in progress: 40 +", "malformed_terminal_answer"),
        ("The answer is 3/0.", "malformed_terminal_answer"),
        ("The answer is 3/2.\nFinal answer: 2", "conflicting_answers"),
        ("Therefore, the values are 12 and 13.", "conflicting_answers"),
        ("Therefore, 40 plus 2 may produce 42.", "ambiguous_terminal_answer"),
        ("Therefore, 6 or 7 dogs will fit.", "conflicting_answers"),
    ],
)
def test_rejects_unclear_or_invalid_terminal_answers(response: str, category: str) -> None:
    with pytest.raises(CanonicalExtractionError) as captured:
        extract_canonical_number(response)

    assert captured.value.category == category


def test_rejects_generation_that_reached_the_token_limit() -> None:
    with pytest.raises(CanonicalExtractionError) as captured:
        extract_canonical_number("The answer is 42.", generation_truncated=True)

    assert captured.value.category == "generation_truncated"


def test_score_separates_extraction_from_correctness() -> None:
    extracted_but_wrong = score_canonical_answer("The answer is 41.5.", "42")
    rejected = score_canonical_answer("I considered 42.", "42")

    assert extracted_but_wrong.predicted == Fraction(83, 2)
    assert extracted_but_wrong.correct is False
    assert extracted_but_wrong.failure_category is None
    assert rejected.predicted is None
    assert rejected.correct is False
    assert rejected.failure_category == "no_terminal_answer"


@pytest.mark.parametrize(
    ("value", "serialized"),
    [(Fraction(42, 1), 42), (Fraction(3, 2), "3/2"), (Fraction(-5, 4), "-5/4")],
)
def test_serializes_canonical_numbers_exactly(
    value: Fraction,
    serialized: int | str,
) -> None:
    assert serialize_canonical_number(value) == serialized


def test_extractor_identity_is_versioned_and_hashed() -> None:
    assert CANONICAL_EXTRACTOR_ID == "foundry-terminal-number-v2"
    assert (
        canonical_extractor_sha256()
        == "e099d1c247968fed982cb849022ec3137b1694c15f23a65663a127b8158c06df"
    )
