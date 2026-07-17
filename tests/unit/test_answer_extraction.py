import pytest

from foundry.evaluation.answer_extraction import (
    CANONICAL_EXTRACTOR_ID,
    CanonicalExtractionError,
    canonical_extractor_sha256,
    extract_canonical_integer,
    score_canonical_answer,
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
    ],
)
def test_extracts_clear_terminal_integer_formats(response: str, expected: int) -> None:
    assert extract_canonical_integer(response) == expected


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("", "empty_response"),
        ("The answer is 42.\nFinal answer: 43", "conflicting_answers"),
        ("Reasoning uses 10, 20, and 42 without a conclusion.", "no_terminal_answer"),
        ("The answer is 42.5.", "non_integral_decimal"),
        (r"Therefore, the answer is \boxed{42", "malformed_terminal_answer"),
        ("The answer might be 42 or 43.", "ambiguous_terminal_answer"),
        ("Final answer: 12,34", "malformed_terminal_answer"),
        ("Reasoning ends here.\n100", "no_terminal_answer"),
        ("The final calculation is still in progress: 40 +", "malformed_terminal_answer"),
    ],
)
def test_rejects_unclear_or_invalid_terminal_answers(response: str, category: str) -> None:
    with pytest.raises(CanonicalExtractionError) as captured:
        extract_canonical_integer(response)

    assert captured.value.category == category


def test_rejects_generation_that_reached_the_token_limit() -> None:
    with pytest.raises(CanonicalExtractionError) as captured:
        extract_canonical_integer("The answer is 42.", generation_truncated=True)

    assert captured.value.category == "generation_truncated"


def test_score_separates_extraction_from_correctness() -> None:
    extracted_but_wrong = score_canonical_answer("The answer is 41.", "42")
    rejected = score_canonical_answer("I considered 42.", "42")

    assert extracted_but_wrong.predicted == 41
    assert extracted_but_wrong.correct is False
    assert extracted_but_wrong.failure_category is None
    assert rejected.predicted is None
    assert rejected.correct is False
    assert rejected.failure_category == "no_terminal_answer"


def test_extractor_identity_is_versioned_and_hashed() -> None:
    assert CANONICAL_EXTRACTOR_ID == "foundry-terminal-integer-v1"
    assert (
        canonical_extractor_sha256()
        == "ffce6538526f9aa21e05ce4d9d6830ec71d3a6334a23fa1e9c7beef3c2053946"
    )
