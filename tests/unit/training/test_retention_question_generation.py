import json
from pathlib import Path

import pytest

from foundry.training.retention import (
    QUESTION_GENERATION_CONFIG,
    RetentionItem,
    question_generation_configuration_sha256,
    question_generation_evidence,
    score_response,
)

FIXTURES = Path("tests/fixtures/training/retention_question_generation_v2_fixtures.json")


def _item(
    prompt: str = "Calculate 6 multiplied by 7 and finish with `Final answer: 42`.",
) -> RetentionItem:
    return RetentionItem(
        "fixture", "arithmetic", "multiplication", "numeric_terminal", prompt, "42"
    )


@pytest.mark.parametrize(
    "response",
    [
        "Would you like another example?",
        "Problem: What is 7 + 5?",
        "Can you provide more information?",
        "What is 7 + 5?",
        "Calculate 6 multiplied by 7?",
    ],
)
def test_true_positive_question_generation_fixtures(response: str) -> None:
    assert score_response(_item(), response)["question_generation"] is True


@pytest.mark.parametrize(
    ("item", "response"),
    [
        (_item(), "6 times 7 is 42.\nFinal answer: 42"),
        (
            _item('The supplied text is "What is 6 times 7?". Answer it.'),
            'The supplied text was "What is 6 times 7?".\nFinal answer: 42',
        ),
        (_item(), r"The work is \(6 \times 7 = ?\), so it is 42.\nFinal answer: 42"),
        (
            RetentionItem(
                "json",
                "format",
                "json",
                "json_exact",
                "Return the JSON object.",
                '{"question?":"stored"}',
            ),
            '{"question?":"stored"}',
        ),
        (_item(), "The question has a direct multiplication answer.\nFinal answer: 42"),
        (_item(), "The result is unquestionably 42.\nFinal answer: 42"),
        (
            _item('Answer the supplied question "What is 6 times 7?".'),
            'You supplied "What is 6 times 7?"; its answer is 42.\nFinal answer: 42',
        ),
        (_item(), "Final answer: 42"),
        (_item(), "Answer: 42\nFinal answer: 42"),
        (_item(), "6 × 7 → 42; confidence 100%.\nFinal answer: 42"),
    ],
)
def test_true_negative_question_generation_fixtures(item: RetentionItem, response: str) -> None:
    assert score_response(item, response)["question_generation"] is False


def test_mathematical_placeholder_mutations_are_consistent() -> None:
    prompt = _item().prompt
    assert question_generation_evidence(prompt, "6 × 7 = ?")["decision"] is False
    assert question_generation_evidence(prompt, "6 × 7 = 42")["decision"] is False
    assert question_generation_evidence(prompt, "6 × 7 is what?")["decision"] is True
    assert (
        question_generation_evidence(prompt, "6 × 7 = ?. Would you like another example?")[
            "decision"
        ]
        is True
    )


def test_exact_failure_structure_is_represented_generically() -> None:
    evidence = question_generation_evidence(
        _item().prompt,
        "Start with multiplication:\n"
        r"\( 6 \times 7 = ? \)"
        "\nPerform it:\n"
        r"\( 6 \times 7 = 42 \)"
        "\nFinal answer: 42",
    )
    assert evidence["decision"] is False
    assert evidence["question_marks"] == [
        {
            "start": 43,
            "end": 44,
            "classification": "mathematical_unknown_placeholder",
        }
    ]


def test_question_or_problem_header_remains_a_true_positive() -> None:
    assert (
        question_generation_evidence(_item().prompt, "Question: compute 6 × 7.")["triggered_rule"]
        == "unprotected_question_or_problem_header"
    )
    assert (
        question_generation_evidence(_item().prompt, "`Question: compute 6 × 7.`")["decision"]
        is False
    )


def test_frozen_question_generation_fixture_corpus() -> None:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["fixtures"]) == 15
    for fixture in payload["fixtures"]:
        item = RetentionItem(
            fixture["name"],
            "format" if fixture["kind"] == "json_exact" else "arithmetic",
            "fixture",
            fixture["kind"],
            fixture["prompt"],
            fixture["expected"],
        )
        assert (
            score_response(item, fixture["response"])["question_generation"]
            is fixture["expected_question_generation"]
        ), fixture["name"]


def test_question_generation_configuration_hash_is_tamper_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = question_generation_configuration_sha256()
    monkeypatch.setitem(QUESTION_GENERATION_CONFIG, "header_pattern", "tampered")
    assert question_generation_configuration_sha256() != before
