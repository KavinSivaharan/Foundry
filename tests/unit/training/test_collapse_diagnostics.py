from foundry.synthesis.contamination import DevelopmentQuestion
from foundry.training.collapse_diagnostics import _classify


def _row(response: str, **values: object) -> dict[str, object]:
    return {
        "stable_id": "a" * 64,
        "response": response,
        "output_tokens": values.pop("output_tokens", 20),
        "generation_truncated": values.pop("generation_truncated", False),
        "exact_format_compliant": values.pop("exact_format_compliant", False),
        "predicted_answer": values.pop("predicted_answer", None),
        "extraction_failure_category": values.pop(
            "extraction_failure_category", "no_terminal_answer"
        ),
        **values,
    }


def _question() -> DevelopmentQuestion:
    return DevelopmentQuestion(
        stable_id="a" * 64,
        row_index=0,
        question="An orchard has 12 trees and plants 5 more. How many trees are there?",
    )


def test_classifies_exact_terminal_contract() -> None:
    category, evidence = _classify(
        _row(
            "Add the two amounts.\nFinal answer: 17",
            exact_format_compliant=True,
            predicted_answer="17",
            extraction_failure_category=None,
        ),
        _question(),
    )
    assert category == "terminal_answer_contract_present"
    assert evidence["appears_to_answer_user"] is True


def test_classifies_question_echo_before_reasoning() -> None:
    category, evidence = _classify(
        _row("An orchard has 12 trees and plants 5 more. How many trees are there?"),
        _question(),
    )
    assert category == "prompt_or_question_echo"
    assert evidence["prefix_echo"] is True


def test_classifies_truncation_before_other_features() -> None:
    category, _ = _classify(
        _row(
            "First add 12 and 5, then continue explaining.",
            generation_truncated=True,
            output_tokens=768,
        ),
        _question(),
    )
    assert category == "token_limit_truncation"


def test_classifies_reasoning_without_terminal_answer() -> None:
    category, _ = _classify(
        _row("First add 12 and 5, so the total is 17."),
        _question(),
    )
    assert category == "reasoning_like_without_terminal_answer"
