from foundry.training.format_alignment_audit import classify_completion


def _record(completion: str, *, enabled: bool = True) -> dict[str, object]:
    return {
        "training_completion": completion,
        "canonical_final_answer": "12",
        "output_contract_enabled": enabled,
        "deterministic_solution_trace": ["Add the two exact quantities: 7 + 5 = 12."],
    }


def test_exact_terminal_contract_is_recognized() -> None:
    result = classify_completion(
        _record("Add the two exact quantities: 7 + 5 = 12.\nFinal answer: 12")
    )
    assert result["exactly_one_clear_final_answer"] is True
    assert result["ends_with_exact_terminal"] is True
    assert result["answer_occurs_before_final_line"] is True
    assert result["contains_multiple_answer_markers"] is False


def test_missing_terminal_contract_is_detected() -> None:
    result = classify_completion(
        _record("Add the two exact quantities: 7 + 5 = 12.", enabled=False)
    )
    assert result["exactly_one_clear_final_answer"] is False
    assert result["ends_with_exact_terminal"] is False
    assert result["looks_like_internal_program_trace"] is True


def test_other_and_multiple_terminal_markers_are_detected() -> None:
    result = classify_completion(_record("Answer: 12\nFinal answer: 12"))
    assert result["uses_another_terminal_form"] is True
    assert result["contains_multiple_answer_markers"] is True
