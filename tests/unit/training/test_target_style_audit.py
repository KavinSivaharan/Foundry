from typing import Any

from foundry.training.target_style_audit import analyze_target


class FixtureTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


def _record(trace: list[str], answer: str = "12") -> dict[str, Any]:
    return {
        "synthetic_id": "fixture",
        "group": "targeted",
        "rendered_question": (
            "A basket has seven tokens and receives five more. How many are there?"
        ),
        "deterministic_solution_trace": trace,
        "canonical_final_answer": answer,
        "training_completion": "\n".join(trace),
    }


def test_procedural_target_is_classified_and_final_contract_is_valid() -> None:
    result = analyze_target(
        _record(["The typed ledger starts with 7 tokens.", "Update 1 adds 5, giving 12."]),
        FixtureTokenizer(),
    )
    assert "procedural_or_program_trace_style" in result["categories"]
    assert "internal_operation_terminology" in result["categories"]
    assert result["final_answer_line_count"] == 1
    assert result["final_answer_is_last"] is True


def test_equation_target_is_concise_and_detects_answer_before_final() -> None:
    result = analyze_target(_record(["7 + 5 = 12."]), FixtureTokenizer())
    assert "concise_equation_based_reasoning" in result["categories"]
    assert "answer_before_final_line" in result["categories"]
    assert result["equation_count"] == 1
