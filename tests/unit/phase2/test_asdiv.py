import inspect
import json
from fractions import Fraction
from pathlib import Path

import pytest

from foundry.phase2 import asdiv
from foundry.phase2.asdiv import (
    VerificationError,
    answer_type,
    classify_family,
    execute_formula,
    extract_answer,
    normalize_text,
    unit_is_compatible,
    verify_asdiv,
)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("7 + 2 = 9", Fraction(9)),
        ("(9 - 6) * 2 = 6", Fraction(6)),
        ("1.25 + 0.75 = 2", Fraction(2)),
        ("3 / 4 = 0.75", Fraction(3, 4)),
        ("25% * 40 = 10", Fraction(10)),
        ("2 ^ 3 = 8", Fraction(8)),
        ("-4 + 7 = 3", Fraction(3)),
    ],
)
def test_formula_executor_uses_exact_safe_arithmetic(formula: str, expected: Fraction) -> None:
    execution = execute_formula(formula)

    assert execution.value == expected
    assert execution == execute_formula(formula)
    assert len(execution.program_sha256) == 64


def test_program_structure_hash_ignores_numeric_literals() -> None:
    first = execute_formula("7 + 2 = 9")
    second = execute_formula("70 + 20 = 90")

    assert first.program_sha256 != second.program_sha256
    assert first.program_structure_sha256 == second.program_structure_sha256


@pytest.mark.parametrize(
    ("formula", "reason"),
    [
        ("x + 2 = 3", "formula_unknown_token"),
        ("gcd(4, 8) = 4", "formula_unknown_token"),
        ("1 / 0 = 0", "formula_division_by_zero"),
        ("2 ^ 11 = 2048", "formula_exponent_out_of_bounds"),
        ("2 ^ 0.5 = 1", "formula_non_integer_exponent"),
        ("2 + 2 = 5", "formula_equality_disagreement"),
        ("2 + 2", "formula_requires_single_equality"),
        ("1 + 1 = 2; 2 + 2 = 4", "formula_requires_single_equality"),
    ],
)
def test_formula_executor_fails_closed(formula: str, reason: str) -> None:
    with pytest.raises(VerificationError, match=reason):
        execute_formula(formula)


@pytest.mark.parametrize(
    ("answer", "expected", "unit"),
    [
        ("9 (apples)", Fraction(9), "apples"),
        ("$1,234.50", Fraction(2469, 2), "$"),
        ("3/4 (mile)", Fraction(3, 4), "mile"),
        ("2 1/2 hours", Fraction(5, 2), "hours"),
        ("-(1/2)", Fraction(-1, 2), ""),
    ],
)
def test_answer_extraction_is_independent_and_exact(
    answer: str, expected: Fraction, unit: str
) -> None:
    extraction = extract_answer(answer)

    assert extraction.value == expected
    assert extraction.unit == unit


@pytest.mark.parametrize("answer", ["3; 4", "1:30", "about 4", "4 = four"])
def test_answer_extraction_rejects_multiple_or_ambiguous_values(answer: str) -> None:
    with pytest.raises(VerificationError):
        extract_answer(answer)


def test_unit_compatibility_uses_only_deterministic_text_identity() -> None:
    assert unit_is_compatible("apples", "A basket contains apples.")
    assert unit_is_compatible("feet", "A board is measured in foot-long sections.")
    assert unit_is_compatible("$", "Each item costs 4 dollars.")
    assert not unit_is_compatible("miles", "The problem asks for apples.")


@pytest.mark.parametrize(
    ("solution_type", "expected"),
    [
        ("Addition", "multi_step_bookkeeping_or_omission"),
        ("Ratio", "rate_ratio_percentage_or_average"),
        ("Common-Division", "constraint_distribution_or_discrete_reasoning"),
        ("Geometry", "unsupported"),
    ],
)
def test_family_classifier_is_frozen_and_non_llm(solution_type: str, expected: str) -> None:
    assert classify_family(solution_type, ("+",), answer_type(Fraction(4))) == expected


def test_verifier_writes_deterministic_ignored_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = tmp_path / "fixture.xml"
    xml.write_text(
        """<?xml version="1.0"?>
<ASDiv>
  <Problem ID="fixture-0001" Grade="2" Source="https://example.invalid/source">
    <Body>A box contains seven blue items.</Body>
    <Question>Two more are added. How many items are present?</Question>
    <Solution-Type>Addition</Solution-Type>
    <Answer>9 (items)</Answer>
    <Formula>7+2=9</Formula>
  </Problem>
</ASDiv>
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(asdiv, "ASDIV_SOURCE_SHA256", asdiv.file_sha256(xml))
    monkeypatch.setattr(asdiv, "ASDIV_SOURCE_COUNT", 1)

    first = verify_asdiv(xml, tmp_path / "first")
    second = verify_asdiv(xml, tmp_path / "second")

    assert first == second
    assert first["supported_verified_count"] == 1
    assert first["parser_nondeterminism"] == 0
    assert (tmp_path / "first" / "asdiv_rows.jsonl").read_bytes() == (
        tmp_path / "second" / "asdiv_rows.jsonl"
    ).read_bytes()
    row = json.loads((tmp_path / "first" / "verified_supported_asdiv.jsonl").read_text())
    assert row["combined_question"].count("seven blue items") == 1


def test_formula_executor_never_uses_eval() -> None:
    source = inspect.getsource(asdiv)
    forbidden_call = "".join(("ev", "al("))
    assert forbidden_call not in source


def test_normalization_changes_only_unicode_and_whitespace() -> None:
    assert normalize_text("  Twelve\r\nitems   remain.  ") == "Twelve items remain."
