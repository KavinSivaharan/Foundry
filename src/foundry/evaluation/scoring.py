"""Strict extraction and exact scoring for final integer answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.0+)?"
_FINAL_LINE = re.compile(
    rf"^Final answer:\s*(?:(?P<boxed>\\boxed\{{\s*(?P<boxed_value>{_NUMBER})\s*\}})"
    rf"|(?P<plain>{_NUMBER}))$"
)
_REFERENCE = re.compile(rf"^(?P<value>{_NUMBER})$")


class AnswerExtractionError(ValueError):
    """Raised when an output does not contain one unambiguous final integer."""


@dataclass(frozen=True)
class Score:
    """Exact-answer result for one response."""

    predicted: int | None
    expected: int
    correct: bool
    error: str | None


def _normalize_integral_number(value: str) -> int:
    normalized = value.replace(",", "")
    try:
        decimal = Decimal(normalized)
    except InvalidOperation as error:
        raise AnswerExtractionError("answer is not a valid number") from error
    integral = decimal.to_integral_value()
    if decimal != integral:
        raise AnswerExtractionError("answer is a non-integral decimal")
    return int(integral)


def extract_final_integer(response: str) -> int:
    """Extract only an exact final line, rejecting ambiguity and non-integers.

    Accepted examples include ``Final answer: 42``, ``Final answer: -7``,
    ``Final answer: 1,234``, ``Final answer: 42.0``, and
    ``Final answer: \\boxed{42}``. A decimal is accepted only when its fractional
    portion is all zeros. Numbers elsewhere in the reasoning are ignored.
    """

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        raise AnswerExtractionError("response is empty")
    marker_lines = [line for line in lines if line.startswith("Final answer:")]
    if len(marker_lines) != 1:
        raise AnswerExtractionError("response must contain exactly one Final answer: line")
    if lines[-1] != marker_lines[0]:
        raise AnswerExtractionError("Final answer: must be the last non-empty line")

    match = _FINAL_LINE.fullmatch(lines[-1])
    if match is None:
        raise AnswerExtractionError("final line does not match the required integer format")
    value = match.group("boxed_value") or match.group("plain")
    if value is None:
        raise AnswerExtractionError("final line did not contain a numeric value")
    return _normalize_integral_number(value)


def parse_reference_integer(reference: str) -> int:
    """Parse the pinned benchmark's raw integer answer without model-output syntax."""

    match = _REFERENCE.fullmatch(reference.strip())
    if match is None:
        raise AnswerExtractionError("benchmark reference is not an integer")
    return _normalize_integral_number(match.group("value"))


def score_response(response: str, reference: str) -> Score:
    """Compare one model response with one exact benchmark answer."""

    expected = parse_reference_integer(reference)
    try:
        predicted = extract_final_integer(response)
    except AnswerExtractionError as error:
        return Score(predicted=None, expected=expected, correct=False, error=str(error))
    return Score(
        predicted=predicted,
        expected=expected,
        correct=predicted == expected,
        error=None,
    )
