"""Deterministic extraction of clear terminal numeric answers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Literal

from foundry.evaluation.scoring import parse_reference_integer

CANONICAL_EXTRACTOR_ID = "foundry-terminal-number-v2"

ExtractionFailureCategory = Literal[
    "empty_response",
    "generation_truncated",
    "conflicting_answers",
    "malformed_terminal_answer",
    "ambiguous_terminal_answer",
    "no_terminal_answer",
]

_GROUPED_INTEGER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_DECIMAL_NUMBER = rf"{_GROUPED_INTEGER}(?:\.\d+)?"
_ASCII_FRACTION = rf"{_GROUPED_INTEGER}\s*/\s*{_GROUPED_INTEGER}"
_LATEX_FRACTION = rf"\\frac\{{\s*{_GROUPED_INTEGER}\s*\}}\{{\s*{_GROUPED_INTEGER}\s*\}}"
_UNSIGNED_NUMBER = rf"(?:{_LATEX_FRACTION}|{_ASCII_FRACTION}|{_DECIMAL_NUMBER})"
_VALUE = rf"[+-]?\s*\$?\s*{_UNSIGNED_NUMBER}"
_CAPTURE = (
    rf"(?:\$?\s*\\boxed\{{\s*(?P<boxed>{_VALUE})\s*\}}\s*\$?"
    rf"|\*\*\s*(?P<bold>{_VALUE})\s*\*\*"
    rf"|\\\(\s*(?P<latex>{_VALUE})\s*\\\)"
    rf"|(?P<plain>{_VALUE})(?![\d,/]|\.\d))"
)
_DECORATED_CAPTURE = (
    rf"(?:\$?\s*\\boxed\{{\s*(?P<boxed>{_VALUE})\s*\}}\s*\$?"
    rf"|\*\*\s*(?P<bold>{_VALUE})\s*\*\*"
    rf"|\\\(\s*(?P<latex>{_VALUE})\s*\\\))"
)
_FORBIDDEN_UNIT_WORDS = (
    "about|after|although|and|approximately|because|before|but|from|hence|if|or|"
    "since|so|therefore|thus|when|where|which"
)
_UNIT_WORD = rf"(?!(?:{_FORBIDDEN_UNIT_WORDS})\b)[A-Za-z%°]+(?:-[A-Za-z%°]+)?"
_UNIT_SUFFIX = rf"(?:\s*%|\s+{_UNIT_WORD}(?:\s+{_UNIT_WORD}){{0,3}})?"
_TERMINAL_PUNCTUATION = r"\s*[.!]?\s*$"
_CONCLUSION_VERBS = (
    r"are|averages|bought|buys?|contains?|costs?|earned|earns?|equals|gave|gives?|"
    r"has|have|had|is|lost|made|makes?|needs?|paid|pays?|should\s+buy|spends?|spent|"
    r"takes?|took|uses?|used|was|were|will\s+be|will\s+make|would\s+use"
)
_LEADING_RESULT_VERBS = r"are|can\s+fit|do|does|is|will\s+fit|would\s+fit"

_TERMINAL_PATTERNS = (
    re.compile(
        rf"(?:^|\n)\s*(?:\*\*\s*)?Final answer\s*:\s*(?:\*\*\s*)?"
        rf"{_CAPTURE}{_UNIT_SUFFIX}{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|\n)\s*(?:\\\[\s*)?{_DECORATED_CAPTURE}(?:\s*\\\])?"
        rf"{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(rf"^{_CAPTURE}{_TERMINAL_PUNCTUATION}", re.IGNORECASE),
    re.compile(
        rf"\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:|is|equals|=|:)\s*"
        rf"{_CAPTURE}{_UNIT_SUFFIX}{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bthe\s+(?:answer|result|total|distance|amount|cost|price|time|length|"
        rf"count|sum|difference|product|quotient|area|perimeter|volume|speed|rate|"
        rf"age|number|value)\s*(?:is|equals|=)\s*{_CAPTURE}{_UNIT_SUFFIX}"
        rf"{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:therefore|thus|hence|so|finally|consequently)[,:]?\s*"
        rf"(?:.{{0,160}}?\b(?:is|are|equals|has|have|contains|costs?)\s*)?"
        rf"{_CAPTURE}{_UNIT_SUFFIX}{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b(?:therefore|thus|hence|so|finally|consequently)[,:]?\s*.{{0,180}}?"
        rf"\b(?:{_CONCLUSION_VERBS})\b[^\n0-9$+\-]{{0,100}}?{_CAPTURE}"
        rf"[^\n]{{0,160}}?{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:therefore|thus|hence|so|finally|consequently)[,:]?\s*"
        rf"[^\n]{{0,240}}?{_DECORATED_CAPTURE}[^\n]{{0,160}}?"
        rf"{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:therefore|thus|hence|so|finally|consequently)[,:]?\s*"
        rf"{_CAPTURE}\s+{_UNIT_WORD}(?:\s+{_UNIT_WORD}){{0,5}}\s+"
        rf"(?:{_LEADING_RESULT_VERBS})\b[^\n]{{0,180}}?{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:therefore|thus|hence|so|finally|consequently)[,:]?\s*"
        rf"[^\n0-9$+\-]{{1,240}}?{_CAPTURE}{_UNIT_SUFFIX}"
        rf"{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|\n)\s*(?:\*\*\s*)?Final answer\s*:\s*(?:\*\*\s*)?"
        rf"[^\n0-9$+\-]{{0,100}}?\b(?:is|are|equals|=)\s*{_CAPTURE}"
        rf"{_UNIT_SUFFIX}{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\\text\{{\s*Final answer:\s*\}}\s*{_CAPTURE}"
        rf"(?:\s*\\text\{{[^{{}}]{{1,60}}\}})?\s*\\\]\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|\n)\s*[A-Za-z][A-Za-z0-9_ ]{{0,40}}\s*=\s*{_CAPTURE}"
        rf"{_UNIT_SUFFIX}{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:^|\n)\s*(?:\\\[\s*)?"
        rf"(?:[A-Za-z][A-Za-z0-9_ ]{{0,40}}\s*=\s*)?"
        rf"(?:{_VALUE}\s*[+\-*/]\s*)+{_VALUE}\s*=\s*{_CAPTURE}"
        rf"{_UNIT_SUFFIX}(?:\s*\\\])?{_TERMINAL_PUNCTUATION}",
        re.IGNORECASE,
    ),
)

_GLOBAL_STRONG_PATTERNS = (
    re.compile(
        rf"(?:\*\*\s*)?Final answer\s*:\s*(?:\*\*\s*)?{_CAPTURE}",
        re.IGNORECASE,
    ),
    re.compile(rf"\\boxed\{{\s*(?P<boxed>{_VALUE})\s*\}}", re.IGNORECASE),
    re.compile(rf"\*\*\s*(?P<bold>{_VALUE})\s*\*\*", re.IGNORECASE),
    re.compile(
        rf"\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:|is|equals|=|:)\s*{_CAPTURE}",
        re.IGNORECASE,
    ),
)

_NUMBER_TOKEN = re.compile(rf"(?<![\d,]){_VALUE}(?![\d,])")
_COMMA_NUMBER = re.compile(r"[+-]?\s*\$?\s*\d+(?:,\d+)+")
_ANSWER_CUE = re.compile(
    r"\b(answer|result|total|therefore|thus|hence|finally|consequently)\b|"
    r"\\boxed|\*\*",
    re.IGNORECASE,
)
_CONFLICTING_TERMINAL_LIST = re.compile(
    rf"\b(?:answer|result|total|therefore|thus|hence|finally|consequently)\b"
    rf"[^\n]{{0,240}}?{_VALUE}\s+(?:and|or)\s+{_VALUE}[^\n]*$",
    re.IGNORECASE,
)
_UNFINISHED_END = re.compile(r"(?:[=:,+\-/]|(?<!\*)\*(?!\*)|\\boxed\{)\s*$")


class CanonicalExtractionError(ValueError):
    """Raised when no single clear terminal number can be extracted."""

    def __init__(self, category: ExtractionFailureCategory, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class CanonicalAnswerScore:
    """Canonical benchmark-answer result for one model response."""

    predicted: Fraction | None
    expected: int
    correct: bool
    error: str | None
    failure_category: ExtractionFailureCategory | None


def _captured_value(match: re.Match[str]) -> str:
    for name in ("boxed", "bold", "latex", "plain"):
        value = match.groupdict().get(name)
        if value is not None:
            return value
    raise CanonicalExtractionError(
        "malformed_terminal_answer",
        "terminal answer pattern did not capture a value",
    )


def _normalize_value(value: str) -> Fraction:
    compact = value.replace(" ", "").replace("$", "").replace(",", "")
    sign = 1
    if compact.startswith(("+", "-")):
        if compact[0] == "-":
            sign = -1
        compact = compact[1:]

    latex_fraction = re.fullmatch(r"\\frac\{(?P<numerator>\d+)\}\{(?P<denominator>\d+)\}", compact)
    ascii_fraction = re.fullmatch(r"(?P<numerator>\d+)/(?P<denominator>\d+)", compact)
    fraction_match = latex_fraction or ascii_fraction
    if fraction_match is not None:
        try:
            return sign * Fraction(
                int(fraction_match.group("numerator")),
                int(fraction_match.group("denominator")),
            )
        except ZeroDivisionError as error:
            raise CanonicalExtractionError(
                "malformed_terminal_answer",
                "terminal fraction has a zero denominator",
            ) from error

    try:
        decimal = Decimal(compact)
    except InvalidOperation as error:
        raise CanonicalExtractionError(
            "malformed_terminal_answer",
            "terminal answer is not a valid number",
        ) from error
    return sign * Fraction(decimal)


def _strong_candidate_values(response: str) -> set[Fraction]:
    values: set[Fraction] = set()
    for pattern in _GLOBAL_STRONG_PATTERNS:
        for match in pattern.finditer(response):
            try:
                values.add(_normalize_value(_captured_value(match)))
            except CanonicalExtractionError:
                continue
    return values


def _has_malformed_comma_number(text: str) -> bool:
    for match in _COMMA_NUMBER.finditer(text):
        token = match.group().replace(" ", "").replace("$", "").lstrip("+-")
        if "," in token and re.fullmatch(r"\d{1,3}(?:,\d{3})+", token) is None:
            return True
    return False


def extract_canonical_number(
    response: str,
    *,
    generation_truncated: bool = False,
) -> Fraction:
    """Extract one clear terminal number without guessing from arbitrary numbers.

    Accepted answers must use an explicit answer cue, conclusion cue, standalone boxed
    or bold terminal value, or a terminal equation/assignment. Intermediate numbers are
    ignored. Conflicting explicit candidates and truncated generations are rejected.
    Decimals and fractions are normalized exactly as :class:`fractions.Fraction` values.
    """

    if generation_truncated:
        raise CanonicalExtractionError(
            "generation_truncated",
            "generation reached its configured token limit",
        )
    stripped = response.strip()
    if not stripped:
        raise CanonicalExtractionError("empty_response", "response is empty")
    tail = stripped[-300:]
    if _ANSWER_CUE.search(tail) and _has_malformed_comma_number(tail):
        raise CanonicalExtractionError(
            "malformed_terminal_answer",
            "terminal answer uses invalid comma grouping",
        )
    if _UNFINISHED_END.search(stripped) or stripped.count("{") != stripped.count("}"):
        raise CanonicalExtractionError(
            "malformed_terminal_answer",
            "response ends with an unfinished answer expression",
        )
    if _CONFLICTING_TERMINAL_LIST.search(tail):
        raise CanonicalExtractionError(
            "conflicting_answers",
            "response contains multiple terminal answer candidates",
        )

    terminal_values: set[Fraction] = set()
    terminal_error: CanonicalExtractionError | None = None
    for pattern in _TERMINAL_PATTERNS:
        match = pattern.search(stripped)
        if match is None:
            continue
        try:
            terminal_values.add(_normalize_value(_captured_value(match)))
        except CanonicalExtractionError as error:
            terminal_error = error

    if len(terminal_values) > 1:
        raise CanonicalExtractionError(
            "conflicting_answers",
            "response contains conflicting terminal answers",
        )
    if not terminal_values:
        if terminal_error is not None:
            raise terminal_error
        if _NUMBER_TOKEN.search(tail) and _ANSWER_CUE.search(tail):
            raise CanonicalExtractionError(
                "ambiguous_terminal_answer",
                "answer-like terminal prose is not unambiguous",
            )
        raise CanonicalExtractionError(
            "no_terminal_answer",
            "response does not contain a clear terminal answer",
        )

    terminal_value = next(iter(terminal_values))
    strong_values = _strong_candidate_values(stripped)
    if any(value != terminal_value for value in strong_values):
        raise CanonicalExtractionError(
            "conflicting_answers",
            "response contains conflicting explicit answer candidates",
        )
    return terminal_value


def serialize_canonical_number(value: Fraction) -> int | str:
    """Return a JSON-safe exact representation of a normalized answer."""

    if value.denominator == 1:
        return value.numerator
    return f"{value.numerator}/{value.denominator}"


def score_canonical_answer(
    response: str,
    reference: str,
    *,
    generation_truncated: bool = False,
) -> CanonicalAnswerScore:
    """Compare the canonical terminal answer with one benchmark reference."""

    expected = parse_reference_integer(reference)
    try:
        predicted = extract_canonical_number(
            response,
            generation_truncated=generation_truncated,
        )
    except CanonicalExtractionError as error:
        return CanonicalAnswerScore(
            predicted=None,
            expected=expected,
            correct=False,
            error=str(error),
            failure_category=error.category,
        )
    return CanonicalAnswerScore(
        predicted=predicted,
        expected=expected,
        correct=predicted == expected,
        error=None,
        failure_category=None,
    )


def canonical_extractor_sha256() -> str:
    """Return a stable digest of the extractor's versioned grammar and rules."""

    specification = {
        "conflict_patterns": [pattern.pattern for pattern in _GLOBAL_STRONG_PATTERNS],
        "conflicting_terminal_list_pattern": _CONFLICTING_TERMINAL_LIST.pattern,
        "extractor_id": CANONICAL_EXTRACTOR_ID,
        "exact_normalization": "fractions.Fraction_decimal_and_fraction_v1",
        "json_non_integral_representation": "reduced_numerator_slash_denominator",
        "malformed_comma_rejection": True,
        "number_grammar": _VALUE,
        "plain_number_requires_entire_response": True,
        "require_balanced_braces": True,
        "reject_generation_truncation": True,
        "terminal_patterns": [pattern.pattern for pattern in _TERMINAL_PATTERNS],
        "unfinished_end_pattern": _UNFINISHED_END.pattern,
    }
    serialized = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
