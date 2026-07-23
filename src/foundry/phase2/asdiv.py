from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

ASDIV_SOURCE_SHA256: Final = "ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929"
ASDIV_SOURCE_COUNT: Final = 2305
MAX_EXPONENT: Final = 10
MAX_NUMBER_DIGITS: Final = 64
MAX_RESULT_BITS: Final = 256

BOOKKEEPING_TYPES: Final = frozenset(
    {
        "Addition",
        "Comparison",
        "Difference",
        "Multiplication",
        "Sequential-Operation",
        "Substraction",
        "Subtraction",
        "Sum",
        "Surplus",
    }
)
RATE_TYPES: Final = frozenset(
    {
        "Ratio",
        "TVQ-Change",
        "TVQ-Final",
        "TVQ-Initial",
        "UnitTrans",
    }
)
DISCRETE_TYPES: Final = frozenset(
    {
        "Algebra-1",
        "Algebra-2",
        "Ceil-Division",
        "Common-Division",
        "Floor-Division",
        "GCD",
        "LCM",
        "Number-Operation",
        "Number-Pattern",
        "Set-Operation",
    }
)

_TOKEN_PATTERN: Final = re.compile(
    r"\s*(?:(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)|"
    r"(?P<operator>[+\-*/^()%]))"
)
_ANSWER_NUMBER_PATTERN: Final = re.compile(
    r"(?<![\w.])(?P<number>[+-]?(?:"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)\s+\d+/\d+|"
    r"\d+/\d+|"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"))(?![\w/])"
)
_UNIT_TOKEN_PATTERN: Final = re.compile(r"[A-Za-z]+")
_UNIT_STOPWORDS: Final = frozenset(
    {
        "a",
        "an",
        "and",
        "approximately",
        "at",
        "each",
        "in",
        "of",
        "or",
        "per",
        "the",
        "total",
    }
)
_UNIT_ALIASES: Final = {
    "dollar": frozenset({"$", "dollar", "usd"}),
    "cent": frozenset({"cent", "penny", "pennie"}),
    "foot": frozenset({"foot", "feet", "ft"}),
    "inch": frozenset({"inch", "inche", "in"}),
    "hour": frozenset({"hour", "hr"}),
    "minute": frozenset({"minute", "min"}),
    "second": frozenset({"second", "sec"}),
    "percent": frozenset({"%", "percent", "percentage"}),
}


class VerificationError(ValueError):
    """A deterministic rejection with a content-free reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class FormulaNode:
    operator: str
    literal: str | None = None
    children: tuple[FormulaNode, ...] = ()

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"operator": self.operator}
        if self.literal is not None:
            payload["literal"] = self.literal
        if self.children:
            payload["children"] = [child.payload() for child in self.children]
        return payload


@dataclass(frozen=True)
class ParsedExpression:
    node: FormulaNode
    value: Fraction


@dataclass(frozen=True)
class FormulaExecution:
    value: Fraction
    left: FormulaNode
    right: FormulaNode
    operation_sequence: tuple[str, ...]
    operation_count: int
    depth: int
    program_sha256: str
    program_structure_sha256: str


@dataclass(frozen=True)
class AnswerExtraction:
    value: Fraction
    unit: str
    answer_type: str


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


def canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(normalized.split())


def serialize_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def answer_type(value: Fraction) -> str:
    if value.denominator == 1:
        return "integer"
    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    return "terminating_decimal" if denominator == 1 else "fraction"


def _checked(value: Fraction) -> Fraction:
    if value.numerator.bit_length() > MAX_RESULT_BITS or value.denominator.bit_length() > (
        MAX_RESULT_BITS
    ):
        raise VerificationError("formula_result_out_of_bounds")
    return value


def _tokenize(expression: str) -> tuple[Token, ...]:
    expression = expression.strip()
    if not expression:
        raise VerificationError("formula_empty_expression")
    tokens: list[Token] = []
    position = 0
    while position < len(expression):
        match = _TOKEN_PATTERN.match(expression, position)
        if match is None:
            raise VerificationError("formula_unknown_token")
        number = match.group("number")
        operator = match.group("operator")
        if number is not None:
            compact = number.replace(",", "")
            if len(compact.replace(".", "")) > MAX_NUMBER_DIGITS:
                raise VerificationError("formula_number_out_of_bounds")
            tokens.append(Token("number", compact))
        elif operator is not None:
            tokens.append(Token("operator", operator))
        position = match.end()
    return tuple(tokens)


class FormulaParser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse(self) -> ParsedExpression:
        parsed = self._parse_sum()
        if self._index != len(self._tokens):
            raise VerificationError("formula_unexpected_token")
        return parsed

    def _peek(self, value: str | None = None) -> bool:
        if self._index >= len(self._tokens):
            return False
        token = self._tokens[self._index]
        return value is None or token.value == value

    def _consume(self, value: str | None = None) -> Token:
        if not self._peek(value):
            raise VerificationError("formula_unexpected_token")
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _parse_sum(self) -> ParsedExpression:
        left = self._parse_product()
        while self._peek("+") or self._peek("-"):
            operator = self._consume().value
            right = self._parse_product()
            value = left.value + right.value if operator == "+" else left.value - right.value
            left = ParsedExpression(
                FormulaNode(operator, children=(left.node, right.node)), _checked(value)
            )
        return left

    def _parse_product(self) -> ParsedExpression:
        left = self._parse_power()
        while self._peek("*") or self._peek("/"):
            operator = self._consume().value
            right = self._parse_power()
            if operator == "/" and right.value == 0:
                raise VerificationError("formula_division_by_zero")
            value = left.value * right.value if operator == "*" else left.value / right.value
            left = ParsedExpression(
                FormulaNode(operator, children=(left.node, right.node)), _checked(value)
            )
        return left

    def _parse_power(self) -> ParsedExpression:
        left = self._parse_unary()
        if self._peek("^"):
            self._consume("^")
            right = self._parse_unary()
            if right.value.denominator != 1:
                raise VerificationError("formula_non_integer_exponent")
            exponent = right.value.numerator
            if exponent < 0 or exponent > MAX_EXPONENT:
                raise VerificationError("formula_exponent_out_of_bounds")
            left = ParsedExpression(
                FormulaNode("^", children=(left.node, right.node)),
                _checked(left.value**exponent),
            )
        return left

    def _parse_unary(self) -> ParsedExpression:
        if self._peek("+") or self._peek("-"):
            operator = self._consume().value
            child = self._parse_unary()
            value = child.value if operator == "+" else -child.value
            return ParsedExpression(
                FormulaNode(f"unary{operator}", children=(child.node,)), _checked(value)
            )
        return self._parse_postfix()

    def _parse_postfix(self) -> ParsedExpression:
        parsed = self._parse_primary()
        if self._peek("%"):
            self._consume("%")
            parsed = ParsedExpression(
                FormulaNode("percent", children=(parsed.node,)), _checked(parsed.value / 100)
            )
        return parsed

    def _parse_primary(self) -> ParsedExpression:
        if self._peek("("):
            self._consume("(")
            parsed = self._parse_sum()
            self._consume(")")
            return parsed
        token = self._consume()
        if token.kind != "number":
            raise VerificationError("formula_expected_number")
        try:
            value = Fraction(token.value)
        except (ValueError, ZeroDivisionError) as error:
            raise VerificationError("formula_invalid_number") from error
        return ParsedExpression(FormulaNode("number", literal=serialize_fraction(value)), value)


def _operations(node: FormulaNode) -> tuple[str, ...]:
    result: list[str] = []
    for child in node.children:
        result.extend(_operations(child))
    if node.operator != "number":
        result.append(node.operator)
    return tuple(result)


def _depth(node: FormulaNode) -> int:
    if not node.children:
        return 0
    return 1 + max(_depth(child) for child in node.children)


def _structure_payload(node: FormulaNode) -> dict[str, object]:
    payload: dict[str, object] = {"operator": node.operator}
    if node.children:
        payload["children"] = [_structure_payload(child) for child in node.children]
    return payload


def execute_formula(formula: str) -> FormulaExecution:
    normalized = unicodedata.normalize("NFKC", formula).strip()
    if normalized.count("=") != 1:
        raise VerificationError("formula_requires_single_equality")
    left_text, right_text = normalized.split("=", 1)
    left = FormulaParser(_tokenize(left_text)).parse()
    right = FormulaParser(_tokenize(right_text)).parse()
    if left.value != right.value:
        raise VerificationError("formula_equality_disagreement")
    operations = _operations(left.node) + _operations(right.node)
    payload = {
        "left": left.node.payload(),
        "right": right.node.payload(),
        "result": serialize_fraction(right.value),
    }
    structure_payload = {
        "left": _structure_payload(left.node),
        "right": _structure_payload(right.node),
    }
    return FormulaExecution(
        value=right.value,
        left=left.node,
        right=right.node,
        operation_sequence=operations,
        operation_count=len(operations),
        depth=max(_depth(left.node), _depth(right.node)),
        program_sha256=canonical_sha256(payload),
        program_structure_sha256=canonical_sha256(structure_payload),
    )


def _parse_number_token(token: str) -> Fraction:
    sign = -1 if token.startswith("-") else 1
    unsigned = token.lstrip("+-").replace(",", "")
    mixed = re.fullmatch(r"(\d+)\s+(\d+)/(\d+)", unsigned)
    if mixed is not None:
        denominator = int(mixed.group(3))
        if denominator == 0:
            raise VerificationError("answer_division_by_zero")
        value = Fraction(int(mixed.group(1))) + Fraction(int(mixed.group(2)), denominator)
        return sign * value
    try:
        return Fraction(token.replace(",", ""))
    except (ValueError, ZeroDivisionError) as error:
        raise VerificationError("answer_invalid_number") from error


def _stem_unit(token: str) -> str:
    lowered = token.casefold()
    irregular = {"feet": "foot", "inches": "inch", "pennies": "penny"}
    if lowered in irregular:
        return irregular[lowered]
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith("es") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("s") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


def extract_answer(answer: str) -> AnswerExtraction:
    normalized = normalize_text(answer).replace("−", "-")
    wrapped_fraction = re.fullmatch(r"-\((\d+/\d+)\)", normalized)
    if wrapped_fraction is not None:
        value = -_parse_number_token(wrapped_fraction.group(1))
        return AnswerExtraction(value, "", answer_type(value))
    matches = list(_ANSWER_NUMBER_PATTERN.finditer(normalized))
    if len(matches) != 1:
        raise VerificationError("answer_requires_single_number")
    match = matches[0]
    prefix = normalized[: match.start()].strip()
    suffix = normalized[match.end() :].strip()
    if prefix not in {"", "$", "£", "€", "("}:
        raise VerificationError("answer_unsupported_prefix")
    if any(marker in suffix for marker in (";", ":", "=")):
        raise VerificationError("answer_multiple_or_ambiguous")
    if re.search(r"\d", prefix + suffix):
        raise VerificationError("answer_multiple_or_ambiguous")
    if prefix == "(" and not suffix.startswith(")"):
        raise VerificationError("answer_unbalanced_parenthesis")
    unit = suffix
    if prefix in {"$", "£", "€"}:
        unit = f"{prefix} {unit}".strip()
    unit = unit.strip().strip("()., ")
    if re.search(r"[^A-Za-z$£€%\s.'\-/]", unit):
        raise VerificationError("answer_unsupported_unit_notation")
    value = _parse_number_token(match.group("number"))
    return AnswerExtraction(value, unit, answer_type(value))


def unit_is_compatible(unit: str, combined_question: str) -> bool:
    if not unit:
        return True
    question = normalize_text(combined_question).casefold()
    question_tokens = {_stem_unit(token) for token in _UNIT_TOKEN_PATTERN.findall(question)}
    answer_tokens = [
        _stem_unit(token)
        for token in _UNIT_TOKEN_PATTERN.findall(unit)
        if token.casefold() not in _UNIT_STOPWORDS
    ]
    symbols = {symbol for symbol in ("$", "£", "€", "%") if symbol in unit}
    if "$" in symbols and "$" not in question and "dollar" not in question_tokens:
        return False
    if "£" in symbols and "£" not in question and "pound" not in question_tokens:
        return False
    if "€" in symbols and "€" not in question and "euro" not in question_tokens:
        return False
    if "%" in symbols and "%" not in question and "percent" not in question_tokens:
        return False
    for token in answer_tokens:
        if token in question_tokens:
            continue
        aliases = _UNIT_ALIASES.get(token)
        if aliases is not None and (aliases & (question_tokens | set(question))):
            continue
        return False
    return True


def classify_family(
    solution_type: str,
    operation_sequence: Sequence[str],
    result_answer_type: str,
) -> str:
    del operation_sequence, result_answer_type
    if solution_type in BOOKKEEPING_TYPES:
        return "multi_step_bookkeeping_or_omission"
    if solution_type in RATE_TYPES:
        return "rate_ratio_percentage_or_average"
    if solution_type in DISCRETE_TYPES:
        return "constraint_distribution_or_discrete_reasoning"
    return "unsupported"


def _problem_text(problem: ET.Element, name: str) -> str:
    element = problem.find(name)
    if element is None or element.text is None:
        raise VerificationError(f"missing_{name.casefold().replace('-', '_')}")
    value = normalize_text(element.text)
    if not value:
        raise VerificationError(f"empty_{name.casefold().replace('-', '_')}")
    return value


def _base_record(problem: ET.Element, source_hash: str) -> dict[str, object]:
    source_id = normalize_text(problem.attrib.get("ID", ""))
    if not source_id:
        raise VerificationError("missing_source_id")
    body = _problem_text(problem, "Body")
    question = _problem_text(problem, "Question")
    solution = problem.find("Solution-Type")
    if solution is None or solution.text is None:
        raise VerificationError("missing_solution_type")
    solution_type = normalize_text(solution.text)
    answer = _problem_text(problem, "Answer")
    formula = _problem_text(problem, "Formula")
    combined = f"{body} {question}"
    return {
        "source_corpus": "asdiv_v1_0",
        "source_id": source_id,
        "grade": normalize_text(problem.attrib.get("Grade", "")),
        "source_url": normalize_text(problem.attrib.get("Source", "")),
        "body": body,
        "question": question,
        "combined_question": combined,
        "solution_type": solution_type,
        "subtypes": {key: normalize_text(value) for key, value in sorted(solution.attrib.items())},
        "answer_text": answer,
        "formula": formula,
        "source_file_sha256": source_hash,
        "question_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "formula_sha256": hashlib.sha256(formula.encode("utf-8")).hexdigest(),
    }


def _verified_record(base: dict[str, object]) -> dict[str, object]:
    formula = str(base["formula"])
    answer_text = str(base["answer_text"])
    combined_question = str(base["combined_question"])
    solution_type = str(base["solution_type"])
    first_execution = execute_formula(formula)
    second_execution = execute_formula(formula)
    if first_execution != second_execution:
        raise VerificationError("formula_replay_nondeterminism")
    first_answer = extract_answer(answer_text)
    second_answer = extract_answer(answer_text)
    if first_answer != second_answer:
        raise VerificationError("answer_replay_nondeterminism")
    if first_execution.value != first_answer.value:
        raise VerificationError("formula_answer_disagreement")
    if not unit_is_compatible(first_answer.unit, combined_question):
        raise VerificationError("unit_incompatible")
    family = classify_family(
        solution_type,
        first_execution.operation_sequence,
        first_answer.answer_type,
    )
    result = dict(base)
    result.update(
        {
            "status": "verified_supported" if family != "unsupported" else "verified_unsupported",
            "family": family,
            "canonical_answer": serialize_fraction(first_answer.value),
            "answer_type": first_answer.answer_type,
            "answer_unit": first_answer.unit,
            "operation_sequence": list(first_execution.operation_sequence),
            "operation_count": first_execution.operation_count,
            "formula_depth": first_execution.depth,
            "program_sha256": first_execution.program_sha256,
            "program_structure_sha256": first_execution.program_structure_sha256,
            "formula_replay_verified": True,
            "answer_replay_verified": True,
            "unit_compatible": True,
        }
    )
    return result


def _json_line(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def _write_lines(path: Path, rows: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = _json_line(row)
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def verify_asdiv(xml_path: Path, output_dir: Path) -> dict[str, object]:
    source_hash = file_sha256(xml_path)
    if source_hash != ASDIV_SOURCE_SHA256:
        raise RuntimeError(f"ASDiv XML hash mismatch: {source_hash}")
    root = ET.parse(xml_path).getroot()
    problems = list(root.iter("Problem"))
    if len(problems) != ASDIV_SOURCE_COUNT:
        raise RuntimeError(f"expected {ASDIV_SOURCE_COUNT} ASDiv rows, found {len(problems)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    supported: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    solution_type_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    parser_nondeterminism = 0

    for problem in problems:
        try:
            base = _base_record(problem, source_hash)
            source_id = str(base["source_id"])
            if source_id in seen_ids:
                duplicate_ids += 1
                raise VerificationError("duplicate_source_id")
            seen_ids.add(source_id)
            first = _verified_record(base)
            second = _verified_record(base)
            if first != second:
                parser_nondeterminism += 1
                raise VerificationError("row_replay_nondeterminism")
            rows.append(first)
            family = str(first["family"])
            family_counts[family] += 1
            solution_type_counts[str(first["solution_type"])] += 1
            if first["status"] == "verified_supported":
                supported.append(first)
        except VerificationError as error:
            rejection_counts[error.code] += 1
            try:
                rejected = _base_record(problem, source_hash)
            except VerificationError:
                rejected = {
                    "source_corpus": "asdiv_v1_0",
                    "source_file_sha256": source_hash,
                    "source_id": normalize_text(problem.attrib.get("ID", "")),
                }
            rejected.update({"status": "rejected", "rejection_reason": error.code})
            rows.append(rejected)

    rows.sort(key=lambda row: str(row.get("source_id", "")))
    supported.sort(key=lambda row: str(row["source_id"]))
    rows_hash = _write_lines(output_dir / "asdiv_rows.jsonl", rows)
    supported_hash = _write_lines(output_dir / "verified_supported_asdiv.jsonl", supported)
    verified_count = sum(1 for row in rows if str(row["status"]).startswith("verified_"))
    unsupported_count = family_counts["unsupported"]
    summary: dict[str, object] = {
        "schema_version": 1,
        "source": {
            "corpus": "ASDiv V1.0",
            "repository": "https://github.com/chaochun/nlu-asdiv-dataset.git",
            "commit": "883f90a9a65bf00304ba8f37423910fe743abc47",
            "tree": "2c3e8723c68436a2a6697329edfdf7fbd44e52ac",
            "xml_sha256": source_hash,
            "license": "CC BY-NC 4.0",
        },
        "source_count": len(problems),
        "unique_source_ids": len(seen_ids),
        "duplicate_source_ids": duplicate_ids,
        "verified_count": verified_count,
        "supported_verified_count": len(supported),
        "unsupported_verified_count": unsupported_count,
        "rejected_count": len(problems) - verified_count,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "solution_type_counts": dict(sorted(solution_type_counts.items())),
        "parser_nondeterminism": parser_nondeterminism,
        "formula_disagreements_accepted": 0,
        "rows_sha256": rows_hash,
        "supported_rows_sha256": supported_hash,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_dir / "asdiv_verification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the pinned ASDiv V1.0 corpus")
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = verify_asdiv(args.xml, args.output_dir)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
