from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

from foundry.phase2.asdiv import (
    FormulaNode,
    VerificationError,
    answer_type,
    canonical_sha256,
    file_sha256,
    normalize_text,
    serialize_fraction,
)

MATHQA_CARD_REVISION: Final = "c4f1cc784c04c4957b50c97858f23893b633eea6"
MATHQA_PARQUET_REVISION: Final = "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6"
MATHQA_TRAIN_SHA256: Final = "c16335ea4f7c9a8da44ccec52146d29e040582d2c11ca712fcfa2dd0ee964a99"
MATHQA_TRAIN_COUNT: Final = 29_837
MATHQA_SUBSET_SIZE: Final = 5_000
MATHQA_SELECTION_SEED: Final = "foundry-phase2-mathqa-subset-v1"
MAX_EXPONENT: Final = 10
MAX_RESULT_BITS: Final = 256

FAMILY_BOOKKEEPING: Final = "multi_step_bookkeeping_or_omission"
FAMILY_RATE: Final = "rate_ratio_percentage_or_average"
FAMILY_DISCRETE: Final = "constraint_distribution_or_discrete_reasoning"

_CONSTANTS: Final = {
    "const_0": Fraction(0),
    "const_0_25": Fraction("0.25"),
    "const_0_2778": Fraction("0.2778"),
    "const_0_33": Fraction("0.33"),
    "const_0_3937": Fraction("0.3937"),
    "const_1": Fraction(1),
    "const_10": Fraction(10),
    "const_100": Fraction(100),
    "const_1000": Fraction(1000),
    "const_12": Fraction(12),
    "const_180": Fraction(180),
    "const_1_6": Fraction("1.6"),
    "const_2": Fraction(2),
    "const_26": Fraction(26),
    "const_3": Fraction(3),
    "const_360": Fraction(360),
    "const_3600": Fraction(3600),
    "const_3_6": Fraction("3.6"),
    "const_4": Fraction(4),
    "const_5": Fraction(5),
    "const_52": Fraction(52),
    "const_6": Fraction(6),
    "const_60": Fraction(60),
}

_RATE_OPERATIONS: Final = frozenset(
    {
        "inverse",
        "negate_prob",
        "original_price_before_gain",
        "original_price_before_loss",
        "p_after_gain",
        "speed",
        "speed_in_still_water",
        "stream_speed",
    }
)
_DISCRETE_OPERATIONS: Final = frozenset(
    {
        "choose",
        "factorial",
        "gcd",
        "lcm",
        "max",
        "min",
        "permutation",
        "reminder",
        "cube_edge_by_volume",
        "diagonal",
        "quadrilateral_area",
        "rectangle_area",
        "rectangle_perimeter",
        "rhombus_area",
        "rhombus_perimeter",
        "square_area",
        "square_edge_by_area",
        "square_edge_by_perimeter",
        "square_perimeter",
        "surface_cube",
        "surface_rectangular_prism",
        "triangle_area",
        "triangle_area_three_edges",
        "triangle_perimeter",
        "volume_cube",
        "volume_rectangular_prism",
    }
)
_OPTION_DEPENDENT = re.compile(
    r"\b(?:which|following|option|choice|choices|statements?|expressions?)\b", re.IGNORECASE
)
_TOKEN = re.compile(
    r"\s*(?:(?P<number>\d+(?:\.\d+)?)|(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<punctuation>[(),+\-]))"
)
_OPTION_LABEL = re.compile(r"(?:^|,\s*)([a-e])\s*\)\s*", re.IGNORECASE)
_OPTION_NUMBER = re.compile(
    r"(?<![\w.])(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?)"
    r"(?![\w/])"
)


@dataclass(frozen=True)
class ProgramExecution:
    value: Fraction
    node: FormulaNode
    operation_sequence: tuple[str, ...]
    operation_count: int
    depth: int
    program_sha256: str
    program_structure_sha256: str


@dataclass(frozen=True)
class OptionValue:
    label: str
    text: str
    value: Fraction | None


@dataclass(frozen=True)
class _TokenValue:
    kind: str
    value: str


@dataclass(frozen=True)
class _EvaluatedNode:
    node: FormulaNode
    value: Fraction


def _checked(value: Fraction) -> Fraction:
    if value.numerator.bit_length() > MAX_RESULT_BITS or value.denominator.bit_length() > (
        MAX_RESULT_BITS
    ):
        raise VerificationError("mathqa_result_out_of_bounds")
    return value


def _exact_sqrt(value: Fraction) -> Fraction:
    if value < 0:
        raise VerificationError("mathqa_sqrt_negative")
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if numerator * numerator != value.numerator or denominator * denominator != value.denominator:
        raise VerificationError("mathqa_sqrt_not_exact")
    return Fraction(numerator, denominator)


def _exact_cuberoot(value: Fraction) -> Fraction:
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    numerator = round(magnitude.numerator ** (1 / 3))
    denominator = round(magnitude.denominator ** (1 / 3))
    if numerator**3 != magnitude.numerator or denominator**3 != magnitude.denominator:
        raise VerificationError("mathqa_cuberoot_not_exact")
    return Fraction(sign * numerator, denominator)


def _require_arity(operator: str, arguments: Sequence[Fraction], expected: int) -> None:
    if len(arguments) != expected:
        raise VerificationError(f"mathqa_{operator}_arity")


def _require_integer(operator: str, value: Fraction, *, nonnegative: bool = False) -> int:
    if value.denominator != 1:
        raise VerificationError(f"mathqa_{operator}_requires_integer")
    integer = value.numerator
    if nonnegative and integer < 0:
        raise VerificationError(f"mathqa_{operator}_requires_nonnegative")
    return integer


def _execute_operation(operator: str, arguments: Sequence[Fraction]) -> Fraction:
    if operator in {"add", "subtract", "multiply", "divide", "power"}:
        _require_arity(operator, arguments, 2)
        left, right = arguments
        if operator == "add":
            return _checked(left + right)
        if operator == "subtract":
            return _checked(left - right)
        if operator == "multiply":
            return _checked(left * right)
        if operator == "divide":
            if right == 0:
                raise VerificationError("mathqa_division_by_zero")
            return _checked(left / right)
        exponent = _require_integer(operator, right, nonnegative=True)
        if exponent > MAX_EXPONENT:
            raise VerificationError("mathqa_power_out_of_bounds")
        return _checked(left**exponent)

    if operator in {"negate", "inverse", "floor", "sqrt", "factorial"}:
        _require_arity(operator, arguments, 1)
        value = arguments[0]
        if operator == "negate":
            return -value
        if operator == "inverse":
            if value == 0:
                raise VerificationError("mathqa_division_by_zero")
            return _checked(1 / value)
        if operator == "floor":
            return Fraction(value.numerator // value.denominator)
        if operator == "sqrt":
            return _exact_sqrt(value)
        integer = _require_integer(operator, value, nonnegative=True)
        if integer > 100:
            raise VerificationError("mathqa_factorial_out_of_bounds")
        return _checked(Fraction(math.factorial(integer)))

    if operator in {"gcd", "lcm", "max", "min", "reminder", "choose", "permutation"}:
        _require_arity(operator, arguments, 2)
        left, right = arguments
        if operator == "max":
            return max(left, right)
        if operator == "min":
            return min(left, right)
        left_integer = _require_integer(operator, left, nonnegative=True)
        right_integer = _require_integer(operator, right, nonnegative=True)
        if operator == "gcd":
            return Fraction(math.gcd(left_integer, right_integer))
        if operator == "lcm":
            return Fraction(math.lcm(left_integer, right_integer))
        if operator == "reminder":
            if right_integer == 0:
                raise VerificationError("mathqa_division_by_zero")
            return Fraction(left_integer % right_integer)
        if right_integer > left_integer or left_integer > 1000:
            raise VerificationError(f"mathqa_{operator}_out_of_bounds")
        if operator == "choose":
            return Fraction(math.comb(left_integer, right_integer))
        return Fraction(math.perm(left_integer, right_integer))

    if operator == "rectangle_area":
        _require_arity(operator, arguments, 2)
        return _checked(arguments[0] * arguments[1])
    if operator == "rectangle_perimeter":
        _require_arity(operator, arguments, 2)
        return _checked(2 * (arguments[0] + arguments[1]))
    if operator == "square_area":
        _require_arity(operator, arguments, 1)
        return _checked(arguments[0] ** 2)
    if operator == "square_perimeter":
        _require_arity(operator, arguments, 1)
        return _checked(4 * arguments[0])
    if operator == "square_edge_by_perimeter":
        _require_arity(operator, arguments, 1)
        return _checked(arguments[0] / 4)
    if operator == "square_edge_by_area":
        _require_arity(operator, arguments, 1)
        return _exact_sqrt(arguments[0])
    if operator == "triangle_area":
        _require_arity(operator, arguments, 2)
        return _checked(arguments[0] * arguments[1] / 2)
    if operator == "triangle_perimeter":
        _require_arity(operator, arguments, 3)
        return _checked(sum(arguments, Fraction()))
    if operator == "triangle_area_three_edges":
        _require_arity(operator, arguments, 3)
        semi = sum(arguments, Fraction()) / 2
        return _exact_sqrt(
            semi * (semi - arguments[0]) * (semi - arguments[1]) * (semi - arguments[2])
        )
    if operator == "rhombus_area":
        _require_arity(operator, arguments, 2)
        return _checked(arguments[0] * arguments[1] / 2)
    if operator == "rhombus_perimeter":
        _require_arity(operator, arguments, 1)
        return _checked(4 * arguments[0])
    if operator == "quadrilateral_area":
        _require_arity(operator, arguments, 3)
        return _checked(arguments[0] * (arguments[1] + arguments[2]) / 2)
    if operator == "volume_rectangular_prism":
        _require_arity(operator, arguments, 3)
        return _checked(arguments[0] * arguments[1] * arguments[2])
    if operator == "volume_cube":
        _require_arity(operator, arguments, 1)
        return _checked(arguments[0] ** 3)
    if operator == "cube_edge_by_volume":
        _require_arity(operator, arguments, 1)
        return _exact_cuberoot(arguments[0])
    if operator == "surface_cube":
        _require_arity(operator, arguments, 1)
        return _checked(6 * arguments[0] ** 2)
    if operator == "surface_rectangular_prism":
        _require_arity(operator, arguments, 3)
        a, b, c = arguments
        return _checked(2 * (a * b + a * c + b * c))
    if operator == "diagonal":
        _require_arity(operator, arguments, 2)
        return _exact_sqrt(arguments[0] ** 2 + arguments[1] ** 2)

    if operator == "speed":
        _require_arity(operator, arguments, 2)
        if arguments[1] == 0:
            raise VerificationError("mathqa_division_by_zero")
        return _checked(arguments[0] / arguments[1])
    if operator == "stream_speed":
        _require_arity(operator, arguments, 2)
        return _checked((arguments[0] - arguments[1]) / 2)
    if operator == "speed_in_still_water":
        _require_arity(operator, arguments, 2)
        return _checked((arguments[0] + arguments[1]) / 2)
    if operator == "p_after_gain":
        _require_arity(operator, arguments, 2)
        return _checked(arguments[0] * (100 + arguments[1]) / 100)
    if operator == "original_price_before_gain":
        _require_arity(operator, arguments, 2)
        if 100 + arguments[1] == 0:
            raise VerificationError("mathqa_division_by_zero")
        return _checked(arguments[0] * 100 / (100 + arguments[1]))
    if operator == "original_price_before_loss":
        _require_arity(operator, arguments, 2)
        if 100 - arguments[1] == 0:
            raise VerificationError("mathqa_division_by_zero")
        return _checked(arguments[0] * 100 / (100 - arguments[1]))
    if operator == "negate_prob":
        _require_arity(operator, arguments, 1)
        return _checked(1 - arguments[0])

    raise VerificationError("mathqa_unsupported_operation")


def _tokenize(program: str) -> tuple[_TokenValue, ...]:
    normalized = normalize_text(program)
    if not normalized:
        raise VerificationError("mathqa_empty_program")
    tokens: list[_TokenValue] = []
    position = 0
    while position < len(normalized):
        match = _TOKEN.match(normalized, position)
        if match is None:
            raise VerificationError("mathqa_unknown_token")
        if (number := match.group("number")) is not None:
            tokens.append(_TokenValue("number", number))
        elif (identifier := match.group("identifier")) is not None:
            tokens.append(_TokenValue("identifier", identifier))
        else:
            tokens.append(_TokenValue("punctuation", cast(str, match.group("punctuation"))))
        position = match.end()
    return tuple(tokens)


class ProgramParser:
    def __init__(self, tokens: Sequence[_TokenValue]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse(self) -> _EvaluatedNode:
        parsed = self._parse_value()
        if self._index != len(self._tokens):
            raise VerificationError("mathqa_unexpected_token")
        return parsed

    def _peek(self, value: str | None = None) -> bool:
        if self._index >= len(self._tokens):
            return False
        return value is None or self._tokens[self._index].value == value

    def _consume(self, value: str | None = None) -> _TokenValue:
        if not self._peek(value):
            raise VerificationError("mathqa_unexpected_token")
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _parse_value(self) -> _EvaluatedNode:
        sign = 1
        if self._peek("+") or self._peek("-"):
            sign = -1 if self._consume().value == "-" else 1
        token = self._consume()
        if token.kind == "number":
            value = _checked(sign * Fraction(token.value))
            return _EvaluatedNode(FormulaNode("number", literal=serialize_fraction(value)), value)
        if token.kind != "identifier":
            raise VerificationError("mathqa_expected_value")
        if not self._peek("("):
            if sign != 1:
                raise VerificationError("mathqa_signed_constant")
            constant = _CONSTANTS.get(token.value)
            if constant is None:
                raise VerificationError("mathqa_unknown_constant")
            return _EvaluatedNode(
                FormulaNode("number", literal=serialize_fraction(constant)), constant
            )
        if sign != 1:
            raise VerificationError("mathqa_signed_function")
        self._consume("(")
        arguments: list[_EvaluatedNode] = []
        if not self._peek(")"):
            arguments.append(self._parse_value())
            while self._peek(","):
                self._consume(",")
                arguments.append(self._parse_value())
        self._consume(")")
        values = [argument.value for argument in arguments]
        value = _execute_operation(token.value, values)
        return _EvaluatedNode(
            FormulaNode(token.value, children=tuple(argument.node for argument in arguments)), value
        )


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


def _structure(node: FormulaNode) -> dict[str, object]:
    return {
        "operator": node.operator,
        "children": [_structure(child) for child in node.children],
    }


def execute_program(program: str) -> ProgramExecution:
    parsed = ProgramParser(_tokenize(program)).parse()
    operations = _operations(parsed.node)
    payload = {"program": parsed.node.payload(), "result": serialize_fraction(parsed.value)}
    return ProgramExecution(
        value=parsed.value,
        node=parsed.node,
        operation_sequence=operations,
        operation_count=len(operations),
        depth=_depth(parsed.node),
        program_sha256=canonical_sha256(payload),
        program_structure_sha256=canonical_sha256(_structure(parsed.node)),
    )


def parse_options(options: str) -> tuple[OptionValue, ...]:
    normalized = normalize_text(options)
    matches = list(_OPTION_LABEL.finditer(normalized))
    if [match.group(1).casefold() for match in matches] != list("abcde"):
        raise VerificationError("mathqa_malformed_options")
    result: list[OptionValue] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        text = normalized[match.end() : end].strip().strip(", ")
        if not text:
            raise VerificationError("mathqa_empty_option")
        result.append(OptionValue(match.group(1).casefold(), text, extract_option_value(text)))
    return tuple(result)


def extract_option_value(text: str) -> Fraction | None:
    normalized = normalize_text(text).replace("−", "-")
    matches = list(_OPTION_NUMBER.finditer(normalized))
    if len(matches) != 1:
        return None
    match = matches[0]
    remainder = (normalized[: match.start()] + " " + normalized[match.end() :]).strip()
    if re.search(r"\d", remainder) or any(marker in remainder for marker in ("=", ":", "±", "√")):
        return None
    token = match.group("number").replace(",", "").replace(" ", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            value = Fraction(int(numerator), int(denominator))
        else:
            value = Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None
    lowered = remainder.casefold()
    multiplier = Fraction(1)
    if "million" in lowered:
        multiplier = Fraction(1_000_000)
    elif "billion" in lowered:
        multiplier = Fraction(1_000_000_000)
    return _checked(value * multiplier)


def classify_family(category: str, operation_sequence: Sequence[str]) -> str:
    normalized_category = category.casefold()
    operations = set(operation_sequence)
    if normalized_category in {"gain", "physics"} or operations & _RATE_OPERATIONS:
        return FAMILY_RATE
    if normalized_category in {"geometry", "probability"} or operations & _DISCRETE_OPERATIONS:
        return FAMILY_DISCRETE
    return FAMILY_BOOKKEEPING


def _requires_options(problem: str) -> bool:
    return _OPTION_DEPENDENT.search(problem) is not None


def _row_hash(row: dict[str, str], row_index: int) -> str:
    return canonical_sha256(
        {
            "revision": MATHQA_PARQUET_REVISION,
            "row_index": row_index,
            "Problem": row["Problem"],
            "options": row["options"],
            "correct": row["correct"],
            "annotated_formula": row["annotated_formula"],
            "linear_formula": row["linear_formula"],
            "category": row["category"],
        }
    )


def _verify_row(row: dict[str, str], row_index: int, source_hash: str) -> dict[str, object]:
    problem = normalize_text(row["Problem"])
    options_text = normalize_text(row["options"])
    correct = normalize_text(row["correct"]).casefold()
    category = normalize_text(row["category"])
    program = normalize_text(row["annotated_formula"])
    if not problem or not program or not category:
        raise VerificationError("mathqa_missing_required_field")
    if correct not in set("abcde") or len(correct) != 1:
        raise VerificationError("mathqa_invalid_correct_option")
    options = parse_options(options_text)
    correct_option = next(item for item in options if item.label == correct)
    if correct_option.value is None:
        raise VerificationError("mathqa_correct_option_non_numeric")
    duplicate_values = [item for item in options if item.value == correct_option.value]
    if len(duplicate_values) != 1:
        raise VerificationError("mathqa_ambiguous_correct_value")
    first = execute_program(program)
    second = execute_program(program)
    if first != second:
        raise VerificationError("mathqa_program_replay_nondeterminism")
    if first.value != correct_option.value:
        raise VerificationError("mathqa_program_option_disagreement")
    include_options = _requires_options(problem)
    combined = f"{problem}\n{options_text}" if include_options else problem
    source_id = f"mathqa-train-{row_index:05d}"
    family = classify_family(category, first.operation_sequence)
    return {
        "source_corpus": "mathqa_train",
        "source_id": source_id,
        "source_row_index": row_index,
        "source_row_sha256": _row_hash(row, row_index),
        "source_url": "https://huggingface.co/datasets/allenai/math_qa",
        "source_file_sha256": source_hash,
        "source_revision": MATHQA_PARQUET_REVISION,
        "grade": category,
        "source_difficulty": category,
        "category": category,
        "body": "",
        "question": problem,
        "combined_question": combined,
        "original_options": options_text if include_options else "",
        "options_included": include_options,
        "correct_option": correct,
        "canonical_answer": serialize_fraction(first.value),
        "answer_type": answer_type(first.value),
        "answer_unit": "",
        "formula": program,
        "formula_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
        "question_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "family": family,
        "operation_sequence": list(first.operation_sequence),
        "operation_count": first.operation_count,
        "formula_depth": first.depth,
        "program_sha256": first.program_sha256,
        "program_structure_sha256": first.program_structure_sha256,
        "program_ast": first.node.payload(),
        "formula_replay_verified": True,
        "rationale_loaded": False,
    }


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


def _load_parquet_rows(path: Path) -> list[dict[str, str]]:
    parquet: Any = importlib.import_module("pyarrow.parquet")
    columns = ["Problem", "options", "correct", "annotated_formula", "linear_formula", "category"]
    table: Any = parquet.read_table(path, columns=columns)
    result: list[dict[str, str]] = []
    for index in range(table.num_rows):
        row: dict[str, str] = {}
        for column in columns:
            value: object = table.column(column)[index].as_py()
            if not isinstance(value, str):
                raise VerificationError("mathqa_non_string_field")
            row[column] = value
        result.append(row)
    return result


def _selection_key(row: dict[str, object]) -> str:
    operations = row.get("operation_sequence")
    if not isinstance(operations, list) or not operations:
        raise ValueError("verified MathQA row has no operation sequence")
    root_operation = operations[-1]
    return hashlib.sha256(
        (
            f"{MATHQA_SELECTION_SEED}:"
            f"{row['source_row_sha256']}:{row['category']}:{root_operation}:{row['answer_type']}"
        ).encode()
    ).hexdigest()


def verify_mathqa(parquet_path: Path, output_dir: Path) -> dict[str, object]:
    source_hash = file_sha256(parquet_path)
    if source_hash != MATHQA_TRAIN_SHA256:
        raise ValueError(f"unexpected MathQA train hash: {source_hash}")
    rows = _load_parquet_rows(parquet_path)
    if len(rows) != MATHQA_TRAIN_COUNT:
        raise ValueError(f"expected {MATHQA_TRAIN_COUNT} MathQA train rows, found {len(rows)}")

    verified: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        source_id = f"mathqa-train-{row_index:05d}"
        try:
            candidate = _verify_row(row, row_index, source_hash)
            verified.append(candidate)
            evidence.append(
                {
                    "source_id": source_id,
                    "source_row_sha256": candidate["source_row_sha256"],
                    "status": "verified",
                    "family": candidate["family"],
                    "program_sha256": candidate["program_sha256"],
                }
            )
        except VerificationError as error:
            rejection_counts[error.code] += 1
            evidence.append({"source_id": source_id, "status": "rejected", "reason": error.code})

    verified.sort(key=lambda item: (_selection_key(item), str(item["source_id"])))
    selected = verified[:MATHQA_SUBSET_SIZE]
    selected.sort(key=lambda item: str(item["source_id"]))
    if len(selected) != MATHQA_SUBSET_SIZE:
        raise ValueError("fewer than 5,000 verified MathQA train rows remain")
    if len({str(row["source_id"]) for row in selected}) != len(selected):
        raise RuntimeError("duplicate MathQA source ID")
    replay = [
        _verify_row(
            rows[cast(int, row["source_row_index"])],
            cast(int, row["source_row_index"]),
            source_hash,
        )
        for row in selected[:30]
    ]
    if replay != selected[:30]:
        raise RuntimeError("MathQA fixed verification replay is not exact")

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_hash = _write_lines(output_dir / "verification_evidence.jsonl", evidence)
    selected_hash = _write_lines(output_dir / "verified_subset.jsonl", selected)
    family_counts = Counter(str(row["family"]) for row in selected)
    category_counts = Counter(str(row["category"]) for row in selected)
    operation_counts = Counter(cast(list[str], row["operation_sequence"])[-1] for row in selected)
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_namespace": "allenai/math_qa",
        "dataset_card_revision": MATHQA_CARD_REVISION,
        "parquet_revision": MATHQA_PARQUET_REVISION,
        "train_artifact_sha256": source_hash,
        "source_split": "train",
        "source_count": len(rows),
        "verified_count": len(verified),
        "rejected_count": len(rows) - len(verified),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected_count": len(selected),
        "selection_seed": MATHQA_SELECTION_SEED,
        "selection_fields": [
            "stable_source_row_hash",
            "category",
            "formula_root_operation",
            "answer_type",
        ],
        "selection_performed_before_model_inference": True,
        "rationale_column_loaded": False,
        "validation_split_accessed": False,
        "test_split_accessed": False,
        "remote_code_executed": False,
        "trust_remote_code": False,
        "verified_replay_size": 30,
        "verified_replay_exact": True,
        "family_counts": dict(sorted(family_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "root_operation_counts": dict(sorted(operation_counts.items())),
        "evidence_sha256": evidence_hash,
        "selected_rows_sha256": selected_hash,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_dir / "verification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the pinned MathQA train artifact")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(verify_mathqa(args.parquet, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
