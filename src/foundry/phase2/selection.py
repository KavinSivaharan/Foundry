from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from statistics import fmean
from typing import Final, cast

from foundry.phase2.asdiv import canonical_sha256, file_sha256
from foundry.phase2.capacity import (
    FAMILY_BOOKKEEPING,
    FAMILY_DISCRETE,
    FAMILY_RATE,
    answer_magnitude_bucket,
    question_token_bucket,
)
from foundry.synthesis.contamination import normalize_text

SELECTION_SEED: Final = "foundry-phase2-vetted-matched-selection-v1"
SIZES: Final = (300, 250, 200)
FAMILIES: Final = (FAMILY_BOOKKEEPING, FAMILY_RATE, FAMILY_DISCRETE)
QUOTAS: Final = {
    300: {
        "targeted": {FAMILY_BOOKKEEPING: 165, FAMILY_RATE: 70, FAMILY_DISCRETE: 65},
        "generic": {FAMILY_BOOKKEEPING: 100, FAMILY_RATE: 100, FAMILY_DISCRETE: 100},
    },
    250: {
        "targeted": {FAMILY_BOOKKEEPING: 138, FAMILY_RATE: 58, FAMILY_DISCRETE: 54},
        "generic": {FAMILY_BOOKKEEPING: 84, FAMILY_RATE: 83, FAMILY_DISCRETE: 83},
    },
    200: {
        "targeted": {FAMILY_BOOKKEEPING: 110, FAMILY_RATE: 47, FAMILY_DISCRETE: 43},
        "generic": {FAMILY_BOOKKEEPING: 67, FAMILY_RATE: 67, FAMILY_DISCRETE: 66},
    },
}
_OUTPUT_BUCKETS: Final = (
    (32, "1_to_32"),
    (64, "33_to_64"),
    (128, "65_to_128"),
    (256, "129_to_256"),
)
_TOKEN: Final = re.compile(r"[a-z0-9]+|<num>")


@dataclass(frozen=True)
class Candidate:
    source_id: str
    source_corpus: str
    family: str
    difficulty: str
    solution_type: str
    question_sha256: str
    program_sha256: str
    program_structure_sha256: str
    question_token_count: int
    question_token_bucket: str
    formula_depth: int
    operation_count: int
    answer_type: str
    answer_magnitude_bucket: str
    base_extractable: bool
    base_output_tokens: int
    base_output_token_bucket: str
    stable_rank: str


@dataclass(frozen=True)
class Match:
    targeted_source_id: str
    generic_source_id: str
    source_corpus: str
    cost: int


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: object = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(cast(dict[str, object], raw))
    return rows


def _string(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def _integer(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"field {key!r} must be an integer")
    return value


def _boolean(row: dict[str, object], key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"field {key!r} must be a boolean")
    return value


def output_token_bucket(value: int) -> str:
    for upper, label in _OUTPUT_BUCKETS:
        if value <= upper:
            return label
    return "257_plus"


def _source_label(value: str) -> str:
    if value == "asdiv_v1_0":
        return "ASDiv"
    if value == "mathqa_train":
        return "MathQA"
    raise ValueError(f"unsupported source corpus: {value}")


def _candidate(row: dict[str, object], prediction: dict[str, object]) -> Candidate:
    source_id = _string(row, "source_id")
    if source_id != _string(prediction, "source_id"):
        raise ValueError("candidate and prediction source IDs differ")
    if _boolean(prediction, "correct"):
        raise ValueError("selection candidate is not a base-model failure")
    if prediction.get("backend_error") is not None:
        raise ValueError("selection candidate has a backend error")
    source_corpus = _source_label(_string(row, "source_corpus"))
    difficulty = _string(row, "grade") if source_corpus == "ASDiv" else "not_available"
    solution_type = (
        _string(row, "solution_type") if source_corpus == "ASDiv" else _string(row, "category")
    )
    question_tokens = _integer(prediction, "question_token_count")
    output_tokens = _integer(prediction, "output_tokens")
    stable_rank = hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()
    return Candidate(
        source_id=source_id,
        source_corpus=source_corpus,
        family=_string(row, "family"),
        difficulty=difficulty,
        solution_type=solution_type,
        question_sha256=_string(row, "question_sha256"),
        program_sha256=_string(row, "program_sha256"),
        program_structure_sha256=_string(row, "program_structure_sha256"),
        question_token_count=question_tokens,
        question_token_bucket=question_token_bucket(question_tokens),
        formula_depth=_integer(row, "formula_depth"),
        operation_count=_integer(row, "operation_count"),
        answer_type=_string(row, "answer_type"),
        answer_magnitude_bucket=answer_magnitude_bucket(Fraction(_string(row, "canonical_answer"))),
        base_extractable=_boolean(prediction, "extractable"),
        base_output_tokens=output_tokens,
        base_output_token_bucket=output_token_bucket(output_tokens),
        stable_rank=stable_rank,
    )


def load_base_failures(
    source_paths: Sequence[Path], prediction_paths: Sequence[Path]
) -> tuple[list[Candidate], dict[str, dict[str, object]]]:
    if len(source_paths) != len(prediction_paths):
        raise ValueError("source and prediction path counts differ")
    candidates: list[Candidate] = []
    raw_by_id: dict[str, dict[str, object]] = {}
    for source_path, prediction_path in zip(source_paths, prediction_paths, strict=True):
        source_rows = {_string(row, "source_id"): row for row in _load_jsonl(source_path)}
        predictions = {_string(row, "source_id"): row for row in _load_jsonl(prediction_path)}
        if set(source_rows) != set(predictions):
            raise ValueError("clean candidate and prediction ID sets differ")
        for source_id in sorted(source_rows):
            prediction = predictions[source_id]
            if _boolean(prediction, "correct"):
                continue
            candidate = _candidate(source_rows[source_id], prediction)
            candidates.append(candidate)
            raw = dict(source_rows[source_id])
            raw.update(
                {
                    "base_extractable": candidate.base_extractable,
                    "base_output_tokens": candidate.base_output_tokens,
                    "base_output_sha256": prediction.get("output_sha256"),
                    "base_exact_format_compliant": prediction.get("exact_format_compliant"),
                    "base_generation_truncated": prediction.get("generation_truncated"),
                }
            )
            raw_by_id[source_id] = raw
    if len({item.source_id for item in candidates}) != len(candidates):
        raise ValueError("duplicate source ID across the combined pool")
    candidates.sort(key=lambda item: item.source_id)
    return candidates, raw_by_id


def _deduplicate(candidates: Sequence[Candidate]) -> tuple[list[Candidate], Counter[str]]:
    accepted: list[Candidate] = []
    seen_questions: set[str] = set()
    seen_programs: set[str] = set()
    rejected: Counter[str] = Counter()
    for item in sorted(candidates, key=lambda value: (value.stable_rank, value.source_id)):
        if item.question_sha256 in seen_questions:
            rejected["duplicate_question_sha256"] += 1
            continue
        if item.program_sha256 in seen_programs:
            rejected["duplicate_latent_program"] += 1
            continue
        seen_questions.add(item.question_sha256)
        seen_programs.add(item.program_sha256)
        accepted.append(item)
    accepted.sort(key=lambda item: item.source_id)
    return accepted, rejected


def _coverage_order(candidates: Sequence[Candidate]) -> list[Candidate]:
    strata: dict[tuple[object, ...], list[Candidate]] = defaultdict(list)
    for item in candidates:
        stratum = (
            item.program_structure_sha256,
            item.solution_type,
            item.operation_count,
            item.answer_type,
            item.difficulty,
        )
        strata[stratum].append(item)
    for values in strata.values():
        values.sort(key=lambda item: (item.stable_rank, item.source_id))
    ordered: list[Candidate] = []
    keys = sorted(strata, key=lambda key: canonical_sha256(key))
    while keys:
        remaining: list[tuple[object, ...]] = []
        for key in keys:
            values = strata[key]
            ordered.append(values.pop(0))
            if values:
                remaining.append(key)
        keys = remaining
    return ordered


def _partition_roles(candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    targeted: list[Candidate] = []
    generic: list[Candidate] = []
    for index, item in enumerate(_coverage_order(candidates)):
        (targeted if index % 2 == 0 else generic).append(item)
    return targeted, generic


def _categorical_cost(left: Candidate, right: Candidate) -> int:
    fields = (
        (left.difficulty, right.difficulty, 80),
        (left.question_token_bucket, right.question_token_bucket, 60),
        (left.answer_type, right.answer_type, 80),
        (left.answer_magnitude_bucket, right.answer_magnitude_bucket, 60),
        (left.base_extractable, right.base_extractable, 50),
        (left.base_output_token_bucket, right.base_output_token_bucket, 50),
    )
    return sum(weight for a, b, weight in fields if a != b)


def matching_cost(left: Candidate, right: Candidate) -> int:
    if left.source_corpus != right.source_corpus:
        raise ValueError("matching across source corpora is forbidden")
    numerical = (
        abs(left.question_token_count - right.question_token_count) * 2
        + abs(left.formula_depth - right.formula_depth) * 20
        + abs(left.operation_count - right.operation_count) * 20
        + abs(left.base_output_tokens - right.base_output_tokens)
    )
    tie = int(hashlib.sha256(f"{left.source_id}:{right.source_id}".encode()).hexdigest()[:4], 16)
    return (_categorical_cost(left, right) + numerical) * 100_000 + tie


def _question_shingles(text: str, size: int = 5) -> frozenset[tuple[str, ...]]:
    tokens = _TOKEN.findall(normalize_text(text, replace_numbers=True))
    if len(tokens) < size:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _near_duplicate(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> bool:
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= 0.85


def _generic_pool_without_near_duplicates(
    pool: Sequence[Candidate],
    targeted: Sequence[Candidate],
    raw_by_id: dict[str, dict[str, object]],
) -> tuple[list[Candidate], int]:
    targeted_shingles = [
        _question_shingles(_string(raw_by_id[item.source_id], "combined_question"))
        for item in targeted
    ]
    accepted: list[Candidate] = []
    rejected = 0
    for item in pool:
        shingles = _question_shingles(_string(raw_by_id[item.source_id], "combined_question"))
        if any(_near_duplicate(shingles, other) for other in targeted_shingles):
            rejected += 1
        else:
            accepted.append(item)
    return accepted, rejected


def _choose_generic(
    targeted: Sequence[Candidate],
    generic_pool: Sequence[Candidate],
    quotas: dict[str, int],
) -> list[Candidate]:
    available: dict[str, list[Candidate]] = {
        family: sorted(
            (item for item in generic_pool if item.family == family),
            key=lambda item: (item.stable_rank, item.source_id),
        )
        for family in FAMILIES
    }
    for family, quota in quotas.items():
        if len(available[family]) < quota:
            raise ValueError(f"generic {family} capacity is {len(available[family])}, need {quota}")
    remaining = dict(quotas)
    unmatched = list(targeted)
    selected: list[Candidate] = []
    used: set[str] = set()
    while unmatched:
        decisions: list[tuple[int, int, str, str, Candidate, Candidate]] = []
        for target in unmatched:
            options: list[tuple[int, str, str, Candidate]] = []
            for family in FAMILIES:
                if remaining[family] <= 0:
                    continue
                choices = [item for item in available[family] if item.source_id not in used]
                if not choices:
                    continue
                best = min(
                    choices,
                    key=lambda item: (
                        matching_cost(target, item),
                        item.stable_rank,
                        item.source_id,
                    ),
                )
                options.append((matching_cost(target, best), family, best.source_id, best))
            if not options:
                raise RuntimeError("quota-constrained generic matching exhausted candidates")
            options.sort(key=lambda value: value[:3])
            best_cost, best_family, _, best_item = options[0]
            second_cost = options[1][0] if len(options) > 1 else best_cost + 1_000_000_000
            regret = second_cost - best_cost
            decisions.append(
                (-regret, best_cost, target.stable_rank, best_family, target, best_item)
            )
        _, _, _, family, target, generic = min(decisions, key=lambda value: value[:4])
        selected.append(generic)
        used.add(generic.source_id)
        remaining[family] -= 1
        unmatched.remove(target)
    if any(remaining.values()):
        raise RuntimeError(f"generic quotas were not exhausted: {remaining}")
    return selected


def _hungarian(cost: Sequence[Sequence[int]]) -> list[int]:
    size = len(cost)
    if size == 0 or any(len(row) != size for row in cost):
        raise ValueError("Hungarian assignment requires a non-empty square matrix")
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row_index in range(1, size + 1):
        p[0] = row_index
        column0 = 0
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            current_row = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[current_row - 1][column - 1] - u[current_row] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            if math.isinf(delta):
                raise RuntimeError("assignment matrix has no complete solution")
            integer_delta = int(delta)
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += integer_delta
                    v[column] -= integer_delta
                else:
                    minimum[column] -= integer_delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[p[column] - 1] = column - 1
    if sorted(assignment) != list(range(size)):
        raise RuntimeError("Hungarian assignment is incomplete")
    return assignment


def _exact_matches(targeted: Sequence[Candidate], generic: Sequence[Candidate]) -> list[Match]:
    ordered_targeted = sorted(targeted, key=lambda item: item.source_id)
    ordered_generic = sorted(generic, key=lambda item: item.source_id)
    assignment = _hungarian(
        [[matching_cost(left, right) for right in ordered_generic] for left in ordered_targeted]
    )
    return [
        Match(
            targeted_source_id=left.source_id,
            generic_source_id=ordered_generic[column].source_id,
            source_corpus=left.source_corpus,
            cost=matching_cost(left, ordered_generic[column]),
        )
        for left, column in zip(ordered_targeted, assignment, strict=True)
    ]


def _smd(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("SMD inputs must be non-empty and equally sized")
    left_mean = fmean(left)
    right_mean = fmean(right)
    if len(left) == 1:
        return 0.0 if left_mean == right_mean else math.inf
    left_variance = sum((value - left_mean) ** 2 for value in left) / (len(left) - 1)
    right_variance = sum((value - right_mean) ** 2 for value in right) / (len(right) - 1)
    pooled = math.sqrt((left_variance + right_variance) / 2)
    if pooled == 0:
        return 0.0 if left_mean == right_mean else math.inf
    return abs(left_mean - right_mean) / pooled


def _categorical_balance(
    targeted: Sequence[Candidate], generic: Sequence[Candidate], field: str
) -> dict[str, float]:
    targeted_counts = Counter(str(getattr(item, field)) for item in targeted)
    generic_counts = Counter(str(getattr(item, field)) for item in generic)
    levels = sorted(set(targeted_counts) | set(generic_counts))
    size = len(targeted)
    return {
        level: abs(targeted_counts[level] / size - generic_counts[level] / size) for level in levels
    }


def balance_report(
    targeted: Sequence[Candidate], generic: Sequence[Candidate]
) -> dict[str, object]:
    numerical_fields = (
        "question_token_count",
        "formula_depth",
        "operation_count",
        "base_output_tokens",
    )
    categorical_fields = (
        "source_corpus",
        "difficulty",
        "question_token_bucket",
        "answer_type",
        "answer_magnitude_bucket",
        "base_extractable",
        "base_output_token_bucket",
    )
    numerical = {
        field: _smd(
            [float(getattr(item, field)) for item in targeted],
            [float(getattr(item, field)) for item in generic],
        )
        for field in numerical_fields
    }
    categorical = {
        field: _categorical_balance(targeted, generic, field) for field in categorical_fields
    }
    source_exact = Counter(item.source_corpus for item in targeted) == Counter(
        item.source_corpus for item in generic
    )
    numerical_pass = all(value <= 0.10 for value in numerical.values())
    categorical_pass = all(
        difference <= 0.05 for field in categorical.values() for difference in field.values()
    )
    return {
        "numerical_smd": numerical,
        "categorical_absolute_proportion_difference": categorical,
        "source_composition_exact": source_exact,
        "numerical_gate_passed": numerical_pass,
        "categorical_gate_passed": categorical_pass,
        "matching_quality_gate_passed": source_exact and numerical_pass and categorical_pass,
    }


def _write_jsonl(path: Path, rows: Iterable[object]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _content_free(item: Candidate, arm: str) -> dict[str, object]:
    payload = asdict(item)
    payload.pop("stable_rank")
    payload["arm"] = arm
    return payload


def _attempt_size(
    size: int,
    candidates: Sequence[Candidate],
    raw_by_id: dict[str, dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for item in candidates:
        grouped[(item.source_corpus, item.family)].append(item)
    target_quotas = QUOTAS[size]["targeted"]
    generic_quotas = QUOTAS[size]["generic"]
    source_family_quotas: dict[str, dict[str, dict[str, int]]] = {
        "ASDiv": {"targeted": {}, "generic": {}},
        "MathQA": {"targeted": {}, "generic": {}},
    }
    for family in FAMILIES:
        asdiv_each = min(
            len(grouped[("ASDiv", family)]) // 2,
            target_quotas[family],
            generic_quotas[family],
        )
        source_family_quotas["ASDiv"]["targeted"][family] = asdiv_each
        source_family_quotas["ASDiv"]["generic"][family] = asdiv_each
        source_family_quotas["MathQA"]["targeted"][family] = target_quotas[family] - asdiv_each
        source_family_quotas["MathQA"]["generic"][family] = generic_quotas[family] - asdiv_each

    targeted: list[Candidate] = []
    generic_pools: dict[str, list[Candidate]] = defaultdict(list)
    for source in ("ASDiv", "MathQA"):
        for family in FAMILIES:
            target_role, generic_role = _partition_roles(grouped[(source, family)])
            target_need = source_family_quotas[source]["targeted"][family]
            generic_need = source_family_quotas[source]["generic"][family]
            if len(target_role) < target_need or len(generic_role) < generic_need:
                raise ValueError(
                    f"{source} {family} role capacity is "
                    f"{len(target_role)}/{len(generic_role)}, need {target_need}/{generic_need}"
                )
            targeted.extend(_coverage_order(target_role)[:target_need])
            generic_pools[source].extend(generic_role)

    generic: list[Candidate] = []
    near_duplicate_rejections = 0
    matches: list[Match] = []
    for source in ("ASDiv", "MathQA"):
        source_targeted = [item for item in targeted if item.source_corpus == source]
        clean_generic, rejected = _generic_pool_without_near_duplicates(
            generic_pools[source], source_targeted, raw_by_id
        )
        near_duplicate_rejections += rejected
        source_generic = _choose_generic(
            source_targeted, clean_generic, source_family_quotas[source]["generic"]
        )
        generic.extend(source_generic)
        matches.extend(_exact_matches(source_targeted, source_generic))

    if len(targeted) != size or len(generic) != size:
        raise RuntimeError("selected arm sizes differ from the requested size")
    targeted_ids = {item.source_id for item in targeted}
    generic_ids = {item.source_id for item in generic}
    all_selected = [*targeted, *generic]
    unique_gate = (
        not targeted_ids & generic_ids
        and len({item.question_sha256 for item in all_selected}) == size * 2
        and len({item.program_sha256 for item in all_selected}) == size * 2
    )
    if not unique_gate:
        raise RuntimeError("selected arms violate ID, question, or latent-program uniqueness")
    balance = balance_report(targeted, generic)
    output_dir.mkdir(parents=True, exist_ok=True)
    targeted_rows = [
        {**raw_by_id[item.source_id], "selection_arm": "targeted"}
        for item in sorted(targeted, key=lambda value: value.source_id)
    ]
    generic_rows = [
        {**raw_by_id[item.source_id], "selection_arm": "generic"}
        for item in sorted(generic, key=lambda value: value.source_id)
    ]
    targeted_hash = _write_jsonl(output_dir / "targeted_full.jsonl", targeted_rows)
    generic_hash = _write_jsonl(output_dir / "generic_full.jsonl", generic_rows)
    targeted_manifest_hash = _write_jsonl(
        output_dir / "targeted_manifest.jsonl",
        [_content_free(item, "targeted") for item in sorted(targeted, key=lambda v: v.source_id)],
    )
    generic_manifest_hash = _write_jsonl(
        output_dir / "generic_manifest.jsonl",
        [_content_free(item, "generic") for item in sorted(generic, key=lambda v: v.source_id)],
    )
    match_hash = _write_jsonl(
        output_dir / "exact_matches.jsonl",
        [asdict(item) for item in sorted(matches, key=lambda value: value.targeted_source_id)],
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "selection_seed": SELECTION_SEED,
        "selected_size_per_arm": size,
        "targeted_family_counts": dict(sorted(Counter(item.family for item in targeted).items())),
        "generic_family_counts": dict(sorted(Counter(item.family for item in generic).items())),
        "targeted_source_counts": dict(
            sorted(Counter(item.source_corpus for item in targeted).items())
        ),
        "generic_source_counts": dict(
            sorted(Counter(item.source_corpus for item in generic).items())
        ),
        "source_family_quotas": source_family_quotas,
        "base_failures_only": True,
        "targeted_coverage_fields": [
            "formula_structure",
            "solution_type_or_category",
            "operation_count",
            "answer_type",
            "grade_or_source_difficulty",
        ],
        "generic_family_balanced": len(set(generic_quotas.values())) <= 2,
        "role_partition_disjoint": True,
        "unique_questions_and_programs": unique_gate,
        "near_duplicate_definition": "number-neutral five-token-shingle Jaccard >= 0.85",
        "generic_candidates_rejected_as_target_near_duplicates": near_duplicate_rejections,
        "cross_arm_near_duplicates": 0,
        "assignment_algorithm": (
            "exact deterministic Hungarian assignment after quota-constrained selection"
        ),
        "matching": balance,
        "targeted_full_sha256": targeted_hash,
        "generic_full_sha256": generic_hash,
        "targeted_manifest_sha256": targeted_manifest_hash,
        "generic_manifest_sha256": generic_manifest_hash,
        "exact_match_sha256": match_hash,
        "gsm1k_used_for_contamination_only": True,
        "gsm1k_row_performance_used": False,
        "sealed_final_accessed": False,
    }
    summary["selection_gate_passed"] = bool(balance["matching_quality_gate_passed"])
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def select_largest_feasible(
    *,
    source_paths: Sequence[Path],
    prediction_paths: Sequence[Path],
    output_root: Path,
) -> dict[str, object]:
    loaded, raw_by_id = load_base_failures(source_paths, prediction_paths)
    candidates, duplicate_rejections = _deduplicate(loaded)
    attempts: list[dict[str, object]] = []
    for size in SIZES:
        attempt_dir = output_root / f"size_{size}"
        try:
            summary = _attempt_size(size, candidates, raw_by_id, attempt_dir)
            attempts.append(
                {
                    "size": size,
                    "status": "passed" if summary["selection_gate_passed"] else "failed",
                    "summary_sha256": summary["summary_sha256"],
                    "matching": summary["matching"],
                }
            )
            if summary["selection_gate_passed"]:
                result = dict(summary)
                result["attempts"] = attempts
                result["input_base_failure_count"] = len(loaded)
                result["eligible_unique_count"] = len(candidates)
                result["deduplication_rejections"] = dict(sorted(duplicate_rejections.items()))
                result["asdiv_source_sha256"] = file_sha256(source_paths[0])
                result["mathqa_source_sha256"] = file_sha256(source_paths[1])
                result["asdiv_prediction_sha256"] = file_sha256(prediction_paths[0])
                result["mathqa_prediction_sha256"] = file_sha256(prediction_paths[1])
                result["result_sha256"] = canonical_sha256(result)
                (output_root / "selection_result.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                return result
        except (RuntimeError, ValueError) as error:
            attempts.append({"size": size, "status": "failed", "reason": str(error)})
    stop: dict[str, object] = {
        "schema_version": 1,
        "status": "stopped_no_feasible_approved_size",
        "attempts": attempts,
        "input_base_failure_count": len(loaded),
        "eligible_unique_count": len(candidates),
        "deduplication_rejections": dict(sorted(duplicate_rejections.items())),
    }
    stop["result_sha256"] = canonical_sha256(stop)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selection_result.json").write_text(
        json.dumps(stop, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    raise RuntimeError("no approved matched size passed; see selection_result.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select matched Phase 2 targeted and generic arms")
    parser.add_argument("--asdiv", type=Path, required=True)
    parser.add_argument("--asdiv-predictions", type=Path, required=True)
    parser.add_argument("--mathqa", type=Path, required=True)
    parser.add_argument("--mathqa-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = select_largest_feasible(
        source_paths=(args.asdiv, args.mathqa),
        prediction_paths=(args.asdiv_predictions, args.mathqa_predictions),
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
