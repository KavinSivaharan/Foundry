from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

from foundry.phase2.asdiv import canonical_sha256, file_sha256
from foundry.phase2.selection import (
    QUOTAS,
    Candidate,
    Match,
    _deduplicate,
    _exact_matches,
    _near_duplicate,
    _question_shingles,
    balance_report,
    load_base_failures,
)
from foundry.synthesis.contamination import normalized_text_sha256

REPAIR_CONTRACT: Final = "foundry-vetted-matching-repair-v1"
EXPECTED_INPUT_FREEZE_SHA256: Final = (
    "0e6332e2933cfb71d1266e8bbee9c5201f72f63a42fba8f81c80fdbba0965979"
)
TARGET_SIZE: Final = 200
MAX_SMD: Final = 0.10
MAX_CATEGORICAL_DIFFERENCE: Final = 0.05
CONTINUOUS_FIELDS: Final = (
    "question_token_count",
    "base_output_tokens",
    "formula_depth",
    "operation_count",
)
CATEGORICAL_FIELDS: Final = (
    "source_corpus",
    "difficulty",
    "question_token_bucket",
    "answer_type",
    "answer_magnitude_bucket",
    "base_extractable",
    "base_output_token_bucket",
)
EXPECTED_BASELINE_SMD: Final = {
    "question_token_count": 0.0,
    "base_output_tokens": 0.022589325494615863,
    "formula_depth": 0.11389459246177541,
    "operation_count": 0.10876528809635315,
}


@dataclass(frozen=True)
class RepairConfig:
    contract: str = REPAIR_CONTRACT
    target_size: int = TARGET_SIZE
    maximum_smd: float = MAX_SMD
    maximum_categorical_difference: float = MAX_CATEGORICAL_DIFFERENCE

    def validate_frozen(self) -> None:
        if asdict(self) != asdict(RepairConfig()):
            raise ValueError("matching-repair configuration differs from the frozen contract")


@dataclass(frozen=True)
class RepairInputs:
    candidates: tuple[Candidate, ...]
    raw_by_id: dict[str, dict[str, object]]
    targeted_ids: tuple[str, ...]
    generic_ids: tuple[str, ...]
    normalized_by_id: dict[str, str]
    input_freeze: dict[str, object]


@dataclass(frozen=True)
class PassingRepair:
    arm: str
    removed: str
    added: str
    targeted_ids: tuple[str, ...]
    generic_ids: tuple[str, ...]
    numerical_smd: dict[str, float]
    categorical_maximum: float
    categorical_total: float
    pairwise_matching_cost: int

    @property
    def objective(self) -> tuple[object, ...]:
        smds = self.numerical_smd.values()
        return (
            max(smds),
            sum(value * value for value in self.numerical_smd.values()),
            self.categorical_total,
            self.pairwise_matching_cost,
            self.targeted_ids,
            self.generic_ids,
        )


@dataclass(frozen=True)
class SearchEvidence:
    checked_replacements: int
    legal_replacements: int
    passing_replacements: int
    selected: PassingRepair
    matches: tuple[Match, ...]


def _load_json(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a JSON object")
    return cast(dict[str, object], raw)


def _load_ids(path: Path) -> tuple[str, ...]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: object = json.loads(line)
            if not isinstance(raw, dict) or not isinstance(raw.get("source_id"), str):
                raise ValueError(f"{path}:{line_number} has no source_id")
            ids.append(cast(str, raw["source_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate source IDs")
    return tuple(ids)


def _text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def _normalized_question_sha256(row: dict[str, object]) -> str:
    return normalized_text_sha256(_text(row, "combined_question"))


def _candidate_payload(candidate: Candidate, normalized_by_id: dict[str, str]) -> dict[str, object]:
    payload = asdict(candidate)
    payload.pop("stable_rank")
    payload["question_normalized_sha256"] = normalized_by_id[candidate.source_id]
    return payload


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from the frozen repair input")


def _validate_freeze_document(document: dict[str, object]) -> None:
    stored = document.get("input_freeze_sha256")
    if stored != EXPECTED_INPUT_FREEZE_SHA256:
        raise ValueError("input-freeze identity differs from the authorized repair")
    payload = dict(document)
    payload.pop("input_freeze_sha256", None)
    _require_equal(canonical_sha256(payload), stored, "input-freeze content")
    _require_equal(document.get("repair_contract"), REPAIR_CONTRACT, "repair contract")
    _require_equal(document.get("source_rows_reparsed"), False, "source parsing state")
    _require_equal(document.get("contamination_rerun"), False, "contamination state")
    _require_equal(document.get("model_inference_rerun"), False, "model inference state")


def load_and_verify_inputs(
    *,
    asdiv: Path,
    asdiv_predictions: Path,
    mathqa: Path,
    mathqa_predictions: Path,
    baseline_targeted: Path,
    baseline_generic: Path,
    baseline_selection: Path,
    input_freeze_path: Path,
) -> RepairInputs:
    freeze = _load_json(input_freeze_path)
    _validate_freeze_document(freeze)
    _require_equal(file_sha256(asdiv), freeze["asdiv_clean_source_sha256"], "ASDiv clean source")
    _require_equal(file_sha256(mathqa), freeze["mathqa_clean_source_sha256"], "MathQA clean source")
    _require_equal(
        file_sha256(asdiv_predictions),
        freeze["asdiv_prediction_sha256"],
        "ASDiv base predictions",
    )
    _require_equal(
        file_sha256(mathqa_predictions),
        freeze["mathqa_prediction_sha256"],
        "MathQA base predictions",
    )
    _require_equal(
        file_sha256(baseline_targeted),
        freeze["failed_targeted_full_sha256"],
        "failed targeted assignment",
    )
    _require_equal(
        file_sha256(baseline_generic),
        freeze["failed_generic_full_sha256"],
        "failed generic assignment",
    )
    selection = _load_json(baseline_selection)
    selection_payload = dict(selection)
    stored_selection_hash = selection_payload.pop("summary_sha256", None)
    _require_equal(
        canonical_sha256(selection_payload), stored_selection_hash, "baseline selection summary"
    )
    _require_equal(selection.get("selection_gate_passed"), False, "baseline stop decision")

    loaded, raw_by_id = load_base_failures((asdiv, mathqa), (asdiv_predictions, mathqa_predictions))
    candidates, rejected = _deduplicate(loaded)
    normalized_by_id = {
        candidate.source_id: _normalized_question_sha256(raw_by_id[candidate.source_id])
        for candidate in candidates
    }
    payloads = [_candidate_payload(candidate, normalized_by_id) for candidate in candidates]
    asdiv_payloads = [row for row in payloads if row["source_corpus"] == "ASDiv"]
    mathqa_payloads = [row for row in payloads if row["source_corpus"] == "MathQA"]
    _require_equal(len(loaded), freeze["input_base_failure_count"], "base-failure count")
    _require_equal(len(candidates), freeze["eligible_unique_count"], "eligible pool count")
    _require_equal(
        dict(sorted(rejected.items())), freeze["deduplication_rejections"], "deduplication"
    )
    _require_equal(len(asdiv_payloads), freeze["eligible_asdiv_count"], "eligible ASDiv count")
    _require_equal(len(mathqa_payloads), freeze["eligible_mathqa_count"], "eligible MathQA count")
    _require_equal(
        canonical_sha256(asdiv_payloads), freeze["eligible_asdiv_sha256"], "eligible ASDiv pool"
    )
    _require_equal(
        canonical_sha256(mathqa_payloads),
        freeze["eligible_mathqa_sha256"],
        "eligible MathQA pool",
    )
    _require_equal(
        canonical_sha256(payloads), freeze["combined_eligible_pool_sha256"], "combined pool"
    )
    _require_equal(
        canonical_sha256(payloads), freeze["candidate_covariates_sha256"], "candidate covariates"
    )

    targeted_ids = _load_ids(baseline_targeted)
    generic_ids = _load_ids(baseline_generic)
    _require_equal(
        canonical_sha256(sorted(targeted_ids)),
        freeze["failed_targeted_assignment_ids_sha256"],
        "targeted assignment IDs",
    )
    _require_equal(
        canonical_sha256(sorted(generic_ids)),
        freeze["failed_generic_assignment_ids_sha256"],
        "generic assignment IDs",
    )
    candidate_ids = {candidate.source_id for candidate in candidates}
    if not set(targeted_ids + generic_ids) <= candidate_ids:
        raise ValueError("baseline assignment contains an ID outside the frozen candidate pool")

    return RepairInputs(
        candidates=tuple(candidates),
        raw_by_id=raw_by_id,
        targeted_ids=targeted_ids,
        generic_ids=generic_ids,
        normalized_by_id=normalized_by_id,
        input_freeze=freeze,
    )


def _categorical_metrics(report: dict[str, object]) -> tuple[float, float]:
    raw = report["categorical_absolute_proportion_difference"]
    if not isinstance(raw, dict):
        raise ValueError("categorical report is malformed")
    values = [
        float(value)
        for levels in raw.values()
        if isinstance(levels, dict)
        for value in levels.values()
    ]
    return max(values, default=0.0), sum(values)


def _validate_assignment(
    targeted_ids: Sequence[str],
    generic_ids: Sequence[str],
    *,
    by_id: dict[str, Candidate],
    normalized_by_id: dict[str, str],
    shingles_by_id: dict[str, frozenset[tuple[str, ...]]],
    require_numerical_gate: bool,
) -> tuple[dict[str, object], float, float]:
    if len(targeted_ids) != TARGET_SIZE or len(generic_ids) != TARGET_SIZE:
        raise ValueError("repaired arm size differs from 200")
    try:
        targeted = [by_id[source_id] for source_id in targeted_ids]
        generic = [by_id[source_id] for source_id in generic_ids]
    except KeyError as error:
        raise ValueError("assignment contains an ID outside the frozen pool") from error
    _require_equal(
        Counter(item.family for item in targeted),
        Counter(QUOTAS[TARGET_SIZE]["targeted"]),
        "targeted family quotas",
    )
    _require_equal(
        Counter(item.family for item in generic),
        Counter(QUOTAS[TARGET_SIZE]["generic"]),
        "generic family quotas",
    )
    _require_equal(
        Counter(item.source_corpus for item in targeted),
        Counter(item.source_corpus for item in generic),
        "source composition",
    )
    combined = [*targeted, *generic]
    if len({item.source_id for item in combined}) != TARGET_SIZE * 2:
        raise ValueError("assignment shares source IDs")
    if len({item.question_sha256 for item in combined}) != TARGET_SIZE * 2:
        raise ValueError("assignment contains exact-question duplicates")
    if len({item.program_sha256 for item in combined}) != TARGET_SIZE * 2:
        raise ValueError("assignment contains latent-program duplicates")
    targeted_normalized = {normalized_by_id[item.source_id] for item in targeted}
    generic_normalized = {normalized_by_id[item.source_id] for item in generic}
    if targeted_normalized & generic_normalized:
        raise ValueError("assignment contains cross-arm normalized-question duplicates")
    if any(
        _near_duplicate(shingles_by_id[left.source_id], shingles_by_id[right.source_id])
        for left in targeted
        for right in generic
    ):
        raise ValueError("assignment contains unresolved cross-arm near duplicates")
    report = balance_report(targeted, generic)
    categorical_maximum, categorical_total = _categorical_metrics(report)
    if categorical_maximum > MAX_CATEGORICAL_DIFFERENCE:
        raise ValueError("assignment exceeds the categorical balance gate")
    if require_numerical_gate and not bool(report["matching_quality_gate_passed"]):
        raise ValueError("assignment exceeds a frozen matching gate")
    return report, categorical_maximum, categorical_total


def _aggregate_smd(left_sum: int, left_sumsq: int, right_sum: int, right_sumsq: int) -> float:
    left_mean = left_sum / TARGET_SIZE
    right_mean = right_sum / TARGET_SIZE
    left_variance = (left_sumsq - TARGET_SIZE * left_mean * left_mean) / (TARGET_SIZE - 1)
    right_variance = (right_sumsq - TARGET_SIZE * right_mean * right_mean) / (TARGET_SIZE - 1)
    pooled = math.sqrt(max(0.0, (left_variance + right_variance) / 2))
    if pooled == 0:
        return 0.0 if left_mean == right_mean else math.inf
    return abs(left_mean - right_mean) / pooled


def _pair_matches(
    targeted: Sequence[Candidate], generic: Sequence[Candidate]
) -> tuple[int, tuple[Match, ...]]:
    matches: list[Match] = []
    for source in ("ASDiv", "MathQA"):
        matches.extend(
            _exact_matches(
                [item for item in targeted if item.source_corpus == source],
                [item for item in generic if item.source_corpus == source],
            )
        )
    matches.sort(key=lambda item: item.targeted_source_id)
    return sum(item.cost for item in matches), tuple(matches)


def search_single_replacements(inputs: RepairInputs) -> SearchEvidence:
    by_id = {candidate.source_id: candidate for candidate in inputs.candidates}
    selected = set(inputs.targeted_ids + inputs.generic_ids)
    unused = [
        candidate.source_id
        for candidate in inputs.candidates
        if candidate.source_id not in selected
    ]
    shingles = {
        source_id: _question_shingles(_text(inputs.raw_by_id[source_id], "combined_question"))
        for source_id in [*unused, *inputs.targeted_ids, *inputs.generic_ids]
    }
    baseline_report, _, _ = _validate_assignment(
        inputs.targeted_ids,
        inputs.generic_ids,
        by_id=by_id,
        normalized_by_id=inputs.normalized_by_id,
        shingles_by_id=shingles,
        require_numerical_gate=False,
    )
    _require_equal(baseline_report["numerical_smd"], EXPECTED_BASELINE_SMD, "baseline SMDs")

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source_id in unused:
        candidate = by_id[source_id]
        buckets[(candidate.source_corpus, candidate.family)].append(source_id)
    for values in buckets.values():
        values.sort()
    near_targeted = {
        source_id: any(
            _near_duplicate(shingles[source_id], shingles[other]) for other in inputs.targeted_ids
        )
        for source_id in unused
    }
    near_generic = {
        source_id: any(
            _near_duplicate(shingles[source_id], shingles[other]) for other in inputs.generic_ids
        )
        for source_id in unused
    }
    targeted_normalized = {inputs.normalized_by_id[item] for item in inputs.targeted_ids}
    generic_normalized = {inputs.normalized_by_id[item] for item in inputs.generic_ids}

    def aggregates(ids: Sequence[str]) -> tuple[dict[str, int], dict[str, int]]:
        return (
            {
                field: sum(int(getattr(by_id[item], field)) for item in ids)
                for field in CONTINUOUS_FIELDS
            },
            {
                field: sum(int(getattr(by_id[item], field)) ** 2 for item in ids)
                for field in CONTINUOUS_FIELDS
            },
        )

    targeted_sums, targeted_sumsq = aggregates(inputs.targeted_ids)
    generic_sums, generic_sumsq = aggregates(inputs.generic_ids)
    targeted_categories = {
        field: Counter(str(getattr(by_id[item], field)) for item in inputs.targeted_ids)
        for field in CATEGORICAL_FIELDS
    }
    generic_categories = {
        field: Counter(str(getattr(by_id[item], field)) for item in inputs.generic_ids)
        for field in CATEGORICAL_FIELDS
    }

    checked = 0
    legal = 0
    preliminary: list[tuple[str, str, str]] = []
    for arm, arm_ids in (
        ("targeted", inputs.targeted_ids),
        ("generic", inputs.generic_ids),
    ):
        for removed in arm_ids:
            old = by_id[removed]
            for added in buckets[(old.source_corpus, old.family)]:
                checked += 1
                if arm == "targeted" and (
                    inputs.normalized_by_id[added] in generic_normalized or near_generic[added]
                ):
                    continue
                if arm == "generic" and (
                    inputs.normalized_by_id[added] in targeted_normalized or near_targeted[added]
                ):
                    continue
                legal += 1
                smds: dict[str, float] = {}
                for field in CONTINUOUS_FIELDS:
                    removed_value = int(getattr(old, field))
                    added_value = int(getattr(by_id[added], field))
                    delta = added_value - removed_value
                    delta_square = added_value * added_value - removed_value * removed_value
                    smds[field] = _aggregate_smd(
                        targeted_sums[field] + (delta if arm == "targeted" else 0),
                        targeted_sumsq[field] + (delta_square if arm == "targeted" else 0),
                        generic_sums[field] + (delta if arm == "generic" else 0),
                        generic_sumsq[field] + (delta_square if arm == "generic" else 0),
                    )
                if max(smds.values()) > MAX_SMD + 1e-12:
                    continue
                categorical_maximum = 0.0
                for field in CATEGORICAL_FIELDS:
                    old_level = str(getattr(old, field))
                    new_level = str(getattr(by_id[added], field))
                    levels = (
                        set(targeted_categories[field])
                        | set(generic_categories[field])
                        | {old_level, new_level}
                    )
                    for level in levels:
                        targeted_count = targeted_categories[field][level]
                        generic_count = generic_categories[field][level]
                        change = int(new_level == level) - int(old_level == level)
                        if arm == "targeted":
                            targeted_count += change
                        else:
                            generic_count += change
                        categorical_maximum = max(
                            categorical_maximum,
                            abs(targeted_count / TARGET_SIZE - generic_count / TARGET_SIZE),
                        )
                if categorical_maximum <= MAX_CATEGORICAL_DIFFERENCE + 1e-12:
                    preliminary.append((arm, removed, added))

    passing_without_cost: list[PassingRepair] = []
    for arm, removed, added in preliminary:
        targeted_ids = list(inputs.targeted_ids)
        generic_ids = list(inputs.generic_ids)
        changed = targeted_ids if arm == "targeted" else generic_ids
        changed[changed.index(removed)] = added
        targeted = [by_id[item] for item in targeted_ids]
        generic = [by_id[item] for item in generic_ids]
        report = balance_report(targeted, generic)
        categorical_maximum, categorical_total = _categorical_metrics(report)
        if not bool(report["matching_quality_gate_passed"]):
            continue
        if categorical_maximum > MAX_CATEGORICAL_DIFFERENCE:
            continue
        passing_without_cost.append(
            PassingRepair(
                arm=arm,
                removed=removed,
                added=added,
                targeted_ids=tuple(sorted(targeted_ids)),
                generic_ids=tuple(sorted(generic_ids)),
                numerical_smd=cast(dict[str, float], report["numerical_smd"]),
                categorical_maximum=categorical_maximum,
                categorical_total=categorical_total,
                pairwise_matching_cost=0,
            )
        )
    if not passing_without_cost:
        raise RuntimeError("no legal single replacement passed the frozen matching gates")

    best_prefix = min(item.objective[:3] for item in passing_without_cost)
    finalists = [item for item in passing_without_cost if item.objective[:3] == best_prefix]
    costed: list[tuple[PassingRepair, tuple[Match, ...]]] = []
    for item in finalists:
        targeted = [by_id[source_id] for source_id in item.targeted_ids]
        generic = [by_id[source_id] for source_id in item.generic_ids]
        cost, matches = _pair_matches(targeted, generic)
        costed.append(
            (
                PassingRepair(
                    arm=item.arm,
                    removed=item.removed,
                    added=item.added,
                    targeted_ids=item.targeted_ids,
                    generic_ids=item.generic_ids,
                    numerical_smd=item.numerical_smd,
                    categorical_maximum=item.categorical_maximum,
                    categorical_total=item.categorical_total,
                    pairwise_matching_cost=cost,
                ),
                matches,
            )
        )
    selected_repair, matches = min(costed, key=lambda value: value[0].objective)
    _validate_assignment(
        selected_repair.targeted_ids,
        selected_repair.generic_ids,
        by_id=by_id,
        normalized_by_id=inputs.normalized_by_id,
        shingles_by_id=shingles,
        require_numerical_gate=True,
    )
    return SearchEvidence(
        checked_replacements=checked,
        legal_replacements=legal,
        passing_replacements=len(passing_without_cost),
        selected=selected_repair,
        matches=matches,
    )


def _write_jsonl(path: Path, rows: Iterable[object]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def run_repair(
    *,
    asdiv: Path,
    asdiv_predictions: Path,
    mathqa: Path,
    mathqa_predictions: Path,
    baseline_targeted: Path,
    baseline_generic: Path,
    baseline_selection: Path,
    input_freeze_path: Path,
    output_root: Path,
    config: RepairConfig | None = None,
) -> dict[str, object]:
    if config is None:
        config = RepairConfig()
    config.validate_frozen()
    inputs = load_and_verify_inputs(
        asdiv=asdiv,
        asdiv_predictions=asdiv_predictions,
        mathqa=mathqa,
        mathqa_predictions=mathqa_predictions,
        baseline_targeted=baseline_targeted,
        baseline_generic=baseline_generic,
        baseline_selection=baseline_selection,
        input_freeze_path=input_freeze_path,
    )
    evidence = search_single_replacements(inputs)
    by_id = {candidate.source_id: candidate for candidate in inputs.candidates}
    output_root.mkdir(parents=True, exist_ok=True)
    targeted_full_hash = _write_jsonl(
        output_root / "targeted_full.jsonl",
        [
            {**inputs.raw_by_id[source_id], "selection_arm": "targeted"}
            for source_id in evidence.selected.targeted_ids
        ],
    )
    generic_full_hash = _write_jsonl(
        output_root / "generic_full.jsonl",
        [
            {**inputs.raw_by_id[source_id], "selection_arm": "generic"}
            for source_id in evidence.selected.generic_ids
        ],
    )

    def manifest_row(source_id: str, arm: str) -> dict[str, object]:
        row = _candidate_payload(by_id[source_id], inputs.normalized_by_id)
        row["arm"] = arm
        return row

    targeted_manifest_hash = _write_jsonl(
        output_root / "targeted_manifest.jsonl",
        [manifest_row(source_id, "targeted") for source_id in evidence.selected.targeted_ids],
    )
    generic_manifest_hash = _write_jsonl(
        output_root / "generic_manifest.jsonl",
        [manifest_row(source_id, "generic") for source_id in evidence.selected.generic_ids],
    )
    match_hash = _write_jsonl(
        output_root / "exact_matches.jsonl", [asdict(item) for item in evidence.matches]
    )
    targeted = [by_id[source_id] for source_id in evidence.selected.targeted_ids]
    generic = [by_id[source_id] for source_id in evidence.selected.generic_ids]
    final_report = balance_report(targeted, generic)
    categorical_maximum, categorical_total = _categorical_metrics(final_report)
    assignment = {
        "targeted": list(evidence.selected.targeted_ids),
        "generic": list(evidence.selected.generic_ids),
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "repair_contract": REPAIR_CONTRACT,
        "input_freeze_sha256": EXPECTED_INPUT_FREEZE_SHA256,
        "matching_configuration": asdict(config),
        "matching_configuration_sha256": canonical_sha256(asdict(config)),
        "method": "deterministic exhaustive legal single-row replacement",
        "checked_replacements": evidence.checked_replacements,
        "legal_replacements": evidence.legal_replacements,
        "passing_replacements": evidence.passing_replacements,
        "selected_replacement": {
            "arm": evidence.selected.arm,
            "removed": evidence.selected.removed,
            "added": evidence.selected.added,
        },
        "two_replacement_search_run": False,
        "global_fallback_run": False,
        "targeted_size": len(targeted),
        "generic_size": len(generic),
        "targeted_family_counts": dict(sorted(Counter(item.family for item in targeted).items())),
        "generic_family_counts": dict(sorted(Counter(item.family for item in generic).items())),
        "targeted_source_counts": dict(
            sorted(Counter(item.source_corpus for item in targeted).items())
        ),
        "generic_source_counts": dict(
            sorted(Counter(item.source_corpus for item in generic).items())
        ),
        "numerical_smd": final_report["numerical_smd"],
        "categorical_absolute_proportion_difference": final_report[
            "categorical_absolute_proportion_difference"
        ],
        "categorical_maximum": categorical_maximum,
        "categorical_total": categorical_total,
        "pairwise_matching_cost": evidence.selected.pairwise_matching_cost,
        "matching_gate_passed": final_report["matching_quality_gate_passed"],
        "source_composition_exact": final_report["source_composition_exact"],
        "cross_arm_source_id_overlap": 0,
        "cross_arm_exact_question_overlap": 0,
        "cross_arm_normalized_question_overlap": 0,
        "cross_arm_latent_program_overlap": 0,
        "cross_arm_near_duplicate_count": 0,
        "contamination_count": 0,
        "source_rows_reparsed": False,
        "contamination_rerun": False,
        "model_inference_rerun": False,
        "targeted_full_sha256": targeted_full_hash,
        "generic_full_sha256": generic_full_hash,
        "targeted_manifest_sha256": targeted_manifest_hash,
        "generic_manifest_sha256": generic_manifest_hash,
        "exact_match_sha256": match_hash,
        "targeted_ids_sha256": canonical_sha256(list(evidence.selected.targeted_ids)),
        "generic_ids_sha256": canonical_sha256(list(evidence.selected.generic_ids)),
        "paired_assignment_sha256": canonical_sha256(assignment),
    }
    summary["covariate_summary_sha256"] = canonical_sha256(
        {
            "numerical_smd": summary["numerical_smd"],
            "categorical": summary["categorical_absolute_proportion_difference"],
        }
    )
    summary["matching_evidence_sha256"] = canonical_sha256(summary)
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_root / "repair_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair the frozen Phase 2 matched assignment")
    parser.add_argument("--asdiv", type=Path, required=True)
    parser.add_argument("--asdiv-predictions", type=Path, required=True)
    parser.add_argument("--mathqa", type=Path, required=True)
    parser.add_argument("--mathqa-predictions", type=Path, required=True)
    parser.add_argument("--baseline-targeted", type=Path, required=True)
    parser.add_argument("--baseline-generic", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path, required=True)
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_repair(
        asdiv=args.asdiv,
        asdiv_predictions=args.asdiv_predictions,
        mathqa=args.mathqa,
        mathqa_predictions=args.mathqa_predictions,
        baseline_targeted=args.baseline_targeted,
        baseline_generic=args.baseline_generic,
        baseline_selection=args.baseline_selection,
        input_freeze_path=args.input_freeze,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
