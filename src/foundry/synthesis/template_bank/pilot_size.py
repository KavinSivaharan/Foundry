"""Deterministic selection of the largest exactly schedulable signal pilot."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.template_bank.signal_pilot import (
    CATEGORY_ORDER,
    GROUP_ORDER,
    canonical_sha256,
)
from foundry.synthesis.template_bank.submode_policy import (
    constrained_attempt_split,
    load_policy_config,
    water_fill,
)

APPROVED_ACCEPTED_SIZES = (900, 800, 700, 600, 500)
ATTEMPT_NUMERATOR = 11
ATTEMPT_DENOMINATOR = 10
SIZE_SELECTION_POLICY_ID = "largest-feasible-matched-signal-pilot-v1"
TARGETED_WEIGHTS = dict(zip(CATEGORY_ORDER, (550, 233, 217), strict=True))
GENERIC_WEIGHTS = dict(zip(CATEGORY_ORDER, (334, 333, 333), strict=True))


@dataclass(frozen=True)
class PilotSizeCandidate:
    """One predeclared matched accepted size and its exact fixed margins."""

    accepted_per_dataset: int
    training_per_dataset: int
    validation_per_dataset: int
    output_contract_per_dataset: int
    targeted_accepted: dict[str, int]
    generic_accepted: dict[str, int]
    targeted_attempts: dict[str, int]
    generic_attempts: dict[str, int]
    total_attempts: int


@dataclass(frozen=True)
class PilotSizePreflight:
    """Exact global scheduler outcome for one descending candidate size."""

    candidate: PilotSizeCandidate
    feasible: bool
    config_sha256: str
    allocation_sha256: str
    schedule_sha256: str | None
    summary_sha256: str | None
    blocker: str | None


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _ceil_fraction(quantity: int, numerator: int, denominator: int) -> int:
    return (quantity * numerator + denominator - 1) // denominator


def _largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Allocate an exact total by stable key-order largest remainder."""

    weight_total = sum(weights.values())
    if total < 0 or weight_total <= 0:
        raise ValueError("largest-remainder margins are invalid")
    result = {key: total * weight // weight_total for key, weight in weights.items()}
    missing = total - sum(result.values())
    stable_order = {key: index for index, key in enumerate(weights)}
    ranked = sorted(
        weights,
        key=lambda key: (-((total * weights[key]) % weight_total), stable_order[key]),
    )
    for key in ranked[:missing]:
        result[key] += 1
    return result


def derive_size_candidate(accepted_per_dataset: int) -> PilotSizeCandidate:
    """Derive exact accepted and fixed-attempt family margins for one size."""

    if accepted_per_dataset not in APPROVED_ACCEPTED_SIZES:
        raise ValueError("accepted size is not one of the five approved candidates")
    targeted_accepted = _largest_remainder(accepted_per_dataset, TARGETED_WEIGHTS)
    generic_accepted = _largest_remainder(accepted_per_dataset, GENERIC_WEIGHTS)
    targeted_attempts = {
        family: _ceil_fraction(quantity, ATTEMPT_NUMERATOR, ATTEMPT_DENOMINATOR)
        for family, quantity in targeted_accepted.items()
    }
    generic_attempts = {
        family: _ceil_fraction(quantity, ATTEMPT_NUMERATOR, ATTEMPT_DENOMINATOR)
        for family, quantity in generic_accepted.items()
    }
    return PilotSizeCandidate(
        accepted_per_dataset=accepted_per_dataset,
        training_per_dataset=accepted_per_dataset * 9 // 10,
        validation_per_dataset=accepted_per_dataset // 10,
        output_contract_per_dataset=accepted_per_dataset // 5,
        targeted_accepted=targeted_accepted,
        generic_accepted=generic_accepted,
        targeted_attempts=targeted_attempts,
        generic_attempts=generic_attempts,
        total_attempts=sum(targeted_attempts.values()) + sum(generic_attempts.values()),
    )


def derive_size_config_payload(
    base_config_path: Path,
    policy_path: Path,
    accepted_per_dataset: int,
) -> dict[str, object]:
    """Derive a complete candidate config without changing any frozen policy."""

    candidate = derive_size_candidate(accepted_per_dataset)
    raw: object = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(_mapping(raw, "base signal-pilot config"))
    policy = load_policy_config(policy_path)
    payload["pilot_id"] = f"foundry-signal-first-pilot-{accepted_per_dataset}-v1"
    payload["capacity_audit_id"] = f"foundry-signal-first-size-selection-{accepted_per_dataset}-v1"
    payload["attempt_multiplier"] = {
        "numerator": ATTEMPT_NUMERATOR,
        "denominator": ATTEMPT_DENOMINATOR,
    }
    datasets = _mapping(payload["datasets"], "datasets")
    accepted_by_group = {
        GROUP_ORDER[0]: candidate.targeted_accepted,
        GROUP_ORDER[1]: candidate.generic_accepted,
    }
    attempts_by_group = {
        GROUP_ORDER[0]: candidate.targeted_attempts,
        GROUP_ORDER[1]: candidate.generic_attempts,
    }

    for group in GROUP_ORDER:
        dataset = _mapping(datasets[group], group)
        dataset["accepted_total"] = accepted_per_dataset
        dataset["training_accepted"] = candidate.training_per_dataset
        dataset["validation_accepted"] = candidate.validation_per_dataset
        dataset["output_contract_accepted"] = candidate.output_contract_per_dataset
        training_by_family = _largest_remainder(
            candidate.training_per_dataset,
            accepted_by_group[group],
        )
        output_by_family = _largest_remainder(
            candidate.output_contract_per_dataset,
            accepted_by_group[group],
        )
        families = _mapping(dataset["families"], f"{group}.families")
        for family in CATEGORY_ORDER:
            item = _mapping(families[family], f"{group}.{family}")
            accepted = accepted_by_group[group][family]
            attempts = attempts_by_group[group][family]
            item["accepted"] = accepted
            item["attempts"] = attempts
            item["training_accepted"] = training_by_family[family]
            item["validation_accepted"] = accepted - training_by_family[family]
            training_attempts = _ceil_fraction(
                training_by_family[family], ATTEMPT_NUMERATOR, ATTEMPT_DENOMINATOR
            )
            item["training_attempts"] = training_attempts
            item["validation_attempts"] = attempts - training_attempts
            item["output_contract_accepted"] = output_by_family[family]

    for family in CATEGORY_ORDER:
        global_accepted = water_fill(
            sum(accepted_by_group[group][family] for group in GROUP_ORDER),
            policy.capacities[family],
        )
        global_attempts = water_fill(
            sum(attempts_by_group[group][family] for group in GROUP_ORDER),
            policy.capacities[family],
        )
        targeted_attempts, generic_attempts, targeted_accepted, generic_accepted = (
            constrained_attempt_split(
                global_attempts,
                global_accepted,
                first_attempt_total=attempts_by_group[GROUP_ORDER[0]][family],
                first_accepted_total=accepted_by_group[GROUP_ORDER[0]][family],
            )
        )
        group_attempt_modes = {
            GROUP_ORDER[0]: targeted_attempts,
            GROUP_ORDER[1]: generic_attempts,
        }
        group_accepted_modes = {
            GROUP_ORDER[0]: targeted_accepted,
            GROUP_ORDER[1]: generic_accepted,
        }
        for group in GROUP_ORDER:
            family_item = _mapping(
                _mapping(_mapping(datasets[group], group)["families"], "families")[family],
                family,
            )
            family_item["accepted_modes"] = group_accepted_modes[group]
            family_item["attempt_modes"] = group_attempt_modes[group]

    for group in GROUP_ORDER:
        dataset = _mapping(datasets[group], group)
        families = _mapping(dataset["families"], f"{group}.families")
        family_attempts = {
            family: cast(int, _mapping(families[family], family)["attempts"])
            for family in CATEGORY_ORDER
        }
        output_attempt_total = _ceil_fraction(
            candidate.output_contract_per_dataset,
            ATTEMPT_NUMERATOR,
            ATTEMPT_DENOMINATOR,
        )
        output_attempts = _largest_remainder(output_attempt_total, family_attempts)
        for family in CATEGORY_ORDER:
            _mapping(families[family], family)["output_contract_attempts"] = output_attempts[family]

    payload["full_schedule_master_seed"] = (
        f"foundry-reduced-signal-pilot-{accepted_per_dataset}-schedule-20260719-v1"
    )
    payload["full_schedule_raw_path"] = (
        f"results/raw/signal_pilot_size_selection/{accepted_per_dataset}/full_schedule.jsonl"
    )
    payload["full_schedule_summary_path"] = (
        f"results/synthesis_smoke/signal_pilot_{accepted_per_dataset}_schedule_summary.json"
    )
    smoke = _mapping(payload["smoke"], "smoke")
    smoke["run_id"] = "foundry-template-bank-reduced-signal-pilot-review-v1"
    smoke["master_seed"] = "foundry-reduced-signal-pilot-review-20260719-v1"
    smoke["raw_directory"] = "results/raw/template_bank_signal_pilot_review"
    smoke["summary_path"] = "results/synthesis_smoke/signal_pilot_review_summary_final.json"
    smoke["schedule_path"] = "configs/synthesis/signal_pilot_review_schedule_final.json"
    smoke["human_review_markdown"] = "results/raw/template_bank_signal_pilot_review/human_review.md"
    smoke["human_review_html"] = "results/raw/template_bank_signal_pilot_review/human_review.html"
    smoke["codex_audit_path"] = (
        "results/raw/template_bank_signal_pilot_review/codex_language_audit.json"
    )
    smoke["codex_assisted_html"] = (
        "results/raw/template_bank_signal_pilot_review/codex_assisted_review.html"
    )
    return payload


def write_size_config(
    base_config_path: Path,
    policy_path: Path,
    output_path: Path,
    accepted_per_dataset: int,
) -> dict[str, object]:
    """Write one deterministic ignored candidate configuration."""

    payload = derive_size_config_payload(base_config_path, policy_path, accepted_per_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")
    return payload


def select_first_feasible(
    results: tuple[PilotSizePreflight, ...],
) -> PilotSizePreflight:
    """Return the first feasible size in the frozen descending order."""

    if not results:
        raise ValueError("no accepted pilot size was evaluated")
    evaluated = tuple(result.candidate.accepted_per_dataset for result in results)
    if evaluated != APPROVED_ACCEPTED_SIZES[: len(results)]:
        raise ValueError("accepted pilot sizes were not evaluated in approved order")
    for index, result in enumerate(results):
        if result.feasible:
            if index != len(results) - 1:
                raise ValueError("a smaller accepted size was evaluated after feasibility")
            if result.schedule_sha256 is None or result.summary_sha256 is None:
                raise ValueError("feasible size lacks exact schedule evidence")
            return result
        if result.blocker is None:
            raise ValueError("infeasible size lacks a blocker")
    if len(results) != len(APPROVED_ACCEPTED_SIZES):
        raise ValueError("size evaluation stopped before feasibility or exhaustion")
    raise ValueError("none of the five approved accepted sizes is exactly schedulable")


def build_selection_evidence(
    results: tuple[PilotSizePreflight, ...],
    *,
    selection_config_sha256: str,
    candidate_allocations: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    """Build content-free evidence for the exact descending selection."""

    if candidate_allocations is not None and len(candidate_allocations) != len(results):
        raise ValueError("candidate allocations must align with evaluated sizes")

    selected: PilotSizePreflight | None
    try:
        selected = select_first_feasible(results)
    except ValueError as error:
        if str(error) != "none of the five approved accepted sizes is exactly schedulable":
            raise
        selected = None
    payload: dict[str, object] = {
        "schema_version": 1,
        "evidence_id": "foundry-reduced-signal-pilot-selection-v1",
        "policy_id": SIZE_SELECTION_POLICY_ID,
        "candidate_order": list(APPROVED_ACCEPTED_SIZES),
        "attempt_multiplier": {
            "numerator": ATTEMPT_NUMERATOR,
            "denominator": ATTEMPT_DENOMINATOR,
            "label": "1.10",
        },
        "selection_config_sha256": selection_config_sha256,
        "results": [
            {
                **asdict(result.candidate),
                "candidate_sha256": canonical_sha256(asdict(result.candidate)),
                "config_sha256": result.config_sha256,
                "allocation_sha256": result.allocation_sha256,
                "allocation": (
                    candidate_allocations[index] if candidate_allocations is not None else None
                ),
                "exact_schedule_feasible": result.feasible,
                "schedule_sha256": result.schedule_sha256,
                "schedule_summary_sha256": result.summary_sha256,
                "blocker": result.blocker,
            }
            for index, result in enumerate(results)
        ],
        "selected_accepted_size_per_dataset": (
            selected.candidate.accepted_per_dataset if selected is not None else None
        ),
        "selected_fixed_attempts": (
            selected.candidate.total_attempts if selected is not None else None
        ),
        "complete_schedule_created": selected is not None,
        "fresh_review_schedule_created": False,
        "fresh_smoke_run": False,
        "deterministic_replay_run": False,
        "review_packet_created": False,
        "size_selection_gate_passed": selected is not None,
        "lower_sizes_not_tested_after_first_feasible": True,
        "family_weights_unchanged": True,
        "frozen_controls_changed": False,
        "sealed_final_accessed": False,
        "next_action": "human-review" if selected is not None else "architecture-stop",
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    return payload
