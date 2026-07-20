"""Deterministic fixed-attempt-pool derivation for the signal-first pilot."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.template_bank.difficulty_reallocation import (
    RATE_FAMILY,
    WEIGHTED_MODE,
    load_difficulty_reallocation_config,
    reallocate_group_difficulties,
)
from foundry.synthesis.template_bank.signal_pilot import (
    CATEGORY_ORDER,
    DIFFICULTY_ORDER,
    GROUP_ORDER,
    balanced_counts,
    canonical_sha256,
    load_signal_pilot_config,
)
from foundry.synthesis.template_bank.submode_policy import (
    balanced_matrix,
    constrained_attempt_split,
    constrained_difficulty_matrices,
    load_policy_config,
    water_fill,
)
from foundry.synthesis.template_bank.surface_reuse import (
    derive_surface_caps,
    load_surface_reuse_config,
)

APPROVED_MULTIPLIERS: tuple[tuple[int, int], ...] = ((23, 20), (9, 8), (11, 10))
LEGACY_MULTIPLIER = (5, 4)
ATTEMPT_POOL_POLICY_ID = "largest-feasible-fixed-attempt-pool-v1"


@dataclass(frozen=True)
class AttemptPoolCandidate:
    """One predeclared candidate multiplier and its exact family totals."""

    numerator: int
    denominator: int
    label: str
    targeted_attempts: dict[str, int]
    generic_attempts: dict[str, int]
    total_attempts: int


@dataclass(frozen=True)
class AttemptPoolPreflight:
    """Exact scheduler outcome for one evaluated candidate pool."""

    candidate: AttemptPoolCandidate
    feasible: bool
    config_sha256: str
    schedule_sha256: str | None
    summary_sha256: str | None
    blocker: str | None


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _ceil_fraction(quantity: int, numerator: int, denominator: int) -> int:
    return (quantity * numerator + denominator - 1) // denominator


def _multiplier_label(numerator: int, denominator: int) -> str:
    labels = {(23, 20): "1.15", (9, 8): "1.125", (11, 10): "1.10", (5, 4): "1.25"}
    try:
        return labels[(numerator, denominator)]
    except KeyError as error:
        raise ValueError("attempt multiplier is not predeclared") from error


def _proportional_counts(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Allocate a fixed total by stable largest remainder."""

    weight_total = sum(weights.values())
    if total < 0 or weight_total <= 0 or total > weight_total:
        raise ValueError("proportional allocation margins are invalid")
    result = {label: total * weight // weight_total for label, weight in weights.items()}
    missing = total - sum(result.values())
    ranked = sorted(
        weights,
        key=lambda label: (
            -((total * weights[label]) % weight_total),
            tuple(weights).index(label),
        ),
    )
    for label in ranked[:missing]:
        result[label] += 1
    return result


def derive_attempt_counts(
    base_config_path: Path, numerator: int, denominator: int
) -> AttemptPoolCandidate:
    """Derive only the approved family-level fixed attempt totals."""

    if (numerator, denominator) not in APPROVED_MULTIPLIERS:
        raise ValueError("only the three approved Milestone 7F multipliers may be evaluated")
    base = load_signal_pilot_config(base_config_path)
    by_group = {
        group: {
            family: _ceil_fraction(
                base.datasets[group].families[family].accepted,
                numerator,
                denominator,
            )
            for family in CATEGORY_ORDER
        }
        for group in GROUP_ORDER
    }
    return AttemptPoolCandidate(
        numerator=numerator,
        denominator=denominator,
        label=_multiplier_label(numerator, denominator),
        targeted_attempts=by_group[GROUP_ORDER[0]],
        generic_attempts=by_group[GROUP_ORDER[1]],
        total_attempts=sum(quantity for group in by_group.values() for quantity in group.values()),
    )


def derive_candidate_config_payload(
    base_config_path: Path,
    policy_path: Path,
    numerator: int,
    denominator: int,
) -> dict[str, object]:
    """Create a selected-multiplier config while preserving every accepted quota."""

    candidate = derive_attempt_counts(base_config_path, numerator, denominator)
    raw: object = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(_mapping(raw, "base signal-pilot config"))
    policy = load_policy_config(policy_path)
    payload["attempt_multiplier"] = {"numerator": numerator, "denominator": denominator}
    datasets = _mapping(payload["datasets"], "datasets")

    group_attempt_totals = {
        GROUP_ORDER[0]: candidate.targeted_attempts,
        GROUP_ORDER[1]: candidate.generic_attempts,
    }
    for family in CATEGORY_ORDER:
        accepted_total = sum(
            cast(
                int,
                _mapping(
                    _mapping(_mapping(datasets[group], group)["families"], "families")[family],
                    family,
                )["accepted"],
            )
            for group in GROUP_ORDER
        )
        attempt_total = sum(group_attempt_totals[group][family] for group in GROUP_ORDER)
        global_accepted = water_fill(accepted_total, policy.capacities[family])
        global_attempts = water_fill(attempt_total, policy.capacities[family])
        targeted_attempts, generic_attempts, targeted_accepted, generic_accepted = (
            constrained_attempt_split(
                global_attempts,
                global_accepted,
                first_attempt_total=group_attempt_totals[GROUP_ORDER[0]][family],
                first_accepted_total=cast(
                    int,
                    _mapping(
                        _mapping(
                            _mapping(datasets[GROUP_ORDER[0]], "targeted")["families"], "families"
                        )[family],
                        family,
                    )["accepted"],
                ),
            )
        )
        attempts_by_group = {
            GROUP_ORDER[0]: targeted_attempts,
            GROUP_ORDER[1]: generic_attempts,
        }
        accepted_by_group = {
            GROUP_ORDER[0]: targeted_accepted,
            GROUP_ORDER[1]: generic_accepted,
        }
        for group in GROUP_ORDER:
            family_map = _mapping(
                _mapping(_mapping(datasets[group], group)["families"], "families")[family],
                f"{group}.{family}",
            )
            if _mapping(family_map["accepted_modes"], "accepted_modes") != accepted_by_group[group]:
                raise ValueError("accepted submode allocation changed")
            attempts = group_attempt_totals[group][family]
            family_map["attempts"] = attempts
            family_map["attempt_modes"] = attempts_by_group[group]
            training_attempts = _ceil_fraction(
                cast(int, family_map["training_accepted"]), numerator, denominator
            )
            family_map["training_attempts"] = training_attempts
            family_map["validation_attempts"] = attempts - training_attempts

    for group in GROUP_ORDER:
        dataset = _mapping(datasets[group], group)
        families = _mapping(dataset["families"], f"{group}.families")
        family_attempts = {
            family: cast(int, _mapping(families[family], family)["attempts"])
            for family in CATEGORY_ORDER
        }
        output_total = _ceil_fraction(
            cast(int, dataset["output_contract_accepted"]), numerator, denominator
        )
        output_by_family = _proportional_counts(output_total, family_attempts)
        for family in CATEGORY_ORDER:
            _mapping(families[family], family)["output_contract_attempts"] = output_by_family[
                family
            ]

    payload["full_schedule_master_seed"] = "foundry-signal-first-full-schedule-20260719-v2"
    payload["full_schedule_raw_path"] = "results/raw/signal_pilot_attempt_pool/full_schedule.jsonl"
    payload["full_schedule_summary_path"] = (
        "results/synthesis_smoke/signal_pilot_attempt_pool_schedule_summary.json"
    )
    smoke = _mapping(payload["smoke"], "smoke")
    smoke["run_id"] = "foundry-template-bank-signal-pilot-review-v3"
    smoke["master_seed"] = "foundry-signal-pilot-review-smoke-20260719-v3"
    smoke["summary_path"] = "results/synthesis_smoke/signal_pilot_review_summary_v3.json"
    smoke["schedule_path"] = "configs/synthesis/signal_pilot_review_schedule_v3.json"
    return payload


def write_candidate_config(
    base_config_path: Path,
    policy_path: Path,
    output_path: Path,
    numerator: int,
    denominator: int,
) -> dict[str, object]:
    """Write one deterministic candidate config for exact preflight."""

    payload = derive_candidate_config_payload(base_config_path, policy_path, numerator, denominator)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")
    return payload


def build_attempt_pool_allocation(config_path: Path, policy_path: Path) -> dict[str, object]:
    """Derive exact submode, difficulty, output, and split margins for a pool."""

    config = load_signal_pilot_config(config_path)
    policy = load_policy_config(policy_path)
    difficulty_policy = load_difficulty_reallocation_config(config.difficulty_reallocation_path)
    surface_policy = load_surface_reuse_config(difficulty_policy.surface_policy_config)
    surface_caps = derive_surface_caps(config, surface_policy)
    dataset_results: dict[str, dict[str, object]] = {group: {} for group in GROUP_ORDER}
    global_modes: dict[str, dict[str, int]] = {}
    required_moves: dict[str, int] = {}
    selected_shifts: dict[str, tuple[dict[str, object], ...]] = {}

    for family in CATEGORY_ORDER:
        global_attempts = water_fill(
            sum(config.datasets[group].families[family].attempts for group in GROUP_ORDER),
            policy.capacities[family],
        )
        global_accepted = water_fill(
            sum(config.datasets[group].families[family].accepted for group in GROUP_ORDER),
            policy.capacities[family],
        )
        targeted_attempts, generic_attempts, targeted_accepted, generic_accepted = (
            constrained_attempt_split(
                global_attempts,
                global_accepted,
                first_attempt_total=config.datasets[GROUP_ORDER[0]].families[family].attempts,
                first_accepted_total=config.datasets[GROUP_ORDER[0]].families[family].accepted,
            )
        )
        attempts_by_group = {
            GROUP_ORDER[0]: targeted_attempts,
            GROUP_ORDER[1]: generic_attempts,
        }
        accepted_by_group = {
            GROUP_ORDER[0]: targeted_accepted,
            GROUP_ORDER[1]: generic_accepted,
        }
        for group in GROUP_ORDER:
            quota = config.datasets[group].families[family]
            if quota.attempt_modes != attempts_by_group[group]:
                raise ValueError(f"{group}/{family} attempt modes differ from frozen water-filling")
            if quota.accepted_modes != accepted_by_group[group]:
                raise ValueError(f"{group}/{family} accepted modes changed")

        difficulty = constrained_difficulty_matrices(
            global_attempts,
            targeted_attempts,
            targeted_total=config.datasets[GROUP_ORDER[0]].families[family].attempts,
            generic_total=config.datasets[GROUP_ORDER[1]].families[family].attempts,
            cell_capacities=policy.difficulty_capacities[family],
        )
        accepted_difficulty = {
            group: balanced_matrix(
                accepted_by_group[group],
                balanced_counts(config.datasets[group].families[family].accepted, DIFFICULTY_ORDER),
                cell_capacities=difficulty[group],
            )
            for group in GROUP_ORDER
        }
        if family == RATE_FAMILY:
            for group in GROUP_ORDER:
                easy_medium_capacity = (
                    len(surface_policy.weighted_easy_medium_identities)
                    * surface_caps[group][family][WEIGHTED_MODE].max_attempts_per_identity
                )
                current = difficulty[group][WEIGHTED_MODE]
                moves = max(0, current["easy"] + current["medium"] - easy_medium_capacity)
                difficulty[group], shifts = reallocate_group_difficulties(
                    difficulty[group],
                    accepted_difficulty[group],
                    required_moves=moves,
                    donor_mode_order=difficulty_policy.donor_mode_order,
                )
                required_moves[group] = moves
                selected_shifts[group] = shifts

        global_modes[family] = global_attempts
        for group in GROUP_ORDER:
            quota = config.datasets[group].families[family]
            if any(
                accepted_difficulty[group][mode][difficulty_label]
                > difficulty[group][mode][difficulty_label]
                for mode in quota.attempt_modes
                for difficulty_label in DIFFICULTY_ORDER
            ):
                raise ValueError("accepted difficulty cell exceeds the attempt pool")
            dataset_results[group][family] = {
                "required_accepted": quota.accepted,
                "required_attempts": quota.attempts,
                "attempt_modes": quota.attempt_modes,
                "accepted_modes": quota.accepted_modes,
                "attempt_subordinate_allocations": {
                    "difficulty": difficulty[group],
                    "output_contract": balanced_matrix(
                        quota.attempt_modes,
                        {
                            "enabled": quota.output_contract_attempts,
                            "disabled": quota.attempts - quota.output_contract_attempts,
                        },
                    ),
                    "future_split": balanced_matrix(
                        quota.attempt_modes,
                        {
                            "training": quota.training_attempts,
                            "synthetic_validation": quota.validation_attempts,
                        },
                    ),
                },
                "accepted_subordinate_allocations": {
                    "difficulty": accepted_difficulty[group],
                },
            }

    payload: dict[str, object] = {
        "schema_version": 1,
        "policy_id": ATTEMPT_POOL_POLICY_ID,
        "config_sha256": config.config_sha256,
        "multiplier": {
            "numerator": config.attempt_numerator,
            "denominator": config.attempt_denominator,
            "label": _multiplier_label(config.attempt_numerator, config.attempt_denominator),
        },
        "total_attempts": sum(
            quota.attempts
            for dataset in config.datasets.values()
            for quota in dataset.families.values()
        ),
        "global_attempt_modes": global_modes,
        "datasets": dataset_results,
        "difficulty_policy_id": difficulty_policy.policy_id,
        "difficulty_required_moves": required_moves,
        "difficulty_shifts": selected_shifts,
        "surface_policy_id": surface_policy.policy_id,
        "surface_policy_sha256": surface_policy.config_sha256,
        "accepted_quotas_unchanged": True,
        "capacity_gate_passed": True,
    }
    payload["allocation_sha256"] = canonical_sha256(payload)
    return payload


def candidate_summary(candidate: AttemptPoolCandidate) -> dict[str, object]:
    """Return the stable content-free report for one multiplier."""

    payload = asdict(candidate)
    payload["candidate_sha256"] = canonical_sha256(payload)
    return payload


def select_first_feasible(
    results: tuple[AttemptPoolPreflight, ...],
) -> AttemptPoolPreflight:
    """Select the first feasible preflight in the approved descending order."""

    if not results:
        raise ValueError("no attempt multiplier was evaluated")
    evaluated = tuple(
        (result.candidate.numerator, result.candidate.denominator) for result in results
    )
    if evaluated != APPROVED_MULTIPLIERS[: len(results)]:
        raise ValueError("attempt multipliers were not evaluated in approved order")
    for index, result in enumerate(results):
        if result.feasible:
            if index != len(results) - 1:
                raise ValueError("a lower multiplier was evaluated after feasibility")
            if result.schedule_sha256 is None or result.summary_sha256 is None:
                raise ValueError("feasible preflight lacks schedule evidence")
            return result
        if result.blocker is None:
            raise ValueError("failed preflight lacks a blocker")
    if len(results) != len(APPROVED_MULTIPLIERS):
        raise ValueError("multiplier evaluation stopped before a feasible result or exhaustion")
    raise ValueError("none of the three approved attempt multipliers is feasible")


def dump_candidate_summary(candidate: AttemptPoolCandidate) -> str:
    """Serialize a candidate report for progress logs and evidence."""

    return json.dumps(candidate_summary(candidate), sort_keys=True)


def build_selection_evidence(
    *,
    base_config_path: Path,
    policy_path: Path,
    candidate_config_paths: tuple[Path, ...],
    blockers: tuple[str, ...],
    selection_config_sha256: str,
) -> dict[str, object]:
    """Build the content-free stopped result after exact candidate exhaustion."""

    if len(candidate_config_paths) != len(APPROVED_MULTIPLIERS) or len(blockers) != len(
        APPROVED_MULTIPLIERS
    ):
        raise ValueError("selection evidence must account for all three approved candidates")
    results: list[dict[str, object]] = []
    preflights: list[AttemptPoolPreflight] = []
    for index, (numerator, denominator) in enumerate(APPROVED_MULTIPLIERS):
        candidate = derive_attempt_counts(base_config_path, numerator, denominator)
        config = load_signal_pilot_config(candidate_config_paths[index])
        allocation = build_attempt_pool_allocation(candidate_config_paths[index], policy_path)
        preflight = AttemptPoolPreflight(
            candidate=candidate,
            feasible=False,
            config_sha256=config.config_sha256,
            schedule_sha256=None,
            summary_sha256=None,
            blocker=blockers[index],
        )
        preflights.append(preflight)
        results.append(
            {
                **candidate_summary(candidate),
                "config_sha256": config.config_sha256,
                "allocation_sha256": allocation["allocation_sha256"],
                "global_attempt_modes": allocation["global_attempt_modes"],
                "difficulty_required_moves": allocation["difficulty_required_moves"],
                "datasets": allocation["datasets"],
                "exact_schedule_attempted": True,
                "exact_schedule_feasible": False,
                "schedule_sha256": None,
                "schedule_summary_sha256": None,
                "blocker": blockers[index],
            }
        )
    try:
        select_first_feasible(tuple(preflights))
    except ValueError as error:
        if str(error) != "none of the three approved attempt multipliers is feasible":
            raise
    else:
        raise ValueError("stopped evidence unexpectedly selected a multiplier")
    payload: dict[str, object] = {
        "schema_version": 1,
        "evidence_id": "foundry-signal-pilot-attempt-pool-selection-v1",
        "policy_id": ATTEMPT_POOL_POLICY_ID,
        "selection_config_sha256": selection_config_sha256,
        "candidate_order": ["1.15", "1.125", "1.10"],
        "results": results,
        "selected_multiplier": None,
        "selected_fixed_attempts": None,
        "complete_schedule_created": False,
        "fresh_review_schedule_created": False,
        "fresh_smoke_run": False,
        "deterministic_replay_run": False,
        "review_packet_created": False,
        "selection_gate_passed": False,
        "final_stop_rule_invoked": True,
        "accepted_quotas_unchanged": True,
        "generators_or_verifiers_changed": False,
        "template_wording_changed": False,
        "benchmark_contamination_changed": False,
        "sealed_final_accessed": False,
        "next_architectural_decision": "reduce-accepted-signal-pilot",
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    return payload


def write_selection_evidence(
    *,
    base_config_path: Path,
    policy_path: Path,
    candidate_config_paths: tuple[Path, ...],
    blockers: tuple[str, ...],
    selection_config_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    """Write the aggregate no-feasible-multiplier result."""

    payload = build_selection_evidence(
        base_config_path=base_config_path,
        policy_path=policy_path,
        candidate_config_paths=candidate_config_paths,
        blockers=blockers,
        selection_config_sha256=selection_config_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
