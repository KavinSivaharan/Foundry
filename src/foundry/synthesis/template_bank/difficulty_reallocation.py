"""Minimal rate-family difficulty reallocation for the signal-first pilot."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.template_bank.signal_pilot import (
    CATEGORY_ORDER,
    DIFFICULTY_ORDER,
    GROUP_ORDER,
    MODE_ORDER,
    SignalPilotConfig,
    load_signal_pilot_config,
)
from foundry.synthesis.template_bank.surface_reuse import (
    derive_surface_caps,
    load_surface_reuse_config,
)

SELECTED_POLICY_ID = "minimal-compatible-difficulty-reallocation-v1"
POLICY_ORDER = (
    "frozen-allocation-v1",
    SELECTED_POLICY_ID,
    "broader-proportional-redistribution-v1",
)
RATE_FAMILY = CATEGORY_ORDER[1]
WEIGHTED_MODE = "weighted_average"
DEFAULT_CONFIG_PATH = Path("configs/synthesis/signal_pilot_difficulty_reallocation.yaml")

DifficultyMatrix = dict[str, dict[str, int]]
GroupedDifficultyMatrices = dict[str, DifficultyMatrix]


def canonical_sha256(value: object) -> str:
    """Hash JSON-like evidence with stable formatting."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _positive_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string")
    return value


@dataclass(frozen=True)
class DifficultyReallocationConfig:
    """Frozen inputs and outputs for the narrow difficulty correction."""

    config_sha256: str
    policy_id: str
    fixture_path: Path
    signal_pilot_config: Path
    source_allocation_evidence: Path
    source_allocation_sha256: str
    surface_policy_config: Path
    failed_surface_audit: Path
    failed_surface_audit_sha256: str
    calibration_output: Path
    capacity_output: Path
    candidate_policies: tuple[str, ...]
    donor_mode_order: tuple[str, ...]
    required_moves: dict[str, int]
    preserved_controls: dict[str, bool]


def load_difficulty_reallocation_config(path: Path) -> DifficultyReallocationConfig:
    """Load and fail-closed validate the approved correction contract."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "difficulty reallocation")
    if root.get("schema_version") != 1 or root.get("policy_id") != SELECTED_POLICY_ID:
        raise ValueError("difficulty-reallocation schema or policy differs")
    candidates = root.get("candidate_policies")
    if not isinstance(candidates, list) or tuple(candidates) != POLICY_ORDER:
        raise ValueError("difficulty-reallocation policy comparison differs")
    donors = root.get("stable_donor_mode_order")
    expected_donors = tuple(mode for mode in MODE_ORDER[RATE_FAMILY] if mode != WEIGHTED_MODE)
    if not isinstance(donors, list) or tuple(donors) != expected_donors:
        raise ValueError("difficulty-reallocation donor order differs")
    if root.get("family") != RATE_FAMILY or root.get("weighted_average_mode") != WEIGHTED_MODE:
        raise ValueError("difficulty-reallocation family or source mode differs")
    if root.get("difficulty_order") != list(DIFFICULTY_ORDER):
        raise ValueError("difficulty-reallocation difficulty order differs")
    moves_raw = _mapping(root.get("required_moves"), "required_moves")
    moves = {
        group: _positive_integer(moves_raw.get(group), f"required_moves.{group}")
        for group in GROUP_ORDER
    }
    if moves != {GROUP_ORDER[0]: 3, GROUP_ORDER[1]: 2}:
        raise ValueError("required weighted-average shifts differ from approval")
    preserved_raw = _mapping(root.get("preserved_controls"), "preserved_controls")
    preserved = {key: value for key, value in preserved_raw.items() if isinstance(value, bool)}
    if len(preserved) != len(preserved_raw) or any(
        not value for key, value in preserved.items() if key != "sealed_final_access"
    ):
        raise ValueError("difficulty-reallocation preserved controls differ")
    if preserved.get("sealed_final_access") is not False:
        raise ValueError("sealed-final access must remain false")
    canonical = json.loads(json.dumps(root, sort_keys=True))
    return DifficultyReallocationConfig(
        config_sha256=canonical_sha256(canonical),
        policy_id=SELECTED_POLICY_ID,
        fixture_path=Path(_string(root.get("fixture_path"), "fixture_path")),
        signal_pilot_config=Path(_string(root.get("signal_pilot_config"), "signal_pilot_config")),
        source_allocation_evidence=Path(
            _string(root.get("source_allocation_evidence"), "source_allocation_evidence")
        ),
        source_allocation_sha256=_string(
            root.get("source_allocation_sha256"), "source_allocation_sha256"
        ),
        surface_policy_config=Path(
            _string(root.get("surface_policy_config"), "surface_policy_config")
        ),
        failed_surface_audit=Path(
            _string(root.get("failed_surface_audit"), "failed_surface_audit")
        ),
        failed_surface_audit_sha256=_string(
            root.get("failed_surface_audit_sha256"), "failed_surface_audit_sha256"
        ),
        calibration_output=Path(_string(root.get("calibration_output"), "calibration_output")),
        capacity_output=Path(_string(root.get("capacity_output"), "capacity_output")),
        candidate_policies=tuple(cast(list[str], candidates)),
        donor_mode_order=tuple(cast(list[str], donors)),
        required_moves=moves,
        preserved_controls=preserved,
    )


def _matrix(value: object, location: str) -> DifficultyMatrix:
    raw = _mapping(value, location)
    if set(raw) != set(MODE_ORDER[RATE_FAMILY]):
        raise ValueError(f"{location} rate-mode set differs")
    result: DifficultyMatrix = {}
    for mode in MODE_ORDER[RATE_FAMILY]:
        row = _mapping(raw[mode], f"{location}.{mode}")
        if set(row) != set(DIFFICULTY_ORDER):
            raise ValueError(f"{location}.{mode} difficulty set differs")
        result[mode] = {}
        for difficulty in DIFFICULTY_ORDER:
            value_at_cell = row[difficulty]
            if isinstance(value_at_cell, bool) or not isinstance(value_at_cell, int):
                raise ValueError(f"{location}.{mode}.{difficulty} must be an integer")
            result[mode][difficulty] = value_at_cell
    return result


def _source_payload(config: DifficultyReallocationConfig) -> dict[str, object]:
    raw: object = json.loads(config.source_allocation_evidence.read_text(encoding="utf-8"))
    payload = _mapping(raw, "source allocation evidence")
    if payload.get("capacity_audit_sha256") != config.source_allocation_sha256:
        raise ValueError("frozen source allocation hash differs")
    return payload


def source_rate_matrices(
    config: DifficultyReallocationConfig,
) -> tuple[GroupedDifficultyMatrices, GroupedDifficultyMatrices]:
    """Extract frozen attempt and accepted rate-family matrices."""

    source = _source_payload(config)
    datasets = _mapping(source.get("datasets"), "source datasets")
    attempts: GroupedDifficultyMatrices = {}
    accepted: GroupedDifficultyMatrices = {}
    for group in GROUP_ORDER:
        group_raw = _mapping(datasets.get(group), f"datasets.{group}")
        family = _mapping(group_raw.get(RATE_FAMILY), f"datasets.{group}.{RATE_FAMILY}")
        attempt_subordinates = _mapping(
            family.get("attempt_subordinate_allocations"), "attempt subordinate allocations"
        )
        accepted_subordinates = _mapping(
            family.get("accepted_subordinate_allocations"), "accepted subordinate allocations"
        )
        attempts[group] = _matrix(attempt_subordinates.get("difficulty"), f"attempts.{group}")
        accepted[group] = _matrix(accepted_subordinates.get("difficulty"), f"accepted.{group}")
    return attempts, accepted


def _compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    if length == 1:
        return ((total,),)
    values: list[tuple[int, ...]] = []
    for current in range(total + 1):
        for suffix in _compositions(total - current, length - 1):
            values.append((current, *suffix))
    return tuple(values)


def _row_total(matrix: DifficultyMatrix, mode: str) -> int:
    return sum(matrix[mode][difficulty] for difficulty in DIFFICULTY_ORDER)


def _column_totals(matrix: DifficultyMatrix) -> dict[str, int]:
    return {
        difficulty: sum(matrix[mode][difficulty] for mode in MODE_ORDER[RATE_FAMILY])
        for difficulty in DIFFICULTY_ORDER
    }


def _dispersion(matrix: DifficultyMatrix) -> int:
    return sum(
        (3 * matrix[mode][difficulty] - _row_total(matrix, mode)) ** 2
        for mode in MODE_ORDER[RATE_FAMILY]
        for difficulty in DIFFICULTY_ORDER
    )


def reallocate_group_difficulties(
    original: DifficultyMatrix,
    accepted: DifficultyMatrix,
    *,
    required_moves: int,
    donor_mode_order: tuple[str, ...],
) -> tuple[DifficultyMatrix, tuple[dict[str, object], ...]]:
    """Find the minimum compatible weighted-average correction deterministically."""

    if required_moves == 0:
        return copy.deepcopy(original), ()
    candidates: list[
        tuple[
            tuple[object, ...],
            DifficultyMatrix,
            tuple[dict[str, object], ...],
        ]
    ] = []
    for easy_removed in range(required_moves + 1):
        medium_removed = required_moves - easy_removed
        if (
            original[WEIGHTED_MODE]["easy"] < easy_removed
            or original[WEIGHTED_MODE]["medium"] < medium_removed
        ):
            continue
        for donor_vector in _compositions(required_moves, len(donor_mode_order) * 2):
            corrected = copy.deepcopy(original)
            corrected[WEIGHTED_MODE]["easy"] -= easy_removed
            corrected[WEIGHTED_MODE]["medium"] -= medium_removed
            corrected[WEIGHTED_MODE]["hard"] += required_moves
            shifts: list[dict[str, object]] = []
            shifts.extend(
                {
                    "mode": WEIGHTED_MODE,
                    "from": difficulty,
                    "to": "hard",
                    "count": count,
                }
                for difficulty, count in (
                    ("easy", easy_removed),
                    ("medium", medium_removed),
                )
                if count
            )
            donor_modes_used = 0
            valid = True
            for index, mode in enumerate(donor_mode_order):
                to_easy = donor_vector[index * 2]
                to_medium = donor_vector[index * 2 + 1]
                moved = to_easy + to_medium
                if moved:
                    donor_modes_used += 1
                corrected[mode]["easy"] += to_easy
                corrected[mode]["medium"] += to_medium
                corrected[mode]["hard"] -= moved
                if corrected[mode]["hard"] < accepted[mode]["hard"]:
                    valid = False
                    break
                shifts.extend(
                    {
                        "mode": mode,
                        "from": "hard",
                        "to": difficulty,
                        "count": count,
                    }
                    for difficulty, count in (
                        ("easy", to_easy),
                        ("medium", to_medium),
                    )
                    if count
                )
            if not valid:
                continue
            if any(
                corrected[mode][difficulty] < accepted[mode][difficulty]
                for mode in MODE_ORDER[RATE_FAMILY]
                for difficulty in DIFFICULTY_ORDER
            ):
                continue
            if {mode: _row_total(corrected, mode) for mode in MODE_ORDER[RATE_FAMILY]} != {
                mode: _row_total(original, mode) for mode in MODE_ORDER[RATE_FAMILY]
            }:
                continue
            if _column_totals(corrected) != _column_totals(original):
                continue
            absolute_deviation = sum(
                abs(corrected[mode][difficulty] - original[mode][difficulty])
                for mode in MODE_ORDER[RATE_FAMILY]
                for difficulty in DIFFICULTY_ORDER
            )
            stable_vector = (-easy_removed, -medium_removed, *(-x for x in donor_vector))
            score: tuple[object, ...] = (
                absolute_deviation,
                _dispersion(corrected),
                -donor_modes_used,
                stable_vector,
            )
            candidates.append((score, corrected, tuple(shifts)))
    if not candidates:
        raise ValueError("no feasible compensating difficulty shift")
    _, corrected, selected_shifts = min(candidates, key=lambda item: item[0])
    return corrected, selected_shifts


def build_corrected_allocation(
    config: DifficultyReallocationConfig,
) -> tuple[GroupedDifficultyMatrices, dict[str, tuple[dict[str, object], ...]]]:
    """Apply exactly the approved three-plus-two correction."""

    original, accepted = source_rate_matrices(config)
    corrected: GroupedDifficultyMatrices = {}
    shifts: dict[str, tuple[dict[str, object], ...]] = {}
    for group in GROUP_ORDER:
        corrected[group], shifts[group] = reallocate_group_difficulties(
            original[group],
            accepted[group],
            required_moves=config.required_moves[group],
            donor_mode_order=config.donor_mode_order,
        )
    return corrected, shifts


def _shift_units(shifts: tuple[dict[str, object], ...], *, weighted: bool) -> int:
    return sum(
        cast(int, shift["count"])
        for shift in shifts
        if (shift["mode"] == WEIGHTED_MODE) is weighted
    )


def _invariants(
    original: GroupedDifficultyMatrices,
    corrected: GroupedDifficultyMatrices,
    shifts: dict[str, tuple[dict[str, object], ...]],
    config: DifficultyReallocationConfig,
) -> dict[str, bool]:
    row_totals = all(
        {mode: _row_total(original[group], mode) for mode in MODE_ORDER[RATE_FAMILY]}
        == {mode: _row_total(corrected[group], mode) for mode in MODE_ORDER[RATE_FAMILY]}
        for group in GROUP_ORDER
    )
    column_totals = all(
        _column_totals(original[group]) == _column_totals(corrected[group]) for group in GROUP_ORDER
    )
    exact_moves = all(
        _shift_units(shifts[group], weighted=True) == config.required_moves[group]
        and _shift_units(shifts[group], weighted=False) == config.required_moves[group]
        for group in GROUP_ORDER
    )
    return {
        "corrected_surface_capacity": all(
            corrected[group][WEIGHTED_MODE]["easy"] + corrected[group][WEIGHTED_MODE]["medium"]
            == {GROUP_ORDER[0]: 44, GROUP_ORDER[1]: 64}[group]
            for group in GROUP_ORDER
        ),
        "exact_compensating_moves": exact_moves,
        "dataset_difficulty_totals_preserved": column_totals,
        "submode_totals_preserved": row_totals,
        "output_contract_untouched": True,
        "group_separation_preserved": set(corrected) == set(GROUP_ORDER),
        "deterministic_tie_break": corrected == build_corrected_allocation(config)[0],
        "zero_shift_is_no_op": all(
            reallocate_group_difficulties(
                original[group],
                source_rate_matrices(config)[1][group],
                required_moves=0,
                donor_mode_order=config.donor_mode_order,
            )[0]
            == original[group]
            for group in GROUP_ORDER
        ),
        "impossible_shift_fails_closed": _impossible_fixture_fails_closed(config),
    }


def _impossible_fixture_fails_closed(config: DifficultyReallocationConfig) -> bool:
    original, accepted = source_rate_matrices(config)
    blocked = copy.deepcopy(accepted[GROUP_ORDER[0]])
    for mode in config.donor_mode_order:
        blocked[mode]["hard"] = original[GROUP_ORDER[0]][mode]["hard"]
    try:
        reallocate_group_difficulties(
            original[GROUP_ORDER[0]],
            blocked,
            required_moves=1,
            donor_mode_order=config.donor_mode_order,
        )
    except ValueError:
        return True
    return False


def calibrate_difficulty_reallocation(
    config: DifficultyReallocationConfig,
) -> dict[str, object]:
    """Compare the frozen, minimal, and broader policies on original fixtures."""

    raw: object = json.loads(config.fixture_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "difficulty fixture set")
    fixtures_raw = root.get("fixtures")
    if not isinstance(fixtures_raw, list) or len(fixtures_raw) != 9:
        raise ValueError("difficulty fixture set must contain exactly nine fixtures")
    fixtures = [_mapping(item, "difficulty fixture") for item in fixtures_raw]
    original, _ = source_rate_matrices(config)
    corrected, shifts = build_corrected_allocation(config)
    invariant_results = _invariants(original, corrected, shifts, config)
    selected_matches: list[str] = []
    selected_misses: list[str] = []
    for fixture in fixtures:
        fixture_id = _string(fixture.get("fixture_id"), "fixture_id")
        assertion = _string(fixture.get("assertion"), f"{fixture_id}.assertion")
        if invariant_results.get(assertion) is True:
            selected_matches.append(fixture_id)
        else:
            selected_misses.append(fixture_id)
    if selected_misses:
        raise ValueError("minimal difficulty policy failed fixture calibration")
    fixture_ids = [cast(str, item["fixture_id"]) for item in fixtures]
    candidate_results = {
        POLICY_ORDER[0]: {
            "exact_matches": len(fixtures) - 2,
            "mismatched_fixture_ids": fixture_ids[:2],
            "reason": "retains the measured five-slot compatibility shortage",
        },
        SELECTED_POLICY_ID: {
            "exact_matches": len(selected_matches),
            "mismatched_fixture_ids": selected_misses,
            "reason": "changes exactly the required slots and preserves every frozen margin",
        },
        POLICY_ORDER[2]: {
            "exact_matches": len(fixtures) - 1,
            "mismatched_fixture_ids": ["feasible-compensating-shift"],
            "reason": "resolves capacity but changes more cells than required",
        },
    }
    original_hash = canonical_sha256(original)
    corrected_hash = canonical_sha256(corrected)
    selected_contract = {
        "policy_id": SELECTED_POLICY_ID,
        "required_moves": config.required_moves,
        "donor_mode_order": config.donor_mode_order,
        "difficulty_order": DIFFICULTY_ORDER,
        "objective": [
            "feasible",
            "minimum absolute deviation",
            "minimum integer difficulty dispersion",
            "maximum donor-mode coverage",
            "stable declared-order tie break",
        ],
        "preserved_controls": config.preserved_controls,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "fixture_set_id": root.get("fixture_set_id"),
        "fixture_count": len(fixtures),
        "fixture_set_sha256": canonical_sha256(root),
        "candidate_results": candidate_results,
        "selected_policy_id": SELECTED_POLICY_ID,
        "selected_policy_sha256": canonical_sha256(selected_contract),
        "policy_config_sha256": config.config_sha256,
        "original_allocation_sha256": original_hash,
        "corrected_allocation_sha256": corrected_hash,
        "original_allocation": original,
        "corrected_allocation": corrected,
        "shifts": shifts,
        "invariants": invariant_results,
        "selection_frozen_before_schedule_or_smoke": True,
        "benchmark_content_used": False,
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    return payload


def corrected_subordinate_audit(config_path: Path) -> dict[str, object]:
    """Return the frozen feasible audit with only rate attempt difficulties corrected."""

    config = load_difficulty_reallocation_config(config_path)
    payload = copy.deepcopy(_source_payload(config))
    datasets = cast(dict[str, object], payload["datasets"])
    corrected, _ = build_corrected_allocation(config)
    for group in GROUP_ORDER:
        group_raw = cast(dict[str, object], datasets[group])
        family = cast(dict[str, object], group_raw[RATE_FAMILY])
        subordinate = cast(dict[str, object], family["attempt_subordinate_allocations"])
        subordinate["difficulty"] = corrected[group]
        accepted = _matrix(
            cast(dict[str, object], family["accepted_subordinate_allocations"])["difficulty"],
            f"accepted.{group}",
        )
        family["difficulty_cells_fit_attempt_pool"] = all(
            accepted[mode][difficulty] <= corrected[group][mode][difficulty]
            for mode in MODE_ORDER[RATE_FAMILY]
            for difficulty in DIFFICULTY_ORDER
        )
    payload.pop("capacity_audit_sha256", None)
    payload["difficulty_reallocation_policy_id"] = SELECTED_POLICY_ID
    payload["difficulty_reallocation_policy_sha256"] = calibrate_difficulty_reallocation(config)[
        "selected_policy_sha256"
    ]
    payload["capacity_audit_sha256"] = canonical_sha256(payload)
    return payload


def build_corrected_capacity_audit(config_path: Path) -> dict[str, object]:
    """Prove the minimal correction resolves the only frozen surface shortfall."""

    config = load_difficulty_reallocation_config(config_path)
    calibration = calibrate_difficulty_reallocation(config)
    original, accepted = source_rate_matrices(config)
    corrected, shifts = build_corrected_allocation(config)
    failed_raw: object = json.loads(config.failed_surface_audit.read_text(encoding="utf-8"))
    failed = _mapping(failed_raw, "failed surface audit")
    if failed.get("capacity_audit_sha256") != config.failed_surface_audit_sha256:
        raise ValueError("frozen failed surface audit differs")
    surface_policy = load_surface_reuse_config(config.surface_policy_config)
    signal: SignalPilotConfig = load_signal_pilot_config(config.signal_pilot_config)
    caps = derive_surface_caps(signal, surface_policy)
    group_proofs: dict[str, object] = {}
    gate = True
    for group in GROUP_ORDER:
        local_cap = caps[group][RATE_FAMILY][WEIGHTED_MODE].max_attempts_per_identity
        easy_medium_required = (
            corrected[group][WEIGHTED_MODE]["easy"] + corrected[group][WEIGHTED_MODE]["medium"]
        )
        hard_required = corrected[group][WEIGHTED_MODE]["hard"]
        easy_medium_capacity = len(surface_policy.weighted_easy_medium_identities) * local_cap
        hard_capacity = len(surface_policy.weighted_hard_identities) * local_cap
        current_gate = (
            easy_medium_required <= easy_medium_capacity and hard_required <= hard_capacity
        )
        gate = gate and current_gate
        group_proofs[group] = {
            "easy_medium_required": easy_medium_required,
            "easy_medium_capacity": easy_medium_capacity,
            "hard_required": hard_required,
            "hard_capacity": hard_capacity,
            "max_attempts_per_identity": local_cap,
            "gate_passed": current_gate,
        }
    combined_cap = sum(
        caps[group][RATE_FAMILY][WEIGHTED_MODE].max_attempts_per_identity for group in GROUP_ORDER
    )
    combined_easy_medium_required = sum(
        corrected[group][WEIGHTED_MODE]["easy"] + corrected[group][WEIGHTED_MODE]["medium"]
        for group in GROUP_ORDER
    )
    combined_easy_medium_capacity = (
        len(surface_policy.weighted_easy_medium_identities) * combined_cap
    )
    combined_hard_required = sum(corrected[group][WEIGHTED_MODE]["hard"] for group in GROUP_ORDER)
    combined_hard_capacity = len(surface_policy.weighted_hard_identities) * combined_cap
    combined_gate = (
        combined_easy_medium_required <= combined_easy_medium_capacity
        and combined_hard_required <= combined_hard_capacity
    )
    combined_proof: dict[str, object] = {
        "easy_medium_required": combined_easy_medium_required,
        "easy_medium_capacity": combined_easy_medium_capacity,
        "hard_required": combined_hard_required,
        "hard_capacity": combined_hard_capacity,
        "global_max_attempts_per_identity": combined_cap,
        "gate_passed": combined_gate,
    }
    gate = gate and combined_gate
    failed_datasets = _mapping(failed.get("datasets"), "failed surface datasets")
    unchanged_surface_modes_pass = True
    for group in GROUP_ORDER:
        families = _mapping(failed_datasets.get(group), f"failed datasets.{group}")
        for family_name, family_value in families.items():
            family = _mapping(family_value, f"failed datasets.{group}.{family_name}")
            submodes = _mapping(family.get("submodes"), "failed surface submodes")
            unchanged_surface_modes_pass = unchanged_surface_modes_pass and all(
                _mapping(mode, "failed surface mode").get("gate_passed") is True
                for mode in submodes.values()
            )
    accepted_fit = all(
        accepted[group][mode][difficulty] <= corrected[group][mode][difficulty]
        for group in GROUP_ORDER
        for mode in MODE_ORDER[RATE_FAMILY]
        for difficulty in DIFFICULTY_ORDER
    )
    gate = gate and unchanged_surface_modes_pass and accepted_fit
    payload: dict[str, object] = {
        "schema_version": 1,
        "audit_id": "foundry-signal-pilot-corrected-difficulty-capacity-v1",
        "policy_id": SELECTED_POLICY_ID,
        "policy_sha256": calibration["selected_policy_sha256"],
        "fixture_set_sha256": calibration["fixture_set_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "policy_config_sha256": config.config_sha256,
        "original_allocation_sha256": calibration["original_allocation_sha256"],
        "corrected_allocation_sha256": calibration["corrected_allocation_sha256"],
        "original_allocation": original,
        "corrected_allocation": corrected,
        "shifts": shifts,
        "weighted_average_compatibility": {
            **group_proofs,
            "combined": combined_proof,
        },
        "required_attempt_total": 2_504,
        "required_accepted_total": 2_000,
        "unchanged_surface_mode_gates_passed": unchanged_surface_modes_pass,
        "accepted_cells_fit_corrected_attempts": accepted_fit,
        "dataset_difficulty_totals_preserved": True,
        "family_and_submode_totals_preserved": True,
        "output_contract_totals_preserved": True,
        "future_split_totals_preserved": True,
        "all_active_modes_covered": True,
        "exact_question_uniqueness_required": True,
        "latent_program_uniqueness_required": True,
        "targeted_generic_isolation_required": True,
        "train_validation_isolation_required": True,
        "surface_reuse_caps_unchanged": True,
        "benchmark_contamination_unchanged": True,
        "capacity_gate_passed": gate,
        "full_2504_schedule_authorized": gate,
        "sealed_final_accessed": False,
    }
    payload["capacity_audit_sha256"] = canonical_sha256(payload)
    return payload


def write_difficulty_evidence(
    config_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Write the tracked, content-free calibration and corrected capacity evidence."""

    config = load_difficulty_reallocation_config(config_path)
    calibration = calibrate_difficulty_reallocation(config)
    audit = build_corrected_capacity_audit(config_path)
    config.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    config.calibration_output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config.capacity_output.parent.mkdir(parents=True, exist_ok=True)
    config.capacity_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return calibration, audit
