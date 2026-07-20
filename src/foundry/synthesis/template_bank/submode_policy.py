"""Capacity-aware submode balancing for the signal-first pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.template_bank.reuse import load_contract
from foundry.synthesis.template_bank.signal_pilot import (
    CATEGORY_ORDER,
    DIFFICULTY_ORDER,
    GROUP_ORDER,
    SPLIT_ORDER,
    TARGET_TYPE_BY_MODE,
    _derive_group_caps,
    balanced_counts,
    load_signal_pilot_config,
)

SELECTED_POLICY_ID = "maximally-balanced-feasible-submodes-v1"


def canonical_sha256(value: object) -> str:
    """Hash JSON-like content with stable formatting."""

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
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a nonempty string")
    return value


@dataclass(frozen=True)
class SubmodePolicyConfig:
    """Frozen policy and raw verified capacities."""

    config_sha256: str
    policy_id: str
    fixture_path: Path
    signal_pilot_config: Path
    source_capacity_audit: Path
    superseded_capacity_audit: Path
    candidate_policies: tuple[str, ...]
    family_order: tuple[str, ...]
    mode_order: dict[str, tuple[str, ...]]
    capacities: dict[str, dict[str, int]]
    difficulty_capacities: dict[str, dict[str, dict[str, int]]]


def load_policy_config(path: Path) -> SubmodePolicyConfig:
    """Load and strictly validate the approved policy configuration."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "submode policy")
    candidates_raw = root.get("candidate_policies")
    families_raw = root.get("stable_family_order")
    if not isinstance(candidates_raw, list) or not all(
        isinstance(item, str) for item in candidates_raw
    ):
        raise ValueError("candidate_policies must be a string list")
    if not isinstance(families_raw, list) or not all(
        isinstance(item, str) for item in families_raw
    ):
        raise ValueError("stable_family_order must be a string list")
    candidate_policies = tuple(cast(list[str], candidates_raw))
    family_order = tuple(cast(list[str], families_raw))
    if candidate_policies != (
        "equal-style-uniform-caps-v1",
        SELECTED_POLICY_ID,
        "proportional-to-capacity-v1",
    ):
        raise ValueError("candidate policy order differs from the approved comparison")
    if root.get("policy_id") != SELECTED_POLICY_ID:
        raise ValueError("selected policy differs from the approved water-filling policy")
    modes_raw = _mapping(root.get("stable_mode_order"), "stable_mode_order")
    capacity_raw = _mapping(
        root.get("verified_unique_attempt_capacity"), "verified_unique_attempt_capacity"
    )
    difficulty_raw = _mapping(
        root.get("verified_unique_attempt_capacity_by_difficulty"),
        "verified_unique_attempt_capacity_by_difficulty",
    )
    if (
        tuple(modes_raw) != family_order
        or tuple(capacity_raw) != family_order
        or tuple(difficulty_raw) != family_order
    ):
        raise ValueError("family order differs across policy sections")
    mode_order: dict[str, tuple[str, ...]] = {}
    capacities: dict[str, dict[str, int]] = {}
    difficulty_capacities: dict[str, dict[str, dict[str, int]]] = {}
    for family in family_order:
        listed = modes_raw[family]
        if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
            raise ValueError(f"stable_mode_order.{family} must be a string list")
        modes = tuple(cast(list[str], listed))
        current_raw = _mapping(capacity_raw[family], f"capacity.{family}")
        if tuple(current_raw) != modes:
            raise ValueError(f"capacity mode order differs for {family}")
        difficulty_family = _mapping(difficulty_raw[family], f"difficulty_capacity.{family}")
        if tuple(difficulty_family) != modes:
            raise ValueError(f"difficulty-capacity mode order differs for {family}")
        mode_order[family] = modes
        capacities[family] = {
            mode: _positive_integer(current_raw[mode], f"capacity.{family}.{mode}")
            for mode in modes
        }
        difficulty_capacities[family] = {}
        for mode in modes:
            mode_difficulties = _mapping(
                difficulty_family[mode], f"difficulty_capacity.{family}.{mode}"
            )
            if tuple(mode_difficulties) != DIFFICULTY_ORDER:
                raise ValueError(f"difficulty-capacity order differs for {family}/{mode}")
            difficulty_capacities[family][mode] = {
                difficulty: _positive_integer(
                    mode_difficulties[difficulty],
                    f"difficulty_capacity.{family}.{mode}.{difficulty}",
                )
                for difficulty in DIFFICULTY_ORDER
            }
    allocation = _mapping(root.get("allocation"), "allocation")
    preserved = _mapping(root.get("preserved_controls"), "preserved_controls")
    if allocation != {
        "global_method": "deterministic-water-filling",
        "dataset_split_method": "largest-remainder",
        "subordinate_method": "stable-minimum-deviation",
        "integer_tie_break": "stable-mode-id-order",
        "unordered_iteration_allowed": False,
        "result_or_benchmark_dependent_weights_allowed": False,
        "manual_weights_allowed": False,
    }:
        raise ValueError("allocation contract differs from Milestone 7B")
    if preserved.get("sealed_final_access") is not False:
        raise ValueError("preserved-control contract is malformed")
    for key, value in preserved.items():
        if key != "sealed_final_access" and value not in {True, "unchanged"}:
            raise ValueError(f"preserved control {key} is not frozen")
    return SubmodePolicyConfig(
        config_sha256=canonical_sha256(root),
        policy_id=SELECTED_POLICY_ID,
        fixture_path=Path(_string(root.get("fixture_path"), "fixture_path")),
        signal_pilot_config=Path(_string(root.get("signal_pilot_config"), "signal_pilot_config")),
        source_capacity_audit=Path(
            _string(root.get("source_capacity_audit"), "source_capacity_audit")
        ),
        superseded_capacity_audit=Path(
            _string(root.get("superseded_capacity_audit"), "superseded_capacity_audit")
        ),
        candidate_policies=candidate_policies,
        family_order=family_order,
        mode_order=mode_order,
        capacities=capacities,
        difficulty_capacities=difficulty_capacities,
    )


def water_fill(total: int, capacities: dict[str, int]) -> dict[str, int]:
    """Allocate maximally evenly, saturating low-capacity modes deterministically."""

    if total < 0 or any(capacity < 0 for capacity in capacities.values()):
        raise ValueError("water-filling quantities must be nonnegative")
    if total > sum(capacities.values()):
        raise ValueError("insufficient_total_capacity")
    result = {mode: 0 for mode in capacities}
    eligible = list(capacities)
    remaining = total
    while eligible:
        base, remainder = divmod(remaining, len(eligible))
        provisional = {mode: base + (index < remainder) for index, mode in enumerate(eligible)}
        saturated = [mode for mode in eligible if capacities[mode] < provisional[mode]]
        if not saturated:
            for mode in eligible:
                result[mode] = provisional[mode]
            return result
        for mode in saturated:
            result[mode] = capacities[mode]
            remaining -= capacities[mode]
        eligible = [mode for mode in eligible if mode not in set(saturated)]
    if remaining:
        raise ValueError("insufficient_total_capacity")
    return result


def largest_remainder_split(
    global_allocation: dict[str, int], first_total: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Split a global allocation proportionally with stable largest remainders."""

    combined = sum(global_allocation.values())
    if first_total < 0 or first_total > combined or combined <= 0:
        raise ValueError("largest-remainder total is invalid")
    first = {
        mode: (quantity * first_total) // combined for mode, quantity in global_allocation.items()
    }
    missing = first_total - sum(first.values())
    ranked = sorted(
        global_allocation,
        key=lambda mode: (
            -((global_allocation[mode] * first_total) % combined),
            tuple(global_allocation).index(mode),
        ),
    )
    for mode in ranked[:missing]:
        first[mode] += 1
    second = {mode: global_allocation[mode] - first[mode] for mode in global_allocation}
    return first, second


def constrained_attempt_split(
    global_attempts: dict[str, int],
    global_accepted: dict[str, int],
    *,
    first_attempt_total: int,
    first_accepted_total: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Split accepted counts by largest remainder and attempts by closest feasible share."""

    first_accepted, second_accepted = largest_remainder_split(global_accepted, first_accepted_total)
    combined = sum(global_attempts.values())
    modes = tuple(global_attempts)
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for mode in modes:
        lower = first_accepted[mode]
        upper = global_attempts[mode] - second_accepted[mode]
        if lower > upper:
            raise ValueError("accepted split cannot fit the attempt pool")
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for allocated, (cost, values) in states.items():
            for quantity in range(lower, upper + 1):
                new_total = allocated + quantity
                if new_total > first_attempt_total:
                    continue
                deviation = quantity * combined - global_attempts[mode] * first_attempt_total
                candidate = (cost + deviation * deviation, values + (quantity,))
                current = next_states.get(new_total)
                if current is None or candidate < current:
                    next_states[new_total] = candidate
        states = next_states
    if first_attempt_total not in states:
        raise ValueError("accepted split cannot fit the attempt pool")
    values = states[first_attempt_total][1]
    first_attempts = {mode: values[index] for index, mode in enumerate(modes)}
    second_attempts = {mode: global_attempts[mode] - first_attempts[mode] for mode in modes}
    return first_attempts, second_attempts, first_accepted, second_accepted


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _add_edge(graph: list[list[_Edge]], start: int, end: int, capacity: int, cost: int) -> None:
    graph[start].append(_Edge(end, len(graph[end]), capacity, cost))
    graph[end].append(_Edge(start, len(graph[start]) - 1, 0, -cost))


def balanced_matrix(
    rows: dict[str, int],
    columns: dict[str, int],
    *,
    cell_capacities: dict[str, dict[str, int]] | None = None,
) -> dict[str, dict[str, int]]:
    """Minimize squared deviation from proportional row/column margins."""

    total = sum(rows.values())
    if total != sum(columns.values()):
        raise ValueError("subordinate margins do not sum")
    row_names = tuple(rows)
    column_names = tuple(columns)
    source = 0
    row_offset = 1
    column_offset = row_offset + len(row_names)
    sink = column_offset + len(column_names)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    for row_index, row in enumerate(row_names):
        _add_edge(graph, source, row_offset + row_index, rows[row], 0)
        for column_index, column in enumerate(column_names):
            cap = (
                cell_capacities[row][column]
                if cell_capacities is not None
                else min(rows[row], columns[column])
            )
            for unit in range(cap):
                incremental = total * (2 * unit + 1) - 2 * rows[row] * columns[column]
                _add_edge(
                    graph,
                    row_offset + row_index,
                    column_offset + column_index,
                    1,
                    incremental,
                )
    for column_index, column in enumerate(column_names):
        _add_edge(graph, column_offset + column_index, sink, columns[column], 0)
    flow = 0
    while flow < total:
        infinity = 10**30
        distance = [infinity] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)
        distance[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for node in range(len(graph)):
                if distance[node] == infinity:
                    continue
                for edge_index, edge in enumerate(graph[node]):
                    candidate = distance[node] + edge.cost
                    if edge.capacity and candidate < distance[edge.to]:
                        distance[edge.to] = candidate
                        previous_node[edge.to] = node
                        previous_edge[edge.to] = edge_index
                        changed = True
            if not changed:
                break
        if previous_node[sink] < 0:
            raise ValueError("impossible_subordinate_compatibility")
        node = sink
        while node != source:
            prior = previous_node[node]
            edge_index = previous_edge[node]
            edge = graph[prior][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = prior
        flow += 1
    result = {row: {column: 0 for column in column_names} for row in row_names}
    for row_index, row in enumerate(row_names):
        node = row_offset + row_index
        for edge in graph[node]:
            if column_offset <= edge.to < sink and edge.cost <= total * 100:
                reverse = graph[edge.to][edge.reverse]
                if reverse.capacity:
                    column = column_names[edge.to - column_offset]
                    result[row][column] += reverse.capacity
    if any(sum(result[row].values()) != rows[row] for row in row_names):
        raise ValueError("balanced matrix row total mismatch")
    if any(
        sum(result[row][column] for row in row_names) != columns[column] for column in column_names
    ):
        raise ValueError("balanced matrix column total mismatch")
    return result


def constrained_difficulty_matrices(
    global_modes: dict[str, int],
    targeted_modes: dict[str, int],
    *,
    targeted_total: int,
    generic_total: int,
    cell_capacities: dict[str, dict[str, int]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Allocate difficulty globally, then split it without exceeding any cell."""

    targeted_difficulties = balanced_counts(targeted_total, DIFFICULTY_ORDER)
    generic_difficulties = balanced_counts(generic_total, DIFFICULTY_ORDER)
    combined_difficulties = {
        difficulty: targeted_difficulties[difficulty] + generic_difficulties[difficulty]
        for difficulty in DIFFICULTY_ORDER
    }
    global_matrix = balanced_matrix(
        global_modes,
        combined_difficulties,
        cell_capacities=cell_capacities,
    )
    targeted_matrix = balanced_matrix(
        targeted_modes,
        targeted_difficulties,
        cell_capacities=global_matrix,
    )
    generic_matrix = {
        mode: {
            difficulty: global_matrix[mode][difficulty] - targeted_matrix[mode][difficulty]
            for difficulty in DIFFICULTY_ORDER
        }
        for mode in global_modes
    }
    if any(
        sum(generic_matrix[mode].values()) != global_modes[mode] - targeted_modes[mode]
        for mode in global_modes
    ):
        raise ValueError("generic difficulty row margin differs")
    if {
        difficulty: sum(generic_matrix[mode][difficulty] for mode in global_modes)
        for difficulty in DIFFICULTY_ORDER
    } != generic_difficulties:
        raise ValueError("generic difficulty column margin differs")
    return {
        "global": global_matrix,
        GROUP_ORDER[0]: targeted_matrix,
        GROUP_ORDER[1]: generic_matrix,
    }


def _equal_style(total: int, capacities: dict[str, int]) -> dict[str, int]:
    base, remainder = divmod(total, len(capacities))
    result = {mode: base + (index < remainder) for index, mode in enumerate(capacities)}
    if any(result[mode] > capacities[mode] for mode in capacities):
        raise ValueError("infeasible_equal_style")
    return result


def _proportional(total: int, capacities: dict[str, int]) -> dict[str, int]:
    capacity_total = sum(capacities.values())
    if total > capacity_total:
        raise ValueError("insufficient_total_capacity")
    result = {mode: (total * cap) // capacity_total for mode, cap in capacities.items()}
    missing = total - sum(result.values())
    ranked = sorted(
        capacities,
        key=lambda mode: (
            -((total * capacities[mode]) % capacity_total),
            tuple(capacities).index(mode),
        ),
    )
    for mode in ranked[:missing]:
        result[mode] += 1
    return result


def calibrate_policy(config: SubmodePolicyConfig) -> dict[str, object]:
    """Compare three frozen policies on original content-free fixtures."""

    raw: object = json.loads(config.fixture_path.read_text(encoding="utf-8"))
    fixture_root = _mapping(raw, "fixture set")
    fixtures_raw = fixture_root.get("fixtures")
    if not isinstance(fixtures_raw, list):
        raise ValueError("fixtures must be a list")
    fixtures = [_mapping(item, "fixture") for item in fixtures_raw]
    results: dict[str, dict[str, object]] = {
        policy: {"exact_matches": 0, "mismatched_fixture_ids": []}
        for policy in config.candidate_policies
    }
    for fixture in fixtures:
        fixture_id = _string(fixture.get("fixture_id"), "fixture_id")
        if "capacities" in fixture:
            capacities = {
                key: _positive_integer(value, f"{fixture_id}.{key}")
                for key, value in _mapping(fixture["capacities"], "capacities").items()
            }
            total = _positive_integer(fixture.get("total"), f"{fixture_id}.total")
            expected_error = fixture.get("expected_error")
            expected = fixture.get("expected_selected")
            for policy, implementation in (
                ("equal-style-uniform-caps-v1", _equal_style),
                (SELECTED_POLICY_ID, water_fill),
                ("proportional-to-capacity-v1", _proportional),
            ):
                try:
                    actual: object = implementation(total, capacities)
                except ValueError as error:
                    actual = str(error)
                wanted: object = expected_error if expected_error is not None else expected
                if actual == wanted:
                    results[policy]["exact_matches"] = (
                        cast(int, results[policy]["exact_matches"]) + 1
                    )
                else:
                    cast(list[str], results[policy]["mismatched_fixture_ids"]).append(fixture_id)
        elif "global_allocation" in fixture:
            allocation = {
                key: _positive_integer(value, f"{fixture_id}.{key}")
                for key, value in _mapping(fixture["global_allocation"], "global").items()
            }
            first, second = largest_remainder_split(
                allocation, _positive_integer(fixture.get("targeted_total"), "targeted_total")
            )
            matched = first == fixture.get("expected_targeted") and second == fixture.get(
                "expected_generic"
            )
            for policy in results:
                if matched:
                    results[policy]["exact_matches"] = (
                        cast(int, results[policy]["exact_matches"]) + 1
                    )
                else:
                    cast(list[str], results[policy]["mismatched_fixture_ids"]).append(fixture_id)
        else:
            rows = {
                key: _positive_integer(value, f"{fixture_id}.{key}")
                for key, value in _mapping(fixture["mode_allocation"], "modes").items()
            }
            columns = {
                key: _positive_integer(value, f"{fixture_id}.{key}")
                for key, value in _mapping(fixture["column_totals"], "columns").items()
            }
            caps_raw = fixture.get("cell_capacities")
            caps = None
            if caps_raw is not None:
                caps = {
                    row: {
                        column: _positive_integer(value, f"{fixture_id}.{row}.{column}")
                        for column, value in _mapping(raw_row, "cell row").items()
                    }
                    for row, raw_row in _mapping(caps_raw, "cell capacities").items()
                }
            try:
                matrix: object = balanced_matrix(rows, columns, cell_capacities=caps)
                matched_matrix = {
                    row: sum(cast(dict[str, dict[str, int]], matrix)[row].values()) for row in rows
                } == fixture.get("expected_row_totals") and {
                    column: sum(
                        cast(dict[str, dict[str, int]], matrix)[row][column] for row in rows
                    )
                    for column in columns
                } == fixture.get("expected_column_totals")
                actual_error = None
            except ValueError as error:
                matched_matrix = False
                actual_error = str(error)
            wanted_error = fixture.get("expected_error")
            matched = actual_error == wanted_error if wanted_error is not None else matched_matrix
            for policy in results:
                if matched:
                    results[policy]["exact_matches"] = (
                        cast(int, results[policy]["exact_matches"]) + 1
                    )
                else:
                    cast(list[str], results[policy]["mismatched_fixture_ids"]).append(fixture_id)
    selected = results[SELECTED_POLICY_ID]
    if selected["exact_matches"] != len(fixtures):
        raise ValueError("selected policy does not satisfy every fixture")
    selected_contract = {
        "policy_id": config.policy_id,
        "algorithm": "iterative deterministic water-filling",
        "dataset_split": "stable largest remainder",
        "subordinate_allocation": "integer minimum squared proportional deviation",
        "family_order": config.family_order,
        "mode_order": config.mode_order,
        "verified_capacity": config.capacities,
        "tie_break": "declared mode and column order",
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "fixture_set_id": fixture_root.get("fixture_set_id"),
        "fixture_count": len(fixtures),
        "fixture_set_sha256": canonical_sha256(fixture_root),
        "config_sha256": config.config_sha256,
        "selected_policy_id": config.policy_id,
        "selected_policy_sha256": canonical_sha256(selected_contract),
        "candidate_results": results,
        "selected_contract": selected_contract,
        "selection_rationale": (
            "complete feasible allocation with minimal concentration and stable broad coverage"
        ),
        "rejected_alternatives": {
            "equal-style-uniform-caps-v1": "fails when a low-capacity mode needs redistribution",
            "proportional-to-capacity-v1": (
                "feasible but concentrates attempts in large modes beyond necessity"
            ),
        },
        "benchmark_or_result_dependent": False,
        "sealed_final_accessed": False,
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    return payload


def write_calibration(config_path: Path, output_path: Path) -> dict[str, object]:
    """Write content-free policy calibration evidence."""

    payload = calibrate_policy(load_policy_config(config_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _normalized_entropy(allocation: dict[str, int]) -> float:
    total = sum(allocation.values())
    if total <= 0 or len(allocation) <= 1:
        return 1.0
    entropy = -sum(
        (quantity / total) * math.log(quantity / total)
        for quantity in allocation.values()
        if quantity
    )
    return entropy / math.log(len(allocation))


def _target_counts(family: str, allocation: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mode, quantity in allocation.items():
        target = TARGET_TYPE_BY_MODE[family][mode]
        counts[target] = counts.get(target, 0) + quantity
    return counts


def _frame_distribution(family: str, allocation: dict[str, int]) -> dict[str, dict[str, int]]:
    frame_count = 18 if family == CATEGORY_ORDER[0] else 4 if family == CATEGORY_ORDER[1] else 5
    return {
        mode: balanced_counts(quantity, tuple(f"frame_{index + 1}" for index in range(frame_count)))
        for mode, quantity in allocation.items()
    }


def _subordinate_allocations(
    mode_allocation: dict[str, int],
    *,
    difficulty_total: int,
    output_enabled: int,
    split_training: int,
    difficulty_matrix: dict[str, dict[str, int]] | None = None,
) -> dict[str, object]:
    total = sum(mode_allocation.values())
    return {
        "difficulty": difficulty_matrix
        if difficulty_matrix is not None
        else balanced_matrix(mode_allocation, balanced_counts(difficulty_total, DIFFICULTY_ORDER)),
        "output_contract": balanced_matrix(
            mode_allocation,
            {"enabled": output_enabled, "disabled": total - output_enabled},
        ),
        "future_split": balanced_matrix(
            mode_allocation,
            {
                SPLIT_ORDER[0]: split_training,
                SPLIT_ORDER[1]: total - split_training,
            },
        ),
    }


def build_revised_capacity_audit(config_path: Path) -> dict[str, object]:
    """Audit the full 2,504-attempt pilot under feasible submode balancing."""

    policy = load_policy_config(config_path)
    calibration = calibrate_policy(policy)
    signal = load_signal_pilot_config(policy.signal_pilot_config)
    reuse = load_contract(signal.reuse_config_path)
    source: object = json.loads(policy.source_capacity_audit.read_text(encoding="utf-8"))
    source_map = _mapping(source, "source capacity")
    superseded: object = json.loads(policy.superseded_capacity_audit.read_text(encoding="utf-8"))
    superseded_map = _mapping(superseded, "superseded capacity")
    if source_map.get("capacity_audit_sha256") != (
        "1a40db7b40005d12f631b534e07f28d4f9974d6516b900975116526575f21129"
    ):
        raise ValueError("raw unique-capacity evidence changed")
    if superseded_map.get("capacity_audit_sha256") != (
        "522b5b4eaeaea3fe7c29542afd78a62532ead6d8c5fa6356abd7db125a307aaf"
    ):
        raise ValueError("Milestone 7A stopped audit changed")
    family_results: dict[str, object] = {}
    dataset_results: dict[str, dict[str, object]] = {group: {} for group in GROUP_ORDER}
    gate = True
    for family in policy.family_order:
        required_attempts = sum(
            signal.datasets[group].families[family].attempts for group in GROUP_ORDER
        )
        required_accepted = sum(
            signal.datasets[group].families[family].accepted for group in GROUP_ORDER
        )
        global_attempts = water_fill(required_attempts, policy.capacities[family])
        global_accepted = water_fill(required_accepted, policy.capacities[family])
        (
            targeted_attempts,
            generic_attempts,
            targeted_accepted,
            generic_accepted,
        ) = constrained_attempt_split(
            global_attempts,
            global_accepted,
            first_attempt_total=signal.datasets[GROUP_ORDER[0]].families[family].attempts,
            first_accepted_total=signal.datasets[GROUP_ORDER[0]].families[family].accepted,
        )
        split_attempts = {
            GROUP_ORDER[0]: targeted_attempts,
            GROUP_ORDER[1]: generic_attempts,
        }
        split_accepted = {
            GROUP_ORDER[0]: targeted_accepted,
            GROUP_ORDER[1]: generic_accepted,
        }
        attempt_difficulties = constrained_difficulty_matrices(
            global_attempts,
            targeted_attempts,
            targeted_total=signal.datasets[GROUP_ORDER[0]].families[family].attempts,
            generic_total=signal.datasets[GROUP_ORDER[1]].families[family].attempts,
            cell_capacities=policy.difficulty_capacities[family],
        )
        accepted_difficulties = {
            group: balanced_matrix(
                split_accepted[group],
                balanced_counts(
                    signal.datasets[group].families[family].accepted,
                    DIFFICULTY_ORDER,
                ),
                cell_capacities=attempt_difficulties[group],
            )
            for group in GROUP_ORDER
        }
        accepted_difficulties["global"] = {
            mode: {
                difficulty: sum(
                    accepted_difficulties[group][mode][difficulty] for group in GROUP_ORDER
                )
                for difficulty in DIFFICULTY_ORDER
            }
            for mode in global_accepted
        }
        for group in GROUP_ORDER:
            quota = signal.datasets[group].families[family]
            if quota.attempt_modes != split_attempts[group]:
                raise ValueError(f"{group}/{family} attempt allocation differs from water-filling")
            if quota.accepted_modes != split_accepted[group]:
                raise ValueError(f"{group}/{family} accepted allocation differs from water-filling")
            caps = _derive_group_caps(signal, reuse.identity_inventory, group, family)
            attempt_layers = cast(dict[str, int], caps["attempt_layer_capacity"])
            accepted_layers = cast(dict[str, int], caps["accepted_layer_capacity"])
            retained_attempt_surface = min(
                attempt_layers[key]
                for key in ("sentence_plan", "number_neutral", "plan_scenario_domain")
            )
            retained_accepted_surface = min(
                accepted_layers[key]
                for key in ("sentence_plan", "number_neutral", "plan_scenario_domain")
            )
            attempt_subordinates = _subordinate_allocations(
                quota.attempt_modes,
                difficulty_total=quota.attempts,
                output_enabled=quota.output_contract_attempts,
                split_training=quota.training_attempts,
                difficulty_matrix=attempt_difficulties[group],
            )
            accepted_subordinates = _subordinate_allocations(
                quota.accepted_modes,
                difficulty_total=quota.accepted,
                output_enabled=quota.output_contract_accepted,
                split_training=quota.training_accepted,
                difficulty_matrix=accepted_difficulties[group],
            )
            difficulty_cells_fit = all(
                accepted_difficulties[group][mode][difficulty]
                <= attempt_difficulties[group][mode][difficulty]
                for mode in quota.attempt_modes
                for difficulty in DIFFICULTY_ORDER
            )
            group_gate = (
                quota.attempts <= retained_attempt_surface
                and quota.accepted <= retained_accepted_surface
                and all(
                    quota.attempt_modes[mode] <= policy.capacities[family][mode]
                    for mode in quota.attempt_modes
                )
                and all(
                    quota.accepted_modes[mode] <= quota.attempt_modes[mode]
                    for mode in quota.accepted_modes
                )
                and difficulty_cells_fit
            )
            gate = gate and group_gate
            dataset_results[group][family] = {
                "required_accepted": quota.accepted,
                "required_attempts": quota.attempts,
                "assigned_attempt_modes": quota.attempt_modes,
                "assigned_accepted_modes": quota.accepted_modes,
                "target_type_attempts": _target_counts(family, quota.attempt_modes),
                "target_type_accepted": _target_counts(family, quota.accepted_modes),
                "semantic_frame_attempt_distribution": _frame_distribution(
                    family, quota.attempt_modes
                ),
                "semantic_frame_accepted_distribution": _frame_distribution(
                    family, quota.accepted_modes
                ),
                "attempt_subordinate_allocations": attempt_subordinates,
                "accepted_subordinate_allocations": accepted_subordinates,
                "retained_surface_attempt_capacity": retained_attempt_surface,
                "retained_surface_accepted_capacity": retained_accepted_surface,
                "sentence_plan_attempt_cap": cast(dict[str, int], caps["attempt_caps"])[
                    "sentence_plan"
                ],
                "number_neutral_attempt_cap": cast(dict[str, int], caps["attempt_caps"])[
                    "number_neutral"
                ],
                "plan_scenario_attempt_cap": cast(dict[str, int], caps["attempt_caps"])[
                    "plan_scenario_domain"
                ],
                "maximum_submode_concentration": max(quota.attempt_modes.values()) / quota.attempts,
                "normalized_submode_entropy": _normalized_entropy(quota.attempt_modes),
                "difficulty_cells_fit_attempt_pool": difficulty_cells_fit,
                "gate_passed": group_gate,
            }
        mode_details = {
            mode: {
                "required_attempts": global_attempts[mode],
                "required_accepted": global_accepted[mode],
                "compatible_unique_capacity": policy.capacities[family][mode],
                "compatible_capacity_by_difficulty": policy.difficulty_capacities[family][mode],
                "assigned_attempts_by_difficulty": attempt_difficulties["global"][mode],
                "remaining_attempt_headroom": policy.capacities[family][mode]
                - global_attempts[mode],
                "remaining_difficulty_headroom": {
                    difficulty: policy.difficulty_capacities[family][mode][difficulty]
                    - attempt_difficulties["global"][mode][difficulty]
                    for difficulty in DIFFICULTY_ORDER
                },
                "target_type": TARGET_TYPE_BY_MODE[family][mode],
                "coverage_nonzero": global_attempts[mode] > 0,
            }
            for mode in policy.mode_order[family]
        }
        family_gate = (
            sum(global_attempts.values()) == required_attempts
            and sum(global_accepted.values()) == required_accepted
            and all(
                cast(int, item["remaining_attempt_headroom"]) >= 0 for item in mode_details.values()
            )
            and all(cast(bool, item["coverage_nonzero"]) for item in mode_details.values())
            and all(_target_counts(family, global_attempts).values())
            and all(
                headroom >= 0
                for item in mode_details.values()
                for headroom in cast(dict[str, int], item["remaining_difficulty_headroom"]).values()
            )
        )
        gate = gate and family_gate
        family_results[family] = {
            "required_accepted": required_accepted,
            "required_attempts": required_attempts,
            "available_unique_capacity": sum(policy.capacities[family].values()),
            "remaining_family_headroom": sum(policy.capacities[family].values())
            - required_attempts,
            "assigned_attempt_modes": global_attempts,
            "assigned_accepted_modes": global_accepted,
            "target_type_attempts": _target_counts(family, global_attempts),
            "mode_details": mode_details,
            "maximum_submode_concentration": max(global_attempts.values()) / required_attempts,
            "normalized_submode_entropy": _normalized_entropy(global_attempts),
            "gate_passed": family_gate,
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "audit_id": "foundry-signal-pilot-feasible-capacity-v1",
        "policy_id": policy.policy_id,
        "policy_sha256": calibration["selected_policy_sha256"],
        "fixture_set_sha256": calibration["fixture_set_sha256"],
        "policy_config_sha256": policy.config_sha256,
        "signal_config_sha256": signal.config_sha256,
        "superseded_audit_sha256": superseded_map["capacity_audit_sha256"],
        "required_accepted_total": 2_000,
        "required_attempt_total": 2_504,
        "families": family_results,
        "datasets": dataset_results,
        "exact_question_uniqueness_required": True,
        "global_latent_uniqueness_required": True,
        "cross_dataset_isolation_required": True,
        "train_validation_isolation_required": True,
        "retained_template_scenario_caps_passed": gate,
        "difficulty_totals_exact": True,
        "output_contract_totals_exact": True,
        "future_split_totals_exact": True,
        "capacity_gate_passed": gate,
        "full_2504_schedule_feasible": gate,
        "allocator_implemented": False,
        "full_schedule_created": False,
        "fresh_smoke_run": False,
        "review_packet_created": False,
        "generator_modes_changed": False,
        "generator_ranges_changed": False,
        "templates_changed": False,
        "labels_or_verifiers_changed": False,
        "benchmark_contamination_changed": False,
        "sealed_final_accessed": False,
    }
    payload["capacity_audit_sha256"] = canonical_sha256(payload)
    return payload


def write_revised_capacity_audit(config_path: Path, output_path: Path) -> dict[str, object]:
    """Write the content-free compatibility-aware capacity evidence."""

    payload = build_revised_capacity_audit(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
