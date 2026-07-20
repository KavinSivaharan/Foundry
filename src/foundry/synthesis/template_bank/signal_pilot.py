"""Signal-first pilot quota, capacity, and deterministic allocation contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.taxonomy import FailureCategory
from foundry.synthesis.template_bank.bank import build_template_bank
from foundry.synthesis.template_bank.reuse import calibrate, load_contract, load_fixtures

_BOOKKEEPING = str(FailureCategory.MULTI_STEP_BOOKKEEPING)
_RATES = str(FailureCategory.RATE_RATIO_PERCENTAGE)
_DISCRETE = str(FailureCategory.CONSTRAINT_DISCRETE)
CATEGORY_ORDER = (_BOOKKEEPING, _RATES, _DISCRETE)
GROUP_ORDER = ("targeted", "generic_control")
DIFFICULTY_ORDER = ("easy", "medium", "hard")
SPLIT_ORDER = ("training", "synthetic_validation")

MODE_ORDER: dict[str, tuple[str, ...]] = {
    _BOOKKEEPING: ("inventory", "grouping"),
    _RATES: (
        "rate_total",
        "ratio_scale",
        "percentage",
        "weighted_average",
        "combined_rate",
    ),
    _DISCRETE: (
        "two_type_allocation",
        "complete_packages",
        "equal_distribution",
        "dual_capacity",
    ),
}
TARGET_TYPE_BY_MODE: dict[str, dict[str, str]] = {
    _BOOKKEEPING: {
        "inventory": "remaining_quantity",
        "grouping": "group_count",
    },
    _RATES: {
        "rate_total": "total_quantity",
        "ratio_scale": "ratio",
        "percentage": "percentage",
        "weighted_average": "weighted_mean",
        "combined_rate": "total_quantity",
    },
    _DISCRETE: {
        "two_type_allocation": "count",
        "complete_packages": "group_count",
        "equal_distribution": "count",
        "dual_capacity": "capacity",
    },
}
_FRAMES_PER_MODE = {_BOOKKEEPING: 9, _RATES: 4, _DISCRETE: 5}
_EXACT_MODE_UPPER_BOUNDS: dict[str, dict[str, int | None]] = {
    _BOOKKEEPING: {"inventory": None, "grouping": None},
    _RATES: {
        "rate_total": 96,
        "ratio_scale": 336,
        "percentage": 104,
        "weighted_average": None,
        "combined_rate": 384,
    },
    _DISCRETE: {
        "two_type_allocation": None,
        "complete_packages": 1344,
        "equal_distribution": 253,
        "dual_capacity": 90,
    },
}


@dataclass(frozen=True)
class FamilyQuota:
    """One dataset/family quota with fixed split, output, and mode allocations."""

    accepted: int
    attempts: int
    training_accepted: int
    validation_accepted: int
    training_attempts: int
    validation_attempts: int
    output_contract_accepted: int
    output_contract_attempts: int
    accepted_modes: dict[str, int]
    attempt_modes: dict[str, int]


@dataclass(frozen=True)
class DatasetQuota:
    """The frozen 1,000-example dataset contract."""

    accepted_total: int
    training_accepted: int
    validation_accepted: int
    output_contract_accepted: int
    families: dict[str, FamilyQuota]


@dataclass(frozen=True)
class SmokeQuota:
    """One group of the 120-question review smoke."""

    family_counts: dict[str, int]
    output_contract_attempts: int
    mode_counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class SmokeContract:
    """Paths and quotas for the fresh bounded review smoke."""

    run_id: str
    master_seed: str
    attempts: int
    datasets: dict[str, SmokeQuota]
    semantic_config: Path
    development_manifest: Path
    evaluation_config: Path
    raw_directory: Path
    summary_path: Path
    schedule_path: Path
    human_review_markdown: Path
    human_review_html: Path
    codex_audit_path: Path
    codex_assisted_html: Path
    export_filename: str


@dataclass(frozen=True)
class SignalPilotConfig:
    """Complete, hashable signal-first pilot contract."""

    config_sha256: str
    pilot_id: str
    capacity_audit_id: str
    allocator_id: str
    difficulty_reallocation_path: Path
    reuse_config_path: Path
    reuse_policy_id: str
    reuse_policy_sha256: str
    attempt_numerator: int
    attempt_denominator: int
    datasets: dict[str, DatasetQuota]
    full_schedule_master_seed: str
    full_schedule_candidate_pool_per_mode_difficulty: int
    full_schedule_raw_path: Path
    full_schedule_summary_path: Path
    smoke: SmokeContract


def canonical_sha256(value: object) -> str:
    """Hash JSON-like content with deterministic formatting."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string")
    return value


def _family_quota(raw: object, category: str, location: str) -> FamilyQuota:
    item = _mapping(raw, location)
    accepted_modes = {
        key: _integer(value, f"{location}.accepted_modes.{key}")
        for key, value in _mapping(item.get("accepted_modes"), location).items()
    }
    attempt_modes = {
        key: _integer(value, f"{location}.attempt_modes.{key}")
        for key, value in _mapping(item.get("attempt_modes"), location).items()
    }
    if (
        tuple(accepted_modes) != MODE_ORDER[category]
        or tuple(attempt_modes) != MODE_ORDER[category]
    ):
        raise ValueError(f"{location} mode order differs from the frozen generator modes")
    quota = FamilyQuota(
        accepted=_integer(item.get("accepted"), f"{location}.accepted"),
        attempts=_integer(item.get("attempts"), f"{location}.attempts"),
        training_accepted=_integer(item.get("training_accepted"), f"{location}.training_accepted"),
        validation_accepted=_integer(
            item.get("validation_accepted"), f"{location}.validation_accepted"
        ),
        training_attempts=_integer(item.get("training_attempts"), f"{location}.training_attempts"),
        validation_attempts=_integer(
            item.get("validation_attempts"), f"{location}.validation_attempts"
        ),
        output_contract_accepted=_integer(
            item.get("output_contract_accepted"), f"{location}.output_contract_accepted"
        ),
        output_contract_attempts=_integer(
            item.get("output_contract_attempts"), f"{location}.output_contract_attempts"
        ),
        accepted_modes=accepted_modes,
        attempt_modes=attempt_modes,
    )
    if quota.training_accepted + quota.validation_accepted != quota.accepted:
        raise ValueError(f"{location} accepted split does not sum")
    if quota.training_attempts + quota.validation_attempts != quota.attempts:
        raise ValueError(f"{location} attempt split does not sum")
    if sum(quota.accepted_modes.values()) != quota.accepted:
        raise ValueError(f"{location} accepted modes do not sum")
    if sum(quota.attempt_modes.values()) != quota.attempts:
        raise ValueError(f"{location} attempt modes do not sum")
    if any(quota.accepted_modes[mode] > quota.attempt_modes[mode] for mode in accepted_modes):
        raise ValueError(f"{location} accepted mode exceeds its attempt pool")
    return quota


def _dataset_quota(raw: object, group: str) -> DatasetQuota:
    item = _mapping(raw, f"datasets.{group}")
    families_raw = _mapping(item.get("families"), f"datasets.{group}.families")
    if tuple(families_raw) != CATEGORY_ORDER:
        raise ValueError(f"datasets.{group} family order differs")
    families = {
        category: _family_quota(
            families_raw[category], category, f"datasets.{group}.families.{category}"
        )
        for category in CATEGORY_ORDER
    }
    result = DatasetQuota(
        accepted_total=_integer(item.get("accepted_total"), f"datasets.{group}.accepted_total"),
        training_accepted=_integer(
            item.get("training_accepted"), f"datasets.{group}.training_accepted"
        ),
        validation_accepted=_integer(
            item.get("validation_accepted"), f"datasets.{group}.validation_accepted"
        ),
        output_contract_accepted=_integer(
            item.get("output_contract_accepted"),
            f"datasets.{group}.output_contract_accepted",
        ),
        families=families,
    )
    if result.accepted_total != 1000:
        raise ValueError("each signal-first dataset must contain exactly 1,000 acceptances")
    if (result.training_accepted, result.validation_accepted) != (900, 100):
        raise ValueError("each signal-first dataset must freeze a 900/100 split")
    if result.output_contract_accepted != 200:
        raise ValueError("each signal-first dataset must contain 200 output-track acceptances")
    if sum(value.accepted for value in families.values()) != result.accepted_total:
        raise ValueError(f"datasets.{group} family acceptances do not sum")
    if sum(value.training_accepted for value in families.values()) != 900:
        raise ValueError(f"datasets.{group} training acceptances do not sum")
    if sum(value.validation_accepted for value in families.values()) != 100:
        raise ValueError(f"datasets.{group} validation acceptances do not sum")
    if sum(value.output_contract_accepted for value in families.values()) != 200:
        raise ValueError(f"datasets.{group} output acceptances do not sum")
    return result


def _smoke_quota(raw: object, group: str) -> SmokeQuota:
    item = _mapping(raw, f"smoke.datasets.{group}")
    families = {
        key: _integer(value, f"smoke.datasets.{group}.family_counts.{key}")
        for key, value in _mapping(item.get("family_counts"), "family_counts").items()
    }
    modes_raw = _mapping(item.get("mode_counts"), "mode_counts")
    if tuple(families) != CATEGORY_ORDER or tuple(modes_raw) != CATEGORY_ORDER:
        raise ValueError("smoke category order differs")
    modes: dict[str, dict[str, int]] = {}
    for category in CATEGORY_ORDER:
        current = {
            key: _integer(value, f"smoke.{group}.{category}.{key}")
            for key, value in _mapping(modes_raw[category], "mode counts").items()
        }
        if tuple(current) != MODE_ORDER[category] or sum(current.values()) != families[category]:
            raise ValueError("smoke mode allocation differs from its family count")
        modes[category] = current
    result = SmokeQuota(
        family_counts=families,
        output_contract_attempts=_integer(
            item.get("output_contract_attempts"), "output_contract_attempts"
        ),
        mode_counts=modes,
    )
    if sum(families.values()) != 60 or result.output_contract_attempts != 12:
        raise ValueError("smoke group must contain 60 attempts and 12 output-track attempts")
    return result


def load_signal_pilot_config(path: Path) -> SignalPilotConfig:
    """Load and strictly validate the approved signal-first contract."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "signal pilot config")
    if root.get("schema_version") != 1:
        raise ValueError("signal-pilot schema version differs")
    reuse = _mapping(root.get("reuse_policy"), "reuse_policy")
    multiplier = _mapping(root.get("attempt_multiplier"), "attempt_multiplier")
    datasets_raw = _mapping(root.get("datasets"), "datasets")
    if tuple(datasets_raw) != GROUP_ORDER:
        raise ValueError("signal-pilot dataset order differs")
    datasets = {group: _dataset_quota(datasets_raw[group], group) for group in GROUP_ORDER}
    expected_attempts = {
        "targeted": {_BOOKKEEPING: 688, _RATES: 292, _DISCRETE: 272},
        "generic_control": {_BOOKKEEPING: 418, _RATES: 417, _DISCRETE: 417},
    }
    if {
        group: {category: quota.attempts for category, quota in datasets[group].families.items()}
        for group in GROUP_ORDER
    } != expected_attempts:
        raise ValueError("signal-first fixed attempt pools differ from approval")
    if (
        sum(quota.attempts for dataset in datasets.values() for quota in dataset.families.values())
        != 2504
    ):
        raise ValueError("signal-first fixed attempt total must equal 2,504")
    smoke_raw = _mapping(root.get("smoke"), "smoke")
    smoke_datasets = _mapping(smoke_raw.get("datasets"), "smoke.datasets")
    smoke = SmokeContract(
        run_id=_string(smoke_raw.get("run_id"), "smoke.run_id"),
        master_seed=_string(smoke_raw.get("master_seed"), "smoke.master_seed"),
        attempts=_integer(smoke_raw.get("attempts"), "smoke.attempts"),
        datasets={group: _smoke_quota(smoke_datasets[group], group) for group in GROUP_ORDER},
        semantic_config=Path(_string(smoke_raw.get("semantic_config"), "semantic_config")),
        development_manifest=Path(
            _string(smoke_raw.get("development_manifest"), "development_manifest")
        ),
        evaluation_config=Path(_string(smoke_raw.get("evaluation_config"), "evaluation_config")),
        raw_directory=Path(_string(smoke_raw.get("raw_directory"), "raw_directory")),
        summary_path=Path(_string(smoke_raw.get("summary_path"), "summary_path")),
        schedule_path=Path(_string(smoke_raw.get("schedule_path"), "schedule_path")),
        human_review_markdown=Path(
            _string(smoke_raw.get("human_review_markdown"), "human_review_markdown")
        ),
        human_review_html=Path(_string(smoke_raw.get("human_review_html"), "human_review_html")),
        codex_audit_path=Path(_string(smoke_raw.get("codex_audit_path"), "codex_audit_path")),
        codex_assisted_html=Path(
            _string(smoke_raw.get("codex_assisted_html"), "codex_assisted_html")
        ),
        export_filename=_string(smoke_raw.get("export_filename"), "export_filename"),
    )
    if smoke.attempts != 120 or smoke.export_filename != (
        "foundry-signal-pilot-template-review.json"
    ):
        raise ValueError("signal-pilot smoke size or export filename differs")
    safety = _mapping(root.get("safety"), "safety")
    if safety != {
        "benchmark_answers_allowed": False,
        "sealed_final_allowed": False,
        "llm_generation_allowed": False,
        "full_dataset_generation_allowed": False,
        "training_allowed": False,
    }:
        raise ValueError("signal-pilot safety boundary differs")
    if root.get("difficulty_order") != list(DIFFICULTY_ORDER):
        raise ValueError("difficulty order differs")
    if root.get("split_order") != list(SPLIT_ORDER):
        raise ValueError("split order differs")
    canonical = json.loads(json.dumps(root, sort_keys=True))
    return SignalPilotConfig(
        config_sha256=canonical_sha256(canonical),
        pilot_id=_string(root.get("pilot_id"), "pilot_id"),
        capacity_audit_id=_string(root.get("capacity_audit_id"), "capacity_audit_id"),
        allocator_id=_string(root.get("allocator_id"), "allocator_id"),
        difficulty_reallocation_path=Path(
            _string(
                root.get("difficulty_reallocation_path"),
                "difficulty_reallocation_path",
            )
        ),
        reuse_config_path=Path(_string(reuse.get("config_path"), "reuse config path")),
        reuse_policy_id=_string(reuse.get("policy_id"), "reuse policy id"),
        reuse_policy_sha256=_string(reuse.get("policy_sha256"), "reuse policy hash"),
        attempt_numerator=_integer(multiplier.get("numerator"), "attempt numerator"),
        attempt_denominator=_integer(multiplier.get("denominator"), "attempt denominator"),
        datasets=datasets,
        full_schedule_master_seed=_string(
            root.get("full_schedule_master_seed"), "full schedule master seed"
        ),
        full_schedule_candidate_pool_per_mode_difficulty=_integer(
            root.get("full_schedule_candidate_pool_per_mode_difficulty"),
            "candidate pool per mode/difficulty",
        ),
        full_schedule_raw_path=Path(
            _string(root.get("full_schedule_raw_path"), "full schedule raw path")
        ),
        full_schedule_summary_path=Path(
            _string(root.get("full_schedule_summary_path"), "full schedule summary path")
        ),
        smoke=smoke,
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _reuse_cap(config: SignalPilotConfig, quantity: int, identities: int) -> int:
    return _ceil_div(
        config.attempt_numerator * quantity,
        config.attempt_denominator * identities,
    )


def balanced_counts(total: int, labels: tuple[str, ...]) -> dict[str, int]:
    """Divide an integer as evenly as possible in stable label order."""

    base, remainder = divmod(total, len(labels))
    return {label: base + (index < remainder) for index, label in enumerate(labels)}


def _reserve_capacity(required: dict[str, int], available: int) -> dict[str, int]:
    """Reserve all required counts, then spread shared headroom deterministically."""

    total_required = sum(required.values())
    if total_required > available:
        result = {
            label: (available * quantity) // total_required for label, quantity in required.items()
        }
        remainder = available - sum(result.values())
        ranked = sorted(
            required,
            key=lambda label: (
                -((available * required[label]) % total_required),
                tuple(required).index(label),
            ),
        )
        for label in ranked[:remainder]:
            result[label] += 1
        return result
    result = dict(required)
    extra = available - total_required
    labels = tuple(required)
    for index in range(extra):
        result[labels[index % len(labels)]] += 1
    return result


def _derive_group_caps(
    config: SignalPilotConfig,
    inventory: dict[str, dict[str, int]],
    group: str,
    category: str,
) -> dict[str, object]:
    quota = config.datasets[group].families[category]
    current = inventory[category]
    attempts = {
        "sentence_plan": _reuse_cap(config, quota.attempts, current["sentence_plans"]),
        "number_neutral": _reuse_cap(config, quota.attempts, current["number_neutral_signatures"]),
        "plan_scenario_domain": _reuse_cap(
            config, quota.attempts, current["plan_scenario_domains"]
        ),
        "semantic_frame": _reuse_cap(config, quota.attempts, current["semantic_frames"]),
        "target_type": _reuse_cap(config, quota.attempts, current["target_types"]),
    }
    accepted = {
        "sentence_plan": _reuse_cap(config, quota.accepted, current["sentence_plans"]),
        "number_neutral": _reuse_cap(config, quota.accepted, current["number_neutral_signatures"]),
        "plan_scenario_domain": _reuse_cap(
            config, quota.accepted, current["plan_scenario_domains"]
        ),
        "semantic_frame": _reuse_cap(config, quota.accepted, current["semantic_frames"]),
        "target_type": _reuse_cap(config, quota.accepted, current["target_types"]),
    }
    attempt_layers = {
        key: attempts[key]
        * current[
            {
                "sentence_plan": "sentence_plans",
                "number_neutral": "number_neutral_signatures",
                "plan_scenario_domain": "plan_scenario_domains",
                "semantic_frame": "semantic_frames",
                "target_type": "target_types",
            }[key]
        ]
        for key in attempts
    }
    accepted_layers = {
        key: accepted[key]
        * current[
            {
                "sentence_plan": "sentence_plans",
                "number_neutral": "number_neutral_signatures",
                "plan_scenario_domain": "plan_scenario_domains",
                "semantic_frame": "semantic_frames",
                "target_type": "target_types",
            }[key]
        ]
        for key in accepted
    }
    return {
        "attempt_caps": attempts,
        "accepted_caps": accepted,
        "attempt_layer_capacity": attempt_layers,
        "accepted_layer_capacity": accepted_layers,
        "surface_attempt_capacity": min(attempt_layers.values()),
        "surface_accepted_capacity": min(accepted_layers.values()),
        "maximum_plan_attempt_concentration": attempts["sentence_plan"] / quota.attempts,
        "maximum_number_neutral_attempt_concentration": (
            attempts["number_neutral"] / quota.attempts
        ),
    }


def _mode_capacities(
    *,
    category: str,
    caps_by_group: dict[str, dict[str, object]],
    cap_kind: str,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    key = "attempt_caps" if cap_kind == "attempt" else "accepted_caps"
    per_group: dict[str, dict[str, int]] = {}
    for group in GROUP_ORDER:
        frame_cap = cast(dict[str, int], caps_by_group[group][key])["semantic_frame"]
        target_cap = cast(dict[str, int], caps_by_group[group][key])["target_type"]
        current: dict[str, int] = {}
        for mode in MODE_ORDER[category]:
            structural_cap = (
                target_cap if category == _BOOKKEEPING else frame_cap * _FRAMES_PER_MODE[category]
            )
            exact = _EXACT_MODE_UPPER_BOUNDS[category][mode]
            current[mode] = structural_cap if exact is None else min(structural_cap, exact)
        per_group[group] = current
    combined: dict[str, int] = {}
    for mode in MODE_ORDER[category]:
        exact = _EXACT_MODE_UPPER_BOUNDS[category][mode]
        summed = sum(per_group[group][mode] for group in GROUP_ORDER)
        combined[mode] = summed if exact is None else min(summed, exact)
    return per_group, combined


def _maximum_flow(capacities: dict[tuple[str, str], int], source: str, sink: str) -> int:
    """Return an exact integer maximum flow for the small compatibility graph."""

    residual: dict[str, dict[str, int]] = {}
    for (start, end), capacity in capacities.items():
        residual.setdefault(start, {})[end] = capacity
        residual.setdefault(end, {}).setdefault(start, 0)
    total = 0
    while True:
        parent: dict[str, str | None] = {source: None}
        queue = [source]
        for node in queue:
            for neighbor in sorted(residual.get(node, {})):
                if residual[node][neighbor] > 0 and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
            if sink in parent:
                break
        if sink not in parent:
            return total
        amount = sum(capacities.values())
        current = sink
        while parent[current] is not None:
            prior = cast(str, parent[current])
            amount = min(amount, residual[prior][current])
            current = prior
        current = sink
        while parent[current] is not None:
            prior = cast(str, parent[current])
            residual[prior][current] -= amount
            residual[current][prior] += amount
            current = prior
        total += amount


def _compatible_capacity(
    *,
    category: str,
    groups: tuple[str, ...],
    caps_by_group: dict[str, dict[str, object]],
    group_mode_capacity: dict[str, dict[str, int]],
    cap_kind: str,
) -> tuple[int, dict[str, object]]:
    """Combine per-target caps with group and globally finite mode supply."""

    cap_key = "attempt_caps" if cap_kind == "attempt" else "accepted_caps"
    edge_capacity: dict[tuple[str, str], int] = {}
    target_caps: dict[str, dict[str, int]] = {}
    mode_caps: dict[str, int] = {}
    for group in groups:
        target_cap = cast(dict[str, int], caps_by_group[group][cap_key])["target_type"]
        target_caps[group] = {
            target: target_cap for target in sorted(set(TARGET_TYPE_BY_MODE[category].values()))
        }
        for target, capacity in target_caps[group].items():
            edge_capacity[("source", f"target:{group}:{target}")] = capacity
        for mode in MODE_ORDER[category]:
            target = TARGET_TYPE_BY_MODE[category][mode]
            group_mode = f"group-mode:{group}:{mode}"
            edge_capacity[(f"target:{group}:{target}", group_mode)] = group_mode_capacity[group][
                mode
            ]
            edge_capacity[(group_mode, f"mode:{mode}")] = group_mode_capacity[group][mode]
    for mode in MODE_ORDER[category]:
        exact = _EXACT_MODE_UPPER_BOUNDS[category][mode]
        available = sum(group_mode_capacity[group][mode] for group in groups)
        mode_caps[mode] = available if exact is None else min(available, exact)
        edge_capacity[(f"mode:{mode}", "sink")] = mode_caps[mode]
    return _maximum_flow(edge_capacity, "source", "sink"), {
        "target_type_caps": target_caps,
        "shared_mode_caps": mode_caps,
        "target_type_by_mode": TARGET_TYPE_BY_MODE[category],
    }


def _stratum_records(
    *,
    required_accepted: dict[str, int],
    required_attempts: dict[str, int],
    accepted_capacity: int,
    attempt_capacity: int,
    limiting_modes: tuple[str, ...],
    maximum_plan_concentration: float,
) -> dict[str, dict[str, object]]:
    accepted_reservation = _reserve_capacity(required_accepted, accepted_capacity)
    attempt_reservation = _reserve_capacity(required_attempts, attempt_capacity)
    return {
        label: {
            "required_accepted": required_accepted[label],
            "required_attempts": required_attempts[label],
            "available_accepted_capacity": accepted_reservation[label],
            "available_attempt_capacity": attempt_reservation[label],
            "attempt_capacity_ratio": (attempt_reservation[label] / required_attempts[label]),
            "limiting_modes": limiting_modes,
            "maximum_plan_concentration": maximum_plan_concentration,
            "capacity_reserved_from_shared_family_pool": True,
            "gate_passed": (
                accepted_reservation[label] >= required_accepted[label]
                and attempt_reservation[label] >= required_attempts[label]
            ),
        }
        for label in required_attempts
    }


def build_signal_capacity_audit(
    config_path: Path,
    *,
    previous_capacity_path: Path,
) -> dict[str, object]:
    """Prove the reduced quotas fit every bounded identity and finite program mode."""

    config = load_signal_pilot_config(config_path)
    reuse_path = config.reuse_config_path
    if not reuse_path.is_absolute():
        reuse_path = Path.cwd() / reuse_path
    reuse = load_contract(reuse_path)
    fixtures_path = reuse.fixture_path
    if not fixtures_path.is_absolute():
        fixtures_path = Path.cwd() / fixtures_path
    calibration = calibrate(reuse, load_fixtures(fixtures_path))
    if (
        calibration["selected_policy_id"] != config.reuse_policy_id
        or calibration["selected_policy_sha256"] != config.reuse_policy_sha256
    ):
        raise ValueError("bounded reuse policy differs from Milestone 6E")
    previous: object = json.loads(previous_capacity_path.read_text(encoding="utf-8"))
    previous_map = _mapping(previous, "previous capacity audit")
    if previous_map.get("capacity_audit_sha256") != (
        "1a40db7b40005d12f631b534e07f28d4f9974d6516b900975116526575f21129"
    ):
        raise ValueError("Milestone 6E capacity evidence changed")
    inventory = reuse.identity_inventory
    dataset_results: dict[str, object] = {}
    combined_results: dict[str, object] = {}
    all_passed = True
    for category in CATEGORY_ORDER:
        group_caps = {
            group: _derive_group_caps(config, inventory, group, category) for group in GROUP_ORDER
        }
        group_attempt_modes, combined_attempt_modes = _mode_capacities(
            category=category, caps_by_group=group_caps, cap_kind="attempt"
        )
        group_accepted_modes, combined_accepted_modes = _mode_capacities(
            category=category, caps_by_group=group_caps, cap_kind="accepted"
        )
        combined_attempt_required = sum(
            config.datasets[group].families[category].attempts for group in GROUP_ORDER
        )
        combined_accepted_required = sum(
            config.datasets[group].families[category].accepted for group in GROUP_ORDER
        )
        combined_surface_attempt = sum(
            cast(int, group_caps[group]["surface_attempt_capacity"]) for group in GROUP_ORDER
        )
        combined_surface_accepted = sum(
            cast(int, group_caps[group]["surface_accepted_capacity"]) for group in GROUP_ORDER
        )
        combined_compatible_attempt, combined_attempt_compatibility = _compatible_capacity(
            category=category,
            groups=GROUP_ORDER,
            caps_by_group=group_caps,
            group_mode_capacity=group_attempt_modes,
            cap_kind="attempt",
        )
        combined_compatible_accepted, combined_accepted_compatibility = _compatible_capacity(
            category=category,
            groups=GROUP_ORDER,
            caps_by_group=group_caps,
            group_mode_capacity=group_accepted_modes,
            cap_kind="accepted",
        )
        combined_latent_attempt = min(combined_surface_attempt, combined_compatible_attempt)
        combined_latent_accepted = min(combined_surface_accepted, combined_compatible_accepted)
        limiting_modes = tuple(
            mode
            for mode in MODE_ORDER[category]
            if _EXACT_MODE_UPPER_BOUNDS[category][mode] is not None
        )
        mode_gate = True
        for mode in MODE_ORDER[category]:
            total_attempt_mode = sum(
                config.datasets[group].families[category].attempt_modes[mode]
                for group in GROUP_ORDER
            )
            total_accepted_mode = sum(
                config.datasets[group].families[category].accepted_modes[mode]
                for group in GROUP_ORDER
            )
            mode_gate = mode_gate and total_attempt_mode <= combined_attempt_modes[mode]
            mode_gate = mode_gate and total_accepted_mode <= combined_accepted_modes[mode]
            for group in GROUP_ORDER:
                quota = config.datasets[group].families[category]
                mode_gate = mode_gate and (
                    quota.attempt_modes[mode] <= group_attempt_modes[group][mode]
                    and quota.accepted_modes[mode] <= group_accepted_modes[group][mode]
                )
        category_gate = (
            combined_latent_attempt >= combined_attempt_required
            and combined_latent_accepted >= combined_accepted_required
            and mode_gate
        )
        all_passed = all_passed and category_gate
        combined_results[category] = {
            "required_accepted": combined_accepted_required,
            "required_attempts": combined_attempt_required,
            "available_surface_accepted_capacity": combined_surface_accepted,
            "available_surface_attempt_capacity": combined_surface_attempt,
            "available_latent_accepted_capacity": combined_latent_accepted,
            "available_latent_attempt_capacity": combined_latent_attempt,
            "attempt_capacity_ratio": combined_latent_attempt / combined_attempt_required,
            "mode_attempt_capacity": combined_attempt_modes,
            "mode_accepted_capacity": combined_accepted_modes,
            "mode_attempt_allocation": {
                mode: sum(
                    config.datasets[group].families[category].attempt_modes[mode]
                    for group in GROUP_ORDER
                )
                for mode in MODE_ORDER[category]
            },
            "mode_accepted_allocation": {
                mode: sum(
                    config.datasets[group].families[category].accepted_modes[mode]
                    for group in GROUP_ORDER
                )
                for mode in MODE_ORDER[category]
            },
            "attempt_compatibility": combined_attempt_compatibility,
            "accepted_compatibility": combined_accepted_compatibility,
            "limiting_modes": limiting_modes,
            "gate_passed": category_gate,
        }
        for group in GROUP_ORDER:
            dataset = cast(dict[str, object], dataset_results.setdefault(group, {}))
            quota = config.datasets[group].families[category]
            caps = group_caps[group]
            group_compatible_attempt, group_attempt_compatibility = _compatible_capacity(
                category=category,
                groups=(group,),
                caps_by_group=group_caps,
                group_mode_capacity=group_attempt_modes,
                cap_kind="attempt",
            )
            group_compatible_accepted, group_accepted_compatibility = _compatible_capacity(
                category=category,
                groups=(group,),
                caps_by_group=group_caps,
                group_mode_capacity=group_accepted_modes,
                cap_kind="accepted",
            )
            group_mode_attempt_capacity = min(
                cast(int, caps["surface_attempt_capacity"]),
                group_compatible_attempt,
            )
            group_mode_accepted_capacity = min(
                cast(int, caps["surface_accepted_capacity"]),
                group_compatible_accepted,
            )
            difficulty = _stratum_records(
                required_accepted=balanced_counts(quota.accepted, DIFFICULTY_ORDER),
                required_attempts=balanced_counts(quota.attempts, DIFFICULTY_ORDER),
                accepted_capacity=group_mode_accepted_capacity,
                attempt_capacity=group_mode_attempt_capacity,
                limiting_modes=limiting_modes,
                maximum_plan_concentration=cast(float, caps["maximum_plan_attempt_concentration"]),
            )
            output = _stratum_records(
                required_accepted={
                    "enabled": quota.output_contract_accepted,
                    "disabled": quota.accepted - quota.output_contract_accepted,
                },
                required_attempts={
                    "enabled": quota.output_contract_attempts,
                    "disabled": quota.attempts - quota.output_contract_attempts,
                },
                accepted_capacity=group_mode_accepted_capacity,
                attempt_capacity=group_mode_attempt_capacity,
                limiting_modes=limiting_modes,
                maximum_plan_concentration=cast(float, caps["maximum_plan_attempt_concentration"]),
            )
            splits = _stratum_records(
                required_accepted={
                    "training": quota.training_accepted,
                    "synthetic_validation": quota.validation_accepted,
                },
                required_attempts={
                    "training": quota.training_attempts,
                    "synthetic_validation": quota.validation_attempts,
                },
                accepted_capacity=group_mode_accepted_capacity,
                attempt_capacity=group_mode_attempt_capacity,
                limiting_modes=limiting_modes,
                maximum_plan_concentration=cast(float, caps["maximum_plan_attempt_concentration"]),
            )
            group_gate = (
                quota.attempts <= group_mode_attempt_capacity
                and quota.accepted <= group_mode_accepted_capacity
                and all(item["gate_passed"] for item in difficulty.values())
                and all(item["gate_passed"] for item in output.values())
                and all(item["gate_passed"] for item in splits.values())
            )
            all_passed = all_passed and group_gate
            dataset[category] = {
                "required_accepted": quota.accepted,
                "required_attempts": quota.attempts,
                "available_surface_accepted_capacity": caps["surface_accepted_capacity"],
                "available_surface_attempt_capacity": caps["surface_attempt_capacity"],
                "available_latent_accepted_capacity": group_mode_accepted_capacity,
                "available_latent_attempt_capacity": group_mode_attempt_capacity,
                "attempt_capacity_ratio": group_mode_attempt_capacity / quota.attempts,
                "caps": caps,
                "mode_attempt_capacity": group_attempt_modes[group],
                "mode_accepted_capacity": group_accepted_modes[group],
                "mode_attempt_allocation": quota.attempt_modes,
                "mode_accepted_allocation": quota.accepted_modes,
                "attempt_compatibility": group_attempt_compatibility,
                "accepted_compatibility": group_accepted_compatibility,
                "difficulty_strata": difficulty,
                "output_contract_strata": output,
                "split_strata": splits,
                "limiting_modes": limiting_modes,
                "gate_passed": group_gate,
            }
    payload: dict[str, object] = {
        "schema_version": 1,
        "audit_id": config.capacity_audit_id,
        "pilot_id": config.pilot_id,
        "config_sha256": config.config_sha256,
        "reuse_policy_id": config.reuse_policy_id,
        "reuse_policy_sha256": config.reuse_policy_sha256,
        "previous_capacity_audit_sha256": previous_map["capacity_audit_sha256"],
        "template_bank_sha256": canonical_sha256([asdict(item) for item in build_template_bank()]),
        "planned_accepted_total": 2000,
        "required_attempt_total": 2504,
        "training_accepted_total": 1800,
        "synthetic_validation_accepted_total": 200,
        "output_contract_accepted_total": 400,
        "datasets": dataset_results,
        "combined_family_capacity": combined_results,
        "cross_dataset_exact_overlap_allowed": False,
        "cross_dataset_latent_overlap_allowed": False,
        "cross_split_exact_overlap_allowed": False,
        "cross_split_latent_overlap_allowed": False,
        "capacity_gate_passed": all_passed,
        "full_2504_schedule_feasible": all_passed,
        "limiting_cross_dataset_families": [
            {
                "category": category,
                "required_attempts": cast(dict[str, object], record)["required_attempts"],
                "available_compatible_attempts": cast(dict[str, object], record)[
                    "available_latent_attempt_capacity"
                ],
                "shortfall": cast(int, cast(dict[str, object], record)["required_attempts"])
                - cast(
                    int,
                    cast(dict[str, object], record)["available_latent_attempt_capacity"],
                ),
            }
            for category, record in combined_results.items()
            if not cast(bool, cast(dict[str, object], record)["gate_passed"])
        ],
        "allocator_implemented": False,
        "full_schedule_created": False,
        "fresh_smoke_run": False,
        "deterministic_replay_run": False,
        "review_packet_created": False,
        "old_10003_schedule_remains_unapproved": True,
        "generators_changed": False,
        "verifiers_changed": False,
        "benchmark_contamination_changed": False,
        "sealed_final_accessed": False,
    }
    payload["capacity_audit_sha256"] = canonical_sha256(payload)
    return payload


def write_signal_capacity_audit(
    config_path: Path,
    output_path: Path,
    *,
    previous_capacity_path: Path,
) -> dict[str, object]:
    """Write the content-free reduced-pilot capacity decision."""

    payload = build_signal_capacity_audit(
        config_path, previous_capacity_path=previous_capacity_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
