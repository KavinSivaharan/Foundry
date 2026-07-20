"""Submode-local bounded reuse for canonical runtime surface identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.template_bank.signal_pilot import (
    CATEGORY_ORDER,
    GROUP_ORDER,
    MODE_ORDER,
    SignalPilotConfig,
    load_signal_pilot_config,
)

SELECTED_POLICY_ID = "submode-local-balanced-surface-reuse-v1"
POLICY_ORDER = (
    "family-level-bounded-surface-reuse-v1",
    SELECTED_POLICY_ID,
    "permissive-exact-latent-only-v1",
)


def canonical_sha256(value: object) -> str:
    """Return a stable SHA-256 for JSON-like content."""

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
class SurfaceReusePolicyConfig:
    """Frozen runtime-identity inventory and policy contract."""

    config_sha256: str
    policy_id: str
    fixture_path: Path
    signal_pilot_config: Path
    subordinate_allocation_evidence: Path
    candidate_policies: tuple[str, ...]
    multiplier_numerator: int
    multiplier_denominator: int
    active_runtime_identities: dict[str, dict[str, int]]
    normalizer_version: str
    normalizer_source_sha256: str
    identity_contract_sha256: str
    weighted_easy_medium_identities: tuple[str, ...]
    weighted_hard_identities: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceReuseFixture:
    """Original content-free cap calibration fixture."""

    fixture_id: str
    relationship: str
    same_exact_question: bool
    same_latent_program: bool
    dataset_use_after: int
    global_use_after: int
    family_level_cap: int
    submode_local_cap: int
    global_submode_cap: int
    expected_allowed: bool


@dataclass(frozen=True)
class SubmodeSurfaceCaps:
    """Attempt and accepted caps for one dataset/family/submode."""

    active_identities: int
    required_attempts: int
    accepted_quota: int
    max_attempts_per_identity: int
    max_accepted_per_identity: int


def load_surface_reuse_config(path: Path) -> SurfaceReusePolicyConfig:
    """Load and strictly validate the approved submode-local policy."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "surface reuse policy")
    if root.get("schema_version") != 1 or root.get("policy_id") != SELECTED_POLICY_ID:
        raise ValueError("surface reuse policy identity differs")
    candidates = root.get("candidate_policies")
    if not isinstance(candidates, list) or tuple(candidates) != POLICY_ORDER:
        raise ValueError("exactly the approved three surface policies must be compared")
    formula = _mapping(root.get("cap_formula"), "cap_formula")
    if formula != {
        "multiplier_numerator": 5,
        "multiplier_denominator": 4,
        "dataset_key": "dataset/family/submode/runtime-number-neutral-identity",
        "global_key": "family/submode/runtime-number-neutral-identity",
        "global_cap": "sum-of-dataset-caps",
    }:
        raise ValueError("surface reuse cap formula differs from approval")
    normalizer = _mapping(root.get("normalizer"), "normalizer")
    expected_normalizer = {
        "version": "foundry-number-neutral-v1",
        "source_sha256": "57d77d32b4631aaecacb02d454f39000f7e13fec457b398ff983817df05c3d1f",
        "identity_contract_sha256": (
            "e57f12e06673c21bb4c257c651d96b791a8b991260a3e5fd7c5fd11bdc3a72eb"
        ),
        "semantics_unchanged": True,
    }
    if normalizer != expected_normalizer:
        raise ValueError("canonical runtime normalizer contract changed")
    identities_raw = _mapping(root.get("active_runtime_identities"), "identities")
    if tuple(identities_raw) != CATEGORY_ORDER:
        raise ValueError("runtime identity family order differs")
    identities: dict[str, dict[str, int]] = {}
    for family in CATEGORY_ORDER:
        current = _mapping(identities_raw[family], f"identities.{family}")
        if tuple(current) != MODE_ORDER[family]:
            raise ValueError(f"runtime identity submode order differs for {family}")
        identities[family] = {
            mode: _positive_integer(current[mode], f"identities.{family}.{mode}")
            for mode in MODE_ORDER[family]
        }
    expected_identities = {
        CATEGORY_ORDER[0]: {"inventory": 384, "grouping": 384},
        CATEGORY_ORDER[1]: {
            "rate_total": 20,
            "ratio_scale": 20,
            "percentage": 20,
            "weighted_average": 8,
            "combined_rate": 20,
        },
        CATEGORY_ORDER[2]: {
            "two_type_allocation": 80,
            "complete_packages": 80,
            "equal_distribution": 80,
            "dual_capacity": 80,
        },
    }
    if identities != expected_identities:
        raise ValueError("measured runtime identity inventory differs")
    compatibility = _mapping(
        root.get("difficulty_compatibility_evidence"), "difficulty compatibility"
    )
    rate_compatibility = _mapping(
        compatibility.get(CATEGORY_ORDER[1]), "rate difficulty compatibility"
    )
    weighted = _mapping(rate_compatibility.get("weighted_average"), "weighted compatibility")
    easy_medium = _mapping(weighted.get("easy_medium"), "weighted easy/medium")
    hard = _mapping(weighted.get("hard"), "weighted hard")
    if easy_medium.get("difficulties") != ["easy", "medium"] or hard.get("difficulties") != [
        "hard"
    ]:
        raise ValueError("weighted-average difficulty identity groups differ")
    easy_medium_raw = easy_medium.get("runtime_identity_sha256_values")
    hard_raw = hard.get("runtime_identity_sha256_values")
    if (
        not isinstance(easy_medium_raw, list)
        or not isinstance(hard_raw, list)
        or not all(isinstance(item, str) and len(item) == 64 for item in easy_medium_raw)
        or not all(isinstance(item, str) and len(item) == 64 for item in hard_raw)
    ):
        raise ValueError("weighted-average difficulty identity hashes are malformed")
    easy_medium_identities = tuple(cast(list[str], easy_medium_raw))
    hard_identities = tuple(cast(list[str], hard_raw))
    if (
        len(easy_medium_identities) != 4
        or len(hard_identities) != 4
        or set(easy_medium_identities) & set(hard_identities)
        or len(set(easy_medium_identities) | set(hard_identities)) != 8
    ):
        raise ValueError("weighted-average difficulty identity partition differs")
    preserved = _mapping(root.get("preserved_controls"), "preserved_controls")
    if preserved.get("sealed_final_access") is not False:
        raise ValueError("sealed-final boundary differs")
    if any(
        value not in {True, "unchanged"}
        for key, value in preserved.items()
        if key != "sealed_final_access"
    ):
        raise ValueError("a preserved reuse or contamination control changed")
    return SurfaceReusePolicyConfig(
        config_sha256=canonical_sha256(root),
        policy_id=SELECTED_POLICY_ID,
        fixture_path=Path(_string(root.get("fixture_path"), "fixture_path")),
        signal_pilot_config=Path(_string(root.get("signal_pilot_config"), "signal_pilot_config")),
        subordinate_allocation_evidence=Path(
            _string(
                root.get("subordinate_allocation_evidence"),
                "subordinate_allocation_evidence",
            )
        ),
        candidate_policies=POLICY_ORDER,
        multiplier_numerator=5,
        multiplier_denominator=4,
        active_runtime_identities=identities,
        normalizer_version=cast(str, normalizer["version"]),
        normalizer_source_sha256=cast(str, normalizer["source_sha256"]),
        identity_contract_sha256=cast(str, normalizer["identity_contract_sha256"]),
        weighted_easy_medium_identities=easy_medium_identities,
        weighted_hard_identities=hard_identities,
    )


def load_surface_reuse_fixtures(path: Path) -> tuple[SurfaceReuseFixture, ...]:
    """Load original content-free policy fixtures."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("surface reuse fixtures must be a nonempty list")
    expected = set(SurfaceReuseFixture.__dataclass_fields__)
    fixtures: list[SurfaceReuseFixture] = []
    for index, value in enumerate(raw):
        item = _mapping(value, f"fixtures[{index}]")
        if set(item) != expected:
            raise ValueError(f"fixtures[{index}] fields differ")
        fixtures.append(
            SurfaceReuseFixture(
                fixture_id=_string(item["fixture_id"], "fixture_id"),
                relationship=_string(item["relationship"], "relationship"),
                same_exact_question=bool(item["same_exact_question"]),
                same_latent_program=bool(item["same_latent_program"]),
                dataset_use_after=_positive_integer(item["dataset_use_after"], "dataset use"),
                global_use_after=_positive_integer(item["global_use_after"], "global use"),
                family_level_cap=_positive_integer(item["family_level_cap"], "family cap"),
                submode_local_cap=_positive_integer(item["submode_local_cap"], "local cap"),
                global_submode_cap=_positive_integer(item["global_submode_cap"], "global cap"),
                expected_allowed=bool(item["expected_allowed"]),
            )
        )
    if len({fixture.fixture_id for fixture in fixtures}) != len(fixtures):
        raise ValueError("surface reuse fixture IDs must be unique")
    return tuple(fixtures)


def fixture_allowed(policy_id: str, fixture: SurfaceReuseFixture) -> bool:
    """Classify a fixture without consulting generated or benchmark outputs."""

    if fixture.same_exact_question or fixture.same_latent_program:
        return False
    if policy_id == POLICY_ORDER[0]:
        return fixture.dataset_use_after <= fixture.family_level_cap
    if policy_id == SELECTED_POLICY_ID:
        return (
            fixture.dataset_use_after <= fixture.submode_local_cap
            and fixture.global_use_after <= fixture.global_submode_cap
        )
    if policy_id == POLICY_ORDER[2]:
        return True
    raise ValueError("unknown surface reuse policy")


def calibrate_surface_reuse(
    config: SurfaceReusePolicyConfig, fixtures: tuple[SurfaceReuseFixture, ...]
) -> dict[str, object]:
    """Compare exactly three policies and freeze the selected result."""

    results: dict[str, object] = {}
    for policy_id in config.candidate_policies:
        decisions = [fixture_allowed(policy_id, fixture) for fixture in fixtures]
        results[policy_id] = {
            "exact_matches": sum(
                actual is fixture.expected_allowed
                for actual, fixture in zip(decisions, fixtures, strict=True)
            ),
            "mismatched_fixture_ids": [
                fixture.fixture_id
                for actual, fixture in zip(decisions, fixtures, strict=True)
                if actual is not fixture.expected_allowed
            ],
        }
    selected = cast(dict[str, object], results[config.policy_id])
    if selected["exact_matches"] != len(fixtures):
        raise ValueError("selected submode-local policy failed fixture calibration")
    contract = {
        "policy_id": config.policy_id,
        "cap_granularity": "dataset/family/submode/runtime-number-neutral-identity",
        "attempt_formula": "ceil(1.25*submode_attempts/active_runtime_identities)",
        "accepted_formula": "ceil(1.25*submode_acceptances/active_runtime_identities)",
        "global_cap": "targeted_cap+generic_control_cap",
        "exact_question_duplicates": "reject",
        "latent_program_duplicates": "reject",
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "policy_id": config.policy_id,
        "config_sha256": config.config_sha256,
        "fixture_set_sha256": canonical_sha256([asdict(item) for item in fixtures]),
        "fixture_count": len(fixtures),
        "candidate_results": results,
        "selected_policy_contract": contract,
        "selected_policy_sha256": canonical_sha256(contract),
        "selection_frozen_before_real_capacity_or_smoke": True,
        "rejected_alternatives": {
            POLICY_ORDER[0]: "underallocates scarce submodes by averaging over a family",
            POLICY_ORDER[2]: "places no deterministic concentration bound on surface reuse",
        },
        "benchmark_contamination_changed": False,
        "sealed_final_accessed": False,
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    return payload


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _cap(config: SurfaceReusePolicyConfig, quantity: int, identities: int) -> int:
    return _ceil_div(
        config.multiplier_numerator * quantity,
        config.multiplier_denominator * identities,
    )


def derive_surface_caps(
    signal: SignalPilotConfig, policy: SurfaceReusePolicyConfig
) -> dict[str, dict[str, dict[str, SubmodeSurfaceCaps]]]:
    """Derive every dataset/family/submode cap mechanically."""

    result: dict[str, dict[str, dict[str, SubmodeSurfaceCaps]]] = {}
    for group in GROUP_ORDER:
        result[group] = {}
        for family in CATEGORY_ORDER:
            quota = signal.datasets[group].families[family]
            result[group][family] = {}
            for mode in MODE_ORDER[family]:
                identities = policy.active_runtime_identities[family][mode]
                result[group][family][mode] = SubmodeSurfaceCaps(
                    active_identities=identities,
                    required_attempts=quota.attempt_modes[mode],
                    accepted_quota=quota.accepted_modes[mode],
                    max_attempts_per_identity=_cap(policy, quota.attempt_modes[mode], identities),
                    max_accepted_per_identity=_cap(policy, quota.accepted_modes[mode], identities),
                )
    return result


def build_surface_capacity_audit(config_path: Path) -> dict[str, object]:
    """Audit all eleven submodes under the selected frozen cap policy."""

    policy = load_surface_reuse_config(config_path)
    fixtures = load_surface_reuse_fixtures(policy.fixture_path)
    calibration = calibrate_surface_reuse(policy, fixtures)
    signal = load_signal_pilot_config(policy.signal_pilot_config)
    caps = derive_surface_caps(signal, policy)
    datasets: dict[str, object] = {}
    gate = True
    for group in GROUP_ORDER:
        families: dict[str, object] = {}
        for family in CATEGORY_ORDER:
            modes: dict[str, object] = {}
            for mode in MODE_ORDER[family]:
                item = caps[group][family][mode]
                attempt_capacity = item.active_identities * item.max_attempts_per_identity
                accepted_capacity = item.active_identities * item.max_accepted_per_identity
                mode_gate = (
                    attempt_capacity >= item.required_attempts
                    and accepted_capacity >= item.accepted_quota
                )
                gate = gate and mode_gate
                modes[mode] = {
                    **asdict(item),
                    "attempt_capacity": attempt_capacity,
                    "accepted_capacity": accepted_capacity,
                    "attempt_headroom": attempt_capacity - item.required_attempts,
                    "accepted_headroom": accepted_capacity - item.accepted_quota,
                    "maximum_attempt_concentration": (
                        item.max_attempts_per_identity / item.required_attempts
                    ),
                    "gate_passed": mode_gate,
                }
            families[family] = {
                "submodes": modes,
                "gate_passed": all(
                    cast(dict[str, object], modes[mode])["gate_passed"]
                    for mode in MODE_ORDER[family]
                ),
            }
        datasets[group] = families
    global_modes: dict[str, object] = {}
    for family in CATEGORY_ORDER:
        current: dict[str, object] = {}
        for mode in MODE_ORDER[family]:
            targeted = caps[GROUP_ORDER[0]][family][mode]
            generic = caps[GROUP_ORDER[1]][family][mode]
            global_cap = targeted.max_attempts_per_identity + generic.max_attempts_per_identity
            required = targeted.required_attempts + generic.required_attempts
            capacity = targeted.active_identities * global_cap
            current[mode] = {
                "required_attempts": required,
                "active_runtime_identities": targeted.active_identities,
                "global_max_attempts_per_identity": global_cap,
                "global_attempt_capacity": capacity,
                "headroom": capacity - required,
                "gate_passed": capacity >= required,
            }
            gate = gate and capacity >= required
        global_modes[family] = current
    subordinate: object = json.loads(
        policy.subordinate_allocation_evidence.read_text(encoding="utf-8")
    )
    subordinate_map = _mapping(subordinate, "subordinate allocation evidence")
    if subordinate_map.get("capacity_audit_sha256") != (
        "7c1d87d7ed43b71343a1af68c531f3d747f9d46c0ef3d6268b380213e830aa0c"
    ):
        raise ValueError("frozen subordinate allocation evidence differs")
    weighted_family = CATEGORY_ORDER[1]
    difficulty_proof: dict[str, object] = {}
    for group in GROUP_ORDER:
        group_evidence = _mapping(_mapping(subordinate_map["datasets"], "datasets")[group], group)
        family_evidence = _mapping(group_evidence[weighted_family], weighted_family)
        allocations = _mapping(
            family_evidence["attempt_subordinate_allocations"], "subordinate allocations"
        )
        difficulty = _mapping(allocations["difficulty"], "difficulty allocation")
        weighted_difficulty = _mapping(difficulty["weighted_average"], "weighted difficulty")
        required_easy_medium = cast(int, weighted_difficulty["easy"]) + cast(
            int, weighted_difficulty["medium"]
        )
        local_cap = caps[group][weighted_family]["weighted_average"].max_attempts_per_identity
        compatible_capacity = len(policy.weighted_easy_medium_identities) * local_cap
        proof_gate = compatible_capacity >= required_easy_medium
        gate = gate and proof_gate
        difficulty_proof[group] = {
            "difficulty_group": ["easy", "medium"],
            "required_attempts": required_easy_medium,
            "compatible_runtime_identities": len(policy.weighted_easy_medium_identities),
            "max_attempts_per_identity": local_cap,
            "compatible_capacity": compatible_capacity,
            "shortfall": max(0, required_easy_medium - compatible_capacity),
            "gate_passed": proof_gate,
        }
    combined_required = sum(
        cast(int, _mapping(difficulty_proof[group], group)["required_attempts"])
        for group in GROUP_ORDER
    )
    combined_cap = sum(
        caps[group][weighted_family]["weighted_average"].max_attempts_per_identity
        for group in GROUP_ORDER
    )
    combined_capacity = len(policy.weighted_easy_medium_identities) * combined_cap
    combined_gate = combined_capacity >= combined_required
    gate = gate and combined_gate
    difficulty_proof["combined"] = {
        "difficulty_group": ["easy", "medium"],
        "required_attempts": combined_required,
        "compatible_runtime_identities": len(policy.weighted_easy_medium_identities),
        "global_max_attempts_per_identity": combined_cap,
        "compatible_capacity": combined_capacity,
        "shortfall": max(0, combined_required - combined_capacity),
        "gate_passed": combined_gate,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "audit_id": "foundry-signal-pilot-submode-surface-capacity-v1",
        "policy_id": policy.policy_id,
        "policy_sha256": calibration["selected_policy_sha256"],
        "policy_config_sha256": policy.config_sha256,
        "fixture_set_sha256": calibration["fixture_set_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "signal_config_sha256": signal.config_sha256,
        "required_attempt_total": 2504,
        "required_accepted_total": 2000,
        "datasets": datasets,
        "global_submodes": global_modes,
        "difficulty_compatibility_proof": {weighted_family: {"weighted_average": difficulty_proof}},
        "runtime_normalizer_version": policy.normalizer_version,
        "runtime_normalizer_source_sha256": policy.normalizer_source_sha256,
        "identity_contract_sha256": policy.identity_contract_sha256,
        "all_eleven_submodes_audited": sum(len(value) for value in MODE_ORDER.values()) == 11,
        "capacity_gate_passed": gate,
        "full_2504_schedule_authorized": gate,
        "first_limiting_stratum": {
            "family": weighted_family,
            "submode": "weighted_average",
            "difficulty_group": ["easy", "medium"],
            "combined_required_attempts": combined_required,
            "combined_compatible_capacity": combined_capacity,
            "shortfall": max(0, combined_required - combined_capacity),
        }
        if not gate
        else None,
        "exact_question_uniqueness_changed": False,
        "latent_program_uniqueness_changed": False,
        "benchmark_contamination_changed": False,
        "sealed_final_accessed": False,
    }
    payload["capacity_audit_sha256"] = canonical_sha256(payload)
    return payload


def write_surface_policy_evidence(
    config_path: Path, calibration_path: Path, audit_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Write content-free calibration and capacity evidence."""

    config = load_surface_reuse_config(config_path)
    calibration = calibrate_surface_reuse(config, load_surface_reuse_fixtures(config.fixture_path))
    audit = build_surface_capacity_audit(config_path)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return calibration, audit
