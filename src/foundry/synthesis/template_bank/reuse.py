"""Bounded generated-to-generated template reuse and finite-capacity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml

from foundry.synthesis.contamination import normalized_text_sha256
from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.generators.bookkeeping import generate_bookkeeping
from foundry.synthesis.generators.discrete import generate_discrete
from foundry.synthesis.generators.rates import generate_rates
from foundry.synthesis.pipeline import _verify
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.taxonomy import FailureCategory

REUSE_AUDIT_ID = "foundry-template-bank-bounded-reuse-v1"

_BOOKKEEPING = str(FailureCategory.MULTI_STEP_BOOKKEEPING)
_RATES = str(FailureCategory.RATE_RATIO_PERCENTAGE)
_DISCRETE = str(FailureCategory.CONSTRAINT_DISCRETE)
_CATEGORY_ORDER = (_BOOKKEEPING, _RATES, _DISCRETE)

_GENERATORS: dict[str, Callable[..., CandidateDraft]] = {
    _BOOKKEEPING: generate_bookkeeping,
    _RATES: generate_rates,
    _DISCRETE: generate_discrete,
}
_VARIANT_PERIODS = {_BOOKKEEPING: 240, _RATES: 40, _DISCRETE: 80}


class ReuseOutcome(StrEnum):
    """Deterministic result of generated-to-generated identity comparison."""

    PASS = "pass"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class ReusePolicy:
    """One predeclared candidate policy compared before new smoke outputs."""

    policy_id: str
    reject_number_neutral_reuse: bool
    enforce_usage_caps: bool
    close_paraphrase_action: ReuseOutcome

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ReuseFixture:
    """Original pairwise fixture with explicit identity and cap evidence."""

    fixture_id: str
    relationship: str
    left: str
    right: str
    expected_outcome: ReuseOutcome
    same_latent_program: bool
    same_complete_candidate: bool
    same_number_neutral_signature: bool
    same_sentence_plan: bool
    same_structural_problem: bool
    cross_dataset: bool
    cross_split: bool
    close_paraphrase: bool
    plan_use_after: int
    plan_cap: int
    number_neutral_use_after: int
    number_neutral_cap: int


@dataclass(frozen=True)
class ReuseDecision:
    """Content-free fixture decision."""

    outcome: ReuseOutcome
    reason: str


@dataclass(frozen=True)
class Contract:
    """Frozen candidate policies, quotas, identity inventory, and probe settings."""

    config_sha256: str
    fixture_path: Path
    selected_policy_id: str
    policies: tuple[ReusePolicy, ...]
    benchmark_contamination: dict[str, object]
    accepted_quotas: dict[str, dict[str, int]]
    attempt_multiplier_numerator: int
    attempt_multiplier_denominator: int
    identity_inventory: dict[str, dict[str, int]]
    latent_probe_id: str
    latent_probe_candidates: int


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be a boolean")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def load_contract(path: Path) -> Contract:
    """Load the bounded-reuse design and enforce the benchmark firewall."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "bounded reuse contract")
    if root.get("schema_version") != 1:
        raise ValueError("bounded reuse schema version differs")
    benchmark = _mapping(root.get("benchmark_contamination"), "benchmark_contamination")
    expected_benchmark: dict[str, object] = {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "manual_review_at": 0.75,
        "automatic_reject_at": 0.82,
        "trust_remote_code": False,
        "unchanged": True,
    }
    if benchmark != expected_benchmark:
        raise ValueError("generated-to-development contamination controls changed")
    policies_raw = root.get("candidate_policies")
    if not isinstance(policies_raw, list) or len(policies_raw) != 3:
        raise ValueError("exactly three bounded-reuse policies must be compared")
    policies: list[ReusePolicy] = []
    for index, value in enumerate(policies_raw):
        item = _mapping(value, f"candidate_policies[{index}]")
        if set(item) != {
            "policy_id",
            "reject_number_neutral_reuse",
            "enforce_usage_caps",
            "close_paraphrase_action",
        }:
            raise ValueError("candidate policy fields differ from the frozen schema")
        policies.append(
            ReusePolicy(
                policy_id=_string(item["policy_id"], "policy_id"),
                reject_number_neutral_reuse=_boolean(
                    item["reject_number_neutral_reuse"], "reject_number_neutral_reuse"
                ),
                enforce_usage_caps=_boolean(item["enforce_usage_caps"], "enforce_usage_caps"),
                close_paraphrase_action=ReuseOutcome(
                    _string(item["close_paraphrase_action"], "close_paraphrase_action")
                ),
            )
        )
    selected = _string(root.get("selected_policy_id"), "selected_policy_id")
    if selected not in {policy.policy_id for policy in policies}:
        raise ValueError("selected policy is not one of the three candidates")
    quota = _mapping(root.get("quota_contract"), "quota_contract")
    numerator = _integer(quota.get("attempt_multiplier_numerator"), "attempt numerator")
    denominator = _integer(quota.get("attempt_multiplier_denominator"), "attempt denominator")
    if (numerator, denominator) != (5, 4):
        raise ValueError("attempt multiplier differs from the frozen 125 percent")
    accepted: dict[str, dict[str, int]] = {}
    for group in ("targeted", "generic_control"):
        values = _mapping(quota.get(group), group)
        if set(values) != set(_CATEGORY_ORDER):
            raise ValueError(f"{group} quota categories differ from the frozen contract")
        accepted[group] = {
            category: _integer(values[category], f"{group}.{category}")
            for category in _CATEGORY_ORDER
        }
    if (
        sum(accepted["targeted"].values()) != 4000
        or sum(accepted["generic_control"].values()) != 4000
    ):
        raise ValueError("accepted quotas must total 4,000 per dataset")
    inventory_raw = _mapping(root.get("identity_inventory"), "identity_inventory")
    inventory: dict[str, dict[str, int]] = {}
    expected_inventory_fields = {
        "sentence_plans",
        "number_neutral_signatures",
        "plan_scenario_domains",
        "semantic_frames",
        "target_types",
    }
    for category in _CATEGORY_ORDER:
        values = _mapping(inventory_raw.get(category), f"identity_inventory.{category}")
        if set(values) != expected_inventory_fields:
            raise ValueError("identity inventory fields differ")
        inventory[category] = {
            key: _integer(values[key], f"identity_inventory.{category}.{key}")
            for key in expected_inventory_fields
        }
    expected_inventory = {
        _BOOKKEEPING: {
            "sentence_plans": 72,
            "number_neutral_signatures": 768,
            "plan_scenario_domains": 1728,
            "semantic_frames": 18,
            "target_types": 2,
        },
        _RATES: {
            "sentence_plans": 80,
            "number_neutral_signatures": 88,
            "plan_scenario_domains": 400,
            "semantic_frames": 20,
            "target_types": 4,
        },
        _DISCRETE: {
            "sentence_plans": 80,
            "number_neutral_signatures": 320,
            "plan_scenario_domains": 1600,
            "semantic_frames": 20,
            "target_types": 3,
        },
    }
    if inventory != expected_inventory:
        raise ValueError("identity inventory differs from Milestone 6D")
    probe = _mapping(root.get("latent_probe"), "latent_probe")
    if (
        probe.get("persisted_programs") is not False
        or probe.get("benchmark_content_allowed") is not False
    ):
        raise ValueError("latent probe violates the no-data boundary")
    canonical = json.loads(json.dumps(root, sort_keys=True))
    return Contract(
        config_sha256=_canonical_sha256(canonical),
        fixture_path=Path(_string(root.get("fixture_path"), "fixture_path")),
        selected_policy_id=selected,
        policies=tuple(policies),
        benchmark_contamination=benchmark,
        accepted_quotas=accepted,
        attempt_multiplier_numerator=numerator,
        attempt_multiplier_denominator=denominator,
        identity_inventory=inventory,
        latent_probe_id=_string(probe.get("probe_id"), "latent_probe.probe_id"),
        latent_probe_candidates=_integer(
            probe.get("candidates_per_family"), "latent_probe.candidates_per_family"
        ),
    )


def load_fixtures(path: Path) -> tuple[ReuseFixture, ...]:
    """Load original fixtures and reject schema drift."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("bounded reuse fixtures must be a nonempty array")
    fixtures: list[ReuseFixture] = []
    expected = set(ReuseFixture.__dataclass_fields__)
    for index, value in enumerate(raw):
        item = _mapping(value, f"fixtures[{index}]")
        if set(item) != expected:
            raise ValueError(f"fixtures[{index}] fields differ from the frozen schema")
        fixture = ReuseFixture(
            fixture_id=_string(item["fixture_id"], "fixture_id"),
            relationship=_string(item["relationship"], "relationship"),
            left=_string(item["left"], "left"),
            right=_string(item["right"], "right"),
            expected_outcome=ReuseOutcome(_string(item["expected_outcome"], "expected_outcome")),
            same_latent_program=_boolean(item["same_latent_program"], "same_latent_program"),
            same_complete_candidate=_boolean(
                item["same_complete_candidate"], "same_complete_candidate"
            ),
            same_number_neutral_signature=_boolean(
                item["same_number_neutral_signature"], "same_number_neutral_signature"
            ),
            same_sentence_plan=_boolean(item["same_sentence_plan"], "same_sentence_plan"),
            same_structural_problem=_boolean(
                item["same_structural_problem"], "same_structural_problem"
            ),
            cross_dataset=_boolean(item["cross_dataset"], "cross_dataset"),
            cross_split=_boolean(item["cross_split"], "cross_split"),
            close_paraphrase=_boolean(item["close_paraphrase"], "close_paraphrase"),
            plan_use_after=_integer(item["plan_use_after"], "plan_use_after"),
            plan_cap=_integer(item["plan_cap"], "plan_cap"),
            number_neutral_use_after=_integer(
                item["number_neutral_use_after"], "number_neutral_use_after"
            ),
            number_neutral_cap=_integer(item["number_neutral_cap"], "number_neutral_cap"),
        )
        fixtures.append(fixture)
    if len({fixture.fixture_id for fixture in fixtures}) != len(fixtures):
        raise ValueError("fixture IDs must be unique")
    return tuple(fixtures)


def classify_fixture(policy: ReusePolicy, fixture: ReuseFixture) -> ReuseDecision:
    """Apply exact/latent identity, frozen caps, then supported near-copy review."""

    exact = normalized_text_sha256(fixture.left) == normalized_text_sha256(fixture.right)
    if exact or fixture.same_complete_candidate:
        return ReuseDecision(ReuseOutcome.REJECT, "exact_complete_candidate")
    if fixture.same_latent_program:
        return ReuseDecision(ReuseOutcome.REJECT, "latent_program_copy")
    if policy.reject_number_neutral_reuse and fixture.same_number_neutral_signature:
        return ReuseDecision(ReuseOutcome.REJECT, "legacy_number_neutral_one_use")
    if policy.enforce_usage_caps:
        if fixture.plan_use_after > fixture.plan_cap:
            return ReuseDecision(ReuseOutcome.REJECT, "sentence_plan_cap_exceeded")
        if fixture.number_neutral_use_after > fixture.number_neutral_cap:
            return ReuseDecision(ReuseOutcome.REJECT, "number_neutral_cap_exceeded")
    if fixture.close_paraphrase and not fixture.same_sentence_plan:
        return ReuseDecision(policy.close_paraphrase_action, "cross_plan_close_paraphrase")
    return ReuseDecision(ReuseOutcome.PASS, "bounded_distinct_example")


def calibrate(contract: Contract, fixtures: tuple[ReuseFixture, ...]) -> dict[str, object]:
    """Compare all three policies and freeze the predeclared selected result."""

    candidates: dict[str, object] = {}
    for policy in contract.policies:
        decisions = [classify_fixture(policy, fixture) for fixture in fixtures]
        matches = sum(
            decision.outcome is fixture.expected_outcome
            for decision, fixture in zip(decisions, fixtures, strict=True)
        )
        candidates[policy.policy_id] = {
            "policy_sha256": policy.sha256,
            "exact_matches": matches,
            "fixture_count": len(fixtures),
            "pass_count": sum(item.outcome is ReuseOutcome.PASS for item in decisions),
            "review_count": sum(item.outcome is ReuseOutcome.REVIEW for item in decisions),
            "reject_count": sum(item.outcome is ReuseOutcome.REJECT for item in decisions),
            "mismatched_fixture_ids": [
                fixture.fixture_id
                for decision, fixture in zip(decisions, fixtures, strict=True)
                if decision.outcome is not fixture.expected_outcome
            ],
            "decisions": [
                {
                    "fixture_id": fixture.fixture_id,
                    "relationship": fixture.relationship,
                    "expected": str(fixture.expected_outcome),
                    "actual": str(decision.outcome),
                    "reason": decision.reason,
                }
                for decision, fixture in zip(decisions, fixtures, strict=True)
            ],
        }
    selected_policy = next(
        policy for policy in contract.policies if policy.policy_id == contract.selected_policy_id
    )
    selected_result = cast(dict[str, object], candidates[selected_policy.policy_id])
    if selected_result["exact_matches"] != len(fixtures):
        raise ValueError("predeclared selected bounded-reuse policy failed calibration")
    payload: dict[str, object] = {
        "schema_version": 1,
        "audit_id": REUSE_AUDIT_ID,
        "config_sha256": contract.config_sha256,
        "fixture_set_sha256": _canonical_sha256([asdict(item) for item in fixtures]),
        "fixture_count": len(fixtures),
        "candidate_policies": candidates,
        "selected_policy_id": selected_policy.policy_id,
        "selected_policy_sha256": selected_policy.sha256,
        "selection_frozen_before_smoke_outputs": True,
        "benchmark_contamination": contract.benchmark_contamination,
        "rejected_alternatives": {
            "legacy-one-use-number-neutral": (
                "rejects harmless distinct programs sharing reviewed wording"
            ),
            "permissive-exact-latent-only": (
                "does not enforce concentration caps or near-copy review"
            ),
        },
    }
    payload["calibration_sha256"] = _canonical_sha256(payload)
    return payload


def _attempt_pool(contract: Contract, accepted: int) -> int:
    return _ceil_div(
        accepted * contract.attempt_multiplier_numerator,
        contract.attempt_multiplier_denominator,
    )


def _reuse_cap(contract: Contract, quantity: int, identities: int) -> int:
    return _ceil_div(
        contract.attempt_multiplier_numerator * quantity,
        contract.attempt_multiplier_denominator * identities,
    )


def derive_caps(contract: Contract) -> dict[str, object]:
    """Derive every cap mechanically from frozen quota and inventory values."""

    result: dict[str, object] = {}
    for group, quotas in contract.accepted_quotas.items():
        categories: dict[str, object] = {}
        for category, accepted in quotas.items():
            attempts = _attempt_pool(contract, accepted)
            inventory = contract.identity_inventory[category]
            enabled_attempts = _ceil_div(attempts, 5)
            enabled_accepted = _ceil_div(accepted, 5)
            difficulty_attempts = {
                label: attempts // 3 + (index < attempts % 3)
                for index, label in enumerate(("easy", "medium", "hard"))
            }
            categories[category] = {
                "accepted_quota": accepted,
                "required_attempts": attempts,
                "max_attempts_per_sentence_plan": _reuse_cap(
                    contract, attempts, inventory["sentence_plans"]
                ),
                "max_accepted_per_sentence_plan": _reuse_cap(
                    contract, accepted, inventory["sentence_plans"]
                ),
                "max_attempts_per_number_neutral_signature": _reuse_cap(
                    contract, attempts, inventory["number_neutral_signatures"]
                ),
                "max_accepted_per_number_neutral_signature": _reuse_cap(
                    contract, accepted, inventory["number_neutral_signatures"]
                ),
                "max_attempts_per_plan_scenario_domain": _reuse_cap(
                    contract, attempts, inventory["plan_scenario_domains"]
                ),
                "max_accepted_per_plan_scenario_domain": _reuse_cap(
                    contract, accepted, inventory["plan_scenario_domains"]
                ),
                "max_attempts_per_semantic_frame": _reuse_cap(
                    contract, attempts, inventory["semantic_frames"]
                ),
                "max_accepted_per_semantic_frame": _reuse_cap(
                    contract, accepted, inventory["semantic_frames"]
                ),
                "max_attempts_per_target_type": _reuse_cap(
                    contract, attempts, inventory["target_types"]
                ),
                "max_accepted_per_target_type": _reuse_cap(
                    contract, accepted, inventory["target_types"]
                ),
                "difficulty_attempts": difficulty_attempts,
                "difficulty_caps": {
                    label: _reuse_cap(contract, count, 1)
                    for label, count in difficulty_attempts.items()
                },
                "output_contract_attempts": {
                    "enabled": enabled_attempts,
                    "disabled": attempts - enabled_attempts,
                },
                "output_contract_accepted": {
                    "enabled": enabled_accepted,
                    "disabled": accepted - enabled_accepted,
                },
                "output_contract_caps": {
                    "enabled": _reuse_cap(contract, enabled_attempts, 1),
                    "disabled": _reuse_cap(contract, attempts - enabled_attempts, 1),
                },
            }
        result[group] = categories
    return result


def _latent_hash(draft: CandidateDraft) -> str:
    return _canonical_sha256(asdict(draft.latent_program))


def _mode(draft: CandidateDraft) -> str:
    return draft.latent_program.program_family.split(":", maxsplit=1)[-1]


def _probe_latent_programs(
    contract: Contract, mode_verification_targets: dict[str, dict[str, int]]
) -> dict[str, object]:
    """Construct a fixed finite seed pool and dual-verify required unique programs."""

    result: dict[str, object] = {}
    difficulties = tuple(DifficultyLevel)
    for category in _CATEGORY_ORDER:
        generator = _GENERATORS[category]
        period = _VARIANT_PERIODS[category]
        seen: set[str] = set()
        seen_by_mode: dict[str, set[str]] = {}
        verified_by_mode: Counter[str] = Counter()
        for index in range(contract.latent_probe_candidates):
            material = f"{contract.latent_probe_id}:{category}:{index}"
            seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
            draft = generator(
                seed=seed,
                difficulty=difficulties[index % len(difficulties)],
                variant=index % period,
                output_contract_enabled=False,
            )
            digest = _latent_hash(draft)
            mode = _mode(draft)
            mode_seen = seen_by_mode.setdefault(mode, set())
            if digest not in mode_seen:
                mode_seen.add(digest)
                if verified_by_mode[mode] < mode_verification_targets[category].get(mode, 0):
                    primary, independent, reasons = _verify(draft)
                    if (
                        reasons
                        or not primary.success
                        or not independent.success
                        or primary.answer != independent.answer
                        or primary.answer != draft.canonical_final_answer
                    ):
                        raise ValueError("constructive latent probe failed dual verification")
                    verified_by_mode[mode] += 1
            seen.add(digest)
        result[category] = {
            "fixed_probe_candidates": contract.latent_probe_candidates,
            "unique_programs": len(seen),
            "unique_by_mode": {mode: len(values) for mode, values in sorted(seen_by_mode.items())},
            "dual_verified_by_mode": dict(sorted(verified_by_mode.items())),
            "persisted_programs": False,
        }
    return result


def _exact_rate_mode_capacities() -> dict[str, int | None]:
    ratio_pairs = {
        (first, second + 1 if first == second else second, scale)
        for first in range(2, 8)
        for second in range(2, 10)
        for scale in range(3, 11)
    }
    return {
        "rate_total": 12 * 8,
        "ratio_scale": len(ratio_pairs),
        "percentage": 8 * 13,
        "weighted_average": None,
        "combined_rate": 8 * 8 * 6,
    }


def _exact_discrete_mode_capacities() -> dict[str, int | None]:
    ranges = {
        DifficultyLevel.EASY: (9, 35),
        DifficultyLevel.MEDIUM: (36, 80),
        DifficultyLevel.HARD: (81, 200),
    }
    equal: set[tuple[int, int]] = set()
    dual: set[tuple[int, int, int, int]] = set()
    for _difficulty, (low, high) in ranges.items():
        for search_space in range(low, high + 1):
            for containers in range(3, 10):
                each = max(2, (search_space - 1) // containers)
                total = containers * each
                while total + 1 < low:
                    each += 1
                    total = containers * each
                equal.add((total, containers))
        for first_per in range(2, 7):
            for second_per in range(2, 8):
                target = 2
                while True:
                    first_resource = target * first_per + (first_per - 1)
                    second_resource = (target + 1) * second_per + (second_per - 1)
                    capacity_domain = max(first_resource, second_resource) + 1
                    if low <= capacity_domain <= high:
                        dual.add((first_resource, second_resource, first_per, second_per))
                        break
                    if capacity_domain > high:
                        break
                    target += 1
    return {
        "two_type_allocation": None,
        "complete_packages": 192 * 7,
        "equal_distribution": len(equal),
        "dual_capacity": len(dual),
    }


def _bounded_identity_capacity(
    contract: Contract, caps: dict[str, object], category: str
) -> tuple[int, dict[str, int]]:
    inventory = contract.identity_inventory[category]
    layer_capacity: Counter[str] = Counter()
    for group in ("targeted", "generic_control"):
        group_caps = cast(dict[str, Any], cast(dict[str, object], caps[group])[category])
        layer_capacity["sentence_plan"] += (
            group_caps["max_attempts_per_sentence_plan"] * inventory["sentence_plans"]
        )
        layer_capacity["number_neutral"] += (
            group_caps["max_attempts_per_number_neutral_signature"]
            * inventory["number_neutral_signatures"]
        )
        layer_capacity["plan_scenario_domain"] += (
            group_caps["max_attempts_per_plan_scenario_domain"] * inventory["plan_scenario_domains"]
        )
        layer_capacity["semantic_frame"] += (
            group_caps["max_attempts_per_semantic_frame"] * inventory["semantic_frames"]
        )
        layer_capacity["target_type"] += (
            group_caps["max_attempts_per_target_type"] * inventory["target_types"]
        )
        layer_capacity["difficulty"] += sum(group_caps["difficulty_caps"].values())
        layer_capacity["output_contract"] += sum(group_caps["output_contract_caps"].values())
    values = dict(sorted(layer_capacity.items()))
    return min(values.values()), values


def _frame_mode_capacity(caps: dict[str, object], category: str, frames_per_mode: int) -> int:
    total = sum(
        cast(
            int,
            cast(
                dict[str, Any],
                cast(dict[str, object], caps[group])[category],
            )["max_attempts_per_semantic_frame"],
        )
        for group in ("targeted", "generic_control")
    )
    return frames_per_mode * total


def build_capacity_audit(contract: Contract, calibration: dict[str, object]) -> dict[str, object]:
    """Apply bounded reuse, exact latent limits, and every future quota stratum."""

    caps = derive_caps(contract)
    rate_frame_mode_cap = _frame_mode_capacity(caps, _RATES, 4)
    discrete_frame_mode_cap = _frame_mode_capacity(caps, _DISCRETE, 5)
    bookkeeping_target_cap = sum(
        cast(dict[str, Any], cast(dict[str, object], caps[group])[_BOOKKEEPING])[
            "max_attempts_per_target_type"
        ]
        for group in ("targeted", "generic_control")
    )
    mode_targets = {
        _BOOKKEEPING: {
            "inventory": bookkeeping_target_cap,
            "grouping": bookkeeping_target_cap,
        },
        _RATES: {
            mode: rate_frame_mode_cap
            for mode in (
                "rate_total",
                "ratio_scale",
                "percentage",
                "weighted_average",
                "combined_rate",
            )
        },
        _DISCRETE: {
            mode: discrete_frame_mode_cap
            for mode in (
                "two_type_allocation",
                "complete_packages",
                "equal_distribution",
                "dual_capacity",
            )
        },
    }
    probe = _probe_latent_programs(contract, mode_targets)
    exact_bounds: dict[str, dict[str, int | None]] = {
        _RATES: _exact_rate_mode_capacities(),
        _DISCRETE: _exact_discrete_mode_capacities(),
    }
    category_result: dict[str, object] = {}
    gate = True
    total_attempts = 0
    for category in _CATEGORY_ORDER:
        required = sum(
            cast(dict[str, Any], cast(dict[str, object], caps[group])[category])[
                "required_attempts"
            ]
            for group in ("targeted", "generic_control")
        )
        accepted = sum(
            contract.accepted_quotas[group][category] for group in contract.accepted_quotas
        )
        total_attempts += required
        bounded_capacity, layers = _bounded_identity_capacity(contract, caps, category)
        probe_data = cast(dict[str, Any], probe[category])
        mode_details: dict[str, object] = {}
        if category == _BOOKKEEPING:
            latent_capacity = min(
                bounded_capacity,
                sum(
                    min(probe_data["unique_by_mode"][mode], target)
                    for mode, target in mode_targets[category].items()
                ),
            )
            for mode, target in mode_targets[category].items():
                mode_details[mode] = {
                    "frame_or_target_cap": target,
                    "constructive_unique_programs": probe_data["unique_by_mode"][mode],
                    "available_bounded_programs": min(target, probe_data["unique_by_mode"][mode]),
                }
        else:
            latent_capacity = 0
            for mode, target in mode_targets[category].items():
                probed = probe_data["unique_by_mode"][mode]
                exact = exact_bounds[category][mode]
                available = min(target, probed if exact is None else exact)
                latent_capacity += available
                mode_details[mode] = {
                    "semantic_frame_cap": target,
                    "constructive_unique_programs": probed,
                    "exact_finite_upper_bound": exact,
                    "available_bounded_programs": available,
                    "dual_verified_programs": probe_data["dual_verified_by_mode"][mode],
                }
            latent_capacity = min(latent_capacity, bounded_capacity)
        passed = latent_capacity >= required
        gate = gate and passed
        category_result[category] = {
            "accepted_quota": accepted,
            "required_attempts": required,
            "bounded_identity_capacity": bounded_capacity,
            "identity_layer_capacities": layers,
            "constructive_probe": probe_data,
            "latent_capacity_under_balanced_frame_or_target_caps": latent_capacity,
            "capacity_ratio": latent_capacity / required,
            "shortfall": max(0, required - latent_capacity),
            "gate_passed": passed,
            "mode_details": mode_details,
            "group_caps": {
                group: cast(dict[str, object], caps[group])[category]
                for group in ("targeted", "generic_control")
            },
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "audit_id": REUSE_AUDIT_ID,
        "config_sha256": contract.config_sha256,
        "fixture_set_sha256": calibration["fixture_set_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "selected_policy_id": calibration["selected_policy_id"],
        "selected_policy_sha256": calibration["selected_policy_sha256"],
        "identity_definitions": {
            "exact_question": "fully rendered and normalized exact text; globally unique",
            "latent_program": (
                "complete executable program plus instantiated parameters; globally unique"
            ),
            "structural_problem": (
                "content-free skill and operation topology; balanced repetition allowed"
            ),
            "surface_template": (
                "plan, scenario, lexical family, ordering, render and number-neutral "
                "signatures; capped reuse allowed"
            ),
        },
        "planned_accepted_total": 8000,
        "required_attempt_total": total_attempts,
        "category_capacity": category_result,
        "capacity_gate_passed": gate,
        "full_generation_feasible": gate,
        "allocator_implemented": False,
        "candidate_schedule_created": False,
        "fresh_smoke_run": False,
        "deterministic_replay_run": False,
        "review_packet_created": False,
        "stop_reason": (
            None
            if gate
            else (
                "finite unique latent-program supply under balanced semantic-frame caps "
                "is insufficient"
            )
        ),
        "benchmark_contamination_unchanged": True,
        "sealed_final_accessed": False,
    }
    payload["capacity_audit_sha256"] = _canonical_sha256(payload)
    return payload


def run_audit(config_path: Path, calibration_path: Path, capacity_path: Path) -> dict[str, object]:
    """Write content-free calibration and capacity evidence only."""

    contract = load_contract(config_path)
    fixture_path = contract.fixture_path
    if not fixture_path.is_absolute():
        fixture_path = Path.cwd() / fixture_path
    fixtures = load_fixtures(fixture_path)
    calibration = calibrate(contract, fixtures)
    capacity = build_capacity_audit(contract, calibration)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    capacity_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    capacity_path.write_text(
        json.dumps(capacity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return capacity


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--capacity-output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_audit(
        args.config.resolve(),
        args.calibration_output.resolve(),
        args.capacity_output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
