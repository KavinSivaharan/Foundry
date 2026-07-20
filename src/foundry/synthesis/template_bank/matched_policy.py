"""Frozen balanced reuse policy for the matched-template signal experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.template_bank.signal_pilot import canonical_sha256

POLICY_ID = "matched-template-signal-v1"


@dataclass(frozen=True)
class MatchedTemplateCaps:
    """Per-dataset/family accepted-example concentration limits."""

    max_sentence_plan_usage: int
    max_frame_usage: int
    max_scenario_usage: int
    max_number_neutral_usage: int


@dataclass(frozen=True)
class MatchedPolicyFixture:
    """Original content-free policy calibration case."""

    fixture_id: str
    same_exact_question: bool
    same_latent_program: bool
    cross_dataset_exact_copy: bool
    same_reviewed_plan: bool
    same_number_neutral_structure: bool
    benchmark_contamination: bool
    sentence_plan_use_after: int
    sentence_plan_cap: int
    frame_use_after: int
    frame_cap: int
    scenario_use_after: int
    scenario_cap: int
    number_neutral_use_after: int
    number_neutral_cap: int
    expected_allowed: bool


@dataclass(frozen=True)
class MatchedPolicyConfig:
    """Pinned matched-template policy configuration."""

    config_sha256: str
    policy_id: str
    fixture_path: Path
    concentration_percent: int
    additive_headroom: int


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _positive(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def load_matched_policy_config(path: Path) -> MatchedPolicyConfig:
    """Load and strictly validate the frozen fast-track policy."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "matched-template policy")
    if root.get("schema_version") != 1 or root.get("policy_id") != POLICY_ID:
        raise ValueError("matched-template policy identity differs")
    uniqueness = _mapping(root.get("global_uniqueness"), "global_uniqueness")
    if uniqueness != {
        "exact_rendered_question": True,
        "normalized_exact_rendered_question": True,
        "latent_program": True,
        "synthetic_example_id": True,
        "targeted_generic_full_example": True,
    }:
        raise ValueError("matched-template global uniqueness contract differs")
    repeatable = root.get("balanced_repeatable_identities")
    if repeatable != [
        "sentence_plan",
        "semantic_frame",
        "number_neutral_structure",
        "reasoning_structure",
        "scenario_domain",
        "target_type",
    ]:
        raise ValueError("matched-template repeatable identity set differs")
    concentration = _mapping(root.get("concentration"), "concentration")
    if concentration != {
        "number_neutral_max_percent": 15,
        "cap_additive_headroom": 2,
        "balance_order": ["sentence_plan", "semantic_frame", "scenario_domain"],
    }:
        raise ValueError("matched-template concentration policy differs")
    preserved = _mapping(root.get("preserved_controls"), "preserved_controls")
    if preserved.get("sealed_final_access") is not False or any(
        value is not True for key, value in preserved.items() if key != "sealed_final_access"
    ):
        raise ValueError("a matched-template preserved control changed")
    return MatchedPolicyConfig(
        config_sha256=canonical_sha256(root),
        policy_id=POLICY_ID,
        fixture_path=Path(str(root.get("fixture_path"))),
        concentration_percent=15,
        additive_headroom=2,
    )


def derive_caps(
    accepted_examples: int,
    *,
    compatible_sentence_plans: int,
    compatible_frames: int,
    compatible_scenarios: int,
) -> MatchedTemplateCaps:
    """Derive the four frozen caps mechanically from accepted quota/inventory."""

    if (
        accepted_examples <= 0
        or min(
            compatible_sentence_plans,
            compatible_frames,
            compatible_scenarios,
        )
        <= 0
    ):
        raise ValueError("matched-template cap margins must be positive")

    def cap(inventory: int) -> int:
        return (accepted_examples + inventory - 1) // inventory + 2

    return MatchedTemplateCaps(
        max_sentence_plan_usage=cap(compatible_sentence_plans),
        max_frame_usage=cap(compatible_frames),
        max_scenario_usage=cap(compatible_scenarios),
        max_number_neutral_usage=max(1, accepted_examples * 15 // 100),
    )


def load_fixtures(path: Path) -> tuple[MatchedPolicyFixture, ...]:
    """Load original content-free policy fixtures."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("matched-template fixtures must be a nonempty list")
    expected = set(MatchedPolicyFixture.__dataclass_fields__)
    fixtures: list[MatchedPolicyFixture] = []
    for index, value in enumerate(raw):
        item = _mapping(value, f"fixtures[{index}]")
        if set(item) != expected:
            raise ValueError(f"fixtures[{index}] fields differ")
        fixtures.append(
            MatchedPolicyFixture(
                fixture_id=cast(str, item["fixture_id"]),
                same_exact_question=cast(bool, item["same_exact_question"]),
                same_latent_program=cast(bool, item["same_latent_program"]),
                cross_dataset_exact_copy=cast(bool, item["cross_dataset_exact_copy"]),
                same_reviewed_plan=cast(bool, item["same_reviewed_plan"]),
                same_number_neutral_structure=cast(bool, item["same_number_neutral_structure"]),
                benchmark_contamination=cast(bool, item["benchmark_contamination"]),
                sentence_plan_use_after=cast(int, item["sentence_plan_use_after"]),
                sentence_plan_cap=cast(int, item["sentence_plan_cap"]),
                frame_use_after=cast(int, item["frame_use_after"]),
                frame_cap=cast(int, item["frame_cap"]),
                scenario_use_after=cast(int, item["scenario_use_after"]),
                scenario_cap=cast(int, item["scenario_cap"]),
                number_neutral_use_after=cast(int, item["number_neutral_use_after"]),
                number_neutral_cap=cast(int, item["number_neutral_cap"]),
                expected_allowed=cast(bool, item["expected_allowed"]),
            )
        )
    if len({item.fixture_id for item in fixtures}) != len(fixtures):
        raise ValueError("matched-template fixture IDs must be unique")
    return tuple(fixtures)


def fixture_allowed(fixture: MatchedPolicyFixture) -> bool:
    """Apply the frozen policy without consulting generated or benchmark output."""

    if (
        fixture.same_exact_question
        or fixture.same_latent_program
        or fixture.cross_dataset_exact_copy
        or fixture.benchmark_contamination
    ):
        return False
    return (
        fixture.sentence_plan_use_after <= fixture.sentence_plan_cap
        and fixture.frame_use_after <= fixture.frame_cap
        and fixture.scenario_use_after <= fixture.scenario_cap
        and fixture.number_neutral_use_after <= fixture.number_neutral_cap
    )


def calibrate(
    config: MatchedPolicyConfig,
    fixtures: tuple[MatchedPolicyFixture, ...],
) -> dict[str, object]:
    """Freeze fixture, policy, and calibration hashes before generation."""

    decisions = [fixture_allowed(item) for item in fixtures]
    mismatches = [
        item.fixture_id
        for item, decision in zip(fixtures, decisions, strict=True)
        if decision is not item.expected_allowed
    ]
    if mismatches:
        raise ValueError(f"matched-template policy fixture mismatch: {mismatches}")
    contract = {
        "policy_id": config.policy_id,
        "exact_question_duplicates": "reject",
        "normalized_exact_duplicates": "reject",
        "latent_program_duplicates": "reject",
        "cross_dataset_exact_or_latent": "reject",
        "number_neutral_reuse": "allow-within-15-percent-stratum-cap",
        "sentence_plan_cap": "ceil(accepted/compatible_plans)+2",
        "frame_cap": "ceil(accepted/compatible_frames)+2",
        "scenario_cap": "ceil(accepted/compatible_scenarios)+2",
        "benchmark_contamination": "unchanged-reject-or-review",
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "policy_id": config.policy_id,
        "config_sha256": config.config_sha256,
        "fixture_count": len(fixtures),
        "fixture_set_sha256": canonical_sha256([asdict(item) for item in fixtures]),
        "policy_contract": contract,
        "policy_sha256": canonical_sha256(contract),
        "exact_fixture_matches": len(fixtures),
        "mismatched_fixture_ids": mismatches,
        "calibration_gate_passed": True,
        "frozen_before_generation": True,
        "benchmark_contamination_changed": False,
        "sealed_final_accessed": False,
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    return payload
