from collections import Counter
from pathlib import Path

from foundry.synthesis.pipeline import (
    GroupName,
    build_attempt_plan,
    deterministic_aggregate_sha256,
    load_smoke_config,
)

FRESH_CONFIG = Path("configs/synthesis/gsm1k_phase1_smoke_v2.yaml")


def test_smoke_plan_has_exact_frozen_distribution() -> None:
    config = load_smoke_config(Path("configs/synthesis/gsm1k_phase1_smoke.yaml"))
    plans = build_attempt_plan(config)

    assert len(plans) == 120
    assert len({plan.random_seed for plan in plans}) == 120
    assert Counter(plan.group for plan in plans) == {
        GroupName.TARGETED: 60,
        GroupName.GENERIC_CONTROL: 60,
    }
    assert set(Counter(plan.category for plan in plans[:60]).values()) == {33, 14, 13}
    assert set(Counter(plan.category for plan in plans[60:]).values()) == {20}
    assert sum(plan.output_contract_enabled for plan in plans[:60]) == 12
    assert sum(plan.output_contract_enabled for plan in plans[60:]) == 12


def test_each_group_has_balanced_difficulty_and_exact_output_rate() -> None:
    plans = build_attempt_plan(load_smoke_config(Path("configs/synthesis/gsm1k_phase1_smoke.yaml")))

    for group in GroupName:
        selected = [plan for plan in plans if plan.group is group]
        assert set(Counter(plan.difficulty for plan in selected).values()) == {20}
        assert sum(plan.output_contract_enabled for plan in selected) / len(selected) == 0.20


def test_plan_replay_is_identical() -> None:
    config = load_smoke_config(Path("configs/synthesis/gsm1k_phase1_smoke.yaml"))

    assert build_attempt_plan(config) == build_attempt_plan(config)


def test_deterministic_aggregate_excludes_runtime_telemetry() -> None:
    config = load_smoke_config(Path("configs/synthesis/gsm1k_phase1_smoke.yaml"))

    assert deterministic_aggregate_sha256(config=config, records=()) == (
        deterministic_aggregate_sha256(config=config, records=())
    )


def test_blocker_resolution_plan_uses_a_fresh_seed_and_same_frozen_distribution() -> None:
    original = load_smoke_config(Path("configs/synthesis/gsm1k_phase1_smoke.yaml"))
    fresh = load_smoke_config(FRESH_CONFIG)
    plans = build_attempt_plan(fresh)

    assert fresh.master_seed != original.master_seed
    assert len(plans) == 120
    assert Counter(plan.group for plan in plans) == {
        GroupName.TARGETED: 60,
        GroupName.GENERIC_CONTROL: 60,
    }
    original_plans = build_attempt_plan(original)
    assert Counter(plan.category for plan in plans[:60]) == Counter(
        plan.category for plan in original_plans[:60]
    )
    assert Counter(plan.category for plan in plans[60:]) == Counter(
        plan.category for plan in original_plans[60:]
    )
    assert sum(plan.output_contract_enabled for plan in plans[:60]) == 12
    assert sum(plan.output_contract_enabled for plan in plans[60:]) == 12
