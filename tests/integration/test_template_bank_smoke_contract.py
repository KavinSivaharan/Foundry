"""Bounded template-bank smoke configuration integration checks."""

from pathlib import Path

from foundry.synthesis.pipeline import build_attempt_plan, load_smoke_config


def test_template_bank_smoke_allocation_is_exact_and_bounded() -> None:
    config = load_smoke_config(Path("configs/synthesis/template_bank_smoke.yaml"))
    plans = build_attempt_plan(config)
    assert len(plans) == 120
    assert sum(plan.output_contract_enabled for plan in plans[:60]) == 12
    assert sum(plan.output_contract_enabled for plan in plans[60:]) == 12
    assert len({plan.random_seed for plan in plans}) == 120
