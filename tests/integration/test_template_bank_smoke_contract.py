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


def test_template_bank_v2_smoke_uses_a_fresh_seed_and_ignored_review_path() -> None:
    first = load_smoke_config(Path("configs/synthesis/template_bank_smoke.yaml"))
    second = load_smoke_config(Path("configs/synthesis/template_bank_smoke_v2.yaml"))
    assert second.attempts == 120
    assert second.master_seed != first.master_seed
    assert second.raw_directory.as_posix() == "results/raw/template_bank_smoke_v2"
    assert second.manual_audit_path.as_posix().endswith("template_bank_smoke_v2/human_review.md")
