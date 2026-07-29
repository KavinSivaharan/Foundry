from __future__ import annotations

from foundry.phase2.l3_grpo_signal_blocker import (
    BLOCKER_ID,
    EXPECTED_EXCEPTION,
    FIXTURE_VALUES,
    OBSERVED_FAMILIES,
)


def test_blocker_contract_freezes_the_observed_failure() -> None:
    assert BLOCKER_ID == "foundry-l3-grpo-signal-audit-blocker-v1"
    assert EXPECTED_EXCEPTION == ("stock TRL advantages differ from frozen reward projection")
    assert OBSERVED_FAMILIES == (
        "constraint_distribution_or_discrete_reasoning",
        "multi_step_bookkeeping_or_omission",
        "rate_ratio_percentage_or_average",
    )
    assert len(FIXTURE_VALUES) == 16
    assert len(FIXTURE_VALUES) ** 4 == 65_536


def test_blocker_source_contains_no_retry_or_model_generation_path() -> None:
    import inspect

    from foundry.phase2 import l3_grpo_signal_blocker as blocker

    source = inspect.getsource(blocker)
    assert "_generate_and_score_completions" not in source
    assert ".generate(" not in source
    assert ".train(" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
