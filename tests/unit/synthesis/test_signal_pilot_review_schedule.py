"""Signal-first 120-question review-schedule tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from foundry.synthesis.template_bank.signal_pilot import canonical_sha256
from foundry.synthesis.template_bank.signal_smoke import build_review_schedule

_CONFIG = Path("configs/synthesis/signal_pilot.yaml")


def test_review_schedule_is_exact_unique_and_reproducible() -> None:
    first = build_review_schedule(_CONFIG)
    second = build_review_schedule(_CONFIG)
    assert first == second
    assert len(first) == 120
    assert canonical_sha256([item.__dict__ for item in first]) == (
        "c00a24542b460b107f165e1e003122b84f1d95e8a4962920396c70f1dddd9492"
    )
    for field in (
        "slot_id",
        "candidate_id",
        "latent_program_sha256",
        "semantic_ir_sha256",
        "render_signature_sha256",
        "predicted_number_neutral_sha256",
    ):
        assert len({getattr(item, field) for item in first}) == 120
    assert Counter((item.group, item.category) for item in first) == {
        ("targeted", "multi_step_bookkeeping_or_omission"): 33,
        ("targeted", "rate_ratio_percentage_or_average"): 14,
        ("targeted", "constraint_distribution_or_discrete_reasoning"): 13,
        ("generic_control", "multi_step_bookkeeping_or_omission"): 20,
        ("generic_control", "rate_ratio_percentage_or_average"): 20,
        ("generic_control", "constraint_distribution_or_discrete_reasoning"): 20,
    }
    assert Counter((item.group, item.output_contract_enabled) for item in first) == {
        ("targeted", True): 12,
        ("targeted", False): 48,
        ("generic_control", True): 12,
        ("generic_control", False): 48,
    }
    assert Counter(item.difficulty for item in first) == {
        "easy": 42,
        "medium": 41,
        "hard": 37,
    }
