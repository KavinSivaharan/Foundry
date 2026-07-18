from foundry.synthesis.audit import ManualAuditSummary, readiness_reasons


def _summary() -> dict[str, object]:
    return {
        "attempted": 120,
        "accepted": 95,
        "deterministic_replay_passed": True,
        "verifier_disagreements": 0,
        "primary_verifier_failures": 0,
        "independent_verifier_failures": 0,
        "accepted_by_category": {"a": 30, "b": 30, "c": 35},
        "benchmark_boundary": {"sealed_final_accessed": False},
    }


def test_readiness_passes_only_when_every_frozen_gate_passes() -> None:
    audit = ManualAuditSummary(120, 0, 0, 0, 0, ())

    assert readiness_reasons(_summary(), audit) == ()


def test_readiness_reports_yield_invalid_acceptance_and_family_defects() -> None:
    summary = _summary()
    summary["accepted"] = 24
    summary["accepted_by_category"] = {"bookkeeping": 4, "rates": 16, "discrete": 4}
    audit = ManualAuditSummary(120, 0, 5, 0, 0, ("limited_template_diversity",))

    reasons = readiness_reasons(summary, audit)

    assert "invalid_accepted_examples_present" in reasons
    assert "acceptance_rate_below_75_percent" in reasons
    assert "accepted_below_15:bookkeeping" in reasons
    assert "accepted_below_15:discrete" in reasons
    assert "systematic_generator_weaknesses_present" in reasons
