from pathlib import Path

from foundry.synthesis.template_bank.matched_policy import (
    calibrate,
    derive_caps,
    fixture_allowed,
    load_fixtures,
    load_matched_policy_config,
)

_CONFIG = Path("configs/synthesis/matched_template_signal.yaml")


def test_matched_policy_calibration_passes_all_original_fixtures() -> None:
    config = load_matched_policy_config(_CONFIG)
    fixtures = load_fixtures(config.fixture_path)
    result = calibrate(config, fixtures)
    assert result["exact_fixture_matches"] == 8
    assert result["mismatched_fixture_ids"] == []
    assert result["calibration_gate_passed"] is True


def test_repeated_reviewed_structure_is_not_identity_rejection() -> None:
    config = load_matched_policy_config(_CONFIG)
    fixtures = {item.fixture_id: item for item in load_fixtures(config.fixture_path)}
    assert fixture_allowed(fixtures["same-plan-distinct-example"])
    assert fixture_allowed(fixtures["same-number-neutral-distinct-example"])
    assert not fixture_allowed(fixtures["exact-rendered-copy"])
    assert not fixture_allowed(fixtures["latent-program-copy"])
    assert not fixture_allowed(fixtures["development-contamination"])


def test_caps_follow_frozen_formula() -> None:
    caps = derive_caps(
        275,
        compatible_sentence_plans=72,
        compatible_frames=18,
        compatible_scenarios=20,
    )
    assert caps.max_sentence_plan_usage == 6
    assert caps.max_frame_usage == 18
    assert caps.max_scenario_usage == 16
    assert caps.max_number_neutral_usage == 41
