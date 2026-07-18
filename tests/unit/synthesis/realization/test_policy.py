"""Frozen policy tests for the separately approved future realization smoke."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from foundry.synthesis.realization.policy import (
    DesignConfigError,
    InternalDiversityMode,
    load_local_realization_design,
)

CONFIG = Path("configs/synthesis/local_realization_design.yaml")


def test_complete_design_config_hash_is_frozen() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        "d6e6ca82681b702e07c71a9732a8c81159ea7a9bca78c73193228f72ca4ec3a5"
    )


def test_design_config_pins_models_and_remains_non_executable() -> None:
    design = load_local_realization_design(CONFIG)
    assert design.design_only
    assert design.primary_model.repo_id == "Qwen/Qwen3-1.7B"
    assert design.primary_model.revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert design.fallback_model.repo_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert design.fallback_model.revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert not design.primary_model.trust_remote_code
    assert not design.primary_model.thinking_enabled


def test_generation_budget_has_fixed_beams_without_retries() -> None:
    design = load_local_realization_design(CONFIG)
    assert design.generation.do_sample is False
    assert design.generation.candidates_per_ir == 3
    assert design.generation.num_beams == design.generation.num_return_sequences == 3
    assert design.future_smoke.ir_attempts == 120
    assert design.future_smoke.ir_attempts * design.generation.candidates_per_ir == 360


def test_future_smoke_preserves_allocations_and_zero_defect_gates() -> None:
    smoke = load_local_realization_design(CONFIG).future_smoke
    assert smoke.targeted_allocations == {
        "bookkeeping": 33,
        "discrete": 13,
        "rates": 14,
    }
    assert smoke.generic_allocations == {
        "bookkeeping": 20,
        "discrete": 20,
        "rates": 20,
    }
    assert smoke.output_contract_per_group == 12
    assert smoke.minimum_clean_accepts == 90
    assert smoke.minimum_accepts_per_family == 15
    assert (
        smoke.maximum_false_labels
        == smoke.maximum_semantic_drift_accepts
        == smoke.maximum_invalid_accepts
        == smoke.maximum_unresolved_contamination
        == 0
    )


def test_benchmark_thresholds_stay_frozen_but_internal_policy_is_unset() -> None:
    policy = load_local_realization_design(CONFIG).semantic_screening
    assert policy.benchmark_manual_review_threshold == 0.75
    assert policy.benchmark_rejection_threshold == 0.82
    assert policy.internal_diversity_mode is InternalDiversityMode.CALIBRATE_SEPARATELY
    assert policy.internal_semantic_threshold is None


def test_policy_objects_reject_lowered_gates() -> None:
    smoke = load_local_realization_design(CONFIG).future_smoke
    with pytest.raises(ValueError, match="cannot be lowered"):
        replace(smoke, minimum_clean_accepts=89)


def test_loader_rejects_executable_design_flag(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace("design_only: true", "design_only: false")
    damaged = tmp_path / "damaged.yaml"
    damaged.write_text(text, encoding="utf-8")
    with pytest.raises(DesignConfigError, match="design-only"):
        load_local_realization_design(damaged)
