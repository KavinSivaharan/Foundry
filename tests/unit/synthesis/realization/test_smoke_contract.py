"""Frozen contract tests for the bounded local-model realization smoke."""

from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path

from foundry.synthesis.realization.local_runtime import PinnedQwenRealizer
from foundry.synthesis.realization.prompting import serialize_value_blind_request
from foundry.synthesis.realization.request_builder import prepare_realization_request
from foundry.synthesis.realization.smoke_contract import (
    build_realization_attempt_plan,
    generate_procedural_ir,
    load_realization_smoke_config,
)

CONFIG = Path("configs/synthesis/local_realization_smoke.yaml")


def test_smoke_plan_is_exactly_120_fresh_matched_irs() -> None:
    config = load_realization_smoke_config(CONFIG)
    plans = build_realization_attempt_plan(config)
    assert len(plans) == 120
    assert len({plan.random_seed for plan in plans}) == 120
    assert Counter(str(plan.group) for plan in plans) == {"targeted": 60, "generic_control": 60}
    assert Counter(str(plan.category) for plan in plans if str(plan.group) == "targeted") == {
        "multi_step_bookkeeping_or_omission": 33,
        "rate_ratio_percentage_or_average": 14,
        "constraint_distribution_or_discrete_reasoning": 13,
    }
    assert Counter(
        str(plan.category) for plan in plans if str(plan.group) == "generic_control"
    ) == {
        "multi_step_bookkeeping_or_omission": 20,
        "rate_ratio_percentage_or_average": 20,
        "constraint_distribution_or_discrete_reasoning": 20,
    }
    assert Counter(str(plan.group) for plan in plans if plan.output_contract_enabled) == {
        "targeted": 12,
        "generic_control": 12,
    }


def test_all_planned_irs_compile_to_value_blind_requests() -> None:
    config = load_realization_smoke_config(CONFIG)
    plans = build_realization_attempt_plan(config)
    request_hashes: set[str] = set()
    for plan in plans:
        draft = generate_procedural_ir(plan)
        prepared = prepare_realization_request(draft, style_variant=plan.style_variant)
        request_hashes.add(prepared.request_sha256)
        assert set(prepared.replacements) == {
            placeholder.token for placeholder in prepared.request.placeholders
        }
        assert (
            draft.canonical_final_answer.render() not in prepared.request.required_question_intent
        )
        serialized = serialize_value_blind_request(prepared.request)
        assert draft.candidate_id not in serialized
        assert "canonical_final_answer" not in serialized
        assert "solution" not in serialized.lower()
        assert "benchmark" not in serialized.lower()
    assert len(request_hashes) == 120


def test_model_runtime_public_generation_interface_cannot_receive_benchmark_content() -> None:
    signature = inspect.signature(PinnedQwenRealizer.generate)
    assert tuple(signature.parameters) == ("self", "prepared")
