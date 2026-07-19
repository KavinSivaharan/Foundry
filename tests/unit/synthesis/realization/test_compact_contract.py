from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from foundry.synthesis.realization.compact_prompting import serialize_compact_request
from foundry.synthesis.realization.compact_request import prepare_compact_request
from foundry.synthesis.realization.compact_runtime import _ClosingTagCriteria
from foundry.synthesis.realization.compact_smoke_contract import (
    build_compact_attempt_plan,
    generate_procedural_ir,
    load_compact_smoke_config,
)
from foundry.synthesis.realization.model_contracts import PlaceholderKind

CONFIG = Path("configs/synthesis/local_realization_compact_micro.yaml")


def test_compact_plan_has_exact_fresh_allocations() -> None:
    config = load_compact_smoke_config(CONFIG)
    plans = build_compact_attempt_plan(config)
    assert len(plans) == 30
    assert len({plan.random_seed for plan in plans}) == 30
    assert sum(plan.output_contract_enabled for plan in plans) == 6
    assert Counter(str(plan.group) for plan in plans) == {"targeted": 15, "generic_control": 15}
    assert Counter(str(plan.category) for plan in plans) == {
        "multi_step_bookkeeping_or_omission": 13,
        "rate_ratio_percentage_or_average": 9,
        "constraint_distribution_or_discrete_reasoning": 8,
    }


def test_all_compact_requests_are_value_blind_and_complete() -> None:
    config = load_compact_smoke_config(CONFIG)
    for plan in build_compact_attempt_plan(config):
        prepared = prepare_compact_request(
            generate_procedural_ir(plan), style_variant=plan.style_variant
        )
        prompt = serialize_compact_request(prepared.request)
        for placeholder in prepared.base.request.placeholders:
            if placeholder.kind in {PlaceholderKind.ENTITY, PlaceholderKind.LOCATION}:
                assert prepared.replacements[placeholder.token] not in prompt
        assert "Final answer:" not in prompt
        assert "canonical_answer" not in prompt
        assert "question_intent" not in prompt
        assert "target_type" not in prompt
        assert "placeholder_inventory" not in prompt
        assert "clause_to_semantic" not in prompt
        assert prepared.request.tag_order[-1] == "Q"
        assert len(prepared.request.segments) >= 3
        assert set(prepared.replacements) == {
            token for segment in prepared.request.segments for token in segment.required_tokens
        }


def test_prompt_contains_one_original_syntax_example_and_compact_spec() -> None:
    config = load_compact_smoke_config(CONFIG)
    plan = build_compact_attempt_plan(config)[0]
    prepared = prepare_compact_request(generate_procedural_ir(plan), style_variant=0)
    prompt = serialize_compact_request(prepared.request)
    assert "Syntax example only" in prompt
    assert "ORDER=E1,Q" in prompt
    assert prompt.count("SPEC:") == 1
    assert f"ORDER={','.join(prepared.request.tag_order)}" in prompt


def test_closing_q_stops_only_beams_with_complete_tag() -> None:
    criteria = _ClosingTagCriteria((8, 9))
    result = criteria(
        torch.tensor([[1, 8, 9], [1, 8, 7]], dtype=torch.long),
        torch.zeros((2, 1), dtype=torch.float32),
    )
    assert tuple(result.shape) == (2,)
    assert result.tolist() == [True, False]
