from __future__ import annotations

from typing import Any

import pytest

from foundry.phase2.l3_grpo_signal_audit import (
    COMPLETIONS_PER_ARM,
    COMPLETIONS_PER_GROUP,
    GROUPS_PER_ARM,
    REWARD_COMPONENT_FIELDS,
    build_signal_summary,
    classify_zero_variance_group,
    signal_audit_method_contract,
)
from foundry.training.config import canonical_sha256


def _components(value: float = 0.0) -> dict[str, list[float]]:
    return {name: [value] * COMPLETIONS_PER_GROUP for name in REWARD_COMPONENT_FIELDS}


def _group(
    *,
    arm: str,
    position: int,
    informative: bool,
    family: str,
) -> dict[str, Any]:
    rewards = [0.0, 1.0, 0.0, 1.0] if informative else [1.0] * 4
    advantages = [-1.0, 1.0, -1.0, 1.0] if informative else [0.0] * 4
    row: dict[str, Any] = {
        "arm": arm,
        "schedule_position": position,
        "group_id": f"{arm}-{position:02d}",
        "source_kind": "task" if position <= 24 else "base_replay",
        "task_family": family if position <= 24 else None,
        "prompt_sha256": f"{position:064x}",
        "completion_count": 4,
        "completion_sha256s": [f"{position * 10 + index:064x}" for index in range(4)],
        "completion_token_counts": [2, 2, 2, 2],
        "distinct_completion_count": 4,
        "reward_vector": rewards,
        "reward_component_vectors": _components(),
        "reward_mean": 0.5 if informative else 1.0,
        "reward_variance": 0.25 if informative else 0.0,
        "advantages": advantages,
        "nonzero_advantage_count": 4 if informative else 0,
        "reward_rank_count": 2 if informative else 1,
        "correctness_vector": ([True, False, True, False] if informative else [True] * 4),
        "correctness_count": 2 if informative else 4,
        "extraction_count": 4,
        "compliant_format_count": 4,
        "malformed_count": 0,
        "prompt_echo_count": 0,
        "question_generation_count": 0,
        "truncation_count": 0,
        "backend_failure_count": 0,
        "valid_completion_token_counts": [2, 2, 2, 2],
        "policy_logprobs_finite": True,
        "reference_logprobs_finite": True,
        "policy_reference_kl": {"finite": True},
        "reward_contract_consistent": True,
        "zero_variance_classification": (None if informative else "all_correct_saturated"),
    }
    row["group_record_sha256"] = canonical_sha256(row)
    return row


def _raw(arm: str, informative_positions: set[int]) -> dict[str, Any]:
    families = ("bookkeeping", "rate_ratio", "discrete")
    groups = [
        _group(
            arm=arm,
            position=position,
            informative=position in informative_positions,
            family=families[(position - 1) // 8] if position <= 24 else "replay",
        )
        for position in range(1, GROUPS_PER_ARM + 1)
    ]
    raw: dict[str, Any] = {
        "audit_id": "foundry-l3-grpo-signal-audit-v1",
        "arm": arm,
        "group_count": GROUPS_PER_ARM,
        "completion_count": COMPLETIONS_PER_ARM,
        "groups": groups,
        "optimizer_created": False,
        "backward_calls": 0,
        "scheduler_created": False,
        "adapter_saved": False,
    }
    raw["raw_audit_sha256"] = canonical_sha256(raw)
    return raw


def test_method_contract_reconstructs_and_freezes_scope() -> None:
    contract = signal_audit_method_contract()
    supplied = contract["method_contract_sha256"]
    payload = dict(contract)
    payload.pop("method_contract_sha256")
    assert supplied == canonical_sha256(payload)
    assert contract["scope"] == {
        "arms": ["generic", "targeted"],
        "groups_per_arm": 32,
        "task_groups_per_arm": 24,
        "replay_groups_per_arm": 8,
        "completions_per_group": 4,
        "completions_per_arm": 128,
        "total_groups": 64,
        "total_completions": 256,
    }


def test_zero_variance_classification_precedence_is_mutually_exclusive() -> None:
    assert (
        classify_zero_variance_group(
            reward_variance=0.0,
            completion_sha256s=["a"] * 4,
            reward_vector=[1.0] * 4,
            reward_component_vectors=_components(),
            correctness=[True] * 4,
        )
        == "output_identical"
    )
    assert (
        classify_zero_variance_group(
            reward_variance=0.0,
            completion_sha256s=["a", "b", "c", "d"],
            reward_vector=[1.0] * 4,
            reward_component_vectors=_components(),
            correctness=[True] * 4,
        )
        == "all_correct_saturated"
    )
    assert (
        classify_zero_variance_group(
            reward_variance=0.0,
            completion_sha256s=["a", "b", "c", "d"],
            reward_vector=[0.0] * 4,
            reward_component_vectors=_components(),
            correctness=[False] * 4,
        )
        == "all_incorrect_saturated"
    )
    assert (
        classify_zero_variance_group(
            reward_variance=0.0,
            completion_sha256s=["a", "b", "c", "d"],
            reward_vector=[0.0] * 4,
            reward_component_vectors=_components(),
            correctness=[True, False, True, False],
        )
        == "output_diverse_reward_indistinguishable"
    )


@pytest.mark.parametrize(
    ("variance", "rewards", "complete"),
    (
        (float("nan"), [1.0] * 4, True),
        (0.0, [1.0, 1.0, 1.0, 2.0], True),
        (0.0, [1.0] * 4, False),
        (0.25, [1.0, float("nan"), 1.0, 2.0], True),
    ),
)
def test_invalid_zero_variance_evidence_is_fail_closed(
    variance: float,
    rewards: list[float],
    complete: bool,
) -> None:
    assert (
        classify_zero_variance_group(
            reward_variance=variance,
            completion_sha256s=["a", "b", "c", "d"],
            reward_vector=rewards,
            reward_component_vectors=_components(),
            correctness=[True] * 4,
            evidence_complete=complete,
        )
        == "invalid_or_ambiguous"
    )


def test_nonzero_variance_is_not_given_a_saturation_class() -> None:
    assert (
        classify_zero_variance_group(
            reward_variance=0.25,
            completion_sha256s=["a", "b", "c", "d"],
            reward_vector=[0.0, 1.0, 0.0, 1.0],
            reward_component_vectors=_components(),
            correctness=[True, False, True, False],
        )
        is None
    )


def test_quantitative_gate_requires_four_groups_across_two_families_per_arm() -> None:
    passed = build_signal_summary(
        _raw("generic", {1, 2, 9, 10}),
        _raw("targeted", {1, 2, 9, 10}),
    )
    assert passed["quantitative_viability_passed"] is True
    assert passed["decision"] == "deterministic_replay_required"
    failed = build_signal_summary(
        _raw("generic", {1, 2, 3, 4}),
        _raw("targeted", {1, 2, 9}),
    )
    assert failed["quantitative_viability_passed"] is False
    assert failed["decision"] == "reward_signal_insufficient"


def test_deterministic_replay_completes_the_viability_gate() -> None:
    generic = _raw("generic", {1, 2, 9, 10})
    targeted = _raw("targeted", {1, 2, 9, 10})
    replays = {arm: {"passed": True, "group_id": f"{arm}-01"} for arm in ("generic", "targeted")}
    result = build_signal_summary(
        generic,
        targeted,
        deterministic_replays=replays,
    )
    assert result["viability_passed"] is True
    assert result["decision"] == "schedule_viable"
    overlap = result["generic_targeted_overlap"]
    assert overlap["shared_prompt_position_count"] == 32
    assert overlap["groups_informative_in_both"] == 4
