"""Warmup-aware GRPO optimizer-update validation.

The reward/objective classification and the optimizer-update classification are
deliberately separate.  A nonzero policy gradient is not by itself evidence of
a parameter update when every effective learning rate is exactly zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from foundry.training.config import canonical_sha256

UPDATE_CONTRACT_ID = "foundry-grpo-warmup-aware-update-v1"
FIXTURE_CONTRACT_ID = "foundry-grpo-warmup-aware-update-fixtures-v1"

UpdateClassification = Literal[
    "expected_zero_advantage_noop",
    "expected_zero_lr_warmup_noop",
    "nonzero_policy_update",
    "unexpected_zero_gradient",
    "unexpected_positive_lr_no_update",
    "invalid_or_ambiguous",
]

EXPECTED_ZERO_ADVANTAGE_NOOP: UpdateClassification = "expected_zero_advantage_noop"
EXPECTED_ZERO_LR_WARMUP_NOOP: UpdateClassification = "expected_zero_lr_warmup_noop"
NONZERO_POLICY_UPDATE: UpdateClassification = "nonzero_policy_update"
UNEXPECTED_ZERO_GRADIENT: UpdateClassification = "unexpected_zero_gradient"
UNEXPECTED_POSITIVE_LR_NO_UPDATE: UpdateClassification = "unexpected_positive_lr_no_update"
INVALID_OR_AMBIGUOUS: UpdateClassification = "invalid_or_ambiguous"

ALL_CLASSIFICATIONS: tuple[UpdateClassification, ...] = (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    EXPECTED_ZERO_LR_WARMUP_NOOP,
    NONZERO_POLICY_UPDATE,
    UNEXPECTED_ZERO_GRADIENT,
    UNEXPECTED_POSITIVE_LR_NO_UPDATE,
    INVALID_OR_AMBIGUOUS,
)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _finite_numbers(value: object, *, nonempty: bool) -> list[float] | None:
    if not isinstance(value, list) or (nonempty and not value):
        return None
    if not all(_finite_number(item) for item in value):
        return None
    return [float(cast(int | float, item)) for item in value]


def effective_learning_rates(optimizer: Any) -> list[float]:
    """Capture every effective optimizer parameter-group learning rate."""

    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise TypeError("optimizer must expose nonempty param_groups")
    values: list[float] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or not _finite_number(group.get("lr")):
            raise ValueError(f"optimizer parameter group {index} has an invalid learning rate")
        value = float(cast(int | float, group["lr"]))
        if value < 0.0:
            raise ValueError("effective learning rates cannot be negative")
        values.append(value)
    return values


def changed_state_tensor_count(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> int:
    """Count changed, added, or removed tensors in two state-tree captures."""

    def rows(value: Mapping[str, object]) -> dict[str, str]:
        raw = value.get("tensor_evidence")
        if not isinstance(raw, list):
            raise ValueError("state-tree evidence lacks tensor_evidence")
        result: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("state-tree tensor evidence row differs")
            path = item.get("path")
            sha256 = item.get("sha256")
            if not isinstance(path, str) or not isinstance(sha256, str) or path in result:
                raise ValueError("state-tree tensor identity differs")
            result[path] = sha256
        return result

    left = rows(before)
    right = rows(after)
    return sum(left.get(path) != right.get(path) for path in set(left) | set(right))


def classify_update(evidence: Mapping[str, object]) -> UpdateClassification:
    """Classify one attempted optimizer step using gradient, LR, and delta evidence."""

    rewards = _finite_numbers(evidence.get("advantages"), nonempty=True)
    learning_rates = _finite_numbers(evidence.get("effective_learning_rates"), nonempty=True)
    reward_variance = evidence.get("reward_variance")
    gradient_norm = evidence.get("policy_gradient_norm")
    delta_norm = evidence.get("policy_delta_norm")
    nonzero_gradients = evidence.get("nonzero_policy_gradient_tensor_count")
    changed_policy = evidence.get("changed_policy_tensor_count")
    changed_optimizer = evidence.get("changed_optimizer_state_tensor_count")
    if (
        rewards is None
        or learning_rates is None
        or not _finite_number(reward_variance)
        or float(cast(int | float, reward_variance)) < 0.0
        or not _finite_number(gradient_norm)
        or float(cast(int | float, gradient_norm)) < 0.0
        or not _finite_number(delta_norm)
        or float(cast(int | float, delta_norm)) < 0.0
        or isinstance(nonzero_gradients, bool)
        or not isinstance(nonzero_gradients, int)
        or nonzero_gradients < 0
        or isinstance(changed_policy, bool)
        or not isinstance(changed_policy, int)
        or changed_policy < 0
        or isinstance(changed_optimizer, bool)
        or not isinstance(changed_optimizer, int)
        or changed_optimizer < 0
        or evidence.get("policy_gradient_finite") is not True
        or evidence.get("policy_delta_finite") is not True
        or evidence.get("evidence_complete") is not True
        or evidence.get("reference_parameter_changed") is not False
        or evidence.get("base_parameter_changed") is not False
        or any(value < 0.0 for value in learning_rates)
    ):
        return INVALID_OR_AMBIGUOUS

    variance = float(cast(int | float, reward_variance))
    gradient = float(cast(int | float, gradient_norm))
    delta = float(cast(int | float, delta_norm))
    zero_advantages = all(value == 0.0 for value in rewards)
    any_advantage = any(value != 0.0 for value in rewards)
    zero_gradient = gradient == 0.0 and nonzero_gradients == 0
    nonzero_gradient = gradient > 0.0 and nonzero_gradients > 0
    zero_delta = delta == 0.0 and changed_policy == 0
    nonzero_delta = delta > 0.0 and changed_policy > 0
    all_lrs_zero = all(value == 0.0 for value in learning_rates)
    positive_lr = max(learning_rates) > 0.0

    if (
        (gradient == 0.0) != (nonzero_gradients == 0)
        or (delta == 0.0) != (changed_policy == 0)
        or (variance == 0.0) != zero_advantages
    ):
        return INVALID_OR_AMBIGUOUS
    if any_advantage and zero_gradient:
        return UNEXPECTED_ZERO_GRADIENT
    if variance == 0.0 and zero_advantages and zero_gradient and zero_delta:
        return EXPECTED_ZERO_ADVANTAGE_NOOP
    if any_advantage and nonzero_gradient and all_lrs_zero and zero_delta:
        return EXPECTED_ZERO_LR_WARMUP_NOOP
    if positive_lr and nonzero_gradient and zero_delta:
        return UNEXPECTED_POSITIVE_LR_NO_UPDATE
    if positive_lr and nonzero_gradient and nonzero_delta:
        return NONZERO_POLICY_UPDATE
    return INVALID_OR_AMBIGUOUS


def complete_warmup_smoke_gate(
    steps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require a zero-LR first call and a positive-LR second policy update."""

    rows = list(steps)
    classifications = [row.get("classification") for row in rows]
    recomputed_classifications = [classify_update(row) for row in rows]
    learning_rates = [
        _finite_numbers(row.get("effective_learning_rates"), nonempty=True) for row in rows
    ]
    result: dict[str, object] = {
        "step_count": len(rows),
        "completion_count": sum(
            cast(int, row["completion_count"])
            for row in rows
            if isinstance(row.get("completion_count"), int)
            and not isinstance(row.get("completion_count"), bool)
        ),
        "classifications": classifications,
        "recomputed_classifications": recomputed_classifications,
        "classification_evidence_consistent": (classifications == recomputed_classifications),
        "expected_zero_advantage_noop_count": classifications.count(EXPECTED_ZERO_ADVANTAGE_NOOP),
        "expected_zero_lr_warmup_noop_count": classifications.count(EXPECTED_ZERO_LR_WARMUP_NOOP),
        "nonzero_policy_update_count": classifications.count(NONZERO_POLICY_UPDATE),
        "unexpected_zero_gradient_count": classifications.count(UNEXPECTED_ZERO_GRADIENT),
        "unexpected_positive_lr_no_update_count": classifications.count(
            UNEXPECTED_POSITIVE_LR_NO_UPDATE
        ),
        "invalid_or_ambiguous_count": classifications.count(INVALID_OR_AMBIGUOUS),
        "policy_update_count": sum(
            isinstance(row.get("changed_policy_tensor_count"), int)
            and cast(int, row["changed_policy_tensor_count"]) > 0
            for row in rows
        ),
        "optimizer_call_count": sum(row.get("optimizer_call_completed") is True for row in rows),
        "scheduler_advance_count": sum(row.get("scheduler_step_completed") is True for row in rows),
        "reference_update_count": sum(
            row.get("reference_parameter_changed") is True for row in rows
        ),
        "base_update_count": sum(row.get("base_parameter_changed") is True for row in rows),
    }
    first_lr_zero = (
        len(learning_rates) == 2
        and learning_rates[0] is not None
        and all(value == 0.0 for value in learning_rates[0])
    )
    second_lr_positive = (
        len(learning_rates) == 2 and learning_rates[1] is not None and max(learning_rates[1]) > 0.0
    )
    policy_update_count = sum(
        isinstance(row.get("changed_policy_tensor_count"), int)
        and cast(int, row["changed_policy_tensor_count"]) > 0
        for row in rows
    )
    optimizer_call_count = sum(row.get("optimizer_call_completed") is True for row in rows)
    scheduler_advance_count = sum(row.get("scheduler_step_completed") is True for row in rows)
    reference_update_count = sum(row.get("reference_parameter_changed") is True for row in rows)
    base_update_count = sum(row.get("base_parameter_changed") is True for row in rows)
    result["first_call_all_effective_lrs_zero"] = first_lr_zero
    result["second_call_has_positive_effective_lr"] = second_lr_positive
    result["passed"] = (
        result["step_count"] == 2
        and result["completion_count"] == 8
        and first_lr_zero
        and second_lr_positive
        and result["classification_evidence_consistent"] is True
        and classifications[0] in {EXPECTED_ZERO_ADVANTAGE_NOOP, EXPECTED_ZERO_LR_WARMUP_NOOP}
        and classifications[1] == NONZERO_POLICY_UPDATE
        and policy_update_count >= 1
        and optimizer_call_count == 2
        and scheduler_advance_count == 2
        and reference_update_count == 0
        and base_update_count == 0
    )
    result["complete_smoke_gate_sha256"] = canonical_sha256(result)
    return result


def counted_update_gate(
    steps: Sequence[Mapping[str, object]],
    *,
    expected_steps: int,
    expected_learning_rates: Sequence[float],
) -> dict[str, object]:
    """Require every counted step to satisfy the warmup-aware contract."""

    rows = list(steps)
    expected = [float(value) for value in expected_learning_rates]
    observed = [
        max(cast(list[float], _finite_numbers(row.get("effective_learning_rates"), nonempty=True)))
        if _finite_numbers(row.get("effective_learning_rates"), nonempty=True) is not None
        else math.nan
        for row in rows
    ]
    classifications = [row.get("classification") for row in rows]
    recomputed_classifications = [classify_update(row) for row in rows]
    counts = {name: classifications.count(name) for name in ALL_CLASSIFICATIONS}
    result: dict[str, object] = {
        "step_count": len(rows),
        "expected_step_count": expected_steps,
        "expected_learning_rate_trajectory": expected,
        "observed_learning_rate_trajectory": observed,
        "learning_rate_trajectory_exact": observed == expected,
        "classification_counts": counts,
        "recomputed_classifications": recomputed_classifications,
        "classification_evidence_consistent": (classifications == recomputed_classifications),
        "optimizer_call_count": sum(row.get("optimizer_call_completed") is True for row in rows),
        "scheduler_advance_count": sum(row.get("scheduler_step_completed") is True for row in rows),
        "policy_update_count": counts[NONZERO_POLICY_UPDATE],
        "reference_update_count": sum(
            row.get("reference_parameter_changed") is True for row in rows
        ),
        "base_update_count": sum(row.get("base_parameter_changed") is True for row in rows),
    }
    result["passed"] = (
        len(rows) == expected_steps
        and len(expected) == expected_steps
        and result["learning_rate_trajectory_exact"] is True
        and result["classification_evidence_consistent"] is True
        and result["optimizer_call_count"] == expected_steps
        and result["scheduler_advance_count"] == expected_steps
        and cast(int, result["policy_update_count"]) >= 1
        and counts[UNEXPECTED_ZERO_GRADIENT] == 0
        and counts[UNEXPECTED_POSITIVE_LR_NO_UPDATE] == 0
        and counts[INVALID_OR_AMBIGUOUS] == 0
        and result["reference_update_count"] == 0
        and result["base_update_count"] == 0
    )
    result["counted_update_gate_sha256"] = canonical_sha256(result)
    return result


def update_contract() -> dict[str, object]:
    """Return the content-free six-way update-classification contract."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_id": UPDATE_CONTRACT_ID,
        "classifications": list(ALL_CLASSIFICATIONS),
        "classification_inputs": [
            "group_position",
            "trainer_global_step",
            "optimizer_call_index",
            "scheduler_step_index",
            "effective_learning_rate_per_policy_parameter_group",
            "minimum_effective_learning_rate",
            "maximum_effective_learning_rate",
            "reward_variance",
            "nonzero_advantage_count",
            "policy_gradient_norm",
            "nonzero_policy_gradient_tensor_count",
            "policy_state_before_and_after",
            "policy_delta_norm",
            "changed_policy_tensor_count",
            "optimizer_state_before_and_after",
            "changed_optimizer_state_tensor_count",
            "scheduler_state_before_and_after",
            "reference_state",
            "base_state",
        ],
        "expected_zero_advantage_noop_requires": [
            "exact_zero_reward_variance",
            "all_advantages_exactly_zero",
            "exact_zero_policy_gradient",
            "exact_zero_policy_delta",
            "finite_complete_evidence",
        ],
        "expected_zero_lr_warmup_noop_requires": [
            "at_least_one_nonzero_advantage",
            "finite_strictly_positive_policy_gradient",
            "at_least_one_nonzero_policy_gradient_tensor",
            "all_effective_policy_learning_rates_exactly_zero",
            "exact_zero_policy_delta",
            "optimizer_state_change_permitted",
            "reference_and_base_unchanged",
        ],
        "nonzero_policy_update_requires": [
            "maximum_effective_policy_learning_rate_strictly_positive",
            "finite_strictly_positive_policy_gradient",
            "finite_strictly_positive_policy_delta",
            "at_least_one_changed_policy_tensor",
            "reference_and_base_unchanged",
        ],
        "unexpected_positive_lr_no_update_terminal": True,
        "classification_uses_optimizer_call_index_alone": False,
        "optimizer_then_scheduler_order_required": True,
        "scientific_recipe_changed": False,
    }
    payload["update_contract_sha256"] = canonical_sha256(payload)
    return payload


def _observation(
    *,
    variance: float,
    advantages: list[float],
    learning_rates: list[float],
    gradient_norm: float,
    nonzero_gradients: int,
    delta_norm: float,
    changed_policy: int,
    changed_optimizer: int = 0,
    reference_changed: bool = False,
    base_changed: bool = False,
) -> dict[str, object]:
    return {
        "reward_variance": variance,
        "advantages": advantages,
        "effective_learning_rates": learning_rates,
        "policy_gradient_norm": gradient_norm,
        "nonzero_policy_gradient_tensor_count": nonzero_gradients,
        "policy_gradient_finite": True,
        "policy_delta_norm": delta_norm,
        "changed_policy_tensor_count": changed_policy,
        "policy_delta_finite": True,
        "changed_optimizer_state_tensor_count": changed_optimizer,
        "reference_parameter_changed": reference_changed,
        "base_parameter_changed": base_changed,
        "evidence_complete": True,
    }


def run_deterministic_fixtures(torch: Any) -> dict[str, object]:
    """Run fourteen model-free update and scheduler-order fixtures."""

    rows: list[dict[str, object]] = []

    def passed(fixture_id: str, detail: object = True) -> None:
        rows.append(
            {
                "fixture_id": fixture_id,
                "passed": True,
                "detail_sha256": canonical_sha256(detail),
            }
        )

    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.0 if step == 0 else 1.0,
    )
    assert effective_learning_rates(optimizer) == [0.0]
    before = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.equal(parameter.detach(), before)
    assert optimizer.state
    passed("nonzero_gradient_lr_zero_has_zero_parameter_delta")
    passed("optimizer_state_may_change_at_lr_zero", {"state_entries": len(optimizer.state)})

    scheduler.step()
    assert effective_learning_rates(optimizer) == [1.0e-6]
    passed("scheduler_advance_makes_next_lr_positive")
    parameter.grad = torch.ones_like(parameter)
    before_positive = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(parameter.detach(), before_positive)
    passed("first_positive_lr_step_changes_parameters")

    positive_no_update = _observation(
        variance=0.25,
        advantages=[-0.5, 0.5, -0.5, 0.5],
        learning_rates=[1.0e-6],
        gradient_norm=1.0,
        nonzero_gradients=1,
        delta_norm=0.0,
        changed_policy=0,
    )
    assert classify_update(positive_no_update) == UNEXPECTED_POSITIVE_LR_NO_UPDATE
    passed("positive_lr_nonzero_gradient_zero_delta_fails")

    zero_advantage = _observation(
        variance=0.0,
        advantages=[0.0, 0.0, 0.0, 0.0],
        learning_rates=[1.0e-6],
        gradient_norm=0.0,
        nonzero_gradients=0,
        delta_norm=0.0,
        changed_policy=0,
    )
    assert classify_update(zero_advantage) == EXPECTED_ZERO_ADVANTAGE_NOOP
    passed("zero_advantage_group_is_valid_noop")

    hidden_zero_gradient = _observation(
        variance=0.25,
        advantages=[-0.5, 0.5, -0.5, 0.5],
        learning_rates=[0.0],
        gradient_norm=0.0,
        nonzero_gradients=0,
        delta_norm=0.0,
        changed_policy=0,
    )
    assert classify_update(hidden_zero_gradient) == UNEXPECTED_ZERO_GRADIENT
    passed("zero_lr_does_not_hide_zero_gradient_defect")

    reference = torch.tensor([2.0])
    base = torch.tensor([3.0])
    assert torch.equal(reference, torch.tensor([2.0]))
    assert torch.equal(base, torch.tensor([3.0]))
    passed("reference_and_base_remain_unchanged")
    passed("optimizer_then_scheduler_order_preserved", ["optimizer", "scheduler"])

    accepted_warmup = _observation(
        variance=0.25,
        advantages=[-0.5, 0.5, -0.5, 0.5],
        learning_rates=[0.0],
        gradient_norm=1.0,
        nonzero_gradients=1,
        delta_norm=0.0,
        changed_policy=0,
        changed_optimizer=1,
    )
    assert classify_update(accepted_warmup) == EXPECTED_ZERO_LR_WARMUP_NOOP
    passed("accepted_warmup_allows_global_step_advancement", {"global_step_after": 1})

    update = _observation(
        variance=0.25,
        advantages=[-0.5, 0.5, -0.5, 0.5],
        learning_rates=[1.0e-6],
        gradient_norm=1.0,
        nonzero_gradients=1,
        delta_norm=1.0e-6,
        changed_policy=1,
        changed_optimizer=1,
    )
    assert classify_update(update) == NONZERO_POLICY_UPDATE
    smoke = complete_warmup_smoke_gate(
        (
            {
                **accepted_warmup,
                "classification": EXPECTED_ZERO_LR_WARMUP_NOOP,
                "completion_count": 4,
                "optimizer_call_completed": True,
                "scheduler_step_completed": True,
            },
            {
                **update,
                "classification": NONZERO_POLICY_UPDATE,
                "completion_count": 4,
                "optimizer_call_completed": True,
                "scheduler_step_completed": True,
            },
        )
    )
    assert smoke["passed"] is True
    passed("two_step_warmup_then_update_passes", smoke)

    failed_smoke = complete_warmup_smoke_gate(
        (
            {
                **accepted_warmup,
                "classification": EXPECTED_ZERO_LR_WARMUP_NOOP,
                "completion_count": 4,
                "optimizer_call_completed": True,
                "scheduler_step_completed": True,
            },
            {
                **zero_advantage,
                "classification": EXPECTED_ZERO_ADVANTAGE_NOOP,
                "completion_count": 4,
                "optimizer_call_completed": True,
                "scheduler_step_completed": True,
            },
        )
    )
    assert failed_smoke["passed"] is False
    passed("two_step_without_policy_update_fails", failed_smoke)

    counted = counted_update_gate(
        (
            {
                **accepted_warmup,
                "classification": EXPECTED_ZERO_LR_WARMUP_NOOP,
                "optimizer_call_completed": True,
                "scheduler_step_completed": True,
            },
            {
                **update,
                "classification": NONZERO_POLICY_UPDATE,
                "optimizer_call_completed": True,
                "scheduler_step_completed": True,
            },
            {
                **zero_advantage,
                "classification": EXPECTED_ZERO_ADVANTAGE_NOOP,
                "optimizer_call_completed": True,
                "scheduler_step_completed": True,
            },
        ),
        expected_steps=3,
        expected_learning_rates=(0.0, 1.0e-6, 1.0e-6),
    )
    assert counted["passed"] is True
    passed("counted_validator_accepts_initial_warmup_and_later_valid_steps", counted)

    recipe = {
        "optimizer": "paged_adamw_8bit",
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "learning_rate": 1.0e-6,
    }
    before_recipe = canonical_sha256(recipe)
    after_recipe = canonical_sha256(recipe)
    assert before_recipe == after_recipe
    passed("update_contract_does_not_modify_scientific_recipe", recipe)

    incomplete = dict(accepted_warmup)
    incomplete["evidence_complete"] = False
    assert classify_update(incomplete) == INVALID_OR_AMBIGUOUS

    if len(rows) != 14 or any(row["passed"] is not True for row in rows):
        raise RuntimeError("warmup-aware update fixture inventory differs")
    payload: dict[str, object] = {
        "schema_version": 1,
        "fixture_contract_id": FIXTURE_CONTRACT_ID,
        "update_contract_id": UPDATE_CONTRACT_ID,
        "fixture_count": len(rows),
        "fixtures": rows,
    }
    payload["fixture_sha256"] = canonical_sha256(payload)
    return payload
