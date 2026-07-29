"""General zero-gradient classification for verifier-GRPO groups.

This module is deliberately model-independent.  It freezes the mathematical
conditions that distinguish an expected group-relative no-op from a broken
policy computation graph, and provides deterministic tensor fixtures for the
Milestone 14A-R1 adjudication.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from foundry.training.config import canonical_sha256
from foundry.training.grpo_replay_evidence import tensor_evidence

CLASSIFICATION_CONTRACT_ID = "foundry-grpo-zero-gradient-classification-v1"
FIXTURE_CONTRACT_ID = "foundry-grpo-zero-gradient-fixtures-v1"
Classification = Literal[
    "expected_zero_advantage_noop",
    "unexpected_zero_gradient",
    "nonzero_gradient_update",
    "invalid_or_ambiguous",
]

EXPECTED_ZERO_ADVANTAGE_NOOP: Classification = "expected_zero_advantage_noop"
UNEXPECTED_ZERO_GRADIENT: Classification = "unexpected_zero_gradient"
NONZERO_GRADIENT_UPDATE: Classification = "nonzero_gradient_update"
INVALID_OR_AMBIGUOUS: Classification = "invalid_or_ambiguous"


@dataclass(frozen=True)
class ObjectiveComponents:
    """Exact stock-TRL DR-GRPO objective components."""

    policy: Any
    kl: Any
    combined: Any
    per_token_kl: Any


def _is_finite_tensor(torch: Any, value: Any) -> bool:
    return bool(torch.isfinite(value).all().item())


def reward_projection(torch: Any, rewards: Sequence[float]) -> dict[str, object]:
    """Project four frozen rewards to unscaled group-relative advantages."""

    if len(rewards) != 4:
        raise ValueError("GRPO reward projection requires exactly four completions")
    tensor = torch.tensor(list(rewards), dtype=torch.float32)
    if not _is_finite_tensor(torch, tensor):
        raise ValueError("completion rewards must be finite")
    mean = tensor.mean()
    advantages = tensor - mean
    if not _is_finite_tensor(torch, advantages):
        raise ValueError("group-relative advantages must be finite")
    variance = ((tensor - mean) ** 2).mean()
    result: dict[str, object] = {
        "rewards": [float(value) for value in tensor.tolist()],
        "reward_mean": float(mean.item()),
        "reward_variance": float(variance.item()),
        "advantages": [float(value) for value in advantages.tolist()],
        "normalized_advantages": [float(value) for value in advantages.tolist()],
        "reward_scaling": False,
    }
    result["reward_projection_sha256"] = canonical_sha256(result)
    return result


def objective_components(
    torch: Any,
    *,
    policy_logprobs: Any,
    reference_logprobs: Any,
    advantages: Any,
    completion_mask: Any,
    beta: float = 0.04,
    epsilon: float = 0.2,
    max_completion_length: int = 256,
) -> ObjectiveComponents:
    """Reconstruct TRL 0.17's num-iterations-one DR-GRPO objective exactly."""

    if policy_logprobs.shape != reference_logprobs.shape:
        raise ValueError("policy and reference log-probability shapes differ")
    if policy_logprobs.shape != completion_mask.shape:
        raise ValueError("completion mask shape differs from token log probabilities")
    if policy_logprobs.ndim != 2 or advantages.ndim != 1:
        raise ValueError("GRPO objective requires token matrices and one advantage vector")
    if policy_logprobs.shape[0] != advantages.shape[0]:
        raise ValueError("advantage count differs from completion count")
    if not bool(policy_logprobs.requires_grad):
        raise ValueError("policy log probabilities are detached from autograd")
    if max_completion_length <= 0 or beta < 0.0 or epsilon <= 0.0:
        raise ValueError("GRPO objective constants are invalid")
    if not all(
        _is_finite_tensor(torch, value)
        for value in (policy_logprobs, reference_logprobs, advantages, completion_mask)
    ):
        raise ValueError("GRPO objective inputs must be finite")
    valid_counts = completion_mask.sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("every completion must have at least one valid token")

    old_policy_logprobs = policy_logprobs.detach()
    coefficient = torch.exp(policy_logprobs - old_policy_logprobs)
    clipped_coefficient = torch.clamp(coefficient, 1.0 - epsilon, 1.0 + epsilon)
    policy_token_loss = -torch.min(
        coefficient * advantages.unsqueeze(1),
        clipped_coefficient * advantages.unsqueeze(1),
    )
    delta = reference_logprobs - policy_logprobs
    per_token_kl = torch.exp(delta) - delta - 1.0
    denominator = policy_token_loss.size(0) * max_completion_length
    policy_objective = (policy_token_loss * completion_mask).sum() / denominator
    kl_objective = beta * (per_token_kl * completion_mask).sum() / denominator
    combined = policy_objective + kl_objective
    if not all(
        _is_finite_tensor(torch, value)
        for value in (policy_objective, kl_objective, combined, per_token_kl)
    ):
        raise ValueError("GRPO objective outputs must be finite")
    return ObjectiveComponents(
        policy=policy_objective,
        kl=kl_objective,
        combined=combined,
        per_token_kl=per_token_kl,
    )


def tensor_graph_evidence(value: Any) -> dict[str, object]:
    """Record graph connectivity without retaining tensor values."""

    grad_fn = getattr(value, "grad_fn", None)
    return {
        "requires_grad": bool(getattr(value, "requires_grad", False)),
        "grad_fn": None if grad_fn is None else type(grad_fn).__name__,
    }


def gradient_projection(
    torch: Any,
    *,
    objective: Any,
    named_parameters: Sequence[tuple[str, Any]],
    retain_graph: bool,
) -> dict[str, object]:
    """Project one objective onto policy tensors without populating ``.grad``."""

    rows = tuple(named_parameters)
    if not rows:
        raise ValueError("gradient projection requires policy parameters")
    if any(not bool(parameter.requires_grad) for _, parameter in rows):
        raise ValueError("gradient projection received a frozen policy parameter")
    if not bool(getattr(objective, "requires_grad", False)):
        raise ValueError("objective is detached from autograd")
    gradients = torch.autograd.grad(
        objective,
        [parameter for _, parameter in rows],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    tensor_rows: list[dict[str, object]] = []
    squared_norms: list[float] = []
    finite = True
    nonzero_count = 0
    present_count = 0
    for (name, _), gradient in zip(rows, gradients, strict=True):
        if gradient is None:
            tensor_rows.append({"name": name, "present": False})
            continue
        detached = gradient.detach()
        present_count += 1
        is_finite = _is_finite_tensor(torch, detached)
        finite = finite and is_finite
        nonzero = bool(torch.count_nonzero(detached).item())
        nonzero_count += int(nonzero)
        norm = float(torch.linalg.vector_norm(detached.float()).item())
        squared_norms.append(norm * norm)
        evidence = tensor_evidence(detached).as_dict()
        tensor_rows.append(
            {
                "name": name,
                "present": True,
                "finite": is_finite,
                "nonzero": nonzero,
                "norm": norm,
                "sha256": evidence["sha256"],
            }
        )
    payload: dict[str, object] = {
        "parameter_count": len(rows),
        "present_gradient_count": present_count,
        "missing_gradient_count": len(rows) - present_count,
        "finite": finite,
        "nonzero_gradient_count": nonzero_count,
        "exactly_zero": present_count == len(rows) and nonzero_count == 0 and finite,
        "graph_connected": present_count == len(rows),
        "global_norm": math.sqrt(math.fsum(squared_norms)),
        "tensors": tensor_rows,
    }
    payload["gradient_projection_sha256"] = canonical_sha256(payload)
    return payload


def populated_gradient_projection(
    torch: Any,
    *,
    named_policy_parameters: Sequence[tuple[str, Any]],
    named_reference_parameters: Sequence[tuple[str, Any]],
    named_base_parameters: Sequence[tuple[str, Any]],
) -> dict[str, object]:
    """Capture populated gradients after the combined backward pass."""

    def rows(values: Sequence[tuple[str, Any]]) -> tuple[list[dict[str, object]], int, int, bool]:
        result: list[dict[str, object]] = []
        present = 0
        nonzero = 0
        finite = True
        for name, parameter in values:
            gradient = parameter.grad
            if gradient is None:
                result.append({"name": name, "present": False})
                continue
            detached = gradient.detach()
            present += 1
            item_finite = _is_finite_tensor(torch, detached)
            finite = finite and item_finite
            item_nonzero = bool(torch.count_nonzero(detached).item())
            nonzero += int(item_nonzero)
            evidence = tensor_evidence(detached).as_dict()
            result.append(
                {
                    "name": name,
                    "present": True,
                    "finite": item_finite,
                    "nonzero": item_nonzero,
                    "norm": float(torch.linalg.vector_norm(detached.float()).item()),
                    "sha256": evidence["sha256"],
                }
            )
        return result, present, nonzero, finite

    policy_rows, policy_present, policy_nonzero, policy_finite = rows(named_policy_parameters)
    reference_rows, reference_present, reference_nonzero, reference_finite = rows(
        named_reference_parameters
    )
    base_rows, base_present, base_nonzero, base_finite = rows(named_base_parameters)
    norms = [
        cast(float, row["norm"])
        for row in policy_rows
        if row.get("present") is True and isinstance(row.get("norm"), float)
    ]
    payload: dict[str, object] = {
        "parameter_count": len(named_policy_parameters),
        "present_gradient_count": policy_present,
        "missing_gradient_count": len(named_policy_parameters) - policy_present,
        "finite": policy_finite and reference_finite and base_finite,
        "nonzero_gradient_count": policy_nonzero,
        "exactly_zero": (
            policy_present == len(named_policy_parameters) and policy_nonzero == 0 and policy_finite
        ),
        "graph_connected": policy_present == len(named_policy_parameters),
        "global_norm": math.sqrt(math.fsum(value * value for value in norms)),
        "reference_gradient_count": reference_present,
        "reference_nonzero_gradient_count": reference_nonzero,
        "base_gradient_count": base_present,
        "base_nonzero_gradient_count": base_nonzero,
        "policy_tensors": policy_rows,
        "reference_tensors": reference_rows,
        "base_tensors": base_rows,
    }
    payload["gradient_projection_sha256"] = canonical_sha256(payload)
    return payload


def _finite_numbers(values: object) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(
            not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
            for value in values
        )
    )


def _projection(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def classify_group(evidence: Mapping[str, object]) -> Classification:
    """Return exactly one frozen zero-gradient classification."""

    rewards = evidence.get("rewards")
    advantages = evidence.get("advantages")
    valid_counts = evidence.get("valid_completion_token_counts")
    policy_projection = _projection(evidence.get("policy_gradient"))
    kl_projection = _projection(evidence.get("kl_gradient"))
    combined_projection = _projection(evidence.get("combined_gradient"))
    if (
        not _finite_numbers(rewards)
        or not _finite_numbers(advantages)
        or not isinstance(valid_counts, list)
        or len(valid_counts) != len(cast(list[object], rewards))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in valid_counts
        )
        or not isinstance(evidence.get("reward_variance"), int | float)
        or not math.isfinite(cast(float, evidence["reward_variance"]))
        or cast(float, evidence["reward_variance"]) < 0.0
        or evidence.get("policy_logprobs_finite") is not True
        or evidence.get("reference_logprobs_finite") is not True
        or evidence.get("kl_finite") is not True
        or evidence.get("controlled_live_policy_fixture_passed") is not True
        or policy_projection is None
        or kl_projection is None
        or combined_projection is None
        or evidence.get("base_gradient_count") != 0
        or evidence.get("reference_gradient_count") != 0
        or evidence.get("optimizer_owned_tensor_count")
        != evidence.get("requires_grad_policy_tensor_count")
    ):
        return INVALID_OR_AMBIGUOUS
    projections = (policy_projection, kl_projection, combined_projection)
    if any(item.get("finite") is not True for item in projections):
        return INVALID_OR_AMBIGUOUS
    reward_variance = float(cast(float, evidence["reward_variance"]))
    advantage_values = cast(list[int | float], advantages)
    zero_advantages = all(float(value) == 0.0 for value in advantage_values)
    any_advantage = any(float(value) != 0.0 for value in advantage_values)
    combined_zero = combined_projection.get("exactly_zero") is True
    combined_nonzero = combined_projection.get(
        "graph_connected"
    ) is True and combined_projection.get("nonzero_gradient_count") not in (None, 0)
    if (
        reward_variance == 0.0
        and zero_advantages
        and evidence.get("adapters_identical_at_step_start") is True
        and policy_projection.get("exactly_zero") is True
        and kl_projection.get("exactly_zero") is True
        and combined_zero
        and all(item.get("graph_connected") is True for item in projections)
    ):
        return EXPECTED_ZERO_ADVANTAGE_NOOP
    if (reward_variance > 0.0 or any_advantage) and combined_zero:
        if (
            policy_projection.get("graph_connected") is True
            and combined_projection.get("graph_connected") is True
        ):
            return UNEXPECTED_ZERO_GRADIENT
        return INVALID_OR_AMBIGUOUS
    if combined_nonzero:
        return NONZERO_GRADIENT_UPDATE
    return INVALID_OR_AMBIGUOUS


def complete_smoke_gate(steps: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Require a complete two-group smoke to contain a real policy update."""

    classifications = [step.get("classification") for step in steps]
    completion_count = 0
    for step in steps:
        value = step.get("completion_count")
        if isinstance(value, int) and not isinstance(value, bool):
            completion_count += value
    result: dict[str, object] = {
        "step_count": len(steps),
        "completion_count": completion_count,
        "zero_variance_group_count": sum(step.get("reward_variance") == 0.0 for step in steps),
        "nonzero_variance_group_count": sum(
            isinstance(step.get("reward_variance"), int | float)
            and float(cast(float, step["reward_variance"])) > 0.0
            for step in steps
        ),
        "expected_noop_group_count": classifications.count(EXPECTED_ZERO_ADVANTAGE_NOOP),
        "nonzero_gradient_group_count": classifications.count(NONZERO_GRADIENT_UPDATE),
        "policy_update_count": sum(step.get("policy_parameter_changed") is True for step in steps),
        "optimizer_step_count": sum(step.get("optimizer_step_completed") is True for step in steps),
        "scheduler_step_count": sum(step.get("scheduler_step_completed") is True for step in steps),
        "reference_update_count": sum(
            step.get("reference_parameter_changed") is True for step in steps
        ),
        "base_update_count": sum(step.get("base_parameter_changed") is True for step in steps),
    }
    result["passed"] = (
        result["step_count"] == 2
        and result["completion_count"] == 8
        and cast(int, result["nonzero_variance_group_count"]) >= 1
        and cast(int, result["nonzero_gradient_group_count"]) >= 1
        and cast(int, result["policy_update_count"]) >= 1
        and result["optimizer_step_count"] == 2
        and result["scheduler_step_count"] == 2
        and result["reference_update_count"] == 0
        and result["base_update_count"] == 0
        and all(
            value in {EXPECTED_ZERO_ADVANTAGE_NOOP, NONZERO_GRADIENT_UPDATE}
            for value in classifications
        )
    )
    result["complete_smoke_gate_sha256"] = canonical_sha256(result)
    return result


def _fixture_observation(
    *,
    rewards: list[float],
    advantages: list[float],
    variance: float,
    policy: Mapping[str, object],
    kl: Mapping[str, object],
    combined: Mapping[str, object],
) -> dict[str, object]:
    return {
        "rewards": rewards,
        "reward_variance": variance,
        "advantages": advantages,
        "valid_completion_token_counts": [2, 2, 2, 2],
        "policy_logprobs_finite": True,
        "reference_logprobs_finite": True,
        "kl_finite": True,
        "adapters_identical_at_step_start": True,
        "controlled_live_policy_fixture_passed": True,
        "requires_grad_policy_tensor_count": 1,
        "optimizer_owned_tensor_count": 1,
        "base_gradient_count": 0,
        "reference_gradient_count": 0,
        "policy_gradient": dict(policy),
        "kl_gradient": dict(kl),
        "combined_gradient": dict(combined),
    }


def run_deterministic_fixtures(torch: Any) -> dict[str, object]:
    """Execute and hash all fifteen pre-model classification fixtures."""

    rows: list[dict[str, object]] = []

    def passed(fixture_id: str, detail: object = True) -> None:
        rows.append(
            {
                "fixture_id": fixture_id,
                "passed": True,
                "detail_sha256": canonical_sha256(detail),
            }
        )

    equal = reward_projection(torch, [1.25, 1.25, 1.25, 1.25])
    assert equal["advantages"] == [0.0, 0.0, 0.0, 0.0]
    passed("equal_rewards_zero_advantages", equal)

    policy = torch.zeros((4, 2), dtype=torch.float32, requires_grad=True)
    reference = policy.detach().clone()
    zero_advantages = torch.zeros(4, dtype=torch.float32)
    mask = torch.ones((4, 2), dtype=torch.float32)
    zero_components = objective_components(
        torch,
        policy_logprobs=policy,
        reference_logprobs=reference,
        advantages=zero_advantages,
        completion_mask=mask,
    )
    zero_policy = gradient_projection(
        torch,
        objective=zero_components.policy,
        named_parameters=(("policy", policy),),
        retain_graph=True,
    )
    zero_kl = gradient_projection(
        torch,
        objective=zero_components.kl,
        named_parameters=(("policy", policy),),
        retain_graph=True,
    )
    zero_combined = gradient_projection(
        torch,
        objective=zero_components.combined,
        named_parameters=(("policy", policy),),
        retain_graph=False,
    )
    assert zero_policy["exactly_zero"] and zero_kl["exactly_zero"] and zero_combined["exactly_zero"]
    passed("zero_advantages_identical_logits_zero_gradients")

    varied = reward_projection(torch, [0.0, 1.0, 0.0, 1.0])
    assert cast(float, varied["reward_variance"]) > 0.0 and any(
        value != 0.0 for value in cast(list[float], varied["advantages"])
    )
    passed("nonzero_variance_nonzero_advantages", varied)

    live_policy = torch.zeros((4, 2), dtype=torch.float32, requires_grad=True)
    live_advantages = torch.tensor(cast(list[float], varied["advantages"]), dtype=torch.float32)
    live_components = objective_components(
        torch,
        policy_logprobs=live_policy,
        reference_logprobs=live_policy.detach().clone(),
        advantages=live_advantages,
        completion_mask=mask,
    )
    live_projection = gradient_projection(
        torch,
        objective=live_components.combined,
        named_parameters=(("policy", live_policy),),
        retain_graph=False,
    )
    assert cast(int, live_projection["nonzero_gradient_count"]) >= 1
    passed("live_policy_nonzero_advantages_nonzero_gradient", live_projection)

    try:
        objective_components(
            torch,
            policy_logprobs=live_policy.detach(),
            reference_logprobs=live_policy.detach(),
            advantages=live_advantages,
            completion_mask=mask,
        )
    except ValueError as error:
        assert "detached" in str(error)
    else:
        raise AssertionError("detached policy log probabilities were accepted")
    passed("detached_policy_logprobs_rejected")

    for fixture_id, overrides, expected in (
        (
            "empty_completion_mask_rejected",
            {"completion_mask": torch.zeros_like(mask)},
            "valid token",
        ),
        (
            "nonfinite_advantages_rejected",
            {"advantages": torch.tensor([0.0, math.inf, 0.0, 0.0])},
            "finite",
        ),
        (
            "nonfinite_logprobs_rejected",
            {"policy_logprobs": torch.tensor([[math.nan, 0.0]] * 4, requires_grad=True)},
            "finite",
        ),
    ):
        values = {
            "policy_logprobs": torch.zeros((4, 2), requires_grad=True),
            "reference_logprobs": torch.zeros((4, 2)),
            "advantages": torch.zeros(4),
            "completion_mask": mask,
        }
        values.update(overrides)
        try:
            objective_components(torch, **values)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"{fixture_id} unexpectedly passed")
        passed(fixture_id)

    try:
        reward_projection(torch, [0.0, 0.0, math.nan, 0.0])
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite rewards were accepted")
    passed("nonfinite_rewards_rejected")

    perturbed = torch.full((4, 2), 0.25, requires_grad=True)
    perturbed_components = objective_components(
        torch,
        policy_logprobs=perturbed,
        reference_logprobs=torch.zeros((4, 2)),
        advantages=torch.zeros(4),
        completion_mask=mask,
    )
    perturbed_kl = gradient_projection(
        torch,
        objective=perturbed_components.kl,
        named_parameters=(("policy", perturbed),),
        retain_graph=False,
    )
    assert cast(int, perturbed_kl["nonzero_gradient_count"]) == 1
    passed("controlled_perturbation_nonzero_kl_gradient", perturbed_kl)

    policy_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    reference_parameter = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    base_parameter = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    (policy_parameter.square().sum()).backward()
    assert reference_parameter.grad is None
    passed("reference_parameters_receive_zero_gradients")
    assert base_parameter.grad is None
    passed("base_parameters_receive_zero_gradients")
    optimizer = torch.optim.SGD([policy_parameter], lr=0.1)
    optimizer_ids = {id(item) for group in optimizer.param_groups for item in group["params"]}
    assert optimizer_ids == {id(policy_parameter)}
    assert id(reference_parameter) not in optimizer_ids and id(base_parameter) not in optimizer_ids
    passed("optimizer_owns_only_policy_parameters")

    zero_map = {
        "finite": True,
        "exactly_zero": True,
        "graph_connected": True,
        "nonzero_gradient_count": 0,
    }
    live_map = {
        "finite": True,
        "exactly_zero": False,
        "graph_connected": True,
        "nonzero_gradient_count": 1,
    }
    noop_observation = _fixture_observation(
        rewards=[1.0] * 4,
        advantages=[0.0] * 4,
        variance=0.0,
        policy=zero_map,
        kl=zero_map,
        combined=zero_map,
    )
    assert classify_group(noop_observation) == EXPECTED_ZERO_ADVANTAGE_NOOP
    incomplete = complete_smoke_gate(
        (
            {
                "classification": EXPECTED_ZERO_ADVANTAGE_NOOP,
                "completion_count": 4,
                "reward_variance": 0.0,
                "policy_parameter_changed": False,
                "optimizer_step_completed": True,
                "scheduler_step_completed": True,
                "reference_parameter_changed": False,
                "base_parameter_changed": False,
            },
        )
    )
    assert incomplete["passed"] is False
    passed("single_noop_does_not_pass_complete_smoke", incomplete)

    update_observation = _fixture_observation(
        rewards=[0.0, 1.0, 0.0, 1.0],
        advantages=[-0.5, 0.5, -0.5, 0.5],
        variance=0.25,
        policy=live_map,
        kl=zero_map,
        combined=live_map,
    )
    assert classify_group(update_observation) == NONZERO_GRADIENT_UPDATE
    complete = complete_smoke_gate(
        (
            {
                "classification": EXPECTED_ZERO_ADVANTAGE_NOOP,
                "completion_count": 4,
                "reward_variance": 0.0,
                "policy_parameter_changed": False,
                "optimizer_step_completed": True,
                "scheduler_step_completed": True,
                "reference_parameter_changed": False,
                "base_parameter_changed": False,
            },
            {
                "classification": NONZERO_GRADIENT_UPDATE,
                "completion_count": 4,
                "reward_variance": 0.25,
                "policy_parameter_changed": True,
                "optimizer_step_completed": True,
                "scheduler_step_completed": True,
                "reference_parameter_changed": False,
                "base_parameter_changed": False,
            },
        )
    )
    assert complete["passed"] is True
    passed("noop_plus_update_passes_complete_smoke", complete)

    if len(rows) != 15 or any(row["passed"] is not True for row in rows):
        raise RuntimeError("zero-gradient fixture inventory differs")
    payload: dict[str, object] = {
        "schema_version": 1,
        "fixture_contract_id": FIXTURE_CONTRACT_ID,
        "classification_contract_id": CLASSIFICATION_CONTRACT_ID,
        "fixture_count": len(rows),
        "fixtures": rows,
    }
    payload["fixture_sha256"] = canonical_sha256(payload)
    return payload


def classification_contract() -> dict[str, object]:
    """Return the frozen model-independent classification definition."""

    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    payload: dict[str, object] = {
        "schema_version": 1,
        "classification_contract_id": CLASSIFICATION_CONTRACT_ID,
        "classifications": [
            EXPECTED_ZERO_ADVANTAGE_NOOP,
            UNEXPECTED_ZERO_GRADIENT,
            NONZERO_GRADIENT_UPDATE,
            INVALID_OR_AMBIGUOUS,
        ],
        "expected_zero_advantage_noop_requires": [
            "finite_rewards",
            "exact_zero_reward_variance",
            "all_advantages_exactly_zero",
            "nonempty_completion_token_masks",
            "finite_policy_and_reference_logprobs",
            "finite_policy_reference_kl",
            "identical_step_start_adapters",
            "exact_zero_policy_component_gradient",
            "exact_zero_kl_component_gradient",
            "exact_zero_combined_gradient",
            "zero_base_and_reference_gradients",
            "policy_graph_connected",
        ],
        "unexpected_zero_gradient_requires": [
            "nonzero_reward_variance_or_advantage",
            "nonempty_completion_token_masks",
            "connected_policy_objective",
            "exact_zero_combined_policy_gradient",
        ],
        "classification_uses_total_norm_only": False,
        "reward_scaling": False,
        "loss_type": "dr_grpo",
        "beta": 0.04,
        "epsilon": 0.2,
        "max_completion_length": 256,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "classify_group_source_sha256": hashlib.sha256(
            inspect.getsource(classify_group).encode("utf-8")
        ).hexdigest(),
        "objective_components_source_sha256": hashlib.sha256(
            inspect.getsource(objective_components).encode("utf-8")
        ).hexdigest(),
    }
    payload["classification_contract_sha256"] = canonical_sha256(payload)
    return payload
