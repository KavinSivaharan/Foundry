"""Cross-device advantage equivalence for the corrected L3 GRPO signal audit.

The stock TRL CUDA tensor remains canonical.  The helpers in this module only
recompute that projection on CUDA and compare it with the historical CPU
float32 diagnostic under a predeclared numerical and invariant contract.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from foundry.phase2.l3_grpo_signal_blocker import FIXTURE_VALUES
from foundry.phase2.l3_grpo_zero_gradient import reward_projection
from foundry.training.config import canonical_sha256
from foundry.training.grpo_replay_evidence import tensor_evidence

CONTRACT_ID = "foundry-l3-grpo-advantage-equivalence-v1"
ABSOLUTE_TOLERANCE = 2.384185791015625e-07
RELATIVE_TOLERANCE = 1e-6
FLOAT32_EPSILON = 1.1920928955078125e-07
EXPECTED_FIXTURE_VECTOR_COUNT = 65_536
EXPECTED_EXACT_MISMATCH_VECTOR_COUNT = 11_915
EXPECTED_MAXIMUM_FIXTURE_DIFFERENCE = FLOAT32_EPSILON
EXPECTED_PRIOR_FIXTURE_SHA256 = "35bbe14e12659645035890450b64d74ec2da37eb4d03f73392f2d1885f8f555"


def _source_sha256(value: Callable[..., object]) -> str:
    """Hash one callable's normalized source text."""

    return hashlib.sha256(inspect.getsource(value).encode("utf-8")).hexdigest()


def _as_float_list(values: Sequence[object]) -> list[float]:
    return [
        float(value)
        for value in values
        if not isinstance(value, bool) and isinstance(value, int | float)
    ]


def recompute_stock_cuda_projection(torch: Any, rewards: Sequence[float]) -> dict[str, object]:
    """Recompute TRL 0.17's one-function, unscaled CUDA reward projection."""

    if len(rewards) != 4:
        raise ValueError("canonical CUDA projection requires exactly four rewards")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("canonical advantage projection requires CUDA")
    reward_outputs = torch.tensor(list(rewards), dtype=torch.float32, device="cuda:0")
    rewards_per_func = torch.zeros((4, 1), dtype=torch.float32, device="cuda:0")
    rewards_per_func[:, 0] = reward_outputs
    reward_weights = torch.tensor([1.0], dtype=torch.float32, device="cuda:0")
    canonical_rewards = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)
    mean = canonical_rewards.view(-1, 4).mean(dim=1).repeat_interleave(4)
    advantages = canonical_rewards - mean
    variance = ((canonical_rewards - mean) ** 2).mean()
    result: dict[str, object] = {
        "implementation_id": "trl-0.17-one-reward-unscaled-cuda-projection",
        "device": str(canonical_rewards.device),
        "dtype": str(canonical_rewards.dtype),
        "reward_vector": [float(value) for value in canonical_rewards.detach().cpu().tolist()],
        "reward_vector_evidence": tensor_evidence(canonical_rewards).as_dict(),
        "reward_mean": float(mean[0].item()),
        "reward_variance": float(variance.item()),
        "advantages": [float(value) for value in advantages.detach().cpu().tolist()],
        "advantage_tensor_evidence": tensor_evidence(advantages).as_dict(),
        "reward_scaling": False,
    }
    result["projection_sha256"] = canonical_sha256(result)
    return result


def cpu_diagnostic_projection(torch: Any, rewards: Sequence[float]) -> dict[str, object]:
    """Run the frozen CPU helper as a diagnostic, never as canonical evidence."""

    projected = reward_projection(torch, rewards)
    return {
        "implementation_id": "foundry-historical-cpu-float32-reward-projection",
        "device": "cpu",
        "dtype": "torch.float32",
        "reward_vector": projected["rewards"],
        "reward_mean": projected["reward_mean"],
        "reward_variance": projected["reward_variance"],
        "advantages": projected["advantages"],
        "reward_projection_sha256": projected["reward_projection_sha256"],
        "scientific_gate_authority": False,
    }


def _ordering_preserved(rewards: Sequence[float], advantages: Sequence[float]) -> bool:
    if len(rewards) != len(advantages):
        return False
    for left in range(len(rewards)):
        for right in range(len(rewards)):
            if rewards[left] == rewards[right]:
                if advantages[left] != advantages[right]:
                    return False
                continue
            if (rewards[left] < rewards[right]) != (advantages[left] < advantages[right]):
                return False
    return True


def _material_sign_preserved(canonical: Sequence[float], diagnostic: Sequence[float]) -> bool:
    """Ignore only numerically indistinguishable signs inside the absolute tolerance."""

    return all(
        not (
            abs(left) > ABSOLUTE_TOLERANCE
            and abs(right) > ABSOLUTE_TOLERANCE
            and ((left < 0.0) != (right < 0.0))
        )
        for left, right in zip(canonical, diagnostic, strict=True)
    )


def evaluate_advantage_equivalence(
    *,
    reward_vector: Sequence[object],
    canonical_cuda_advantages: Sequence[object],
    duplicate_cuda_advantages: Sequence[object],
    cpu_diagnostic_advantages: Sequence[object],
    canonical_reward_vector_sha256: str,
    duplicate_reward_vector_sha256: str,
    canonical_projection_matches_stock: bool,
) -> dict[str, object]:
    """Evaluate one four-completion group under the frozen equivalence contract."""

    shapes = {
        "reward_vector": len(reward_vector),
        "canonical_cuda_advantages": len(canonical_cuda_advantages),
        "duplicate_cuda_advantages": len(duplicate_cuda_advantages),
        "cpu_diagnostic_advantages": len(cpu_diagnostic_advantages),
    }
    shape_identical = len(set(shapes.values())) == 1 and next(iter(shapes.values())) == 4
    numeric_lengths = {
        "reward_vector": len(_as_float_list(reward_vector)),
        "canonical_cuda_advantages": len(_as_float_list(canonical_cuda_advantages)),
        "duplicate_cuda_advantages": len(_as_float_list(duplicate_cuda_advantages)),
        "cpu_diagnostic_advantages": len(_as_float_list(cpu_diagnostic_advantages)),
    }
    numeric_shape_valid = shape_identical and all(value == 4 for value in numeric_lengths.values())
    rewards = _as_float_list(reward_vector)
    canonical = _as_float_list(canonical_cuda_advantages)
    duplicate = _as_float_list(duplicate_cuda_advantages)
    diagnostic = _as_float_list(cpu_diagnostic_advantages)
    finite_classifications_identical = False
    all_finite = False
    maximum_difference: float | None = None
    cuda_duplicate_exact = False
    cpu_cuda_close = False
    reward_ordering_preserved = False
    advantage_ordering_preserved = False
    material_sign_preserved = False
    group_zero_nonzero_classification_identical = False
    zero_variance_exact_zero = False
    nonzero_variance_has_canonical_nonzero = False
    reward_variance_zero = False
    if numeric_shape_valid:
        classifications = [
            [math.isfinite(value) for value in values]
            for values in (rewards, canonical, duplicate, diagnostic)
        ]
        finite_classifications_identical = all(
            value == classifications[0] for value in classifications[1:]
        )
        all_finite = all(all(value) for value in classifications)
        if all_finite:
            differences = [
                abs(left - right) for left, right in zip(canonical, diagnostic, strict=True)
            ]
            maximum_difference = max(differences)
            cuda_duplicate_exact = canonical == duplicate
            cpu_cuda_close = all(
                math.isclose(
                    left,
                    right,
                    rel_tol=RELATIVE_TOLERANCE,
                    abs_tol=ABSOLUTE_TOLERANCE,
                )
                for left, right in zip(canonical, diagnostic, strict=True)
            )
            reward_ordering_preserved = _ordering_preserved(rewards, canonical)
            advantage_ordering_preserved = _ordering_preserved(rewards, diagnostic)
            material_sign_preserved = _material_sign_preserved(canonical, diagnostic)
            group_zero_nonzero_classification_identical = any(
                value != 0.0 for value in canonical
            ) == any(value != 0.0 for value in diagnostic)
            reward_variance_zero = all(value == rewards[0] for value in rewards[1:])
            zero_variance_exact_zero = not reward_variance_zero or (
                all(value == 0.0 for value in canonical)
                and all(value == 0.0 for value in duplicate)
                and all(value == 0.0 for value in diagnostic)
            )
            nonzero_variance_has_canonical_nonzero = reward_variance_zero or any(
                value != 0.0 for value in canonical
            )
    conditions = {
        "reward_vector_bytes_identical": (
            len(canonical_reward_vector_sha256) == 64
            and canonical_reward_vector_sha256 == duplicate_reward_vector_sha256
        ),
        "shape_identical": shape_identical,
        "numeric_shape_valid": numeric_shape_valid,
        "finite_nonfinite_classification_identical": finite_classifications_identical,
        "all_values_finite": all_finite,
        "zero_variance_rewards_produce_exact_zero_advantages": zero_variance_exact_zero,
        "nonzero_variance_rewards_produce_canonical_nonzero_advantage": (
            nonzero_variance_has_canonical_nonzero
        ),
        "duplicate_cuda_projection_exact": cuda_duplicate_exact,
        "cpu_projection_within_frozen_tolerance": cpu_cuda_close,
        "reward_ordering_preserved_by_canonical_cuda": reward_ordering_preserved,
        "advantage_ordering_preserved_for_unequal_rewards": advantage_ordering_preserved,
        "material_sign_classification_preserved": material_sign_preserved,
        "group_zero_nonzero_classification_identical": (
            group_zero_nonzero_classification_identical
        ),
        "canonical_recomputation_matches_stock_cuda": canonical_projection_matches_stock,
    }
    passed = all(conditions.values())
    result: dict[str, object] = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "canonical_source": "stock_trl_cuda_advantage_tensor",
        "cpu_projection_role": "diagnostic_only",
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "shapes": shapes,
        "reward_variance_classification": "zero" if reward_variance_zero else "nonzero",
        "maximum_absolute_difference": maximum_difference,
        "conditions": conditions,
        "passed": passed,
        "failure_classification": None if passed else "advantage_projection_mismatch",
    }
    result["equivalence_sha256"] = canonical_sha256(result)
    return result


def exhaustive_cross_device_fixture(torch: Any) -> dict[str, object]:
    """Replay all 16^4 historical reward vectors on CPU and CUDA."""

    if not bool(torch.cuda.is_available()):
        raise RuntimeError("cross-device advantage fixture requires CUDA")
    rows = torch.tensor(
        list(itertools.product(FIXTURE_VALUES, repeat=4)),
        dtype=torch.float32,
    )
    cpu = rows - rows.mean(dim=1, keepdim=True)
    cuda_rows_first = rows.to("cuda:0")
    cuda_first = cuda_rows_first - cuda_rows_first.mean(dim=1, keepdim=True)
    cuda_rows_second = rows.to("cuda:0")
    cuda_second = cuda_rows_second - cuda_rows_second.mean(dim=1, keepdim=True)
    cuda_first_cpu = cuda_first.cpu()
    differing = torch.any(cpu != cuda_first_cpu, dim=1)
    differences = torch.abs(cpu - cuda_first_cpu)
    maximum = float(differences.max().item())
    close = torch.isclose(
        cpu,
        cuda_first_cpu,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    material_sign_change = (
        (torch.abs(cpu) > ABSOLUTE_TOLERANCE)
        & (torch.abs(cuda_first_cpu) > ABSOLUTE_TOLERANCE)
        & ((cpu < 0.0) != (cuda_first_cpu < 0.0))
    )
    cpu_nonzero = torch.any(cpu != 0.0, dim=1)
    cuda_nonzero = torch.any(cuda_first_cpu != 0.0, dim=1)
    result: dict[str, object] = {
        "schema_version": 1,
        "fixture_id": "foundry-l3-grpo-cross-device-advantage-equivalence-fixture-v1",
        "prior_fixture_sha256": EXPECTED_PRIOR_FIXTURE_SHA256,
        "input_value_count": len(FIXTURE_VALUES),
        "fixture_vector_count": int(rows.shape[0]),
        "exact_mismatch_vector_count": int(differing.sum().item()),
        "maximum_absolute_difference": maximum,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "all_values_within_tolerance": bool(close.all().item()),
        "material_sign_change_vector_count": int(torch.any(material_sign_change, dim=1).sum()),
        "group_zero_nonzero_classification_difference_count": int(
            (cpu_nonzero != cuda_nonzero).sum().item()
        ),
        "duplicate_cuda_projection_exact": bool(torch.equal(cuda_first, cuda_second)),
        "cpu_tensor_evidence": tensor_evidence(cpu).as_dict(),
        "first_cuda_tensor_evidence": tensor_evidence(cuda_first).as_dict(),
        "second_cuda_tensor_evidence": tensor_evidence(cuda_second).as_dict(),
        "model_loaded": False,
        "generation_calls": 0,
    }
    if (
        result["fixture_vector_count"] != EXPECTED_FIXTURE_VECTOR_COUNT
        or result["exact_mismatch_vector_count"] != EXPECTED_EXACT_MISMATCH_VECTOR_COUNT
        or result["maximum_absolute_difference"] != EXPECTED_MAXIMUM_FIXTURE_DIFFERENCE
        or result["all_values_within_tolerance"] is not True
        or result["material_sign_change_vector_count"] != 0
        or result["group_zero_nonzero_classification_difference_count"] != 0
        or result["duplicate_cuda_projection_exact"] is not True
    ):
        raise RuntimeError("cross-device advantage fixture differs from the frozen observation")
    result["fixture_sha256"] = canonical_sha256(result)
    return result


def advantage_equivalence_contract(
    fixture: Mapping[str, object],
) -> dict[str, object]:
    """Freeze the corrected numerical contract around a verified CUDA fixture."""

    supplied = fixture.get("fixture_sha256")
    fixture_payload = {key: value for key, value in fixture.items() if key != "fixture_sha256"}
    if not isinstance(supplied, str) or supplied != canonical_sha256(fixture_payload):
        raise ValueError("advantage-equivalence fixture does not reconstruct")
    if (
        fixture.get("fixture_vector_count") != EXPECTED_FIXTURE_VECTOR_COUNT
        or fixture.get("exact_mismatch_vector_count") != EXPECTED_EXACT_MISMATCH_VECTOR_COUNT
        or fixture.get("maximum_absolute_difference") != EXPECTED_MAXIMUM_FIXTURE_DIFFERENCE
        or fixture.get("absolute_tolerance") != ABSOLUTE_TOLERANCE
        or fixture.get("relative_tolerance") != RELATIVE_TOLERANCE
        or fixture.get("all_values_within_tolerance") is not True
    ):
        raise ValueError("advantage-equivalence fixture values differ")
    contract: dict[str, object] = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "canonical_advantage_source": "stock_trl_cuda_tensor_used_by_training_path",
        "cpu_projection_role": "diagnostic_only",
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "float32_machine_epsilon": FLOAT32_EPSILON,
        "tolerance_in_float32_epsilons": 2,
        "observed_maximum_in_float32_epsilons": 1,
        "canonical_implementation_sha256": _source_sha256(recompute_stock_cuda_projection),
        "cpu_diagnostic_implementation_sha256": _source_sha256(cpu_diagnostic_projection),
        "equivalence_implementation_sha256": _source_sha256(evaluate_advantage_equivalence),
        "exhaustive_fixture_sha256": supplied,
        "exhaustive_fixture": dict(fixture),
        "zero_nonzero_gate_authority": "canonical_cuda_only",
        "schedule_viability_authority": "canonical_cuda_reward_variance_and_advantages",
        "failure_classification": "advantage_projection_mismatch",
        "tolerance_may_not_change_after_model_output": True,
    }
    contract["advantage_equivalence_contract_sha256"] = canonical_sha256(contract)
    return contract


def verify_advantage_equivalence_contract(value: Mapping[str, object]) -> None:
    supplied = value.get("advantage_equivalence_contract_sha256")
    payload = {
        key: item for key, item in value.items() if key != "advantage_equivalence_contract_sha256"
    }
    if (
        not isinstance(supplied, str)
        or supplied != canonical_sha256(payload)
        or value.get("contract_id") != CONTRACT_ID
        or value.get("absolute_tolerance") != ABSOLUTE_TOLERANCE
        or value.get("relative_tolerance") != RELATIVE_TOLERANCE
        or value.get("canonical_implementation_sha256")
        != _source_sha256(recompute_stock_cuda_projection)
        or value.get("cpu_diagnostic_implementation_sha256")
        != _source_sha256(cpu_diagnostic_projection)
        or value.get("equivalence_implementation_sha256")
        != _source_sha256(evaluate_advantage_equivalence)
    ):
        raise ValueError("advantage-equivalence contract differs")
    fixture = value.get("exhaustive_fixture")
    if not isinstance(fixture, dict):
        raise ValueError("advantage-equivalence fixture is missing")
    advantage_equivalence_contract(cast(Mapping[str, object], fixture))
