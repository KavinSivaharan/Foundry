"""Adjudicate the two frozen Milestone 14A-R1 diagnostic reproductions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_zero_gradient import (
    EXPECTED_ZERO_ADVANTAGE_NOOP,
    classification_contract,
    classify_group,
)
from foundry.training.config import canonical_sha256

DECISION_ID = "foundry-milestone14a-r1-zero-gradient-decision-v1"
PARTIAL_SHA256 = "2c9bc725e91e3924d400bfb29904e027a4fc47eeb286b91804d77fe8e0917f6a"
STDERR_SHA256 = "ba098a00f03f3f3d5cb1e2e787f7c878dc342adf15037737a5c4aa7ca7ba5212"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"diagnostic evidence must be an object: {path}")
    return cast(dict[str, Any], value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(value: Mapping[str, object], key: str) -> None:
    projected = dict(value)
    expected = projected.pop(key, None)
    if expected != canonical_sha256(projected):
        raise ValueError(f"{key} does not reconstruct")


def _finite_nested(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_nested(item) for item in value)
    if isinstance(value, Mapping):
        return all(_finite_nested(item) for item in value.values())
    return True


def _projection(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} gradient projection is absent")
    result = cast(dict[str, Any], value)
    _verify(result, "gradient_projection_sha256")
    if (
        result.get("parameter_count") != 112
        or result.get("present_gradient_count") != 112
        or result.get("missing_gradient_count") != 0
        or result.get("finite") is not True
        or result.get("graph_connected") is not True
        or result.get("exactly_zero") is not True
        or result.get("nonzero_gradient_count") != 0
        or result.get("global_norm") != 0.0
    ):
        raise ValueError(f"{name} gradient projection is not an exact connected zero")
    return result


def build_decision(root: Path) -> dict[str, object]:
    """Build a content-free Case-1 decision from byte-identical raw evidence."""

    root = root.resolve()
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a_r1/diagnostic"
    paths = (raw / "run-1/partial_evidence.json", raw / "run-2/partial_evidence.json")
    stderr_paths = (raw / "logs/run-1.stderr.txt", raw / "logs/run-2.stderr.txt")
    if (
        any(_file_sha256(path) != PARTIAL_SHA256 for path in paths)
        or paths[0].read_bytes() != paths[1].read_bytes()
        or any(_file_sha256(path) != STDERR_SHA256 for path in stderr_paths)
        or stderr_paths[0].read_bytes() != stderr_paths[1].read_bytes()
    ):
        raise RuntimeError("diagnostic reproductions are not byte-identical")
    first = _read(paths[0])
    second = _read(paths[1])
    if first != second:
        raise RuntimeError("diagnostic JSON values differ")
    _verify(first, "failure_sha256")
    if (
        first.get("status") != "failed"
        or first.get("stage") != "combined_projection_persisted"
        or first.get("error_type") != "KeyError"
        or first.get("error_message") != "'optimizer_owned_tensors'"
    ):
        raise ValueError("diagnostic did not stop at the frozen bookkeeping lookup")
    reward = first.get("reward_projection")
    if not isinstance(reward, dict):
        raise TypeError("diagnostic reward projection is absent")
    reward = cast(dict[str, Any], reward)
    _verify(reward, "reward_projection_sha256")
    projected_rewards = reward.get("rewards")
    if (
        projected_rewards != [1.149999976158142] * 4
        or reward.get("reward_variance") != 0.0
        or reward.get("advantages") != [0.0, 0.0, 0.0, 0.0]
        or reward.get("normalized_advantages") != [0.0, 0.0, 0.0, 0.0]
        or reward.get("reward_scaling") is not False
    ):
        raise ValueError("diagnostic reward/advantage result differs from Case 1")
    valid_counts = first.get("valid_completion_token_counts")
    if valid_counts != [76, 80, 76, 68]:
        raise ValueError("diagnostic valid-token counts differ")
    if not all(
        _finite_nested(first.get(name))
        for name in (
            "policy_token_logprobs",
            "reference_token_logprobs",
            "policy_reference_kl",
            "objective_values",
        )
    ):
        raise ValueError("diagnostic log probabilities, KL, or objectives are nonfinite")
    objectives = first.get("objective_values")
    graphs = first.get("objective_graph")
    if (
        objectives
        != {
            "combined_objective": 0.0,
            "kl_objective": 0.0,
            "policy_objective": 0.0,
            "stock_loss": 0.0,
        }
        or not isinstance(graphs, dict)
        or any(
            not isinstance(value, dict)
            or value.get("requires_grad") is not True
            or not isinstance(value.get("grad_fn"), str)
            for value in graphs.values()
        )
    ):
        raise ValueError("diagnostic objective graph is detached or differs")
    policy = _projection(first.get("policy_gradient"), "policy")
    kl = _projection(first.get("kl_gradient"), "KL")
    combined = _projection(first.get("combined_gradient"), "combined")
    warning = first.get("warning")
    if (
        not isinstance(warning, dict)
        or warning.get("active_adapters_before") != ["default"]
        or warning.get("active_adapters_after") != ["default"]
        or warning.get("state_unchanged") is not True
        or warning.get("strict_entry") is not True
        or warning.get("strict_restored") is not True
    ):
        raise ValueError("diagnostic policy adapter generation audit differs")
    freeze = _read(root / "results/phase2_vetted_corpus/milestone14a_r1_zero_gradient_freeze.json")
    _verify(freeze, "freeze_sha256")
    contract = classification_contract()
    if freeze.get("classification_contract") != contract:
        raise ValueError("classification contract differs from its pre-model freeze")
    fixtures = freeze.get("fixture_contract")
    if (
        not isinstance(fixtures, dict)
        or fixtures.get("fixture_count") != 15
        or any(row.get("passed") is not True for row in fixtures.get("fixtures", []))
    ):
        raise ValueError("controlled live-policy fixtures did not pass")
    classification_input: dict[str, object] = {
        "rewards": reward["rewards"],
        "reward_variance": reward["reward_variance"],
        "advantages": reward["advantages"],
        "valid_completion_token_counts": valid_counts,
        "policy_logprobs_finite": True,
        "reference_logprobs_finite": True,
        "kl_finite": True,
        "adapters_identical_at_step_start": True,
        "controlled_live_policy_fixture_passed": True,
        "requires_grad_policy_tensor_count": 112,
        "optimizer_owned_tensor_count": 112,
        "base_gradient_count": 0,
        "reference_gradient_count": 0,
        "policy_gradient": policy,
        "kl_gradient": kl,
        "combined_gradient": combined,
    }
    classification = classify_group(classification_input)
    if classification != EXPECTED_ZERO_ADVANTAGE_NOOP:
        raise RuntimeError(f"diagnostic classification differs: {classification}")
    payload: dict[str, object] = {
        "schema_version": 1,
        "decision_id": DECISION_ID,
        "starting_commit": "b0635a7c0f551dfb8efd846da5cfe83b28f7af18",
        "original_blocker_sha256": (
            "d4b23d898ef3c53db46882a4a218c2a43cd85298ebdfa75139eaf3a7c08e8752"
        ),
        "classification_contract_sha256": contract["classification_contract_sha256"],
        "fixture_sha256": fixtures["fixture_sha256"],
        "diagnostic_reproductions": 2,
        "diagnostic_partial_sha256s": [PARTIAL_SHA256, PARTIAL_SHA256],
        "diagnostic_stderr_sha256s": [STDERR_SHA256, STDERR_SHA256],
        "diagnostic_evidence_byte_identical": True,
        "diagnostic_stderr_byte_identical": True,
        "reward_vector": reward["rewards"],
        "reward_mean": reward["reward_mean"],
        "reward_variance": reward["reward_variance"],
        "advantages": reward["advantages"],
        "normalized_advantages": reward["normalized_advantages"],
        "valid_completion_token_counts": valid_counts,
        "policy_logprobs_sha256": canonical_sha256(first["policy_token_logprobs"]),
        "reference_logprobs_sha256": canonical_sha256(first["reference_token_logprobs"]),
        "policy_reference_kl_sha256": canonical_sha256(first["policy_reference_kl"]),
        "objective_values": objectives,
        "objective_graph": graphs,
        "policy_gradient_projection_sha256": policy["gradient_projection_sha256"],
        "policy_gradient_global_norm": policy["global_norm"],
        "kl_gradient_projection_sha256": kl["gradient_projection_sha256"],
        "kl_gradient_global_norm": kl["global_norm"],
        "combined_gradient_projection_sha256": combined["gradient_projection_sha256"],
        "combined_gradient_global_norm": combined["global_norm"],
        "policy_gradient_tensors_present": policy["present_gradient_count"],
        "reference_gradient_count": 0,
        "base_gradient_count": 0,
        "optimizer_owned_policy_tensors": 112,
        "initial_policy_reference_identity_sha256": first["initial_identity_sha256"],
        "controlled_positive_kl": first["controlled_positive_kl"],
        "active_policy_before_and_after_generation": True,
        "reference_proxy_fail_closed_switching": True,
        "classification": classification,
        "original_exception_classification": "overstrict_per_group_update_gate",
        "scientific_grpo_contract_changed": False,
        "general_validation_semantics_correction_authorized": True,
        "diagnostic_bookkeeping_error": {
            "type": "KeyError",
            "missing_key": "optimizer_owned_tensors",
            "correct_frozen_runtime_key": "optimizer_parameter_tensors",
            "occurred_after_all_three_decisive_gradient_projections": True,
            "changed_scientific_classification": False,
        },
        "optimizer_steps": 0,
        "scheduler_advancements": 0,
        "adapter_or_checkpoint_saved": False,
        "counted_training_started": False,
        "retention_started": False,
        "gsm1k_started": False,
        "sealed_content_use": 0,
    }
    payload["diagnostic_decision_sha256"] = canonical_sha256(payload)
    return payload


def write_decision(root: Path) -> dict[str, object]:
    result = build_decision(root)
    path = (
        root.resolve() / "results/phase2_vetted_corpus/milestone14a_r1_zero_gradient_decision.json"
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError("existing zero-gradient decision differs")
    else:
        path.write_text(rendered, encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_decision(args.root)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "diagnostic_decision_sha256": result["diagnostic_decision_sha256"],
                "diagnostic_evidence_byte_identical": result["diagnostic_evidence_byte_identical"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
