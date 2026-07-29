"""Content-free Milestone 14B reward-signal classification and aggregation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any, Literal, cast

from foundry.training.config import canonical_sha256

AUDIT_ID = "foundry-l3-grpo-signal-audit-v1"
CONTRACT_ID = "foundry-l3-grpo-signal-audit-contract-v1"
ARMS = ("generic", "targeted")
TASK_FAMILIES = ("bookkeeping", "rate_ratio", "discrete")
GROUPS_PER_ARM = 32
TASK_GROUPS_PER_ARM = 24
REPLAY_GROUPS_PER_ARM = 8
COMPLETIONS_PER_GROUP = 4
COMPLETIONS_PER_ARM = 128
TOTAL_GROUPS = 64
TOTAL_COMPLETIONS = 256
MINIMUM_NONZERO_VARIANCE_TASK_GROUPS = 4
MINIMUM_INFORMATIVE_TASK_FAMILIES = 2

ZERO_VARIANCE_CLASSIFICATIONS = (
    "all_correct_saturated",
    "all_incorrect_saturated",
    "output_diverse_reward_indistinguishable",
    "output_identical",
    "invalid_or_ambiguous",
)
ZeroVarianceClassification = Literal[
    "all_correct_saturated",
    "all_incorrect_saturated",
    "output_diverse_reward_indistinguishable",
    "output_identical",
    "invalid_or_ambiguous",
]

REWARD_COMPONENT_FIELDS = (
    "task_answer_correctness",
    "replay_scorer_correctness",
    "extraction",
    "canonical_or_required_format",
    "instruction_compliance",
    "truncation_penalty",
    "prompt_echo_penalty",
    "question_generation_penalty",
    "malformed_output_penalty",
    "backend_failure_penalty",
)


def _finite_numbers(values: Sequence[object]) -> bool:
    return len(values) == COMPLETIONS_PER_GROUP and all(
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        for value in values
    )


def _all_equal(values: Sequence[object]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def classify_zero_variance_group(
    *,
    reward_variance: float,
    completion_sha256s: Sequence[str],
    reward_vector: Sequence[float],
    reward_component_vectors: Mapping[str, Sequence[float]],
    correctness: Sequence[bool],
    evidence_complete: bool = True,
) -> ZeroVarianceClassification | None:
    """Classify one zero-variance group using frozen, mutually exclusive precedence."""

    if (
        not evidence_complete
        or not math.isfinite(reward_variance)
        or reward_variance < 0.0
        or len(completion_sha256s) != COMPLETIONS_PER_GROUP
        or len(correctness) != COMPLETIONS_PER_GROUP
        or not _finite_numbers(cast(Sequence[object], reward_vector))
        or set(reward_component_vectors) != set(REWARD_COMPONENT_FIELDS)
        or any(
            not _finite_numbers(cast(Sequence[object], values))
            for values in reward_component_vectors.values()
        )
    ):
        return "invalid_or_ambiguous"
    if reward_variance > 0.0:
        return None
    rewards_identical = _all_equal(cast(Sequence[object], reward_vector))
    components_identical = all(
        _all_equal(cast(Sequence[object], values)) for values in reward_component_vectors.values()
    )
    if not rewards_identical or not components_identical:
        return "invalid_or_ambiguous"

    # Output identity takes precedence because it is direct evidence that sampling
    # produced no candidate diversity. The remaining classes therefore describe
    # diverse-output groups without overlap.
    if _all_equal(cast(Sequence[object], completion_sha256s)):
        return "output_identical"
    if all(correctness):
        return "all_correct_saturated"
    if not any(correctness):
        return "all_incorrect_saturated"
    return "output_diverse_reward_indistinguishable"


def signal_audit_method_contract() -> dict[str, object]:
    """Return the model-independent audit and viability contract."""

    contract: dict[str, object] = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "audit_id": AUDIT_ID,
        "scope": {
            "arms": list(ARMS),
            "groups_per_arm": GROUPS_PER_ARM,
            "task_groups_per_arm": TASK_GROUPS_PER_ARM,
            "replay_groups_per_arm": REPLAY_GROUPS_PER_ARM,
            "completions_per_group": COMPLETIONS_PER_GROUP,
            "completions_per_arm": COMPLETIONS_PER_ARM,
            "total_groups": TOTAL_GROUPS,
            "total_completions": TOTAL_COMPLETIONS,
        },
        "execution_prohibitions": {
            "optimizer_creation": True,
            "backward": True,
            "scheduler_creation_or_advancement": True,
            "adapter_mutation": True,
            "trained_adapter_save": True,
            "schedule_reordering": True,
        },
        "per_group_fields": [
            "arm",
            "schedule_position",
            "source_kind",
            "task_family",
            "prompt_sha256",
            "completion_sha256s",
            "completion_token_counts",
            "distinct_completion_count",
            "reward_vector",
            "reward_component_vectors",
            "reward_mean",
            "reward_variance",
            "advantages",
            "nonzero_advantage_count",
            "reward_rank_count",
            "correctness_vector",
            "correctness_count",
            "extraction_count",
            "compliant_format_count",
            "malformed_count",
            "prompt_echo_count",
            "question_generation_count",
            "truncation_count",
            "backend_failure_count",
            "valid_completion_token_counts",
            "policy_reference_kl",
            "zero_variance_classification",
        ],
        "zero_variance_classifications": list(ZERO_VARIANCE_CLASSIFICATIONS),
        "classification_precedence": [
            "invalid_or_ambiguous_on_missing_nonfinite_or_inconsistent_evidence",
            "output_identical",
            "all_correct_saturated",
            "all_incorrect_saturated",
            "output_diverse_reward_indistinguishable",
        ],
        "reward_component_fields": list(REWARD_COMPONENT_FIELDS),
        "family_summary_fields": [
            "total_groups",
            "nonzero_variance_groups",
            "zero_variance_groups",
            "nonzero_variance_percentage",
            "zero_variance_classifications",
            "identical_output_groups",
            "average_distinct_completions",
            "average_reward_variance",
            "average_nonzero_advantage_count",
            "average_completion_correctness",
            "average_extraction_rate",
            "average_format_compliance_rate",
        ],
        "overlap_definitions": {
            "shared_prompt_positions": "same schedule position and prompt SHA-256",
            "completion_output_overlap": "set intersection over union at shared prompt positions",
            "reward_pattern_overlap": (
                "exact float32 reward-vector equality at shared prompt positions"
            ),
        },
        "viability_gate": {
            "minimum_nonzero_variance_task_groups_per_arm": (MINIMUM_NONZERO_VARIANCE_TASK_GROUPS),
            "minimum_informative_task_families_per_arm": (MINIMUM_INFORMATIVE_TASK_FAMILIES),
            "usable_task_group_requires": [
                "nonzero_advantage",
                "nonempty_valid_completion_token_mask",
                "finite_policy_logprobs",
                "finite_reference_logprobs",
            ],
            "invalid_or_ambiguous_groups": 0,
            "backend_failures": 0,
            "reward_contract_inconsistencies": 0,
            "deterministic_fresh_process_task_replays_per_arm_minimum": 1,
            "replay_groups_may_be_zero_variance_noops": True,
        },
        "scientific_settings_changed": False,
        "counted_training_schedule_changed": False,
        "holdout_v2_use": 0,
        "gsm1k_use": 0,
        "sealed_content_use": 0,
    }
    contract["method_contract_sha256"] = canonical_sha256(contract)
    return contract


def _verify_self_hash(value: Mapping[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _validate_group_evidence(row: Mapping[str, Any]) -> None:
    rewards = row.get("reward_vector")
    components = row.get("reward_component_vectors")
    completions = row.get("completion_sha256s")
    correctness = row.get("correctness_vector")
    advantages = row.get("advantages")
    valid_counts = row.get("valid_completion_token_counts")
    kl = row.get("policy_reference_kl")
    if (
        not isinstance(rewards, list)
        or not isinstance(components, dict)
        or not isinstance(completions, list)
        or not all(isinstance(value, str) and value for value in completions)
        or not isinstance(correctness, list)
        or not all(isinstance(value, bool) for value in correctness)
        or not isinstance(advantages, list)
        or not _finite_numbers(cast(Sequence[object], advantages))
        or not isinstance(valid_counts, list)
        or len(valid_counts) != COMPLETIONS_PER_GROUP
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in valid_counts
        )
        or not isinstance(kl, dict)
    ):
        raise ValueError("raw audit group evidence shape differs")
    variance = row.get("reward_variance")
    if isinstance(variance, bool) or not isinstance(variance, int | float):
        raise ValueError("raw audit reward variance differs")
    evidence_complete = (
        row.get("policy_logprobs_finite") is True
        and row.get("reference_logprobs_finite") is True
        and kl.get("finite") is True
        and row.get("reward_contract_consistent") is True
    )
    expected_classification = classify_zero_variance_group(
        reward_variance=float(variance),
        completion_sha256s=cast(list[str], completions),
        reward_vector=cast(list[float], rewards),
        reward_component_vectors=cast(dict[str, list[float]], components),
        correctness=cast(list[bool], correctness),
        evidence_complete=evidence_complete,
    )
    counts = (
        "correctness_count",
        "extraction_count",
        "compliant_format_count",
        "malformed_count",
        "prompt_echo_count",
        "question_generation_count",
        "truncation_count",
        "backend_failure_count",
    )
    if (
        row.get("zero_variance_classification") != expected_classification
        or row.get("distinct_completion_count") != len(set(completions))
        or row.get("reward_rank_count") != len(set(cast(list[float], rewards)))
        or row.get("nonzero_advantage_count") != sum(float(value) != 0.0 for value in advantages)
        or any(
            isinstance(row.get(name), bool)
            or not isinstance(row.get(name), int)
            or not 0 <= cast(int, row[name]) <= COMPLETIONS_PER_GROUP
            for name in counts
        )
    ):
        raise ValueError("raw audit group evidence does not reconstruct")


def _group_rows(raw: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    _verify_self_hash(raw, "raw_audit_sha256")
    if (
        raw.get("audit_id") != AUDIT_ID
        or raw.get("arm") != arm
        or raw.get("group_count") != GROUPS_PER_ARM
        or raw.get("completion_count") != COMPLETIONS_PER_ARM
        or raw.get("optimizer_created") is not False
        or raw.get("backward_calls") != 0
        or raw.get("scheduler_created") is not False
        or raw.get("adapter_saved") is not False
    ):
        raise ValueError(f"{arm} raw signal audit contract differs")
    groups_value = raw.get("groups")
    if not isinstance(groups_value, list) or len(groups_value) != GROUPS_PER_ARM:
        raise ValueError(f"{arm} raw audit must contain exactly 32 groups")
    groups: list[dict[str, Any]] = []
    for index, value in enumerate(groups_value, start=1):
        if not isinstance(value, dict):
            raise ValueError("raw audit group must be an object")
        row = cast(dict[str, Any], value)
        _verify_self_hash(row, "group_record_sha256")
        if (
            row.get("arm") != arm
            or row.get("schedule_position") != index
            or row.get("completion_count") != COMPLETIONS_PER_GROUP
        ):
            raise ValueError("raw audit group identity or cardinality differs")
        _validate_group_evidence(row)
        groups.append(row)
    if sum(row.get("source_kind") == "task" for row in groups) != TASK_GROUPS_PER_ARM:
        raise ValueError("task-group count differs")
    if sum(row.get("source_kind") == "base_replay" for row in groups) != REPLAY_GROUPS_PER_ARM:
        raise ValueError("replay-group count differs")
    return groups


def _classification_counts(groups: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        cast(str, row["zero_variance_classification"])
        for row in groups
        if row.get("reward_variance") == 0.0
    )
    return {name: counts[name] for name in ZERO_VARIANCE_CLASSIFICATIONS}


def _metric_summary(groups: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if not groups:
        raise ValueError("signal-density summary requires at least one group")
    variances = [float(cast(float, row["reward_variance"])) for row in groups]
    nonzero = sum(value > 0.0 for value in variances)
    completions = len(groups) * COMPLETIONS_PER_GROUP
    return {
        "total_groups": len(groups),
        "nonzero_variance_groups": nonzero,
        "zero_variance_groups": len(groups) - nonzero,
        "nonzero_variance_percentage": 100.0 * nonzero / len(groups),
        "zero_variance_classifications": _classification_counts(groups),
        "reward_indistinguishable_diversity_groups": sum(
            row.get("zero_variance_classification") == "output_diverse_reward_indistinguishable"
            for row in groups
        ),
        "identical_output_groups": sum(
            cast(int, row["distinct_completion_count"]) == 1 for row in groups
        ),
        "average_distinct_completions": fmean(
            cast(int, row["distinct_completion_count"]) for row in groups
        ),
        "average_reward_variance": fmean(variances),
        "average_nonzero_advantage_count": fmean(
            cast(int, row["nonzero_advantage_count"]) for row in groups
        ),
        "average_completion_correctness": (
            sum(cast(int, row["correctness_count"]) for row in groups) / completions
        ),
        "average_extraction_rate": (
            sum(cast(int, row["extraction_count"]) for row in groups) / completions
        ),
        "average_format_compliance_rate": (
            sum(cast(int, row["compliant_format_count"]) for row in groups) / completions
        ),
        "backend_failures": sum(cast(int, row["backend_failure_count"]) for row in groups),
        "reward_contract_inconsistencies": sum(
            row.get("reward_contract_consistent") is not True for row in groups
        ),
        "invalid_or_ambiguous_groups": sum(
            row.get("zero_variance_classification") == "invalid_or_ambiguous" for row in groups
        ),
        "completion_tokens": sum(
            sum(cast(list[int], row["completion_token_counts"])) for row in groups
        ),
    }


def _arm_summary(
    groups: Sequence[Mapping[str, Any]],
    deterministic_replay: Mapping[str, Any] | None,
) -> dict[str, object]:
    tasks = [row for row in groups if row.get("source_kind") == "task"]
    replay = [row for row in groups if row.get("source_kind") == "base_replay"]
    family = {
        name: _metric_summary(
            [row for row in tasks if str(row.get("task_family")).replace("/", "_") == name]
        )
        for name in TASK_FAMILIES
    }
    task_summary = _metric_summary(tasks)
    informative = [row for row in tasks if float(cast(float, row["reward_variance"])) > 0.0]
    informative_families = sorted(
        {
            str(row["task_family"]).replace("/", "_")
            for row in informative
            if str(row["task_family"]).replace("/", "_") in TASK_FAMILIES
        }
    )
    usable = [
        row
        for row in informative
        if cast(int, row["nonzero_advantage_count"]) > 0
        and sum(cast(list[int], row["valid_completion_token_counts"])) > 0
        and row.get("policy_logprobs_finite") is True
        and row.get("reference_logprobs_finite") is True
    ]
    candidates = [
        {
            "schedule_position": row["schedule_position"],
            "group_id": row["group_id"],
            "task_family": row["task_family"],
            "prompt_sha256": row["prompt_sha256"],
            "group_record_sha256": row["group_record_sha256"],
            "reward_variance": row["reward_variance"],
            "nonzero_advantage_count": row["nonzero_advantage_count"],
            "valid_completion_token_count": sum(
                cast(list[int], row["valid_completion_token_counts"])
            ),
        }
        for row in usable
    ]
    deterministic_passed = (
        deterministic_replay is not None
        and deterministic_replay.get("passed") is True
        and deterministic_replay.get("group_id") in {cast(str, row["group_id"]) for row in usable}
    )
    conditions = {
        "minimum_nonzero_variance_task_groups": (
            len(informative) >= MINIMUM_NONZERO_VARIANCE_TASK_GROUPS
        ),
        "minimum_informative_task_families": (
            len(informative_families) >= MINIMUM_INFORMATIVE_TASK_FAMILIES
        ),
        "usable_informative_task_group": bool(usable),
        "zero_invalid_or_ambiguous_groups": (
            _metric_summary(groups)["invalid_or_ambiguous_groups"] == 0
        ),
        "zero_backend_failures": _metric_summary(groups)["backend_failures"] == 0,
        "zero_reward_contract_inconsistencies": (
            _metric_summary(groups)["reward_contract_inconsistencies"] == 0
        ),
        "deterministic_fresh_process_replay": deterministic_passed,
    }
    quantitative_conditions = {
        name: passed
        for name, passed in conditions.items()
        if name != "deterministic_fresh_process_replay"
    }
    return {
        "all_groups": _metric_summary(groups),
        "task_groups": task_summary,
        "task_families": family,
        "shared_replay_groups": _metric_summary(replay),
        "informative_task_group_count": len(informative),
        "informative_task_families": informative_families,
        "usable_informative_candidates": candidates,
        "viability_conditions": conditions,
        "quantitative_viability_passed": all(quantitative_conditions.values()),
        "deterministic_replay_status": (
            "pending" if deterministic_replay is None else deterministic_replay
        ),
        "viability_passed": all(conditions.values()),
    }


def _overlap(
    generic: Sequence[Mapping[str, Any]],
    targeted: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    shared: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = [
        (left, right)
        for left, right in zip(generic, targeted, strict=True)
        if left.get("prompt_sha256") == right.get("prompt_sha256")
    ]
    intersections = 0
    unions = 0
    for left, right in shared:
        left_hashes = set(cast(list[str], left["completion_sha256s"]))
        right_hashes = set(cast(list[str], right["completion_sha256s"]))
        intersections += len(left_hashes & right_hashes)
        unions += len(left_hashes | right_hashes)
    return {
        "shared_prompt_positions": [left["schedule_position"] for left, _ in shared],
        "shared_prompt_position_count": len(shared),
        "groups_saturated_in_both": sum(
            left.get("reward_variance") == 0.0 and right.get("reward_variance") == 0.0
            for left, right in shared
        ),
        "groups_informative_in_both": sum(
            float(cast(float, left["reward_variance"])) > 0.0
            and float(cast(float, right["reward_variance"])) > 0.0
            for left, right in shared
        ),
        "groups_informative_only_generic": sum(
            float(cast(float, left["reward_variance"])) > 0.0
            and right.get("reward_variance") == 0.0
            for left, right in shared
        ),
        "groups_informative_only_targeted": sum(
            left.get("reward_variance") == 0.0
            and float(cast(float, right["reward_variance"])) > 0.0
            for left, right in shared
        ),
        "completion_output_overlap_count": intersections,
        "completion_output_union_count": unions,
        "completion_output_overlap_ratio": intersections / unions if unions else 0.0,
        "reward_pattern_overlap_count": sum(
            left.get("reward_vector") == right.get("reward_vector") for left, right in shared
        ),
        "reward_pattern_overlap_ratio": (
            sum(left.get("reward_vector") == right.get("reward_vector") for left, right in shared)
            / len(shared)
            if shared
            else 0.0
        ),
    }


def _failure_attribution(arms: Mapping[str, Mapping[str, Any]]) -> str:
    summaries = [cast(Mapping[str, Any], value["all_groups"]) for value in arms.values()]
    indistinguishable = sum(
        cast(int, summary["reward_indistinguishable_diversity_groups"]) for summary in summaries
    )
    uniform = sum(
        cast(int, summary["zero_variance_classifications"]["all_correct_saturated"])
        + cast(int, summary["zero_variance_classifications"]["all_incorrect_saturated"])
        + cast(int, summary["zero_variance_classifications"]["output_identical"])
        for summary in summaries
    )
    if indistinguishable and uniform:
        return "mixed_prompt_saturation_and_reward_resolution_failure"
    if indistinguishable:
        return "reward_resolution_failure"
    return "prompt_saturation"


def build_signal_summary(
    generic_raw: Mapping[str, Any],
    targeted_raw: Mapping[str, Any],
    *,
    deterministic_replays: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Build the content-free two-arm signal-density and viability decision."""

    groups = {
        "generic": _group_rows(generic_raw, "generic"),
        "targeted": _group_rows(targeted_raw, "targeted"),
    }
    arm_summaries = {
        arm: _arm_summary(
            groups[arm],
            None if deterministic_replays is None else deterministic_replays.get(arm),
        )
        for arm in ARMS
    }
    quantitative_passed = all(
        cast(bool, arm_summaries[arm]["quantitative_viability_passed"]) for arm in ARMS
    )
    viability_ready = not quantitative_passed or deterministic_replays is not None
    viability_passed = viability_ready and all(
        cast(bool, arm_summaries[arm]["viability_passed"]) for arm in ARMS
    )
    if viability_passed:
        decision = "schedule_viable"
        attribution: str | None = None
    elif viability_ready:
        decision = "reward_signal_insufficient"
        attribution = _failure_attribution(arm_summaries)
    else:
        decision = "deterministic_replay_required"
        attribution = None
    result: dict[str, object] = {
        "schema_version": 1,
        "summary_id": "foundry-l3-grpo-signal-density-summary-v1",
        "audit_id": AUDIT_ID,
        "method_contract_sha256": signal_audit_method_contract()["method_contract_sha256"],
        "arms": arm_summaries,
        "generic_targeted_overlap": _overlap(groups["generic"], groups["targeted"]),
        "quantitative_viability_passed": quantitative_passed,
        "viability_ready": viability_ready,
        "viability_passed": viability_passed,
        "decision": decision,
        "failure_attribution": attribution,
        "counted_training_authorized": False,
        "counted_training_started": False,
        "holdout_v2_started": False,
        "gsm1k_started": False,
        "sealed_content_use": 0,
    }
    result["signal_summary_sha256"] = canonical_sha256(result)
    return result
