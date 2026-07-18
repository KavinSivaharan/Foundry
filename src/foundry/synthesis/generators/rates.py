"""Exact rate, ratio, percentage, and weighted-average generation."""

from __future__ import annotations

import random
from fractions import Fraction

from foundry.synthesis.generators import (
    CandidateDraft,
    GeneratorVerification,
    candidate_id,
    exact_value,
    training_completion,
    verification_result,
)
from foundry.synthesis.schema import (
    DifficultyLevel,
    LatentProgramSpec,
    ProgramParameter,
    ProgramStep,
)
from foundry.synthesis.taxonomy import FailureCategory

GENERATOR_ID = "exact-rate-ratio-relations"
GENERATOR_VERSION = "1"

_CONTEXTS = (
    "a solar ink printer",
    "a deep-ocean beacon lab",
    "an alpine lens workshop",
    "a floating algae nursery",
    "a desert acoustics station",
    "an orbital fabric loom",
    "a polar battery archive",
    "a volcanic crystal kiln",
)
_OBJECTS = (
    "signal cards",
    "sample capsules",
    "lens blanks",
    "culture panels",
    "echo tags",
    "woven strips",
    "power cells",
    "crystal plates",
)
_PERCENTS = (10, 20, 25, 40, 50, 60, 75, 80)


def _mode(variant: int) -> str:
    return ("rate_total", "ratio_scale", "percentage", "weighted_average", "combined_rate")[
        variant % 5
    ]


def generate_rates(
    *,
    seed: int,
    difficulty: DifficultyLevel,
    variant: int,
    output_contract_enabled: bool,
) -> CandidateDraft:
    """Generate one exact rational-relationship candidate without external text."""

    if seed < 0 or variant < 0:
        raise ValueError("rate seed and variant must be non-negative")
    rng = random.Random(seed)
    mode = _mode(variant)
    context = _CONTEXTS[variant % len(_CONTEXTS)]
    item = _OBJECTS[(variant * 3) % len(_OBJECTS)]
    parameters: list[ProgramParameter] = []
    steps: list[ProgramStep] = []
    trace: list[str] = []
    payload: dict[str, object] = {"mode": mode, "unit_check": "exact"}

    if mode == "rate_total":
        rate = rng.randint(3, 14)
        intervals = rng.randint(2, 7) + (2 if difficulty is DifficultyLevel.HARD else 0)
        answer_fraction = Fraction(rate * intervals)
        parameters.extend(
            (
                ProgramParameter("rate", exact_value(rate), "items/interval"),
                ProgramParameter("intervals", exact_value(intervals), "intervals"),
            )
        )
        steps.append(
            ProgramStep("total", "multiply", ("rate", "intervals"), exact_value(answer_fraction))
        )
        question = (
            f"At {context}, one calibrated cycle produces {rate} {item}. "
            f"The device completes exactly {intervals} cycles. What total number of {item} "
            "does it produce?"
        )
        trace.append(f"Multiply the exact per-cycle rate {rate} by {intervals} cycles.")
        payload.update(rate=rate, intervals=intervals)
        answer_symbol = "total"
    elif mode == "ratio_scale":
        first_part = rng.randint(2, 7)
        second_part = rng.randint(2, 9)
        if first_part == second_part:
            second_part += 1
        scale = rng.randint(3, 10)
        known = first_part * scale
        answer_fraction = Fraction(second_part * scale)
        parameters.extend(
            (
                ProgramParameter("first_part", exact_value(first_part), "ratio_part"),
                ProgramParameter("second_part", exact_value(second_part), "ratio_part"),
                ProgramParameter("known", exact_value(known), "items"),
            )
        )
        steps.extend(
            (
                ProgramStep("scale", "divide", ("known", "first_part"), exact_value(scale)),
                ProgramStep(
                    "paired_amount",
                    "multiply",
                    ("scale", "second_part"),
                    exact_value(answer_fraction),
                ),
            )
        )
        question = (
            f"At {context}, amber and cobalt {item} are prepared in the exact ratio "
            f"{first_part}:{second_part}. If the amber share contains {known}, how many "
            "belong to the cobalt share?"
        )
        trace.extend(
            (
                f"Divide {known} by the first ratio part {first_part} to obtain scale {scale}.",
                f"Multiply scale {scale} by the second part {second_part}.",
            )
        )
        payload.update(first_part=first_part, second_part=second_part, known=known)
        answer_symbol = "paired_amount"
    elif mode == "percentage":
        percent = _PERCENTS[variant % len(_PERCENTS)]
        base = rng.randint(4, 16) * 100
        answer_fraction = Fraction(base * percent, 100)
        parameters.extend(
            (
                ProgramParameter("base", exact_value(base), "items"),
                ProgramParameter("percent", exact_value(percent), "percent"),
                ProgramParameter("hundred", exact_value(100), "percent_base"),
            )
        )
        steps.extend(
            (
                ProgramStep("scaled", "multiply", ("base", "percent"), exact_value(base * percent)),
                ProgramStep(
                    "selected",
                    "divide",
                    ("scaled", "hundred"),
                    exact_value(answer_fraction),
                ),
            )
        )
        question = (
            f"At {context}, a quality scan examines {percent}% of an independently prepared "
            f"batch of {base} {item}. How many {item} are examined?"
        )
        trace.extend(
            (
                f"Represent {percent}% exactly as {percent}/100.",
                f"Multiply {base} by {percent}/100 without floating point.",
            )
        )
        payload.update(base=base, percent=percent)
        answer_symbol = "selected"
    elif mode == "weighted_average":
        group_count = 3 if difficulty is DifficultyLevel.HARD else 2
        weights = [rng.randint(2, 7) for _ in range(group_count)]
        values = [rng.randint(4, 18) for _ in range(group_count)]
        weighted_total = sum(weight * value for weight, value in zip(weights, values, strict=True))
        total_weight = sum(weights)
        answer_fraction = Fraction(weighted_total, total_weight)
        for index, (weight, value) in enumerate(zip(weights, values, strict=True), start=1):
            parameters.extend(
                (
                    ProgramParameter(f"weight_{index}", exact_value(weight), "panels"),
                    ProgramParameter(f"value_{index}", exact_value(value), "marks/panel"),
                )
            )
            steps.append(
                ProgramStep(
                    f"weighted_{index}",
                    "multiply",
                    (f"weight_{index}", f"value_{index}"),
                    exact_value(weight * value),
                )
            )
        parameters.append(ProgramParameter("total_weight", exact_value(total_weight), "panels"))
        steps.append(
            ProgramStep(
                "weighted_total",
                "sum_many",
                tuple(f"weighted_{index}" for index in range(1, group_count + 1)),
                exact_value(weighted_total),
            )
        )
        steps.append(
            ProgramStep(
                "weighted_mean",
                "divide",
                ("weighted_total", "total_weight"),
                exact_value(answer_fraction),
            )
        )
        groups = "; ".join(
            f"{weight} panels carry {value} marks each"
            for weight, value in zip(weights, values, strict=True)
        )
        question = (
            f"At {context}, {groups}. What is the exact weighted average number of marks per panel?"
        )
        trace.extend(
            (
                f"The exact weighted total is {weighted_total}.",
                f"Divide by the total weight {total_weight} to obtain the weighted average.",
            )
        )
        payload.update(weights=weights, values=values)
        answer_symbol = "weighted_mean"
    else:
        first_rate = rng.randint(2, 9)
        second_rate = rng.randint(2, 9)
        intervals = rng.randint(3, 8)
        answer_fraction = Fraction((first_rate + second_rate) * intervals)
        parameters.extend(
            (
                ProgramParameter("first_rate", exact_value(first_rate), "items/interval"),
                ProgramParameter("second_rate", exact_value(second_rate), "items/interval"),
                ProgramParameter("intervals", exact_value(intervals), "intervals"),
            )
        )
        steps.extend(
            (
                ProgramStep(
                    "combined_rate",
                    "add",
                    ("first_rate", "second_rate"),
                    exact_value(first_rate + second_rate),
                ),
                ProgramStep(
                    "combined_total",
                    "multiply",
                    ("combined_rate", "intervals"),
                    exact_value(answer_fraction),
                ),
            )
        )
        question = (
            f"At {context}, two independent channels deliver {first_rate} and {second_rate} "
            f"{item} per interval. Both run for {intervals} intervals. How many arrive altogether?"
        )
        trace.extend(
            (
                f"Add the two compatible rates to obtain {first_rate + second_rate} per interval.",
                f"Multiply that exact rate by {intervals} intervals.",
            )
        )
        payload.update(first_rate=first_rate, second_rate=second_rate, intervals=intervals)
        answer_symbol = "combined_total"

    answer = exact_value(answer_fraction)
    trace_tuple = tuple(trace)
    program = LatentProgramSpec(
        program_family=f"{GENERATOR_ID}:{mode}",
        parameters=tuple(parameters),
        steps=tuple(steps),
        constraints=(
            "relationships use exact rational arithmetic",
            "units are dimensionally valid",
        ),
        answer_symbol=answer_symbol,
    )
    return CandidateDraft(
        candidate_id=candidate_id(GENERATOR_ID, seed, variant),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        random_seed=seed,
        target_failure_category=FailureCategory.RATE_RATIO_PERCENTAGE,
        secondary_skill_tags=(mode, "exact_rational"),
        difficulty_level=difficulty,
        output_contract_enabled=output_contract_enabled,
        latent_program=program,
        rendered_question=question,
        deterministic_solution_trace=trace_tuple,
        canonical_final_answer=answer,
        training_completion=training_completion(
            trace_tuple, answer, output_contract_enabled=output_contract_enabled
        ),
        structure_signature={
            "generator": GENERATOR_ID,
            "mode": mode,
            "difficulty": difficulty,
            "template_variant": variant,
            "step_operations": [step.operation for step in steps],
            "answer_symbol": answer_symbol,
        },
        verifier_payload=payload,
    )


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid integer payload field {key}")
    return value


def _payload_int_list(payload: dict[str, object], key: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"invalid integer-list payload field {key}")
    return [int(item) for item in value]


def _equation_answer(payload: dict[str, object]) -> Fraction:
    mode = payload.get("mode")
    if mode == "rate_total":
        return Fraction(_payload_int(payload, "rate") * _payload_int(payload, "intervals"))
    if mode == "ratio_scale":
        return Fraction(
            _payload_int(payload, "known") * _payload_int(payload, "second_part"),
            _payload_int(payload, "first_part"),
        )
    if mode == "percentage":
        return Fraction(_payload_int(payload, "base") * _payload_int(payload, "percent"), 100)
    if mode == "weighted_average":
        weights = _payload_int_list(payload, "weights")
        values = _payload_int_list(payload, "values")
        if len(weights) != len(values) or not weights:
            raise ValueError("weighted-average payload lengths differ")
        return Fraction(
            sum(weight * value for weight, value in zip(weights, values, strict=True)),
            sum(weights),
        )
    if mode == "combined_rate":
        return Fraction(
            (_payload_int(payload, "first_rate") + _payload_int(payload, "second_rate"))
            * _payload_int(payload, "intervals")
        )
    raise ValueError("unsupported rational-relation mode")


def verify_rate_equation(draft: CandidateDraft) -> GeneratorVerification:
    """Compute the answer directly from the exact mode equation."""

    try:
        answer = _equation_answer(draft.verifier_payload)
    except (ValueError, ZeroDivisionError) as error:
        return verification_result(
            verifier_id="rate-equation-v1",
            method_family="exact_rational_equation",
            answer=None,
            failure_reason="invalid_equation_payload",
            evidence_payload={"error": type(error).__name__},
        )
    return verification_result(
        verifier_id="rate-equation-v1",
        method_family="exact_rational_equation",
        answer=answer,
        failure_reason=None,
        evidence_payload={"mode": draft.verifier_payload.get("mode")},
    )


def verify_rate_inverse(draft: CandidateDraft) -> GeneratorVerification:
    """Check the answer with cross-products, inverse substitution, and units."""

    payload = draft.verifier_payload
    if payload.get("unit_check") != "exact":
        return verification_result(
            verifier_id="rate-inverse-v1",
            method_family="cross_product_inverse_units",
            answer=None,
            failure_reason="unit_consistency_failure",
            evidence_payload={},
        )
    try:
        candidate = draft.canonical_final_answer.fraction
        mode = payload.get("mode")
        if mode == "rate_total":
            valid = candidate / _payload_int(payload, "intervals") == _payload_int(payload, "rate")
        elif mode == "ratio_scale":
            valid = _payload_int(payload, "known") * _payload_int(
                payload, "second_part"
            ) == candidate * _payload_int(payload, "first_part")
        elif mode == "percentage":
            valid = candidate * 100 == _payload_int(payload, "base") * _payload_int(
                payload, "percent"
            )
        elif mode == "weighted_average":
            weights = _payload_int_list(payload, "weights")
            values = _payload_int_list(payload, "values")
            valid = candidate * sum(weights) == sum(
                weight * value for weight, value in zip(weights, values, strict=True)
            )
        elif mode == "combined_rate":
            valid = candidate / _payload_int(payload, "intervals") == (
                _payload_int(payload, "first_rate") + _payload_int(payload, "second_rate")
            )
        else:
            valid = False
    except (ValueError, ZeroDivisionError):
        valid = False
    if not valid:
        return verification_result(
            verifier_id="rate-inverse-v1",
            method_family="cross_product_inverse_units",
            answer=None,
            failure_reason="inverse_or_cross_product_failure",
            evidence_payload={"mode": payload.get("mode")},
        )
    return verification_result(
        verifier_id="rate-inverse-v1",
        method_family="cross_product_inverse_units",
        answer=candidate,
        failure_reason=None,
        evidence_payload={"mode": payload.get("mode"), "unit_check": True},
    )


def validate_rate_constraints(draft: CandidateDraft) -> tuple[str, ...]:
    """Reject unsafe or ambiguous rational candidates."""

    reasons: list[str] = []
    if draft.ambiguity_flags:
        reasons.append("ambiguous_target")
    if draft.canonical_final_answer.denominator <= 0:
        reasons.append("invalid_denominator")
    if draft.verifier_payload.get("unit_check") != "exact":
        reasons.append("unit_consistency_failure")
    return tuple(reasons)
