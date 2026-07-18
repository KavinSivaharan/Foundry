"""Bounded integer allocation generation with constructive and enumerative checks."""

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

GENERATOR_ID = "bounded-discrete-allocation"
GENERATOR_VERSION = "1"

_CONTEXTS = (
    "a comet-sample registry",
    "an underwater mosaic shop",
    "a polar drone hangar",
    "a high-altitude seed exchange",
    "a tidal instrument depot",
    "a basalt sculpture yard",
    "an orbital repair cooperative",
    "a desert light laboratory",
)
_OBJECTS = (
    "sample cases",
    "mosaic frames",
    "drone kits",
    "seed canisters",
    "instrument racks",
    "stone modules",
    "repair bundles",
    "light panels",
)


def _mode(variant: int) -> str:
    return ("two_type_allocation", "complete_packages", "equal_distribution", "dual_capacity")[
        variant % 4
    ]


def generate_discrete(
    *,
    seed: int,
    difficulty: DifficultyLevel,
    variant: int,
    output_contract_enabled: bool,
) -> CandidateDraft:
    """Generate one bounded integer problem with a well-defined target."""

    if seed < 0 or variant < 0:
        raise ValueError("discrete seed and variant must be non-negative")
    rng = random.Random(seed)
    mode = _mode(variant)
    context = _CONTEXTS[variant % len(_CONTEXTS)]
    item = _OBJECTS[(variant * 5) % len(_OBJECTS)]
    parameters: list[ProgramParameter] = []
    steps: list[ProgramStep] = []
    trace: list[str] = []
    payload: dict[str, object] = {"mode": mode, "finite_domain": True}

    if mode == "two_type_allocation":
        total = rng.randint(8, 18) + (8 if difficulty is DifficultyLevel.HARD else 0)
        first_count = rng.randint(2, total - 2)
        second_count = total - first_count
        first_cost = rng.randint(2, 6)
        second_cost = rng.randint(7, 12)
        resource_total = first_count * first_cost + second_count * second_cost
        answer_fraction = Fraction(first_count)
        parameters.extend(
            (
                ProgramParameter("total", exact_value(total), "assemblies"),
                ProgramParameter("resource_total", exact_value(resource_total), "parts"),
                ProgramParameter("first_cost", exact_value(first_cost), "parts/assembly"),
                ProgramParameter("second_cost", exact_value(second_cost), "parts/assembly"),
            )
        )
        steps.extend(
            (
                ProgramStep(
                    "all_second_cost",
                    "multiply",
                    ("total", "second_cost"),
                    exact_value(total * second_cost),
                ),
                ProgramStep(
                    "resource_difference",
                    "subtract",
                    ("all_second_cost", "resource_total"),
                    exact_value(total * second_cost - resource_total),
                ),
            )
        )
        parameters.append(
            ProgramParameter(
                "cost_difference", exact_value(second_cost - first_cost), "parts/assembly"
            )
        )
        steps.append(
            ProgramStep(
                "first_count",
                "divide",
                ("resource_difference", "cost_difference"),
                exact_value(first_count),
            )
        )
        question = (
            f"At {context}, exactly {total} {item} are assembled in two designs. Design A uses "
            f"{first_cost} calibrated parts and design B uses {second_cost}. The complete batch "
            f"uses {resource_total} parts. How many are design A?"
        )
        trace.extend(
            (
                f"Treat all {total} as design B, which would use {total * second_cost} parts.",
                f"The resource difference is {total * second_cost - resource_total}.",
                f"Divide by the per-design difference {second_cost - first_cost}.",
            )
        )
        payload.update(
            total=total,
            resource_total=resource_total,
            first_cost=first_cost,
            second_cost=second_cost,
        )
        answer_symbol = "first_count"
    elif mode == "complete_packages":
        package_size = rng.randint(3, 9)
        complete = rng.randint(4, 14) + (8 if difficulty is DifficultyLevel.HARD else 0)
        remainder = rng.randint(0, package_size - 1)
        total = complete * package_size + remainder
        answer_fraction = Fraction(complete)
        parameters.extend(
            (
                ProgramParameter("total", exact_value(total), "items"),
                ProgramParameter("package_size", exact_value(package_size), "items/package"),
            )
        )
        steps.append(
            ProgramStep(
                "complete_packages",
                "floor_divide",
                ("total", "package_size"),
                exact_value(complete),
            )
        )
        question = (
            f"At {context}, {total} {item} must be placed into complete packages of exactly "
            f"{package_size}. Partial packages do not count. What is the maximum number of "
            "complete packages?"
        )
        trace.append(
            f"Use exact integer division: {total} divided by {package_size} gives {complete}."
        )
        payload.update(total=total, package_size=package_size)
        answer_symbol = "complete_packages"
    elif mode == "equal_distribution":
        containers = rng.randint(3, 9)
        each = rng.randint(4, 15) + (5 if difficulty is DifficultyLevel.HARD else 0)
        total = containers * each
        answer_fraction = Fraction(each)
        parameters.extend(
            (
                ProgramParameter("total", exact_value(total), "items"),
                ProgramParameter("containers", exact_value(containers), "containers"),
            )
        )
        steps.append(
            ProgramStep(
                "per_container",
                "divide",
                ("total", "containers"),
                exact_value(each),
            )
        )
        question = (
            f"At {context}, {total} {item} are distributed equally among {containers} sealed "
            "containers with none left over. How many go into each container?"
        )
        trace.append(f"Exact equal distribution gives {total}/{containers} = {each} per container.")
        payload.update(total=total, containers=containers)
        answer_symbol = "per_container"
    else:
        first_per = rng.randint(2, 6)
        second_per = rng.randint(2, 7)
        target = rng.randint(5, 14) + (5 if difficulty is DifficultyLevel.HARD else 0)
        first_extra = rng.randint(0, first_per - 1)
        second_extra = rng.randint(0, second_per - 1)
        first_resource = target * first_per + first_extra
        second_resource = target * second_per + second_extra
        answer_fraction = Fraction(target)
        parameters.extend(
            (
                ProgramParameter("first_resource", exact_value(first_resource), "parts"),
                ProgramParameter("second_resource", exact_value(second_resource), "parts"),
                ProgramParameter("first_per", exact_value(first_per), "parts/assembly"),
                ProgramParameter("second_per", exact_value(second_per), "parts/assembly"),
            )
        )
        steps.extend(
            (
                ProgramStep(
                    "first_capacity",
                    "floor_divide",
                    ("first_resource", "first_per"),
                    exact_value(first_resource // first_per),
                ),
                ProgramStep(
                    "second_capacity",
                    "floor_divide",
                    ("second_resource", "second_per"),
                    exact_value(second_resource // second_per),
                ),
                ProgramStep(
                    "assembly_capacity",
                    "minimum",
                    ("first_capacity", "second_capacity"),
                    exact_value(target),
                ),
            )
        )
        question = (
            f"At {context}, each {item} needs {first_per} amber parts and {second_per} cobalt "
            f"parts. Stocks contain {first_resource} amber and {second_resource} cobalt parts. "
            f"What maximum number of complete {item} can be assembled?"
        )
        trace.extend(
            (
                f"Amber parts permit {first_resource // first_per} complete assemblies.",
                f"Cobalt parts permit {second_resource // second_per} complete assemblies.",
                "The smaller exact capacity controls the result.",
            )
        )
        payload.update(
            first_resource=first_resource,
            second_resource=second_resource,
            first_per=first_per,
            second_per=second_per,
        )
        answer_symbol = "assembly_capacity"

    answer = exact_value(answer_fraction)
    trace_tuple = tuple(trace)
    program = LatentProgramSpec(
        program_family=f"{GENERATOR_ID}:{mode}",
        parameters=tuple(parameters),
        steps=tuple(steps),
        constraints=(
            "all decision variables are nonnegative integers",
            "the requested target is unique within a finite domain",
        ),
        answer_symbol=answer_symbol,
    )
    return CandidateDraft(
        candidate_id=candidate_id(GENERATOR_ID, seed, variant),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        random_seed=seed,
        target_failure_category=FailureCategory.CONSTRAINT_DISCRETE,
        secondary_skill_tags=(mode, "bounded_integer_domain"),
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
            "constraint_topology": [step.operation for step in steps],
            "answer_symbol": answer_symbol,
        },
        verifier_payload=payload,
    )


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid integer payload field {key}")
    return value


def _constructive_answer(payload: dict[str, object]) -> Fraction:
    mode = payload.get("mode")
    if mode == "two_type_allocation":
        total = _payload_int(payload, "total")
        resource_total = _payload_int(payload, "resource_total")
        first_cost = _payload_int(payload, "first_cost")
        second_cost = _payload_int(payload, "second_cost")
        return Fraction(total * second_cost - resource_total, second_cost - first_cost)
    if mode == "complete_packages":
        return Fraction(_payload_int(payload, "total") // _payload_int(payload, "package_size"))
    if mode == "equal_distribution":
        return Fraction(_payload_int(payload, "total"), _payload_int(payload, "containers"))
    if mode == "dual_capacity":
        return Fraction(
            min(
                _payload_int(payload, "first_resource") // _payload_int(payload, "first_per"),
                _payload_int(payload, "second_resource") // _payload_int(payload, "second_per"),
            )
        )
    raise ValueError("unsupported discrete mode")


def verify_discrete_constructive(draft: CandidateDraft) -> GeneratorVerification:
    """Solve the sampled bounded constraint directly."""

    try:
        answer = _constructive_answer(draft.verifier_payload)
    except (ValueError, ZeroDivisionError) as error:
        return verification_result(
            verifier_id="discrete-constructive-v1",
            method_family="constructive_integer_solver",
            answer=None,
            failure_reason="constructive_solver_failure",
            evidence_payload={"error": type(error).__name__},
        )
    if answer.denominator != 1 or answer < 0:
        return verification_result(
            verifier_id="discrete-constructive-v1",
            method_family="constructive_integer_solver",
            answer=None,
            failure_reason="nonintegral_or_negative_solution",
            evidence_payload={"mode": draft.verifier_payload.get("mode")},
        )
    return verification_result(
        verifier_id="discrete-constructive-v1",
        method_family="constructive_integer_solver",
        answer=answer,
        failure_reason=None,
        evidence_payload={"mode": draft.verifier_payload.get("mode")},
    )


def _enumerated_solutions(payload: dict[str, object]) -> list[int]:
    mode = payload.get("mode")
    if mode == "two_type_allocation":
        total = _payload_int(payload, "total")
        resource_total = _payload_int(payload, "resource_total")
        first_cost = _payload_int(payload, "first_cost")
        second_cost = _payload_int(payload, "second_cost")
        return [
            first
            for first in range(total + 1)
            if first * first_cost + (total - first) * second_cost == resource_total
        ]
    if mode == "complete_packages":
        total = _payload_int(payload, "total")
        size = _payload_int(payload, "package_size")
        return [
            count
            for count in range(total + 1)
            if count * size <= total and (count + 1) * size > total
        ]
    if mode == "equal_distribution":
        total = _payload_int(payload, "total")
        containers = _payload_int(payload, "containers")
        return [
            per_container
            for per_container in range(total + 1)
            if per_container * containers == total
        ]
    if mode == "dual_capacity":
        first_resource = _payload_int(payload, "first_resource")
        second_resource = _payload_int(payload, "second_resource")
        first_per = _payload_int(payload, "first_per")
        second_per = _payload_int(payload, "second_per")
        upper = max(first_resource, second_resource)
        return [
            count
            for count in range(upper + 1)
            if count * first_per <= first_resource
            and count * second_per <= second_resource
            and (
                (count + 1) * first_per > first_resource
                or (count + 1) * second_per > second_resource
            )
        ]
    raise ValueError("unsupported discrete mode")


def verify_discrete_enumeration(draft: CandidateDraft) -> GeneratorVerification:
    """Enumerate the finite domain and require exactly one requested result."""

    try:
        solutions = _enumerated_solutions(draft.verifier_payload)
    except (ValueError, ZeroDivisionError) as error:
        return verification_result(
            verifier_id="discrete-enumerator-v1",
            method_family="bounded_brute_force",
            answer=None,
            failure_reason="enumeration_failure",
            evidence_payload={"error": type(error).__name__},
        )
    if len(solutions) != 1:
        return verification_result(
            verifier_id="discrete-enumerator-v1",
            method_family="bounded_brute_force",
            answer=None,
            failure_reason="nonunique_or_missing_solution",
            evidence_payload={"solution_count": len(solutions)},
        )
    return verification_result(
        verifier_id="discrete-enumerator-v1",
        method_family="bounded_brute_force",
        answer=Fraction(solutions[0]),
        failure_reason=None,
        evidence_payload={"solution_count": 1},
    )


def validate_discrete_constraints(draft: CandidateDraft) -> tuple[str, ...]:
    """Reject ambiguity, nonintegrality, or a nonunique finite target."""

    reasons: list[str] = []
    if draft.ambiguity_flags:
        reasons.append("ambiguous_target")
    if draft.canonical_final_answer.denominator != 1:
        reasons.append("nonintegral_discrete_target")
    try:
        if len(_enumerated_solutions(draft.verifier_payload)) != 1:
            reasons.append("nonunique_or_missing_solution")
    except (ValueError, ZeroDivisionError):
        reasons.append("invalid_constraint_payload")
    return tuple(reasons)
