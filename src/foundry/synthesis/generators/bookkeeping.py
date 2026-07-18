"""Independent multi-step state-transition generator and verifier pair."""

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

GENERATOR_ID = "bookkeeping-state-transitions"
GENERATOR_VERSION = "1"

_LOCATIONS = (
    "orbital herbarium",
    "deep-sea archive",
    "volcanic glass studio",
    "polar observatory",
    "desert seed vault",
    "floating repair dock",
    "mountain signal station",
    "underground map library",
    "tidal research garden",
    "solar instrument workshop",
)
_ITEMS = (
    "silverleaf trays",
    "ceramic markers",
    "cobalt tiles",
    "calibration rings",
    "sealed specimen tubes",
    "navigation prisms",
    "etched index plates",
    "thermal sensor clips",
    "woven sample sleeves",
    "quartz alignment blocks",
)
_ADD_PHRASES = ("receives", "recovers", "assembles", "adds from reserve", "accepts")
_SUBTRACT_PHRASES = ("ships away", "retires", "transfers out", "uses", "sets aside")
_QUESTION_PHRASES = (
    "What exact quantity is left in the active inventory?",
    "How many units does the final ledger show?",
    "What is the resulting stored quantity?",
    "How large is the inventory after every recorded change?",
    "What final amount remains available?",
)


def _update_count(difficulty: DifficultyLevel) -> int:
    return {
        DifficultyLevel.EASY: 2,
        DifficultyLevel.MEDIUM: 3,
        DifficultyLevel.HARD: 4,
    }[difficulty]


def generate_bookkeeping(
    *,
    seed: int,
    difficulty: DifficultyLevel,
    variant: int,
    output_contract_enabled: bool,
) -> CandidateDraft:
    """Generate one exact, independently created bookkeeping candidate."""

    if seed < 0 or variant < 0:
        raise ValueError("bookkeeping seed and variant must be non-negative")
    rng = random.Random(seed)
    grouping = variant % 4 == 3
    transfer_mode = variant % 4 in {1, 2}
    update_count = _update_count(difficulty)
    group_size = rng.randint(2, 6) if grouping else 1
    start = rng.randint(70, 150) * group_size
    parameters = [ProgramParameter("start", exact_value(start), "items")]
    steps: list[ProgramStep] = []
    trace: list[str] = [f"Begin with {start} items in the active ledger."]
    clauses: list[str] = []
    operations: list[str] = []
    current = Fraction(start)
    current_symbol = "start"
    signed_updates: list[int] = []
    for index in range(update_count):
        magnitude = rng.randint(3, 18) * group_size
        add = ((variant >> index) & 1) == 0
        signed = magnitude if add else -magnitude
        if current + signed <= 0:
            signed = magnitude
            add = True
        parameter = f"change_{index + 1}"
        output = f"state_{index + 1}"
        parameters.append(ProgramParameter(parameter, exact_value(magnitude), "items"))
        current = current + signed
        operation = "add" if add else "subtract"
        steps.append(
            ProgramStep(
                output_symbol=output,
                operation=operation,
                input_symbols=(current_symbol, parameter),
                exact_result=exact_value(current),
            )
        )
        phrase_pool = _ADD_PHRASES if add else _SUBTRACT_PHRASES
        phrase = phrase_pool[(variant + index) % len(phrase_pool)]
        if transfer_mode and add:
            phrase = "receives by transfer"
        clauses.append(f"it {phrase} {magnitude} {_ITEMS[(variant + index) % len(_ITEMS)]}")
        trace.append(f"Apply {operation} {magnitude}; the exact state becomes {current}.")
        operations.append(operation)
        signed_updates.append(signed)
        current_symbol = output

    target_kind = "items"
    if grouping:
        parameters.append(ProgramParameter("group_size", exact_value(group_size), "items/group"))
        groups = current / group_size
        steps.append(
            ProgramStep(
                output_symbol="group_count",
                operation="divide",
                input_symbols=(current_symbol, "group_size"),
                exact_result=exact_value(groups),
            )
        )
        trace.append(
            f"Divide {current} items into groups of {group_size}; this gives {groups} groups."
        )
        clauses.append(f"the final stock is packed into equal groups of {group_size}")
        current = groups
        current_symbol = "group_count"
        target_kind = "groups"
        question = "How many complete equal groups are formed after every change?"
    else:
        question = _QUESTION_PHRASES[variant % len(_QUESTION_PHRASES)]

    location = _LOCATIONS[variant % len(_LOCATIONS)]
    item = _ITEMS[(variant * 3) % len(_ITEMS)]
    rendered = (
        f"At the {location}, the active ledger begins with {start} {item}. "
        + "; then ".join(clauses)
        + f". {question}"
    )
    answer = exact_value(current)
    program = LatentProgramSpec(
        program_family=f"{GENERATOR_ID}:{'grouping' if grouping else 'inventory'}",
        parameters=tuple(parameters),
        steps=tuple(steps),
        constraints=("all ledger states remain positive", "all grouping is exact"),
        answer_symbol=current_symbol,
    )
    trace_tuple = tuple(trace)
    return CandidateDraft(
        candidate_id=candidate_id(GENERATOR_ID, seed, variant),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        random_seed=seed,
        target_failure_category=FailureCategory.MULTI_STEP_BOOKKEEPING,
        secondary_skill_tags=("state_transition", target_kind),
        difficulty_level=difficulty,
        output_contract_enabled=output_contract_enabled,
        latent_program=program,
        rendered_question=rendered,
        deterministic_solution_trace=trace_tuple,
        canonical_final_answer=answer,
        training_completion=training_completion(
            trace_tuple, answer, output_contract_enabled=output_contract_enabled
        ),
        structure_signature={
            "generator": GENERATOR_ID,
            "mode": "grouping" if grouping else "inventory",
            "difficulty": difficulty,
            "template_variant": variant,
            "operations": operations + (["divide"] if grouping else []),
            "topology": "linear_state_chain",
            "target_kind": target_kind,
        },
        verifier_payload={
            "start": start,
            "signed_updates": signed_updates,
            "group_size": group_size,
            "grouping": grouping,
        },
    )


def verify_bookkeeping_dag(draft: CandidateDraft) -> GeneratorVerification:
    """Execute the latent DAG and validate every recorded exact intermediate."""

    values = {
        parameter.name: parameter.value.fraction for parameter in draft.latent_program.parameters
    }
    for step in draft.latent_program.steps:
        left = values[step.input_symbols[0]]
        right = values[step.input_symbols[1]]
        if step.operation == "add":
            computed = left + right
        elif step.operation == "subtract":
            computed = left - right
        elif step.operation == "divide" and right != 0:
            computed = left / right
        else:
            return verification_result(
                verifier_id="bookkeeping-dag-v1",
                method_family="exact_arithmetic_dag",
                answer=None,
                failure_reason="unsupported_or_invalid_operation",
                evidence_payload={"step": step.output_symbol},
            )
        if computed != step.exact_result.fraction:
            return verification_result(
                verifier_id="bookkeeping-dag-v1",
                method_family="exact_arithmetic_dag",
                answer=None,
                failure_reason="latent_intermediate_disagreement",
                evidence_payload={"step": step.output_symbol},
            )
        values[step.output_symbol] = computed
    answer = values[draft.latent_program.answer_symbol]
    return verification_result(
        verifier_id="bookkeeping-dag-v1",
        method_family="exact_arithmetic_dag",
        answer=answer,
        failure_reason=None,
        evidence_payload={"step_count": len(draft.latent_program.steps)},
    )


def verify_bookkeeping_ledger(draft: CandidateDraft) -> GeneratorVerification:
    """Replay a separate signed ledger and prove the result by inversion."""

    payload = draft.verifier_payload
    start_raw = payload.get("start")
    updates_raw = payload.get("signed_updates")
    group_size_raw = payload.get("group_size")
    grouping_raw = payload.get("grouping")
    if (
        isinstance(start_raw, bool)
        or not isinstance(start_raw, int)
        or not isinstance(updates_raw, list)
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in updates_raw)
        or isinstance(group_size_raw, bool)
        or not isinstance(group_size_raw, int)
        or not isinstance(grouping_raw, bool)
    ):
        return verification_result(
            verifier_id="bookkeeping-ledger-v1",
            method_family="state_ledger_conservation",
            answer=None,
            failure_reason="invalid_ledger_payload",
            evidence_payload={},
        )
    start = start_raw
    updates = [int(item) for item in updates_raw]
    ending_items = start + sum(updates)
    grouping = grouping_raw
    group_size = group_size_raw
    if ending_items <= 0 or (grouping and ending_items % group_size != 0):
        return verification_result(
            verifier_id="bookkeeping-ledger-v1",
            method_family="state_ledger_conservation",
            answer=None,
            failure_reason="ledger_constraint_failure",
            evidence_payload={"update_count": len(updates)},
        )
    answer = Fraction(ending_items, group_size if grouping else 1)
    reconstructed_start = int(answer * (group_size if grouping else 1)) - sum(updates)
    if reconstructed_start != start:
        return verification_result(
            verifier_id="bookkeeping-ledger-v1",
            method_family="state_ledger_conservation",
            answer=None,
            failure_reason="inverse_check_failure",
            evidence_payload={"update_count": len(updates)},
        )
    return verification_result(
        verifier_id="bookkeeping-ledger-v1",
        method_family="state_ledger_conservation",
        answer=answer,
        failure_reason=None,
        evidence_payload={"update_count": len(updates), "inverse_check": True},
    )


def validate_bookkeeping_constraints(draft: CandidateDraft) -> tuple[str, ...]:
    """Return content-free rejection reasons for ambiguous or invalid bookkeeping drafts."""

    reasons: list[str] = []
    if draft.ambiguity_flags:
        reasons.append("ambiguous_target")
    if draft.canonical_final_answer.fraction <= 0:
        reasons.append("nonpositive_final_state")
    if len(draft.latent_program.steps) < 2:
        reasons.append("insufficient_dependency_depth")
    return tuple(reasons)
