"""Bounded integer allocation generation with constructive and enumerative checks."""

# ruff: noqa: E501  # controlled natural-language templates are kept readable as full clauses

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

from foundry.synthesis.generators import (
    CandidateDraft,
    GeneratorVerification,
    candidate_id,
    exact_value,
    training_completion,
    verification_result,
)
from foundry.synthesis.object_units import Countability, ObjectKind, QuantityUnit
from foundry.synthesis.quality import NounFormEvidence, RenderQualityMetadata
from foundry.synthesis.schema import (
    DifficultyLevel,
    LatentProgramSpec,
    ProgramParameter,
    ProgramStep,
)
from foundry.synthesis.taxonomy import FailureCategory

GENERATOR_ID = "bounded-discrete-allocation"
GENERATOR_VERSION = "2"


def _kind(family: str, singular: str, plural: str) -> ObjectKind:
    return ObjectKind(
        family=family,
        singular=singular,
        plural=plural,
        countability=Countability.DISCRETE,
        unit=QuantityUnit.ITEM,
        combination_key=f"{family}:count",
        transferable=False,
        supported_verbs=("assemble", "pack", "distribute", "allocate"),
        supported_containers=("package", "container", "group"),
    )


@dataclass(frozen=True)
class _DiscreteScenario:
    scenario_id: str
    setting: str
    actor: str
    object_kind: ObjectKind
    container: str
    safe_context: str


_SCENARIOS = (
    _DiscreteScenario(
        "registry",
        "comet-sample registry",
        "Ada",
        _kind("sample_case", "sample case", "sample cases"),
        "vault",
        "The catalog code is recorded separately.",
    ),
    _DiscreteScenario(
        "mosaic",
        "underwater mosaic shop",
        "Ben",
        _kind("mosaic_frame", "mosaic frame", "mosaic frames"),
        "rack",
        "Paint colors do not affect capacity.",
    ),
    _DiscreteScenario(
        "drone",
        "polar drone hangar",
        "Cleo",
        _kind("drone_kit", "drone kit", "drone kits"),
        "bay",
        "Flight schedules use another register.",
    ),
    _DiscreteScenario(
        "seed",
        "high-altitude seed exchange",
        "Dev",
        _kind("seed_canister", "seed canister", "seed canisters"),
        "crate",
        "Humidity readings are independent.",
    ),
    _DiscreteScenario(
        "instrument",
        "tidal instrument depot",
        "Emi",
        _kind("instrument_rack", "instrument rack", "instrument racks"),
        "room",
        "Inspection dates are tracked elsewhere.",
    ),
    _DiscreteScenario(
        "stone",
        "basalt sculpture yard",
        "Finn",
        _kind("stone_module", "stone module", "stone modules"),
        "pallet",
        "Surface finish does not change the count.",
    ),
    _DiscreteScenario(
        "repair",
        "orbital repair cooperative",
        "Gia",
        _kind("repair_bundle", "repair bundle", "repair bundles"),
        "locker",
        "Tool serials are logged independently.",
    ),
    _DiscreteScenario(
        "light",
        "desert light laboratory",
        "Hale",
        _kind("light_panel", "light panel", "light panels"),
        "cabinet",
        "Brightness tests happen after allocation.",
    ),
    _DiscreteScenario(
        "marine",
        "marine sensor workshop",
        "Ira",
        _kind("sensor_pack", "sensor pack", "sensor packs"),
        "shelf",
        "Calibration notes use a separate form.",
    ),
    _DiscreteScenario(
        "archive",
        "city archive annex",
        "Jin",
        _kind("record_box", "record box", "record boxes"),
        "alcove",
        "Document dates do not affect grouping.",
    ),
    _DiscreteScenario(
        "theater",
        "theater prop room",
        "Koa",
        _kind("prop_case", "prop case", "prop cases"),
        "closet",
        "Rehearsal times are irrelevant.",
    ),
    _DiscreteScenario(
        "orchard",
        "orchard tool shed",
        "Luz",
        _kind("tool_set", "tool set", "tool sets"),
        "stall",
        "Maintenance notes are stored elsewhere.",
    ),
    _DiscreteScenario(
        "ceramic",
        "ceramic testing studio",
        "Mina",
        _kind("tile_sample", "tile sample", "tile samples"),
        "drawer",
        "Glaze colors are not part of the allocation.",
    ),
    _DiscreteScenario(
        "signal",
        "rail signal lab",
        "Nico",
        _kind("signal_unit", "signal unit", "signal units"),
        "bench",
        "Wire lengths are checked separately.",
    ),
    _DiscreteScenario(
        "reef",
        "reef restoration station",
        "Opal",
        _kind("reef_block", "reef block", "reef blocks"),
        "pen",
        "Water temperature is monitored separately.",
    ),
    _DiscreteScenario(
        "audio",
        "audio equipment library",
        "Paz",
        _kind("audio_module", "audio module", "audio modules"),
        "case",
        "Cable checkout uses another counter.",
    ),
    _DiscreteScenario(
        "weather",
        "weather probe depot",
        "Rui",
        _kind("probe_kit", "probe kit", "probe kits"),
        "cage",
        "Forecast records are unrelated.",
    ),
    _DiscreteScenario(
        "festival",
        "festival supply room",
        "Sana",
        _kind("banner_roll", "banner roll", "banner rolls"),
        "bin",
        "Event dates do not alter the totals.",
    ),
    _DiscreteScenario(
        "robot",
        "robotics classroom",
        "Tao",
        _kind("robot_part_set", "robot part set", "robot part sets"),
        "station",
        "Lesson plans are filed separately.",
    ),
    _DiscreteScenario(
        "garden",
        "botanical research garden",
        "Uma",
        _kind("plant_tray", "plant tray", "plant trays"),
        "greenhouse",
        "Watering records are independent.",
    ),
)

_RENDERER_FAMILIES = (
    "setting_first",
    "actor_first",
    "question_first",
    "inventory_report",
    "passive_plan",
    "constraint_summary",
)

_SEARCH_SPACE_RANGES = {
    DifficultyLevel.EASY: (9, 35),
    DifficultyLevel.MEDIUM: (36, 80),
    DifficultyLevel.HARD: (81, 200),
}
TEMPLATE_FAMILIES = (
    "two_type_allocation",
    "complete_packages",
    "equal_distribution",
    "dual_capacity",
)
RENDERING_VARIANTS_PER_FAMILY = len(_RENDERER_FAMILIES)
SCENARIO_DOMAIN_COUNT = len(_SCENARIOS)


def _mode(variant: int) -> str:
    return ("two_type_allocation", "complete_packages", "equal_distribution", "dual_capacity")[
        variant % 4
    ]


def _difficulty_search_space(rng: random.Random, difficulty: DifficultyLevel) -> int:
    low, high = _SEARCH_SPACE_RANGES[difficulty]
    return rng.randint(low, high)


def _render_discrete_question(
    *,
    mode: str,
    family: str,
    scenario: _DiscreteScenario,
    values: dict[str, object],
) -> tuple[str, tuple[str, ...], str]:
    plural = scenario.object_kind.plural
    singular = scenario.object_kind.singular
    setting = scenario.setting
    actor = scenario.actor
    container = scenario.container
    if mode == "two_type_allocation":
        total = _payload_int(values, "total")
        resource = _payload_int(values, "resource_total")
        first = _payload_int(values, "first_cost")
        second = _payload_int(values, "second_cost")
        forms = (
            (
                f"The {setting} assembles {total} {plural} in designs A and B.",
                f"Each A design uses {first} parts, each B design uses {second}, and the batch uses {resource} parts.",
                "How many design A units were assembled?",
            ),
            (
                f"{actor} records a batch of {total} {plural} using {resource} parts.",
                f"An A design requires {first} parts while a B design requires {second}.",
                "Determine the number with design A.",
            ),
            (
                "The requested value is the design A portion of the completed batch.",
                f"The {setting} made {total} {plural}, with A using {first} parts and B using {second} parts.",
                f"Together they consumed {resource} parts; how many were design A?",
            ),
            (
                f"An inventory report from the {setting} lists {total} {plural} and {resource} consumed parts.",
                f"The two possible designs require {first} and {second} parts respectively.",
                "Find the design A count.",
            ),
            (
                f"At the {setting}, {total} {plural} were assembled from {resource} parts.",
                f"Design A was allotted {first} parts per unit and design B was allotted {second}.",
                "Calculate how many were design A.",
            ),
            (
                f"The {setting} has two assembly rules: A uses {first} parts and B uses {second}.",
                f"Its completed batch contains {total} {plural} and accounts for {resource} parts.",
                "What is the unique design A quantity?",
            ),
        )
    elif mode == "complete_packages":
        total = _payload_int(values, "total")
        size = _payload_int(values, "package_size")
        forms = (
            (
                f"The {setting} has {total} {plural} ready for packing.",
                f"A complete package holds exactly {size} {plural}; partial packages do not count.",
                "What is the maximum number of complete packages?",
            ),
            (
                f"{actor} packs {total} {plural} in groups of {size}.",
                "Only full groups are recorded.",
                "How many complete packages can be recorded?",
            ),
            (
                "The requested value is the number of full packages.",
                f"There are {total} {plural} at the {setting}, and each full package requires {size}.",
                "Any remainder stays unpacked; how many full packages can be made?",
            ),
            (
                f"The {setting} inventory report shows {total} {plural}.",
                f"Package capacity is fixed at {size} {plural}.",
                "Find the greatest number of filled packages.",
            ),
            (
                f"At the {setting}, {total} {plural} are to be placed into packages of {size}.",
                "Incomplete packages are excluded from the count.",
                "Determine the full-package total.",
            ),
            (
                f"The packing rule at the {setting} assigns exactly {size} {plural} to every complete package.",
                f"A stock of {total} {plural} is available.",
                "What maximum count satisfies the rule?",
            ),
        )
    elif mode == "equal_distribution":
        total = _payload_int(values, "total")
        containers = _payload_int(values, "containers")
        forms = (
            (
                f"The {setting} distributes {total} {plural} equally among {containers} {container}s.",
                "Nothing is left over.",
                f"How many {plural} go in each {container}?",
            ),
            (
                f"{actor} divides {total} {plural} across {containers} identical {container}s.",
                "Every destination receives the same quantity.",
                f"What is the count per {container}?",
            ),
            (
                f"The requested value is the number of {plural} in one {container}.",
                f"At the {setting}, {total} are shared evenly by {containers} {container}s with no remainder.",
                "What is the common allocation?",
            ),
            (
                f"An allocation report lists {total} {plural} and {containers} equal destinations.",
                f"Each destination is one {container} at the {setting}.",
                "Find the equal share.",
            ),
            (
                f"At the {setting}, {total} {plural} are assigned to {containers} {container}s in equal quantities.",
                "No item remains outside the allocation.",
                "Determine one destination's share.",
            ),
            (
                f"The distribution rule requires all {containers} {container}s to receive matching counts.",
                f"The available inventory is {total} {plural}.",
                "What unique count satisfies the rule?",
            ),
        )
    else:
        first_resource = _payload_int(values, "first_resource")
        second_resource = _payload_int(values, "second_resource")
        first_per = _payload_int(values, "first_per")
        second_per = _payload_int(values, "second_per")
        forms = (
            (
                f"The {setting} can build {plural} from amber and cobalt parts.",
                f"Each {singular} needs {first_per} amber parts and {second_per} cobalt parts; stocks contain {first_resource} and {second_resource} respectively.",
                f"What maximum number of complete {plural} can be built?",
            ),
            (
                f"{actor} has {first_resource} amber parts and {second_resource} cobalt parts.",
                f"One {singular} consumes {first_per} amber and {second_per} cobalt parts.",
                "Determine the maximum complete-unit count.",
            ),
            (
                f"The requested value is the complete {plural} capacity.",
                f"Available stocks are {first_resource} amber parts and {second_resource} cobalt parts.",
                f"Every {singular} requires {first_per} amber and {second_per} cobalt parts; what maximum can the {setting} assemble?",
            ),
            (
                f"The {setting} inventory lists {first_resource} amber parts and {second_resource} cobalt parts.",
                f"The construction rule assigns {first_per} amber and {second_per} cobalt parts to each {singular}.",
                "Find the limiting complete-build capacity.",
            ),
            (
                f"At the {setting}, each {singular} is assembled with {first_per} amber parts and {second_per} cobalt parts.",
                f"The available supplies are {first_resource} amber and {second_resource} cobalt parts.",
                "Calculate the maximum number of complete units.",
            ),
            (
                f"Two independent capacity constraints govern production at the {setting}.",
                f"For each {singular}, {first_per} of {first_resource} amber parts and {second_per} of {second_resource} cobalt parts are needed.",
                "What unique maximum satisfies both constraints?",
            ),
        )
    selected = forms[_RENDERER_FAMILIES.index(family)]
    return " ".join(selected), selected[:-1], selected[-1]


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
    occurrence = variant // 4
    scenario = _SCENARIOS[(variant * 7 + occurrence) % len(_SCENARIOS)]
    renderer_family = _RENDERER_FAMILIES[occurrence % len(_RENDERER_FAMILIES)]
    search_space_target = _difficulty_search_space(rng, difficulty)
    parameters: list[ProgramParameter] = []
    steps: list[ProgramStep] = []
    trace: list[str] = []
    payload: dict[str, object] = {"mode": mode, "finite_domain": True}

    if mode == "two_type_allocation":
        total = search_space_target - 1
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
        total = search_space_target - 1
        complete, remainder = divmod(total, package_size)
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
        trace.append(
            f"Use exact integer division: {total} divided by {package_size} gives {complete}."
        )
        payload.update(total=total, package_size=package_size)
        answer_symbol = "complete_packages"
    elif mode == "equal_distribution":
        containers = rng.randint(3, 9)
        each = max(2, (search_space_target - 1) // containers)
        total = containers * each
        while total + 1 < _SEARCH_SPACE_RANGES[difficulty][0]:
            each += 1
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
        trace.append(f"Exact equal distribution gives {total}/{containers} = {each} per container.")
        payload.update(total=total, containers=containers)
        answer_symbol = "per_container"
    else:
        first_per = rng.randint(2, 6)
        second_per = rng.randint(2, 7)
        low, high = _SEARCH_SPACE_RANGES[difficulty]
        target = 2
        while True:
            first_resource = target * first_per + (first_per - 1)
            second_resource = (target + 1) * second_per + (second_per - 1)
            capacity_domain = max(first_resource, second_resource) + 1
            if low <= capacity_domain <= high:
                break
            if capacity_domain > high:
                raise ValueError("unable to construct the requested dual-capacity difficulty")
            target += 1
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

    question, rendered_clauses, conclusion = _render_discrete_question(
        mode=mode,
        family=renderer_family,
        scenario=scenario,
        values=payload,
    )
    if mode in {"two_type_allocation", "complete_packages", "equal_distribution"}:
        actual_search_space = _payload_int(payload, "total") + 1
    else:
        actual_search_space = (
            max(
                _payload_int(payload, "first_resource"),
                _payload_int(payload, "second_resource"),
            )
            + 1
        )
    low, high = _SEARCH_SPACE_RANGES[difficulty]
    if not low <= actual_search_space <= high:
        raise ValueError("discrete search space is outside the documented difficulty range")
    constraints_independent = True
    if mode == "dual_capacity":
        first_capacity = _payload_int(payload, "first_resource") // _payload_int(
            payload, "first_per"
        )
        second_capacity = _payload_int(payload, "second_resource") // _payload_int(
            payload, "second_per"
        )
        constraints_independent = first_capacity != second_capacity
        if not constraints_independent:
            raise ValueError("dual-capacity constraints must not be tied")
    payload["difficulty_evidence"] = {
        "variable_count": 2 if mode == "two_type_allocation" else 1,
        "domain_size": actual_search_space,
        "independent_constraints": 2 if mode in {"two_type_allocation", "dual_capacity"} else 1,
        "dependency_depth": len(steps),
        "elimination_steps": max(1, len(steps) - 1),
    }

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
        quality_metadata=RenderQualityMetadata(
            scenario_id=scenario.scenario_id,
            renderer_family=renderer_family,
            clauses=rendered_clauses,
            declared_entity_ids=("actor", "inventory", "target"),
            referenced_entity_ids=("inventory", "target"),
            pronoun_referent_ids=(),
            noun_forms=(
                NounFormEvidence(
                    quantity=2,
                    rendered_noun=scenario.object_kind.plural,
                    object_kind=scenario.object_kind,
                ),
            ),
            combination_groups=(),
            operations=(),
            unit_transitions=(),
            target_symbol=answer_symbol,
            target_mentions=1,
            conclusion=conclusion,
            constraints_tied=not constraints_independent,
            grammar_complete=True,
        ),
        structure_signature={
            "generator": GENERATOR_ID,
            "mode": mode,
            "difficulty": difficulty,
            "scenario_family": scenario.scenario_id,
            "renderer_family": renderer_family,
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
