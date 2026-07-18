"""Independent multi-step state-transition generator and verifier pair."""

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
from foundry.synthesis.object_units import (
    Countability,
    LedgerOperationKind,
    LocationSpec,
    ObjectKind,
    QuantityUnit,
    TypedLedgerOperation,
    TypedQuantity,
    validate_ledger_plan,
)
from foundry.synthesis.quality import (
    NounFormEvidence,
    RenderQualityMetadata,
    UnitTransitionEvidence,
)
from foundry.synthesis.realization import compile_problem, select_plan
from foundry.synthesis.realization.domains import make_domain
from foundry.synthesis.realization.ir import (
    BookkeepingProblemIR,
    LedgerChangeKind,
    LedgerChangeSpec,
    QuantitySpec,
    TargetKind,
    TargetSpec,
    UnitSpec,
)
from foundry.synthesis.realization.morphology import GROUP, TRANSFER
from foundry.synthesis.schema import (
    DifficultyLevel,
    LatentProgramSpec,
    ProgramParameter,
    ProgramStep,
)
from foundry.synthesis.taxonomy import FailureCategory

GENERATOR_ID = "bookkeeping-state-transitions"
GENERATOR_VERSION = "3"


def _kind(family: str, singular: str, plural: str) -> ObjectKind:
    return ObjectKind(
        family=family,
        singular=singular,
        plural=plural,
        countability=Countability.DISCRETE,
        unit=QuantityUnit.ITEM,
        combination_key=f"{family}:count",
        transferable=True,
        supported_verbs=("move", "place", "deliver", "remove", "send", "return"),
        supported_containers=("inventory",),
    )


@dataclass(frozen=True)
class _Scenario:
    scenario_id: str
    setting: str
    actor: str
    object_kind: ObjectKind
    ledger: str
    source: str
    destination: str
    safe_context: str


_SCENARIOS = (
    _Scenario(
        "botanical",
        "botanical station",
        "Mara",
        _kind("seed_packet", "seed packet", "seed packets"),
        "sorting cabinet",
        "dry-storage shelf",
        "planting cart",
        "The humidity log is kept separately.",
    ),
    _Scenario(
        "lunar_shop",
        "lunar repair shop",
        "Ilan",
        _kind("bearing", "bearing", "bearings"),
        "parts bin",
        "inspection bench",
        "assembly trolley",
        "The calibration tools stay on another bench.",
    ),
    _Scenario(
        "marine_lab",
        "marine field lab",
        "Nia",
        _kind("sample_vial", "sample vial", "sample vials"),
        "cold locker",
        "intake cooler",
        "analysis rack",
        "The temperature record does not alter the vial count.",
    ),
    _Scenario(
        "map_archive",
        "map archive",
        "Oren",
        _kind("map_tube", "map tube", "map tubes"),
        "catalog rack",
        "return desk",
        "scanning room",
        "The reading tables remain outside the storage count.",
    ),
    _Scenario(
        "bakery",
        "bakery store room",
        "Priya",
        _kind("flour_sack", "flour sack", "flour sacks"),
        "supply bay",
        "delivery pallet",
        "mixing station",
        "A separate cleaning check occurs after the count.",
    ),
    _Scenario(
        "theater",
        "theater wardrobe",
        "Ravi",
        _kind("costume", "costume", "costumes"),
        "costume rail",
        "laundry return",
        "rehearsal room",
        "Props are cataloged in a different collection.",
    ),
    _Scenario(
        "observatory",
        "hilltop observatory",
        "Sora",
        _kind("lens_cap", "lens cap", "lens caps"),
        "equipment drawer",
        "maintenance tray",
        "telescope deck",
        "The weather instruments use a separate register.",
    ),
    _Scenario(
        "apiary",
        "apiary workshop",
        "Tomas",
        _kind("hive_frame", "hive frame", "hive frames"),
        "drying rack",
        "repair table",
        "field wagon",
        "Protective clothing is stored elsewhere.",
    ),
    _Scenario(
        "rescue",
        "rescue depot",
        "Uma",
        _kind("blanket", "blanket", "blankets"),
        "supply cage",
        "wash station",
        "dispatch van",
        "Medical kits follow an independent inventory.",
    ),
    _Scenario(
        "ceramics",
        "ceramics studio",
        "Vik",
        _kind("clay_block", "clay block", "clay blocks"),
        "material shelf",
        "delivery bench",
        "throwing room",
        "Glaze containers are counted on another sheet.",
    ),
    _Scenario(
        "robotics",
        "robotics laboratory",
        "Wren",
        _kind("sensor_module", "sensor module", "sensor modules"),
        "component cabinet",
        "testing table",
        "prototype cart",
        "Power cables are excluded from this ledger.",
    ),
    _Scenario(
        "vineyard",
        "vineyard cellar",
        "Xena",
        _kind("bottle_crate", "bottle crate", "bottle crates"),
        "cellar bay",
        "loading dock",
        "tasting room",
        "Empty barrels use a separate storage list.",
    ),
    _Scenario(
        "museum",
        "museum restoration room",
        "Yara",
        _kind("display_hook", "display hook", "display hooks"),
        "hardware cabinet",
        "receiving tray",
        "gallery cart",
        "Lighting equipment is not part of this count.",
    ),
    _Scenario(
        "radio",
        "community radio station",
        "Zane",
        _kind("cable_reel", "cable reel", "cable reels"),
        "storage wall",
        "recording booth",
        "outside-broadcast case",
        "Microphones have their own checkout record.",
    ),
    _Scenario(
        "clinic",
        "field clinic",
        "Asha",
        _kind("bandage_roll", "bandage roll", "bandage rolls"),
        "medical cabinet",
        "supply tent",
        "treatment cart",
        "Medication is tracked by another system.",
    ),
    _Scenario(
        "harbor",
        "harbor workshop",
        "Bram",
        _kind("mooring_rope", "mooring rope", "mooring ropes"),
        "gear locker",
        "inspection dock",
        "service boat",
        "Safety flags are stored in a separate locker.",
    ),
    _Scenario(
        "print_shop",
        "print shop",
        "Cleo",
        _kind("paper_ream", "paper ream", "paper reams"),
        "stock room",
        "delivery platform",
        "press floor",
        "Ink supplies are excluded from this paper ledger.",
    ),
    _Scenario(
        "geology",
        "geology laboratory",
        "Dara",
        _kind("specimen_tray", "specimen tray", "specimen trays"),
        "collection rack",
        "intake counter",
        "study room",
        "Field notebooks remain in the archive.",
    ),
    _Scenario(
        "aviation",
        "aviation workshop",
        "Enzo",
        _kind("filter_cartridge", "filter cartridge", "filter cartridges"),
        "service cabinet",
        "inspection cart",
        "engine bay",
        "Fasteners are counted in another inventory.",
    ),
    _Scenario(
        "aquarium",
        "public aquarium",
        "Faye",
        _kind("feed_bucket", "feed bucket", "feed buckets"),
        "food locker",
        "preparation room",
        "habitat cart",
        "Water-testing supplies are tracked separately.",
    ),
    _Scenario(
        "library",
        "neighborhood library",
        "Galen",
        _kind("book_crate", "book crate", "book crates"),
        "receiving alcove",
        "return desk",
        "branch van",
        "Loose books are cataloged outside this crate ledger.",
    ),
    _Scenario(
        "solar_farm",
        "solar farm workshop",
        "Hana",
        _kind("junction_box", "junction box", "junction boxes"),
        "parts cage",
        "test bench",
        "maintenance truck",
        "Cable bundles use a separate stock record.",
    ),
    _Scenario(
        "textile",
        "textile cooperative",
        "Idris",
        _kind("thread_spool", "thread spool", "thread spools"),
        "dye-room shelf",
        "drying table",
        "loom cart",
        "Fabric rolls are recorded elsewhere.",
    ),
    _Scenario(
        "weather",
        "weather station",
        "Juno",
        _kind("battery_pack", "battery pack", "battery packs"),
        "power cabinet",
        "charging rack",
        "sensor hut",
        "Data cards do not share this inventory.",
    ),
)

_RENDERER_FAMILIES = (
    "active_chronology",
    "ledger_report",
    "location_first",
    "passive_inventory",
    "audit_entries",
    "shift_narrative",
    "before_after",
    "custodian_report",
)
TEMPLATE_FAMILIES = ("inventory", "grouping")
RENDERING_VARIANTS_PER_FAMILY = len(_RENDERER_FAMILIES)
SCENARIO_DOMAIN_COUNT = len(_SCENARIOS)


def _update_count(difficulty: DifficultyLevel) -> int:
    return {
        DifficultyLevel.EASY: 2,
        DifficultyLevel.MEDIUM: 3,
        DifficultyLevel.HARD: 4,
    }[difficulty]


def _operation_sentence(
    *,
    family: str,
    add: bool,
    actor: str,
    amount: int,
    noun: str,
    ledger: str,
    source: str,
    destination: str,
    ordinal: int,
) -> str:
    if add:
        forms = {
            "active_chronology": f"{actor} moved {amount} {noun} from the {source} into the {ledger}.",
            "ledger_report": f"The {source} supplied {amount} {noun}, which {actor} placed in the {ledger}.",
            "location_first": f"Into the {ledger}, {actor} delivered {amount} {noun} from the {source}.",
            "passive_inventory": f"From the {source}, {amount} {noun} were delivered to the {ledger}.",
            "audit_entries": f"Entry {ordinal} records an increase of {amount} {noun} from the {source}.",
            "shift_narrative": f"During update {ordinal}, {actor} returned {amount} {noun} from the {source} to the {ledger}.",
            "before_after": f"The next change added {amount} {noun} from the {source} to the {ledger}.",
            "custodian_report": f"According to {actor}'s report, the {ledger} received {amount} {noun} from the {source}.",
        }
    else:
        forms = {
            "active_chronology": f"{actor} sent {amount} {noun} from the {ledger} to the {destination}.",
            "ledger_report": f"The {ledger} released {amount} {noun} for the {destination} under {actor}'s record.",
            "location_first": f"Out of the {ledger}, {actor} moved {amount} {noun} to the {destination}.",
            "passive_inventory": f"From the {ledger}, {amount} {noun} were transferred to the {destination}.",
            "audit_entries": f"Entry {ordinal} records a decrease of {amount} {noun} sent to the {destination}.",
            "shift_narrative": f"During update {ordinal}, {actor} removed {amount} {noun} for the {destination}.",
            "before_after": f"The next change removed {amount} {noun} from the {ledger} for the {destination}.",
            "custodian_report": f"According to {actor}'s report, {amount} {noun} left the {ledger} for the {destination}.",
        }
    return f"Update {ordinal}: {forms[family]}"


def _opening_sentence(family: str, scenario: _Scenario, start: int, noun: str) -> str:
    forms = {
        "active_chronology": f"At the {scenario.setting}, {scenario.actor} counted {start} {noun} in the {scenario.ledger}.",
        "ledger_report": f"The opening ledger for the {scenario.ledger} at the {scenario.setting} listed {start} {noun}.",
        "location_first": f"Inside the {scenario.ledger} were {start} {noun} when work began at the {scenario.setting}.",
        "passive_inventory": f"At the start of the shift, {start} {noun} were stored in the {scenario.ledger}.",
        "audit_entries": f"An inventory audit at the {scenario.setting} starts with {start} {noun} assigned to the {scenario.ledger}.",
        "shift_narrative": f"When {scenario.actor}'s shift began, the {scenario.ledger} held {start} {noun}.",
        "before_after": f"Before any movement occurred at the {scenario.setting}, the {scenario.ledger} contained {start} {noun}.",
        "custodian_report": f"{scenario.actor}, the custodian of the {scenario.ledger}, reported an initial balance of {start} {noun}.",
    }
    return forms[family]


def _question_sentence(
    *, scenario: _Scenario, family: str, grouping: bool, group_size: int, noun: str
) -> str:
    inventory_forms = (
        f"How many {noun} remain in the {scenario.ledger}?",
        f"What final count of {noun} should the {scenario.ledger} report?",
        f"After every recorded movement, how many {noun} does the {scenario.ledger} hold?",
        f"Determine the closing number of {noun} in the {scenario.ledger}.",
        f"What is the {scenario.ledger}'s final {noun} balance?",
        f"At the end of the shift, how many {noun} are stored in the {scenario.ledger}?",
        f"What number of {noun} should {scenario.actor} record for the {scenario.ledger}?",
        f"Once all updates are applied, how many {noun} are left in the {scenario.ledger}?",
    )
    grouping_forms = (
        f"How many complete groups of {group_size} {noun} can {scenario.actor} make from the final balance?",
        f"The remaining {noun} are grouped {group_size} at a time; how many full groups result?",
        f"After the ledger updates, what number of complete {group_size}-{noun} groups can be formed?",
        f"How many full sets, each containing {group_size} {noun}, come from the closing inventory?",
        f"Using every final {noun} in equal sets of {group_size}, how many sets are available?",
        f"What is the number of complete groups when the closing {noun} count is divided into {group_size}s?",
        f"How many complete batches of {group_size} {noun} does the final ledger support?",
        f"From the ending balance, how many whole groups containing {group_size} {noun} can {scenario.actor} prepare?",
    )
    return (grouping_forms if grouping else inventory_forms)[_RENDERER_FAMILIES.index(family)]


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
    grouping = variant % 5 == 4
    scenario = _SCENARIOS[variant % len(_SCENARIOS)]
    update_count = _update_count(difficulty)
    group_size = rng.randint(2, 6) if grouping else 1
    start = rng.randint(70, 150) * group_size
    item_kind = scenario.object_kind
    parameters = [ProgramParameter("start", exact_value(start), item_kind.plural)]
    steps: list[ProgramStep] = []
    trace: list[str] = [f"The typed ledger starts with {start} {item_kind.plural}."]
    operations: list[str] = []
    current = Fraction(start)
    current_symbol = "start"
    signed_updates: list[int] = []
    quantities: list[TypedQuantity] = [TypedQuantity("start", start, item_kind, "ledger")]
    typed_operations: list[TypedLedgerOperation] = []
    noun_forms: list[NounFormEvidence] = [
        NounFormEvidence(start, item_kind.render_noun(start), item_kind)
    ]
    for index in range(update_count):
        magnitude = rng.randint(3, 18) * group_size
        add = ((variant >> index) & 1) == 0
        signed = magnitude if add else -magnitude
        if current + signed <= 0:
            signed = magnitude
            add = True
        parameter = f"change_{index + 1}"
        output = f"state_{index + 1}"
        parameters.append(ProgramParameter(parameter, exact_value(magnitude), item_kind.plural))
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
        noun = item_kind.render_noun(magnitude)
        trace.append(
            f"Update {index + 1} changes the {item_kind.singular} balance by {signed:+d}, "
            f"giving {int(current)}."
        )
        operations.append(operation)
        signed_updates.append(signed)
        quantity_location = "source" if add else "ledger"
        quantities.append(TypedQuantity(parameter, magnitude, item_kind, quantity_location))
        typed_operations.append(
            TypedLedgerOperation(
                LedgerOperationKind.TRANSFER_IN if add else LedgerOperationKind.TRANSFER_OUT,
                parameter,
                "move",
                "source" if add else "ledger",
                "ledger" if add else "destination",
            )
        )
        noun_forms.append(NounFormEvidence(magnitude, noun, item_kind))
        current_symbol = output

    target_kind = "items"
    if grouping:
        parameters.append(
            ProgramParameter("group_size", exact_value(group_size), f"{item_kind.plural}/group")
        )
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
            f"Partition the closing balance of {int(current)} into groups of {group_size}; "
            f"this gives {int(groups)} complete groups."
        )
        current = groups
        current_symbol = "group_count"
        target_kind = "groups"
    locations = (
        LocationSpec("ledger", scenario.ledger, "inventory", (item_kind.family,)),
        LocationSpec("source", scenario.source, "inventory", (item_kind.family,)),
        LocationSpec("destination", scenario.destination, "inventory", (item_kind.family,)),
    )
    plan_rejections = validate_ledger_plan(
        ledger_kind=item_kind,
        quantities=tuple(quantities),
        operations=tuple(typed_operations),
        locations=locations,
    )
    if plan_rejections:
        raise ValueError(f"typed bookkeeping plan is invalid: {plan_rejections[0]}")
    answer = exact_value(current)
    program = LatentProgramSpec(
        program_family=f"{GENERATOR_ID}:{'grouping' if grouping else 'inventory'}",
        parameters=tuple(parameters),
        steps=tuple(steps),
        constraints=("all ledger states remain positive", "all grouping is exact"),
        answer_symbol=current_symbol,
    )
    domain = make_domain(
        domain_id=scenario.scenario_id,
        setting=scenario.setting,
        actor=scenario.actor,
        item_id=scenario.object_kind.family,
        item_singular=scenario.object_kind.singular,
        item_plural=scenario.object_kind.plural,
        primary_location=scenario.ledger,
        secondary_location=scenario.source,
        destination_location=scenario.destination,
        container_singular="group",
        container_plural="groups",
        safe_context=scenario.safe_context,
    )
    item_unit = UnitSpec(f"{scenario.object_kind.family}:count", domain.item.lexeme)
    initial_ir = QuantitySpec("initial", start, domain.item.entity_id, item_unit)
    change_nodes: list[LedgerChangeSpec] = []
    for index, signed in enumerate(signed_updates, start=1):
        amount = abs(signed)
        quantity_ir = QuantitySpec(
            f"change_quantity_{index}", amount, domain.item.entity_id, item_unit
        )
        incoming = signed > 0
        change_nodes.append(
            LedgerChangeSpec(
                f"change_{index}",
                LedgerChangeKind.TRANSFER_IN if incoming else LedgerChangeKind.TRANSFER_OUT,
                quantity_ir,
                domain.secondary_location.entity_id
                if incoming
                else domain.primary_location.entity_id,
                domain.primary_location.entity_id
                if incoming
                else domain.destination_location.entity_id,
                TRANSFER,
            )
        )
    target_kind_ir = TargetKind.GROUP_COUNT if grouping else TargetKind.REMAINING_QUANTITY
    target_unit = UnitSpec("group:count", GROUP) if grouping else item_unit
    problem_ir = BookkeepingProblemIR(
        problem_id=candidate_id(GENERATOR_ID, seed, variant),
        domain=domain,
        initial=initial_ir,
        changes=tuple(change_nodes),
        target=TargetSpec(
            "target", target_kind_ir, current_symbol, domain.item.entity_id, target_unit
        ),
        group_size=group_size if grouping else None,
        context_node_id="safe_context" if variant % 3 == 2 else None,
    )
    realization = compile_problem(
        problem_ir, select_plan(seed=seed, variant=variant, family="bookkeeping")
    )
    question = realization.text
    rendered_clauses = tuple(
        clause for clause in realization.clauses if clause != realization.question_clause
    )
    conclusion = realization.question_clause
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
        rendered_question=question,
        deterministic_solution_trace=trace_tuple,
        canonical_final_answer=answer,
        training_completion=training_completion(
            trace_tuple, answer, output_contract_enabled=output_contract_enabled
        ),
        quality_metadata=RenderQualityMetadata(
            scenario_id=scenario.scenario_id,
            renderer_family=realization.signature.sha256,
            clauses=rendered_clauses,
            declared_entity_ids=("actor", "ledger", "source", "destination"),
            referenced_entity_ids=("actor", "ledger", "source", "destination"),
            pronoun_referent_ids=(),
            noun_forms=tuple(noun_forms),
            combination_groups=((item_kind, *(item_kind for _ in signed_updates)),),
            unit_transitions=(
                UnitTransitionEvidence(
                    QuantityUnit.ITEM,
                    QuantityUnit.PACKAGE if grouping else QuantityUnit.ITEM,
                    grouping,
                ),
            ),
            target_symbol=current_symbol,
            target_mentions=1,
            conclusion=conclusion,
            constraints_tied=False,
            grammar_complete=True,
            quantities=tuple(quantities),
            operations=tuple(typed_operations),
        ),
        problem_ir=problem_ir,
        realization=realization,
        structure_signature={
            "generator": GENERATOR_ID,
            "version": GENERATOR_VERSION,
            "mode": "grouping" if grouping else "inventory",
            "difficulty": difficulty,
            "scenario_family": scenario.scenario_id,
            "renderer_family": realization.signature.sha256,
            "operations": operations + (["divide"] if grouping else []),
            "topology": "linear_state_chain",
            "target_kind": target_kind,
            "object_family": item_kind.family,
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
