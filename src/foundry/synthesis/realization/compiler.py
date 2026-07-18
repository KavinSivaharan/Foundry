"""Deterministic typed English realization compiler.

The compiler owns every complete sentence. Mathematical generators provide only
semantic IR and never select a question by concatenating prose fragments.
"""

# ruff: noqa: E501  # controlled sentence plans remain readable as complete clauses

from __future__ import annotations

import random
from collections.abc import Iterable

from foundry.synthesis.realization.ir import (
    BookkeepingProblemIR,
    ClausePlan,
    CompiledRealization,
    CoverageEntry,
    DiscreteProblemIR,
    DiscreteRelationKind,
    MorphologyUse,
    ProblemIR,
    RateProblemIR,
    RateRelationKind,
    RenderedUnitUse,
    RenderSignature,
    TargetKind,
    Voice,
)
from foundry.synthesis.realization.morphology import noun_form

COMPILER_VERSION = "foundry-realizer-v1"
INTRO_STYLE_COUNT = 6
EVENT_STYLE_COUNT = 8
QUESTION_STYLE_COUNT = 8
CONJUNCTION_STYLE_COUNT = 4
NUMERIC_STYLE_COUNT = 2


def select_plan(*, seed: int, variant: int, family: str) -> ClausePlan:
    """Select independent realization dimensions deterministically."""

    rng = random.Random(f"{seed}:{variant}:{family}:{COMPILER_VERSION}")
    return ClausePlan(
        intro_style=rng.randrange(INTRO_STYLE_COUNT),
        event_style=rng.randrange(EVENT_STYLE_COUNT),
        question_style=rng.randrange(QUESTION_STYLE_COUNT),
        context_position=rng.randrange(3),
        conjunction_style=rng.randrange(CONJUNCTION_STYLE_COUNT),
        numeric_style=rng.randrange(NUMERIC_STYLE_COUNT),
        voice=Voice.ACTIVE if rng.randrange(2) == 0 else Voice.PASSIVE,
    )


def _join(clauses: Iterable[str]) -> str:
    return " ".join(clause.strip() for clause in clauses if clause.strip())


def _question(text: str) -> str:
    """Normalize controlled request forms to an explicit question boundary."""

    return text.rstrip(".?") + "?"


def _surface_number(value: int, style: int) -> str:
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        12: "twelve",
    }
    return words.get(value, str(value)) if style == 1 else str(value)


def _contextualize(core: list[str], context: str, position: int) -> list[str]:
    if not context.strip():
        return core
    if position == 0:
        return [context, *core]
    if position == 1 and len(core) > 1:
        return [core[0], context, *core[1:]]
    return [*core[:-1], context, core[-1]]


def _bookkeeping(problem: BookkeepingProblemIR, plan: ClausePlan) -> CompiledRealization:
    domain = problem.domain
    actor = domain.actor.proper_name or "The custodian"
    item = domain.item.lexeme
    initial_noun, initial_use = noun_form(item, problem.initial.value)
    intro_forms = (
        f"At the {domain.setting}, {actor} begins with {problem.initial.value} {initial_noun} in the {domain.primary_location.lexeme.singular}.",
        f"The opening inventory at the {domain.primary_location.lexeme.singular} contains {problem.initial.value} {initial_noun}.",
        f"Before any transfers, {problem.initial.value} {initial_noun} are recorded in the {domain.primary_location.lexeme.singular}.",
        f"An inventory check at the {domain.setting} finds {problem.initial.value} {initial_noun} in the {domain.primary_location.lexeme.singular}.",
        f"When the work period starts, the {domain.primary_location.lexeme.singular} holds {problem.initial.value} {initial_noun}.",
        f"{actor}'s ledger opens with {problem.initial.value} {initial_noun} assigned to the {domain.primary_location.lexeme.singular}.",
    )
    clauses = [intro_forms[plan.intro_style]]
    coverage = [CoverageEntry(problem.initial.node_id, 0)]
    morphology: list[MorphologyUse] = [initial_use]
    for ordinal, change in enumerate(problem.changes, start=1):
        amount = change.quantity.value
        noun, use = noun_form(item, amount)
        morphology.append(use)
        incoming = change.kind.value == "transfer_in"
        source = domain.secondary_location.lexeme.singular
        destination = domain.destination_location.lexeme.singular
        ledger = domain.primary_location.lexeme.singular
        if incoming:
            forms = (
                f"Next, {actor} transfers {amount} {noun} from the {source} into the {ledger}.",
                f"The {ledger} then receives {amount} {noun} from the {source}.",
                f"From the {source}, {amount} {noun} are moved into the {ledger}.",
                f"Inventory update {ordinal} adds {amount} {noun} from the {source} to the {ledger}.",
                f"Afterward, a delivery of {amount} {noun} moves from the {source} to the {ledger}.",
                f"{actor} records an increase of {amount} {noun} supplied by the {source}.",
                f"The next ledger entry shows {amount} {noun} arriving from the {source}.",
                f"A transfer from the {source} places {amount} {noun} in the {ledger}.",
            )
        else:
            forms = (
                f"Next, {actor} transfers {amount} {noun} from the {ledger} to the {destination}.",
                f"The {ledger} then sends {amount} {noun} to the {destination}.",
                f"From the {ledger}, {amount} {noun} are moved to the {destination}.",
                f"Inventory update {ordinal} removes {amount} {noun} from the {ledger} for the {destination}.",
                f"Afterward, a shipment of {amount} {noun} leaves the {ledger} for the {destination}.",
                f"{actor} records a decrease of {amount} {noun} sent to the {destination}.",
                f"The next ledger entry shows {amount} {noun} departing for the {destination}.",
                f"A transfer to the {destination} removes {amount} {noun} from the {ledger}.",
            )
        clauses.append(forms[(plan.event_style + ordinal - 1) % len(forms)])
        coverage.append(CoverageEntry(change.node_id, len(clauses) - 1))
    plural, plural_use = noun_form(item, 2)
    morphology.append(plural_use)
    ledger = domain.primary_location.lexeme.singular
    if problem.target.kind is TargetKind.GROUP_COUNT:
        if problem.group_size is None:
            raise ValueError("group-count target lacks a group size")
        size = _surface_number(problem.group_size, plan.numeric_style)
        questions = (
            f"How many complete groups of {size} {plural} can be made from the final inventory?",
            f"After all transfers, how many full sets containing {size} {plural} are available?",
            f"What number of complete batches, each with {size} {plural}, can the closing inventory form?",
            f"How many whole groups result when the remaining {plural} are arranged {size} per group?",
            f"Determine the number of full groups of {size} {plural} supported by the ending balance.",
            f"How many complete sets of {size} {plural} can {actor} prepare after the updates?",
            f"From the closing inventory, what is the number of whole batches containing {size} {plural}?",
            f"Once every transfer is applied, how many complete {plural} groups of size {size} remain?",
        )
    else:
        questions = (
            f"How many {plural} remain in the {ledger}?",
            f"What final count of {plural} should the {ledger} report?",
            f"After every transfer, how many {plural} does the {ledger} hold?",
            f"Determine the closing quantity of {plural} in the {ledger}.",
            f"What is the final inventory count of {plural} at the {ledger}?",
            f"At the end of the updates, how many {plural} are stored in the {ledger}?",
            f"What number of {plural} should {actor} record for the {ledger}?",
            f"Once all movements are applied, how many {plural} are left in the {ledger}?",
        )
    question = _question(questions[plan.question_style])
    clauses.append(question)
    coverage.append(CoverageEntry(problem.target.node_id, len(clauses) - 1))
    if problem.context_node_id is not None:
        clauses = _contextualize(clauses, domain.safe_context, plan.context_position)
        # Context insertion shifts clause indices; rebuild by matching ordered semantic clauses.
        semantic_clause_count = len(problem.required_node_ids)
        semantic_indices = [i for i in range(len(clauses)) if clauses[i] != domain.safe_context]
        coverage = [
            CoverageEntry(node_id, semantic_indices[index])
            for index, node_id in enumerate(problem.required_node_ids[:semantic_clause_count])
        ]
    signature = RenderSignature(
        COMPILER_VERSION,
        "bookkeeping",
        "grouping" if problem.target.kind is TargetKind.GROUP_COUNT else "inventory",
        domain.domain_id,
        plan,
    )
    return CompiledRealization(
        _join(clauses),
        tuple(clauses),
        question,
        signature,
        tuple(coverage),
        tuple(morphology),
        (RenderedUnitUse(problem.target.unit.unit_id, plural, None),),
        problem.target.kind,
        (),
        True,
    )


def _scalar(problem: RateProblemIR | DiscreteProblemIR, name: str) -> int:
    for scalar in problem.scalars:
        if scalar.name == name:
            return scalar.value
    raise ValueError(f"missing scalar {name}")


def _rate(problem: RateProblemIR, plan: ClausePlan) -> CompiledRealization:
    d = problem.domain
    actor = d.actor.proper_name or "The operator"
    item_plural, item_use = noun_form(d.item.lexeme, 2)
    coverage: list[CoverageEntry] = []
    clauses: list[str] = []
    units: list[RenderedUnitUse] = []
    kind = problem.relation_kind
    if kind is RateRelationKind.RATE_TOTAL:
        rate, intervals = _scalar(problem, "rate"), _scalar(problem, "intervals")
        frames = (
            (
                f"At the {d.setting}, a process produces {rate} {item_plural} per interval.",
                f"The process runs for {intervals} intervals.",
            ),
            (
                f"{actor} records a production rate of {rate} {item_plural} for each interval.",
                f"Exactly {intervals} intervals are completed.",
            ),
            (
                f"A calibrated interval yields {rate} {item_plural} at the {d.setting}.",
                f"Production continues for {intervals} equal intervals.",
            ),
            (
                f"During each interval, {rate} {item_plural} leave the process.",
                f"The run contains {intervals} intervals in total.",
            ),
            (
                f"The per-interval output is {rate} {item_plural}.",
                f"{actor} operates the process through {intervals} intervals.",
            ),
            (
                f"One interval contributes {rate} {item_plural} to the output.",
                f"The {d.setting} completes {intervals} such intervals.",
            ),
        )
        intro_first, intro_second = frames[plan.intro_style]
        clauses.extend((intro_first, intro_second))
        coverage.extend(
            (
                CoverageEntry(problem.scalars[0].node_id, 0),
                CoverageEntry(problem.scalars[1].node_id, 1),
            )
        )
        units.append(RenderedUnitUse(problem.scalars[0].unit.unit_id, item_plural, "interval"))
        questions = (
            f"How many {item_plural} are produced altogether?",
            f"What is the total output of {item_plural}?",
            f"Determine the complete production count of {item_plural}.",
            f"How many {item_plural} does the full run produce?",
            f"Calculate the number of {item_plural} made across all intervals.",
            f"What combined quantity of {item_plural} leaves the process?",
            f"Find the final production total in {item_plural}.",
            f"What total number of {item_plural} should {actor} record?",
        )
    elif kind is RateRelationKind.RATIO_SCALE:
        ratio_first, ratio_second, known = (
            _scalar(problem, key) for key in ("first_part", "second_part", "known")
        )
        clauses.extend(
            (
                f"At the {d.setting}, two collections of {item_plural} have an exact ratio of {ratio_first}:{ratio_second}.",
                f"The first collection contains {known} {item_plural}.",
            )
        )
        coverage.extend(
            (
                CoverageEntry(problem.scalars[0].node_id, 0),
                CoverageEntry(problem.scalars[1].node_id, 0),
                CoverageEntry(problem.scalars[2].node_id, 1),
            )
        )
        questions = (
            f"How many {item_plural} are in the second collection?",
            f"What count of {item_plural} belongs to the second collection?",
            f"Determine the corresponding quantity of {item_plural} in the second collection.",
            f"What is the size of the second collection of {item_plural}?",
            f"Calculate the second collection's count of {item_plural}.",
            f"How large is the matching second collection of {item_plural}?",
            f"Find the number of {item_plural} assigned to the second collection.",
            f"What exact quantity of {item_plural} completes the ratio?",
        )
    elif kind is RateRelationKind.PERCENTAGE:
        base, percent = _scalar(problem, "base"), _scalar(problem, "percent")
        clauses.extend(
            (
                f"At the {d.setting}, a batch contains {base} {item_plural}.",
                f"A quality check selects exactly {percent}% of that batch.",
            )
        )
        coverage.extend(
            (
                CoverageEntry(problem.scalars[0].node_id, 0),
                CoverageEntry(problem.scalars[1].node_id, 1),
            )
        )
        questions = (
            f"How many {item_plural} are selected?",
            f"What is the selected count of {item_plural}?",
            f"Determine the exact number of {item_plural} in the sample.",
            f"How large is the selected subset of {item_plural}?",
            f"Calculate the sample size in {item_plural}.",
            f"What quantity of {item_plural} does the percentage represent?",
            f"Find the exact selected total of {item_plural}.",
            f"How many of the {item_plural} belong to the quality sample?",
        )
    elif kind is RateRelationKind.WEIGHTED_MEAN:
        if len({(group.weight, group.value) for group in problem.groups}) != len(problem.groups):
            raise ValueError("weighted groups must be semantically unique")
        for index, group in enumerate(problem.groups, start=1):
            clauses.append(
                f"Group {index} contains {group.weight} panels with {group.value} marks per panel."
            )
            coverage.append(CoverageEntry(group.node_id, len(clauses) - 1))
        questions = (
            "What is the exact weighted mean in marks per panel?",
            "Calculate the panel-weighted average number of marks.",
            "Determine the exact mean marks per panel across all groups.",
            "What weighted average describes the marks per panel?",
            "Find the weighted mean of the panel measurements.",
            "What is the average mark value after weighting by panel count?",
            "Compute the exact panel-weighted mean.",
            "Report the weighted average in marks per panel.",
        )
        units.append(RenderedUnitUse(problem.target.unit.unit_id, "marks", "panel"))
    else:
        rate_first, rate_second, intervals = (
            _scalar(problem, key) for key in ("first_rate", "second_rate", "intervals")
        )
        clauses.extend(
            (
                f"At the {d.setting}, two channels deliver {rate_first} and {rate_second} {item_plural} per interval, respectively.",
                f"Both channels operate for {intervals} intervals.",
            )
        )
        coverage.extend(
            (
                CoverageEntry(problem.scalars[0].node_id, 0),
                CoverageEntry(problem.scalars[1].node_id, 0),
                CoverageEntry(problem.scalars[2].node_id, 1),
            )
        )
        units.extend(
            (
                RenderedUnitUse(problem.scalars[0].unit.unit_id, item_plural, "interval"),
                RenderedUnitUse(problem.scalars[1].unit.unit_id, item_plural, "interval"),
            )
        )
        questions = (
            f"How many {item_plural} do the channels deliver altogether?",
            f"What is their combined total of {item_plural}?",
            f"Determine the aggregate number of {item_plural} delivered.",
            f"How many {item_plural} arrive across both channels?",
            f"Calculate the shared output in {item_plural}.",
            f"What total quantity of {item_plural} is delivered?",
            f"Find the combined delivery count of {item_plural}.",
            f"How many {item_plural} should {actor} record in total?",
        )
    question = _question(questions[plan.question_style])
    clauses.append(question)
    coverage.append(CoverageEntry(problem.target.node_id, len(clauses) - 1))
    if problem.context_node_id is not None:
        clauses = _contextualize(clauses, d.safe_context, plan.context_position)
        semantic = [i for i, clause in enumerate(clauses) if clause != d.safe_context]
        coverage = [
            CoverageEntry(entry.node_id, semantic[entry.clause_index]) for entry in coverage
        ]
    signature = RenderSignature(COMPILER_VERSION, "rates", str(kind), d.domain_id, plan)
    return CompiledRealization(
        _join(clauses),
        tuple(clauses),
        question,
        signature,
        tuple(coverage),
        (item_use,),
        tuple(units),
        problem.target.kind,
        (),
        True,
    )


def _discrete(problem: DiscreteProblemIR, plan: ClausePlan) -> CompiledRealization:
    d = problem.domain
    actor = d.actor.proper_name or "The planner"
    plural, item_use = noun_form(d.item.lexeme, 2)
    container_plural, container_use = noun_form(d.container.lexeme, 2)
    kind = problem.relation_kind
    clauses: list[str] = []
    coverage: list[CoverageEntry] = []
    if kind is DiscreteRelationKind.TWO_TYPE_ALLOCATION:
        total, resource, first, second = (
            _scalar(problem, key)
            for key in ("total", "resource_total", "first_cost", "second_cost")
        )
        clauses.extend(
            (
                f"At the {d.setting}, {actor} must assemble exactly {total} {plural} of two designs.",
                f"Design A uses {first} parts and design B uses {second} parts; together they use {resource} parts.",
            )
        )
        coverage.extend(
            (
                CoverageEntry(problem.scalars[0].node_id, 0),
                CoverageEntry(problem.scalars[1].node_id, 1),
                CoverageEntry(problem.scalars[2].node_id, 1),
                CoverageEntry(problem.scalars[3].node_id, 1),
            )
        )
        questions = tuple(
            f"{lead} number of design A {plural}?"
            for lead in (
                "What is the",
                "Determine the",
                "Calculate the",
                "Find the",
                "Report the",
                "What should be the",
                "How many is the exact",
                "Identify the",
            )
        )
    elif kind is DiscreteRelationKind.COMPLETE_PACKAGES:
        total, size = _scalar(problem, "total"), _scalar(problem, "package_size")
        clauses.append(
            f"At the {d.setting}, {total} {plural} are packed into complete {container_plural} holding {size} {plural} each."
        )
        coverage.extend(
            (
                CoverageEntry(problem.scalars[0].node_id, 0),
                CoverageEntry(problem.scalars[1].node_id, 0),
            )
        )
        questions = (
            f"How many complete {container_plural} can be filled?",
            f"What is the number of full {container_plural}?",
            f"Determine how many whole {container_plural} are available.",
            f"How many filled {container_plural} result?",
            f"Calculate the count of complete {container_plural}.",
            f"What number of full {container_plural} can {actor} prepare?",
            f"Find the complete-{container_plural} count.",
            f"How many {container_plural} contain a full allocation?",
        )
    elif kind is DiscreteRelationKind.EQUAL_DISTRIBUTION:
        total, containers = _scalar(problem, "total"), _scalar(problem, "containers")
        container_head, use = noun_form(d.container.lexeme, containers)
        clauses.append(
            f"At the {d.setting}, {actor} distributes {total} {plural} equally among {containers} {container_head}."
        )
        coverage.extend(
            (
                CoverageEntry(problem.scalars[0].node_id, 0),
                CoverageEntry(problem.scalars[1].node_id, 0),
            )
        )
        container_use = use
        questions = (
            f"How many {plural} go into each {d.container.lexeme.singular}?",
            f"What equal quantity of {plural} belongs in one {d.container.lexeme.singular}?",
            f"Determine the number of {plural} assigned to each {d.container.lexeme.singular}.",
            f"How many {plural} does every {d.container.lexeme.singular} receive?",
            f"Calculate the per-{d.container.lexeme.singular} allocation of {plural}.",
            f"What is the equal share of {plural} for one {d.container.lexeme.singular}?",
            f"Find the number of {plural} placed in each {d.container.lexeme.singular}.",
            f"How large is each {d.container.lexeme.singular}'s share of {plural}?",
        )
    else:
        first_resource, second_resource, first_per, second_per = (
            _scalar(problem, key)
            for key in ("first_resource", "second_resource", "first_per", "second_per")
        )
        clauses.extend(
            (
                f"At the {d.setting}, {actor} has {first_resource} amber parts and {second_resource} cobalt parts.",
                f"Each {d.item.lexeme.singular} requires {first_per} amber parts and {second_per} cobalt parts.",
            )
        )
        coverage.extend(
            (
                CoverageEntry(problem.scalars[0].node_id, 0),
                CoverageEntry(problem.scalars[1].node_id, 0),
                CoverageEntry(problem.scalars[2].node_id, 1),
                CoverageEntry(problem.scalars[3].node_id, 1),
            )
        )
        questions = (
            f"How many complete {plural} can be assembled?",
            f"What is the maximum number of whole {plural} that can be made?",
            f"Determine the assembly capacity measured in complete {plural}.",
            f"How many {plural} can both supplies support?",
            f"Calculate the maximum feasible count of {plural}.",
            f"What complete-{plural} capacity do the resources permit?",
            f"Find the greatest number of whole {plural} that can be produced.",
            f"How many completed {plural} should {actor} plan for?",
        )
    question = _question(questions[plan.question_style])
    clauses.append(question)
    coverage.append(CoverageEntry(problem.target.node_id, len(clauses) - 1))
    if problem.context_node_id is not None:
        clauses = _contextualize(clauses, d.safe_context, plan.context_position)
        semantic = [i for i, clause in enumerate(clauses) if clause != d.safe_context]
        coverage = [
            CoverageEntry(entry.node_id, semantic[entry.clause_index]) for entry in coverage
        ]
    signature = RenderSignature(COMPILER_VERSION, "discrete", str(kind), d.domain_id, plan)
    return CompiledRealization(
        _join(clauses),
        tuple(clauses),
        question,
        signature,
        tuple(coverage),
        (item_use, container_use),
        (),
        problem.target.kind,
        (),
        True,
    )


def compile_problem(problem: ProblemIR, plan: ClausePlan) -> CompiledRealization:
    """Compile typed semantic IR into English and auditable realization evidence."""

    if isinstance(problem, BookkeepingProblemIR):
        return _bookkeeping(problem, plan)
    if isinstance(problem, RateProblemIR):
        return _rate(problem, plan)
    if isinstance(problem, DiscreteProblemIR):
        return _discrete(problem, plan)
    raise TypeError("unsupported problem IR")


def compiler_capacity() -> dict[str, int]:
    """Return conservative content-free combinatorial capacity estimates."""

    independent = (
        INTRO_STYLE_COUNT
        * EVENT_STYLE_COUNT
        * QUESTION_STYLE_COUNT
        * 3
        * CONJUNCTION_STYLE_COUNT
        * NUMERIC_STYLE_COUNT
        * 2
    )
    return {
        "semantic_frames_bookkeeping": 2,
        "semantic_frames_rates": 5,
        "semantic_frames_discrete": 4,
        "sentence_plans_per_family": independent,
        "static_incompatible_combinations": 0,
    }
