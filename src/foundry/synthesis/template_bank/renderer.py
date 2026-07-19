"""Deterministic realization of trusted semantic IR through the vetted-pending bank."""

# ruff: noqa: E501  # original reviewed-pending sentences remain legible as complete clauses

from __future__ import annotations

from dataclasses import replace

from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.realization.ir import (
    BookkeepingProblemIR,
    ClausePlan,
    CompiledRealization,
    CoverageEntry,
    DiscreteProblemIR,
    DiscreteRelationKind,
    LexemeSpec,
    MorphologyUse,
    RateProblemIR,
    RateRelationKind,
    RenderedUnitUse,
    RenderSignature,
    TargetKind,
    Voice,
)
from foundry.synthesis.template_bank.composition import NounPhraseSpec, numeric_ordinal
from foundry.synthesis.template_bank.contracts import SentencePlanSpec, TemplateSpec

RENDERER_VERSION = "foundry-template-bank-realizer-v3"


def _noun_form(lexeme: LexemeSpec, quantity: int) -> tuple[str, MorphologyUse]:
    """Route every emitted noun through the typed one-head composition layer."""

    return NounPhraseSpec(head=lexeme, quantity=quantity).render()


def _sentence(text: str) -> str:
    rendered = " ".join(text.split()).strip()
    if rendered and rendered[0].isalpha():
        rendered = rendered[0].upper() + rendered[1:]
    return rendered if rendered.endswith((".", "?")) else rendered + "."


def _question(text: str) -> str:
    rendered = " ".join(text.split()).strip().rstrip(".?")
    if rendered and rendered[0].isalpha():
        rendered = rendered[0].upper() + rendered[1:]
    return rendered + "?"


def _scalar(problem: RateProblemIR | DiscreteProblemIR, name: str) -> int:
    for scalar in problem.scalars:
        if scalar.name == name:
            return scalar.value
    raise ValueError(f"required scalar {name!r} is missing")


def _plan_index(template: TemplateSpec, plan: SentencePlanSpec) -> int:
    try:
        return template.sentence_plan_variants.index(plan)
    except ValueError as error:
        raise ValueError("sentence plan is not part of the selected template") from error


def _signature(
    template: TemplateSpec, plan: SentencePlanSpec, family: str, domain: str
) -> RenderSignature:
    index = _plan_index(template, plan)
    return RenderSignature(
        compiler_version=RENDERER_VERSION,
        problem_family=family,
        semantic_frame=f"{template.template_id}:{plan.plan_id}",
        domain_id=domain,
        plan=ClausePlan(
            intro_style=index,
            event_style=index,
            question_style=index,
            context_position=0,
            conjunction_style=index,
            numeric_style=0,
            voice=Voice.PASSIVE if index == 1 else Voice.ACTIVE,
        ),
    )


def _bookkeeping(
    problem: BookkeepingProblemIR, template: TemplateSpec, plan: SentencePlanSpec
) -> CompiledRealization:
    index = _plan_index(template, plan)
    domain = problem.domain
    actor = domain.actor.proper_name or "The inventory coordinator"
    item = domain.item.lexeme
    initial_noun, initial_use = _noun_form(item, problem.initial.value)
    ledger = domain.primary_location.lexeme.singular
    openings = (
        f"At the {domain.setting}, the {ledger} initially contains {problem.initial.value} {initial_noun}.",
        f"The {ledger} at the {domain.setting} starts with {problem.initial.value} {initial_noun}.",
        f"At the {domain.setting}, {actor} starts with {problem.initial.value} {initial_noun} in the {ledger}.",
        f"Before any transfers, the {ledger} in the {domain.setting} contains {problem.initial.value} {initial_noun}.",
    )
    clauses = [_sentence(openings[index])]
    coverage = [CoverageEntry(problem.initial.node_id, 0)]
    morphology: list[MorphologyUse] = [initial_use]
    for ordinal, change in enumerate(problem.changes, start=1):
        amount = change.quantity.value
        noun, use = _noun_form(item, amount)
        morphology.append(use)
        incoming = change.destination_id == "primary_location"
        origin = (
            domain.secondary_location.lexeme.singular
            if incoming
            else domain.primary_location.lexeme.singular
        )
        destination = (
            domain.primary_location.lexeme.singular
            if incoming
            else domain.destination_location.lexeme.singular
        )
        transition = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}.get(
            ordinal, f"Step {ordinal}"
        )
        events = (
            f"{transition}, {actor} moves {amount} {noun} from the {origin} to the {destination}.",
            f"{transition}, {amount} {noun} move from the {origin} to the {destination}.",
            f"{transition}, {actor} moves {amount} {noun} from the {origin} to the {destination}.",
            f"{transition}, {amount} {noun} are moved from the {origin} to the {destination}.",
        )
        clauses.append(_sentence(events[index]))
        coverage.append(CoverageEntry(change.node_id, len(clauses) - 1))
    plural, plural_use = _noun_form(item, 2)
    morphology.append(plural_use)
    if problem.target.kind is TargetKind.GROUP_COUNT:
        if problem.group_size is None:
            raise ValueError("group-count target requires a group size")
        questions = (
            f"After these changes, how many complete groups of {problem.group_size} {plural} can be made from the {plural} in the {ledger}",
            f"How many full groups of {problem.group_size} {plural} can be formed from the final inventory in the {ledger}",
            f"After all the moves, how many complete groups of {problem.group_size} {plural} can be made from the {ledger}'s contents",
            f"After every transfer, how many groups of {problem.group_size} {plural} can be formed from the {plural} left in the {ledger}",
        )
    else:
        questions = (
            f"How many {plural} are in the {ledger} after these changes",
            f"What is the final number of {plural} in the {ledger}",
            f"After all the moves, how many {plural} remain in the {ledger}",
            f"How many {plural} are left in the {ledger} after every transfer",
        )
    question = _question(questions[index])
    clauses.append(question)
    coverage.append(CoverageEntry(problem.target.node_id, len(clauses) - 1))
    return CompiledRealization(
        text=" ".join(clauses),
        clauses=tuple(clauses),
        question_clause=question,
        signature=_signature(template, plan, "bookkeeping", domain.domain_id),
        coverage=tuple(coverage),
        morphology_uses=tuple(morphology),
        unit_uses=(RenderedUnitUse(problem.target.unit.unit_id, plural, None),),
        rendered_target_kind=problem.target.kind,
        licensed_elisions=(),
        grammar_complete=True,
    )


def _rate(
    problem: RateProblemIR, template: TemplateSpec, plan: SentencePlanSpec
) -> CompiledRealization:
    index = _plan_index(template, plan)
    domain = problem.domain
    actor = domain.actor.proper_name or "The operator"
    plural, item_use = _noun_form(domain.item.lexeme, 2)
    clauses: list[str] = []
    coverage: list[CoverageEntry] = []
    morphology: list[MorphologyUse] = [item_use]
    units: list[RenderedUnitUse] = []

    if problem.relation_kind is RateRelationKind.RATE_TOTAL:
        rate, intervals = _scalar(problem, "rate"), _scalar(problem, "intervals")
        facts = (
            f"a process produces {rate} {plural} per interval",
            f"{actor} produces {rate} {plural} during each interval",
            f"each interval produces {rate} {plural}",
            f"the output rate is {rate} {plural} per interval",
        )
        supports = (
            f"The process runs for {intervals} equal intervals.",
            f"Exactly {intervals} intervals are completed.",
            f"This continues for {intervals} intervals.",
            f"The process lasts for {intervals} intervals.",
        )
        questions = (
            f"How many {plural} are produced in total",
            f"What total number of {plural} does the run produce",
            f"How many {plural} are produced across all {intervals} intervals",
            f"How many {plural} does the process produce altogether",
        )
        clauses = [
            _sentence(f"At the {domain.setting}, {facts[index]}"),
            _sentence(supports[index]),
        ]
        coverage = [
            CoverageEntry(problem.scalars[0].node_id, 0),
            CoverageEntry(problem.scalars[1].node_id, 1),
        ]
        units.append(RenderedUnitUse(problem.scalars[0].unit.unit_id, plural, "interval"))
    elif problem.relation_kind is RateRelationKind.RATIO_SCALE:
        first, second, known = (
            _scalar(problem, name) for name in ("first_part", "second_part", "known")
        )
        facts = (
            f"the ratio of {plural} in the first collection to {plural} in the second collection is {first}:{second}",
            f"the first collection and second collection contain {plural} in a {first}-to-{second} ratio",
            f"for every {first} parts in the first collection, there are {second} parts in the second collection",
            f"the two collections have a first-to-second ratio of {first}:{second}",
        )
        supports = (
            f"The first collection contains {known} {plural}.",
            f"There are {known} {plural} in the first collection.",
            f"The first collection has {known} {plural}.",
            f"{actor} counts {known} {plural} in the first collection.",
        )
        questions = (
            f"How many {plural} are in the second collection",
            f"How many {plural} does the second collection contain",
            f"How many {plural} are in the second collection",
            f"What is the number of {plural} in the second collection",
        )
        clauses = [
            _sentence(f"At the {domain.setting}, {facts[index]}"),
            _sentence(supports[index]),
        ]
        coverage = [
            CoverageEntry(problem.scalars[0].node_id, 0),
            CoverageEntry(problem.scalars[1].node_id, 0),
            CoverageEntry(problem.scalars[2].node_id, 1),
        ]
    elif problem.relation_kind is RateRelationKind.PERCENTAGE:
        base, percent = _scalar(problem, "base"), _scalar(problem, "percent")
        facts = (
            f"a collection contains {base} {plural}",
            f"{actor} begins with a batch of {base} {plural}",
            f"there are {base} {plural} in a collection",
            f"a batch contains {base} {plural}",
        )
        supports = (
            f"Exactly {percent}% of the collection is selected.",
            f"The selected portion is {percent}% of that batch.",
            f"Exactly {percent}% of the {plural} are chosen for inspection.",
            f"The selection covers {percent}% of the batch.",
        )
        questions = (
            f"How many {plural} are selected",
            f"What is the selected quantity of {plural}",
            f"How many {plural} are chosen for inspection",
            f"How many {plural} are selected",
        )
        clauses = [
            _sentence(f"At the {domain.setting}, {facts[index]}"),
            _sentence(supports[index]),
        ]
        coverage = [
            CoverageEntry(problem.scalars[0].node_id, 0),
            CoverageEntry(problem.scalars[1].node_id, 1),
        ]
    elif problem.relation_kind is RateRelationKind.WEIGHTED_MEAN:
        if len({(group.weight, group.value) for group in problem.groups}) != len(problem.groups):
            raise ValueError("weighted groups must be unique")
        for ordinal, group in enumerate(problem.groups, start=1):
            panel, panel_use = _noun_form(
                problem.target.unit.denominator or domain.item.lexeme, group.weight
            )
            mark, mark_use = _noun_form(problem.target.unit.numerator, group.value)
            morphology.extend((panel_use, mark_use))
            forms = (
                f"Group {ordinal} has {group.weight} {panel}, each worth {group.value} {mark}.",
                f"The {numeric_ordinal(ordinal)} group has {group.weight} {panel} worth {group.value} {mark} each.",
                f"Group {ordinal} includes {group.weight} {panel}, with each one worth {group.value} {mark}.",
                f"There are {group.weight} {panel} in group {ordinal}, and each is worth {group.value} {mark}.",
            )
            clauses.append(_sentence(forms[index]))
            coverage.append(CoverageEntry(group.node_id, len(clauses) - 1))
        questions = (
            "What is the average value per panel across all the panels",
            "What is the average number of marks per panel across all the groups",
            "Across all the groups, what is the average value per panel",
            "What is the combined average in marks per panel",
        )
        units.append(RenderedUnitUse(problem.target.unit.unit_id, "marks", "panel"))
    else:
        first, second, intervals = (
            _scalar(problem, name) for name in ("first_rate", "second_rate", "intervals")
        )
        facts = (
            f"two channels deliver {first} and {second} {plural} per interval, respectively",
            f"one channel supplies {first} {plural} per interval while the other supplies {second} {plural} per interval",
            f"one stream produces {first} {plural} per interval and a second stream produces {second} {plural} per interval",
            f"two processes produce {first} and {second} {plural} per interval, respectively",
        )
        supports = (
            f"Both channels operate for {intervals} intervals.",
            f"The two channels each run for {intervals} intervals.",
            f"Both streams run for {intervals} intervals.",
            f"{actor} runs both processes for {intervals} intervals.",
        )
        questions = (
            f"How many {plural} do the two channels deliver altogether",
            f"What is the combined output of {plural}",
            f"How many {plural} do both streams produce altogether",
            f"How many {plural} do the two processes produce in total",
        )
        clauses = [
            _sentence(f"At the {domain.setting}, {facts[index]}"),
            _sentence(supports[index]),
        ]
        coverage = [
            CoverageEntry(problem.scalars[0].node_id, 0),
            CoverageEntry(problem.scalars[1].node_id, 0),
            CoverageEntry(problem.scalars[2].node_id, 1),
        ]
        units.extend(
            (
                RenderedUnitUse(problem.scalars[0].unit.unit_id, plural, "interval"),
                RenderedUnitUse(problem.scalars[1].unit.unit_id, plural, "interval"),
            )
        )
    question = _question(questions[index])
    clauses.append(question)
    coverage.append(CoverageEntry(problem.target.node_id, len(clauses) - 1))
    return CompiledRealization(
        text=" ".join(clauses),
        clauses=tuple(clauses),
        question_clause=question,
        signature=_signature(template, plan, "rates", domain.domain_id),
        coverage=tuple(coverage),
        morphology_uses=tuple(morphology),
        unit_uses=tuple(units),
        rendered_target_kind=problem.target.kind,
        licensed_elisions=(),
        grammar_complete=True,
    )


def _discrete(
    problem: DiscreteProblemIR, template: TemplateSpec, plan: SentencePlanSpec
) -> CompiledRealization:
    index = _plan_index(template, plan)
    domain = problem.domain
    actor = domain.actor.proper_name or "The planner"
    plural, item_use = _noun_form(domain.item.lexeme, 2)
    singular, singular_use = _noun_form(domain.item.lexeme, 1)
    container_plural, container_use = _noun_form(domain.container.lexeme, 2)
    coverage: list[CoverageEntry] = []
    morphology = [item_use, singular_use, container_use]
    if problem.relation_kind is DiscreteRelationKind.TWO_TYPE_ALLOCATION:
        total, resource, first, second = (
            _scalar(problem, name)
            for name in ("total", "resource_total", "first_cost", "second_cost")
        )
        facts = (
            f"{actor} will make exactly {total} {plural}, with each one classified as type A or type B",
            f"a batch contains {total} {plural} split between type A and type B",
            f"the order contains {total} {plural} divided between types A and B",
            f"there are exactly {total} {plural} in total, consisting of type A and type B items",
        )
        conditions = (
            f"each type A {singular} uses {first} parts, each type B {singular} uses {second} parts, and all {plural} use {resource} parts altogether",
            f"a type A {singular} requires {first} parts, a type B {singular} requires {second} parts, and the batch uses {resource} parts in total",
            f"type A {plural} use {first} parts each, type B {plural} use {second} parts each, and the combined requirement is {resource} parts",
            f"each type A {singular} needs {first} parts and each type B {singular} needs {second} parts, for a total of {resource} parts",
        )
        questions = (
            f"How many of the {plural} are type A",
            f"How many type A {plural} are in the batch",
            f"How many of the {plural} must be type A",
            f"What is the number of type A {plural}",
        )
    elif problem.relation_kind is DiscreteRelationKind.COMPLETE_PACKAGES:
        total, size = _scalar(problem, "total"), _scalar(problem, "package_size")
        facts = (
            f"{actor} has {total} {plural} available for packing",
            f"there are {total} {plural} available",
            f"there are {total} {plural} available for packing",
            f"{actor} needs to place {total} {plural} into containers",
        )
        conditions = (
            f"each full {domain.container.lexeme.singular} contains {size} {plural}",
            f"every {domain.container.lexeme.singular} holds exactly {size} {plural}",
            f"a {domain.container.lexeme.singular} is full when it contains {size} {plural}",
            f"{size} {plural} must be placed in each {domain.container.lexeme.singular}",
        )
        questions = (
            f"How many full {container_plural} can be filled",
            f"How many full {container_plural} can {actor} prepare",
            f"How many complete {container_plural} can be made",
            f"How many {container_plural} can be filled completely",
        )
    elif problem.relation_kind is DiscreteRelationKind.EQUAL_DISTRIBUTION:
        total, containers = _scalar(problem, "total"), _scalar(problem, "containers")
        facts = (
            f"{actor} has {total} {plural} to place among {containers} {container_plural}",
            f"the plan distributes {total} {plural} across {containers} {container_plural}",
            f"there are {total} {plural} to divide among {containers} {container_plural}",
            f"a total of {total} {plural} must be shared by {containers} {container_plural}",
        )
        conditions = (
            f"every {domain.container.lexeme.singular} receives the same number of {plural}",
            f"the {plural} are divided equally with no remainder",
            f"each {domain.container.lexeme.singular} gets the same number of {plural}",
            f"the allocation must be even across all {container_plural}",
        )
        questions = (
            f"How many {plural} does each {domain.container.lexeme.singular} receive",
            f"How many {plural} go to each {domain.container.lexeme.singular}",
            f"How many {plural} does every {domain.container.lexeme.singular} receive",
            f"How many {plural} are assigned to each {domain.container.lexeme.singular}",
        )
    else:
        first_resource, second_resource, first_per, second_per = (
            _scalar(problem, name)
            for name in ("first_resource", "second_resource", "first_per", "second_per")
        )
        facts = (
            f"{actor} has {first_resource} amber parts and {second_resource} cobalt parts",
            f"there are {first_resource} amber parts and {second_resource} cobalt parts available",
            f"the supplies include {first_resource} amber parts and {second_resource} cobalt parts",
            f"the available materials are {first_resource} amber parts and {second_resource} cobalt parts",
        )
        conditions = (
            f"each {domain.item.lexeme.singular} requires {first_per} amber parts and {second_per} cobalt parts",
            f"one {domain.item.lexeme.singular} uses {first_per} amber parts and {second_per} cobalt parts",
            f"every {domain.item.lexeme.singular} needs {first_per} amber parts and {second_per} cobalt parts",
            f"making one {domain.item.lexeme.singular} requires {first_per} amber parts and {second_per} cobalt parts",
        )
        questions = (
            f"How many complete {plural} can be made",
            f"What is the greatest number of {plural} that can be made",
            f"How many {plural} can be made before either type of part runs out",
            f"How many complete {plural} can the available parts produce",
        )
    if index == 2:
        clauses = [
            _sentence(f"At the {domain.setting}, {conditions[index]}"),
            _sentence(f"In total, {facts[index]}"),
        ]
        fact_scalar_names = {
            DiscreteRelationKind.TWO_TYPE_ALLOCATION: {"total"},
            DiscreteRelationKind.COMPLETE_PACKAGES: {"total"},
            DiscreteRelationKind.EQUAL_DISTRIBUTION: {"total", "containers"},
            DiscreteRelationKind.DUAL_CAPACITY: {
                "first_resource",
                "second_resource",
            },
        }[problem.relation_kind]
        coverage_offset = {
            scalar.node_id: 1 if scalar.name in fact_scalar_names else 0
            for scalar in problem.scalars
        }
    else:
        clauses = [
            _sentence(f"At the {domain.setting}, {facts[index]}"),
            _sentence(conditions[index]),
        ]
        coverage_offset = {
            scalar.node_id: (
                0
                if scalar.name in {"total", "containers", "first_resource", "second_resource"}
                else 1
            )
            for scalar in problem.scalars
        }
    for scalar in problem.scalars:
        coverage.append(CoverageEntry(scalar.node_id, coverage_offset[scalar.node_id]))
    question = _question(questions[index])
    clauses.append(question)
    coverage.append(CoverageEntry(problem.target.node_id, len(clauses) - 1))
    return CompiledRealization(
        text=" ".join(clauses),
        clauses=tuple(clauses),
        question_clause=question,
        signature=_signature(template, plan, "discrete", domain.domain_id),
        coverage=tuple(coverage),
        morphology_uses=tuple(morphology),
        unit_uses=(RenderedUnitUse(problem.target.unit.unit_id, plural, None),),
        rendered_target_kind=problem.target.kind,
        licensed_elisions=(),
        grammar_complete=True,
    )


def render_with_template(
    draft: CandidateDraft, template: TemplateSpec, plan: SentencePlanSpec
) -> CandidateDraft:
    """Fill one compatible bank plan from trusted IR and return a new immutable draft."""

    if draft.difficulty_level not in template.supported_difficulty_levels:
        raise ValueError("template does not support this difficulty")
    if draft.problem_ir.target.kind not in template.compatible_target_types:
        raise ValueError("template target type is incompatible with semantic IR")
    if draft.target_failure_category != template.reasoning_category:
        raise ValueError("template category is incompatible with semantic IR")
    if isinstance(draft.problem_ir, BookkeepingProblemIR):
        realization = _bookkeeping(draft.problem_ir, template, plan)
    elif isinstance(draft.problem_ir, RateProblemIR):
        expected = str(draft.problem_ir.relation_kind)
        if not template.semantic_frame.startswith(expected + "."):
            raise ValueError("rate semantic frame is incompatible with relation")
        realization = _rate(draft.problem_ir, template, plan)
    elif isinstance(draft.problem_ir, DiscreteProblemIR):
        expected = str(draft.problem_ir.relation_kind)
        if not template.semantic_frame.startswith(expected + "."):
            raise ValueError("discrete semantic frame is incompatible with relation")
        realization = _discrete(draft.problem_ir, template, plan)
    else:
        raise TypeError("unsupported semantic IR")
    quality = replace(
        draft.quality_metadata,
        scenario_id=template.template_id,
        renderer_family=RENDERER_VERSION,
        clauses=realization.clauses,
        target_mentions=1,
        conclusion=realization.question_clause,
        grammar_complete=realization.grammar_complete,
    )
    structure = {
        **draft.structure_signature,
        "template_id": template.template_id,
        "template_version": template.template_version,
        "sentence_plan_id": plan.plan_id,
        "render_signature_sha256": template.render_signature_hash(plan),
    }
    return replace(
        draft,
        rendered_question=realization.text,
        realization=realization,
        quality_metadata=quality,
        structure_signature=structure,
    )
