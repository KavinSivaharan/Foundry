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
    MorphologyUse,
    RateProblemIR,
    RateRelationKind,
    RenderedUnitUse,
    RenderSignature,
    TargetKind,
    Voice,
)
from foundry.synthesis.realization.morphology import noun_form
from foundry.synthesis.template_bank.contracts import SentencePlanSpec, TemplateSpec

RENDERER_VERSION = "foundry-template-bank-realizer-v1"


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
    initial_noun, initial_use = noun_form(item, problem.initial.value)
    lead = template.semantic_frame.replace("_", " ")
    ledger = domain.primary_location.lexeme.singular
    openings = (
        f"For the {lead} at the {domain.setting}, {actor} records {problem.initial.value} {initial_noun} in the {ledger}.",
        f"The {lead} record for the {ledger} at the {domain.setting} starts with {problem.initial.value} {initial_noun}.",
        f"Before the scheduled movements at the {domain.setting}, the {ledger} contains {problem.initial.value} {initial_noun} for the {lead} count.",
        f"At the {domain.setting}, {actor}'s {lead} register opens with {problem.initial.value} {initial_noun} stored in the {ledger}.",
    )
    clauses = [_sentence(openings[index])]
    coverage = [CoverageEntry(problem.initial.node_id, 0)]
    morphology: list[MorphologyUse] = [initial_use]
    for ordinal, change in enumerate(problem.changes, start=1):
        amount = change.quantity.value
        noun, use = noun_form(item, amount)
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
        events = (
            f"Update {ordinal} has {actor} move {amount} {noun} from the {origin} to the {destination}.",
            f"In update {ordinal}, a transfer of {amount} {noun} is recorded from the {origin} to the {destination}.",
            f"From the {origin}, {actor} next moves {amount} {noun} into the {destination}.",
            f"The {lead} register then shows {amount} {noun} moving from the {origin} to the {destination}.",
        )
        clauses.append(_sentence(events[index]))
        coverage.append(CoverageEntry(change.node_id, len(clauses) - 1))
    plural, plural_use = noun_form(item, 2)
    morphology.append(plural_use)
    if problem.target.kind is TargetKind.GROUP_COUNT:
        if problem.group_size is None:
            raise ValueError("group-count target requires a group size")
        questions = (
            f"After all updates, how many complete groups of {problem.group_size} {plural} can be formed from the {ledger}",
            f"How many full sets, each containing {problem.group_size} {plural}, does the closing {ledger} inventory provide",
            f"Using the final balance, what number of complete {problem.group_size}-{item.attributive} groups can be made",
            f"When the register closes, how many whole groups of {problem.group_size} {plural} are available",
        )
    else:
        questions = (
            f"After all updates, how many {plural} remain in the {ledger}",
            f"What final quantity of {plural} should the {ledger} record",
            f"Using every listed movement, how many {plural} are left in the {ledger}",
            f"When the register closes, what is the {ledger}'s inventory of {plural}",
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
    plural, item_use = noun_form(domain.item.lexeme, 2)
    lead = template.semantic_frame.split(".", 1)[1].replace("_", " ")
    clauses: list[str] = []
    coverage: list[CoverageEntry] = []
    morphology: list[MorphologyUse] = [item_use]
    units: list[RenderedUnitUse] = []

    if problem.relation_kind is RateRelationKind.RATE_TOTAL:
        rate, intervals = _scalar(problem, "rate"), _scalar(problem, "intervals")
        facts = (
            f"a process produces {rate} {plural} per interval",
            f"{actor} records an output of {rate} {plural} during each interval",
            f"each operating interval yields {rate} {plural}",
            f"the {lead} process contributes {rate} {plural} every interval",
        )
        supports = (
            f"The process runs for {intervals} equal intervals.",
            f"Exactly {intervals} intervals are completed.",
            f"Production continues through {intervals} intervals.",
            f"The recorded run spans {intervals} intervals.",
        )
        questions = (
            f"How many {plural} are produced in total",
            f"What total number of {plural} does the run produce",
            f"Across all intervals, what is the complete output of {plural}",
            f"What quantity of {plural} should {actor} record for the full run",
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
            f"two collections of {plural} are in the ratio {first}:{second}",
            f"the first and second {lead} collections follow a {first}-to-{second} ratio",
            f"for every {first} parts in the first collection, the second collection has {second} parts",
            f"the paired collections maintain the exact proportion {first}:{second}",
        )
        supports = (
            f"The first collection contains {known} {plural}.",
            f"There are {known} {plural} in the first collection.",
            f"The known first collection has {known} {plural}.",
            f"{actor} counts {known} {plural} on the first side of the proportion.",
        )
        questions = (
            f"How many {plural} are in the second collection",
            f"What quantity of {plural} belongs in the second collection",
            f"What is the corresponding count of {plural} on the second side",
            f"How many {plural} complete the paired proportion",
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
            f"the full {lead} inventory consists of {base} {plural}",
            f"a recorded batch has {base} {plural} altogether",
        )
        supports = (
            f"Exactly {percent}% of the collection is selected.",
            f"The selected portion is {percent}% of that batch.",
            f"A {percent}% share is assigned to the sample.",
            f"The plan calls for choosing {percent}% of the complete inventory.",
        )
        questions = (
            f"How many {plural} are selected",
            f"What is the selected quantity of {plural}",
            f"How many {plural} make up that percentage share",
            f"What number of {plural} belongs to the chosen portion",
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
            panel, panel_use = noun_form(
                problem.target.unit.denominator or domain.item.lexeme, group.weight
            )
            mark, mark_use = noun_form(problem.target.unit.numerator, group.value)
            morphology.extend((panel_use, mark_use))
            forms = (
                f"Group {ordinal} contains {group.weight} {panel}, each with a value of {group.value} {mark}.",
                f"In group {ordinal}, {group.weight} {panel} are recorded at {group.value} {mark} per panel.",
                f"The {ordinal}th group contributes {group.weight} {panel} whose individual value is {group.value} {mark}.",
                f"For group {ordinal}, the record lists {group.weight} {panel} and {group.value} {mark} per panel.",
            )
            clauses.append(_sentence(forms[index]))
            coverage.append(CoverageEntry(group.node_id, len(clauses) - 1))
        questions = (
            "What is the exact weighted mean in marks per panel",
            "What panel-weighted average do all groups produce",
            "Across the groups, what is the exact mean value per panel",
            "What weighted average in marks per panel should be recorded",
        )
        units.append(RenderedUnitUse(problem.target.unit.unit_id, "marks", "panel"))
    else:
        first, second, intervals = (
            _scalar(problem, name) for name in ("first_rate", "second_rate", "intervals")
        )
        facts = (
            f"two channels deliver {first} and {second} {plural} per interval, respectively",
            f"one channel supplies {first} {plural} per interval while the other supplies {second}",
            f"the paired streams operate at {first} and {second} {plural} per interval",
            f"the {lead} process combines rates of {first} and {second} {plural} per interval",
        )
        supports = (
            f"Both channels operate for {intervals} intervals.",
            f"The two channels each run for {intervals} intervals.",
            f"Their shared operating period lasts {intervals} intervals.",
            f"{actor} keeps both streams active through {intervals} intervals.",
        )
        questions = (
            f"How many {plural} do the two channels deliver altogether",
            f"What is the combined output of {plural}",
            f"Across both streams and all intervals, how many {plural} are delivered",
            f"What total quantity of {plural} should {actor} record",
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
    plural, item_use = noun_form(domain.item.lexeme, 2)
    container_plural, container_use = noun_form(domain.container.lexeme, 2)
    lead = template.semantic_frame.split(".", 1)[1].replace("_", " ")
    coverage: list[CoverageEntry] = []
    morphology = [item_use, container_use]
    if problem.relation_kind is DiscreteRelationKind.TWO_TYPE_ALLOCATION:
        total, resource, first, second = (
            _scalar(problem, name)
            for name in ("total", "resource_total", "first_cost", "second_cost")
        )
        facts = (
            f"{actor} will make exactly {total} {plural}, split between type A and type B",
            f"a production order calls for {total} {plural} of two types",
            f"the {lead} plan contains a total of {total} {plural} across designs A and B",
            f"two designs together must account for exactly {total} {plural}",
        )
        conditions = (
            f"type A uses {first} parts, type B uses {second} parts, and the complete order uses {resource} parts",
            f"each A requires {first} parts and each B requires {second}, with {resource} parts used altogether",
            f"the two per-item requirements are {first} and {second} parts, and their combined use is {resource} parts",
            f"design A consumes {first} parts per item, design B consumes {second}, and total consumption is {resource} parts",
        )
        questions = (
            f"How many of the {plural} are type A",
            f"What is the exact count of type A {plural}",
            f"How many type A {plural} satisfy both conditions",
            f"What number of type A {plural} belongs in the completed order",
        )
    elif problem.relation_kind is DiscreteRelationKind.COMPLETE_PACKAGES:
        total, size = _scalar(problem, "total"), _scalar(problem, "package_size")
        facts = (
            f"{actor} has {total} {plural} available for packing",
            f"the packing inventory contains {total} {plural}",
            f"a total of {total} {plural} is available for the {lead} task",
            f"the packing area has {total} {plural} ready to be grouped",
        )
        conditions = (
            f"every complete {domain.container.lexeme.singular} must contain {size} {plural}",
            f"a full {domain.container.lexeme.singular} holds exactly {size} {plural}",
            f"only {domain.container.lexeme.plural} with all {size} {plural} count as complete",
            f"the packing rule assigns {size} {plural} to each complete {domain.container.lexeme.singular}",
        )
        questions = (
            f"How many complete {container_plural} can be filled",
            f"What number of full {container_plural} can {actor} prepare",
            f"How many whole {container_plural} are possible",
            f"What is the count of completely filled {container_plural}",
        )
    elif problem.relation_kind is DiscreteRelationKind.EQUAL_DISTRIBUTION:
        total, containers = _scalar(problem, "total"), _scalar(problem, "containers")
        facts = (
            f"{actor} has {total} {plural} to place among {containers} {container_plural}",
            f"the {lead} task distributes {total} {plural} across {containers} {container_plural}",
            f"there are {total} {plural} and {containers} receiving {container_plural}",
            f"{containers} {container_plural} must share a supply of {total} {plural}",
        )
        conditions = (
            f"every {domain.container.lexeme.singular} receives the same number of {plural}",
            f"the {plural} are divided equally with no remainder",
            "each destination gets an identical share",
            f"the allocation must be even across all {container_plural}",
        )
        questions = (
            f"How many {plural} does each {domain.container.lexeme.singular} receive",
            f"What equal share of {plural} goes to one {domain.container.lexeme.singular}",
            f"What is the number of {plural} assigned to every {domain.container.lexeme.singular}",
            f"How large is each {domain.container.lexeme.singular}'s share in {plural}",
        )
    else:
        first_resource, second_resource, first_per, second_per = (
            _scalar(problem, name)
            for name in ("first_resource", "second_resource", "first_per", "second_per")
        )
        facts = (
            f"{actor} has {first_resource} amber parts and {second_resource} cobalt parts",
            f"the {lead} inventory provides {first_resource} amber parts together with {second_resource} cobalt parts",
            f"two material stocks contain {first_resource} amber parts and {second_resource} cobalt parts",
            f"the available supplies are {first_resource} amber parts and {second_resource} cobalt parts",
        )
        conditions = (
            f"each {domain.item.lexeme.singular} requires {first_per} amber parts and {second_per} cobalt parts",
            f"one completed {domain.item.lexeme.singular} consumes {first_per} amber parts plus {second_per} cobalt parts",
            f"both requirements—{first_per} amber and {second_per} cobalt parts—must be met for every {domain.item.lexeme.singular}",
            f"a valid build uses {first_per} amber parts and {second_per} cobalt parts per {domain.item.lexeme.singular}",
        )
        questions = (
            f"How many complete {plural} can the two supplies support",
            f"What is the greatest number of {plural} that can be completed",
            f"How many {plural} can be made before either material becomes insufficient",
            f"What production capacity, measured in complete {plural}, do the supplies allow",
        )
    if index == 2:
        clauses = [
            _sentence(
                f"Provided that {conditions[index]}, the {lead} task can proceed at the {domain.setting}"
            ),
            _sentence(facts[index]),
        ]
        coverage_offset = {scalar.node_id: 1 for scalar in problem.scalars}
    else:
        clauses = [
            _sentence(f"At the {domain.setting}, {facts[index]}"),
            _sentence(f"The exact condition is that {conditions[index]}"),
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
