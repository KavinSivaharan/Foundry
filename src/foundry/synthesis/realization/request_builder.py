"""Value-blind realization requests compiled from procedural semantic IR."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import count

from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.realization.ir import (
    BookkeepingProblemIR,
    CountBehavior,
    DiscreteProblemIR,
    DiscreteRelationKind,
    ProblemIR,
    RateProblemIR,
    RateRelationKind,
    TargetKind,
    UnitSpec,
)
from foundry.synthesis.realization.model_contracts import (
    PlaceholderKind,
    PlaceholderSpec,
    RealizationRequest,
    SemanticEventSpec,
    StyleControls,
)


@dataclass(frozen=True)
class PreparedRealizationRequest:
    """A model request plus values withheld until deterministic compilation."""

    draft: CandidateDraft
    request: RealizationRequest
    replacements: dict[str, str]
    semantic_frame: str
    realization_signature: str

    @property
    def request_sha256(self) -> str:
        """Hash the value-blind request for replay evidence."""

        payload = json.dumps(asdict(self.request), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _RequestBuilder:
    def __init__(self, draft: CandidateDraft) -> None:
        self.draft = draft
        self.slots: list[PlaceholderSpec] = []
        self.replacements: dict[str, str] = {}
        self.events: list[SemanticEventSpec] = []
        self._slot_counter = count(1)

    def slot(
        self,
        prefix: str,
        kind: PlaceholderKind,
        node_id: str,
        replacement: str,
    ) -> str:
        token = f"<{prefix}_{next(self._slot_counter)}>"
        self.slots.append(PlaceholderSpec(token, kind, node_id))
        self.replacements[token] = replacement
        return token

    def event(self, node_id: str, description: str, slots: tuple[str, ...]) -> None:
        self.events.append(SemanticEventSpec(node_id, description, slots))


def _noun(unit: UnitSpec, quantity: int) -> str:
    lexeme = unit.numerator
    if lexeme.count_behavior is CountBehavior.MASS or quantity == 1:
        return lexeme.singular
    return lexeme.plural


def _location_name(problem: ProblemIR, which: str) -> str:
    entity = {
        "primary": problem.domain.primary_location,
        "secondary": problem.domain.secondary_location,
        "destination": problem.domain.destination_location,
    }[which]
    return entity.lexeme.singular


def _target_intent(kind: TargetKind) -> str:
    return {
        TargetKind.COUNT: "ask for the count of the specified target type",
        TargetKind.TOTAL_QUANTITY: "ask for the total quantity produced or available",
        TargetKind.REMAINING_QUANTITY: "ask for the quantity remaining after every event",
        TargetKind.RATE: "ask for the specified rate with its interval",
        TargetKind.PERCENTAGE: "ask for the quantity represented by the stated percentage",
        TargetKind.RATIO: "ask for the unknown quantity implied by the stated ratio",
        TargetKind.WEIGHTED_MEAN: "ask for the weighted mean across all stated groups",
        TargetKind.VALID_ASSIGNMENT_COUNT: "ask for the number of valid assignments",
        TargetKind.CAPACITY: "ask for the maximum capacity allowed by every constraint",
        TargetKind.GROUP_COUNT: "ask for the number of complete groups",
    }[kind]


def _build_bookkeeping(builder: _RequestBuilder, problem: BookkeepingProblemIR) -> str:
    actor = builder.slot(
        "ENTITY",
        PlaceholderKind.ENTITY,
        problem.initial.node_id,
        problem.domain.actor.proper_name or "the operator",
    )
    initial_quantity = builder.slot(
        "QUANTITY", PlaceholderKind.QUANTITY, problem.initial.node_id, str(problem.initial.value)
    )
    initial_unit = builder.slot(
        "UNIT",
        PlaceholderKind.UNIT,
        problem.initial.node_id,
        _noun(problem.initial.unit, problem.initial.value),
    )
    location = builder.slot(
        "LOCATION",
        PlaceholderKind.LOCATION,
        problem.initial.node_id,
        _location_name(problem, "primary"),
    )
    builder.event(
        problem.initial.node_id,
        f"state that {actor} begins with {initial_quantity} {initial_unit} at {location}",
        (actor, initial_quantity, initial_unit, location),
    )
    for change in problem.changes:
        quantity = builder.slot(
            "QUANTITY", PlaceholderKind.QUANTITY, change.node_id, str(change.quantity.value)
        )
        unit = builder.slot(
            "UNIT",
            PlaceholderKind.UNIT,
            change.node_id,
            _noun(change.quantity.unit, change.quantity.value),
        )
        if change.origin_id == problem.domain.secondary_location.entity_id:
            origin_value = _location_name(problem, "secondary")
            destination_value = _location_name(problem, "primary")
            event_kind = "state a transfer into the tracked inventory"
        else:
            origin_value = _location_name(problem, "primary")
            destination_value = _location_name(problem, "destination")
            event_kind = "state a transfer out of the tracked inventory"
        origin = builder.slot("LOCATION", PlaceholderKind.LOCATION, change.node_id, origin_value)
        destination = builder.slot(
            "LOCATION", PlaceholderKind.LOCATION, change.node_id, destination_value
        )
        builder.event(
            change.node_id,
            f"{event_kind}: move {quantity} {unit} from {origin} to {destination}",
            (quantity, unit, origin, destination),
        )
    target_location = builder.slot(
        "LOCATION",
        PlaceholderKind.LOCATION,
        problem.target.node_id,
        _location_name(problem, "primary"),
    )
    target_unit = builder.slot(
        "TARGET_ENTITY",
        PlaceholderKind.TARGET_ENTITY,
        problem.target.node_id,
        problem.target.unit.numerator.plural,
    )
    target_slots = [target_location, target_unit]
    description = f"ask how many {target_unit} remain at {target_location} after all events"
    if problem.group_size is not None:
        group_size = builder.slot(
            "QUANTITY", PlaceholderKind.QUANTITY, problem.target.node_id, str(problem.group_size)
        )
        item = builder.slot(
            "UNIT",
            PlaceholderKind.UNIT,
            problem.target.node_id,
            problem.domain.item.lexeme.plural,
        )
        target_slots.extend((group_size, item))
        description = (
            f"ask how many complete {target_unit} of {group_size} {item} can be formed "
            f"from the final inventory at {target_location}"
        )
    builder.event(problem.target.node_id, description, tuple(target_slots))
    return "bookkeeping:" + ("grouping" if problem.group_size is not None else "inventory")


def _rate_scalar_slots(
    builder: _RequestBuilder, problem: RateProblemIR
) -> dict[str, tuple[str, str, str | None]]:
    result: dict[str, tuple[str, str, str | None]] = {}
    for scalar in problem.scalars:
        quantity = builder.slot(
            "QUANTITY", PlaceholderKind.QUANTITY, scalar.node_id, str(scalar.value)
        )
        unit = builder.slot(
            "UNIT", PlaceholderKind.UNIT, scalar.node_id, _noun(scalar.unit, scalar.value)
        )
        denominator: str | None = None
        if scalar.unit.denominator is not None:
            denominator = builder.slot(
                "RATE_INTERVAL",
                PlaceholderKind.RATE_INTERVAL,
                scalar.node_id,
                scalar.unit.denominator.singular,
            )
        result[scalar.node_id] = (quantity, unit, denominator)
    return result


def _build_rates(builder: _RequestBuilder, problem: RateProblemIR) -> str:
    slots = _rate_scalar_slots(builder, problem)
    actor_name = problem.domain.actor.proper_name or "the operator"
    location_name = problem.domain.setting
    kind = problem.relation_kind
    if kind is RateRelationKind.RATE_TOTAL:
        rate_q, rate_u, interval = slots["rate"]
        duration_q, duration_u, _ = slots["intervals"]
        actor = builder.slot("ENTITY", PlaceholderKind.ENTITY, "rate", actor_name)
        location = builder.slot("LOCATION", PlaceholderKind.LOCATION, "rate", location_name)
        assert interval is not None
        builder.event(
            "rate",
            f"state that {actor} handles {rate_q} {rate_u} per {interval} at {location}",
            (actor, rate_q, rate_u, interval, location),
        )
        builder.event(
            "intervals",
            f"state that this rate continues for {duration_q} {duration_u}",
            (duration_q, duration_u),
        )
    elif kind is RateRelationKind.RATIO_SCALE:
        first_q, first_u, _ = slots["first_part"]
        second_q, second_u, _ = slots["second_part"]
        known_q, known_u, _ = slots["known"]
        first = builder.slot("ENTITY", PlaceholderKind.ENTITY, "first_part", "the first group")
        second = builder.slot("ENTITY", PlaceholderKind.ENTITY, "second_part", "the second group")
        builder.event(
            "first_part",
            (
                f"state that {first} and {second} have a ratio of {first_q} {first_u} "
                f"to {second_q} {second_u}"
            ),
            (first, second, first_q, first_u, second_q, second_u),
        )
        known_group = builder.slot("ENTITY", PlaceholderKind.ENTITY, "known", "the first group")
        builder.event(
            "known",
            f"state that {known_group} is represented by {known_q} {known_u}",
            (known_group, known_q, known_u),
        )
    elif kind is RateRelationKind.PERCENTAGE:
        base_q, base_u, _ = slots["base"]
        percent_q, percent_u, _ = slots["percent"]
        collection = builder.slot(
            "ENTITY", PlaceholderKind.ENTITY, "base", "the complete collection"
        )
        builder.event(
            "base",
            f"state that {collection} contains {base_q} {base_u}",
            (collection, base_q, base_u),
        )
        builder.event(
            "percent",
            (
                f"state that the selected share is {percent_q} {percent_u} "
                "of that collection without calculating it"
            ),
            (percent_q, percent_u),
        )
    elif kind is RateRelationKind.WEIGHTED_MEAN:
        for index, group in enumerate(problem.groups, start=1):
            group_name = builder.slot(
                "ENTITY", PlaceholderKind.ENTITY, group.node_id, f"group {chr(64 + index)}"
            )
            weight = builder.slot(
                "QUANTITY", PlaceholderKind.QUANTITY, group.node_id, str(group.weight)
            )
            panels = builder.slot(
                "UNIT",
                PlaceholderKind.UNIT,
                group.node_id,
                "panel" if group.weight == 1 else "panels",
            )
            value = builder.slot(
                "QUANTITY", PlaceholderKind.QUANTITY, group.node_id, str(group.value)
            )
            marks = builder.slot(
                "UNIT", PlaceholderKind.UNIT, group.node_id, "mark" if group.value == 1 else "marks"
            )
            interval = builder.slot(
                "RATE_INTERVAL", PlaceholderKind.RATE_INTERVAL, group.node_id, "panel"
            )
            builder.event(
                group.node_id,
                (
                    f"state once that {group_name} contains {weight} {panels} "
                    f"averaging {value} {marks} per {interval}"
                ),
                (group_name, weight, panels, value, marks, interval),
            )
    else:
        first_q, first_u, first_interval = slots["first_rate"]
        second_q, second_u, second_interval = slots["second_rate"]
        duration_q, duration_u, _ = slots["intervals"]
        first = builder.slot("ENTITY", PlaceholderKind.ENTITY, "first_rate", "the first stream")
        second = builder.slot("ENTITY", PlaceholderKind.ENTITY, "second_rate", "the second stream")
        assert first_interval is not None and second_interval is not None
        builder.event(
            "first_rate",
            f"state that {first} provides {first_q} {first_u} per {first_interval}",
            (first, first_q, first_u, first_interval),
        )
        builder.event(
            "second_rate",
            f"state that {second} provides {second_q} {second_u} per {second_interval}",
            (second, second_q, second_u, second_interval),
        )
        builder.event(
            "intervals",
            f"state that both streams operate for {duration_q} {duration_u}",
            (duration_q, duration_u),
        )

    target_item = builder.slot(
        "TARGET_ENTITY",
        PlaceholderKind.TARGET_ENTITY,
        problem.target.node_id,
        problem.target.unit.numerator.plural,
    )
    builder.event(
        problem.target.node_id,
        f"{_target_intent(problem.target.kind)} for {target_item}",
        (target_item,),
    )
    return f"rates:{kind}"


def _build_discrete(builder: _RequestBuilder, problem: DiscreteProblemIR) -> str:
    scalar_map = {scalar.node_id: scalar for scalar in problem.scalars}

    def scalar_slots(node_id: str) -> tuple[str, str, str | None]:
        scalar = scalar_map[node_id]
        quantity = builder.slot("QUANTITY", PlaceholderKind.QUANTITY, node_id, str(scalar.value))
        unit = builder.slot("UNIT", PlaceholderKind.UNIT, node_id, _noun(scalar.unit, scalar.value))
        denominator = None
        if scalar.unit.denominator is not None:
            denominator = builder.slot(
                "RATE_INTERVAL",
                PlaceholderKind.RATE_INTERVAL,
                node_id,
                scalar.unit.denominator.singular,
            )
        return quantity, unit, denominator

    kind = problem.relation_kind
    if kind is DiscreteRelationKind.TWO_TYPE_ALLOCATION:
        total_q, total_u, _ = scalar_slots("total")
        resource_q, resource_u, _ = scalar_slots("resource_total")
        first_q, first_u, first_den = scalar_slots("first_cost")
        second_q, second_u, second_den = scalar_slots("second_cost")
        first = builder.slot("ENTITY", PlaceholderKind.ENTITY, "first_cost", "design A")
        second = builder.slot("ENTITY", PlaceholderKind.ENTITY, "second_cost", "design B")
        assert first_den is not None and second_den is not None
        builder.event(
            "total",
            f"state that exactly {total_q} {total_u} consist of {first} and {second} together",
            (total_q, total_u, first, second),
        )
        builder.event(
            "resource_total",
            f"state that all items together use {resource_q} {resource_u}",
            (resource_q, resource_u),
        )
        builder.event(
            "first_cost",
            f"state that each {first} uses {first_q} {first_u} per {first_den}",
            (first_q, first_u, first_den),
        )
        builder.event(
            "second_cost",
            f"state that each {second} uses {second_q} {second_u} per {second_den}",
            (second_q, second_u, second_den),
        )
        target_name = "design A items"
    elif kind is DiscreteRelationKind.COMPLETE_PACKAGES:
        total_q, total_u, _ = scalar_slots("total")
        size_q, size_u, _ = scalar_slots("package_size")
        collection = builder.slot("ENTITY", PlaceholderKind.ENTITY, "total", "the inventory")
        container = builder.slot(
            "ENTITY", PlaceholderKind.ENTITY, "package_size", problem.domain.container.lexeme.plural
        )
        builder.event(
            "total",
            f"state that {collection} contains {total_q} {total_u}",
            (collection, total_q, total_u),
        )
        builder.event(
            "package_size",
            f"state that each of the {container} requires exactly {size_q} {size_u}",
            (container, size_q, size_u),
        )
        target_name = f"complete {problem.domain.container.lexeme.plural}"
    elif kind is DiscreteRelationKind.EQUAL_DISTRIBUTION:
        total_q, total_u, _ = scalar_slots("total")
        containers_q, containers_u, _ = scalar_slots("containers")
        collection = builder.slot("ENTITY", PlaceholderKind.ENTITY, "total", "the inventory")
        builder.event(
            "total",
            f"state that {collection} contains {total_q} {total_u}",
            (collection, total_q, total_u),
        )
        builder.event(
            "containers",
            (
                "state that the full inventory is distributed equally among "
                f"{containers_q} {containers_u}"
            ),
            (containers_q, containers_u),
        )
        target_name = (
            f"{problem.domain.item.lexeme.plural} per {problem.domain.container.lexeme.singular}"
        )
    else:
        first_resource_q, first_resource_u, _ = scalar_slots("first_resource")
        second_resource_q, second_resource_u, _ = scalar_slots("second_resource")
        first_per_q, first_per_u, first_per_den = scalar_slots("first_per")
        second_per_q, second_per_u, second_per_den = scalar_slots("second_per")
        amber = builder.slot("ENTITY", PlaceholderKind.ENTITY, "first_resource", "amber stock")
        cobalt = builder.slot("ENTITY", PlaceholderKind.ENTITY, "second_resource", "cobalt stock")
        assert first_per_den is not None and second_per_den is not None
        builder.event(
            "first_resource",
            f"state that {amber} contains {first_resource_q} {first_resource_u}",
            (amber, first_resource_q, first_resource_u),
        )
        builder.event(
            "second_resource",
            f"state that {cobalt} contains {second_resource_q} {second_resource_u}",
            (cobalt, second_resource_q, second_resource_u),
        )
        builder.event(
            "first_per",
            (
                f"state that each target item requires {first_per_q} {first_per_u} "
                f"per {first_per_den} from amber stock"
            ),
            (first_per_q, first_per_u, first_per_den),
        )
        builder.event(
            "second_per",
            (
                f"state that each target item requires {second_per_q} {second_per_u} "
                f"per {second_per_den} from cobalt stock"
            ),
            (second_per_q, second_per_u, second_per_den),
        )
        target_name = problem.domain.item.lexeme.plural

    target = builder.slot(
        "TARGET_ENTITY", PlaceholderKind.TARGET_ENTITY, problem.target.node_id, target_name
    )
    builder.event(
        problem.target.node_id,
        f"{_target_intent(problem.target.kind)} for {target}",
        (target,),
    )
    return f"discrete:{kind}"


def _allowed_orders(
    events: tuple[SemanticEventSpec, ...], family: str
) -> tuple[tuple[str, ...], ...]:
    node_ids = tuple(event.node_id for event in events)
    if family.startswith("bookkeeping") or len(node_ids) <= 2:
        return (node_ids,)
    facts, target = node_ids[:-1], node_ids[-1]
    alternatives = [node_ids]
    reversed_facts = (*reversed(facts), target)
    if reversed_facts != node_ids:
        alternatives.append(reversed_facts)
    return tuple(alternatives)


def prepare_realization_request(
    draft: CandidateDraft, *, style_variant: int
) -> PreparedRealizationRequest:
    """Compile one procedural draft into a value-blind model request."""

    if style_variant < 0:
        raise ValueError("style_variant must be nonnegative")
    builder = _RequestBuilder(draft)
    problem = draft.problem_ir
    if isinstance(problem, BookkeepingProblemIR):
        semantic_frame = _build_bookkeeping(builder, problem)
    elif isinstance(problem, RateProblemIR):
        semantic_frame = _build_rates(builder, problem)
    elif isinstance(problem, DiscreteProblemIR):
        semantic_frame = _build_discrete(builder, problem)
    else:
        raise TypeError("unsupported procedural semantic IR")
    events = tuple(builder.events)
    style_ids = (
        "concise_chronological",
        "direct_operational",
        "compact_narrative",
        "formal_neutral",
        "location_first",
        "question_focused",
    )
    style_id = style_ids[style_variant % len(style_ids)]
    request = RealizationRequest(
        request_id=draft.candidate_id,
        category=draft.target_failure_category,
        semantic_frame=semantic_frame,
        ordered_events=events,
        placeholders=tuple(builder.slots),
        target_type=problem.target.kind,
        required_question_intent=_target_intent(problem.target.kind),
        allowed_discourse_orders=_allowed_orders(events, semantic_frame),
        forbidden_transformations=(
            "do not add, remove, merge, duplicate, or calculate semantic facts",
            "do not alter placeholders, units, intervals, constraints, target, or intent",
            "do not use pronouns, raw numbers, equations, answers, or reasoning",
            "do not emit markdown or text outside the one JSON object",
        ),
        style=StyleControls(
            style_id=style_id,
            difficulty=str(draft.difficulty_level),
            permitted_voices=("active", "natural_passive"),
            allow_safe_context=False,
        ),
    )
    signature_material = (
        f"{semantic_frame}:{style_id}:{','.join(event.node_id for event in events)}"
    )
    return PreparedRealizationRequest(
        draft=draft,
        request=request,
        replacements=builder.replacements,
        semantic_frame=semantic_frame,
        realization_signature=hashlib.sha256(signature_material.encode("utf-8")).hexdigest(),
    )


__all__ = ["PreparedRealizationRequest", "prepare_realization_request"]
