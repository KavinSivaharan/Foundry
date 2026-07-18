"""Typed semantic, sentence-plan, and render-evidence representations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum


class CountBehavior(StrEnum):
    """How a lexeme participates in quantity agreement."""

    COUNT = "count"
    MASS = "mass"


class EntityRole(StrEnum):
    """Semantic entity roles available to the realization compiler."""

    ACTOR = "actor"
    ITEM = "item"
    LOCATION = "location"
    CONTAINER = "container"
    RESOURCE = "resource"
    GROUP = "group"


class TargetKind(StrEnum):
    """Approved answer targets with distinct question semantics."""

    COUNT = "count"
    TOTAL_QUANTITY = "total_quantity"
    REMAINING_QUANTITY = "remaining_quantity"
    RATE = "rate"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    WEIGHTED_MEAN = "weighted_mean"
    VALID_ASSIGNMENT_COUNT = "valid_assignment_count"
    CAPACITY = "capacity"
    GROUP_COUNT = "group_count"


class LedgerChangeKind(StrEnum):
    """Direction of one typed inventory transition."""

    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"


class RateRelationKind(StrEnum):
    """Frozen mathematical relation families for rate generation."""

    RATE_TOTAL = "rate_total"
    RATIO_SCALE = "ratio_scale"
    PERCENTAGE = "percentage"
    WEIGHTED_MEAN = "weighted_average"
    COMBINED_RATE = "combined_rate"


class DiscreteRelationKind(StrEnum):
    """Frozen mathematical relation families for discrete generation."""

    TWO_TYPE_ALLOCATION = "two_type_allocation"
    COMPLETE_PACKAGES = "complete_packages"
    EQUAL_DISTRIBUTION = "equal_distribution"
    DUAL_CAPACITY = "dual_capacity"


class Voice(StrEnum):
    """Controlled grammatical voice."""

    ACTIVE = "active"
    PASSIVE = "passive"


@dataclass(frozen=True)
class LexemeSpec:
    """Explicit morphology; no inflection is guessed by the compiler."""

    lexeme_id: str
    singular: str
    plural: str
    attributive: str
    count_behavior: CountBehavior
    supported_prepositions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = (self.lexeme_id, self.singular, self.plural, self.attributive)
        if any(not item.strip() for item in required):
            raise ValueError("lexeme morphology must be explicit and nonempty")
        if self.count_behavior is CountBehavior.COUNT and self.singular == self.plural:
            raise ValueError("count lexemes require distinct singular and plural forms")


@dataclass(frozen=True)
class VerbSpec:
    """Explicit finite and participial verb forms."""

    verb_id: str
    base: str
    third_person_singular: str
    past: str
    past_participle: str

    def __post_init__(self) -> None:
        if any(
            not item.strip()
            for item in (
                self.verb_id,
                self.base,
                self.third_person_singular,
                self.past,
                self.past_participle,
            )
        ):
            raise ValueError("verb morphology must be explicit and nonempty")


@dataclass(frozen=True)
class EntitySpec:
    """One semantic entity with an explicit role and lexeme."""

    entity_id: str
    role: EntityRole
    lexeme: LexemeSpec
    proper_name: str | None = None

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity IDs cannot be empty")
        if self.role is EntityRole.ACTOR and not (self.proper_name or "").strip():
            raise ValueError("actors require an explicit proper name")


@dataclass(frozen=True)
class UnitSpec:
    """Typed quantity unit with an optional explicit rate denominator."""

    unit_id: str
    numerator: LexemeSpec
    denominator: LexemeSpec | None = None


@dataclass(frozen=True)
class QuantitySpec:
    """One integer semantic quantity."""

    node_id: str
    value: int
    entity_id: str
    unit: UnitSpec

    def __post_init__(self) -> None:
        if not self.node_id.strip() or self.value < 0:
            raise ValueError("quantities require a node ID and nonnegative value")


@dataclass(frozen=True)
class LedgerChangeSpec:
    """One exact state transition rendered exactly once."""

    node_id: str
    kind: LedgerChangeKind
    quantity: QuantitySpec
    origin_id: str
    destination_id: str
    verb: VerbSpec


@dataclass(frozen=True)
class WeightedGroupSpec:
    """One unique weighted-mean group."""

    node_id: str
    weight: int
    value: int

    def __post_init__(self) -> None:
        if self.weight <= 0 or self.value < 0:
            raise ValueError("weighted groups require positive weights and nonnegative values")


@dataclass(frozen=True)
class ScalarSpec:
    """One named exact integer used by a relation."""

    node_id: str
    name: str
    value: int
    unit: UnitSpec


@dataclass(frozen=True)
class TargetSpec:
    """The explicit semantic target that the question and answer must share."""

    node_id: str
    kind: TargetKind
    answer_symbol: str
    entity_id: str | None
    unit: UnitSpec


@dataclass(frozen=True)
class DomainSpec:
    """Compositional lexical domain selected independently of mathematics."""

    domain_id: str
    setting: str
    actor: EntitySpec
    item: EntitySpec
    primary_location: EntitySpec
    secondary_location: EntitySpec
    destination_location: EntitySpec
    container: EntitySpec
    safe_context: str


@dataclass(frozen=True)
class BookkeepingProblemIR:
    """Semantic source of truth for a ledger problem."""

    problem_id: str
    domain: DomainSpec
    initial: QuantitySpec
    changes: tuple[LedgerChangeSpec, ...]
    target: TargetSpec
    group_size: int | None
    context_node_id: str | None

    @property
    def required_node_ids(self) -> tuple[str, ...]:
        nodes = (self.initial.node_id, *(change.node_id for change in self.changes))
        return (*nodes, self.target.node_id)


@dataclass(frozen=True)
class RateProblemIR:
    """Semantic source of truth for one frozen rational relation."""

    problem_id: str
    domain: DomainSpec
    relation_kind: RateRelationKind
    scalars: tuple[ScalarSpec, ...]
    groups: tuple[WeightedGroupSpec, ...]
    target: TargetSpec
    context_node_id: str | None

    @property
    def required_node_ids(self) -> tuple[str, ...]:
        nodes = (*(scalar.node_id for scalar in self.scalars), *(g.node_id for g in self.groups))
        return (*nodes, self.target.node_id)


@dataclass(frozen=True)
class DiscreteProblemIR:
    """Semantic source of truth for one frozen bounded-integer relation."""

    problem_id: str
    domain: DomainSpec
    relation_kind: DiscreteRelationKind
    scalars: tuple[ScalarSpec, ...]
    target: TargetSpec
    context_node_id: str | None

    @property
    def required_node_ids(self) -> tuple[str, ...]:
        return (*(scalar.node_id for scalar in self.scalars), self.target.node_id)


type ProblemIR = BookkeepingProblemIR | RateProblemIR | DiscreteProblemIR


@dataclass(frozen=True)
class ClausePlan:
    """Independent syntactic choices for one semantic clause class."""

    intro_style: int
    event_style: int
    question_style: int
    context_position: int
    conjunction_style: int
    numeric_style: int
    voice: Voice


@dataclass(frozen=True)
class RenderSignature:
    """Stable content-free realization choices."""

    compiler_version: str
    problem_family: str
    semantic_frame: str
    domain_id: str
    plan: ClausePlan

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CoverageEntry:
    """Maps one semantic node to exactly one rendered clause."""

    node_id: str
    clause_index: int


@dataclass(frozen=True)
class MorphologyUse:
    """Auditable morphology choice made by the compiler."""

    lexeme: LexemeSpec
    grammatical_role: str
    quantity: int | None
    rendered: str


@dataclass(frozen=True)
class RenderedUnitUse:
    """Surface evidence for one numerator/denominator unit."""

    unit_id: str
    numerator_rendered: str
    denominator_rendered: str | None


@dataclass(frozen=True)
class CompiledRealization:
    """Final prose plus complete typed realization evidence."""

    text: str
    clauses: tuple[str, ...]
    question_clause: str
    signature: RenderSignature
    coverage: tuple[CoverageEntry, ...]
    morphology_uses: tuple[MorphologyUse, ...]
    unit_uses: tuple[RenderedUnitUse, ...]
    rendered_target_kind: TargetKind
    licensed_elisions: tuple[str, ...]
    grammar_complete: bool


def problem_ir_sha256(problem: ProblemIR) -> str:
    """Hash semantic IR independently of rendered wording."""

    payload = json.dumps(asdict(problem), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
