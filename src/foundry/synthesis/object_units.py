"""Typed objects, units, locations, and ledger-operation safety."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Countability(StrEnum):
    """Whether a generated quantity is counted or measured."""

    DISCRETE = "discrete"
    MEASURED = "measured"


class QuantityUnit(StrEnum):
    """Canonical unit classes used by procedural generators."""

    ITEM = "item"
    PART = "part"
    PACKAGE = "package"
    CONTAINER = "container"
    ASSEMBLY = "assembly"
    INTERVAL = "interval"
    PERCENT = "percent"
    MARK = "mark"


class LedgerOperationKind(StrEnum):
    """Supported typed inventory transitions."""

    ADD = "add"
    REMOVE = "remove"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    GROUP = "group"


@dataclass(frozen=True)
class ObjectKind:
    """One renderable object family with compatibility and grammar metadata."""

    family: str
    singular: str
    plural: str
    countability: Countability
    unit: QuantityUnit
    combination_key: str
    transferable: bool
    supported_verbs: tuple[str, ...]
    supported_containers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (self.family, self.singular, self.plural, self.combination_key)
        ):
            raise ValueError("object metadata cannot contain empty identifiers")
        if self.singular == self.plural:
            raise ValueError("object singular and plural forms must differ")
        if not self.supported_verbs or not self.supported_containers:
            raise ValueError("object kinds require verbs and compatible containers")

    def render_noun(self, quantity: int) -> str:
        """Render the grammatically correct noun form for an integer quantity."""

        return self.singular if quantity == 1 else self.plural

    def compatible_with(self, other: ObjectKind) -> bool:
        """Return whether two quantities may enter one arithmetic ledger."""

        return (
            self.family == other.family
            and self.countability is other.countability
            and self.unit is other.unit
            and self.combination_key == other.combination_key
        )


@dataclass(frozen=True)
class LocationSpec:
    """A named collection endpoint that accepts explicit object families."""

    location_id: str
    rendered_name: str
    container_kind: str
    accepted_families: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.location_id.strip() or not self.rendered_name.strip():
            raise ValueError("locations require stable IDs and rendered names")
        if not self.accepted_families:
            raise ValueError("locations must declare accepted object families")

    def accepts(self, object_kind: ObjectKind) -> bool:
        return (
            object_kind.family in self.accepted_families
            and self.container_kind in object_kind.supported_containers
        )


@dataclass(frozen=True)
class TypedQuantity:
    """One exact integer quantity associated with a typed object and location."""

    symbol: str
    amount: int
    object_kind: ObjectKind
    location_id: str

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.location_id.strip():
            raise ValueError("typed quantities require symbols and locations")
        if self.amount < 0:
            raise ValueError("typed quantities cannot be negative")


@dataclass(frozen=True)
class TypedLedgerOperation:
    """One typed state transition before natural-language rendering."""

    kind: LedgerOperationKind
    quantity_symbol: str
    verb: str
    origin_id: str | None
    destination_id: str | None


def validate_combination(kinds: tuple[ObjectKind, ...]) -> tuple[str, ...]:
    """Reject quantities that cannot share one ledger total."""

    if not kinds:
        return ("empty_quantity_combination",)
    anchor = kinds[0]
    if any(not anchor.compatible_with(kind) for kind in kinds[1:]):
        return ("incompatible_object_or_unit_combination",)
    return ()


def validate_transfer(
    *,
    object_kind: ObjectKind,
    origin: LocationSpec,
    destination: LocationSpec,
    verb: str,
) -> tuple[str, ...]:
    """Validate a transfer between two compatible and distinct endpoints."""

    reasons: list[str] = []
    if not object_kind.transferable:
        reasons.append("object_is_not_transferable")
    if origin.location_id == destination.location_id:
        reasons.append("transfer_endpoints_are_identical")
    if not origin.accepts(object_kind) or not destination.accepts(object_kind):
        reasons.append("transfer_location_is_incompatible")
    if verb not in object_kind.supported_verbs:
        reasons.append("unsupported_object_verb")
    return tuple(reasons)


def validate_ledger_plan(
    *,
    ledger_kind: ObjectKind,
    quantities: tuple[TypedQuantity, ...],
    operations: tuple[TypedLedgerOperation, ...],
    locations: tuple[LocationSpec, ...],
) -> tuple[str, ...]:
    """Validate type compatibility, verbs, and transfer endpoints before rendering."""

    reasons = list(validate_combination((ledger_kind, *(item.object_kind for item in quantities))))
    quantity_by_symbol = {item.symbol: item for item in quantities}
    location_by_id = {location.location_id: location for location in locations}
    if len(quantity_by_symbol) != len(quantities):
        reasons.append("duplicate_quantity_symbol")
    if len(location_by_id) != len(locations):
        reasons.append("duplicate_location_id")
    for operation in operations:
        quantity = quantity_by_symbol.get(operation.quantity_symbol)
        if quantity is None:
            reasons.append("operation_references_missing_quantity")
            continue
        if operation.verb not in quantity.object_kind.supported_verbs:
            reasons.append("unsupported_object_verb")
        if operation.kind in {
            LedgerOperationKind.TRANSFER_IN,
            LedgerOperationKind.TRANSFER_OUT,
        }:
            origin = location_by_id.get(operation.origin_id or "")
            destination = location_by_id.get(operation.destination_id or "")
            if origin is None or destination is None:
                reasons.append("transfer_endpoint_is_missing")
            else:
                reasons.extend(
                    validate_transfer(
                        object_kind=quantity.object_kind,
                        origin=origin,
                        destination=destination,
                        verb=operation.verb,
                    )
                )
    return tuple(dict.fromkeys(reasons))
