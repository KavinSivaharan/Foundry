"""Adapters that turn approved generator vocabularies into typed lexical domains."""

from __future__ import annotations

from foundry.synthesis.realization.ir import DomainSpec, EntityRole, EntitySpec
from foundry.synthesis.realization.morphology import count_lexeme


def make_domain(
    *,
    domain_id: str,
    setting: str,
    actor: str,
    item_id: str,
    item_singular: str,
    item_plural: str,
    primary_location: str,
    secondary_location: str,
    destination_location: str,
    container_singular: str,
    container_plural: str,
    safe_context: str,
) -> DomainSpec:
    """Build explicit morphology for every lexical slot used by the compiler."""

    item = EntitySpec(
        "item",
        EntityRole.ITEM,
        count_lexeme(item_id, item_singular, item_plural, attributive=item_singular),
    )
    return DomainSpec(
        domain_id=domain_id,
        setting=setting,
        actor=EntitySpec(
            "actor",
            EntityRole.ACTOR,
            count_lexeme("person", "person", "people"),
            proper_name=actor,
        ),
        item=item,
        primary_location=EntitySpec(
            "primary_location",
            EntityRole.LOCATION,
            count_lexeme("primary_location", primary_location, f"{primary_location} locations"),
        ),
        secondary_location=EntitySpec(
            "secondary_location",
            EntityRole.LOCATION,
            count_lexeme(
                "secondary_location", secondary_location, f"{secondary_location} locations"
            ),
        ),
        destination_location=EntitySpec(
            "destination_location",
            EntityRole.LOCATION,
            count_lexeme(
                "destination_location",
                destination_location,
                f"{destination_location} locations",
            ),
        ),
        container=EntitySpec(
            "container",
            EntityRole.CONTAINER,
            count_lexeme("container", container_singular, container_plural),
        ),
        safe_context=safe_context,
    )
