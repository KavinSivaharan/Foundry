"""Controlled-template inventory, diversity, and rendering-quality tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from foundry.synthesis.contamination import numeric_template_sha256
from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.generators.bookkeeping import (
    RENDERING_VARIANTS_PER_FAMILY as BOOKKEEPING_RENDERERS,
)
from foundry.synthesis.generators.bookkeeping import (
    SCENARIO_DOMAIN_COUNT as BOOKKEEPING_SCENARIOS,
)
from foundry.synthesis.generators.bookkeeping import (
    TEMPLATE_FAMILIES as BOOKKEEPING_FAMILIES,
)
from foundry.synthesis.generators.bookkeeping import generate_bookkeeping
from foundry.synthesis.generators.discrete import (
    RENDERING_VARIANTS_PER_FAMILY as DISCRETE_RENDERERS,
)
from foundry.synthesis.generators.discrete import (
    SCENARIO_DOMAIN_COUNT as DISCRETE_SCENARIOS,
)
from foundry.synthesis.generators.discrete import TEMPLATE_FAMILIES as DISCRETE_FAMILIES
from foundry.synthesis.generators.discrete import generate_discrete
from foundry.synthesis.generators.rates import (
    RENDERING_VARIANTS_PER_FAMILY as RATE_RENDERERS,
)
from foundry.synthesis.generators.rates import SCENARIO_DOMAIN_COUNT as RATE_SCENARIOS
from foundry.synthesis.generators.rates import TEMPLATE_FAMILIES as RATE_FAMILIES
from foundry.synthesis.generators.rates import generate_rates
from foundry.synthesis.quality import validate_rendered_candidate
from foundry.synthesis.schema import DifficultyLevel

Generator = Callable[..., CandidateDraft]


@pytest.mark.parametrize(
    ("families", "renderers", "scenarios", "expected_families", "expected_renderers"),
    (
        (BOOKKEEPING_FAMILIES, BOOKKEEPING_RENDERERS, BOOKKEEPING_SCENARIOS, 2, 8),
        (RATE_FAMILIES, RATE_RENDERERS, RATE_SCENARIOS, 5, 6),
        (DISCRETE_FAMILIES, DISCRETE_RENDERERS, DISCRETE_SCENARIOS, 4, 6),
    ),
)
def test_template_inventory_is_explicit_and_broad(
    families: tuple[str, ...],
    renderers: int,
    scenarios: int,
    expected_families: int,
    expected_renderers: int,
) -> None:
    assert len(families) == expected_families
    assert len(set(families)) == expected_families
    assert renderers == expected_renderers
    assert scenarios >= 20


@pytest.mark.parametrize(
    ("generator", "attempts"),
    ((generate_bookkeeping, 53), (generate_rates, 34), (generate_discrete, 33)),
)
def test_fresh_smoke_scale_has_distinct_number_neutral_renderings(
    generator: Generator, attempts: int
) -> None:
    hashes: list[str] = []
    for variant in range(attempts):
        draft = generator(
            seed=10_000 + variant,
            difficulty=tuple(DifficultyLevel)[variant % 3],
            variant=variant,
            output_contract_enabled=variant % 5 == 0,
        )
        hashes.append(numeric_template_sha256(draft.rendered_question))
    assert len(set(hashes)) == attempts


@pytest.mark.parametrize(
    ("generator", "attempts"),
    ((generate_bookkeeping, 53), (generate_rates, 34), (generate_discrete, 33)),
)
def test_every_fresh_smoke_scale_rendering_passes_rule_based_quality(
    generator: Generator, attempts: int
) -> None:
    for variant in range(attempts):
        draft = generator(
            seed=20_000 + variant,
            difficulty=tuple(DifficultyLevel)[variant % 3],
            variant=variant,
            output_contract_enabled=variant % 5 == 0,
        )
        assert (
            validate_rendered_candidate(
                question=draft.rendered_question,
                completion=draft.training_completion,
                answer=draft.canonical_final_answer,
                output_contract_enabled=draft.output_contract_enabled,
                metadata=draft.quality_metadata,
            )
            == ()
        )
