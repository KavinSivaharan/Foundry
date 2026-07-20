from pathlib import Path

import pytest

from foundry.training.assistant_only_config import load_assistant_only_recipe

RECIPE = Path("configs/training/assistant_only_sft_v3.yaml")


def test_assistant_only_recipe_loads_two_predeclared_rates() -> None:
    recipe = load_assistant_only_recipe(RECIPE)
    assert recipe.arms["generic_control"].loss_bearing_tokens == 90_000
    assert recipe.arms["targeted"].loss_bearing_tokens == 89_995
    assert recipe.execution_sha256(0.0002) != recipe.execution_sha256(0.00005)


def test_assistant_only_recipe_rejects_other_rate() -> None:
    recipe = load_assistant_only_recipe(RECIPE)
    with pytest.raises(ValueError, match="outside"):
        recipe.execution_sha256(0.0001)
