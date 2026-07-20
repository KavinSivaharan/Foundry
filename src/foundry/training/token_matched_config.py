"""Frozen configuration contract for token-matched QLoRA v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from foundry.training.config import QLoRARecipe, canonical_sha256, load_qlora_recipe

TOKEN_MATCHED_RECIPE_ID = "foundry-token-matched-qlora-v2"


@dataclass(frozen=True)
class TokenMatchedArm:
    """One arm's frozen content-bearing schedule reference and totals."""

    schedule_path: Path
    schedule_sha256: str
    occurrences: int
    loss_bearing_tokens: int
    first_four_step_tokens: int


@dataclass(frozen=True)
class TokenMatchedRecipe:
    """Complete token-matching extension over the immutable v1 QLoRA recipe."""

    recipe_id: str
    selected_method: str
    base_recipe_path: Path
    base_recipe_sha256: str
    base_recipe: QLoRARecipe
    nominal_loss_bearing_tokens: int
    nominal_tokens_per_step: int
    optimizer_steps: int
    micro_batch_size: int
    parity_limit: float
    loss_weighting: str
    whole_examples_only: bool
    split_examples: bool
    packed_examples: bool
    arms: dict[str, TokenMatchedArm]
    census_sha256: dict[str, str]
    method_a_summary_sha256: str
    method_b_summary_sha256: str
    recipe_sha256: str


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def load_token_matched_recipe(path: Path) -> TokenMatchedRecipe:
    """Load and reject any drift from the approved Method B contract."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "recipe")
    if root.get("schema_version") != 1:
        raise ValueError("token-matched recipe schema differs")
    canonical = dict(root)
    canonical.pop("recipe_sha256", None)
    recipe_sha256 = canonical_sha256(canonical)
    declared = root.get("recipe_sha256")
    if declared not in {None, recipe_sha256}:
        raise ValueError("declared token-matched recipe hash differs")

    base = _mapping(root.get("base_recipe"), "base_recipe")
    matching = _mapping(root.get("token_matching"), "token_matching")
    arm_values = _mapping(matching.get("arms"), "token_matching.arms")
    census = _mapping(root.get("census_sha256"), "census_sha256")
    evidence = _mapping(root.get("evidence_sha256"), "evidence_sha256")
    base_path = Path(_string(base.get("path"), "base_recipe.path"))
    base_recipe = load_qlora_recipe(base_path)
    arms: dict[str, TokenMatchedArm] = {}
    for arm in ("generic_control", "targeted"):
        item = _mapping(arm_values.get(arm), f"token_matching.arms.{arm}")
        arms[arm] = TokenMatchedArm(
            schedule_path=Path(_string(item.get("schedule_path"), f"{arm}.schedule_path")),
            schedule_sha256=_string(item.get("schedule_sha256"), f"{arm}.schedule_sha256"),
            occurrences=_integer(item.get("occurrences"), f"{arm}.occurrences"),
            loss_bearing_tokens=_integer(
                item.get("loss_bearing_tokens"), f"{arm}.loss_bearing_tokens"
            ),
            first_four_step_tokens=_integer(
                item.get("first_four_step_tokens"), f"{arm}.first_four_step_tokens"
            ),
        )
    result = TokenMatchedRecipe(
        recipe_id=_string(root.get("recipe_id"), "recipe_id"),
        selected_method=_string(root.get("selected_method"), "selected_method"),
        base_recipe_path=base_path,
        base_recipe_sha256=_string(base.get("recipe_sha256"), "base_recipe.recipe_sha256"),
        base_recipe=base_recipe,
        nominal_loss_bearing_tokens=_integer(
            matching.get("nominal_loss_bearing_tokens"),
            "token_matching.nominal_loss_bearing_tokens",
        ),
        nominal_tokens_per_step=_integer(
            matching.get("nominal_tokens_per_step"), "token_matching.nominal_tokens_per_step"
        ),
        optimizer_steps=_integer(matching.get("optimizer_steps"), "token_matching.optimizer_steps"),
        micro_batch_size=_integer(
            matching.get("micro_batch_size"), "token_matching.micro_batch_size"
        ),
        parity_limit=_float(matching.get("parity_limit"), "token_matching.parity_limit"),
        loss_weighting=_string(matching.get("loss_weighting"), "token_matching.loss_weighting"),
        whole_examples_only=matching.get("whole_examples_only") is True,
        split_examples=matching.get("split_examples") is True,
        packed_examples=matching.get("packed_examples") is True,
        arms=arms,
        census_sha256={arm: _string(census.get(arm), f"census_sha256.{arm}") for arm in arms},
        method_a_summary_sha256=_string(
            evidence.get("method_a_summary_sha256"), "evidence_sha256.method_a_summary_sha256"
        ),
        method_b_summary_sha256=_string(
            evidence.get("method_b_summary_sha256"), "evidence_sha256.method_b_summary_sha256"
        ),
        recipe_sha256=recipe_sha256,
    )
    expected = (
        result.recipe_id == TOKEN_MATCHED_RECIPE_ID,
        result.selected_method == "whole_example_token_budgeted_gradient_accumulation",
        result.base_recipe_sha256 == result.base_recipe.recipe_sha256,
        result.nominal_loss_bearing_tokens == 271200,
        result.nominal_tokens_per_step == 1356,
        result.optimizer_steps == 200,
        result.micro_batch_size == 1,
        result.parity_limit == 0.005,
        result.loss_weighting == "microexample_mean_times_loss_tokens_over_step_loss_tokens",
        result.whole_examples_only,
        not result.split_examples,
        not result.packed_examples,
        result.arms["generic_control"].occurrences == 1578,
        result.arms["targeted"].occurrences == 1398,
    )
    if not all(expected):
        raise ValueError("token-matched recipe differs from the frozen Method B values")
    return result
