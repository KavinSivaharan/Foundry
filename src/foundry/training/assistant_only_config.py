"""Frozen configuration loader for assistant-only SFT v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from foundry.training.config import (
    QLoRARecipe,
    assistant_only_v3_format_contract_sha256,
    canonical_sha256,
    load_qlora_recipe,
)

ASSISTANT_ONLY_RECIPE_ID = "foundry-assistant-only-sft-v3"


@dataclass(frozen=True)
class AssistantOnlyArm:
    """One arm's census and complete Method B schedule."""

    census_sha256: str
    schedule_path: Path
    schedule_sha256: str
    occurrences: int
    loss_bearing_tokens: int
    first_32_step_tokens: int


@dataclass(frozen=True)
class AssistantOnlyRecipe:
    """Complete frozen assistant-only training extension."""

    recipe_id: str
    base_recipe_path: Path
    base_recipe_sha256: str
    base_recipe: QLoRARecipe
    format_sha256: str
    label_construction: str
    method: str
    nominal_loss_bearing_tokens: int
    optimizer_steps: int
    retention_smoke_steps: int
    parity_limit: float
    arms: dict[str, AssistantOnlyArm]
    primary_learning_rate: float
    fallback_learning_rate: float
    retention_suite_path: Path
    retention_suite_sha256: str
    retention_prompt_sha256: str
    retention_generation_sha256: str
    retention_base_summary_sha256: str
    checkpoints: tuple[int, ...]
    recipe_sha256: str

    def execution_sha256(self, learning_rate: float) -> str:
        """Hash one of the two predeclared learning-rate executions."""

        if learning_rate not in {self.primary_learning_rate, self.fallback_learning_rate}:
            raise ValueError("learning rate is outside the predeclared choices")
        return canonical_sha256(
            {"recipe_sha256": self.recipe_sha256, "learning_rate": learning_rate}
        )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(dict[str, Any], value)


def load_assistant_only_recipe(path: Path) -> AssistantOnlyRecipe:
    """Load and reject any drift from the approved assistant-only contract."""

    value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(value, "recipe")
    if root.get("schema_version") != 1 or root.get("recipe_id") != ASSISTANT_ONLY_RECIPE_ID:
        raise ValueError("assistant-only recipe identity differs")
    canonical = dict(root)
    canonical.pop("recipe_sha256", None)
    recipe_sha256 = canonical_sha256(canonical)
    if root.get("recipe_sha256") not in {None, recipe_sha256}:
        raise ValueError("declared assistant-only recipe hash differs")
    base = _mapping(root.get("base_recipe"), "base_recipe")
    format_config = _mapping(root.get("format"), "format")
    matching = _mapping(root.get("token_matching"), "token_matching")
    arm_values = _mapping(matching.get("arms"), "token_matching.arms")
    rates = _mapping(root.get("learning_rates"), "learning_rates")
    retention = _mapping(root.get("retention"), "retention")
    base_path = Path(str(base["path"]))
    base_recipe = load_qlora_recipe(base_path)
    arms: dict[str, AssistantOnlyArm] = {}
    for arm in ("generic_control", "targeted"):
        item = _mapping(arm_values.get(arm), f"arms.{arm}")
        arms[arm] = AssistantOnlyArm(
            census_sha256=str(item["census_sha256"]),
            schedule_path=Path(str(item["schedule_path"])),
            schedule_sha256=str(item["schedule_sha256"]),
            occurrences=int(item["occurrences"]),
            loss_bearing_tokens=int(item["loss_bearing_tokens"]),
            first_32_step_tokens=int(item["first_32_step_tokens"]),
        )
    checkpoints_value = root.get("checkpoints")
    if not isinstance(checkpoints_value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in checkpoints_value
    ):
        raise ValueError("checkpoints must be an integer list")
    result = AssistantOnlyRecipe(
        recipe_id=str(root["recipe_id"]),
        base_recipe_path=base_path,
        base_recipe_sha256=str(base["recipe_sha256"]),
        base_recipe=base_recipe,
        format_sha256=str(format_config["format_sha256"]),
        label_construction=str(format_config["label_construction"]),
        method=str(matching["method"]),
        nominal_loss_bearing_tokens=int(matching["nominal_loss_bearing_tokens"]),
        optimizer_steps=int(matching["optimizer_steps"]),
        retention_smoke_steps=int(matching["retention_smoke_steps"]),
        parity_limit=float(matching["parity_limit"]),
        arms=arms,
        primary_learning_rate=float(rates["primary"]),
        fallback_learning_rate=float(rates["fallback"]),
        retention_suite_path=Path(str(retention["suite_path"])),
        retention_suite_sha256=str(retention["suite_sha256"]),
        retention_prompt_sha256=str(retention["prompt_sha256"]),
        retention_generation_sha256=str(retention["generation_sha256"]),
        retention_base_summary_sha256=str(retention["base_summary_sha256"]),
        checkpoints=tuple(cast(list[int], checkpoints_value)),
        recipe_sha256=recipe_sha256,
    )
    expected = (
        result.base_recipe_sha256 == result.base_recipe.recipe_sha256,
        result.format_sha256 == assistant_only_v3_format_contract_sha256(),
        result.label_construction == "assistant_completion_content_and_final_eos_only",
        result.method == "assistant_only_whole_example_token_budgeted_gradient_accumulation",
        result.nominal_loss_bearing_tokens == 90_000,
        result.optimizer_steps == 200,
        result.retention_smoke_steps == 32,
        result.parity_limit == 0.005,
        result.primary_learning_rate == 0.0002,
        result.fallback_learning_rate == 0.00005,
        result.checkpoints == (50, 100, 200),
        result.arms["generic_control"].loss_bearing_tokens == 90_000,
        result.arms["targeted"].loss_bearing_tokens == 89_995,
        result.arms["generic_control"].first_32_step_tokens == 14_404,
        result.arms["targeted"].first_32_step_tokens == 14_404,
    )
    if not all(expected):
        raise ValueError("assistant-only recipe differs from the frozen contract")
    return result
