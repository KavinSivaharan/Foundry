from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from foundry.training.config import (
    TARGET_MODULES,
    load_qlora_recipe,
    sft_format_contract_sha256,
    sft_messages,
)

RECIPE = Path("configs/training/qwen2_5_1_5b_signal_qlora.yaml")


def test_frozen_recipe_loads_and_has_exact_training_contract() -> None:
    recipe = load_qlora_recipe(RECIPE)
    assert recipe.max_optimizer_steps == 200
    assert recipe.effective_batch_size == 8
    assert recipe.target_modules == TARGET_MODULES
    assert recipe.sft_format_sha256 == sft_format_contract_sha256()
    assert len(recipe.recipe_sha256) == 64


def test_recipe_rejects_hyperparameter_drift(tmp_path: Path) -> None:
    raw = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    raw["training"]["learning_rate"] = 0.0003
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="approved frozen values"):
        load_qlora_recipe(changed)


def test_sft_messages_keep_answer_out_of_user_prompt() -> None:
    messages = sft_messages("An original question?", "Reasoning.\nFinal answer: 7")
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert "Final answer: 7" not in messages[1]["content"]
    assert messages[2]["content"].endswith("Final answer: 7")


def test_dependency_lock_hash_is_frozen() -> None:
    recipe = load_qlora_recipe(RECIPE)
    actual = hashlib.sha256(Path("requirements-training.lock.txt").read_bytes()).hexdigest()
    assert actual == recipe.requirements_lock_sha256
