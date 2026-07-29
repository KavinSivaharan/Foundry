from __future__ import annotations

from foundry.phase2.l3_grpo_contract import (
    DETERMINISTIC_ENVIRONMENT,
    GRPO_RECIPE,
    GRPO_RECIPE_SHA256,
    LAYERS,
    PROJECTIONS,
    STARTING_ADAPTER_SHA256,
)
from foundry.training.config import canonical_sha256


def test_l3_starting_pair_and_layer_scope_are_exact() -> None:
    assert STARTING_ADAPTER_SHA256 == {
        "generic": "67c6f1dd34c0fa1ddebb354dfe14c43e61c48fdd90c687ba1a9290d2401479cd",
        "targeted": "4e195ff2cb32c4faa6858915b95507862c911bb2eb853b060717416d825df91d",
    }
    assert LAYERS == tuple(range(14, 28))
    assert PROJECTIONS == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_grpo_recipe_freezes_every_authorized_scientific_field() -> None:
    expected = {
        "beta": 0.04,
        "optimizer_steps": 32,
        "groups_per_arm": 32,
        "task_groups_per_arm": 24,
        "replay_groups_per_arm": 8,
        "generations_per_group": 4,
        "completions_per_arm": 128,
        "checkpoints": [8, 16, 32],
        "max_prompt_length": 512,
        "max_completion_length": 256,
        "learning_rate": 0.000001,
        "optimizer": "paged_adamw_8bit",
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "epsilon": 0.2,
        "policy_iterations": 1,
        "reward_scaling": False,
        "loss_form": "dr_grpo",
        "mask_truncated_completions": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "seed": 20260720,
        "use_vllm": False,
        "cpu_offload": False,
        "gradient_checkpointing": False,
    }
    for key, value in expected.items():
        assert GRPO_RECIPE[key] == value
    assert GRPO_RECIPE_SHA256 == canonical_sha256(GRPO_RECIPE)


def test_deterministic_environment_is_exact() -> None:
    assert DETERMINISTIC_ENVIRONMENT == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "20260720",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
