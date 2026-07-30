from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from foundry.cycle.contract import (
    CYCLE_ID,
    STAGES,
    CycleContractError,
    content_free_projection,
    load_cycle_config,
    normalized_completion,
    prompt_subseed,
    validate_process_environment,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/cycles/cycle1_verifier_filtered.yaml"


def test_cycle_configuration_freezes_the_authorized_contract() -> None:
    config = load_cycle_config(CONFIG)

    assert config.payload["cycle_id"] == CYCLE_ID
    assert len(STAGES) == 15
    assert config.section("generation")["completions_per_prompt"] == 8
    assert config.section("corpus")["total_assistant_tokens"] == 32_000
    assert config.section("training")["adapted_layers"] == list(range(14, 28))


def test_prompt_subseed_uses_the_frozen_sha256_projection() -> None:
    source_id = "targeted-example-017"
    completion_index = 6
    digest = hashlib.sha256(f"{CYCLE_ID}{source_id}{completion_index}".encode()).digest()
    expected = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    assert prompt_subseed(CYCLE_ID, source_id, completion_index) == expected


def test_completion_normalization_is_stable() -> None:
    assert normalized_completion("  A\u212b\r\n\t B  ") == "AÅ B"


def test_content_free_projection_removes_scientific_content() -> None:
    value = {
        "source_id": "id-1",
        "prompt": "private prompt",
        "nested": [{"completion": "private output", "sha256": "abc"}],
    }

    assert content_free_projection(value) == {
        "source_id": "id-1",
        "nested": [{"sha256": "abc"}],
    }


def test_process_environment_rejects_wrong_hash_seed() -> None:
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }

    with pytest.raises(CycleContractError, match="deterministic process environment differs"):
        validate_process_environment(environment)
