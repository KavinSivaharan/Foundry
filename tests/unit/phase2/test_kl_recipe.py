from __future__ import annotations

import json
from pathlib import Path

from foundry.phase2 import kl_recipe

ROOT = Path(__file__).resolve().parents[3]
RECORD_PATH = ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r3_v1_kl_recipe.json"


def test_runner_reconstructs_exact_v1_lora_call() -> None:
    projection = kl_recipe.runner_lora_projection(
        ROOT / "src" / "foundry" / "phase2" / "vetted_qlora.py"
    )
    assert projection["rank"] == 8
    assert projection["alpha"] == 16
    assert projection["construction_target_module_order"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert projection["lora_scaling"] == 2.0


def test_all_historical_adapter_configs_and_tensors_agree() -> None:
    record = kl_recipe.reconstruct(ROOT)
    assert len(record["historical_artifacts"]) == 6
    assert record["equality_evidence"]["all_six_adapter_config_files_byte_identical"]
    assert record["equality_evidence"]["all_six_tensor_inventories_identical"]
    config = record["canonical_lora_configuration"]
    assert config["trainable_tensor_count"] == 224
    assert config["trainable_parameter_count"] == 2_179_072
    assert config["adapted_layer_count"] == 28
    assert config["adapted_module_count"] == 112


def test_published_recipe_record_reconstructs() -> None:
    expected = kl_recipe.reconstruct(ROOT)
    actual = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    assert actual == expected
    assert actual["decision"] == "historical_v1_rank8_recipe_is_canonical_for_kl"
    assert actual["rank16_draft_rejection"]["classification"] == (
        "conflicting_unexecuted_draft_recipe"
    )
    assert actual["comparator_contract"]["lambda_kl"] == 0
    assert actual["comparator_contract"]["retrained"] is False
    assert actual["holdout_v2_adapter_exposure"] is False
