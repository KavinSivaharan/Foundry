from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from foundry.training.config import canonical_sha256
from foundry.training.grpo_config import (
    BASE_REVISION,
    TARGET_MODULES,
    frozen_config_sha256,
    load_grpo_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FROZEN_CONFIG = PROJECT_ROOT / "configs/training/verifier_grpo_v1.json"


def _raw_config() -> dict[str, Any]:
    value: object = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_rehashed(path: Path, value: dict[str, Any]) -> None:
    payload = {key: item for key, item in value.items() if key != "config_sha256"}
    value["config_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_frozen_config_loads_every_approved_value() -> None:
    config = load_grpo_config(FROZEN_CONFIG)

    assert config.protocol_id == "foundry-verifier-grpo-v1"
    assert config.base_model.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.base_model.revision == BASE_REVISION
    assert config.base_model.trust_remote_code is False
    assert config.quantization == config.quantization.__class__(
        load_in_4bit=True,
        quantization_type="nf4",
        double_quantization=True,
        compute_dtype="float16",
        cpu_offload=False,
    )
    assert config.lora.rank == 16
    assert config.lora.alpha == 32
    assert config.lora.dropout == 0.05
    assert config.lora.bias == "none"
    assert config.lora.target_modules == TARGET_MODULES
    assert config.lora.merge_adapter is False

    grpo = config.grpo
    assert grpo.optimizer_steps == 64
    assert grpo.num_generations == 4
    assert grpo.max_prompt_length == 512
    assert grpo.max_completion_length == 256
    assert grpo.per_device_train_batch_size == 4
    assert grpo.gradient_accumulation_steps == 1
    assert grpo.learning_rate == 0.000001
    assert grpo.optimizer == "paged_adamw_8bit"
    assert grpo.scheduler == "cosine"
    assert grpo.warmup_ratio == 0.05
    assert grpo.weight_decay == 0.0
    assert grpo.max_gradient_norm == 1.0
    assert grpo.epsilon == 0.2
    assert grpo.num_iterations == 1
    assert grpo.scale_rewards is False
    assert grpo.loss_type == "dr_grpo"
    assert grpo.mask_truncated_completions is True
    assert grpo.disable_dropout is True
    assert grpo.temperature == 0.8
    assert grpo.top_p == 0.95
    assert grpo.top_k == 50
    assert grpo.use_vllm is False
    assert grpo.seed == 20260720
    assert grpo.checkpoints == (16, 32, 64)

    assert config.reference_policy.implementation == ("untouched_base_with_active_adapter_disabled")
    assert config.reference_policy.revision == BASE_REVISION
    assert config.reference_policy.trainable is False
    assert config.reference_policy.sync_ref is False
    assert config.reference_policy.cpu_offload is False
    assert config.memory_and_reproducibility.gradient_checkpointing is True
    assert config.memory_and_reproducibility.full_determinism is True
    assert config.memory_and_reproducibility.shuffle is False
    assert [(item.variant_id, item.beta) for item in config.kl_variants] == [
        ("G1", 0.04),
        ("G2", 0.10),
    ]


def test_config_hash_and_variant_execution_hashes_are_stable() -> None:
    config = load_grpo_config(FROZEN_CONFIG)

    assert config.config_sha256 == frozen_config_sha256()
    assert config.execution_sha256("G1") == config.execution_sha256("G1")
    assert config.execution_sha256("G1") != config.execution_sha256("G2")
    with pytest.raises(ValueError, match="outside"):
        config.execution_sha256("G3")


def test_stale_hash_rejects(tmp_path: Path) -> None:
    value = _raw_config()
    value["grpo"]["learning_rate"] = 0.000002
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        load_grpo_config(path)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("base_model", "revision", "f" * 40),
        ("base_model", "trust_remote_code", True),
        ("quantization", "load_in_4bit", False),
        ("quantization", "quantization_type", "fp4"),
        ("quantization", "cpu_offload", True),
        ("lora", "rank", 8),
        ("lora", "dropout", 0.0),
        ("grpo", "optimizer_steps", 65),
        ("grpo", "num_generations", 3),
        ("grpo", "learning_rate", 0.000002),
        ("grpo", "scale_rewards", True),
        ("grpo", "loss_type", "grpo"),
        ("grpo", "use_vllm", True),
        ("reference_policy", "sync_ref", True),
        ("reference_policy", "cpu_offload", True),
        ("memory_and_reproducibility", "shuffle", True),
    ],
)
def test_rehashed_frozen_value_drift_rejects(
    tmp_path: Path, section: str, field: str, replacement: object
) -> None:
    value = _raw_config()
    value[section][field] = replacement
    path = tmp_path / "drift.json"
    _write_rehashed(path, value)

    with pytest.raises(ValueError, match="frozen contract"):
        load_grpo_config(path)


def test_bool_integer_and_integer_float_coercions_reject(tmp_path: Path) -> None:
    for index, (section, field, replacement) in enumerate(
        (
            ("quantization", "load_in_4bit", 1),
            ("grpo", "optimizer_steps", 64.0),
            ("grpo", "weight_decay", 0),
        )
    ):
        value = _raw_config()
        value[section][field] = replacement
        path = tmp_path / f"typed-drift-{index}.json"
        _write_rehashed(path, value)
        with pytest.raises(ValueError, match="frozen contract"):
            load_grpo_config(path)


def test_unknown_and_missing_fields_reject_even_when_rehashed(tmp_path: Path) -> None:
    missing = _raw_config()
    del missing["grpo"]
    missing_path = tmp_path / "missing.json"
    _write_rehashed(missing_path, missing)
    with pytest.raises(ValueError, match="fields differ"):
        load_grpo_config(missing_path)

    unknown = _raw_config()
    unknown["unapproved_search"] = {"beta": 0.2}
    unknown_path = tmp_path / "unknown.json"
    _write_rehashed(unknown_path, unknown)
    with pytest.raises(ValueError, match="fields differ"):
        load_grpo_config(unknown_path)


def test_target_modules_and_kl_variant_order_are_frozen(tmp_path: Path) -> None:
    target_order = copy.deepcopy(_raw_config())
    target_order["lora"]["target_modules"].reverse()
    variant_order = copy.deepcopy(_raw_config())
    variant_order["kl_variants"].reverse()
    extra_variant = copy.deepcopy(_raw_config())
    extra_variant["kl_variants"].append({"variant_id": "G3", "beta": 0.2})

    for index, value in enumerate((target_order, variant_order, extra_variant)):
        path = tmp_path / f"ordered-drift-{index}.json"
        _write_rehashed(path, value)
        with pytest.raises(ValueError, match="frozen contract"):
            load_grpo_config(path)
