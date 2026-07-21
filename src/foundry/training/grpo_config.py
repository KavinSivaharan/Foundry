"""Strict loader for the frozen verifier-reward GRPO configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256

PROTOCOL_ID = "foundry-verifier-grpo-v1"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_FROZEN_PAYLOAD: dict[str, object] = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "base_model": {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "trust_remote_code": False,
    },
    "quantization": {
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "float16",
        "cpu_offload": False,
    },
    "lora": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "bias": "none",
        "target_modules": list(TARGET_MODULES),
        "merge_adapter": False,
    },
    "grpo": {
        "optimizer_steps": 64,
        "num_generations": 4,
        "max_prompt_length": 512,
        "max_completion_length": 256,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.000001,
        "optimizer": "paged_adamw_8bit",
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
        "epsilon": 0.2,
        "num_iterations": 1,
        "scale_rewards": False,
        "loss_type": "dr_grpo",
        "mask_truncated_completions": True,
        "disable_dropout": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "use_vllm": False,
        "seed": 20260720,
        "checkpoints": [16, 32, 64],
    },
    "reference_policy": {
        "implementation": "untouched_base_with_active_adapter_disabled",
        "revision": BASE_REVISION,
        "trainable": False,
        "sync_ref": False,
        "cpu_offload": False,
    },
    "memory_and_reproducibility": {
        "gradient_checkpointing": True,
        "full_determinism": True,
        "shuffle": False,
    },
    "kl_variants": [
        {"variant_id": "G1", "beta": 0.04},
        {"variant_id": "G2", "beta": 0.10},
    ],
}


@dataclass(frozen=True)
class BaseModelConfig:
    model_id: str
    revision: str
    trust_remote_code: bool


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool
    quantization_type: str
    double_quantization: bool
    compute_dtype: str
    cpu_offload: bool


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    bias: str
    target_modules: tuple[str, ...]
    merge_adapter: bool


@dataclass(frozen=True)
class GRPOHyperparameters:
    optimizer_steps: int
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    optimizer: str
    scheduler: str
    warmup_ratio: float
    weight_decay: float
    max_gradient_norm: float
    epsilon: float
    num_iterations: int
    scale_rewards: bool
    loss_type: str
    mask_truncated_completions: bool
    disable_dropout: bool
    temperature: float
    top_p: float
    top_k: int
    use_vllm: bool
    seed: int
    checkpoints: tuple[int, ...]


@dataclass(frozen=True)
class ReferencePolicyConfig:
    implementation: str
    revision: str
    trainable: bool
    sync_ref: bool
    cpu_offload: bool


@dataclass(frozen=True)
class MemoryAndReproducibilityConfig:
    gradient_checkpointing: bool
    full_determinism: bool
    shuffle: bool


@dataclass(frozen=True)
class KLVariant:
    variant_id: str
    beta: float


@dataclass(frozen=True)
class VerifierGRPOConfig:
    protocol_id: str
    base_model: BaseModelConfig
    quantization: QuantizationConfig
    lora: LoRAConfig
    grpo: GRPOHyperparameters
    reference_policy: ReferencePolicyConfig
    memory_and_reproducibility: MemoryAndReproducibilityConfig
    kl_variants: tuple[KLVariant, ...]
    config_sha256: str

    def variant(self, variant_id: str) -> KLVariant:
        """Return one of the only two predeclared KL variants."""

        matches = [item for item in self.kl_variants if item.variant_id == variant_id]
        if len(matches) != 1:
            raise ValueError("variant is outside the frozen G1/G2 choices")
        return matches[0]

    def execution_sha256(self, variant_id: str) -> str:
        """Bind one training execution to this config and one frozen beta."""

        variant = self.variant(variant_id)
        return canonical_sha256(
            {
                "config_sha256": self.config_sha256,
                "variant_id": variant.variant_id,
                "beta": variant.beta,
            }
        )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(dict[str, Any], value)


def _same_typed_value(actual: object, expected: object) -> bool:
    """Compare values without allowing JSON bool/int or int/float coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return False
        return all(_same_typed_value(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _same_typed_value(left, right) for left, right in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def frozen_config_sha256() -> str:
    """Return the canonical identity of the approved config payload."""

    return canonical_sha256(_FROZEN_PAYLOAD)


def load_grpo_config(path: Path) -> VerifierGRPOConfig:
    """Load the exact frozen GRPO recipe and reject every schema or value drift."""

    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load GRPO configuration: {error}") from error
    root = _mapping(value, "GRPO configuration")
    expected_root_keys = {*_FROZEN_PAYLOAD, "config_sha256"}
    if set(root) != expected_root_keys:
        raise ValueError("GRPO configuration fields differ from the frozen schema")
    declared_hash = root.get("config_sha256")
    payload = {key: item for key, item in root.items() if key != "config_sha256"}
    computed_hash = canonical_sha256(payload)
    if not isinstance(declared_hash, str) or declared_hash != computed_hash:
        raise ValueError("GRPO configuration hash differs")
    if not _same_typed_value(payload, _FROZEN_PAYLOAD):
        raise ValueError("GRPO configuration differs from the frozen contract")

    base = _mapping(payload["base_model"], "base_model")
    quantization = _mapping(payload["quantization"], "quantization")
    lora = _mapping(payload["lora"], "lora")
    grpo = _mapping(payload["grpo"], "grpo")
    reference = _mapping(payload["reference_policy"], "reference_policy")
    reproducibility = _mapping(payload["memory_and_reproducibility"], "memory_and_reproducibility")
    raw_variants = payload["kl_variants"]
    if not isinstance(raw_variants, list):
        raise ValueError("kl_variants must be a list")
    variants = tuple(
        KLVariant(
            variant_id=str(_mapping(item, "kl_variant")["variant_id"]),
            beta=float(_mapping(item, "kl_variant")["beta"]),
        )
        for item in raw_variants
    )
    return VerifierGRPOConfig(
        protocol_id=str(payload["protocol_id"]),
        base_model=BaseModelConfig(
            model_id=str(base["model_id"]),
            revision=str(base["revision"]),
            trust_remote_code=cast(bool, base["trust_remote_code"]),
        ),
        quantization=QuantizationConfig(
            load_in_4bit=cast(bool, quantization["load_in_4bit"]),
            quantization_type=str(quantization["quantization_type"]),
            double_quantization=cast(bool, quantization["double_quantization"]),
            compute_dtype=str(quantization["compute_dtype"]),
            cpu_offload=cast(bool, quantization["cpu_offload"]),
        ),
        lora=LoRAConfig(
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            target_modules=tuple(cast(list[str], lora["target_modules"])),
            merge_adapter=cast(bool, lora["merge_adapter"]),
        ),
        grpo=GRPOHyperparameters(
            optimizer_steps=int(grpo["optimizer_steps"]),
            num_generations=int(grpo["num_generations"]),
            max_prompt_length=int(grpo["max_prompt_length"]),
            max_completion_length=int(grpo["max_completion_length"]),
            per_device_train_batch_size=int(grpo["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(grpo["gradient_accumulation_steps"]),
            learning_rate=float(grpo["learning_rate"]),
            optimizer=str(grpo["optimizer"]),
            scheduler=str(grpo["scheduler"]),
            warmup_ratio=float(grpo["warmup_ratio"]),
            weight_decay=float(grpo["weight_decay"]),
            max_gradient_norm=float(grpo["max_gradient_norm"]),
            epsilon=float(grpo["epsilon"]),
            num_iterations=int(grpo["num_iterations"]),
            scale_rewards=cast(bool, grpo["scale_rewards"]),
            loss_type=str(grpo["loss_type"]),
            mask_truncated_completions=cast(bool, grpo["mask_truncated_completions"]),
            disable_dropout=cast(bool, grpo["disable_dropout"]),
            temperature=float(grpo["temperature"]),
            top_p=float(grpo["top_p"]),
            top_k=int(grpo["top_k"]),
            use_vllm=cast(bool, grpo["use_vllm"]),
            seed=int(grpo["seed"]),
            checkpoints=tuple(cast(list[int], grpo["checkpoints"])),
        ),
        reference_policy=ReferencePolicyConfig(
            implementation=str(reference["implementation"]),
            revision=str(reference["revision"]),
            trainable=cast(bool, reference["trainable"]),
            sync_ref=cast(bool, reference["sync_ref"]),
            cpu_offload=cast(bool, reference["cpu_offload"]),
        ),
        memory_and_reproducibility=MemoryAndReproducibilityConfig(
            gradient_checkpointing=cast(bool, reproducibility["gradient_checkpointing"]),
            full_determinism=cast(bool, reproducibility["full_determinism"]),
            shuffle=cast(bool, reproducibility["shuffle"]),
        ),
        kl_variants=variants,
        config_sha256=declared_hash,
    )
