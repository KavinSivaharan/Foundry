"""Strict QLoRA configuration and SFT-text contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

RECIPE_ID = "foundry-qwen2.5-1.5b-signal-qlora-v1"
SFT_SYSTEM_PROMPT = (
    "You solve grade-school arithmetic word problems carefully. Use only the facts in the "
    "problem, show concise reasoning, and follow the required final-answer format."
)
SFT_USER_PREFIX = "Solve this problem.\n\n"
SFT_USER_SUFFIX = (
    "\n\nEnd with exactly one final line in this form:\nFinal answer: <canonical-number>"
)
ASSISTANT_ONLY_V3_FORMAT_ID = "foundry-assistant-only-sft-v3"
CONCISE_ASSISTANT_V4_FORMAT_ID = "foundry-concise-assistant-sft-v4"
ASSISTANT_ONLY_V3_SYSTEM_PROMPT = (
    "You solve grade-school arithmetic word problems carefully. Use the information "
    "in the problem, show concise reasoning, and follow the required final-answer format."
)
ASSISTANT_ONLY_V3_USER_SUFFIX = (
    "\n\nEnd with exactly one final line in this form:\nFinal answer: <integer>"
)
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value with one stable representation."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sft_format_contract_sha256() -> str:
    """Return the immutable hash of the three-message formatting contract."""

    return canonical_sha256(
        {
            "assistant": "training_completion",
            "system": SFT_SYSTEM_PROMPT,
            "user_prefix": SFT_USER_PREFIX,
            "user_suffix": SFT_USER_SUFFIX,
        }
    )


def assistant_only_v3_format_contract_sha256() -> str:
    """Hash the corrected assistant-only formatting and label contract."""

    return canonical_sha256(
        {
            "format_id": ASSISTANT_ONLY_V3_FORMAT_ID,
            "messages": {
                "assistant": "unchanged deterministic trace plus one normalized terminal line",
                "system": ASSISTANT_ONLY_V3_SYSTEM_PROMPT,
                "user_prefix": SFT_USER_PREFIX,
                "user_suffix": ASSISTANT_ONLY_V3_USER_SUFFIX,
            },
            "loss_bearing": ["assistant_completion_content", "final_eos"],
            "masked": [
                "system",
                "user",
                "assistant_header",
                "padding",
                "post_eos_newline",
            ],
        }
    )


def concise_assistant_v4_format_contract_sha256() -> str:
    """Hash the concise equation-grounded assistant-only format contract."""

    return canonical_sha256(
        {
            "format_id": CONCISE_ASSISTANT_V4_FORMAT_ID,
            "messages": {
                "assistant": "one to four equation-grounded lines plus one terminal line",
                "system": ASSISTANT_ONLY_V3_SYSTEM_PROMPT,
                "user_prefix": SFT_USER_PREFIX,
                "user_suffix": ASSISTANT_ONLY_V3_USER_SUFFIX,
            },
            "assistant_target_max_tokens": 128,
            "loss_bearing": ["assistant_completion_content", "final_eos"],
            "masked": [
                "system",
                "user",
                "assistant_header",
                "padding",
                "post_eos_newline",
            ],
            "forbidden": [
                "question_restatement",
                "internal_trace_terminology",
                "code_block",
                "json",
                "duplicate_conclusion",
                "multiple_final_answer_lines",
            ],
        }
    )


def assistant_only_v3_messages(question: str, completion: str) -> list[dict[str, str]]:
    """Build the corrected evaluation-aligned three-message conversation."""

    if not question.strip() or not completion.strip():
        raise ValueError("assistant-only question and completion must be non-empty")
    return [
        {"role": "system", "content": ASSISTANT_ONLY_V3_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{SFT_USER_PREFIX}{question}{ASSISTANT_ONLY_V3_USER_SUFFIX}",
        },
        {"role": "assistant", "content": completion},
    ]


@dataclass(frozen=True)
class QLoRARecipe:
    """Complete, immutable recipe shared by generic and targeted runs."""

    recipe_id: str
    base_model_id: str
    base_revision: str
    tokenizer_sha256: str
    tokenizer_config_sha256: str
    chat_template_sha256: str
    trust_remote_code: bool
    load_in_4bit: bool
    quantization_type: str
    double_quantization: bool
    compute_dtype: str
    rank: int
    alpha: int
    dropout: float
    bias: str
    target_modules: tuple[str, ...]
    max_sequence_length: int
    packing: bool
    micro_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    optimizer: str
    learning_rate: float
    scheduler: str
    warmup_ratio: float
    max_optimizer_steps: int
    weight_decay: float
    max_gradient_norm: float
    gradient_checkpointing: bool
    fp16: bool
    bf16: bool
    seed: int
    logging_steps: int
    evaluation_steps: int
    save_rule: str
    sft_format_sha256: str
    requirements_lock_sha256: str
    dataset_sha256: dict[str, str]
    split_sha256: dict[str, str]
    recipe_sha256: str


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _string_mapping(value: object, name: str) -> dict[str, str]:
    mapping = _mapping(value, name)
    if not all(isinstance(item, str) for item in mapping.values()):
        raise ValueError(f"{name} values must be strings")
    return cast(dict[str, str], mapping)


def load_qlora_recipe(path: Path) -> QLoRARecipe:
    """Load the one approved recipe and reject any hyperparameter drift."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "recipe")
    if root.get("schema_version") != 1 or root.get("recipe_id") != RECIPE_ID:
        raise ValueError("QLoRA recipe identity differs")
    model = _mapping(root.get("model"), "model")
    quant = _mapping(root.get("quantization"), "quantization")
    lora = _mapping(root.get("lora"), "lora")
    training = _mapping(root.get("training"), "training")
    hashes = _mapping(root.get("hashes"), "hashes")
    target_modules = lora.get("target_modules")
    if not isinstance(target_modules, list) or not all(
        isinstance(item, str) for item in target_modules
    ):
        raise ValueError("lora.target_modules must be a string list")

    canonical = dict(root)
    canonical.pop("recipe_sha256", None)
    recipe_sha256 = canonical_sha256(canonical)
    declared_sha = root.get("recipe_sha256")
    if declared_sha not in {None, recipe_sha256}:
        raise ValueError("declared QLoRA recipe hash differs")

    recipe = QLoRARecipe(
        recipe_id=cast(str, root["recipe_id"]),
        base_model_id=cast(str, model["repo_id"]),
        base_revision=cast(str, model["revision"]),
        tokenizer_sha256=cast(str, hashes["tokenizer_sha256"]),
        tokenizer_config_sha256=cast(str, hashes["tokenizer_config_sha256"]),
        chat_template_sha256=cast(str, hashes["chat_template_sha256"]),
        trust_remote_code=cast(bool, model["trust_remote_code"]),
        load_in_4bit=cast(bool, quant["load_in_4bit"]),
        quantization_type=cast(str, quant["quantization_type"]),
        double_quantization=cast(bool, quant["double_quantization"]),
        compute_dtype=cast(str, quant["compute_dtype"]),
        rank=cast(int, lora["rank"]),
        alpha=cast(int, lora["alpha"]),
        dropout=cast(float, lora["dropout"]),
        bias=cast(str, lora["bias"]),
        target_modules=tuple(cast(list[str], target_modules)),
        max_sequence_length=cast(int, training["max_sequence_length"]),
        packing=cast(bool, training["packing"]),
        micro_batch_size=cast(int, training["micro_batch_size"]),
        gradient_accumulation_steps=cast(int, training["gradient_accumulation_steps"]),
        effective_batch_size=cast(int, training["effective_batch_size"]),
        optimizer=cast(str, training["optimizer"]),
        learning_rate=cast(float, training["learning_rate"]),
        scheduler=cast(str, training["scheduler"]),
        warmup_ratio=cast(float, training["warmup_ratio"]),
        max_optimizer_steps=cast(int, training["max_optimizer_steps"]),
        weight_decay=cast(float, training["weight_decay"]),
        max_gradient_norm=cast(float, training["max_gradient_norm"]),
        gradient_checkpointing=cast(bool, training["gradient_checkpointing"]),
        fp16=cast(bool, training["fp16"]),
        bf16=cast(bool, training["bf16"]),
        seed=cast(int, training["seed"]),
        logging_steps=cast(int, training["logging_steps"]),
        evaluation_steps=cast(int, training["evaluation_steps"]),
        save_rule=cast(str, training["save_rule"]),
        sft_format_sha256=cast(str, hashes["sft_format_sha256"]),
        requirements_lock_sha256=cast(str, hashes["requirements_lock_sha256"]),
        dataset_sha256=_string_mapping(hashes["dataset_sha256"], "dataset_sha256"),
        split_sha256=_string_mapping(hashes["split_sha256"], "split_sha256"),
        recipe_sha256=recipe_sha256,
    )
    _validate_frozen_values(recipe)
    return recipe


def _validate_frozen_values(recipe: QLoRARecipe) -> None:
    expected: tuple[bool, ...] = (
        recipe.base_model_id == "Qwen/Qwen2.5-1.5B-Instruct",
        recipe.base_revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        recipe.trust_remote_code is False,
        recipe.load_in_4bit is True,
        recipe.quantization_type == "nf4",
        recipe.double_quantization is True,
        recipe.compute_dtype == "float16",
        recipe.rank == 16,
        recipe.alpha == 32,
        recipe.dropout == 0.05,
        recipe.bias == "none",
        recipe.target_modules == TARGET_MODULES,
        recipe.max_sequence_length == 512,
        recipe.packing is False,
        recipe.micro_batch_size == 1,
        recipe.gradient_accumulation_steps == 8,
        recipe.effective_batch_size == 8,
        recipe.optimizer == "paged_adamw_8bit",
        recipe.learning_rate == 0.0002,
        recipe.scheduler == "cosine",
        recipe.warmup_ratio == 0.05,
        recipe.max_optimizer_steps == 200,
        recipe.weight_decay == 0.0,
        recipe.max_gradient_norm == 1.0,
        recipe.gradient_checkpointing is True,
        recipe.fp16 is True,
        recipe.bf16 is False,
        recipe.seed == 20260720,
        recipe.logging_steps == 5,
        recipe.evaluation_steps == 25,
        recipe.save_rule == "final_adapter_only",
        recipe.sft_format_sha256 == sft_format_contract_sha256(),
    )
    if not all(expected):
        raise ValueError("QLoRA recipe differs from the approved frozen values")


def sft_messages(question: str, completion: str) -> list[dict[str, str]]:
    """Build the exact SFT conversation without exposing benchmark content."""

    if not question.strip() or not completion.strip():
        raise ValueError("SFT question and completion must be non-empty")
    return [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{SFT_USER_PREFIX}{question}{SFT_USER_SUFFIX}"},
        {"role": "assistant", "content": completion},
    ]
