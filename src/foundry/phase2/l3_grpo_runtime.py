"""CUDA-only Milestone 14A dual-adapter verifier-GRPO runtime.

Heavy training dependencies are imported only after the frozen Phase 2 launch
contract has validated the interpreter and process environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import random
import time
import weakref
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal, cast

from foundry.phase2 import vetted_qlora_kl
from foundry.phase2.l3_grpo_contract import (
    DETERMINISTIC_ENVIRONMENT,
    FIXED_LIBRARY_NOTICE_CONTRACT_SHA256,
    GRPO_RECIPE,
    GRPO_RECIPE_SHA256,
    INTERPRETER_SHA256,
    MODEL_REVISION,
    STARTING_ADAPTER_SHA256,
)
from foundry.phase2.l3_grpo_reference import (
    EXPECTED_ADAPTER_TENSORS,
    POLICY_ADAPTER_NAME,
    REFERENCE_ADAPTER_NAME,
    SharedStartingPolicyReference,
    active_adapter_name,
    assert_policy_reference_identity,
    capture_adapter_state,
    reference_mechanism_contract,
    set_policy_active,
    set_reference_active_frozen,
)
from foundry.phase2.l3_grpo_reward import (
    ReplayRewardMetadata,
    RewardBreakdown,
    TaskRewardMetadata,
    reward_configuration_sha256,
    reward_contract_sha256,
    reward_implementation_sha256,
    score_reward,
)
from foundry.phase2.l3_grpo_schedule import (
    CHECKPOINT_STEPS,
    COMPLETIONS_PER_ARM,
    COMPLETIONS_PER_GROUP,
    GROUPS_PER_ARM,
    OPTIMIZER_STEPS,
    REPLAY_GROUPS_PER_ARM,
    REPLAY_POSITIONS,
    REPLAY_SECTION_QUOTAS,
    SCHEDULE_ID,
    SCHEMA_VERSION,
    SEED,
    TASK_GROUPS_PER_ARM,
    TASK_QUOTAS,
    Arm,
    PromptMessage,
)
from foundry.phase2.launch_contract import validate_postimport, validate_preimport
from foundry.training.config import canonical_sha256
from foundry.training.grpo_compatibility import (
    TopPWarningOnlyGenerationContract,
    model_adapter_state,
)
from foundry.training.grpo_replay_evidence import (
    CompatibilityStepEvidence,
    GenerationEvidence,
    build_compatibility_step_evidence,
    capture_base_parameter_state,
    capture_generation_evidence,
    capture_gradient_state,
    capture_lora_state,
    capture_optimizer_state,
    capture_rng_state,
    capture_scheduler_state,
    tensor_evidence,
)
from foundry.training.grpo_runtime import (
    _peak_process_ram,
    assert_cuda_only_model,
    assert_dropout_disabled,
)
from foundry.training.grpo_trainer import (
    get_active_truncation_flags,
    make_truncation_aware_grpo_trainer,
)
from foundry.training.qlora import directory_sha256, file_sha256
from foundry.training.retention import RetentionItem

RuntimeMode = Literal["compatibility", "train"]
SourceKind = Literal["task", "base_replay"]
RUNTIME_ID = "foundry-l3-verifier-grpo-runtime-v1"
MAX_RESERVED_VRAM_BYTES = 10_240 * 1024**2
COMPATIBILITY_STEPS = 2
COMPATIBILITY_COMPLETIONS = 8
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_TASK_METADATA_FIELDS = frozenset(
    {
        "reward_kind",
        "canonical_answer",
        "answer_type",
        "family",
        "difficulty",
        "verifier_metadata_sha256",
        "provenance_sha256",
    }
)
_REPLAY_METADATA_FIELDS = frozenset(
    {
        "reward_kind",
        "section",
        "skill",
        "kind",
        "expected",
        "scorer_sha256",
        "provenance_sha256",
    }
)


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed object")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _read(path: Path) -> dict[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class RuntimeGroup:
    """One prompt-bearing group bound to a content-free manifest row."""

    group_id: str
    arm: Arm
    position: int
    source_kind: SourceKind
    source_id: str
    category: str
    messages: tuple[PromptMessage, ...]
    prompt_sha256: str
    prompt_tokens: int
    reward_metadata_json: str

    def policy_row(self) -> dict[str, object]:
        return {
            "prompt": [message.as_dict() for message in self.messages],
            "group_id": self.group_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "prompt_sha256": self.prompt_sha256,
            "reward_metadata_json": self.reward_metadata_json,
        }


@dataclass(frozen=True)
class RuntimeSchedule:
    arm: Arm
    groups: tuple[RuntimeGroup, ...]
    packet_sha256: str
    manifest_sha256: str


def _messages(value: object, group_id: str) -> tuple[PromptMessage, ...]:
    rows = _array(value, f"{group_id}.messages")
    if len(rows) != 2:
        raise ValueError("each prompt must contain exactly system and user messages")
    result: list[PromptMessage] = []
    for index, item in enumerate(rows):
        row = _object(item, f"{group_id}.messages[{index}]")
        role = "system" if index == 0 else "user"
        if set(row) != {"role", "content"} or row.get("role") != role:
            raise ValueError("prompt messages must be ordered system then user")
        result.append(
            PromptMessage(
                cast(Literal["system", "user"], role),
                _require_text(row.get("content"), "message.content"),
            )
        )
    return tuple(result)


def _metadata(value: object, source_kind: SourceKind, group_id: str) -> dict[str, object]:
    result = _object(value, f"{group_id}.reward_metadata")
    expected = _TASK_METADATA_FIELDS if source_kind == "task" else _REPLAY_METADATA_FIELDS
    if set(result) != expected:
        raise ValueError("trusted reward metadata schema differs")
    expected_kind = "task" if source_kind == "task" else "base_replay"
    if result.get("reward_kind") != expected_kind:
        raise ValueError("reward metadata kind differs")
    for key, item in result.items():
        if key != "reward_kind":
            _require_text(item, f"reward_metadata.{key}")
    for key in ("verifier_metadata_sha256", "scorer_sha256", "provenance_sha256"):
        if key in result:
            _require_sha256(result[key], f"reward_metadata.{key}")
    return result


def load_schedule(packet_path: Path, manifest_path: Path, arm: Arm) -> RuntimeSchedule:
    """Load one exact 32-group prompt packet and its content-free manifest."""

    packet = _read(packet_path)
    manifest = _read(manifest_path)
    packet_sha256 = canonical_sha256(packet)
    declared_manifest = manifest.get("manifest_sha256")
    if declared_manifest != canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("schedule manifest self-hash differs")
    if (
        packet.get("schema_version") != SCHEMA_VERSION
        or manifest.get("schema_version") != SCHEMA_VERSION
        or packet.get("schedule_id") != SCHEDULE_ID
        or manifest.get("schedule_id") != SCHEDULE_ID
        or packet.get("seed") != SEED
        or manifest.get("seed") != SEED
        or packet.get("arm") != arm
        or manifest.get("arm") != arm
        or manifest.get("prompt_packet_sha256") != packet_sha256
        or manifest.get("groups_per_arm") != GROUPS_PER_ARM
        or manifest.get("task_groups") != TASK_GROUPS_PER_ARM
        or manifest.get("replay_groups") != REPLAY_GROUPS_PER_ARM
        or manifest.get("optimizer_steps") != OPTIMIZER_STEPS
        or manifest.get("completions_per_group") != COMPLETIONS_PER_GROUP
        or manifest.get("total_completions") != COMPLETIONS_PER_ARM
        or manifest.get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or manifest.get("gsm1k_prompt_use") != 0
        or manifest.get("holdout_v2_prompt_use") != 0
        or manifest.get("sealed_content_use") != 0
        or manifest.get("human_written_task_source") is not True
        or manifest.get("groups_fixed_before_generation") is not True
        or manifest.get("model_outputs_observed_during_scheduling") is not False
        or manifest.get("prompts_or_answers_in_manifest") is not False
    ):
        raise ValueError("schedule packet or manifest identity differs")
    packet_groups = _array(packet.get("groups"), "packet.groups")
    manifest_groups = _array(manifest.get("groups"), "manifest.groups")
    if len(packet_groups) != GROUPS_PER_ARM or len(manifest_groups) != GROUPS_PER_ARM:
        raise ValueError("schedule must contain exactly 32 groups")
    groups: list[RuntimeGroup] = []
    ids: set[str] = set()
    source_ids: set[str] = set()
    for position, (packet_item, manifest_item) in enumerate(
        zip(packet_groups, manifest_groups, strict=True), start=1
    ):
        row = _object(packet_item, f"packet.groups[{position - 1}]")
        content_free = {
            key: value for key, value in row.items() if key not in {"messages", "reward_metadata"}
        }
        if content_free != _object(manifest_item, f"manifest.groups[{position - 1}]"):
            raise ValueError("prompt group differs from content-free manifest")
        source_kind_value = row.get("source_kind")
        if source_kind_value not in {"task", "base_replay"}:
            raise ValueError("group source kind differs")
        source_kind = cast(SourceKind, source_kind_value)
        group_id = _require_text(row.get("group_id"), "group_id")
        source_id = _require_text(row.get("source_id"), "source_id")
        prompt_tokens = row.get("prompt_tokens")
        if (
            row.get("position") != position
            or row.get("completions_per_group") != COMPLETIONS_PER_GROUP
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
            or prompt_tokens > 512
        ):
            raise ValueError("group position, completions, or prompt tokens differ")
        messages = _messages(row.get("messages"), group_id)
        prompt_sha256 = _require_sha256(row.get("prompt_sha256"), "prompt_sha256")
        if canonical_sha256([message.as_dict() for message in messages]) != prompt_sha256:
            raise ValueError("model-visible prompt differs from its hash")
        trusted = _metadata(row.get("reward_metadata"), source_kind, group_id)
        if group_id in ids or source_id in source_ids:
            raise ValueError("group and source IDs must be unique")
        ids.add(group_id)
        source_ids.add(source_id)
        groups.append(
            RuntimeGroup(
                group_id=group_id,
                arm=arm,
                position=position,
                source_kind=source_kind,
                source_id=source_id,
                category=_require_text(row.get("category"), "category"),
                messages=messages,
                prompt_sha256=prompt_sha256,
                prompt_tokens=prompt_tokens,
                reward_metadata_json=json.dumps(
                    trusted, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ),
            )
        )
    task = [group for group in groups if group.source_kind == "task"]
    replay = [group for group in groups if group.source_kind == "base_replay"]
    if len(task) != TASK_GROUPS_PER_ARM or len(replay) != REPLAY_GROUPS_PER_ARM:
        raise ValueError("schedule task/replay composition differs")
    if [group.position for group in replay] != list(REPLAY_POSITIONS):
        raise ValueError("shared replay positions differ")
    if Counter(group.category for group in task) != Counter(TASK_QUOTAS[arm]):
        raise ValueError("task-family quotas differ")
    if Counter(group.category for group in replay) != Counter(REPLAY_SECTION_QUOTAS):
        raise ValueError("replay-section quotas differ")
    return RuntimeSchedule(
        arm=arm,
        groups=tuple(groups),
        packet_sha256=packet_sha256,
        manifest_sha256=_require_sha256(declared_manifest, "manifest_sha256"),
    )


def _prompt_rows(value: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError("prompt must be a message sequence")
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise ValueError("prompt message structure differs")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError("prompt contains an assistant target or malformed content")
        rows.append({"role": role, "content": content})
    if [row["role"] for row in rows] != ["system", "user"]:
        raise ValueError("prompt must remain system then user")
    return tuple(rows)


def _completion_text(value: object) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("completion must be non-empty")
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        rows = list(value)
        if len(rows) == 1 and isinstance(rows[0], Mapping):
            row = rows[0]
            if row.get("role") == "assistant" and isinstance(row.get("content"), str):
                content = cast(str, row["content"])
                if content.strip():
                    return content
    raise ValueError("completion structure differs from stock TRL")


def _reward_metadata(
    group: RuntimeGroup, prompt: object, metadata_json: str
) -> TaskRewardMetadata | ReplayRewardMetadata:
    value = _metadata(json.loads(metadata_json), group.source_kind, group.group_id)
    prompt_text = _prompt_rows(prompt)[-1]["content"]
    if group.source_kind == "task":
        return TaskRewardMetadata(
            source_id=group.source_id,
            prompt=prompt_text,
            canonical_answer=_require_text(value.get("canonical_answer"), "canonical_answer"),
            answer_type=_require_text(value.get("answer_type"), "answer_type"),
            family=_require_text(value.get("family"), "family"),
            difficulty=_require_text(value.get("difficulty"), "difficulty"),
            verifier_metadata_sha256=_require_sha256(
                value.get("verifier_metadata_sha256"), "verifier_metadata_sha256"
            ),
            provenance_sha256=_require_sha256(value.get("provenance_sha256"), "provenance_sha256"),
        )
    section = _require_text(value.get("section"), "section")
    kind = _require_text(value.get("kind"), "kind")
    if section not in {"arithmetic", "format", "instruction"} or kind not in {
        "numeric_terminal",
        "exact_text",
        "json_exact",
    }:
        raise ValueError("replay scorer section or kind differs")
    item = RetentionItem(
        item_id=group.source_id,
        section=cast(Literal["arithmetic", "format", "instruction"], section),
        skill=_require_text(value.get("skill"), "skill"),
        kind=cast(Literal["numeric_terminal", "exact_text", "json_exact"], kind),
        prompt=prompt_text,
        expected=_require_text(value.get("expected"), "expected"),
    )
    return ReplayRewardMetadata(
        replay_id=group.source_id,
        prompt=prompt_text,
        retention_item=item,
        scorer_sha256=_require_sha256(value.get("scorer_sha256"), "scorer_sha256"),
        provenance_sha256=_require_sha256(value.get("provenance_sha256"), "provenance_sha256"),
    )


@dataclass(frozen=True)
class RewardAudit:
    sequence: int
    group_id: str
    source_kind: SourceKind
    source_id: str
    completion: str
    completion_sha256: str
    completion_tokens: int
    reward: RewardBreakdown

    def raw_record(self) -> dict[str, object]:
        return {**self.content_free_record(), "completion": self.completion}

    def content_free_record(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "group_id": self.group_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "completion_sha256": self.completion_sha256,
            "completion_tokens": self.completion_tokens,
            "reward": self.reward.as_dict(),
        }


class VerifierRewardCallback:
    """Exact TRL reward callback bound to one immutable schedule."""

    __name__ = "foundry_l3_verifier_reward"

    def __init__(
        self,
        groups: Sequence[RuntimeGroup],
        *,
        completion_token_counter: Callable[[str], int],
    ) -> None:
        self.groups = {group.group_id: group for group in groups}
        if not groups or len(self.groups) != len(groups):
            raise ValueError("reward callback requires unique scheduled groups")
        self.completion_token_counter = completion_token_counter
        self.records: list[RewardAudit] = []

    def __call__(
        self,
        prompts: list[object],
        completions: list[object],
        *,
        group_id: list[object],
        source_kind: list[object],
        source_id: list[object],
        prompt_sha256: list[object],
        reward_metadata_json: list[object],
        **unused: object,
    ) -> list[float]:
        del unused
        count = len(completions)
        columns = (
            prompts,
            group_id,
            source_kind,
            source_id,
            prompt_sha256,
            reward_metadata_json,
        )
        if count <= 0 or any(len(column) != count for column in columns):
            raise ValueError("reward callback columns differ")
        truncation = get_active_truncation_flags(expected_count=count)
        values: list[float] = []
        for index in range(count):
            current_id = _require_text(group_id[index], "group_id")
            group = self.groups.get(current_id)
            if group is None:
                raise ValueError("reward group is outside the frozen schedule")
            rows = _prompt_rows(prompts[index])
            if (
                source_kind[index] != group.source_kind
                or source_id[index] != group.source_id
                or prompt_sha256[index] != group.prompt_sha256
                or canonical_sha256(list(rows)) != group.prompt_sha256
            ):
                raise ValueError("reward-side identity differs from the schedule")
            metadata_json = _require_text(reward_metadata_json[index], "reward_metadata_json")
            if metadata_json != group.reward_metadata_json:
                raise ValueError("trusted reward metadata differs")
            completion = _completion_text(completions[index])
            result = score_reward(
                _reward_metadata(group, prompts[index], metadata_json),
                completion,
                generation_truncated=truncation[index],
            )
            tokens = self.completion_token_counter(completion)
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                raise ValueError("completion-token count must be positive")
            if not math.isfinite(result.total):
                raise RuntimeError("verifier reward is not finite")
            self.records.append(
                RewardAudit(
                    sequence=len(self.records),
                    group_id=group.group_id,
                    source_kind=group.source_kind,
                    source_id=group.source_id,
                    completion=completion,
                    completion_sha256=hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                    completion_tokens=tokens,
                    reward=result,
                )
            )
            values.append(result.total)
        return values


def summarize_rewards(
    records: Sequence[RewardAudit],
    groups: Sequence[RuntimeGroup],
    *,
    require_nonzero_variance: bool,
) -> dict[str, object]:
    """Validate exact group accounting and project content-free reward metrics."""

    expected = [group.group_id for group in groups]
    if len(records) != len(expected) * COMPLETIONS_PER_GROUP:
        raise RuntimeError("reward completion accounting differs")
    grouped: dict[str, list[float]] = defaultdict(list)
    order: list[str] = []
    for record in records:
        if not order or order[-1] != record.group_id:
            order.append(record.group_id)
        grouped[record.group_id].append(record.reward.total)
    if order != expected or any(
        len(grouped[group_id]) != COMPLETIONS_PER_GROUP for group_id in expected
    ):
        raise RuntimeError("reward group order or cardinality differs")
    deviations = [pstdev(grouped[group_id]) for group_id in expected]
    nonzero = sum(value > 0.0 for value in deviations)
    if require_nonzero_variance and nonzero == 0:
        raise RuntimeError("compatibility reward groups all have zero variance")
    totals = [record.reward.total for record in records]
    task_records = [record for record in records if record.source_kind == "task"]
    return {
        "groups": len(groups),
        "task_groups": sum(group.source_kind == "task" for group in groups),
        "replay_groups": sum(group.source_kind == "base_replay" for group in groups),
        "completions": len(records),
        "mean_reward": fmean(totals),
        "minimum_reward": min(totals),
        "maximum_reward": max(totals),
        "zero_variance_groups": len(groups) - nonzero,
        "nonzero_variance_groups": nonzero,
        "mean_within_group_reward_std": fmean(deviations),
        "task_correct": sum(record.reward.task_answer_correctness > 0.0 for record in task_records),
        "task_correctness_rate": (
            sum(record.reward.task_answer_correctness > 0.0 for record in task_records)
            / len(task_records)
            if task_records
            else 0.0
        ),
        "task_extractable": sum(record.reward.extractable for record in task_records),
        "task_extraction_rate": (
            sum(record.reward.extractable for record in task_records) / len(task_records)
            if task_records
            else 0.0
        ),
        "task_exact_format": sum(record.reward.exact_format for record in task_records),
        "task_exact_format_rate": (
            sum(record.reward.exact_format for record in task_records) / len(task_records)
            if task_records
            else 0.0
        ),
        "truncated": sum(record.reward.generation_truncated for record in records),
        "prompt_echo": sum(record.reward.prompt_echo for record in records),
        "question_generation": sum(record.reward.question_generation for record in records),
        "malformed_output": sum(record.reward.malformed_output for record in records),
        "backend_failures": sum(record.reward.backend_failure for record in records),
        "completion_tokens": sum(record.completion_tokens for record in records),
        "scheduled_prompt_tokens": sum(group.prompt_tokens for group in groups),
        "generation_input_prompt_tokens": (
            sum(group.prompt_tokens for group in groups) * COMPLETIONS_PER_GROUP
        ),
        "content_free_records_sha256": canonical_sha256(
            [record.content_free_record() for record in records]
        ),
    }


def _runtime_modules() -> tuple[dict[str, Any], dict[str, object]]:
    """Validate launch state, then import and initialize the frozen model stack."""

    preimport = validate_preimport()
    modules = {
        name: importlib.import_module(name)
        for name in (
            "bitsandbytes",
            "datasets",
            "numpy",
            "peft",
            "psutil",
            "torch",
            "transformers",
            "trl",
        )
    }
    torch = modules["torch"]
    numpy = modules["numpy"]
    seed = cast(int, GRPO_RECIPE["seed"])
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    postimport = validate_postimport(
        preimport,
        torch,
        {
            name: modules[name]
            for name in ("bitsandbytes", "peft", "psutil", "torch", "transformers")
        },
    )
    module_files = [getattr(module, "__file__", None) for module in modules.values()]
    if any(
        not isinstance(module_file, str) or ".venv-training" not in str(Path(module_file).resolve())
        for module_file in module_files
    ):
        raise RuntimeError("model stack was imported outside .venv-training")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3080":
        raise RuntimeError("CUDA device differs from the frozen RTX 3080")
    torch.cuda.synchronize(0)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    return modules, {"preimport": preimport, "postimport": postimport}


def _strict(torch: Any, stage: str) -> None:
    if not bool(torch.are_deterministic_algorithms_enabled()) or bool(
        torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise RuntimeError(f"strict deterministic enforcement is absent at {stage}")
    actual = {name: __import__("os").environ.get(name, "") for name in DETERMINISTIC_ENVIRONMENT}
    if actual != DETERMINISTIC_ENVIRONMENT:
        raise RuntimeError(f"deterministic process environment changed at {stage}")


def _load_dual_adapter_model(
    *,
    model_path: Path,
    starting_adapter: Path,
    expected_starting_sha256: str,
    modules: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, object], float]:
    """Load one NF4 base and two byte-identical L3 adapters."""

    if directory_sha256(starting_adapter) != expected_starting_sha256:
        raise ValueError("starting L3 adapter directory hash differs")
    started = time.perf_counter()
    base, tokenizer = vetted_qlora_kl._load_base(model_path, dict(modules))
    peft = modules["peft"]
    base = peft.prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=False,
    )
    model = peft.PeftModel.from_pretrained(
        base,
        str(starting_adapter),
        adapter_name=POLICY_ADAPTER_NAME,
        local_files_only=True,
        is_trainable=True,
        low_cpu_mem_usage=True,
    )
    model.load_adapter(
        str(starting_adapter),
        adapter_name=REFERENCE_ADAPTER_NAME,
        local_files_only=True,
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    set_policy_active(model)
    model.config.use_cache = True
    identity = assert_policy_reference_identity(model, require_policy_trainable=True)
    assert_cuda_only_model(model)
    return model, tokenizer, identity, time.perf_counter() - started


def _completion_token_counter(tokenizer: Any) -> Callable[[str], int]:
    def count(text: str) -> int:
        value = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not isinstance(value, list):
            raise TypeError("tokenizer completion IDs must be a list")
        return len(value)

    return count


def _base_output_hash(model: Any, tokenizer: Any, group: RuntimeGroup, torch: Any) -> str:
    """Hash one adapter-disabled base logit slice without generation."""

    text = tokenizer.apply_chat_template(
        [message.as_dict() for message in group.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    prior = bool(model.training)
    model.eval()
    try:
        with model.disable_adapter(), torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, -1, :128]
        evidence = tensor_evidence(logits).as_dict()
    finally:
        model.train(prior)
    return canonical_sha256(evidence)


def _policy_reference_logps(
    model: Any,
    tokenizer: Any,
    group: RuntimeGroup,
    torch: Any,
) -> tuple[Any, Any]:
    text = tokenizer.apply_chat_template(
        [message.as_dict() for message in group.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    prior = bool(model.training)
    model.eval()
    try:
        set_policy_active(model)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            policy = torch.log_softmax(
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                .logits[:, -1, :]
                .float()
                / cast(float, GRPO_RECIPE["temperature"]),
                dim=-1,
            )
        set_reference_active_frozen(model)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            reference = torch.log_softmax(
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                .logits[:, -1, :]
                .float()
                / cast(float, GRPO_RECIPE["temperature"]),
                dim=-1,
            )
    finally:
        set_policy_active(model)
        model.train(prior)
    return policy, reference


def _reference_perturbation_calibration(
    model: Any,
    tokenizer: Any,
    group: RuntimeGroup,
    torch: Any,
) -> dict[str, object]:
    """Prove identity KL and positive controlled KL, then restore exact bytes."""

    identity_before = assert_policy_reference_identity(model, require_policy_trainable=True)
    initial_policy, initial_reference = _policy_reference_logps(model, tokenizer, group, torch)
    if not bool(torch.equal(initial_policy, initial_reference)):
        raise RuntimeError(
            "byte-identical starting adapters produced non-identical log probabilities"
        )
    initial_delta = initial_reference - initial_policy
    initial_kl = torch.exp(initial_delta) - initial_delta - 1
    if float(initial_kl.abs().max().item()) != 0.0:
        raise RuntimeError("initial policy/reference KL is not exactly zero")
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_B.default" in name
    ]
    if not candidates:
        raise RuntimeError("policy perturbation fixture found no L3 LoRA-B tensor")
    name, parameter = sorted(candidates, key=lambda item: item[0])[0]
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.add_(0.05)
        perturbed_policy, perturbed_reference = _policy_reference_logps(
            model, tokenizer, group, torch
        )
        difference = (perturbed_reference - perturbed_policy).abs()
        index = int(difference.argmax().item())
        delta = perturbed_reference.flatten()[index] - perturbed_policy.flatten()[index]
        positive_kl = float((torch.exp(delta) - delta - 1).item())
        if not math.isfinite(positive_kl) or positive_kl <= 0.0:
            raise RuntimeError("controlled policy perturbation did not produce positive KL")
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    restored = assert_policy_reference_identity(model, require_policy_trainable=True)
    if restored["identity_sha256"] != identity_before["identity_sha256"]:
        raise RuntimeError("controlled policy perturbation was not restored exactly")
    return {
        "calibration_id": "foundry-l3-reference-controlled-perturbation-v1",
        "initial_identity_sha256": identity_before["identity_sha256"],
        "initial_max_per_token_kl": 0.0,
        "controlled_parameter_name_sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
        "controlled_positive_per_token_kl": positive_kl,
        "policy_restored_byte_exact": True,
        "reference_unchanged": True,
        "calibration_sha256": canonical_sha256(
            {
                "initial_identity_sha256": identity_before["identity_sha256"],
                "initial_max_per_token_kl": 0.0,
                "controlled_parameter_name_sha256": hashlib.sha256(
                    name.encode("utf-8")
                ).hexdigest(),
                "controlled_positive_per_token_kl": positive_kl,
                "policy_restored_byte_exact": True,
                "reference_unchanged": True,
            }
        ),
    }


def _trainer_arguments(
    trl: Any,
    *,
    output_dir: Path,
    max_steps: int,
) -> Any:
    values: dict[str, object] = {
        "output_dir": str(output_dir / "trainer_state"),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_steps": max_steps,
        "learning_rate": 0.000001,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "fp16": True,
        "bf16": False,
        "tf32": False,
        "gradient_checkpointing": False,
        "full_determinism": False,
        "seed": 20260720,
        "data_seed": 20260720,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "disable_tqdm": True,
        "max_prompt_length": 512,
        "num_generations": 4,
        "max_completion_length": 256,
        "shuffle_dataset": False,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "use_vllm": False,
        "beta": 0.04,
        "num_iterations": 1,
        "epsilon": 0.2,
        "scale_rewards": False,
        "loss_type": "dr_grpo",
        "mask_truncated_completions": True,
        "disable_dropout": True,
        "sync_ref_model": False,
        "log_completions": False,
        "use_liger_loss": False,
    }
    arguments = trl.GRPOConfig(**values)
    for name, expected in values.items():
        actual = getattr(arguments, name)
        if name in {"logging_strategy", "save_strategy", "lr_scheduler_type"}:
            actual = getattr(actual, "value", actual)
        if actual != expected:
            raise ValueError(f"TRL argument {name} differs: expected {expected!r}, got {actual!r}")
    return arguments


def _save_policy_adapter(model: Any, tokenizer: Any, path: Path) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(f"policy adapter output already exists: {path}")
    if active_adapter_name(model) != POLICY_ADAPTER_NAME:
        raise RuntimeError("policy adapter is not active at save")
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        path,
        safe_serialization=True,
        selected_adapters=[POLICY_ADAPTER_NAME],
    )
    tokenizer.save_pretrained(path)
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    if any(not item.is_file() for item in required):
        raise RuntimeError("policy-only adapter save is incomplete")
    return {
        "directory_sha256": directory_sha256(path),
        "adapter_config_sha256": file_sha256(required[0]),
        "adapter_weights_sha256": file_sha256(required[1]),
        "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
    }


def _token_lengths(completion_ids: Any, eos_token_id: int) -> list[int]:
    rows = completion_ids.detach().cpu().tolist()
    if not isinstance(rows, list):
        raise TypeError("completion token IDs must be rows")
    lengths: list[int] = []
    for row in rows:
        if not isinstance(row, list) or not row:
            raise ValueError("completion token row is empty")
        try:
            lengths.append(row.index(eos_token_id) + 1)
        except ValueError:
            lengths.append(len(row))
    return lengths


def _capture_l3_generation_evidence(
    *,
    group: RuntimeGroup,
    generated_token_ids: Any,
    decoded_completions: Sequence[str],
    completion_token_lengths: Sequence[int],
    truncation_flags: Sequence[bool],
    reward_components: Sequence[Mapping[str, object]],
    rng_before_sha256: str,
    rng_after_sha256: str,
    warning_sha256s: Sequence[str],
    reference_logprobs: Any,
    policy_logprobs: Any,
    per_token_kl: Any,
) -> GenerationEvidence:
    """Reuse the frozen tensor capture while labeling human-written task groups exactly."""

    evidence = capture_generation_evidence(
        group_id=group.group_id,
        source_kind=("synthetic" if group.source_kind == "task" else "base_replay"),
        prompt_sha256=group.prompt_sha256,
        generated_token_ids=generated_token_ids,
        decoded_completions=decoded_completions,
        completion_token_lengths=completion_token_lengths,
        truncation_flags=truncation_flags,
        reward_components=reward_components,
        rng_before_sha256=rng_before_sha256,
        rng_after_sha256=rng_after_sha256,
        warning_sha256s=warning_sha256s,
        reference_logprobs=reference_logprobs,
        policy_logprobs=policy_logprobs,
        per_token_kl=per_token_kl,
    )
    if group.source_kind == "base_replay":
        return evidence
    core = evidence.as_dict()
    core.pop("warning_count")
    core.pop("evidence_sha256")
    core["source_kind"] = "task"
    return replace(
        evidence,
        source_kind=cast(Any, "task"),
        evidence_sha256=canonical_sha256(core),
    )


def _assert_gradient_ownership(model: Any, torch: Any) -> dict[str, object]:
    policy_gradients: list[Any] = []
    reference_gradients: list[str] = []
    base_gradients: list[str] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if f".{POLICY_ADAPTER_NAME}." in name and "lora_" in name:
            if gradient is not None:
                policy_gradients.append(gradient)
        elif f".{REFERENCE_ADAPTER_NAME}." in name and "lora_" in name:
            if gradient is not None:
                reference_gradients.append(name)
        elif "lora_" not in name and gradient is not None:
            base_gradients.append(name)
    if (
        not policy_gradients
        or reference_gradients
        or base_gradients
        or any(not bool(torch.isfinite(gradient).all().item()) for gradient in policy_gradients)
    ):
        raise RuntimeError("gradient finiteness or policy-only ownership failed")
    norms = [
        float(torch.linalg.vector_norm(gradient.detach().float()).item())
        for gradient in policy_gradients
    ]
    if not any(value > 0.0 for value in norms):
        raise RuntimeError("policy gradients are all zero")
    return {
        "policy_gradient_tensors": len(policy_gradients),
        "reference_gradient_tensors": len(reference_gradients),
        "base_gradient_tensors": len(base_gradients),
        "finite": True,
        "nonzero": True,
        "global_norm": math.sqrt(math.fsum(value * value for value in norms)),
    }


class SmokeRecorder:
    """Capture exact tensor-bound evidence for two stock Trainer updates."""

    def __init__(
        self,
        *,
        torch_module: Any,
        numpy_random: Any,
        tokenizer: Any,
        reward_callback: VerifierRewardCallback,
        warning_contract: TopPWarningOnlyGenerationContract,
        groups: Sequence[RuntimeGroup],
        reference_initial_sha256: str,
    ) -> None:
        if (
            len(groups) != COMPATIBILITY_STEPS
            or groups[0].source_kind != "task"
            or groups[1].source_kind != "base_replay"
        ):
            raise ValueError("smoke requires one task then one replay group")
        self.torch = torch_module
        self.numpy_random = numpy_random
        self.tokenizer = tokenizer
        self.reward_callback = reward_callback
        self.warning_contract = warning_contract
        self.groups = tuple(groups)
        self.reference_initial_sha256 = reference_initial_sha256
        self.steps: list[CompatibilityStepEvidence] = []
        self.reference_states_after_steps: list[str] = []
        self._generation_calls = 0
        self._record_start = 0
        self._warning_start = 0
        self._pending_result: Mapping[str, Any] | None = None
        self._pending_records: list[RewardAudit] = []
        self._pending_warning: Any | None = None
        self._pending_step: dict[str, Any] | None = None
        self._policy_capture_active = False
        self._captured_policy: Any | None = None

    def step_begin(self, *, state: Any, model: Any, optimizer: Any, scheduler: Any) -> None:
        _strict(self.torch, "smoke step begin")
        expected = len(self.steps) + 1
        if expected not in {1, 2} or int(state.global_step) != expected - 1:
            raise RuntimeError("smoke Trainer step ordering differs")
        if self._pending_step is not None:
            raise RuntimeError("prior smoke step did not finalize")
        self._pending_step = {
            "step": expected,
            "rng_before": capture_rng_state(self.torch, numpy_random=self.numpy_random),
            "lora_before": capture_lora_state(model),
            "optimizer_before": capture_optimizer_state(optimizer),
            "scheduler_before": capture_scheduler_state(scheduler),
            "strict_mode_evidence": {},
        }

    def start_generation(self) -> None:
        _strict(self.torch, "smoke generation entry")
        if self._pending_step is None or self._pending_result is not None:
            raise RuntimeError("smoke generation lifecycle differs")
        self._record_start = len(self.reward_callback.records)
        self._warning_start = len(self.warning_contract.call_records())

    def finish_generation(self, result: Mapping[str, Any]) -> None:
        _strict(self.torch, "smoke generation exit")
        records = list(self.reward_callback.records[self._record_start :])
        warning_rows = list(self.warning_contract.call_records()[self._warning_start :])
        if len(records) != COMPLETIONS_PER_GROUP or len(warning_rows) != 1:
            raise RuntimeError("smoke generation completion or warning count differs")
        expected = self.groups[self._generation_calls]
        if {record.group_id for record in records} != {expected.group_id}:
            raise RuntimeError("smoke reward rows differ from the scheduled group")
        self._pending_result = result
        self._pending_records = records
        self._pending_warning = warning_rows[0]
        self._generation_calls += 1

    def begin_policy_capture(self) -> None:
        if self._policy_capture_active or self._captured_policy is not None:
            raise RuntimeError("nested policy-logprob capture is prohibited")
        self._policy_capture_active = True

    def capture_policy(self, value: Any) -> None:
        if self._policy_capture_active:
            if self._captured_policy is not None:
                raise RuntimeError("stock loss requested policy log probabilities twice")
            self._captured_policy = value

    def end_policy_capture(self) -> Any:
        self._policy_capture_active = False
        if self._captured_policy is None:
            raise RuntimeError("stock loss produced no policy log probabilities")
        value = self._captured_policy
        self._captured_policy = None
        return value

    def abort_policy_capture(self) -> None:
        self._policy_capture_active = False
        self._captured_policy = None

    def record_loss(
        self,
        *,
        loss: Any,
        reference: Any,
        policy: Any,
        completion_mask: Any,
    ) -> None:
        if (
            self._pending_step is None
            or self._pending_result is None
            or self._pending_warning is None
        ):
            raise RuntimeError("smoke loss lacks matching generation evidence")
        delta = reference - policy
        per_token_kl = self.torch.exp(delta) - delta - 1
        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        if not bool(self.torch.isfinite(loss).all().item()) or not bool(
            self.torch.isfinite(per_token_kl).all().item()
        ):
            raise RuntimeError("smoke loss or KL is non-finite")
        group = self.groups[self._generation_calls - 1]
        warning = self._pending_warning
        generation = _capture_l3_generation_evidence(
            group=group,
            generated_token_ids=self._pending_result["completion_ids"],
            decoded_completions=[record.completion for record in self._pending_records],
            completion_token_lengths=_token_lengths(
                self._pending_result["completion_ids"],
                int(self.tokenizer.eos_token_id),
            ),
            truncation_flags=[
                record.reward.generation_truncated for record in self._pending_records
            ],
            reward_components=[record.reward.as_dict() for record in self._pending_records],
            rng_before_sha256=str(warning.rng_before_sha256),
            rng_after_sha256=str(warning.rng_after_sha256),
            warning_sha256s=warning.warning_sha256s,
            reference_logprobs=reference,
            policy_logprobs=policy,
            per_token_kl=per_token_kl,
        )
        self._pending_step.update(
            {
                "generation": generation,
                "loss": float(loss.detach().float().item()),
                "loss_tensor": loss.detach(),
                "mean_kl": float(mean_kl.detach().float().item()),
                "mean_kl_tensor": mean_kl.detach(),
            }
        )
        self._pending_result = None
        self._pending_records = []
        self._pending_warning = None

    def after_backward(self, model: Any) -> None:
        _strict(self.torch, "smoke backward")
        if self._pending_step is None or "loss_tensor" not in self._pending_step:
            raise RuntimeError("smoke backward lacks matching loss")
        _assert_gradient_ownership(model, self.torch)
        self._pending_step["gradients_after_backward"] = capture_gradient_state(model)

    def pre_optimizer(self, model: Any) -> None:
        _strict(self.torch, "smoke pre-optimizer")
        if self._pending_step is None or "gradients_after_backward" not in self._pending_step:
            raise RuntimeError("smoke clipping preceded backward")
        _assert_gradient_ownership(model, self.torch)
        self._pending_step["gradients_after_clipping"] = capture_gradient_state(model)
        self._pending_step["strict_mode_evidence"]["after_backward"] = True
        self._pending_step["strict_mode_evidence"]["before_optimizer"] = True

    def post_optimizer(self, model: Any, optimizer: Any) -> None:
        _strict(self.torch, "smoke optimizer")
        if self._pending_step is None or "gradients_after_clipping" not in self._pending_step:
            raise RuntimeError("smoke optimizer preceded clipping")
        reference = capture_adapter_state(model, REFERENCE_ADAPTER_NAME)
        reference_hash = str(reference["normalized_tensor_state_sha256"])
        if reference_hash != self.reference_initial_sha256:
            raise RuntimeError("reference adapter changed during smoke optimizer update")
        self.reference_states_after_steps.append(reference_hash)
        self._pending_step["optimizer_after"] = capture_optimizer_state(optimizer)
        self._pending_step["lora_after"] = capture_lora_state(model)
        self._pending_step["strict_mode_evidence"]["after_optimizer"] = True

    def step_end(self, state: Any, scheduler: Any) -> None:
        _strict(self.torch, "smoke scheduler")
        pending = self._pending_step
        if pending is None or int(state.global_step) != int(pending["step"]):
            raise RuntimeError("smoke scheduler ordering differs")
        pending["scheduler_after"] = capture_scheduler_state(scheduler)
        pending["rng_after"] = capture_rng_state(self.torch, numpy_random=self.numpy_random)
        pending["strict_mode_evidence"]["after_scheduler"] = True
        self.steps.append(
            build_compatibility_step_evidence(
                step=int(pending["step"]),
                generation=cast(GenerationEvidence, pending["generation"]),
                loss=float(pending["loss"]),
                loss_tensor=pending["loss_tensor"],
                mean_kl=float(pending["mean_kl"]),
                mean_kl_tensor=pending["mean_kl_tensor"],
                rng_before=cast(dict[str, object], pending["rng_before"]),
                rng_after=cast(dict[str, object], pending["rng_after"]),
                lora_before=cast(dict[str, object], pending["lora_before"]),
                lora_after=cast(dict[str, object], pending["lora_after"]),
                gradients_after_backward=cast(
                    dict[str, object], pending["gradients_after_backward"]
                ),
                gradients_after_clipping=cast(
                    dict[str, object], pending["gradients_after_clipping"]
                ),
                optimizer_before=cast(dict[str, object], pending["optimizer_before"]),
                optimizer_after=cast(dict[str, object], pending["optimizer_after"]),
                scheduler_before=cast(dict[str, object], pending["scheduler_before"]),
                scheduler_after=cast(dict[str, object], pending["scheduler_after"]),
                strict_mode_evidence=cast(dict[str, bool], pending["strict_mode_evidence"]),
            )
        )
        self._pending_step = None

    def assert_complete(self) -> None:
        if (
            len(self.steps) != COMPATIBILITY_STEPS
            or self._generation_calls != COMPATIBILITY_STEPS
            or self._pending_step is not None
            or self._pending_result is not None
            or self.reference_states_after_steps
            != [self.reference_initial_sha256] * COMPATIBILITY_STEPS
        ):
            raise RuntimeError("exact two-step smoke evidence is incomplete")


def make_smoke_callback(base: type[Any], recorder: SmokeRecorder) -> object:
    class SmokeCallback(base):  # type: ignore[misc]
        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args
            recorder.step_begin(
                state=state,
                model=kwargs["model"],
                optimizer=kwargs["optimizer"],
                scheduler=kwargs["lr_scheduler"],
            )
            return control

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state
            recorder.pre_optimizer(kwargs["model"])
            return control

        def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state
            recorder.post_optimizer(kwargs["model"], kwargs["optimizer"])
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args
            recorder.step_end(state, kwargs["lr_scheduler"])
            return control

    SmokeCallback.__name__ = "L3ExactSmokeCallback"
    return SmokeCallback()


def make_smoke_trainer(base: type[Any], recorder: SmokeRecorder) -> type[Any]:
    class SmokeTrainer(base):  # type: ignore[misc]
        def _generate_and_score_completions(self, inputs: Any) -> Any:
            recorder.start_generation()
            result = super()._generate_and_score_completions(inputs)
            if not isinstance(result, Mapping):
                raise TypeError("stock GRPO generation result must be a mapping")
            recorder.finish_generation(cast(Mapping[str, Any], result))
            return result

        def _get_per_token_logps(self, *args: Any, **kwargs: Any) -> Any:
            value = super()._get_per_token_logps(*args, **kwargs)
            recorder.capture_policy(value)
            return value

        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            recorder.begin_policy_capture()
            try:
                loss = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
                policy = recorder.end_policy_capture()
            except BaseException:
                recorder.abort_policy_capture()
                raise
            reference = inputs.get("ref_per_token_logps")
            completion_mask = inputs.get("completion_mask")
            if reference is None or completion_mask is None:
                raise RuntimeError("stock GRPO loss lacks reference log probabilities")
            recorder.record_loss(
                loss=loss,
                reference=reference,
                policy=policy,
                completion_mask=completion_mask,
            )
            return loss

        def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
            loss = super().training_step(model, inputs, num_items_in_batch)
            recorder.after_backward(model)
            return loss

    SmokeTrainer.__name__ = "L3ExactSmokeTrainer"
    return SmokeTrainer


class FullAuditRecorder:
    """Capture content-free per-group, loss, KL, gradient, and update trajectories."""

    def __init__(
        self,
        *,
        torch_module: Any,
        tokenizer: Any,
        reward_callback: VerifierRewardCallback,
        warning_contract: TopPWarningOnlyGenerationContract,
        groups: Sequence[RuntimeGroup],
        reference_initial_sha256: str,
    ) -> None:
        if len(groups) != GROUPS_PER_ARM:
            raise ValueError("counted recorder requires all 32 groups")
        self.torch = torch_module
        self.tokenizer = tokenizer
        self.reward_callback = reward_callback
        self.warning_contract = warning_contract
        self.groups = tuple(groups)
        self.reference_initial_sha256 = reference_initial_sha256
        self.generations: list[dict[str, object]] = []
        self.loss_trajectory: list[float] = []
        self.kl_trajectory: list[float] = []
        self.gradient_trajectory: list[dict[str, object]] = []
        self.policy_state_trajectory: list[str] = []
        self.learning_rate_trajectory: list[float] = []
        self._record_start = 0
        self._warning_start = 0
        self._pending_result: Mapping[str, Any] | None = None
        self._pending_records: list[RewardAudit] = []
        self._pending_warning: Any | None = None
        self._policy_capture = False
        self._policy_value: Any | None = None

    def start_generation(self) -> None:
        _strict(self.torch, "counted generation entry")
        if self._pending_result is not None or len(self.generations) >= GROUPS_PER_ARM:
            raise RuntimeError("counted generation lifecycle differs")
        self._record_start = len(self.reward_callback.records)
        self._warning_start = len(self.warning_contract.call_records())

    def finish_generation(self, result: Mapping[str, Any]) -> None:
        _strict(self.torch, "counted generation exit")
        records = list(self.reward_callback.records[self._record_start :])
        warnings = list(self.warning_contract.call_records()[self._warning_start :])
        expected = self.groups[len(self.generations)]
        if (
            len(records) != COMPLETIONS_PER_GROUP
            or len(warnings) != 1
            or {record.group_id for record in records} != {expected.group_id}
        ):
            raise RuntimeError("counted generation records differ from schedule")
        self._pending_result = result
        self._pending_records = records
        self._pending_warning = warnings[0]

    def begin_policy_capture(self) -> None:
        if self._policy_capture or self._policy_value is not None:
            raise RuntimeError("counted policy capture lifecycle differs")
        self._policy_capture = True

    def capture_policy(self, value: Any) -> None:
        if self._policy_capture:
            if self._policy_value is not None:
                raise RuntimeError("counted loss requested policy log probabilities twice")
            self._policy_value = value

    def finish_loss(
        self,
        *,
        loss: Any,
        reference: Any,
        completion_mask: Any,
    ) -> None:
        self._policy_capture = False
        policy = self._policy_value
        self._policy_value = None
        if policy is None or self._pending_result is None or self._pending_warning is None:
            raise RuntimeError("counted loss lacks matching generation")
        delta = reference - policy
        per_token_kl = self.torch.exp(delta) - delta - 1
        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        if not bool(self.torch.isfinite(loss).all().item()) or not bool(
            self.torch.isfinite(per_token_kl).all().item()
        ):
            raise RuntimeError("counted GRPO loss or KL is non-finite")
        group = self.groups[len(self.generations)]
        warning = self._pending_warning
        generation = _capture_l3_generation_evidence(
            group=group,
            generated_token_ids=self._pending_result["completion_ids"],
            decoded_completions=[record.completion for record in self._pending_records],
            completion_token_lengths=_token_lengths(
                self._pending_result["completion_ids"],
                int(self.tokenizer.eos_token_id),
            ),
            truncation_flags=[
                record.reward.generation_truncated for record in self._pending_records
            ],
            reward_components=[record.reward.as_dict() for record in self._pending_records],
            rng_before_sha256=str(warning.rng_before_sha256),
            rng_after_sha256=str(warning.rng_after_sha256),
            warning_sha256s=warning.warning_sha256s,
            reference_logprobs=reference,
            policy_logprobs=policy,
            per_token_kl=per_token_kl,
        )
        self.generations.append(generation.as_dict())
        self.loss_trajectory.append(float(loss.detach().float().item()))
        self.kl_trajectory.append(float(mean_kl.detach().float().item()))
        self._pending_result = None
        self._pending_records = []
        self._pending_warning = None

    def abort_policy_capture(self) -> None:
        self._policy_capture = False
        self._policy_value = None

    def after_backward(self, model: Any) -> None:
        _strict(self.torch, "counted backward")
        evidence = _assert_gradient_ownership(model, self.torch)
        evidence["step"] = len(self.gradient_trajectory) + 1
        self.gradient_trajectory.append(evidence)

    def pre_optimizer(self, model: Any) -> None:
        _strict(self.torch, "counted clipping")
        _assert_gradient_ownership(model, self.torch)

    def post_optimizer(self, model: Any) -> None:
        _strict(self.torch, "counted optimizer")
        reference = capture_adapter_state(model, REFERENCE_ADAPTER_NAME)
        if reference["normalized_tensor_state_sha256"] != self.reference_initial_sha256:
            raise RuntimeError("reference adapter changed in counted training")
        policy = capture_adapter_state(model, POLICY_ADAPTER_NAME)
        self.policy_state_trajectory.append(str(policy["normalized_tensor_state_sha256"]))

    def step_end(self, state: Any, scheduler: Any) -> None:
        _strict(self.torch, "counted scheduler")
        if int(state.global_step) != len(self.learning_rate_trajectory) + 1:
            raise RuntimeError("counted scheduler step ordering differs")
        state_dict = scheduler.state_dict()
        if int(state_dict["last_epoch"]) != int(state.global_step):
            raise RuntimeError("counted scheduler epoch differs from global step")
        self.learning_rate_trajectory.append(float(scheduler.optimizer.param_groups[0]["lr"]))

    def assert_complete(self) -> None:
        lengths = {
            len(self.generations),
            len(self.loss_trajectory),
            len(self.kl_trajectory),
            len(self.gradient_trajectory),
            len(self.policy_state_trajectory),
            len(self.learning_rate_trajectory),
        }
        if lengths != {GROUPS_PER_ARM} or self._pending_result is not None:
            raise RuntimeError("counted per-step evidence is incomplete")

    def evidence(self) -> dict[str, object]:
        self.assert_complete()
        payload: dict[str, object] = {
            "groups": self.generations,
            "loss_trajectory": self.loss_trajectory,
            "kl_trajectory": self.kl_trajectory,
            "gradient_trajectory": self.gradient_trajectory,
            "policy_state_trajectory": self.policy_state_trajectory,
            "learning_rate_trajectory": self.learning_rate_trajectory,
        }
        payload["trajectory_sha256"] = canonical_sha256(payload)
        return payload


def make_full_callback(base: type[Any], recorder: FullAuditRecorder) -> object:
    class FullCallback(base):  # type: ignore[misc]
        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state
            recorder.pre_optimizer(kwargs["model"])
            return control

        def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state
            recorder.post_optimizer(kwargs["model"])
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args
            recorder.step_end(state, kwargs["lr_scheduler"])
            return control

    FullCallback.__name__ = "L3FullAuditCallback"
    return FullCallback()


def make_full_trainer(base: type[Any], recorder: FullAuditRecorder) -> type[Any]:
    class FullTrainer(base):  # type: ignore[misc]
        def _generate_and_score_completions(self, inputs: Any) -> Any:
            recorder.start_generation()
            result = super()._generate_and_score_completions(inputs)
            if not isinstance(result, Mapping):
                raise TypeError("stock GRPO generation result must be a mapping")
            recorder.finish_generation(cast(Mapping[str, Any], result))
            return result

        def _get_per_token_logps(self, *args: Any, **kwargs: Any) -> Any:
            value = super()._get_per_token_logps(*args, **kwargs)
            recorder.capture_policy(value)
            return value

        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            recorder.begin_policy_capture()
            try:
                loss = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except BaseException:
                recorder.abort_policy_capture()
                raise
            reference = inputs.get("ref_per_token_logps")
            completion_mask = inputs.get("completion_mask")
            if reference is None or completion_mask is None:
                raise RuntimeError("counted loss lacks reference log probabilities")
            recorder.finish_loss(
                loss=loss,
                reference=reference,
                completion_mask=completion_mask,
            )
            return loss

        def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
            loss = super().training_step(model, inputs, num_items_in_batch)
            recorder.after_backward(model)
            return loss

    FullTrainer.__name__ = "L3FullAuditTrainer"
    return FullTrainer


def make_checkpoint_callback(
    base: type[Any],
    *,
    tokenizer: Any,
    output_dir: Path,
    checkpoint_evidence: dict[str, dict[str, object]],
) -> object:
    class CheckpointCallback(base):  # type: ignore[misc]
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if int(state.max_steps) != OPTIMIZER_STEPS:
                raise RuntimeError("checkpoint callback requires exactly 32 steps")
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args
            step = int(state.global_step)
            if step in CHECKPOINT_STEPS:
                model = kwargs["model"]
                reference = capture_adapter_state(model, REFERENCE_ADAPTER_NAME)
                checkpoint = output_dir / f"checkpoint-{step}" / "adapter"
                checkpoint_evidence[str(step)] = {
                    **_save_policy_adapter(model, tokenizer, checkpoint),
                    "policy_tensor_state_sha256": capture_adapter_state(model, POLICY_ADAPTER_NAME)[
                        "normalized_tensor_state_sha256"
                    ],
                    "reference_tensor_state_sha256": reference["normalized_tensor_state_sha256"],
                }
            return control

    CheckpointCallback.__name__ = "L3ExactCheckpointCallback"
    return CheckpointCallback()


def _optimizer_ownership(trainer: Any) -> dict[str, object]:
    model = trainer.model
    policy_parameters = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if f".{POLICY_ADAPTER_NAME}." in name and "lora_" in name and parameter.requires_grad
    }
    reference_parameters = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if f".{REFERENCE_ADAPTER_NAME}." in name and "lora_" in name
    }
    optimizer_parameters = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in cast(list[Any], group["params"])
    }
    if (
        len(policy_parameters) != EXPECTED_ADAPTER_TENSORS
        or optimizer_parameters != policy_parameters
        or optimizer_parameters & reference_parameters
    ):
        raise RuntimeError("optimizer ownership differs from policy-only L3 tensors")
    return {
        "optimizer_parameter_tensors": len(optimizer_parameters),
        "optimizer_parameter_count": sum(
            int(parameter.numel())
            for name, parameter in model.named_parameters()
            if id(parameter) in optimizer_parameters
        ),
        "policy_only": True,
        "reference_owned": False,
        "base_owned": False,
    }


def _finite_history(log_history: Sequence[Mapping[str, object]]) -> dict[str, list[float]]:
    keys = ("loss", "grad_norm", "learning_rate", "kl", "reward", "reward_std")
    result: dict[str, list[float]] = {}
    for key in keys:
        values: list[float] = []
        for row in log_history:
            raw = row.get(key)
            if isinstance(raw, int | float):
                values.append(float(raw))
        if values:
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"Trainer history contains non-finite {key}")
            result[key] = values
    if "loss" not in result or "kl" not in result:
        raise RuntimeError("Trainer history lacks finite loss or KL")
    return result


def _validate_experiment_contract(
    path: Path,
    *,
    schedule: RuntimeSchedule,
    arm: Arm,
) -> dict[str, object]:
    contract = _read(path)
    declared = contract.get("experiment_contract_sha256")
    if declared != canonical_sha256(
        {key: value for key, value in contract.items() if key != "experiment_contract_sha256"}
    ):
        raise ValueError("experiment contract self-hash differs")
    if (
        contract.get("recipe_sha256") != GRPO_RECIPE_SHA256
        or contract.get("recipe") != GRPO_RECIPE
        or contract.get("schedule_id") != SCHEDULE_ID
        or cast(dict[str, object], contract["reward"]).get("contract_sha256")
        != reward_contract_sha256()
        or cast(dict[str, object], contract["reward"]).get("implementation_sha256")
        != reward_implementation_sha256()
        or cast(dict[str, object], contract["reward"]).get("configuration_sha256")
        != reward_configuration_sha256()
        or contract.get("reference") != reference_mechanism_contract()
        or contract.get("fixed_library_notice_contract_sha256")
        != FIXED_LIBRARY_NOTICE_CONTRACT_SHA256
    ):
        raise ValueError("experiment recipe or reward contract differs")
    root = path.parents[2]
    implementation = _read(path.with_name("milestone14a_implementation.json"))
    if implementation.get("implementation_sha256") != canonical_sha256(
        {key: value for key, value in implementation.items() if key != "implementation_sha256"}
    ) or implementation.get("implementation_sha256") != contract.get("implementation_sha256"):
        raise ValueError("frozen implementation manifest differs")
    for row_value in _array(implementation.get("files"), "implementation.files"):
        row = _object(row_value, "implementation.files[]")
        relative = _require_text(row.get("path"), "implementation.path")
        source = (root / relative).resolve()
        if not source.is_relative_to(root.resolve()) or file_sha256(source) != row.get("sha256"):
            raise ValueError("frozen implementation source differs")
    starting = _read(path.with_name("milestone14a_starting_state.json"))
    if starting.get("starting_state_sha256") != canonical_sha256(
        {key: value for key, value in starting.items() if key != "starting_state_sha256"}
    ) or starting.get("starting_state_sha256") != contract.get("starting_state_sha256"):
        raise ValueError("frozen starting-state contract differs")
    paired_path = path.with_name("milestone14a_paired_schedule.json")
    paired = _read(paired_path)
    if (
        paired.get("paired_schedule_sha256") != contract.get("paired_schedule_sha256")
        or paired.get(f"{arm}_manifest_sha256") != schedule.manifest_sha256
    ):
        raise ValueError("runtime schedule differs from the paired experiment contract")
    return contract


def _run(
    *,
    root: Path,
    arm: Arm,
    mode: RuntimeMode,
    packet_path: Path,
    manifest_path: Path,
    experiment_contract_path: Path,
    starting_adapter: Path,
    output_dir: Path,
    raw_evidence_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    if any(path.exists() for path in (output_dir, raw_evidence_path, summary_path)):
        raise FileExistsError("Milestone 14A runtime outputs must start unused")
    if file_sha256(root / ".venv-training/Scripts/python.exe") != INTERPRETER_SHA256:
        raise ValueError("authorized training interpreter differs")
    schedule = load_schedule(packet_path, manifest_path, arm)
    contract = _validate_experiment_contract(experiment_contract_path, schedule=schedule, arm=arm)
    expected_starting = STARTING_ADAPTER_SHA256[arm]
    if directory_sha256(starting_adapter) != expected_starting:
        raise ValueError("selected L3 starting adapter differs")
    groups: tuple[RuntimeGroup, ...]
    if mode == "compatibility":
        task = next(group for group in schedule.groups if group.source_kind == "task")
        replay = next(group for group in schedule.groups if group.source_kind == "base_replay")
        groups = (task, replay)
        max_steps = COMPATIBILITY_STEPS
    else:
        groups = schedule.groups
        max_steps = OPTIMIZER_STEPS

    modules, launch = _runtime_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    trl = modules["trl"]
    datasets = modules["datasets"]
    peft = modules["peft"]
    psutil = modules["psutil"]
    numpy = modules["numpy"]
    _strict(torch, "before model load")
    process = psutil.Process()
    started = time.perf_counter()
    model_path = (
        root
        / "data/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
        / f"snapshots/{MODEL_REVISION}"
    )
    model, tokenizer, initial_identity, model_load_seconds = _load_dual_adapter_model(
        model_path=model_path,
        starting_adapter=starting_adapter,
        expected_starting_sha256=expected_starting,
        modules=modules,
    )
    _strict(torch, "after model load")
    base_before = capture_base_parameter_state(model)
    base_output_before = _base_output_hash(model, tokenizer, groups[0], torch)
    reference_calibration = _reference_perturbation_calibration(model, tokenizer, groups[0], torch)
    initial_policy = capture_adapter_state(model, POLICY_ADAPTER_NAME)
    initial_reference = capture_adapter_state(model, REFERENCE_ADAPTER_NAME)
    reference_initial_sha256 = str(initial_reference["normalized_tensor_state_sha256"])
    reward_callback = VerifierRewardCallback(
        groups,
        completion_token_counter=_completion_token_counter(tokenizer),
    )
    arguments = _trainer_arguments(trl, output_dir=output_dir, max_steps=max_steps)
    _strict(torch, "after GRPOConfig")
    warning_contract = TopPWarningOnlyGenerationContract(
        torch_module=torch,
        generation_owner=transformers.GenerationMixin,
        top_p_call=transformers.generation.logits_process.TopPLogitsWarper.__call__,
    )
    audited_base = make_truncation_aware_grpo_trainer(
        trl.GRPOTrainer,
        generation_scope_factory=partial(warning_contract.install, "generation"),
    )
    callbacks: list[object] = []
    checkpoint_evidence: dict[str, dict[str, object]] = {}
    if mode == "compatibility":
        recorder: SmokeRecorder | FullAuditRecorder = SmokeRecorder(
            torch_module=torch,
            numpy_random=numpy.random,
            tokenizer=tokenizer,
            reward_callback=reward_callback,
            warning_contract=warning_contract,
            groups=groups,
            reference_initial_sha256=reference_initial_sha256,
        )
        trainer_type = make_smoke_trainer(audited_base, cast(SmokeRecorder, recorder))
        callbacks.append(
            make_smoke_callback(transformers.TrainerCallback, cast(SmokeRecorder, recorder))
        )
    else:
        recorder = FullAuditRecorder(
            torch_module=torch,
            tokenizer=tokenizer,
            reward_callback=reward_callback,
            warning_contract=warning_contract,
            groups=groups,
            reference_initial_sha256=reference_initial_sha256,
        )
        trainer_type = make_full_trainer(audited_base, recorder)
        callbacks.extend(
            [
                make_full_callback(transformers.TrainerCallback, recorder),
                make_checkpoint_callback(
                    transformers.TrainerCallback,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    checkpoint_evidence=checkpoint_evidence,
                ),
            ]
        )
    train_dataset = datasets.Dataset.from_list([group.policy_row() for group in groups])
    trainer = trainer_type(
        model=model,
        reward_funcs=reward_callback,
        args=arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=None,
    )
    reference_proxy = SharedStartingPolicyReference(trainer.model, torch)
    trainer.ref_model = reference_proxy
    warning_contract.bind_state_probe(partial(model_adapter_state, trainer.model))
    set_policy_active(trainer.model)
    assert_policy_reference_identity(trainer.model, require_policy_trainable=True)
    assert_cuda_only_model(trainer.model)
    dropout_count = assert_dropout_disabled(trainer.model, torch)
    _strict(torch, "before training")
    training_started = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize(0)
    training_seconds = time.perf_counter() - training_started
    _strict(torch, "after training")
    if int(trainer.state.global_step) != max_steps:
        raise RuntimeError("GRPO optimizer-step accounting differs")
    if mode == "compatibility":
        cast(SmokeRecorder, recorder).assert_complete()
    else:
        cast(FullAuditRecorder, recorder).assert_complete()
        if set(checkpoint_evidence) != {"8", "16", "32"}:
            raise RuntimeError("counted checkpoint set differs")
    reward_summary = summarize_rewards(
        reward_callback.records,
        groups,
        require_nonzero_variance=mode == "compatibility",
    )
    expected_completions = (
        COMPATIBILITY_COMPLETIONS if mode == "compatibility" else COMPLETIONS_PER_ARM
    )
    if (
        reward_summary["completions"] != expected_completions
        or reward_summary["backend_failures"] != 0
    ):
        raise RuntimeError("completion accounting or backend gate failed")
    warning_evidence = warning_contract.evidence()
    if warning_evidence["generation_calls"] != len(groups):
        raise RuntimeError("warning-only generation call count differs")
    history = _finite_history(cast(Sequence[Mapping[str, object]], trainer.state.log_history))
    optimizer_ownership = _optimizer_ownership(trainer)
    final_policy = capture_adapter_state(trainer.model, POLICY_ADAPTER_NAME)
    final_reference = capture_adapter_state(trainer.model, REFERENCE_ADAPTER_NAME)
    if (
        final_policy["normalized_tensor_state_sha256"]
        == initial_policy["normalized_tensor_state_sha256"]
    ):
        raise RuntimeError("GRPO policy adapter did not update")
    if final_reference["normalized_tensor_state_sha256"] != reference_initial_sha256:
        raise RuntimeError("frozen L3 reference adapter changed")
    base_after = capture_base_parameter_state(trainer.model)
    base_output_after = _base_output_hash(trainer.model, tokenizer, groups[0], torch)
    if (
        base_after["base_parameter_state_sha256"] != base_before["base_parameter_state_sha256"]
        or base_output_after != base_output_before
    ):
        raise RuntimeError("frozen base changed during GRPO")
    final_optimizer = capture_optimizer_state(trainer.optimizer)
    final_scheduler = capture_scheduler_state(trainer.lr_scheduler)
    final_adapter = _save_policy_adapter(trainer.model, tokenizer, output_dir / "final_adapter")
    if mode == "train" and (
        final_adapter["directory_sha256"] != checkpoint_evidence["32"]["directory_sha256"]
    ):
        raise RuntimeError("final policy differs from checkpoint 32")

    raw_evidence: dict[str, object] = {
        "schema_version": 1,
        "runtime_id": RUNTIME_ID,
        "arm": arm,
        "mode": mode,
        "schedule_packet_sha256": schedule.packet_sha256,
        "records": [record.raw_record() for record in reward_callback.records],
    }
    raw_evidence["raw_evidence_sha256"] = canonical_sha256(raw_evidence)
    _write_json_new(raw_evidence_path, raw_evidence)

    reference_runtime = reference_proxy.evidence()
    reference_call_count = reference_runtime.get("call_count")
    if (
        isinstance(reference_call_count, bool)
        or not isinstance(reference_call_count, int)
        or reference_call_count != len(groups)
    ):
        raise RuntimeError("starting-policy reference call count differs")
    if mode == "compatibility":
        exact_packet: dict[str, object] = {
            "schema_version": 1,
            "packet_id": "foundry-l3-verifier-grpo-two-step-exact-v1",
            "runtime_id": RUNTIME_ID,
            "arm": arm,
            "starting_adapter_sha256": expected_starting,
            "experiment_contract_sha256": contract["experiment_contract_sha256"],
            "schedule_packet_sha256": schedule.packet_sha256,
            "schedule_manifest_sha256": schedule.manifest_sha256,
            "group_ids": [group.group_id for group in groups],
            "source_kinds": [group.source_kind for group in groups],
            "optimizer_steps": COMPATIBILITY_STEPS,
            "completion_count": COMPATIBILITY_COMPLETIONS,
            "recipe_sha256": GRPO_RECIPE_SHA256,
            "reward_summary": reward_summary,
            "steps": [step.as_dict() for step in cast(SmokeRecorder, recorder).steps],
            "initial_identity": initial_identity,
            "initial_policy": initial_policy,
            "initial_reference": initial_reference,
            "final_policy": final_policy,
            "final_reference": final_reference,
            "base_before": base_before,
            "base_after": base_after,
            "base_output_before": base_output_before,
            "base_output_after": base_output_after,
            "reference_calibration": reference_calibration,
            "reference_runtime": reference_runtime,
            "warning_evidence": warning_evidence,
            "optimizer_ownership": optimizer_ownership,
            "final_optimizer": final_optimizer,
            "final_scheduler": final_scheduler,
            "final_adapter": final_adapter,
        }
        exact_packet["packet_sha256"] = canonical_sha256(exact_packet)
    else:
        exact_packet = {}
    trajectory = None if mode == "compatibility" else cast(FullAuditRecorder, recorder).evidence()

    trained_model_reference = weakref.ref(trainer.model)
    warning_contract.release_state_probe()
    del callbacks, trainer_type, audited_base
    del trainer, model, reference_proxy, recorder, warning_contract
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    if trained_model_reference() is not None:
        raise RuntimeError("trained dual-adapter model remained alive before reload")
    pre_reload_allocated = int(torch.cuda.memory_allocated(0))
    if pre_reload_allocated >= 512 * 1024**2:
        raise RuntimeError("trained model was not released before offline reload")

    reload_started = time.perf_counter()
    reload_base, reload_tokenizer = vetted_qlora_kl._load_base(model_path, modules)
    reload_base = peft.prepare_model_for_kbit_training(
        reload_base, use_gradient_checkpointing=False
    )
    reloaded = peft.PeftModel.from_pretrained(
        reload_base,
        str(output_dir / "final_adapter"),
        local_files_only=True,
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    reload_seconds = time.perf_counter() - reload_started
    assert_cuda_only_model(reloaded)
    if any(parameter.requires_grad for parameter in reloaded.parameters()):
        raise RuntimeError("offline policy reload left trainable parameters")
    reloaded_policy = capture_adapter_state(reloaded, POLICY_ADAPTER_NAME)
    reloaded_base_state = capture_base_parameter_state(reloaded)
    reloaded_base_output = _base_output_hash(reloaded, reload_tokenizer, groups[0], torch)
    if (
        reloaded_policy["normalized_tensor_state_sha256"]
        != final_policy["normalized_tensor_state_sha256"]
        or reloaded_base_state["base_parameter_state_sha256"]
        != base_before["base_parameter_state_sha256"]
        or reloaded_base_output != base_output_before
    ):
        raise RuntimeError("offline adapter reload or base restoration differs")
    if mode == "compatibility":
        exact_packet["reloaded_policy"] = reloaded_policy
        exact_packet["reloaded_base"] = reloaded_base_state
        exact_packet["reloaded_base_output"] = reloaded_base_output
        exact_packet["packet_sha256"] = canonical_sha256(
            {key: value for key, value in exact_packet.items() if key != "packet_sha256"}
        )
    torch.cuda.synchronize(0)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    physical = int(torch.cuda.get_device_properties(0).total_memory)
    if peak_reserved >= physical:
        raise RuntimeError("peak reserved VRAM reached or exceeded physical GPU capacity")
    runtime_seconds = time.perf_counter() - started
    output_disk_bytes = sum(item.stat().st_size for item in output_dir.rglob("*") if item.is_file())
    summary: dict[str, object] = {
        "schema_version": 1,
        "runtime_id": RUNTIME_ID,
        "run_kind": ("compatibility_smoke" if mode == "compatibility" else "counted_training"),
        "arm": arm,
        "starting_adapter_sha256": expected_starting,
        "experiment_contract_sha256": contract["experiment_contract_sha256"],
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "recipe_sha256": GRPO_RECIPE_SHA256,
        "optimizer_steps": max_steps,
        "groups": len(groups),
        "completions": expected_completions,
        "reward": reward_summary,
        "history": history,
        "trajectory": trajectory,
        "initial_policy": initial_policy,
        "initial_reference": initial_reference,
        "initial_identity": initial_identity,
        "final_policy": final_policy,
        "final_reference": final_reference,
        "reloaded_policy": reloaded_policy,
        "reference_calibration": reference_calibration,
        "reference_runtime": reference_runtime,
        "optimizer_ownership": optimizer_ownership,
        "base_parameter_state_sha256": base_before["base_parameter_state_sha256"],
        "base_unchanged": True,
        "reference_unchanged": True,
        "policy_updated": True,
        "second_full_base_model": False,
        "cpu_offload": False,
        "warning_evidence": warning_evidence,
        "checkpoint_evidence": checkpoint_evidence,
        "final_adapter": final_adapter,
        "offline_reload_passed": True,
        "adapter_disabled_base_restoration_passed": True,
        "dropout_disabled": True,
        "dropout_module_count": dropout_count,
        "launch_evidence": launch,
        "model_load_seconds": model_load_seconds,
        "training_seconds": training_seconds,
        "reload_seconds": reload_seconds,
        "runtime_seconds": runtime_seconds,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "physical_vram_bytes": physical,
        "peak_reserved_below_physical": True,
        "peak_process_rss_bytes": _peak_process_ram(process),
        "output_disk_bytes": output_disk_bytes,
        "pre_reload_allocated_vram_bytes": pre_reload_allocated,
        "raw_evidence_file_sha256": file_sha256(raw_evidence_path),
        "prompts_completions_or_answers_present": False,
        "gate_passed": True,
    }
    if mode == "compatibility":
        summary["exact_packet_sha256"] = exact_packet["packet_sha256"]
        summary["exact_packet"] = exact_packet
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json_new(summary_path, summary)
    del reloaded, reload_base
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    _strict(torch, "result publication")
    return summary


def run(
    *,
    root: Path,
    arm: Arm,
    mode: RuntimeMode,
    packet_path: Path,
    manifest_path: Path,
    experiment_contract_path: Path,
    starting_adapter: Path,
    output_dir: Path,
    raw_evidence_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Run one fresh compatibility or counted process."""

    try:
        return _run(
            root=root.resolve(),
            arm=arm,
            mode=mode,
            packet_path=packet_path.resolve(),
            manifest_path=manifest_path.resolve(),
            experiment_contract_path=experiment_contract_path.resolve(),
            starting_adapter=starting_adapter.resolve(),
            output_dir=output_dir.resolve(),
            raw_evidence_path=raw_evidence_path.resolve(),
            summary_path=summary_path.resolve(),
        )
    finally:
        gc.collect()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("generic", "targeted"), required=True)
    parser.add_argument("--mode", choices=("compatibility", "train"), required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-contract", type=Path, required=True)
    parser.add_argument("--starting-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(
        root=args.root,
        arm=cast(Arm, args.arm),
        mode=cast(RuntimeMode, args.mode),
        packet_path=args.packet,
        manifest_path=args.manifest,
        experiment_contract_path=args.experiment_contract,
        starting_adapter=args.starting_adapter,
        output_dir=args.output_dir,
        raw_evidence_path=args.raw_evidence,
        summary_path=args.summary,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
