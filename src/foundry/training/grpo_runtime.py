"""Audited runtime for the frozen verifier-reward GRPO experiment.

The prompt schedule is loaded from an ignored packet while its content-free
manifest is independently checked.  Only the packet's two-message
conversation is presented to the policy; answers and scorer metadata remain
opaque columns consumed by :class:`VerifierRewardCallback`.

Heavy training dependencies are imported only inside :func:`run_grpo`.  This
keeps the contract and its unit tests usable in the main development
environment while the counted run remains confined to ``.venv-training``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import sysconfig
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal, cast

from foundry.training.config import canonical_sha256
from foundry.training.grpo_compatibility import (
    TopPWarningOnlyGenerationContract,
    model_adapter_state,
)
from foundry.training.grpo_config import (
    BASE_REVISION,
    VerifierGRPOConfig,
    load_grpo_config,
)
from foundry.training.grpo_environment import (
    FROZEN_TRANSFORMERS_DETERMINISM_FUNCTION_SHA256,
    FROZEN_TRANSFORMERS_DETERMINISM_MODULE,
    FROZEN_TRANSFORMERS_DETERMINISM_QUALNAME,
    FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT,
    assert_idempotent_deterministic_initialization,
    make_environment_guarded_trainer,
    make_environment_validation_callback,
    transformers_determinism_source_evidence,
    validate_deterministic_process_environment,
)
from foundry.training.grpo_gpu import (
    EXPECTED_GPU_TOTAL_MEMORY_BYTES,
    EXPECTED_TORCH_CUDA_RUNTIME,
    ChildCudaComputeEvidence,
    collect_child_cuda_compute_evidence,
    current_cuda_memory_evidence,
)
from foundry.training.grpo_paths import (
    GrpoRuntimePaths,
    assert_artifact_path,
    assert_source_path,
    deterministic_process_contract,
    frozen_process_environment,
    load_runtime_paths,
    same_canonical_path,
    validate_runtime_paths,
)
from foundry.training.grpo_reference import (
    FROZEN_CHECKPOINT_STEPS,
    assert_only_lora_trainable,
    make_exact_checkpoint_callback,
    run_adapter_disabled_reference,
    validate_installed_reference_contract,
)
from foundry.training.grpo_replay_evidence import capture_base_parameter_state
from foundry.training.grpo_reward import (
    ReplayRewardMetadata,
    RewardBreakdown,
    SyntheticRewardMetadata,
    reward_configuration_sha256,
    reward_implementation_sha256,
    score_reward,
)
from foundry.training.grpo_schedule import (
    COMPLETIONS_PER_GROUP,
    FAMILY_ORDER,
    GROUPS_PER_ARM,
    REPLAY_GROUPS_PER_ARM,
    REPLAY_POSITIONS,
    REPLAY_SECTION_ORDER,
    SCHEDULE_ID,
    SCHEDULE_SCHEMA_VERSION,
    SCHEDULE_SEED,
    SYNTHETIC_GROUPS_PER_ARM,
    SYNTHETIC_QUOTAS,
    Arm,
    PromptMessage,
)
from foundry.training.grpo_trainer import (
    get_active_truncation_flags,
    make_truncation_aware_grpo_trainer,
)
from foundry.training.qlora import directory_sha256
from foundry.training.retention import RetentionItem

RuntimeMode = Literal["compatibility", "train"]

RUNTIME_ID = "foundry-verifier-grpo-runtime-v1"
RUNTIME_SCHEMA_VERSION = 1
COMPATIBILITY_UPDATE_GROUPS = 2
COMPATIBILITY_GENERATION_ONLY_GROUPS = 1
COMPATIBILITY_COMPLETIONS = 12
FULL_COMPLETIONS = GROUPS_PER_ARM * COMPLETIONS_PER_GROUP
MAX_RESERVED_VRAM_BYTES = int(9.6 * 1024**3)
FROZEN_GPU_TOTAL_MEMORY_BYTES = EXPECTED_GPU_TOTAL_MEMORY_BYTES
FROZEN_TORCH_CUDA_RUNTIME = EXPECTED_TORCH_CUDA_RUNTIME
FROZEN_PROCESS_SEED = 20_260_720
FROZEN_CUBLAS_WORKSPACE_CONFIG = ":16:8"
FROZEN_TRANSFORMERS_CUBLAS_WORKSPACE_CONFIG = FROZEN_CUBLAS_WORKSPACE_CONFIG
FROZEN_FULL_DETERMINISM_SOURCE_SHA256 = FROZEN_TRANSFORMERS_DETERMINISM_FUNCTION_SHA256
FROZEN_FULL_DETERMINISM_MODULE = FROZEN_TRANSFORMERS_DETERMINISM_MODULE
FROZEN_FULL_DETERMINISM_QUALNAME = FROZEN_TRANSFORMERS_DETERMINISM_QUALNAME
FROZEN_PYTHON_IMPLEMENTATION = "CPython"
FROZEN_PYTHON_VERSION = "3.12.10"
_PYTHON_HASH_PROBE = "foundry-verifier-grpo-python-hash-probe-v1"
FROZEN_SOFTWARE_VERSIONS = {
    "accelerate": "1.7.0",
    "bitsandbytes": "0.49.2",
    "datasets": "5.0.0",
    "numpy": "2.5.1",
    "peft": "0.15.2",
    "psutil": "7.2.2",
    "tokenizers": "0.21.4",
    "torch": "2.5.1+cu121",
    "transformers": "4.51.3",
    "trl": "0.17.0",
}

_SHA256 = frozenset("0123456789abcdef")
_SYNTHETIC_METADATA_FIELDS = frozenset(
    {
        "reward_kind",
        "canonical_final_answer",
        "family",
        "mode",
        "difficulty",
        "output_contract_enabled",
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
_GROUP_FIELDS = frozenset(
    {
        "group_id",
        "position",
        "source_kind",
        "source_id",
        "category",
        "prompt_sha256",
        "prompt_tokens",
        "completions_per_group",
        "messages",
        "reward_metadata",
    }
)


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
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
    return cast(list[object], value)


def _load_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {name}: {error}") from error
    return _object(value, name)


def _serialized(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class RuntimePromptGroup:
    """One verified runtime prompt plus hidden reward-side metadata."""

    group_id: str
    arm: Arm
    position: int
    source_kind: Literal["synthetic", "base_replay"]
    source_id: str
    category: str
    messages: tuple[PromptMessage, ...]
    prompt_sha256: str
    prompt_tokens: int
    reward_metadata_json: str

    def policy_row(self) -> dict[str, object]:
        """Return a Trainer row whose only model-visible field is ``prompt``."""

        return {
            "prompt": [message.as_dict() for message in self.messages],
            "group_id": self.group_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "prompt_sha256": self.prompt_sha256,
            "reward_metadata_json": self.reward_metadata_json,
        }

    def content_free_record(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "position": self.position,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "category": self.category,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tokens": self.prompt_tokens,
        }


@dataclass(frozen=True)
class RuntimeSchedule:
    """One strict 64-group schedule loaded from a prompt-bearing local packet."""

    arm: Arm
    groups: tuple[RuntimePromptGroup, ...]
    packet_sha256: str
    manifest_sha256: str


def _parse_messages(value: object, *, group_id: str) -> tuple[PromptMessage, ...]:
    rows = _array(value, f"{group_id}.messages")
    if len(rows) != 2:
        raise ValueError("each GRPO prompt must contain exactly system and user messages")
    messages: list[PromptMessage] = []
    for index, raw in enumerate(rows):
        row = _object(raw, f"{group_id}.messages[{index}]")
        if set(row) != {"role", "content"}:
            raise ValueError("prompt message fields differ from the frozen prompt-only schema")
        expected_role = "system" if index == 0 else "user"
        if row.get("role") != expected_role:
            raise ValueError(
                "GRPO prompts must be ordered system then user with no assistant target"
            )
        messages.append(
            PromptMessage(
                cast(Literal["system", "user"], expected_role),
                _require_text(row.get("content"), "message.content"),
            )
        )
    return tuple(messages)


def _validate_reward_metadata(
    value: object, *, source_kind: str, group_id: str
) -> dict[str, object]:
    metadata = _object(value, f"{group_id}.reward_metadata")
    expected_fields = (
        _SYNTHETIC_METADATA_FIELDS if source_kind == "synthetic" else _REPLAY_METADATA_FIELDS
    )
    if set(metadata) != expected_fields:
        raise ValueError("reward metadata differs from the frozen hidden schema")
    expected_kind = "synthetic" if source_kind == "synthetic" else "base_replay"
    if metadata.get("reward_kind") != expected_kind:
        raise ValueError("reward metadata kind differs from the scheduled source kind")
    for key, item in metadata.items():
        if key == "output_contract_enabled":
            if not isinstance(item, bool):
                raise ValueError("output_contract_enabled must be boolean")
        elif key != "reward_kind":
            _require_text(item, f"reward_metadata.{key}")
    for key in ("verifier_metadata_sha256", "scorer_sha256", "provenance_sha256"):
        if key in metadata:
            _require_sha256(metadata[key], f"reward_metadata.{key}")
    return metadata


def load_runtime_schedule(
    packet_path: Path,
    manifest_path: Path,
    *,
    expected_arm: Arm,
) -> RuntimeSchedule:
    """Load one ignored packet and bind it to its content-free tracked manifest."""

    packet = _load_json_object(packet_path, "GRPO prompt packet")
    manifest = _load_json_object(manifest_path, "GRPO schedule manifest")
    packet_hash = canonical_sha256(packet)
    declared_manifest_hash = manifest.get("manifest_sha256")
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("manifest_sha256", None)
    if declared_manifest_hash != canonical_sha256(manifest_without_hash):
        raise ValueError("GRPO schedule manifest self-hash differs")
    if (
        packet.get("schema_version") != SCHEDULE_SCHEMA_VERSION
        or manifest.get("schema_version") != SCHEDULE_SCHEMA_VERSION
        or packet.get("schedule_id") != SCHEDULE_ID
        or manifest.get("schedule_id") != SCHEDULE_ID
        or packet.get("seed") != SCHEDULE_SEED
        or manifest.get("seed") != SCHEDULE_SEED
        or packet.get("arm") != expected_arm
        or manifest.get("arm") != expected_arm
        or manifest.get("prompt_packet_sha256") != packet_hash
        or manifest.get("groups_per_arm") != GROUPS_PER_ARM
        or manifest.get("completions_per_group") != COMPLETIONS_PER_GROUP
        or manifest.get("total_completions") != FULL_COMPLETIONS
        or manifest.get("prompts_or_outputs_in_manifest") is not False
    ):
        raise ValueError("GRPO packet or manifest identity differs from the frozen schedule")
    if set(packet) != {"schema_version", "schedule_id", "seed", "arm", "groups"}:
        raise ValueError("GRPO prompt packet contains fields outside the frozen schema")
    packet_groups = _array(packet.get("groups"), "packet.groups")
    manifest_groups = _array(manifest.get("groups"), "manifest.groups")
    if len(packet_groups) != GROUPS_PER_ARM or len(manifest_groups) != GROUPS_PER_ARM:
        raise ValueError("GRPO schedule must contain exactly 64 prompt groups")

    groups: list[RuntimePromptGroup] = []
    group_ids: set[str] = set()
    source_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for expected_position, (packet_value, manifest_value) in enumerate(
        zip(packet_groups, manifest_groups, strict=True), start=1
    ):
        row = _object(packet_value, f"packet.groups[{expected_position - 1}]")
        manifest_row = _object(manifest_value, f"manifest.groups[{expected_position - 1}]")
        if set(row) != _GROUP_FIELDS:
            raise ValueError("runtime group fields differ from the prompt-only schema")
        content_free = {
            key: item for key, item in row.items() if key not in {"messages", "reward_metadata"}
        }
        if content_free != manifest_row:
            raise ValueError("runtime group differs from its content-free manifest record")
        group_id = _require_text(row.get("group_id"), "group_id")
        source_id = _require_text(row.get("source_id"), "source_id")
        source_kind_value = row.get("source_kind")
        if source_kind_value not in {"synthetic", "base_replay"}:
            raise ValueError("runtime group has an unknown source kind")
        source_kind = cast(Literal["synthetic", "base_replay"], source_kind_value)
        position = row.get("position")
        prompt_tokens = row.get("prompt_tokens")
        completions = row.get("completions_per_group")
        if position != expected_position:
            raise ValueError("runtime group positions are not contiguous and stable")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
        ):
            raise ValueError("prompt token count must be a positive integer")
        if completions != COMPLETIONS_PER_GROUP:
            raise ValueError("runtime group completion count differs from the frozen group size")
        messages = _parse_messages(row.get("messages"), group_id=group_id)
        prompt_hash = _require_sha256(row.get("prompt_sha256"), "prompt_sha256")
        if canonical_sha256([message.as_dict() for message in messages]) != prompt_hash:
            raise ValueError("runtime prompt text differs from its frozen hash")
        metadata = _validate_reward_metadata(
            row.get("reward_metadata"), source_kind=source_kind, group_id=group_id
        )
        if group_id in group_ids or source_id in source_ids or prompt_hash in prompt_hashes:
            raise ValueError("runtime group IDs, source IDs, and prompts must be unique")
        group_ids.add(group_id)
        source_ids.add(source_id)
        prompt_hashes.add(prompt_hash)
        groups.append(
            RuntimePromptGroup(
                group_id=group_id,
                arm=expected_arm,
                position=expected_position,
                source_kind=source_kind,
                source_id=source_id,
                category=_require_text(row.get("category"), "category"),
                messages=messages,
                prompt_sha256=prompt_hash,
                prompt_tokens=prompt_tokens,
                reward_metadata_json=_serialized(metadata),
            )
        )
    synthetic = [group for group in groups if group.source_kind == "synthetic"]
    replay = [group for group in groups if group.source_kind == "base_replay"]
    if len(synthetic) != SYNTHETIC_GROUPS_PER_ARM or len(replay) != REPLAY_GROUPS_PER_ARM:
        raise ValueError("runtime source composition differs from the frozen 52/12 schedule")
    if [group.position for group in replay] != list(REPLAY_POSITIONS):
        raise ValueError("runtime replay positions differ from the frozen schedule")
    if Counter(group.category for group in synthetic) != Counter(SYNTHETIC_QUOTAS[expected_arm]):
        raise ValueError("runtime synthetic family quotas differ from the frozen arm")
    if Counter(group.category for group in replay) != Counter(
        {section: 4 for section in REPLAY_SECTION_ORDER}
    ):
        raise ValueError("runtime replay category quotas differ from the frozen 4/4/4 mix")
    if any(group.category not in FAMILY_ORDER for group in synthetic):
        raise ValueError("runtime synthetic group has an unknown reasoning family")
    return RuntimeSchedule(
        expected_arm,
        tuple(groups),
        packet_hash,
        _require_sha256(declared_manifest_hash, "manifest_sha256"),
    )


def select_compatibility_groups(
    schedule: RuntimeSchedule,
) -> tuple[tuple[RuntimePromptGroup, ...], RuntimePromptGroup]:
    """Select two fixed synthetic updates and one distinct replay-only probe."""

    synthetic = tuple(group for group in schedule.groups if group.source_kind == "synthetic")
    replay = tuple(group for group in schedule.groups if group.source_kind == "base_replay")
    if len(synthetic) < COMPATIBILITY_UPDATE_GROUPS or not replay:
        raise ValueError("schedule lacks the frozen compatibility group composition")
    return synthetic[:COMPATIBILITY_UPDATE_GROUPS], replay[0]


def _prompt_rows(prompt: object) -> tuple[dict[str, str], ...]:
    if not isinstance(prompt, Sequence) or isinstance(prompt, str | bytes | bytearray):
        raise TypeError("conversational prompt must be a message sequence")
    values: list[dict[str, str]] = []
    for raw in prompt:
        if not isinstance(raw, Mapping) or set(raw) != {"role", "content"}:
            raise ValueError("runtime prompt message differs from the frozen projection")
        role = raw.get("role")
        content = raw.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError("runtime prompt contains an assistant target or malformed content")
        values.append({"role": role, "content": content})
    if [item["role"] for item in values] != ["system", "user"]:
        raise ValueError("runtime prompt must remain system then user")
    return tuple(values)


def _completion_text(value: object) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("completion must contain non-empty text")
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        rows = list(value)
        if len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise ValueError("conversational completion must contain one assistant message")
        row = cast(Mapping[object, object], rows[0])
        if set(row) != {"role", "content"} or row.get("role") != "assistant":
            raise ValueError("completion structure differs from the stock TRL contract")
        text = row.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("assistant completion must contain non-empty text")
        return text
    raise TypeError("completion has an unsupported representation")


def _user_prompt_text(prompt: object) -> str:
    rows = _prompt_rows(prompt)
    return rows[-1]["content"]


def _reward_metadata(
    group: RuntimePromptGroup, metadata_json: str, prompt: object
) -> SyntheticRewardMetadata | ReplayRewardMetadata:
    try:
        value: object = json.loads(metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError("runtime reward metadata is malformed") from error
    metadata = _validate_reward_metadata(
        value, source_kind=group.source_kind, group_id=group.group_id
    )
    prompt_text = _user_prompt_text(prompt)
    if group.source_kind == "synthetic":
        return SyntheticRewardMetadata(
            synthetic_id=group.source_id,
            prompt=prompt_text,
            canonical_answer=_require_text(
                metadata.get("canonical_final_answer"), "canonical_final_answer"
            ),
            family=_require_text(metadata.get("family"), "family"),
            submode=_require_text(metadata.get("mode"), "mode"),
            difficulty=_require_text(metadata.get("difficulty"), "difficulty"),
            output_contract_enabled=cast(bool, metadata["output_contract_enabled"]),
            verifier_metadata_sha256=_require_sha256(
                metadata.get("verifier_metadata_sha256"), "verifier_metadata_sha256"
            ),
            provenance_sha256=_require_sha256(
                metadata.get("provenance_sha256"), "provenance_sha256"
            ),
        )
    section = _require_text(metadata.get("section"), "section")
    kind = _require_text(metadata.get("kind"), "kind")
    if section not in {"arithmetic", "format", "instruction"}:
        raise ValueError("replay section differs from the frozen scorer contract")
    if kind not in {"numeric_terminal", "exact_text", "json_exact"}:
        raise ValueError("replay scorer kind differs from the frozen scorer contract")
    item = RetentionItem(
        item_id=group.source_id,
        section=cast(Literal["arithmetic", "format", "instruction"], section),
        skill=_require_text(metadata.get("skill"), "skill"),
        kind=cast(Literal["numeric_terminal", "exact_text", "json_exact"], kind),
        prompt=prompt_text,
        expected=_require_text(metadata.get("expected"), "expected"),
    )
    return ReplayRewardMetadata(
        replay_id=group.source_id,
        prompt=prompt_text,
        retention_item=item,
        scorer_metadata_sha256=_require_sha256(metadata.get("scorer_sha256"), "scorer_sha256"),
        provenance_sha256=_require_sha256(metadata.get("provenance_sha256"), "provenance_sha256"),
    )


@dataclass(frozen=True)
class CompletionRewardAudit:
    """Local completion-bearing audit row; only its projection may be tracked."""

    sequence: int
    group_id: str
    source_kind: str
    source_id: str
    completion: str
    completion_sha256: str
    completion_tokens: int
    reward: RewardBreakdown

    def raw_record(self) -> dict[str, object]:
        value = self.content_free_record()
        value["completion"] = self.completion
        return value

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


TruncationProvider = Callable[..., tuple[bool, ...]]
CompletionTokenCounter = Callable[[str], int]


class VerifierRewardCallback:
    """TRL reward callback bound to one already-verified prompt schedule."""

    __name__ = "foundry_verifier_reward"

    def __init__(
        self,
        groups: Sequence[RuntimePromptGroup],
        *,
        truncation_provider: TruncationProvider = get_active_truncation_flags,
        completion_token_counter: CompletionTokenCounter | None = None,
    ) -> None:
        if not groups:
            raise ValueError("reward callback requires at least one scheduled group")
        self._groups = {group.group_id: group for group in groups}
        if len(self._groups) != len(groups):
            raise ValueError("reward callback group IDs must be unique")
        self._truncation_provider = truncation_provider
        self._token_counter = completion_token_counter or (lambda text: len(text.split()))
        self.records: list[CompletionRewardAudit] = []

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
        columns = (prompts, group_id, source_kind, source_id, prompt_sha256, reward_metadata_json)
        if count == 0 or any(len(column) != count for column in columns):
            raise ValueError("reward callback columns must be non-empty and equally sized")
        flags = self._truncation_provider(expected_count=count)
        rewards: list[float] = []
        for index in range(count):
            item_group_id = _require_text(group_id[index], "group_id")
            group = self._groups.get(item_group_id)
            if group is None:
                raise ValueError("reward callback received a group outside the frozen schedule")
            if source_kind[index] != group.source_kind or source_id[index] != group.source_id:
                raise ValueError("reward-side identity differs from the frozen schedule")
            rows = _prompt_rows(prompts[index])
            actual_prompt_hash = canonical_sha256(list(rows))
            if (
                prompt_sha256[index] != group.prompt_sha256
                or actual_prompt_hash != group.prompt_sha256
            ):
                raise ValueError("model-visible prompt differs from its frozen schedule hash")
            metadata_json = _require_text(reward_metadata_json[index], "reward_metadata_json")
            if metadata_json != group.reward_metadata_json:
                raise ValueError("trusted reward metadata differs from the frozen schedule")
            completion = _completion_text(completions[index])
            metadata = _reward_metadata(group, metadata_json, prompts[index])
            breakdown = score_reward(metadata, completion, generation_truncated=flags[index])
            completion_tokens = self._token_counter(completion)
            if (
                isinstance(completion_tokens, bool)
                or not isinstance(completion_tokens, int)
                or completion_tokens <= 0
            ):
                raise ValueError("completion-token counter must return a positive integer")
            if not math.isfinite(breakdown.total):
                raise RuntimeError("deterministic reward is not finite")
            self.records.append(
                CompletionRewardAudit(
                    sequence=len(self.records),
                    group_id=group.group_id,
                    source_kind=group.source_kind,
                    source_id=group.source_id,
                    completion=completion,
                    completion_sha256=hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                    completion_tokens=completion_tokens,
                    reward=breakdown,
                )
            )
            rewards.append(breakdown.total)
        return rewards


def _component_means(records: Sequence[CompletionRewardAudit]) -> dict[str, float]:
    fields = (
        "correctness",
        "extractability",
        "exact_contract",
        "replay_task_correctness",
        "replay_format_contract",
        "truncation_penalty",
        "echo_or_question_penalty",
        "conflicting_answers_penalty",
        "total",
    )
    return {
        field: fmean(float(getattr(record.reward, field)) for record in records) for field in fields
    }


def _source_reward_means(
    records: Sequence[CompletionRewardAudit], source_kind: str
) -> dict[str, float] | None:
    selected = [record for record in records if record.source_kind == source_kind]
    return None if not selected else _component_means(selected)


def summarize_reward_records(
    records: Sequence[CompletionRewardAudit],
    expected_groups: Sequence[RuntimePromptGroup],
    *,
    require_nonzero_variance: bool,
) -> dict[str, object]:
    """Validate exact group accounting and return a content-free reward summary."""

    expected_ids = [group.group_id for group in expected_groups]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected reward groups must be unique")
    expected_completions = len(expected_ids) * COMPLETIONS_PER_GROUP
    if len(records) != expected_completions:
        raise RuntimeError(
            f"completion count differs: expected {expected_completions}, got {len(records)}"
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    encountered: list[str] = []
    for record in records:
        if not encountered or encountered[-1] != record.group_id:
            encountered.append(record.group_id)
        grouped[record.group_id].append(record.reward.total)
    if encountered != expected_ids or set(grouped) != set(expected_ids):
        raise RuntimeError("reward group order or membership differs from the frozen schedule")
    if any(len(values) != COMPLETIONS_PER_GROUP for values in grouped.values()):
        raise RuntimeError("every prompt group must produce exactly four completions")
    standard_deviations = {group_id: pstdev(grouped[group_id]) for group_id in expected_ids}
    nonzero = sum(value > 0.0 for value in standard_deviations.values())
    if require_nonzero_variance and nonzero == 0:
        raise RuntimeError("compatibility smoke produced no within-group reward variance")
    truncated = sum(record.reward.generation_truncated for record in records)
    synthetic_correct = sum(record.reward.correctness > 0.0 for record in records)
    replay_correct = sum(record.reward.replay_task_correctness > 0.0 for record in records)
    extractable = sum(record.reward.extractability > 0.0 for record in records)
    exact_contract = sum(record.reward.exact_contract > 0.0 for record in records)
    replay_format = sum(record.reward.replay_format_contract > 0.0 for record in records)
    echo_or_question = sum(
        record.reward.prompt_echo or record.reward.question_generation for record in records
    )
    conflicts = sum(record.reward.conflicting_answers for record in records)
    return {
        "groups": len(expected_ids),
        "completions": len(records),
        "synthetic_groups": sum(group.source_kind == "synthetic" for group in expected_groups),
        "replay_groups": sum(group.source_kind == "base_replay" for group in expected_groups),
        "completion_tokens": sum(record.completion_tokens for record in records),
        "scheduled_prompt_tokens": sum(group.prompt_tokens for group in expected_groups),
        "generation_input_prompt_tokens": sum(group.prompt_tokens for group in expected_groups)
        * COMPLETIONS_PER_GROUP,
        "mean_completion_tokens": fmean(record.completion_tokens for record in records),
        "truncated_completions": truncated,
        "truncated_completion_rate": truncated / len(records),
        "nonzero_variance_groups": nonzero,
        "zero_variance_groups": len(expected_ids) - nonzero,
        "mean_within_group_reward_std": fmean(standard_deviations.values()),
        "reward_component_means": _component_means(records),
        "synthetic_reward_component_means": _source_reward_means(records, "synthetic"),
        "replay_reward_component_means": _source_reward_means(records, "base_replay"),
        "reward_event_counts": {
            "synthetic_correct": synthetic_correct,
            "replay_correct": replay_correct,
            "task_correct_total": synthetic_correct + replay_correct,
            "synthetic_extractable": extractable,
            "synthetic_exact_contract": exact_contract,
            "replay_exact_format": replay_format,
            "truncated": truncated,
            "prompt_echo_or_question": echo_or_question,
            "conflicting_answers": conflicts,
        },
        "content_free_records_sha256": canonical_sha256(
            [record.content_free_record() for record in records]
        ),
    }


def frozen_grpo_argument_values(
    config: VerifierGRPOConfig,
    *,
    variant_id: str,
    output_dir: Path,
    mode: RuntimeMode,
) -> dict[str, object]:
    """Translate the frozen contract to exact TRL 0.17 ``GRPOConfig`` values."""

    grpo = config.grpo
    max_steps = grpo.optimizer_steps if mode == "train" else COMPATIBILITY_UPDATE_GROUPS
    return {
        "output_dir": str(output_dir / "trainer_state"),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": grpo.per_device_train_batch_size,
        "gradient_accumulation_steps": grpo.gradient_accumulation_steps,
        "max_steps": max_steps,
        "learning_rate": grpo.learning_rate,
        "optim": grpo.optimizer,
        "lr_scheduler_type": grpo.scheduler,
        "warmup_ratio": grpo.warmup_ratio,
        "weight_decay": grpo.weight_decay,
        "max_grad_norm": grpo.max_gradient_norm,
        "fp16": True,
        "bf16": False,
        "tf32": False,
        "gradient_checkpointing": config.memory_and_reproducibility.gradient_checkpointing,
        "full_determinism": config.memory_and_reproducibility.full_determinism,
        "seed": grpo.seed,
        "data_seed": grpo.seed,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "no",
        "save_only_model": True,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "disable_tqdm": False,
        "max_prompt_length": grpo.max_prompt_length,
        "num_generations": grpo.num_generations,
        "max_completion_length": grpo.max_completion_length,
        "shuffle_dataset": config.memory_and_reproducibility.shuffle,
        "temperature": grpo.temperature,
        "top_p": grpo.top_p,
        "top_k": grpo.top_k,
        "use_vllm": grpo.use_vllm,
        "beta": config.variant(variant_id).beta,
        "num_iterations": grpo.num_iterations,
        "epsilon": grpo.epsilon,
        "scale_rewards": grpo.scale_rewards,
        "loss_type": grpo.loss_type,
        "mask_truncated_completions": grpo.mask_truncated_completions,
        "disable_dropout": grpo.disable_dropout,
        "sync_ref_model": config.reference_policy.sync_ref,
        "log_completions": False,
        "use_liger_loss": False,
    }


def assert_frozen_grpo_arguments(
    arguments: object,
    config: VerifierGRPOConfig,
    *,
    variant_id: str,
    output_dir: Path,
    mode: RuntimeMode,
) -> None:
    """Fail closed if TRL coerced or defaulted any decision-bearing argument."""

    for name, expected in frozen_grpo_argument_values(
        config, variant_id=variant_id, output_dir=output_dir, mode=mode
    ).items():
        actual = getattr(arguments, name)
        if name in {"logging_strategy", "save_strategy", "lr_scheduler_type"}:
            actual = getattr(actual, "value", actual)
        if actual != expected:
            raise ValueError(f"TRL argument {name} differs: expected {expected!r}, got {actual!r}")


def _assert_offline_model_snapshot(model_path: Path, config: VerifierGRPOConfig) -> None:
    resolved = model_path.resolve()
    if not resolved.is_dir() or resolved.name != config.base_model.revision:
        raise ValueError("local model path is not the exact frozen revision snapshot")
    expected_repository_component = "models--" + config.base_model.model_id.replace("/", "--")
    if expected_repository_component not in resolved.parts:
        raise ValueError("local model path does not belong to the frozen model ID")
    required = ("config.json", "tokenizer_config.json")
    if any(not (resolved / name).is_file() for name in required):
        raise FileNotFoundError("frozen model snapshot lacks configuration or tokenizer files")
    if (
        not any(resolved.glob("*.safetensors"))
        and not (resolved / "model.safetensors.index.json").is_file()
    ):
        raise FileNotFoundError("frozen model snapshot lacks safetensors weights")


def _assert_git_ignored(repository_root: Path, path: Path, description: str) -> None:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"{description} must stay inside the repository") from error
    result = subprocess.run(
        ["git", "-C", str(repository_root), "check-ignore", "--quiet", str(relative)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"{description} path is not Git ignored: {path}")


def assert_cuda_only_model(model: object) -> tuple[str, ...]:
    """Require all model parameters and any explicit device map to remain on CUDA."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model does not expose named_parameters")
    devices: set[str] = set()
    offloaded: list[str] = []
    for name, parameter in named_parameters():
        device = str(parameter.device)
        devices.add(device)
        if not device.startswith("cuda"):
            offloaded.append(str(name))
    if offloaded:
        raise RuntimeError(f"CPU or disk model offloading is prohibited: {offloaded[:3]}")
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        invalid = [value for value in device_map.values() if value not in {0, "cuda", "cuda:0"}]
        if invalid:
            raise RuntimeError(
                f"model device map contains CPU, disk, or another device: {invalid[:3]}"
            )
    if not devices:
        raise ValueError("model has no parameters")
    return tuple(sorted(devices))


def assert_frozen_base_has_no_gradients(model: object) -> None:
    """Require that no frozen non-LoRA parameter accumulated a gradient."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model does not expose named_parameters")
    offenders = [
        str(name)
        for name, parameter in named_parameters()
        if "lora_" not in str(name) and getattr(parameter, "grad", None) is not None
    ]
    if offenders:
        raise RuntimeError(f"frozen base parameters received gradients: {offenders[:3]}")


def assert_dropout_disabled(model: object, torch_module: Any) -> int:
    """Require TRL's frozen dropout-disable switch to have taken effect."""

    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise TypeError("model does not expose modules")
    dropout_modules = [item for item in modules() if isinstance(item, torch_module.nn.Dropout)]
    if any(float(item.p) != 0.0 for item in dropout_modules):
        raise RuntimeError("TRL did not disable all policy dropout modules")
    return len(dropout_modules)


def _completion_token_counter(tokenizer: object) -> CompletionTokenCounter:
    def count(text: str) -> int:
        encoded = tokenizer(text, add_special_tokens=False)  # type: ignore[operator]
        if not isinstance(encoded, Mapping):
            raise TypeError("tokenizer completion output must be a mapping")
        token_ids = encoded.get("input_ids")
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, str | bytes | bytearray):
            raise TypeError("tokenizer completion IDs must be a sequence")
        return len(token_ids)

    return count


def _base_reference_hash(model: Any, tokenizer: Any, group: RuntimePromptGroup, torch: Any) -> str:
    encoded = tokenizer.apply_chat_template(
        [message.as_dict() for message in group.messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if int(encoded.shape[-1]) > 64:
        encoded = encoded[:, -64:]
    encoded = encoded.to("cuda:0")

    def forward() -> Any:
        return model(input_ids=encoded, attention_mask=torch.ones_like(encoded)).logits[:, -1, :]

    logits = run_adapter_disabled_reference(model, forward, torch)
    values = logits.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _capture_base_tensor_state(model: object, torch: Any) -> dict[str, object]:
    """Capture a compact exact-byte identity for every non-LoRA parameter."""

    _synchronize_cuda(torch)
    state = capture_base_parameter_state(model)
    _synchronize_cuda(torch)
    required = (
        "parameter_count",
        "total_numel",
        "total_bytes",
        "base_parameter_state_sha256",
    )
    if any(name not in state for name in required):
        raise RuntimeError("exact base-parameter evidence is incomplete")
    digest = _require_sha256(state["base_parameter_state_sha256"], "base parameter state")
    counts: dict[str, int] = {}
    for name in required[:3]:
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"exact base-parameter evidence has invalid {name}")
        counts[name] = value
    return {**counts, "base_parameter_state_sha256": digest}


def _assert_base_tensor_state_unchanged(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    stage: str,
) -> None:
    if dict(after) != dict(before):
        raise RuntimeError(f"exact non-LoRA base tensor bytes changed during {stage}")


def adapter_artifact_sha256(path: Path) -> str:
    """Hash adapter weights/config only, excluding optimizer and Trainer state."""

    names = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    files = [path / name for name in names if (path / name).is_file()]
    if len(files) < 2 or not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter files are incomplete: {path}")
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.name):
        digest.update(item.name.encode("utf-8"))
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def save_final_adapter(output_dir: Path, model: Any, tokenizer: Any) -> tuple[Path, str]:
    """Save into an unused child even when Trainer already created its output parent."""

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter"
    if adapter_path.exists():
        raise FileExistsError("refusing to overwrite an existing final GRPO adapter")
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    return adapter_path, adapter_artifact_sha256(adapter_path)


def _finite_history_metrics(history: Sequence[Mapping[str, object]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in history:
        for name, value in row.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError(f"Trainer metric {name} is NaN or infinite")
            values[str(name)].append(number)
    if not values.get("loss"):
        raise RuntimeError("GRPO training did not report a finite loss")
    if not values.get("kl"):
        raise RuntimeError("GRPO training did not report a finite KL")
    return dict(values)


def _peak_process_ram(process: Any) -> int:
    memory = process.memory_info()
    for name in ("peak_wset", "peak_rss", "rss"):
        value = getattr(memory, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("process memory information lacks a positive RAM measurement")


def _external_process_evidence(seed: int, runtime_paths: GrpoRuntimePaths) -> dict[str, object]:
    """Prove process-start hash seeding and the complete pre-launch contract."""

    if seed != FROZEN_PROCESS_SEED:
        raise ValueError("GRPO seed differs from the frozen process seed")
    launch_evidence = validate_deterministic_process_environment(
        runtime_paths, "counted_grpo_process_entry"
    )
    expected_seed = str(seed)
    current_hash = hash(_PYTHON_HASH_PROBE)
    completed = subprocess.run(
        [
            str(runtime_paths.python_executable),
            "-S",
            "-c",
            f"print(hash({_PYTHON_HASH_PROBE!r}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=frozen_process_environment(runtime_paths),
    )
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if values != [str(current_hash)]:
        raise RuntimeError(
            "PYTHONHASHSEED environment does not match the running interpreter hash seed"
        )
    return {
        "python_hash_randomization": True,
        "python_hash_seed": expected_seed,
        "python_hash_probe_sha256": canonical_sha256(
            {
                "probe_text_sha256": hashlib.sha256(_PYTHON_HASH_PROBE.encode("utf-8")).hexdigest(),
                "probe_value": str(current_hash),
            }
        ),
        "cublas_workspace_config": FROZEN_CUBLAS_WORKSPACE_CONFIG,
        "deterministic_process_environment": launch_evidence,
        "deterministic_process_contract": deterministic_process_contract(runtime_paths).evidence(),
    }


def _stock_full_determinism_source_evidence(
    transformers: Any,
    *,
    expected_source_sha256: str = FROZEN_FULL_DETERMINISM_SOURCE_SHA256,
    expected_module: str = FROZEN_FULL_DETERMINISM_MODULE,
    expected_qualname: str = FROZEN_FULL_DETERMINISM_QUALNAME,
) -> dict[str, object]:
    """Pin the stock helper and its exact idempotent environment writes."""

    if (
        expected_module != FROZEN_TRANSFORMERS_DETERMINISM_MODULE
        or expected_qualname != FROZEN_TRANSFORMERS_DETERMINISM_QUALNAME
    ):
        raise RuntimeError("Transformers full-determinism callable identity differs")
    evidence = transformers_determinism_source_evidence(
        transformers, expected_function_sha256=expected_source_sha256
    )
    return {
        **evidence,
        "source_sha256": evidence["function_source_sha256"],
        "prelaunch_environment": dict(FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT),
        "expected_active_environment": dict(FROZEN_TRANSFORMERS_DETERMINISTIC_ENVIRONMENT),
    }


def _require_transformers_environment(
    runtime_paths: GrpoRuntimePaths, stage: str, torch: Any
) -> dict[str, object]:
    """Require every Transformers-written field and strict Torch state."""

    return validate_deterministic_process_environment(
        runtime_paths,
        stage,
        torch_module=torch,
        require_strict=True,
    )


def _full_determinism_transition_evidence(
    source_evidence: Mapping[str, object],
    *,
    before_initialization: Mapping[str, object],
    after_arguments: Mapping[str, object],
    after_trainer: Mapping[str, object],
) -> dict[str, object]:
    """Bind source-pinned initialization to a no-transition environment."""

    after_arguments_idempotence = assert_idempotent_deterministic_initialization(
        before_initialization, after_arguments
    )
    after_trainer_idempotence = assert_idempotent_deterministic_initialization(
        before_initialization, after_trainer
    )
    value: dict[str, object] = {
        **source_evidence,
        "after_grpo_arguments": after_arguments_idempotence,
        "after_trainer_construction": after_trainer_idempotence,
        "environment_transition_occurred": False,
        "environment_restoration_required": False,
    }
    value["transition_sha256"] = canonical_sha256(value)
    return value


def _software_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in FROZEN_SOFTWARE_VERSIONS}
    if versions != FROZEN_SOFTWARE_VERSIONS:
        raise RuntimeError(f"GRPO software versions differ from the frozen stack: {versions}")
    return versions


def _runtime_environment_evidence(
    modules: Mapping[str, Any], runtime_paths: GrpoRuntimePaths
) -> dict[str, object]:
    """Require exact runtime-visible Python and package versions."""

    python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    expected_python = {
        "implementation": FROZEN_PYTHON_IMPLEMENTATION,
        "version": FROZEN_PYTHON_VERSION,
    }
    if python != expected_python:
        raise RuntimeError(f"Python runtime differs from the frozen contract: {python}")
    executable = Path(sys.executable).resolve(strict=True)
    if not same_canonical_path(executable, runtime_paths.python_executable):
        raise RuntimeError("running interpreter differs from the runtime-path contract")
    versions = _software_versions()
    imported_versions: dict[str, str] = {}
    for name in ("datasets", "numpy", "peft", "psutil", "torch", "transformers", "trl"):
        module = modules[name]
        value = str(getattr(module, "__version__", ""))
        if not value:
            raise RuntimeError(f"imported {name} does not expose a runtime version")
        imported_versions[name] = value
        if value != versions[name]:
            raise RuntimeError(f"imported {name} version differs from package metadata")
    return {
        "python": python,
        "sys_executable": str(executable),
        "sys_executable_file_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "software_versions": versions,
        "imported_software_versions": imported_versions,
    }


def _strict_determinism(torch: Any) -> bool:
    return bool(torch.are_deterministic_algorithms_enabled()) and not bool(
        torch.is_deterministic_algorithms_warn_only_enabled()
    )


def _seed_everything(modules: Mapping[str, Any], seed: int) -> None:
    """Seed every frozen RNG surface and enable strict deterministic execution."""

    if seed != FROZEN_PROCESS_SEED:
        raise ValueError("GRPO seed differs from the frozen process seed")
    numpy = modules["numpy"]
    torch = modules["torch"]
    transformers = modules["transformers"]
    random.seed(seed)
    numpy.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not _strict_determinism(torch):
        raise RuntimeError("strict deterministic mode was not enabled before GRPO")


def _synchronize_cuda(torch: Any) -> None:
    torch.cuda.synchronize(0)


def _prepare_cuda_resources(torch: Any) -> None:
    _synchronize_cuda(torch)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    _synchronize_cuda(torch)
    torch.cuda.reset_peak_memory_stats(0)


def _cleanup_cuda_resources(torch: Any) -> None:
    _synchronize_cuda(torch)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    _synchronize_cuda(torch)


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing GRPO artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _assert_checkpoint_set(output_dir: Path) -> dict[str, str]:
    checkpoint_paths = sorted(
        (path for path in (output_dir / "trainer_state").glob("checkpoint-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    actual_steps: list[int] = []
    hashes: dict[str, str] = {}
    for path in checkpoint_paths:
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError as error:
            raise RuntimeError(f"unexpected checkpoint directory: {path.name}") from error
        actual_steps.append(step)
        hashes[str(step)] = adapter_artifact_sha256(path)
    if tuple(actual_steps) != FROZEN_CHECKPOINT_STEPS:
        raise RuntimeError(
            f"checkpoint steps differ: expected {FROZEN_CHECKPOINT_STEPS}, got {actual_steps}"
        )
    return hashes


def _runtime_modules() -> dict[str, Any]:
    return {
        name: importlib.import_module(name)
        for name in ("datasets", "numpy", "peft", "psutil", "torch", "transformers", "trl")
    }


def _load_quantized_base(
    model_path: Path, config: VerifierGRPOConfig, modules: Mapping[str, Any]
) -> tuple[Any, Any, float]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    _synchronize_cuda(torch)
    started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=config.base_model.trust_remote_code,
        revision=config.base_model.revision,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quantization_type,
        bnb_4bit_use_double_quant=config.quantization.double_quantization,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        revision=config.base_model.revision,
        local_files_only=True,
        trust_remote_code=config.base_model.trust_remote_code,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if not bool(getattr(model, "is_loaded_in_4bit", False)):
        raise RuntimeError("base model did not load in frozen 4-bit mode")
    assert_cuda_only_model(model)
    model.config.use_cache = False
    _synchronize_cuda(torch)
    return model, tokenizer, time.perf_counter() - started


def _prepare_runtime(
    config: VerifierGRPOConfig,
    model_path: Path,
    modules: Mapping[str, Any],
) -> tuple[Any, Any, Any, float]:
    torch = modules["torch"]
    peft = modules["peft"]
    model, tokenizer, load_seconds = _load_quantized_base(model_path, config, modules)
    model = peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.memory_and_reproducibility.gradient_checkpointing,
    )
    lora_config = peft.LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=list(config.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    if config.quantization.compute_dtype != "float16" or torch.float16 is None:
        raise ValueError("runtime compute dtype differs from frozen FP16")
    return model, tokenizer, lora_config, load_seconds


def _validate_cuda(
    torch: Any, runtime_paths: GrpoRuntimePaths, *, stage: str
) -> ChildCudaComputeEvidence:
    """Validate the child through direct PyTorch CUDA computation, never NVML."""

    return collect_child_cuda_compute_evidence(
        torch,
        runtime_paths,
        stage=stage,
    )


def _repeat_row(group: RuntimePromptGroup) -> list[dict[str, object]]:
    return [group.policy_row() for _ in range(COMPLETIONS_PER_GROUP)]


def _run_grpo_impl(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    arm: Arm,
    variant_id: str,
    mode: RuntimeMode,
    output_dir: Path,
    raw_evidence_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Run one frozen compatibility smoke or one complete 64-step GRPO arm."""

    if mode not in {"compatibility", "train"}:
        raise ValueError("runtime mode must be compatibility or train")
    assert_source_path(runtime_paths, config_path, "GRPO configuration")
    assert_source_path(runtime_paths, manifest_path, "GRPO schedule manifest")
    assert_artifact_path(runtime_paths, packet_path, "GRPO schedule packet")
    assert_artifact_path(runtime_paths, output_dir, "GRPO adapter/checkpoint")
    assert_artifact_path(runtime_paths, raw_evidence_path, "raw GRPO evidence")
    assert_artifact_path(runtime_paths, summary_path, "GRPO summary")
    model_path = runtime_paths.model_snapshot_root
    external_process = _external_process_evidence(FROZEN_PROCESS_SEED, runtime_paths)
    config = load_grpo_config(config_path)
    if config.grpo.seed != FROZEN_PROCESS_SEED:
        raise ValueError("configured GRPO seed differs from the frozen process seed")
    if config.base_model.revision != BASE_REVISION:
        raise ValueError("base revision differs from the frozen Qwen checkpoint")
    _assert_offline_model_snapshot(model_path, config)
    if output_dir.exists() or raw_evidence_path.exists() or summary_path.exists():
        raise FileExistsError("GRPO run outputs must start from unused paths")
    schedule = load_runtime_schedule(packet_path, manifest_path, expected_arm=arm)
    if any(group.prompt_tokens > config.grpo.max_prompt_length for group in schedule.groups):
        raise ValueError("scheduled prompt exceeds the frozen 512-token maximum")
    if mode == "compatibility":
        update_groups, generation_only_group = select_compatibility_groups(schedule)
        expected_groups = (*update_groups, generation_only_group)
    else:
        update_groups = schedule.groups
        generation_only_group = None
        expected_groups = schedule.groups

    deterministic_stages = [
        validate_deterministic_process_environment(runtime_paths, "before_transformers_import")
    ]
    modules = _runtime_modules()
    torch = modules["torch"]
    transformers = modules["transformers"]
    trl = modules["trl"]
    peft = modules["peft"]
    datasets = modules["datasets"]
    psutil = modules["psutil"]
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_transformers_import",
            torch_module=torch,
        )
    )
    runtime_environment = _runtime_environment_evidence(modules, runtime_paths)
    full_determinism_source = _stock_full_determinism_source_evidence(transformers)
    _seed_everything(modules, config.grpo.seed)
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_initial_deterministic_setup",
            torch_module=torch,
            require_strict=True,
        )
    )
    child_cuda = _validate_cuda(torch, runtime_paths, stage="counted_grpo")
    cuda = child_cuda.stable_evidence()
    child_cuda_resource = child_cuda.resource_payload()
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_cuda_initialization",
            torch_module=torch,
            require_strict=True,
        )
    )
    reference_audit = validate_installed_reference_contract(Path(sysconfig.get_paths()["purelib"]))

    _prepare_cuda_resources(torch)
    process = psutil.Process()
    run_started = time.perf_counter()
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "before_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    model, tokenizer, lora_config, model_load_seconds = _prepare_runtime(
        config, model_path, modules
    )
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    chat_template_sha256 = hashlib.sha256(
        (tokenizer.chat_template or "").encode("utf-8")
    ).hexdigest()
    reward_callback = VerifierRewardCallback(
        expected_groups,
        completion_token_counter=_completion_token_counter(tokenizer),
    )
    argument_values = frozen_grpo_argument_values(
        config, variant_id=variant_id, output_dir=output_dir, mode=mode
    )
    before_full_determinism = deterministic_stages[-1]
    arguments = trl.GRPOConfig(**argument_values)
    environment_after_arguments = _require_transformers_environment(
        runtime_paths, "after_grpo_config_full_determinism", torch
    )
    deterministic_stages.append(environment_after_arguments)
    assert_frozen_grpo_arguments(
        arguments,
        config,
        variant_id=variant_id,
        output_dir=output_dir,
        mode=mode,
    )
    warning_contract = TopPWarningOnlyGenerationContract(
        torch_module=torch,
        generation_owner=transformers.GenerationMixin,
        top_p_call=transformers.generation.logits_process.TopPLogitsWarper.__call__,
    )
    boundary_counts: Counter[str] = Counter()
    boundary_environment_sha256s: set[str] = set()

    def validate_boundary(stage: str) -> None:
        evidence = _require_transformers_environment(runtime_paths, stage, torch)
        boundary_counts[stage] += 1
        boundary_environment_sha256s.add(str(evidence["environment_sha256"]))

    audited_trainer_type = make_truncation_aware_grpo_trainer(
        trl.GRPOTrainer,
        generation_scope_factory=warning_contract.install,
    )
    trainer_type = make_environment_guarded_trainer(audited_trainer_type, validate_boundary)
    callbacks: list[object] = [
        make_environment_validation_callback(transformers.TrainerCallback, validate_boundary)
    ]
    if mode == "train":
        callbacks.append(make_exact_checkpoint_callback(transformers.TrainerCallback))
    train_dataset = datasets.Dataset.from_list([group.policy_row() for group in update_groups])
    trainer = trainer_type(
        model=model,
        reward_funcs=reward_callback,
        args=arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=lora_config,
    )
    environment_after_trainer = _require_transformers_environment(
        runtime_paths, "after_grpo_trainer_construction", torch
    )
    deterministic_stages.append(environment_after_trainer)
    full_determinism_transition = _full_determinism_transition_evidence(
        full_determinism_source,
        before_initialization=before_full_determinism,
        after_arguments=environment_after_arguments,
        after_trainer=environment_after_trainer,
    )
    if trainer.ref_model is not None:
        raise RuntimeError(
            "official PEFT reference path unexpectedly created a second reference model"
        )
    if not hasattr(trainer.model, "disable_adapter"):
        raise RuntimeError("GRPO trainer did not create the required PEFT policy adapter")
    warning_contract.bind_state_probe(partial(model_adapter_state, trainer.model))
    trainability = assert_only_lora_trainable(trainer.model)
    assert_cuda_only_model(trainer.model)
    dropout_module_count = assert_dropout_disabled(trainer.model, torch)
    base_tensor_state_before = _capture_base_tensor_state(trainer.model, torch)
    base_hash_before = _base_reference_hash(trainer.model, tokenizer, expected_groups[0], torch)

    _synchronize_cuda(torch)
    training_started = time.perf_counter()
    validate_boundary("before_training")
    trainer.train()
    validate_boundary("after_training")
    _synchronize_cuda(torch)
    training_seconds = time.perf_counter() - training_started
    if int(trainer.state.global_step) != len(update_groups):
        raise RuntimeError("GRPO optimizer-step count differs from the frozen schedule")
    assert_cuda_only_model(trainer.model)
    assert_frozen_base_has_no_gradients(trainer.model)
    if generation_only_group is not None:
        trainer._generate_and_score_completions(_repeat_row(generation_only_group))
    metrics = _finite_history_metrics(
        cast(Sequence[Mapping[str, object]], trainer.state.log_history)
    )
    reward_summary = summarize_reward_records(
        reward_callback.records,
        expected_groups,
        require_nonzero_variance=mode == "compatibility",
    )
    expected_completion_count = (
        COMPATIBILITY_COMPLETIONS if mode == "compatibility" else FULL_COMPLETIONS
    )
    if reward_summary["completions"] != expected_completion_count:
        raise RuntimeError("GRPO completion count differs from the frozen mode")
    warning_evidence = warning_contract.evidence()
    expected_generation_calls = len(expected_groups)
    if warning_evidence["generation_calls"] != expected_generation_calls:
        raise RuntimeError("warning-only generation call count differs from the schedule")
    base_tensor_state_after = _capture_base_tensor_state(trainer.model, torch)
    _assert_base_tensor_state_unchanged(
        base_tensor_state_before,
        base_tensor_state_after,
        stage="GRPO compatibility/train execution",
    )
    base_hash_after = _base_reference_hash(trainer.model, tokenizer, expected_groups[0], torch)
    if base_hash_after != base_hash_before:
        raise RuntimeError("adapter-disabled base output changed during GRPO")

    adapter_path, adapter_sha256 = save_final_adapter(output_dir, trainer.model, tokenizer)
    checkpoint_hashes: dict[str, str] = {}
    if mode == "train":
        checkpoint_hashes = _assert_checkpoint_set(output_dir)
    del model
    del trainer
    _cleanup_cuda_resources(torch)

    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "before_adapter_reload_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    reloaded_base, reloaded_tokenizer, reload_base_seconds = _load_quantized_base(
        model_path, config, modules
    )
    _synchronize_cuda(torch)
    reload_started = time.perf_counter()
    reloaded = peft.PeftModel.from_pretrained(
        reloaded_base,
        str(adapter_path),
        local_files_only=True,
        is_trainable=False,
    )
    deterministic_stages.append(
        validate_deterministic_process_environment(
            runtime_paths,
            "after_adapter_reload_model_loading",
            torch_module=torch,
            require_strict=True,
        )
    )
    _synchronize_cuda(torch)
    adapter_reload_seconds = time.perf_counter() - reload_started
    assert_cuda_only_model(reloaded)
    if not any("lora_" in name for name, _ in reloaded.named_parameters()):
        raise RuntimeError("offline adapter reload did not restore LoRA parameters")
    if any(parameter.requires_grad for parameter in reloaded.parameters()):
        raise RuntimeError("offline validation reload unexpectedly left trainable parameters")
    base_tensor_state_after_reload = _capture_base_tensor_state(reloaded, torch)
    _assert_base_tensor_state_unchanged(
        base_tensor_state_before,
        base_tensor_state_after_reload,
        stage="offline adapter reload",
    )
    reloaded_base_hash = _base_reference_hash(
        reloaded, reloaded_tokenizer, expected_groups[0], torch
    )
    if reloaded_base_hash != base_hash_before:
        raise RuntimeError("adapter-disabled hash differs after offline adapter reload")
    _synchronize_cuda(torch)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    cuda_memory_before_final_cleanup = current_cuda_memory_evidence(torch)
    if peak_reserved >= MAX_RESERVED_VRAM_BYTES:
        raise RuntimeError(f"peak reserved VRAM exceeds the 9.6 GiB gate: {peak_reserved} bytes")
    del reloaded
    del reloaded_base
    _cleanup_cuda_resources(torch)
    post_cleanup_allocated = int(torch.cuda.memory_allocated(0))
    post_cleanup_reserved = int(torch.cuda.memory_reserved(0))
    cuda_memory_after_final_cleanup = current_cuda_memory_evidence(torch)
    final_deterministic_environment = validate_deterministic_process_environment(
        runtime_paths,
        "process_result_publication",
        torch_module=torch,
        require_strict=True,
    )
    deterministic_stages.append(final_deterministic_environment)
    deterministic_environment_evidence = {
        "stages": deterministic_stages,
        "boundary_validation_counts": dict(sorted(boundary_counts.items())),
        "boundary_environment_sha256s": sorted(boundary_environment_sha256s),
        "environment_mutation_observed": False,
    }
    deterministic_environment_evidence["evidence_sha256"] = canonical_sha256(
        deterministic_environment_evidence
    )
    raw_evidence: dict[str, object] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_id": RUNTIME_ID,
        "arm": arm,
        "variant_id": variant_id,
        "mode": mode,
        "schedule_packet_sha256": schedule.packet_sha256,
        "records": [record.raw_record() for record in reward_callback.records],
    }
    _write_json_new(raw_evidence_path, raw_evidence)
    _synchronize_cuda(torch)
    total_seconds = time.perf_counter() - run_started
    summary: dict[str, object] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_id": RUNTIME_ID,
        "run_kind": "compatibility_smoke" if mode == "compatibility" else "counted_training",
        "arm": arm,
        "variant_id": variant_id,
        "beta": config.variant(variant_id).beta,
        "config_sha256": config.config_sha256,
        "execution_sha256": config.execution_sha256(variant_id),
        "reward_configuration_sha256": reward_configuration_sha256(),
        "reward_implementation_sha256": reward_implementation_sha256(),
        "base_model_id": config.base_model.model_id,
        "base_revision": config.base_model.revision,
        "schedule_packet_sha256": schedule.packet_sha256,
        "schedule_manifest_sha256": schedule.manifest_sha256,
        "optimizer_steps": len(update_groups),
        "generation_only_groups": int(generation_only_group is not None),
        "reward": reward_summary,
        "reported_metrics": {
            key: {
                "count": len(values),
                "mean": fmean(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for key, values in sorted(metrics.items())
        },
        "only_lora_trainable": True,
        "trainable_parameters": trainability.trainable_parameters,
        "total_parameters": trainability.total_parameters,
        "reference_policy": "untouched_base_with_active_adapter_disabled",
        "reference_implementation_audit": asdict(reference_audit),
        "external_process_contract": external_process,
        "runtime_paths": runtime_paths.evidence(),
        "runtime_environment": runtime_environment,
        "stock_full_determinism_transition": full_determinism_transition,
        "deterministic_process_environment": deterministic_environment_evidence,
        "warning_only_generation_contract": warning_evidence,
        "second_reference_model_created": False,
        "cpu_offload": False,
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_hash_after_reload": reloaded_base_hash,
        "base_tensor_state_before": base_tensor_state_before,
        "base_tensor_state_after": base_tensor_state_after,
        "base_tensor_state_after_reload": base_tensor_state_after_reload,
        "base_tensor_bytes_unchanged": True,
        "base_restoration_passed": True,
        "adapter_sha256": adapter_sha256,
        "adapter_directory_sha256": directory_sha256(adapter_path),
        "adapter_size_bytes": sum(
            path.stat().st_size for path in adapter_path.rglob("*") if path.is_file()
        ),
        "output_dir_disk_bytes": sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        ),
        "checkpoint_adapter_sha256": checkpoint_hashes,
        "offline_adapter_reload_passed": True,
        "chat_template_sha256": chat_template_sha256,
        "dropout_disabled": True,
        "dropout_module_count": dropout_module_count,
        "software_versions": runtime_environment["software_versions"],
        **cuda,
        "child_cuda_probe_resource": child_cuda_resource,
        "cuda_memory_before_final_cleanup": cuda_memory_before_final_cleanup,
        "cuda_memory_after_final_cleanup": cuda_memory_after_final_cleanup,
        "model_load_seconds": model_load_seconds,
        "reload_base_seconds": reload_base_seconds,
        "adapter_reload_seconds": adapter_reload_seconds,
        "training_seconds": training_seconds,
        "completion_tokens_per_second": (
            cast(int, reward_summary["completion_tokens"]) / training_seconds
        ),
        "total_runtime_seconds": total_seconds,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "reserved_vram_gate_bytes": MAX_RESERVED_VRAM_BYTES,
        "reserved_vram_gate_passed": True,
        "post_cleanup_allocated_vram_bytes": post_cleanup_allocated,
        "post_cleanup_reserved_vram_bytes": post_cleanup_reserved,
        "peak_process_ram_bytes": _peak_process_ram(process),
        "raw_evidence_sha256": hashlib.sha256(raw_evidence_path.read_bytes()).hexdigest(),
        "prompts_completions_or_answers_in_summary": False,
        "gate_passed": True,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json_new(summary_path, summary)
    return summary


def run_grpo(
    *,
    runtime_paths: GrpoRuntimePaths,
    config_path: Path,
    packet_path: Path,
    manifest_path: Path,
    arm: Arm,
    variant_id: str,
    mode: RuntimeMode,
    output_dir: Path,
    raw_evidence_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Validate immutable roots before and after one GRPO process."""

    validate_runtime_paths(runtime_paths)
    try:
        return _run_grpo_impl(
            runtime_paths=runtime_paths,
            config_path=config_path,
            packet_path=packet_path,
            manifest_path=manifest_path,
            arm=arm,
            variant_id=variant_id,
            mode=mode,
            output_dir=output_dir,
            raw_evidence_path=raw_evidence_path,
            summary_path=summary_path,
        )
    finally:
        validate_runtime_paths(runtime_paths)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-paths", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("generic_control", "targeted"), required=True)
    parser.add_argument("--variant", choices=("G1", "G2"), required=True)
    parser.add_argument("--mode", choices=("compatibility", "train"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    runtime_paths = load_runtime_paths(args.runtime_paths)
    summary = run_grpo(
        runtime_paths=runtime_paths,
        config_path=args.config,
        packet_path=args.packet,
        manifest_path=args.manifest,
        arm=cast(Arm, args.arm),
        variant_id=cast(str, args.variant),
        mode=cast(RuntimeMode, args.mode),
        output_dir=args.output_dir,
        raw_evidence_path=args.raw_evidence,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
