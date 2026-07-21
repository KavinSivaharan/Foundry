"""Deterministic prompt-only schedules for the frozen two-arm GRPO experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from foundry.training.base_replay import ANCHOR_ID, REPLAY_CORPUS_ID, REPLAY_FORMAT_ID
from foundry.training.config import (
    ASSISTANT_ONLY_V3_SYSTEM_PROMPT,
    ASSISTANT_ONLY_V3_USER_SUFFIX,
    SFT_USER_PREFIX,
    canonical_sha256,
)
from foundry.training.qlora import file_sha256

Arm = Literal["generic_control", "targeted"]
Family = Literal[
    "multi_step_bookkeeping_or_omission",
    "rate_ratio_percentage_or_average",
    "constraint_distribution_or_discrete_reasoning",
]
ReplaySection = Literal["arithmetic", "format", "instruction"]
SourceKind = Literal["synthetic", "base_replay"]
RetentionKind = Literal["numeric_terminal", "exact_text", "json_exact"]

SCHEDULE_ID = "foundry-verifier-grpo-schedule-v1"
SCHEDULE_SCHEMA_VERSION = 1
SCHEDULE_SEED = 20260720
GROUPS_PER_ARM = 64
SYNTHETIC_GROUPS_PER_ARM = 52
REPLAY_GROUPS_PER_ARM = 12
COMPLETIONS_PER_GROUP = 4
COMPLETIONS_PER_ARM = GROUPS_PER_ARM * COMPLETIONS_PER_GROUP
PROMPT_TOKEN_PARITY_MAXIMUM = 0.01
REPLAY_POSITIONS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
FAMILY_ORDER: tuple[Family, ...] = (
    "multi_step_bookkeeping_or_omission",
    "rate_ratio_percentage_or_average",
    "constraint_distribution_or_discrete_reasoning",
)
REPLAY_SECTION_ORDER: tuple[ReplaySection, ...] = ("arithmetic", "format", "instruction")
SYNTHETIC_QUOTAS: Mapping[Arm, Mapping[Family, int]] = {
    "generic_control": {
        "multi_step_bookkeeping_or_omission": 17,
        "rate_ratio_percentage_or_average": 17,
        "constraint_distribution_or_discrete_reasoning": 18,
    },
    "targeted": {
        "multi_step_bookkeeping_or_omission": 29,
        "rate_ratio_percentage_or_average": 12,
        "constraint_distribution_or_discrete_reasoning": 11,
    },
}


@dataclass(frozen=True)
class PromptMessage:
    """One immutable message supplied before generation."""

    role: Literal["system", "user"]
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


PromptTokenCounter = Callable[[tuple[PromptMessage, ...]], int]


class ChatTemplateTokenizer(Protocol):
    """Minimum tokenizer surface needed for exact prompt-token accounting."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class SyntheticPrompt:
    """Prompt-only projection of one frozen accepted synthetic training row."""

    synthetic_id: str
    arm: Arm
    family: Family
    messages: tuple[PromptMessage, ...]
    prompt_sha256: str
    canonical_final_answer: str
    mode: str
    difficulty: str
    output_contract_enabled: bool
    verifier_metadata_sha256: str
    provenance_sha256: str


@dataclass(frozen=True)
class SyntheticRewardMetadata:
    """Hidden deterministic reward fields for one synthetic prompt."""

    canonical_final_answer: str
    family: Family
    mode: str
    difficulty: str
    output_contract_enabled: bool
    verifier_metadata_sha256: str
    provenance_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "reward_kind": "synthetic",
            "canonical_final_answer": self.canonical_final_answer,
            "family": self.family,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "output_contract_enabled": self.output_contract_enabled,
            "verifier_metadata_sha256": self.verifier_metadata_sha256,
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True)
class ReplayPrompt:
    """Prompt-only projection of one frozen correct untouched-base replay row."""

    replay_id: str
    section: ReplaySection
    skill: str
    messages: tuple[PromptMessage, ...]
    prompt_sha256: str
    kind: RetentionKind
    expected: str
    scorer_sha256: str
    provenance_sha256: str


@dataclass(frozen=True)
class ReplayRewardMetadata:
    """Hidden objective-scorer fields for one untouched-base replay prompt."""

    section: ReplaySection
    skill: str
    kind: RetentionKind
    expected: str
    scorer_sha256: str
    provenance_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "reward_kind": "base_replay",
            "section": self.section,
            "skill": self.skill,
            "kind": self.kind,
            "expected": self.expected,
            "scorer_sha256": self.scorer_sha256,
            "provenance_sha256": self.provenance_sha256,
        }


RewardMetadata = SyntheticRewardMetadata | ReplayRewardMetadata


@dataclass(frozen=True)
class ScheduledPromptGroup:
    """One frozen prompt group; model outputs never participate in scheduling."""

    group_id: str
    arm: Arm
    position: int
    source_kind: SourceKind
    source_id: str
    category: str
    messages: tuple[PromptMessage, ...]
    prompt_sha256: str
    prompt_tokens: int
    reward_metadata: RewardMetadata
    completions_per_group: int = COMPLETIONS_PER_GROUP

    def manifest_record(self) -> dict[str, object]:
        """Return the content-free schedule record."""

        return {
            "group_id": self.group_id,
            "position": self.position,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "category": self.category,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tokens": self.prompt_tokens,
            "completions_per_group": self.completions_per_group,
        }

    def packet_record(self) -> dict[str, object]:
        """Return prompt text plus hidden reward metadata, with no assistant target or output."""

        value = self.manifest_record()
        value["messages"] = [message.as_dict() for message in self.messages]
        value["reward_metadata"] = self.reward_metadata.as_dict()
        return value


@dataclass(frozen=True)
class GRPOArmSchedule:
    """One immutable 64-group arm schedule and its content-free manifest."""

    arm: Arm
    groups: tuple[ScheduledPromptGroup, ...]
    manifest: dict[str, object]
    packet_sha256: str

    def prompt_packet(self) -> dict[str, object]:
        """Build the local prompt-bearing packet supplied to the runtime."""

        packet: dict[str, object] = {
            "schema_version": SCHEDULE_SCHEMA_VERSION,
            "schedule_id": SCHEDULE_ID,
            "seed": SCHEDULE_SEED,
            "arm": self.arm,
            "groups": [group.packet_record() for group in self.groups],
        }
        if canonical_sha256(packet) != self.packet_sha256:
            raise RuntimeError("GRPO prompt packet differs from its frozen hash")
        return packet


@dataclass(frozen=True)
class GRPOScheduleBundle:
    """The paired schedules plus a content-free parity summary."""

    generic_control: GRPOArmSchedule
    targeted: GRPOArmSchedule
    summary: dict[str, object]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_sha256(messages: tuple[PromptMessage, ...]) -> str:
    return canonical_sha256([message.as_dict() for message in messages])


def _string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _load_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a string-keyed JSON object")
    return cast(dict[str, object], value)


def _synthetic_messages(question: str) -> tuple[PromptMessage, ...]:
    return (
        PromptMessage("system", ASSISTANT_ONLY_V3_SYSTEM_PROMPT),
        PromptMessage("user", f"{SFT_USER_PREFIX}{question}{ASSISTANT_ONLY_V3_USER_SUFFIX}"),
    )


def load_synthetic_prompts(path: Path, arm: Arm) -> tuple[SyntheticPrompt, ...]:
    """Load only trusted prompt fields from one accepted training JSONL artifact."""

    if arm not in SYNTHETIC_QUOTAS:
        raise ValueError("unknown GRPO arm")
    prompts: list[SyntheticPrompt] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value: object = json.loads(line)
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"synthetic row {line_number} must be a string-keyed object")
        row = cast(dict[str, object], value)
        synthetic_id = _string(row, "synthetic_id")
        family_value = _string(row, "family")
        if family_value not in FAMILY_ORDER:
            raise ValueError(f"synthetic row {line_number} has an unknown family")
        family = cast(Family, family_value)
        if (
            row.get("group") != arm
            or row.get("future_split") != "training"
            or row.get("final_decision") != "accepted"
            or row.get("primary_verifier_success") is not True
            or row.get("independent_verifier_success") is not True
            or row.get("verifier_agreement") is not True
        ):
            raise ValueError(f"synthetic row {line_number} is not a verified accepted training row")
        question = _string(row, "rendered_question")
        canonical_final_answer = _string(row, "canonical_final_answer")
        mode = _string(row, "mode")
        difficulty = _string(row, "difficulty")
        output_contract_enabled = row.get("output_contract_enabled")
        if not isinstance(output_contract_enabled, bool):
            raise ValueError("output_contract_enabled must be boolean")
        primary_evidence_sha256 = _string(row, "primary_evidence_sha256")
        independent_evidence_sha256 = _string(row, "independent_evidence_sha256")
        verifier_metadata_sha256 = canonical_sha256(
            {
                "primary_evidence_sha256": primary_evidence_sha256,
                "independent_evidence_sha256": independent_evidence_sha256,
                "primary_verifier_success": True,
                "independent_verifier_success": True,
                "verifier_agreement": True,
            }
        )
        provenance_sha256 = canonical_sha256(
            {
                "synthetic_id": synthetic_id,
                "group": arm,
                "future_split": "training",
                "family": family,
                "mode": mode,
                "difficulty": difficulty,
                "output_contract_enabled": output_contract_enabled,
                "latent_program_sha256": _string(row, "latent_program_sha256"),
                "semantic_ir_sha256": _string(row, "semantic_ir_sha256"),
                "rendered_text_sha256": _string(row, "rendered_text_sha256"),
                "primary_evidence_sha256": primary_evidence_sha256,
                "independent_evidence_sha256": independent_evidence_sha256,
            }
        )
        messages = _synthetic_messages(question)
        prompt_hash = _prompt_sha256(messages)
        if synthetic_id in seen_ids or prompt_hash in seen_hashes:
            raise ValueError("synthetic prompt IDs and rendered prompts must be unique")
        seen_ids.add(synthetic_id)
        seen_hashes.add(prompt_hash)
        prompts.append(
            SyntheticPrompt(
                synthetic_id,
                arm,
                family,
                messages,
                prompt_hash,
                canonical_final_answer,
                mode,
                difficulty,
                output_contract_enabled,
                verifier_metadata_sha256,
                provenance_sha256,
            )
        )
    if not prompts:
        raise ValueError("synthetic prompt source is empty")
    return tuple(prompts)


def load_replay_prompts(
    raw_path: Path, manifest_path: Path, anchor_path: Path
) -> tuple[ReplayPrompt, ...]:
    """Validate the frozen replay corpus, then discard every assistant target."""

    raw = _load_json_object(raw_path)
    manifest = _load_json_object(manifest_path)
    anchor = _load_json_object(anchor_path)
    declared_manifest_hash = manifest.get("manifest_sha256")
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("manifest_sha256", None)
    if declared_manifest_hash != canonical_sha256(manifest_without_hash):
        raise ValueError("replay manifest self-hash differs")
    if (
        raw.get("schema_version") != 1
        or raw.get("replay_corpus_id") != REPLAY_CORPUS_ID
        or raw.get("replay_format_id") != REPLAY_FORMAT_ID
        or manifest.get("replay_corpus_id") != REPLAY_CORPUS_ID
        or manifest.get("replay_format_id") != REPLAY_FORMAT_ID
        or manifest.get("gate_passed") is not True
        or manifest.get("prompts_or_outputs_in_manifest") is not False
        or manifest.get("replay_corpus_sha256") != canonical_sha256(raw)
        or manifest.get("raw_replay_packet_sha256") != file_sha256(raw_path)
        or anchor.get("anchor_id") != ANCHOR_ID
        or manifest.get("anchor_sha256") != canonical_sha256(anchor)
        or raw.get("source_anchor_sha256") != canonical_sha256(anchor)
    ):
        raise ValueError("replay corpus identity, gate, or packet hash differs")
    raw_items = raw.get("items")
    manifest_items = manifest.get("items")
    if not isinstance(raw_items, list) or not isinstance(manifest_items, list):
        raise ValueError("replay items must be arrays")
    if len(raw_items) != len(manifest_items) or manifest.get("total") != len(raw_items):
        raise ValueError("replay raw and manifest item counts differ")
    anchor_items = anchor.get("items")
    system_prompt = anchor.get("system_prompt")
    if not isinstance(anchor_items, list) or not isinstance(system_prompt, str):
        raise ValueError("replay anchor fields differ")
    anchor_by_id: dict[str, dict[str, object]] = {}
    for anchor_value in anchor_items:
        if not isinstance(anchor_value, dict):
            raise ValueError("replay anchor items must be objects")
        anchor_row = cast(dict[str, object], anchor_value)
        anchor_id = _string(anchor_row, "id")
        if anchor_id in anchor_by_id:
            raise ValueError("replay anchor IDs must be unique")
        anchor_by_id[anchor_id] = anchor_row
    prompts: list[ReplayPrompt] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for raw_value, manifest_value in zip(raw_items, manifest_items, strict=True):
        if not isinstance(raw_value, dict) or not isinstance(manifest_value, dict):
            raise ValueError("replay items must be objects")
        raw_row = cast(dict[str, object], raw_value)
        manifest_row = cast(dict[str, object], manifest_value)
        replay_id = _string(raw_row, "id")
        section_value = _string(raw_row, "section")
        if section_value not in REPLAY_SECTION_ORDER:
            raise ValueError("replay item has an unknown section")
        section = cast(ReplaySection, section_value)
        skill = _string(raw_row, "skill")
        matched_anchor = anchor_by_id.get(replay_id)
        if matched_anchor is None:
            raise ValueError("replay item does not exist in the frozen anchor")
        kind_value = _string(matched_anchor, "kind")
        if kind_value not in {"numeric_terminal", "exact_text", "json_exact"}:
            raise ValueError("replay anchor item has an unknown scorer kind")
        kind = cast(RetentionKind, kind_value)
        expected = _string(matched_anchor, "expected")
        response = _string(raw_row, "assistant_response")
        response_hash = _string(raw_row, "assistant_response_sha256")
        if response_hash != _sha256_text(response):
            raise ValueError("replay assistant-response hash differs")
        if (
            manifest_row.get("id") != replay_id
            or manifest_row.get("section") != section
            or manifest_row.get("skill") != skill
            or manifest_row.get("base_output_sha256") != response_hash
            or matched_anchor.get("section") != section
            or matched_anchor.get("skill") != skill
            or matched_anchor.get("prompt") != raw_row.get("prompt")
            or raw_row.get("system_prompt") != system_prompt
        ):
            raise ValueError("replay raw and manifest item identities differ")
        messages = (
            PromptMessage("system", _string(raw_row, "system_prompt")),
            PromptMessage("user", _string(raw_row, "prompt")),
        )
        prompt_hash = _prompt_sha256(messages)
        if replay_id in seen_ids or prompt_hash in seen_hashes:
            raise ValueError("replay IDs and prompts must be unique")
        seen_ids.add(replay_id)
        seen_hashes.add(prompt_hash)
        scorer_sha256 = canonical_sha256(
            {
                "scorer": "foundry.training.retention.score_response",
                "item": {
                    "id": replay_id,
                    "section": section,
                    "skill": skill,
                    "kind": kind,
                    "prompt_sha256": _sha256_text(_string(raw_row, "prompt")),
                    "expected": expected,
                },
            }
        )
        provenance_sha256 = canonical_sha256(
            {
                "anchor_sha256": canonical_sha256(anchor),
                "replay_corpus_sha256": canonical_sha256(raw),
                "replay_id": replay_id,
                "base_output_sha256": response_hash,
            }
        )
        prompts.append(
            ReplayPrompt(
                replay_id,
                section,
                skill,
                messages,
                prompt_hash,
                kind,
                expected,
                scorer_sha256,
                provenance_sha256,
            )
        )
    return tuple(prompts)


def count_transformers_prompt_tokens(
    tokenizer: ChatTemplateTokenizer, messages: tuple[PromptMessage, ...]
) -> int:
    """Count the exact chat-template prompt tokens before any completion exists."""

    tokenized = tokenizer.apply_chat_template(
        [message.as_dict() for message in messages],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(tokenized, "numel"):
        count = int(tokenized.numel())
    elif isinstance(tokenized, Sequence) and not isinstance(tokenized, str | bytes):
        count = len(tokenized)
    else:
        raise TypeError("tokenizer returned an unsupported prompt-token container")
    if count <= 0:
        raise ValueError("prompt-token count must be positive")
    return count


def _selection_rank(*parts: str) -> str:
    return canonical_sha256({"seed": SCHEDULE_SEED, "parts": list(parts)})


def _checked_token_count(counter: PromptTokenCounter, messages: tuple[PromptMessage, ...]) -> int:
    value = counter(messages)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("prompt-token counter must return a positive integer")
    return value


def _subset_options(
    prompts: Sequence[SyntheticPrompt], count: int, token_counts: Mapping[str, int]
) -> dict[int, tuple[SyntheticPrompt, ...]]:
    ordered = sorted(
        prompts,
        key=lambda item: (
            _selection_rank(item.arm, item.family, item.synthetic_id),
            item.synthetic_id,
        ),
    )
    states: list[dict[int, tuple[SyntheticPrompt, ...]]] = [dict() for _ in range(count + 1)]
    states[0][0] = ()
    for prompt in ordered:
        tokens = token_counts[prompt.synthetic_id]
        for size in range(count, 0, -1):
            previous = states[size - 1]
            current = states[size]
            for subtotal, chosen in previous.items():
                total = subtotal + tokens
                if total not in current:
                    current[total] = (*chosen, prompt)
    if not states[count]:
        raise ValueError("synthetic family has insufficient prompts for its frozen quota")
    return states[count]


def _arm_options(
    arm: Arm,
    prompts: Sequence[SyntheticPrompt],
    token_counts: Mapping[str, int],
) -> dict[int, tuple[SyntheticPrompt, ...]]:
    by_family = {
        family: [item for item in prompts if item.family == family] for family in FAMILY_ORDER
    }
    combined: dict[int, tuple[SyntheticPrompt, ...]] = {0: ()}
    for family in FAMILY_ORDER:
        quota = SYNTHETIC_QUOTAS[arm][family]
        family_options = _subset_options(by_family[family], quota, token_counts)
        next_combined: dict[int, tuple[SyntheticPrompt, ...]] = {}
        for left_total, left_items in combined.items():
            for right_total, right_items in family_options.items():
                total = left_total + right_total
                if total not in next_combined:
                    next_combined[total] = (*left_items, *right_items)
        combined = next_combined
    return combined


def _choose_token_matched_subsets(
    generic_options: Mapping[int, tuple[SyntheticPrompt, ...]],
    targeted_options: Mapping[int, tuple[SyntheticPrompt, ...]],
    replay_tokens: int,
) -> tuple[tuple[SyntheticPrompt, ...], tuple[SyntheticPrompt, ...], int, int]:
    generic_totals = sorted(generic_options)
    targeted_totals = sorted(targeted_options)
    best: tuple[int, int, int] | None = None
    left = 0
    for targeted_total in targeted_totals:
        while left + 1 < len(generic_totals) and abs(
            generic_totals[left + 1] - targeted_total
        ) <= abs(generic_totals[left] - targeted_total):
            left += 1
        generic_total = generic_totals[left]
        key = (abs(generic_total - targeted_total), generic_total, targeted_total)
        if best is None or key < best:
            best = key
    if best is None:
        raise ValueError("no paired synthetic prompt selection exists")
    _, generic_synthetic_tokens, targeted_synthetic_tokens = best
    generic_total = generic_synthetic_tokens + replay_tokens
    targeted_total = targeted_synthetic_tokens + replay_tokens
    parity = abs(generic_total - targeted_total) / max(generic_total, targeted_total)
    if parity > PROMPT_TOKEN_PARITY_MAXIMUM:
        raise ValueError(f"prompt-token parity exceeds 1%: {parity:.8f}")
    return (
        generic_options[generic_synthetic_tokens],
        targeted_options[targeted_synthetic_tokens],
        generic_total,
        targeted_total,
    )


def _selected_replay(prompts: Sequence[ReplayPrompt]) -> tuple[ReplayPrompt, ...]:
    selected_by_section: dict[ReplaySection, list[ReplayPrompt]] = {}
    for section in REPLAY_SECTION_ORDER:
        eligible = [item for item in prompts if item.section == section]
        eligible.sort(
            key=lambda item: (
                _selection_rank("replay", section, item.replay_id),
                item.replay_id,
            )
        )
        if len(eligible) < 4:
            raise ValueError(f"replay section {section} has fewer than four prompts")
        selected_by_section[section] = eligible[:4]
    return tuple(
        selected_by_section[section][index]
        for index in range(4)
        for section in REPLAY_SECTION_ORDER
    )


def _balanced_family_order(arm: Arm) -> tuple[Family, ...]:
    quotas = dict(SYNTHETIC_QUOTAS[arm])
    used = {family: 0 for family in FAMILY_ORDER}
    order: list[Family] = []
    while len(order) < SYNTHETIC_GROUPS_PER_ARM:
        available = [family for family in FAMILY_ORDER if used[family] < quotas[family]]
        family = min(
            available,
            key=lambda item: (used[item] / quotas[item], FAMILY_ORDER.index(item)),
        )
        used[family] += 1
        order.append(family)
    return tuple(order)


def _make_arm_schedule(
    arm: Arm,
    synthetic: Sequence[SyntheticPrompt],
    replay: Sequence[ReplayPrompt],
    token_counts: Mapping[str, int],
) -> GRPOArmSchedule:
    by_family: dict[Family, list[SyntheticPrompt]] = {
        family: sorted(
            (item for item in synthetic if item.family == family),
            key=lambda item: (
                _selection_rank("selected", arm, item.synthetic_id),
                item.synthetic_id,
            ),
        )
        for family in FAMILY_ORDER
    }
    synthetic_order = _balanced_family_order(arm)
    family_indexes = {family: 0 for family in FAMILY_ORDER}
    replay_index = 0
    groups: list[ScheduledPromptGroup] = []
    for position in range(1, GROUPS_PER_ARM + 1):
        category: str
        if position in REPLAY_POSITIONS:
            replay_source = replay[replay_index]
            replay_index += 1
            source_kind: SourceKind = "base_replay"
            source_id = replay_source.replay_id
            category = replay_source.section
            messages = replay_source.messages
            prompt_hash = replay_source.prompt_sha256
            tokens = token_counts[f"replay:{replay_source.replay_id}"]
            reward_metadata: RewardMetadata = ReplayRewardMetadata(
                section=replay_source.section,
                skill=replay_source.skill,
                kind=replay_source.kind,
                expected=replay_source.expected,
                scorer_sha256=replay_source.scorer_sha256,
                provenance_sha256=replay_source.provenance_sha256,
            )
        else:
            family = synthetic_order[position - 1 - replay_index]
            synthetic_source = by_family[family][family_indexes[family]]
            family_indexes[family] += 1
            source_kind = "synthetic"
            source_id = synthetic_source.synthetic_id
            category = synthetic_source.family
            messages = synthetic_source.messages
            prompt_hash = synthetic_source.prompt_sha256
            tokens = token_counts[f"{arm}:{synthetic_source.synthetic_id}"]
            reward_metadata = SyntheticRewardMetadata(
                canonical_final_answer=synthetic_source.canonical_final_answer,
                family=synthetic_source.family,
                mode=synthetic_source.mode,
                difficulty=synthetic_source.difficulty,
                output_contract_enabled=synthetic_source.output_contract_enabled,
                verifier_metadata_sha256=synthetic_source.verifier_metadata_sha256,
                provenance_sha256=synthetic_source.provenance_sha256,
            )
        groups.append(
            ScheduledPromptGroup(
                group_id=f"grpo-{arm}-g{position:03d}",
                arm=arm,
                position=position,
                source_kind=source_kind,
                source_id=source_id,
                category=category,
                messages=messages,
                prompt_sha256=prompt_hash,
                prompt_tokens=tokens,
                reward_metadata=reward_metadata,
            )
        )
    packet: dict[str, object] = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "schedule_id": SCHEDULE_ID,
        "seed": SCHEDULE_SEED,
        "arm": arm,
        "groups": [group.packet_record() for group in groups],
    }
    packet_hash = canonical_sha256(packet)
    family_counts = Counter(
        cast(Family, group.category) for group in groups if group.source_kind == "synthetic"
    )
    replay_counts = Counter(
        cast(ReplaySection, group.category)
        for group in groups
        if group.source_kind == "base_replay"
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "schedule_id": SCHEDULE_ID,
        "seed": SCHEDULE_SEED,
        "arm": arm,
        "groups_per_arm": GROUPS_PER_ARM,
        "synthetic_groups": SYNTHETIC_GROUPS_PER_ARM,
        "replay_groups": REPLAY_GROUPS_PER_ARM,
        "completions_per_group": COMPLETIONS_PER_GROUP,
        "total_completions": COMPLETIONS_PER_ARM,
        "family_counts": {family: family_counts[family] for family in FAMILY_ORDER},
        "replay_section_counts": {
            section: replay_counts[section] for section in REPLAY_SECTION_ORDER
        },
        "replay_positions": list(REPLAY_POSITIONS),
        "prompt_token_total": sum(group.prompt_tokens for group in groups),
        "prompt_packet_sha256": packet_hash,
        "groups": [group.manifest_record() for group in groups],
        "groups_fixed_before_generation": True,
        "retry_until_success": False,
        "replacement_groups": False,
        "model_output_selection": False,
        "model_outputs_observed_during_scheduling": False,
        "prompts_or_outputs_in_manifest": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return GRPOArmSchedule(arm, tuple(groups), manifest, packet_hash)


def build_grpo_schedules(
    *,
    generic_prompts: Sequence[SyntheticPrompt],
    targeted_prompts: Sequence[SyntheticPrompt],
    replay_prompts: Sequence[ReplayPrompt],
    prompt_token_counter: PromptTokenCounter,
) -> GRPOScheduleBundle:
    """Build paired fixed schedules without observing or selecting model outputs."""

    if any(item.arm != "generic_control" for item in generic_prompts) or any(
        item.arm != "targeted" for item in targeted_prompts
    ):
        raise ValueError("synthetic prompt supplied to the wrong arm")
    generic_ids = {item.synthetic_id for item in generic_prompts}
    targeted_ids = {item.synthetic_id for item in targeted_prompts}
    generic_hashes = {item.prompt_sha256 for item in generic_prompts}
    targeted_hashes = {item.prompt_sha256 for item in targeted_prompts}
    if (
        len(generic_ids) != len(generic_prompts)
        or len(targeted_ids) != len(targeted_prompts)
        or generic_ids & targeted_ids
        or generic_hashes & targeted_hashes
    ):
        raise ValueError("synthetic source pools must be unique and disjoint across arms")
    replay = _selected_replay(replay_prompts)
    token_counts: dict[str, int] = {}
    generic_by_id = {item.synthetic_id: item for item in generic_prompts}
    for generic_prompt in generic_prompts:
        token_counts[generic_prompt.synthetic_id] = _checked_token_count(
            prompt_token_counter, generic_prompt.messages
        )
        token_counts[f"generic_control:{generic_prompt.synthetic_id}"] = token_counts[
            generic_prompt.synthetic_id
        ]
    targeted_family_counts: dict[str, int] = {}
    for targeted_prompt in targeted_prompts:
        count = _checked_token_count(prompt_token_counter, targeted_prompt.messages)
        token_counts[targeted_prompt.synthetic_id] = count
        token_counts[f"targeted:{targeted_prompt.synthetic_id}"] = count
        targeted_family_counts[targeted_prompt.synthetic_id] = count
    for replay_prompt in replay:
        token_counts[f"replay:{replay_prompt.replay_id}"] = _checked_token_count(
            prompt_token_counter, replay_prompt.messages
        )
    generic_family_counts = {item_id: token_counts[item_id] for item_id in generic_by_id}
    generic_options = _arm_options("generic_control", generic_prompts, generic_family_counts)
    targeted_options = _arm_options("targeted", targeted_prompts, targeted_family_counts)
    replay_tokens = sum(token_counts[f"replay:{item.replay_id}"] for item in replay)
    generic_selected, targeted_selected, generic_total, targeted_total = (
        _choose_token_matched_subsets(generic_options, targeted_options, replay_tokens)
    )
    generic_schedule = _make_arm_schedule("generic_control", generic_selected, replay, token_counts)
    targeted_schedule = _make_arm_schedule("targeted", targeted_selected, replay, token_counts)
    if generic_schedule.manifest["prompt_token_total"] != generic_total or (
        targeted_schedule.manifest["prompt_token_total"] != targeted_total
    ):
        raise RuntimeError("schedule prompt-token totals differ from paired selection")
    generic_groups = generic_schedule.groups
    targeted_groups = targeted_schedule.groups
    replay_generic = [group for group in generic_groups if group.source_kind == "base_replay"]
    replay_targeted = [group for group in targeted_groups if group.source_kind == "base_replay"]
    if [group.position for group in replay_generic] != list(REPLAY_POSITIONS) or [
        group.source_id for group in replay_generic
    ] != [group.source_id for group in replay_targeted]:
        raise RuntimeError("paired replay positions or source order differ")
    all_group_ids = [group.group_id for group in (*generic_groups, *targeted_groups)]
    if len(set(all_group_ids)) != GROUPS_PER_ARM * 2:
        raise RuntimeError("GRPO group IDs must be globally unique")
    parity = abs(generic_total - targeted_total) / max(generic_total, targeted_total)
    summary: dict[str, object] = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "schedule_id": SCHEDULE_ID,
        "seed": SCHEDULE_SEED,
        "arms": ["generic_control", "targeted"],
        "groups_per_arm": GROUPS_PER_ARM,
        "completions_per_group": COMPLETIONS_PER_GROUP,
        "completions_per_arm": COMPLETIONS_PER_ARM,
        "synthetic_groups_per_arm": SYNTHETIC_GROUPS_PER_ARM,
        "replay_groups_per_arm": REPLAY_GROUPS_PER_ARM,
        "replay_positions": list(REPLAY_POSITIONS),
        "generic_prompt_tokens": generic_total,
        "targeted_prompt_tokens": targeted_total,
        "prompt_token_absolute_difference": abs(generic_total - targeted_total),
        "prompt_token_parity_ratio": parity,
        "prompt_token_parity_maximum": PROMPT_TOKEN_PARITY_MAXIMUM,
        "prompt_token_parity_passed": parity <= PROMPT_TOKEN_PARITY_MAXIMUM,
        "generic_manifest_sha256": generic_schedule.manifest["manifest_sha256"],
        "targeted_manifest_sha256": targeted_schedule.manifest["manifest_sha256"],
        "replay_source_ids_sha256": canonical_sha256([group.source_id for group in replay_generic]),
        "same_replay_ids_positions_and_order": True,
        "synthetic_source_ids_disjoint": True,
        "groups_fixed_before_generation": True,
        "retry_until_success": False,
        "replacement_groups": False,
        "model_output_selection": False,
        "model_outputs_observed_during_scheduling": False,
        "prompts_or_outputs_in_summary": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return GRPOScheduleBundle(generic_schedule, targeted_schedule, summary)


def _serialize_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _git_ignored(repository_root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "-C", str(repository_root), "check-ignore", "--quiet", str(relative)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _write_new_or_identical(path: Path, value: object) -> None:
    payload = _serialize_json(value)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite different schedule artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def write_grpo_schedule_bundle(
    bundle: GRPOScheduleBundle,
    *,
    repository_root: Path,
    generic_packet_path: Path,
    targeted_packet_path: Path,
    generic_manifest_path: Path,
    targeted_manifest_path: Path,
    summary_path: Path,
) -> None:
    """Write prompt packets only to ignored paths and tracked-safe manifests elsewhere."""

    for packet_path in (generic_packet_path, targeted_packet_path):
        if not _git_ignored(repository_root, packet_path):
            raise ValueError(f"prompt-bearing GRPO packet path is not Git ignored: {packet_path}")
    _write_new_or_identical(generic_packet_path, bundle.generic_control.prompt_packet())
    _write_new_or_identical(targeted_packet_path, bundle.targeted.prompt_packet())
    _write_new_or_identical(generic_manifest_path, bundle.generic_control.manifest)
    _write_new_or_identical(targeted_manifest_path, bundle.targeted.manifest)
    _write_new_or_identical(summary_path, bundle.summary)
