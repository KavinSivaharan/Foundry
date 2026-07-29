"""Matched prompt-only schedules for Milestone 14A L3 verifier GRPO.

The tracked manifests produced by this module contain IDs, hashes, counts, and
token accounting only.  Prompt text and trusted reward metadata are written
only to the repository's ignored ``results/raw`` tree.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from foundry.training.config import (
    ASSISTANT_ONLY_V3_SYSTEM_PROMPT,
    ASSISTANT_ONLY_V3_USER_SUFFIX,
    SFT_USER_PREFIX,
    canonical_sha256,
)
from foundry.training.grpo_schedule import (
    PromptMessage as _PromptMessage,
)
from foundry.training.grpo_schedule import (
    ReplayPrompt,
    count_transformers_prompt_tokens,
    load_replay_prompts,
)
from foundry.training.qlora import file_sha256

PromptMessage = _PromptMessage
Arm = Literal["generic", "targeted"]
Family = Literal[
    "multi_step_bookkeeping_or_omission",
    "rate_ratio_percentage_or_average",
    "constraint_distribution_or_discrete_reasoning",
]
SourceKind = Literal["task", "base_replay"]

SCHEDULE_ID = "foundry-l3-verifier-grpo-v1"
SCHEMA_VERSION = 1
SEED = 20260720
GROUPS_PER_ARM = 32
TASK_GROUPS_PER_ARM = 24
REPLAY_GROUPS_PER_ARM = 8
COMPLETIONS_PER_GROUP = 4
COMPLETIONS_PER_ARM = 128
OPTIMIZER_STEPS = 32
CHECKPOINT_STEPS = (8, 16, 32)
PROMPT_TOKEN_PARITY_MAXIMUM = 0.01
REPLAY_POSITIONS = (4, 8, 12, 16, 20, 24, 28, 32)
DATASET_SHA256 = "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31"
TRAINING_FILE_SHA256: Mapping[Arm, str] = {
    "generic": "f3c59ffbea616486537ba911df5c0374227385f56471a64ebbed6fe01b922214",
    "targeted": "3322a9de2e56d24e170129112a8973dd45e5c6575969da52e5cac44f23cea215",
}
FAMILY_ORDER: tuple[Family, ...] = (
    "multi_step_bookkeeping_or_omission",
    "rate_ratio_percentage_or_average",
    "constraint_distribution_or_discrete_reasoning",
)
TASK_QUOTAS: Mapping[Arm, Mapping[Family, int]] = {
    "generic": {
        "multi_step_bookkeeping_or_omission": 8,
        "rate_ratio_percentage_or_average": 8,
        "constraint_distribution_or_discrete_reasoning": 8,
    },
    "targeted": {
        "multi_step_bookkeeping_or_omission": 13,
        "rate_ratio_percentage_or_average": 6,
        "constraint_distribution_or_discrete_reasoning": 5,
    },
}
REPLAY_SECTION_QUOTAS = {"arithmetic": 3, "format": 2, "instruction": 3}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")

PromptTokenCounter = Callable[[tuple[PromptMessage, ...]], int]


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


def _prompt_sha256(messages: tuple[PromptMessage, ...]) -> str:
    return canonical_sha256([message.as_dict() for message in messages])


def _selection_rank(*parts: str) -> str:
    return canonical_sha256({"schedule_id": SCHEDULE_ID, "seed": SEED, "parts": list(parts)})


def _task_messages(question: str) -> tuple[PromptMessage, ...]:
    return (
        PromptMessage("system", ASSISTANT_ONLY_V3_SYSTEM_PROMPT),
        PromptMessage("user", f"{SFT_USER_PREFIX}{question}{ASSISTANT_ONLY_V3_USER_SUFFIX}"),
    )


@dataclass(frozen=True)
class TaskPrompt:
    """One vetted human-written task prompt and trusted answer-side metadata."""

    source_id: str
    arm: Arm
    family: Family
    messages: tuple[PromptMessage, ...]
    prompt_sha256: str
    canonical_answer: str
    answer_type: str
    difficulty: str
    verifier_metadata_sha256: str
    provenance_sha256: str


@dataclass(frozen=True)
class ScheduledGroup:
    """One immutable prompt group."""

    group_id: str
    arm: Arm
    position: int
    source_kind: SourceKind
    source_id: str
    category: str
    messages: tuple[PromptMessage, ...]
    prompt_sha256: str
    prompt_tokens: int
    reward_metadata: Mapping[str, object]

    def manifest_record(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "position": self.position,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "category": self.category,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tokens": self.prompt_tokens,
            "completions_per_group": COMPLETIONS_PER_GROUP,
        }

    def packet_record(self) -> dict[str, object]:
        return {
            **self.manifest_record(),
            "messages": [message.as_dict() for message in self.messages],
            "reward_metadata": dict(self.reward_metadata),
        }


@dataclass(frozen=True)
class ArmSchedule:
    """One 32-group schedule plus its content-free manifest."""

    arm: Arm
    groups: tuple[ScheduledGroup, ...]
    packet: dict[str, object]
    manifest: dict[str, object]

    @property
    def packet_sha256(self) -> str:
        return canonical_sha256(self.packet)


@dataclass(frozen=True)
class ScheduleBundle:
    """The paired arm schedules and shared content-free projections."""

    generic: ArmSchedule
    targeted: ArmSchedule
    shared_replay: dict[str, object]
    summary: dict[str, object]


def load_task_prompts(path: Path, arm: Arm) -> tuple[TaskPrompt, ...]:
    """Load one exact Phase 2 training split without assistant targets in prompts."""

    if arm not in TASK_QUOTAS:
        raise ValueError("task arm is outside the frozen pair")
    if file_sha256(path) != TRAINING_FILE_SHA256[arm]:
        raise ValueError(f"{arm} training artifact hash differs")
    prompts: list[TaskPrompt] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"task row {line_number} must be an object")
        row = cast(dict[str, object], value)
        if row.get("arm") != arm or row.get("split") != "training":
            raise ValueError(f"task row {line_number} is outside the frozen arm/split")
        family_value = _require_text(row.get("family"), "family")
        if family_value not in FAMILY_ORDER:
            raise ValueError(f"task row {line_number} has an unknown family")
        family = cast(Family, family_value)
        source_id = _require_text(row.get("source_id"), "source_id")
        question = _require_text(row.get("question"), "question")
        question_sha256 = _require_sha256(row.get("question_sha256"), "question_sha256")
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != question_sha256:
            raise ValueError(f"task row {line_number} question hash differs")
        canonical_answer = _require_text(row.get("canonical_answer"), "canonical_answer")
        answer_type = _require_text(row.get("answer_type"), "answer_type")
        difficulty = _require_text(row.get("difficulty"), "difficulty")
        program_sha256 = _require_sha256(row.get("program_sha256"), "program_sha256")
        messages = _task_messages(question)
        prompt_hash = _prompt_sha256(messages)
        if source_id in ids or prompt_hash in hashes:
            raise ValueError("task IDs and model-visible prompts must be unique")
        ids.add(source_id)
        hashes.add(prompt_hash)
        verifier_metadata_sha256 = canonical_sha256(
            {
                "program_sha256": program_sha256,
                "canonical_answer": canonical_answer,
                "answer_type": answer_type,
            }
        )
        provenance_sha256 = canonical_sha256(
            {
                "dataset_sha256": DATASET_SHA256,
                "arm": arm,
                "split": "training",
                "source_id": source_id,
                "family": family,
                "question_sha256": question_sha256,
                "question_normalized_sha256": _require_sha256(
                    row.get("question_normalized_sha256"), "question_normalized_sha256"
                ),
                "assistant_completion_sha256": _require_sha256(
                    row.get("assistant_completion_sha256"), "assistant_completion_sha256"
                ),
            }
        )
        prompts.append(
            TaskPrompt(
                source_id=source_id,
                arm=arm,
                family=family,
                messages=messages,
                prompt_sha256=prompt_hash,
                canonical_answer=canonical_answer,
                answer_type=answer_type,
                difficulty=difficulty,
                verifier_metadata_sha256=verifier_metadata_sha256,
                provenance_sha256=provenance_sha256,
            )
        )
    if len(prompts) != 180:
        raise ValueError(f"{arm} training pool must contain exactly 180 records")
    return tuple(prompts)


def _checked_token_count(counter: PromptTokenCounter, messages: tuple[PromptMessage, ...]) -> int:
    value = counter(messages)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("prompt token counter must return a positive integer")
    if value > 512:
        raise ValueError("prompt exceeds the frozen 512-token maximum")
    return value


def _subset_options(
    prompts: Sequence[TaskPrompt],
    count: int,
    token_counts: Mapping[str, int],
) -> dict[int, tuple[TaskPrompt, ...]]:
    ordered = sorted(
        prompts,
        key=lambda item: (
            _selection_rank("task", item.arm, item.family, item.source_id),
            item.source_id,
        ),
    )
    states: list[dict[int, tuple[TaskPrompt, ...]]] = [dict() for _ in range(count + 1)]
    states[0][0] = ()
    for prompt in ordered:
        tokens = token_counts[prompt.source_id]
        for size in range(count, 0, -1):
            for subtotal, chosen in states[size - 1].items():
                total = subtotal + tokens
                if total not in states[size]:
                    states[size][total] = (*chosen, prompt)
    if not states[count]:
        raise ValueError("task family has insufficient rows for its quota")
    return states[count]


def _arm_options(
    arm: Arm,
    prompts: Sequence[TaskPrompt],
    token_counts: Mapping[str, int],
) -> dict[int, tuple[TaskPrompt, ...]]:
    combined: dict[int, tuple[TaskPrompt, ...]] = {0: ()}
    for family in FAMILY_ORDER:
        family_options = _subset_options(
            [prompt for prompt in prompts if prompt.family == family],
            TASK_QUOTAS[arm][family],
            token_counts,
        )
        next_combined: dict[int, tuple[TaskPrompt, ...]] = {}
        for left_total, left_items in combined.items():
            for right_total, right_items in family_options.items():
                total = left_total + right_total
                if total not in next_combined:
                    next_combined[total] = (*left_items, *right_items)
        combined = next_combined
    return combined


def _choose_matched_subsets(
    generic_options: Mapping[int, tuple[TaskPrompt, ...]],
    targeted_options: Mapping[int, tuple[TaskPrompt, ...]],
    replay_tokens: int,
) -> tuple[tuple[TaskPrompt, ...], tuple[TaskPrompt, ...], int, int]:
    generic_totals = sorted(generic_options)
    targeted_totals = sorted(targeted_options)
    if not generic_totals or not targeted_totals:
        raise ValueError("paired task options are empty")
    best: tuple[int, int, int] | None = None
    generic_index = 0
    for targeted_total in targeted_totals:
        while generic_index + 1 < len(generic_totals) and abs(
            generic_totals[generic_index + 1] - targeted_total
        ) <= abs(generic_totals[generic_index] - targeted_total):
            generic_index += 1
        generic_total = generic_totals[generic_index]
        candidate = (abs(generic_total - targeted_total), generic_total, targeted_total)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("paired task selection produced no candidate")
    _, generic_task_tokens, targeted_task_tokens = best
    generic_total = generic_task_tokens + replay_tokens
    targeted_total = targeted_task_tokens + replay_tokens
    parity = abs(generic_total - targeted_total) / max(generic_total, targeted_total)
    if parity > PROMPT_TOKEN_PARITY_MAXIMUM:
        raise ValueError(f"prompt-token parity exceeds 1%: {parity:.8f}")
    return (
        generic_options[generic_task_tokens],
        targeted_options[targeted_task_tokens],
        generic_total,
        targeted_total,
    )


def _select_replay(prompts: Sequence[ReplayPrompt]) -> tuple[ReplayPrompt, ...]:
    selected: list[ReplayPrompt] = []
    for section in ("arithmetic", "format", "instruction"):
        eligible = sorted(
            (prompt for prompt in prompts if prompt.section == section),
            key=lambda item: (
                _selection_rank("replay", section, item.replay_id),
                item.replay_id,
            ),
        )
        quota = REPLAY_SECTION_QUOTAS[section]
        if len(eligible) < quota:
            raise ValueError(f"replay section {section} has fewer than {quota} eligible rows")
        selected.extend(eligible[:quota])
    selected.sort(
        key=lambda item: (
            _selection_rank("replay-order", item.section, item.replay_id),
            item.replay_id,
        )
    )
    if len(selected) != REPLAY_GROUPS_PER_ARM:
        raise RuntimeError("shared replay selection differs from eight rows")
    return tuple(selected)


def _balanced_family_order(arm: Arm) -> tuple[Family, ...]:
    quotas = dict(TASK_QUOTAS[arm])
    used = {family: 0 for family in FAMILY_ORDER}
    order: list[Family] = []
    while len(order) < TASK_GROUPS_PER_ARM:
        available = [family for family in FAMILY_ORDER if used[family] < quotas[family]]
        family = min(
            available,
            key=lambda item: (used[item] / quotas[item], FAMILY_ORDER.index(item)),
        )
        used[family] += 1
        order.append(family)
    return tuple(order)


def _make_schedule(
    arm: Arm,
    tasks: Sequence[TaskPrompt],
    replay: Sequence[ReplayPrompt],
    token_counts: Mapping[str, int],
) -> ArmSchedule:
    by_family: dict[Family, list[TaskPrompt]] = {
        family: sorted(
            (task for task in tasks if task.family == family),
            key=lambda item: (
                _selection_rank("selected", arm, item.source_id),
                item.source_id,
            ),
        )
        for family in FAMILY_ORDER
    }
    task_order = _balanced_family_order(arm)
    family_indexes = {family: 0 for family in FAMILY_ORDER}
    replay_index = 0
    task_index = 0
    groups: list[ScheduledGroup] = []
    for position in range(1, GROUPS_PER_ARM + 1):
        if position in REPLAY_POSITIONS:
            replay_source = replay[replay_index]
            replay_index += 1
            group = ScheduledGroup(
                group_id=f"l3-grpo-{arm}-g{position:03d}",
                arm=arm,
                position=position,
                source_kind="base_replay",
                source_id=replay_source.replay_id,
                category=replay_source.section,
                messages=replay_source.messages,
                prompt_sha256=replay_source.prompt_sha256,
                prompt_tokens=token_counts[f"replay:{replay_source.replay_id}"],
                reward_metadata={
                    "reward_kind": "base_replay",
                    "section": replay_source.section,
                    "skill": replay_source.skill,
                    "kind": replay_source.kind,
                    "expected": replay_source.expected,
                    "scorer_sha256": replay_source.scorer_sha256,
                    "provenance_sha256": replay_source.provenance_sha256,
                },
            )
        else:
            family = task_order[task_index]
            task_index += 1
            task_source = by_family[family][family_indexes[family]]
            family_indexes[family] += 1
            group = ScheduledGroup(
                group_id=f"l3-grpo-{arm}-g{position:03d}",
                arm=arm,
                position=position,
                source_kind="task",
                source_id=task_source.source_id,
                category=task_source.family,
                messages=task_source.messages,
                prompt_sha256=task_source.prompt_sha256,
                prompt_tokens=token_counts[f"{arm}:{task_source.source_id}"],
                reward_metadata={
                    "reward_kind": "task",
                    "canonical_answer": task_source.canonical_answer,
                    "answer_type": task_source.answer_type,
                    "family": task_source.family,
                    "difficulty": task_source.difficulty,
                    "verifier_metadata_sha256": task_source.verifier_metadata_sha256,
                    "provenance_sha256": task_source.provenance_sha256,
                },
            )
        groups.append(group)
    packet: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "schedule_id": SCHEDULE_ID,
        "seed": SEED,
        "arm": arm,
        "groups": [group.packet_record() for group in groups],
    }
    family_counts = Counter(
        cast(Family, group.category) for group in groups if group.source_kind == "task"
    )
    replay_counts = Counter(
        group.category for group in groups if group.source_kind == "base_replay"
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "schedule_id": SCHEDULE_ID,
        "seed": SEED,
        "arm": arm,
        "groups_per_arm": GROUPS_PER_ARM,
        "task_groups": TASK_GROUPS_PER_ARM,
        "replay_groups": REPLAY_GROUPS_PER_ARM,
        "optimizer_steps": OPTIMIZER_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "completions_per_group": COMPLETIONS_PER_GROUP,
        "total_completions": COMPLETIONS_PER_ARM,
        "task_family_counts": {family: family_counts[family] for family in FAMILY_ORDER},
        "replay_section_counts": dict(sorted(replay_counts.items())),
        "replay_positions": list(REPLAY_POSITIONS),
        "prompt_token_total": sum(group.prompt_tokens for group in groups),
        "prompt_packet_sha256": canonical_sha256(packet),
        "groups": [group.manifest_record() for group in groups],
        "human_written_task_source": True,
        "gsm1k_prompt_use": 0,
        "holdout_v2_prompt_use": 0,
        "sealed_content_use": 0,
        "groups_fixed_before_generation": True,
        "model_outputs_observed_during_scheduling": False,
        "prompts_or_answers_in_manifest": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return ArmSchedule(arm=arm, groups=tuple(groups), packet=packet, manifest=manifest)


def build_schedules(
    *,
    generic_prompts: Sequence[TaskPrompt],
    targeted_prompts: Sequence[TaskPrompt],
    replay_prompts: Sequence[ReplayPrompt],
    prompt_token_counter: PromptTokenCounter,
) -> ScheduleBundle:
    """Build the exact paired schedule without observing any model output."""

    if any(prompt.arm != "generic" for prompt in generic_prompts) or any(
        prompt.arm != "targeted" for prompt in targeted_prompts
    ):
        raise ValueError("task prompt supplied to the wrong arm")
    generic_ids = {prompt.source_id for prompt in generic_prompts}
    targeted_ids = {prompt.source_id for prompt in targeted_prompts}
    if (
        len(generic_ids) != len(generic_prompts)
        or len(targeted_ids) != len(targeted_prompts)
        or generic_ids & targeted_ids
    ):
        raise ValueError("task source IDs must be unique and disjoint across arms")
    replay = _select_replay(replay_prompts)
    token_counts: dict[str, int] = {}
    generic_tokens: dict[str, int] = {}
    targeted_tokens: dict[str, int] = {}
    for generic_prompt in generic_prompts:
        count = _checked_token_count(prompt_token_counter, generic_prompt.messages)
        token_counts[f"generic:{generic_prompt.source_id}"] = count
        generic_tokens[generic_prompt.source_id] = count
    for targeted_prompt in targeted_prompts:
        count = _checked_token_count(prompt_token_counter, targeted_prompt.messages)
        token_counts[f"targeted:{targeted_prompt.source_id}"] = count
        targeted_tokens[targeted_prompt.source_id] = count
    for replay_prompt in replay:
        token_counts[f"replay:{replay_prompt.replay_id}"] = _checked_token_count(
            prompt_token_counter, replay_prompt.messages
        )
    replay_tokens = sum(token_counts[f"replay:{prompt.replay_id}"] for prompt in replay)
    selected_generic, selected_targeted, generic_total, targeted_total = _choose_matched_subsets(
        _arm_options("generic", generic_prompts, generic_tokens),
        _arm_options("targeted", targeted_prompts, targeted_tokens),
        replay_tokens,
    )
    generic = _make_schedule("generic", selected_generic, replay, token_counts)
    targeted = _make_schedule("targeted", selected_targeted, replay, token_counts)
    generic_replay = [group for group in generic.groups if group.source_kind == "base_replay"]
    targeted_replay = [group for group in targeted.groups if group.source_kind == "base_replay"]
    if [
        (group.position, group.source_id, group.reward_metadata["scorer_sha256"])
        for group in generic_replay
    ] != [
        (group.position, group.source_id, group.reward_metadata["scorer_sha256"])
        for group in targeted_replay
    ]:
        raise RuntimeError("shared replay IDs, positions, order, or scorers differ")
    if generic.manifest["prompt_token_total"] != generic_total or (
        targeted.manifest["prompt_token_total"] != targeted_total
    ):
        raise RuntimeError("paired prompt-token totals differ from selection")
    parity = abs(generic_total - targeted_total) / max(generic_total, targeted_total)
    shared_replay: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "schedule_id": SCHEDULE_ID,
        "selection": "deterministic_frozen_base_behavior_replay",
        "source_corpus_file_sha256": (
            "a9f25258d23f05a785dfea9f8ae0e05a246b52c9798a0d10e683fdc4e01a87f6"
        ),
        "record_count": REPLAY_GROUPS_PER_ARM,
        "positions": list(REPLAY_POSITIONS),
        "records": [
            {
                "position": group.position,
                "source_id": group.source_id,
                "section": group.category,
                "prompt_sha256": group.prompt_sha256,
                "prompt_tokens": group.prompt_tokens,
                "scorer_sha256": group.reward_metadata["scorer_sha256"],
                "provenance_sha256": group.reward_metadata["provenance_sha256"],
            }
            for group in generic_replay
        ],
        "same_ids_positions_order_and_scorers": True,
        "prompts_answers_or_outputs_present": False,
    }
    shared_replay["shared_replay_sha256"] = canonical_sha256(shared_replay)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "schedule_id": SCHEDULE_ID,
        "seed": SEED,
        "arms": ["generic", "targeted"],
        "groups_per_arm": GROUPS_PER_ARM,
        "task_groups_per_arm": TASK_GROUPS_PER_ARM,
        "replay_groups_per_arm": REPLAY_GROUPS_PER_ARM,
        "completions_per_group": COMPLETIONS_PER_GROUP,
        "completions_per_arm": COMPLETIONS_PER_ARM,
        "optimizer_steps_per_arm": OPTIMIZER_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "generic_task_quotas": dict(TASK_QUOTAS["generic"]),
        "targeted_task_quotas": dict(TASK_QUOTAS["targeted"]),
        "generic_prompt_tokens": generic_total,
        "targeted_prompt_tokens": targeted_total,
        "prompt_token_absolute_difference": abs(generic_total - targeted_total),
        "prompt_token_parity_ratio": parity,
        "prompt_token_parity_maximum": PROMPT_TOKEN_PARITY_MAXIMUM,
        "prompt_token_parity_passed": parity <= PROMPT_TOKEN_PARITY_MAXIMUM,
        "generic_manifest_sha256": generic.manifest["manifest_sha256"],
        "targeted_manifest_sha256": targeted.manifest["manifest_sha256"],
        "shared_replay_sha256": shared_replay["shared_replay_sha256"],
        "exact_group_and_completion_parity": True,
        "same_replay_ids_positions_order_and_scorers": True,
        "dataset_sha256": DATASET_SHA256,
        "gsm1k_prompt_use": 0,
        "holdout_v2_prompt_use": 0,
        "sealed_content_use": 0,
        "model_outputs_observed_during_scheduling": False,
        "prompts_answers_or_outputs_present": False,
    }
    summary["paired_schedule_sha256"] = canonical_sha256(summary)
    return ScheduleBundle(generic, targeted, shared_replay, summary)


def build_production_schedules(root: Path, tokenizer: Any) -> ScheduleBundle:
    """Reconstruct the production schedules from exact allowlisted inputs."""

    dataset_summary = json.loads(
        (root / "results/phase2_vetted_corpus/dataset_summary.json").read_text(encoding="utf-8")
    )
    if not isinstance(dataset_summary, dict) or (
        dataset_summary.get("dataset_sha256") != DATASET_SHA256
    ):
        raise ValueError("Phase 2 dataset identity differs")
    replay = load_replay_prompts(
        root / "results/raw/training/base_replay_kl/replay_corpus.json",
        root / "results/training/base_replay_corpus.json",
        root
        / "results/raw/training/retention_powered_adjudication"
        / "shared_retention_anchor_v1.json",
    )
    return build_schedules(
        generic_prompts=load_task_prompts(
            root / "results/raw/phase2_vetted_corpus/dataset/generic_training.jsonl",
            "generic",
        ),
        targeted_prompts=load_task_prompts(
            root / "results/raw/phase2_vetted_corpus/dataset/targeted_training.jsonl",
            "targeted",
        ),
        replay_prompts=replay,
        prompt_token_counter=lambda messages: count_transformers_prompt_tokens(tokenizer, messages),
    )


def _serialize(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _write_new_or_identical(path: Path, value: object) -> None:
    rendered = _serialize(value)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"existing schedule artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _require_ignored(root: Path, path: Path) -> None:
    relative = path.resolve().relative_to(root.resolve())
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", str(relative)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"prompt-bearing schedule path is not ignored: {path}")


def write_schedule_bundle(
    root: Path,
    bundle: ScheduleBundle,
    *,
    raw_directory: Path,
    tracked_directory: Path,
) -> dict[str, Path]:
    """Write prompt packets only to ignored storage and manifests to tracked storage."""

    generic_packet = raw_directory / "generic_prompt_packet.json"
    targeted_packet = raw_directory / "targeted_prompt_packet.json"
    for path in (generic_packet, targeted_packet):
        _require_ignored(root, path)
    paths = {
        "generic_packet": generic_packet,
        "targeted_packet": targeted_packet,
        "generic_manifest": tracked_directory / "milestone14a_generic_schedule.json",
        "targeted_manifest": tracked_directory / "milestone14a_targeted_schedule.json",
        "shared_replay": tracked_directory / "milestone14a_shared_replay.json",
        "paired_summary": tracked_directory / "milestone14a_paired_schedule.json",
    }
    _write_new_or_identical(paths["generic_packet"], bundle.generic.packet)
    _write_new_or_identical(paths["targeted_packet"], bundle.targeted.packet)
    _write_new_or_identical(paths["generic_manifest"], bundle.generic.manifest)
    _write_new_or_identical(paths["targeted_manifest"], bundle.targeted.manifest)
    _write_new_or_identical(paths["shared_replay"], bundle.shared_replay)
    _write_new_or_identical(paths["paired_summary"], bundle.summary)
    return paths
