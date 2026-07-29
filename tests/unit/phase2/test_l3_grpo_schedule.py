from __future__ import annotations

from collections import Counter

from foundry.phase2.l3_grpo_schedule import (
    FAMILY_ORDER,
    REPLAY_POSITIONS,
    TASK_QUOTAS,
    PromptMessage,
    TaskPrompt,
    build_schedules,
)
from foundry.training.config import canonical_sha256
from foundry.training.grpo_schedule import ReplayPrompt


def _task_prompts(arm: str) -> tuple[TaskPrompt, ...]:
    result: list[TaskPrompt] = []
    for family_index, family in enumerate(FAMILY_ORDER):
        for index in range(60):
            source_id = f"{arm}-{family_index}-{index:03d}"
            token_marker = 100 if arm == "generic" else 101
            messages = (
                PromptMessage("system", "fixture system"),
                PromptMessage("user", f"fixture {source_id} tokens={token_marker}"),
            )
            result.append(
                TaskPrompt(
                    source_id=source_id,
                    arm=arm,  # type: ignore[arg-type]
                    family=family,
                    messages=messages,
                    prompt_sha256=canonical_sha256([message.as_dict() for message in messages]),
                    canonical_answer="7",
                    answer_type="integer",
                    difficulty="easy",
                    verifier_metadata_sha256="a" * 64,
                    provenance_sha256="b" * 64,
                )
            )
    return tuple(result)


def _replay_prompts() -> tuple[ReplayPrompt, ...]:
    sections = ("arithmetic",) * 3 + ("format",) * 2 + ("instruction",) * 3
    result: list[ReplayPrompt] = []
    for index, section in enumerate(sections):
        messages = (
            PromptMessage("system", "fixture system"),
            PromptMessage("user", f"replay-{index} tokens=80"),
        )
        result.append(
            ReplayPrompt(
                replay_id=f"replay-{index}",
                section=section,  # type: ignore[arg-type]
                skill="fixture",
                messages=messages,
                prompt_sha256=canonical_sha256([message.as_dict() for message in messages]),
                kind="exact_text",
                expected="cedar",
                scorer_sha256="c" * 64,
                provenance_sha256="d" * 64,
            )
        )
    return tuple(result)


def _token_count(messages: tuple[PromptMessage, ...]) -> int:
    return int(messages[-1].content.rsplit("tokens=", maxsplit=1)[1])


def test_matched_schedule_freezes_exact_quotas_replay_and_parity() -> None:
    bundle = build_schedules(
        generic_prompts=_task_prompts("generic"),
        targeted_prompts=_task_prompts("targeted"),
        replay_prompts=_replay_prompts(),
        prompt_token_counter=_token_count,
    )
    for arm, schedule in (
        ("generic", bundle.generic),
        ("targeted", bundle.targeted),
    ):
        tasks = [group for group in schedule.groups if group.source_kind == "task"]
        replay = [group for group in schedule.groups if group.source_kind == "base_replay"]
        assert len(schedule.groups) == 32
        assert len(tasks) == 24
        assert len(replay) == 8
        assert Counter(group.category for group in tasks) == Counter(TASK_QUOTAS[arm])
        assert [group.position for group in replay] == list(REPLAY_POSITIONS)
        assert schedule.manifest["total_completions"] == 128
        assert schedule.manifest["checkpoint_steps"] == [8, 16, 32]

    generic_replay = [
        (group.position, group.source_id, group.reward_metadata["scorer_sha256"])
        for group in bundle.generic.groups
        if group.source_kind == "base_replay"
    ]
    targeted_replay = [
        (group.position, group.source_id, group.reward_metadata["scorer_sha256"])
        for group in bundle.targeted.groups
        if group.source_kind == "base_replay"
    ]
    assert generic_replay == targeted_replay
    assert bundle.summary["prompt_token_parity_passed"] is True
    assert float(bundle.summary["prompt_token_parity_ratio"]) <= 0.01
    assert bundle.summary["gsm1k_prompt_use"] == 0
    assert bundle.summary["holdout_v2_prompt_use"] == 0
    assert bundle.summary["sealed_content_use"] == 0


def test_schedule_reconstruction_is_deterministic() -> None:
    arguments = {
        "generic_prompts": _task_prompts("generic"),
        "targeted_prompts": _task_prompts("targeted"),
        "replay_prompts": _replay_prompts(),
        "prompt_token_counter": _token_count,
    }
    first = build_schedules(**arguments)
    second = build_schedules(**arguments)
    assert first.generic.packet == second.generic.packet
    assert first.targeted.packet == second.targeted.packet
    assert first.shared_replay == second.shared_replay
    assert first.summary == second.summary
