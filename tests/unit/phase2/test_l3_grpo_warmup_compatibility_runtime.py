from __future__ import annotations

import pytest

from foundry.phase2.l3_grpo_runtime import RuntimeGroup, RuntimeSchedule
from foundry.phase2.l3_grpo_warmup_compatibility_runtime import (
    _restricted_schedule,
)


def _group(position: int, kind: str) -> RuntimeGroup:
    return RuntimeGroup(
        group_id=f"group-{position}",
        arm="generic",
        position=position,
        source_kind=kind,  # type: ignore[arg-type]
        source_id=f"source-{position}",
        category="family",
        messages=(),
        prompt_sha256=str(position) * 64,
        prompt_tokens=1,
        reward_metadata_json="{}",
    )


def test_warmup_restricted_schedule_is_replay_then_task() -> None:
    full = RuntimeSchedule(
        arm="generic",
        groups=(
            _group(1, "task"),
            _group(4, "base_replay"),
            _group(5, "task"),
        ),
        packet_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    selected = {
        "task_group_id": "group-5",
        "task_schedule_position": 5,
        "task_prompt_sha256": "5" * 64,
        "replay_group_id": "group-4",
        "replay_schedule_position": 4,
        "replay_prompt_sha256": "4" * 64,
    }
    result = _restricted_schedule(full, selected)
    assert [group.group_id for group in result.groups] == ["group-4", "group-5"]
    assert [group.source_kind for group in result.groups] == [
        "base_replay",
        "task",
    ]


def test_warmup_restricted_schedule_rejects_identity_drift() -> None:
    full = RuntimeSchedule(
        arm="generic",
        groups=(_group(1, "task"), _group(4, "base_replay")),
        packet_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="do not reconstruct"):
        _restricted_schedule(
            full,
            {
                "task_group_id": "group-1",
                "task_schedule_position": 1,
                "task_prompt_sha256": "x" * 64,
                "replay_group_id": "group-4",
                "replay_schedule_position": 4,
                "replay_prompt_sha256": "4" * 64,
            },
        )
