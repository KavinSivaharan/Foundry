from __future__ import annotations

import pytest

from foundry.phase2.l3_grpo_runtime import RuntimeGroup, RuntimeSchedule
from foundry.phase2.l3_grpo_signal_compatibility_runtime import (
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


def test_restricted_schedule_uses_only_frozen_task_and_replay() -> None:
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
    assert [group.group_id for group in result.groups] == ["group-5", "group-4"]
    assert result.packet_sha256 == full.packet_sha256
    assert result.manifest_sha256 == full.manifest_sha256


def test_restricted_schedule_rejects_prompt_hash_drift() -> None:
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
