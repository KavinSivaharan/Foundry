from __future__ import annotations

from pathlib import Path

import pytest

from foundry.phase2 import l3_grpo_warmup_compatibility_campaign as campaign
from foundry.training.config import canonical_sha256


def _hashed(value: dict[str, object], key: str) -> dict[str, object]:
    value[key] = canonical_sha256(value)
    return value


def test_campaign_runs_generic_pair_before_targeted_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    selection = _hashed(
        {
            "arms": {
                arm: {
                    "task_group_id": f"{arm}-task",
                    "replay_group_id": f"{arm}-replay",
                }
                for arm in ("generic", "targeted")
            }
        },
        "selection_decision_sha256",
    )
    warmup = _hashed({}, "warmup_update_contract_sha256")
    monkeypatch.setattr(campaign, "_environment", lambda root: {"FROZEN": "1"})
    monkeypatch.setattr(
        campaign,
        "_require_clean_synchronized_main",
        lambda root: "a" * 40,
    )
    monkeypatch.setattr(
        campaign,
        "_read",
        lambda path: (
            selection if path.name.endswith("selection_and_gradient_decision.json") else warmup
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_runtime_command",
        lambda root, **kwargs: [f"{kwargs['arm']}-run-{kwargs['run_index']}"],
    )

    def fake_run(command: list[str], **kwargs: object) -> None:
        del kwargs
        calls.append(command[0])

    monkeypatch.setattr(campaign, "_run", fake_run)
    monkeypatch.setattr(
        campaign,
        "_validated_run",
        lambda **kwargs: {
            "arm": kwargs["arm"],
            "run_index": kwargs["run_index"],
        },
    )
    monkeypatch.setattr(
        campaign,
        "build_compatibility_result",
        lambda **kwargs: {"gate_passed": True},
    )
    monkeypatch.setattr(campaign, "_write_new", lambda *args, **kwargs: None)
    assert campaign.run_campaign(tmp_path) == {"gate_passed": True}
    assert calls == [
        "generic-run-1",
        "generic-run-2",
        "targeted-run-1",
        "targeted-run-2",
    ]


def test_notice_evidence_rejects_unclassified_stderr(tmp_path: Path) -> None:
    path = tmp_path / "stderr.txt"
    path.write_text("unapproved warning\n", encoding="utf-8")
    result = campaign._notice_evidence(path)
    assert result["only_authorized_notices"] is False
    assert result["unrecognized_line_count"] == 1
