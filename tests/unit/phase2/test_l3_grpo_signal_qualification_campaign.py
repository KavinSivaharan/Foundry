from __future__ import annotations

from pathlib import Path

import pytest

from foundry.phase2 import l3_grpo_signal_qualification_campaign as campaign
from foundry.training.config import canonical_sha256


def test_campaign_runs_four_fresh_processes_in_frozen_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    contract = {
        "selected_candidates": {
            arm: {"task_group_id": f"{arm}-task"} for arm in ("generic", "targeted")
        }
    }
    contract["qualification_contract_sha256"] = canonical_sha256(contract)
    monkeypatch.setattr(campaign, "_environment", lambda root: {"FROZEN": "1"})
    monkeypatch.setattr(
        campaign,
        "_require_clean_synchronized_main",
        lambda root: "a" * 40,
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
        "_validate_pair",
        lambda **kwargs: (
            [
                {"source_commit": "a" * 40},
                {"source_commit": "a" * 40},
            ],
            kwargs["arm"][0] * 64,
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_read",
        lambda path: (
            contract if path.name.endswith("qualification_contract.json") else {"groups": []}
        ),
    )
    monkeypatch.setattr(
        campaign,
        "build_signal_summary",
        lambda *args, **kwargs: {
            "decision": "schedule_viable",
            "viability_passed": True,
            "signal_summary_sha256": "s" * 64,
        },
    )
    monkeypatch.setattr(
        campaign,
        "build_selection_and_gradient_decision",
        lambda **kwargs: {"selection_decision_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "write_selection_and_signal_decision",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(campaign, "_write_new", lambda *args, **kwargs: None)
    result = campaign.run_campaign(tmp_path)
    assert result["gate_passed"] is True
    assert calls == [
        "generic-run-1",
        "generic-run-2",
        "targeted-run-1",
        "targeted-run-2",
    ]


def test_campaign_has_no_retry_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    monkeypatch.setattr(campaign, "_environment", lambda root: {"FROZEN": "1"})
    monkeypatch.setattr(
        campaign,
        "_require_clean_synchronized_main",
        lambda root: "a" * 40,
    )
    monkeypatch.setattr(campaign, "_runtime_command", lambda *args, **kwargs: ["run"])

    def fail(command: list[str], **kwargs: object) -> None:
        nonlocal calls
        del command, kwargs
        calls += 1
        raise RuntimeError("projection failed")

    monkeypatch.setattr(campaign, "_run", fail)
    with pytest.raises(RuntimeError, match="projection failed"):
        campaign.run_campaign(tmp_path)
    assert calls == 1
