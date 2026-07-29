from __future__ import annotations

from pathlib import Path

import pytest

from foundry.phase2 import l3_grpo_zero_gradient_campaign as campaign


def test_campaign_runs_exactly_two_sequential_smokes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(campaign, "_environment", lambda root: {"FROZEN": "1"})
    monkeypatch.setattr(
        campaign,
        "_require_clean_synchronized_main",
        lambda root: "a" * 40,
    )
    monkeypatch.setattr(
        campaign,
        "_runtime_command",
        lambda root, **kwargs: [str(kwargs["summary"])],
    )

    def fake_run(command: list[str], **kwargs: object) -> None:
        del kwargs
        calls.append(command[0])

    monkeypatch.setattr(campaign, "_run", fake_run)
    monkeypatch.setattr(
        campaign,
        "write_compatibility_result",
        lambda root: {"gate_passed": True},
    )
    result = campaign.run_campaign(tmp_path)
    assert result == {"gate_passed": True}
    assert calls == [
        str(
            tmp_path / "results/raw/phase2_vetted_corpus/milestone14a_r1/"
            "compatibility/run-1/summary.json"
        ),
        str(
            tmp_path / "results/raw/phase2_vetted_corpus/milestone14a_r1/"
            "compatibility/run-2/summary.json"
        ),
    ]


def test_campaign_has_no_retry_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0
    published = False
    monkeypatch.setattr(campaign, "_environment", lambda root: {"FROZEN": "1"})
    monkeypatch.setattr(
        campaign,
        "_require_clean_synchronized_main",
        lambda root: "a" * 40,
    )
    monkeypatch.setattr(
        campaign,
        "_runtime_command",
        lambda root, **kwargs: [str(kwargs["summary"])],
    )

    def fail_first(command: list[str], **kwargs: object) -> None:
        nonlocal calls
        del command, kwargs
        calls += 1
        raise RuntimeError("official failure")

    def publish(root: Path) -> dict[str, object]:
        nonlocal published
        del root
        published = True
        return {}

    monkeypatch.setattr(campaign, "_run", fail_first)
    monkeypatch.setattr(campaign, "write_compatibility_result", publish)
    with pytest.raises(RuntimeError, match="official failure"):
        campaign.run_campaign(tmp_path)
    assert calls == 1
    assert published is False
