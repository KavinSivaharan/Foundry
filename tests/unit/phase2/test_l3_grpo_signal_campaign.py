from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foundry.phase2 import l3_grpo_signal_campaign as campaign
from foundry.training.config import canonical_sha256


def test_runtime_command_uses_only_the_authorized_model_interpreter(
    monkeypatch: Any,
) -> None:
    root = Path("C:/Foundry")
    monkeypatch.setattr(campaign, "file_sha256", lambda _path: campaign.INTERPRETER_SHA256)
    monkeypatch.setattr(campaign, "adapter_path", lambda base, arm: base / f"{arm}-adapter")
    command = campaign._runtime_command(root, "generic", root / "raw/generic")
    assert command[0] == str(root / ".venv-training/Scripts/python.exe")
    assert command[1:3] == ["-m", "foundry.phase2.l3_grpo_signal_runtime"]
    assert command[command.index("--arm") + 1] == "generic"
    assert "--advantage-contract" in command
    assert "--prior-diagnostic-manifest" in command
    joined = " ".join(command).lower()
    assert "holdout" not in joined
    assert "gsm1k" not in joined
    assert "sealed" not in joined


def test_campaign_runs_generic_then_targeted_without_retry(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(campaign, "_environment", lambda _root: {"FROZEN": "1"})
    monkeypatch.setattr(
        campaign,
        "_require_clean_synchronized_main",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        campaign,
        "_runtime_command",
        lambda _root, arm, _run_root: [arm],
    )

    def fake_run(
        command: list[str],
        *,
        root: Path,
        environment: dict[str, str],
        stdout: Path,
        stderr: Path,
    ) -> None:
        del root, environment, stdout, stderr
        arm = command[0]
        calls.append(arm)
        summary: dict[str, object] = {
            "groups": 32,
            "completions": 128,
            "optimizer_created": False,
            "backward_calls": 0,
            "scheduler_created": False,
            "adapter_saved": False,
            "gate_passed": True,
            "raw_evidence_file_sha256": arm * 8,
        }
        summary["summary_sha256"] = canonical_sha256(summary)
        path = (
            tmp_path
            / "results/raw/phase2_vetted_corpus/milestone14b_r1/signal_audit"
            / arm
            / "summary.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(campaign, "_run", fake_run)
    result = campaign.run_campaign(tmp_path)
    assert calls == ["generic", "targeted"]
    assert result["processes"] == 2
    assert result["groups"] == 64
    assert result["completions"] == 256
    assert result["backward_calls"] == 0
