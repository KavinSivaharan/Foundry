from __future__ import annotations

from pathlib import Path

from foundry.phase2.l3_grpo_campaign import (
    _development_paths,
    _gsm1k_command,
    _runtime_command,
)


def test_runtime_command_transports_only_the_frozen_arm_and_contract() -> None:
    root = Path("C:/Foundry")
    command = _runtime_command(
        root,
        arm="generic",
        mode="compatibility",
        output_dir=root / "raw/artifacts",
        raw_evidence=root / "raw/evidence.json",
        summary=root / "raw/summary.json",
    )
    assert command[1:3] == ["-m", "foundry.phase2.l3_grpo_runtime"]
    assert command[command.index("--arm") + 1] == "generic"
    assert command[command.index("--mode") + 1] == "compatibility"
    assert command[command.index("--starting-adapter") + 1].endswith(
        "milestone13e\\full\\generic\\training\\checkpoint-64\\adapter"
    )
    joined = " ".join(command).lower()
    assert "gsm1k" not in joined
    assert "candidate_suite.json" not in joined


def test_development_paths_exclude_holdout_v2() -> None:
    root = Path("C:/Foundry")
    values = (*_development_paths(root, "adjudication"), *_development_paths(root, "anchor"))
    joined = " ".join(str(value) for value in values)
    assert "retention_adjudication_v2" in joined
    assert "retention_anchor_holdout_v1" in joined
    assert "candidate_suite.json" not in joined
    assert "milestone13c_r2" not in joined


def test_gsm1k_command_uses_frozen_development_evaluator_without_base_rerun() -> None:
    root = Path("C:/Foundry")
    command = _gsm1k_command(
        root,
        adapter=root / "adapter",
        adapter_sha256="a" * 64,
        output_dir=root / "raw/output",
        summary=root / "summary.json",
    )
    assert command[command.index("--adapter-scale") + 1] == "1.0"
    assert command[command.index("--baseline-manifest") + 1].endswith(
        "gsm1k_development_baseline_814.json"
    )
    assert "--adapter" in command
    assert "gsm1k_sealed_final.json" not in " ".join(command)
