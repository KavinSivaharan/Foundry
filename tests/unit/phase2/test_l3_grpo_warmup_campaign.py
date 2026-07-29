from __future__ import annotations

from pathlib import Path

import pytest

from foundry.phase2 import l3_grpo_warmup_campaign as campaign
from foundry.phase2.l3_grpo_source_binding import argv_projection_sha256
from foundry.training.config import canonical_sha256


def test_counted_runtime_command_freezes_contract_and_partial_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("C:/Foundry")
    monkeypatch.setattr(
        campaign,
        "_training_python",
        lambda root: root / ".venv-training/Scripts/python.exe",
    )
    layer1: dict[str, object] = {}
    layer1["layer1_manifest_sha256"] = canonical_sha256(layer1)
    layer2: dict[str, object] = {"source_commit": "a" * 40, "source_tree": "b" * 40}
    layer2["layer2_manifest_sha256"] = canonical_sha256(layer2)
    binding: dict[str, object] = {"qualification_decision_sha256": "c" * 64}
    binding["source_binding_contract_sha256"] = canonical_sha256(binding)
    monkeypatch.setattr(
        campaign,
        "_read",
        lambda path: (
            layer1
            if path.name.endswith("layer1_scientific_qualification_manifest.json")
            else (
                layer2
                if path.name.endswith("layer2_compatibility_runtime_manifest.json")
                else binding
            )
        ),
    )
    command = campaign._runtime_command(
        root,
        arm="generic",
        output_dir=root / "raw/training/generic",
        raw_evidence=root / "raw/training/generic_raw.json",
        partial_evidence=root / "raw/training/generic_partial.json",
        summary=root / "raw/training/generic_summary.json",
    )
    assert command[1:3] == ["-m", "foundry.phase2.l3_grpo_runtime"]
    assert command[command.index("--mode") + 1] == "train"
    assert command[command.index("--arm") + 1] == "generic"
    assert command[command.index("--partial-evidence") + 1].endswith("generic_partial.json")
    assert command[command.index("--warmup-update-contract") + 1].endswith(
        "milestone14b_r2_warmup_update_contract.json"
    )
    assert command[command.index("--expected-argv-sha256") + 1] == argv_projection_sha256(command)


def test_development_paths_exclude_holdout_v2() -> None:
    root = Path("C:/Foundry")
    values = (
        *campaign._development_paths(root, "adjudication"),
        *campaign._development_paths(root, "anchor"),
    )
    joined = " ".join(str(value) for value in values)
    assert "retention_adjudication_v2" in joined
    assert "retention_anchor_holdout_v1" in joined
    assert "candidate_suite.json" not in joined
    assert "milestone13c_r2" not in joined


def test_gsm1k_command_is_adapter_only_and_unsealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("C:/Foundry")
    monkeypatch.setattr(
        campaign,
        "_training_python",
        lambda root: root / ".venv-training/Scripts/python.exe",
    )
    command = campaign._gsm1k_command(
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
    assert "gsm1k_sealed_final.json" not in " ".join(command)
