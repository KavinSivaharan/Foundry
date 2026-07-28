from __future__ import annotations

import sys
from pathlib import Path

from foundry.phase2.layer_restricted_campaign import (
    _development_paths,
    _gsm1k_command,
    _training_command,
)


def test_training_command_transports_exact_scope_and_frozen_schedule() -> None:
    root = Path("C:/Foundry")
    command = _training_command(
        root,
        arm="generic",
        scope_label="L2",
        max_steps=16,
        run_root=root / "raw/L2/generic",
    )
    assert command[0] == sys.executable
    assert command[1:3] == [
        "-m",
        "foundry.phase2.vetted_qlora_layer_restricted",
    ]
    assert command[command.index("--scope-label") + 1] == "L2"
    assert command[command.index("--max-steps") + 1] == "16"
    assert (
        command[command.index("--schedule-sha256") + 1]
        == "4bc00d29d5cf308c12c77111d7943567521cc533b13440dc06c3d8b39c74e9df"
    )
    joined = " ".join(command)
    assert "candidate_suite.json" not in joined
    assert "gsm1k" not in joined.lower()


def test_development_paths_exclude_independent_holdout() -> None:
    root = Path("C:/Foundry")
    adjudication = _development_paths(root, "adjudication")
    anchor = _development_paths(root, "anchor")
    joined = " ".join(str(path) for path in (*adjudication, *anchor))
    assert "retention_adjudication_v2" in joined
    assert "retention_anchor_holdout_v1" in joined
    assert "milestone13c_r2" not in joined
    assert "candidate_suite.json" not in joined


def test_gsm1k_command_uses_frozen_814_manifest_and_unit_scale() -> None:
    root = Path("C:/Foundry")
    command = _gsm1k_command(
        root,
        root / "raw/generic",
        root / "adapter",
        "a" * 64,
    )
    assert command[command.index("--adapter-scale") + 1] == "1.0"
    assert command[command.index("--baseline-manifest") + 1].endswith(
        "gsm1k_development_baseline_814.json"
    )
    assert "gsm1k_sealed_final.json" not in " ".join(command)
