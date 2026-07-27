from __future__ import annotations

from pathlib import Path

from foundry.phase2 import kl_campaign

ROOT = Path(__file__).resolve().parents[3]


def test_campaign_matrix_is_exact_and_ordered() -> None:
    assert kl_campaign.RUNS == (
        ("lambda-001", "generic", "0.01"),
        ("lambda-001", "targeted", "0.01"),
        ("lambda-003", "generic", "0.03"),
        ("lambda-003", "targeted", "0.03"),
        ("lambda-010", "generic", "0.10"),
        ("lambda-010", "targeted", "0.10"),
        ("lambda-030", "generic", "0.30"),
        ("lambda-030", "targeted", "0.30"),
    )


def test_training_command_preserves_frozen_calibration_fields() -> None:
    command = kl_campaign._training_command(
        ROOT,
        "generic",
        "0.01",
        ROOT / "results/raw/example",
    )
    assert command[0].endswith("python.exe")
    assert command[1:5] == [
        "-m",
        "foundry.phase2.vetted_qlora_kl",
        "train",
        "--arm",
    ]
    assert command[command.index("--max-steps") + 1] == "16"
    assert command[command.index("--coefficient") + 1] == "0.01"
    assert command[command.index("--schedule-sha256") + 1] == (kl_campaign.SCHEDULES["generic"])
