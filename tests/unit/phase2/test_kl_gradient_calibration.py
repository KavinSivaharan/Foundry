from __future__ import annotations

from pathlib import Path

import pytest

from foundry.phase2 import (
    kl_gradient_calibration,
    kl_gradient_campaign,
    vetted_qlora_kl_gradient,
)
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
LADDER = ROOT / "results/phase2_vetted_corpus/milestone13d_gradient_ladder.json"


def test_frozen_ladder_is_accepted_only_with_exact_coefficient() -> None:
    ladder = vetted_qlora_kl_gradient._validate_ladder(
        LADDER,
        "0.10",
        "0.88772676877271567348280710717339159364685227428036",
    )
    assert ladder["ladder_sha256"] == (
        "ab6b7a61ca0d4deda7762daf3d03e66c3ab55e2e18370b919f6def5626e54cda"
    )
    with pytest.raises(ValueError, match="frozen common ladder"):
        vetted_qlora_kl_gradient._validate_ladder(LADDER, "0.10", "0.89")


def test_campaign_order_is_frozen_and_sequential() -> None:
    assert kl_gradient_campaign.RHO_LABELS == (
        ("rho-010", "0.10"),
        ("rho-030", "0.30"),
        ("rho-100", "1.00"),
        ("rho-300", "3.00"),
    )
    assert kl_gradient_campaign.ARMS == ("generic", "targeted")


def test_selection_chooses_smallest_common_eligible_rho() -> None:
    smoke = {
        "coefficient_results": [
            {
                "rho_exact": rho,
                "coefficient_exact": str(index + 1),
                "common_eligible": True,
            }
            for index, (_, rho) in enumerate(kl_gradient_calibration.RHO_LABELS)
        ]
    }
    smoke["smoke_summary_sha256"] = canonical_sha256(smoke)
    runs = []
    for index, (_, rho) in enumerate(kl_gradient_calibration.RHO_LABELS):
        for arm in ("generic", "targeted"):
            row = {
                "rho_exact": rho,
                "coefficient_exact": str(index + 1),
                "arm": arm,
                "criteria": {"pass": index >= 1},
                "eligible": index >= 1,
            }
            row["calibration_run_sha256"] = canonical_sha256(row)
            runs.append(row)
    calibration = {"runs": runs}
    calibration["calibration_summary_sha256"] = canonical_sha256(calibration)
    value = kl_gradient_calibration.selection_record(calibration, smoke)
    assert value["selected_rho"] == "0.30"
    assert value["selected_coefficient_exact"] == "2"
    assert value["full_training_run_in_milestone"] is False
