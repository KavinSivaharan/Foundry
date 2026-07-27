from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_calibration
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_PATH = (
    ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r3_historical_comparator.json"
)


def _historical() -> dict[str, Any]:
    value: object = json.loads(HISTORICAL_PATH.read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


def test_published_historical_comparator_self_hashes() -> None:
    value = _historical()
    assert value["objective_contract_sha256"] == (kl_calibration.OBJECTIVE_CONTRACT_SHA256)
    supplied = value.pop("historical_comparator_summary_sha256")
    assert supplied == canonical_sha256(value)


def test_historical_comparator_is_lambda_zero_without_updates() -> None:
    value = _historical()
    for arm in ("generic", "targeted"):
        row = value["arms"][arm]
        assert row["lambda_kl"] == 0
        assert row["optimizer_steps"] == 16
        assert row["loss_bearing_tokens"] == 16_000
        assert row["model_update_performed"] is False
        assert row["base_restoration"]
        assert row["adjudication_retention"]["passed"]
        assert row["anchor_retention"]["passed"]
    assert value["holdout_v2_use"] is False
    assert value["gsm1k_use"] is False
