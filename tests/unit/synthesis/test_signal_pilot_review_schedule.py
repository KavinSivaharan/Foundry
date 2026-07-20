"""Signal-first 120-question review-schedule tests."""

from __future__ import annotations

import json
from pathlib import Path

_BLOCKER = Path("results/synthesis_smoke/signal_pilot_runtime_identity_capacity_blocker.json")
_PRESERVED_SCHEDULE = Path("configs/synthesis/signal_pilot_review_schedule.json")


def test_review_schedule_is_not_rebuilt_after_full_capacity_gate_failure() -> None:
    blocker = json.loads(_BLOCKER.read_text(encoding="utf-8"))
    preserved = json.loads(_PRESERVED_SCHEDULE.read_text(encoding="utf-8"))

    assert blocker["gate"]["fresh_smoke_authorized"] is False
    assert blocker["gate"]["review_schedule_authorized"] is False
    assert (
        preserved["schedule_sha256"]
        == "021956fff2321bd28779390870b5d90030806cb69d8b97d18165b4aea3f67332"
    )
