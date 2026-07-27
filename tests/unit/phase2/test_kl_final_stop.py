from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_final_stop
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
RECORD_PATH = ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r3_final_stop.json"


def _record() -> dict[str, Any]:
    value: object = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


def test_terminal_stop_reconstructs_from_published_evidence() -> None:
    assert _record() == kl_final_stop.build(ROOT)


def test_terminal_stop_is_self_hashed_and_pre_full_training() -> None:
    value = _record()
    supplied = value.pop("terminal_stop_sha256")
    assert supplied == canonical_sha256(value)
    assert value["result"] == "stopped_at_calibration"
    assert value["selected_coefficient"] is None
    assert value["full_training_runs"] == 0
    assert value["holdout_v2"]["adapter_evaluations"] == 0
    assert value["gsm1k_adapter_evaluations"] == 0
    assert value["sealed_boundary_status"] == ("metadata_accessed_example_content_unseen")
    assert value["sealed_paths_accessed"] is False
