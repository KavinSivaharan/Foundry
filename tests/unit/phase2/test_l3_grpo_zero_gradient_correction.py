from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.phase2 import l3_grpo_zero_gradient_correction as correction
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

ROOT = Path(__file__).resolve().parents[3]
WARMUP_AWARE_OVERRIDES = frozenset(
    {
        "src/foundry/phase2/l3_grpo_runtime.py",
        "tests/unit/phase2/test_l3_grpo_runtime.py",
        "tests/unit/phase2/test_l3_grpo_zero_gradient_correction.py",
    }
)


def test_file_rows_bind_bytes_and_sha256(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    rows = correction._file_rows(tmp_path, ("source.py",))
    assert rows == [
        {
            "path": "source.py",
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
        }
    ]


def test_correction_freeze_rejects_another_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="wrong repository"):
        correction.freeze_correction(tmp_path)


def test_published_correction_manifests_reconstruct_and_bind_sources() -> None:
    tracked = ROOT / "results/phase2_vetted_corpus"
    implementation = json.loads(
        (tracked / "milestone14a_r1_corrected_implementation.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (tracked / "milestone14a_r1_correction_contract.json").read_text(encoding="utf-8")
    )
    implementation_hash = implementation.pop("corrected_implementation_sha256")
    contract_hash = contract.pop("correction_contract_sha256")
    assert implementation_hash == canonical_sha256(implementation)
    assert contract_hash == canonical_sha256(contract)
    assert contract["corrected_implementation_sha256"] == implementation_hash
    assert contract["scientific_settings_changed"] is False
    assert contract["counted_training_gradient_gate_changed"] is False
    assert contract["official_smoke_runs_authorized"] == 2
    assert contract["official_smoke_retries_authorized"] == 0
    assert WARMUP_AWARE_OVERRIDES <= {row["path"] for row in implementation["files"]}
    for row in implementation["files"]:
        if row["path"] in WARMUP_AWARE_OVERRIDES:
            continue
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert file_sha256(path) == row["sha256"]
