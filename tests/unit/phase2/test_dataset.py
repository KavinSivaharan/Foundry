from __future__ import annotations

from pathlib import Path

import pytest

from foundry.phase2.asdiv import execute_formula
from foundry.phase2.dataset import build_datasets, construct_completion, deterministic_split

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "results" / "raw" / "phase2_vetted_corpus"
MODEL_PATH = (
    ROOT
    / "data"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-1.5B-Instruct"
    / "snapshots"
    / "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)


def test_construct_completion_replays_formula() -> None:
    execution = execute_formula("2+3=5")
    row: dict[str, object] = {
        "source_id": "fixture-1",
        "source_corpus": "asdiv_v1_0",
        "combined_question": "A fixture asks for two plus three.",
        "formula": "2+3=5",
        "canonical_answer": "5",
        "program_sha256": execution.program_sha256,
        "operation_count": 1,
    }
    assert construct_completion(row) == "Calculation: 2+3=5\nFinal answer: 5"


def test_deterministic_split_is_exact_and_stable() -> None:
    rows = [
        {
            "source_id": f"row-{index:03d}",
            "family": f"family-{index % 3}",
            "source_corpus": f"source-{index % 2}",
            "grade": f"grade-{index % 4}",
            "answer_type": f"type-{index % 2}",
            "operation_count": index % 5,
        }
        for index in range(200)
    ]
    first = deterministic_split(rows)
    second = deterministic_split(list(reversed(rows)))
    assert first == second
    assert len(first[0]) == 180
    assert len(first[1]) == 20
    assert not set(first[0]) & set(first[1])


def test_live_dataset_reconstruction_is_byte_identical(tmp_path: Path) -> None:
    repair_root = RAW_ROOT / "matching_repair"
    required = (
        repair_root / "targeted_full.jsonl",
        repair_root / "generic_full.jsonl",
        repair_root / "repair_summary.json",
        MODEL_PATH,
    )
    if not all(path.exists() for path in required):
        pytest.skip("ignored repaired dataset inputs are not available")
    arguments = {
        "targeted_path": repair_root / "targeted_full.jsonl",
        "generic_path": repair_root / "generic_full.jsonl",
        "repair_summary_path": repair_root / "repair_summary.json",
        "model_path": MODEL_PATH,
    }
    first = build_datasets(**arguments, output_root=tmp_path / "first")
    second = build_datasets(**arguments, output_root=tmp_path / "second")
    assert first == second
    assert first["target_format_sha256"] == (
        "4239aad327ba78941609dccddfc16d1b32e701e194099d3db67ca8d5517c55a2"
    )
    for arm in ("targeted", "generic"):
        evidence = first[arm]
        assert evidence["records"] == 200  # type: ignore[index]
        assert evidence["training_records"] == 180  # type: ignore[index]
        assert evidence["validation_records"] == 20  # type: ignore[index]
        assert evidence["assistant_tokens_maximum"] <= 128  # type: ignore[index,operator]
        assert evidence["target_replay_exact"] is True  # type: ignore[index]
    for name in (
        "targeted_training.jsonl",
        "targeted_validation.jsonl",
        "targeted_manifest.jsonl",
        "generic_training.jsonl",
        "generic_validation.jsonl",
        "generic_manifest.jsonl",
        "dataset_summary.json",
    ):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
