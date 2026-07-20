import json
from pathlib import Path

import pytest

from foundry.training.paired_analysis import analyze_paired_results, paired_bootstrap_interval


def _write_predictions(path: Path, correctness: tuple[bool, ...]) -> list[str]:
    stable_ids = [f"{index:064x}" for index in range(len(correctness))]
    path.write_text(
        "".join(
            json.dumps({"stable_id": stable_id, "correct": correct}) + "\n"
            for stable_id, correct in zip(stable_ids, correctness, strict=True)
        ),
        encoding="utf-8",
    )
    return stable_ids


def test_paired_bootstrap_is_exactly_reproducible() -> None:
    differences = (1, 0, -1, 1, 0, 1)
    first = paired_bootstrap_interval(differences, seed=7, replicates=200)
    second = paired_bootstrap_interval(differences, seed=7, replicates=200)
    assert first == second
    assert first[0] <= sum(differences) / len(differences) <= first[1]


def test_analysis_validates_alignment_and_applies_frozen_gate(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    generic_path = tmp_path / "generic.jsonl"
    targeted_path = tmp_path / "targeted.jsonl"
    stable_ids = _write_predictions(base_path, (True, False, False, True))
    _write_predictions(generic_path, (False, True, False, True))
    _write_predictions(targeted_path, (True, False, True, False))

    taxonomy_path = tmp_path / "taxonomy.jsonl"
    taxonomy_path.write_text(
        json.dumps(
            {
                "stable_id": stable_ids[1],
                "primary_category": "multi_step_bookkeeping_or_omission",
            }
        )
        + "\n"
        + json.dumps({"stable_id": stable_ids[2], "primary_category": "arithmetic_execution"})
        + "\n",
        encoding="utf-8",
    )
    generic_summary = tmp_path / "generic-summary.json"
    generic_summary.write_text(
        json.dumps(
            {"correct_examples": 2, "extractable_answer_rate": 1.0, "generation_failures": 0}
        ),
        encoding="utf-8",
    )
    targeted_summary = tmp_path / "targeted-summary.json"
    targeted_summary.write_text(
        json.dumps(
            {"correct_examples": 2, "extractable_answer_rate": 1.0, "generation_failures": 0}
        ),
        encoding="utf-8",
    )
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"benchmark_evaluation_authorized": True}), encoding="utf-8")
    output = tmp_path / "result.json"

    result = analyze_paired_results(
        base_predictions_path=base_path,
        generic_predictions_path=generic_path,
        targeted_predictions_path=targeted_path,
        generic_summary_path=generic_summary,
        targeted_summary_path=targeted_summary,
        taxonomy_path=taxonomy_path,
        final_parity_path=parity,
        output_path=output,
        expected_examples=4,
    )

    assert result["correct"] == {"base": 2, "generic_control": 2, "targeted": 2}
    assert result["paired_changes"]["targeted_net_wins"] == 0
    assert result["one_seed_signal_gate_passed"] is False
    assert output.exists()


def test_analysis_rejects_different_stable_id_sets(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    generic_path = tmp_path / "generic.jsonl"
    targeted_path = tmp_path / "targeted.jsonl"
    _write_predictions(base_path, (True, False))
    _write_predictions(generic_path, (True, False))
    targeted_path.write_text(
        json.dumps({"stable_id": "f" * 64, "correct": True}) + "\n",
        encoding="utf-8",
    )
    placeholder = tmp_path / "placeholder.json"
    placeholder.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="stable-ID sets differ"):
        analyze_paired_results(
            base_predictions_path=base_path,
            generic_predictions_path=generic_path,
            targeted_predictions_path=targeted_path,
            generic_summary_path=placeholder,
            targeted_summary_path=placeholder,
            taxonomy_path=placeholder,
            final_parity_path=placeholder,
            output_path=tmp_path / "result.json",
            expected_examples=2,
        )
