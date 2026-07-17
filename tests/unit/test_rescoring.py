import json
from pathlib import Path

from foundry.evaluation.rescoring import rescore_predictions


def test_rescores_existing_records_without_copying_raw_content(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    records = [
        {
            "response": "Work.\nFinal answer: 42",
            "reference_answer": 42,
            "output_tokens": 10,
            "error": None,
        },
        {
            "response": "Work.\n" r"\boxed{7}",
            "reference_answer": 7,
            "output_tokens": 12,
            "error": "response must contain exactly one Final answer: line",
        },
        {
            "response": "The answer is 8.",
            "reference_answer": 9,
            "output_tokens": 15,
            "error": "response must contain exactly one Final answer: line",
        },
        {
            "response": "The answer is 10.",
            "reference_answer": 10,
            "output_tokens": 64,
            "error": "response must contain exactly one Final answer: line",
        },
    ]
    predictions.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"

    summary = rescore_predictions(predictions, output, max_new_tokens=64)

    assert summary.processed_examples == 4
    assert summary.exact_format_compliant_examples == 1
    assert summary.extractable_examples == 3
    assert summary.correct_examples == 2
    assert summary.benchmark_accuracy == 0.5
    assert summary.ambiguous_or_rejected_examples == 1
    assert summary.extraction_failure_categories == {"generation_truncated": 1}
    serialized = output.read_text(encoding="utf-8")
    assert "Work." not in serialized
    assert "reference_answer" not in serialized
