import json
from pathlib import Path

import yaml

from foundry.cli import main
from foundry.config import load_config
from foundry.evaluation.manifests import load_manifest


def _write_config(tmp_path: Path) -> Path:
    development_manifest = tmp_path / "development.json"
    sealed_final_manifest = tmp_path / "sealed_final.json"
    config = {
        "schema_version": 1,
        "model": {
            "repo_id": "example/fake-model",
            "revision": "a" * 40,
            "dtype": "float16",
            "device": "cuda",
        },
        "dataset": {
            "repo_id": "example/fake-dataset",
            "revision": "b" * 40,
            "config_name": "default",
            "source_split": "test",
            "expected_examples": 4,
        },
        "partition": {
            "seed": "integration-test",
            "sealed_final_size": 1,
            "development_manifest": str(development_manifest),
            "sealed_final_manifest": str(sealed_final_manifest),
        },
        "prompt": {
            "system": "Solve arithmetic carefully.",
            "user_template": (
                "Solve this problem.\n\n{question}\n\n"
                "End with exactly one final line in this form:\n"
                "Final answer: <integer>"
            ),
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 64,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_fake_model_runs_end_to_end_without_gpu_or_download(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    assert main(["build-manifests", "--config", str(config_path)]) == 0

    config = load_config(config_path)
    manifest_path = Path(config.partition.development_manifest)
    manifest = load_manifest(manifest_path, config)
    fixture = []
    for position, entry in enumerate(manifest.entries):
        expected = 10 + position
        response = (
            f"Computed from {position} and 10.\nFinal answer: {expected}"
            if position < 2
            else "I cannot produce the required final line."
        )
        fixture.append(
            {
                "stable_id": entry.stable_id,
                "row_index": entry.row_index,
                "question": f"What is 10 plus {position}?",
                "answer": str(expected),
                "response": response,
            }
        )
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    output_dir = tmp_path / "results"

    result = main(
        [
            "evaluate-fixture",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--fixture",
            str(fixture_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "fake"
    assert summary["processed_examples"] == 3
    assert summary["schema_version"] == 2
    assert summary["exact_format_compliant_examples"] == 2
    assert summary["exact_format_compliance_rate"] == 2 / 3
    assert summary["extractable_examples"] == 2
    assert summary["extractable_answer_rate"] == 2 / 3
    assert summary["correct_examples"] == 2
    assert summary["invalid_examples"] == 1
    assert summary["ambiguous_or_rejected_examples"] == 1
    assert summary["generation_failures"] == 0
    assert summary["accuracy"] == 2 / 3
    assert summary["extraction_failure_categories"] == {"no_terminal_answer": 1}
    raw_lines = (output_dir / "raw" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3
