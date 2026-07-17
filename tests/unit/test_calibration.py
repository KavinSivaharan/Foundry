import json
from dataclasses import replace
from pathlib import Path

import pytest

from foundry.config import EvaluationConfig, load_config
from foundry.evaluation.calibration import (
    MAIN_DEVELOPMENT_BASELINE,
    PROMPT_FORMAT_CALIBRATION,
    CalibrationError,
    as_development_benchmark_manifest,
    assert_prompt_only_variant,
    build_format_calibration_manifests,
    load_development_subset,
    save_development_subset,
    validate_format_calibration_pair,
)
from foundry.evaluation.manifests import BenchmarkManifest, load_manifest

CONFIG_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml")
DEVELOPMENT_PATH = Path("configs/eval/manifests/gsm1k_development.json")
SELECTION_SEED = "foundry-gsm1k-prompt-format-calibration-v1"


def _source_manifest() -> tuple[EvaluationConfig, BenchmarkManifest]:
    config = load_config(CONFIG_PATH)
    return config, load_manifest(DEVELOPMENT_PATH, config)


def test_calibration_split_is_deterministic_disjoint_and_identifier_only(tmp_path: Path) -> None:
    config, source = _source_manifest()

    first = build_format_calibration_manifests(
        source,
        calibration_size=30,
        selection_seed=SELECTION_SEED,
    )
    second = build_format_calibration_manifests(
        source,
        calibration_size=30,
        selection_seed=SELECTION_SEED,
    )

    assert first == second
    calibration, baseline = first
    assert calibration.purpose == PROMPT_FORMAT_CALIBRATION
    assert baseline.purpose == MAIN_DEVELOPMENT_BASELINE
    assert len(calibration.entries) == 30
    assert len(baseline.entries) == 874
    validate_format_calibration_pair(calibration, baseline, source)

    calibration_rows = {entry.row_index for entry in calibration.entries}
    baseline_rows = {entry.row_index for entry in baseline.entries}
    source_rows = {entry.row_index for entry in source.entries}
    assert calibration_rows.isdisjoint(baseline_rows)
    assert calibration_rows | baseline_rows == source_rows

    path = tmp_path / "calibration.json"
    save_development_subset(calibration, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["purpose"] == PROMPT_FORMAT_CALIBRATION
    assert set(payload["entries"][0]) == {"stable_id", "row_index"}
    serialized = json.dumps(payload).lower()
    assert '"question"' not in serialized
    assert '"answer"' not in serialized
    assert load_development_subset(path, source) == calibration

    adapted = as_development_benchmark_manifest(calibration, source, config)
    assert adapted.partition == "development"
    assert adapted.entries == calibration.entries
    assert adapted.manifest_sha256 == calibration.manifest_sha256


def test_modified_calibration_manifest_is_rejected(tmp_path: Path) -> None:
    _, source = _source_manifest()
    calibration, _ = build_format_calibration_manifests(
        source,
        calibration_size=30,
        selection_seed=SELECTION_SEED,
    )
    path = tmp_path / "calibration.json"
    save_development_subset(calibration, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["row_index"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CalibrationError):
        load_development_subset(path, source)


def test_prompt_calibration_rejects_non_prompt_changes() -> None:
    config = load_config(CONFIG_PATH)
    prompt_variant = replace(
        config,
        prompt=replace(
            config.prompt,
            user_template=(
                "Solve this problem.\n\n{question}\n\n"
                "Your final line must contain only:\nFinal answer: <integer>"
            ),
        ),
    )
    assert_prompt_only_variant(config, prompt_variant)

    generation_variant = replace(
        prompt_variant,
        generation=replace(prompt_variant.generation, max_new_tokens=256),
    )
    with pytest.raises(CalibrationError, match="may change only prompt"):
        assert_prompt_only_variant(config, generation_variant)
