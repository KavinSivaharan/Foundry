import json
from dataclasses import replace
from pathlib import Path

import pytest

from foundry.config import load_config
from foundry.evaluation.calibration import load_development_subset
from foundry.evaluation.manifests import load_manifest
from foundry.evaluation.validation import (
    ANSWER_EXTRACTION_VALIDATION,
    ValidationManifestError,
    as_benchmark_manifest,
    build_answer_validation_manifests,
    load_answer_validation_manifest,
    save_answer_validation_manifest,
    validate_answer_validation_pair,
)

CONFIG_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml")
DEVELOPMENT_PATH = Path("configs/eval/manifests/gsm1k_development.json")
POOL_PATH = Path("configs/eval/manifests/gsm1k_development_baseline.json")
SEED = "foundry-gsm1k-answer-extraction-validation-v1"


def test_fresh_validation_split_is_deterministic_disjoint_and_identifier_only(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    development = load_manifest(DEVELOPMENT_PATH, config)
    pool = load_development_subset(POOL_PATH, development)

    first = build_answer_validation_manifests(
        pool,
        development,
        validation_size=30,
        selection_seed=SEED,
    )
    second = build_answer_validation_manifests(
        pool,
        development,
        validation_size=30,
        selection_seed=SEED,
    )
    assert first == second
    validation, baseline = first
    assert validation.purpose == ANSWER_EXTRACTION_VALIDATION
    assert len(validation.entries) == 30
    assert len(baseline.entries) == 844
    validate_answer_validation_pair(validation, baseline, pool, development)

    calibration = load_development_subset(
        Path("configs/eval/manifests/gsm1k_prompt_format_calibration.json"),
        development,
    )
    calibration_ids = {(entry.stable_id, entry.row_index) for entry in calibration.entries}
    validation_ids = {(entry.stable_id, entry.row_index) for entry in validation.entries}
    baseline_ids = {(entry.stable_id, entry.row_index) for entry in baseline.entries}
    assert calibration_ids.isdisjoint(validation_ids)
    assert calibration_ids.isdisjoint(baseline_ids)
    assert validation_ids.isdisjoint(baseline_ids)

    path = tmp_path / "validation.json"
    save_answer_validation_manifest(validation, path)
    serialized = path.read_text(encoding="utf-8").lower()
    assert '"question"' not in serialized
    assert '"answer"' not in serialized
    assert load_answer_validation_manifest(path, pool, development) == validation

    adapted = as_benchmark_manifest(validation, pool, development, config)
    assert adapted.partition == "development"
    assert adapted.entries == validation.entries


def test_modified_answer_validation_manifest_is_rejected(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    development = load_manifest(DEVELOPMENT_PATH, config)
    pool = load_development_subset(POOL_PATH, development)
    validation, _ = build_answer_validation_manifests(
        pool,
        development,
        validation_size=30,
        selection_seed=SEED,
    )
    path = tmp_path / "validation.json"
    save_answer_validation_manifest(validation, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["row_index"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationManifestError):
        load_answer_validation_manifest(path, pool, development)


def test_baseline_manifest_cannot_be_adapted_for_validation() -> None:
    config = load_config(CONFIG_PATH)
    development = load_manifest(DEVELOPMENT_PATH, config)
    pool = load_development_subset(POOL_PATH, development)
    validation, _ = build_answer_validation_manifests(
        pool,
        development,
        validation_size=30,
        selection_seed=SEED,
    )
    wrong_purpose = replace(validation, purpose="main_development_baseline")

    with pytest.raises(ValidationManifestError):
        as_benchmark_manifest(wrong_purpose, pool, development, config)
