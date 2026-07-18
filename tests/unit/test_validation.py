import json
from dataclasses import replace
from pathlib import Path

import pytest

from foundry.config import load_config
from foundry.evaluation.calibration import load_development_subset
from foundry.evaluation.manifests import load_manifest
from foundry.evaluation.validation import (
    ANSWER_EXTRACTION_VALIDATION,
    FINAL_EVALUATOR_VALIDATION,
    ValidationManifestError,
    as_benchmark_manifest,
    as_final_benchmark_manifest,
    assert_final_evaluator_config,
    build_answer_validation_manifests,
    build_final_evaluator_manifests,
    load_answer_validation_manifest,
    load_final_evaluator_manifest,
    save_answer_validation_manifest,
    save_final_evaluator_manifest,
    validate_answer_validation_pair,
    validate_final_evaluator_pair,
)

CONFIG_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml")
FINAL_EVALUATOR_CONFIG_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml")
DEVELOPMENT_PATH = Path("configs/eval/manifests/gsm1k_development.json")
POOL_PATH = Path("configs/eval/manifests/gsm1k_development_baseline.json")
PREVIOUS_VALIDATION_PATH = Path("configs/eval/manifests/gsm1k_answer_extraction_validation.json")
SOURCE_BASELINE_844_PATH = Path("configs/eval/manifests/gsm1k_development_baseline_844.json")
SEED = "foundry-gsm1k-answer-extraction-validation-v1"
FINAL_SEED = "foundry-gsm1k-final-evaluator-validation-v1"


def test_final_evaluator_config_changes_only_the_verified_token_limit() -> None:
    base = load_config(CONFIG_PATH)
    evaluator = load_config(FINAL_EVALUATOR_CONFIG_PATH)

    assert_final_evaluator_config(base, evaluator)
    assert base.generation.max_new_tokens == 512
    assert evaluator.generation.max_new_tokens == 768


def test_other_final_evaluator_config_changes_are_rejected() -> None:
    base = load_config(CONFIG_PATH)
    evaluator = load_config(FINAL_EVALUATOR_CONFIG_PATH)
    changed_prompt = replace(
        evaluator,
        prompt=replace(evaluator.prompt, system="A different prompt."),
    )

    with pytest.raises(ValidationManifestError, match="only max_new_tokens"):
        assert_final_evaluator_config(base, changed_prompt)


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


def test_final_evaluator_split_is_deterministic_complete_and_identifier_only(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    evaluator_config = load_config(FINAL_EVALUATOR_CONFIG_PATH)
    development = load_manifest(DEVELOPMENT_PATH, config)
    pool = load_development_subset(POOL_PATH, development)
    previous_validation = load_answer_validation_manifest(
        PREVIOUS_VALIDATION_PATH,
        pool,
        development,
    )
    source_baseline = load_answer_validation_manifest(
        SOURCE_BASELINE_844_PATH,
        pool,
        development,
    )

    first = build_final_evaluator_manifests(
        source_baseline,
        development,
        validation_size=30,
        selection_seed=FINAL_SEED,
    )
    second = build_final_evaluator_manifests(
        source_baseline,
        development,
        validation_size=30,
        selection_seed=FINAL_SEED,
    )
    assert first == second
    validation, baseline = first
    assert validation.purpose == FINAL_EVALUATOR_VALIDATION
    assert len(validation.entries) == 30
    assert len(baseline.entries) == 814
    assert (
        validation.manifest_sha256
        == "2234e5ee82cf57e8fb74839a21f7f0ca0d2ff02ddd0fb0e42d93934415b2db93"
    )
    assert (
        baseline.manifest_sha256
        == "5e810d3ab644bef1d43c598a14a6164ba6464b27fde50e92a2f241816ce87897"
    )
    validate_final_evaluator_pair(validation, baseline, source_baseline, development)

    calibration = load_development_subset(
        Path("configs/eval/manifests/gsm1k_prompt_format_calibration.json"),
        development,
    )
    partitions = [
        {(entry.stable_id, entry.row_index) for entry in manifest.entries}
        for manifest in (calibration, previous_validation, validation, baseline)
    ]
    for position, identities in enumerate(partitions):
        for other in partitions[position + 1 :]:
            assert identities.isdisjoint(other)
    development_ids = {(entry.stable_id, entry.row_index) for entry in development.entries}
    assert set().union(*partitions) == development_ids

    validation_path = tmp_path / "final_validation.json"
    baseline_path = tmp_path / "final_baseline.json"
    save_final_evaluator_manifest(validation, validation_path)
    save_final_evaluator_manifest(baseline, baseline_path)
    for path in (validation_path, baseline_path):
        serialized = path.read_text(encoding="utf-8").lower()
        assert '"question"' not in serialized
        assert '"answer"' not in serialized
    assert (
        load_final_evaluator_manifest(validation_path, source_baseline, development) == validation
    )
    assert load_final_evaluator_manifest(baseline_path, source_baseline, development) == baseline
    adapted = as_final_benchmark_manifest(
        validation,
        source_baseline,
        development,
        evaluator_config,
    )
    assert adapted.entries == validation.entries
    assert adapted.config_sha256 == evaluator_config.sha256


def test_modified_final_evaluator_manifest_is_rejected(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    development = load_manifest(DEVELOPMENT_PATH, config)
    pool = load_development_subset(POOL_PATH, development)
    source_baseline = load_answer_validation_manifest(
        SOURCE_BASELINE_844_PATH,
        pool,
        development,
    )
    validation, _ = build_final_evaluator_manifests(
        source_baseline,
        development,
        validation_size=30,
        selection_seed=FINAL_SEED,
    )
    path = tmp_path / "final_validation.json"
    save_final_evaluator_manifest(validation, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["stable_id"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationManifestError):
        load_final_evaluator_manifest(path, source_baseline, development)
