from pathlib import Path

import pytest

from foundry.synthesis.contamination import ContaminationOutcome
from foundry.synthesis.semantic import (
    SemanticArtifactConfig,
    SemanticThresholds,
    load_semantic_artifact_config,
)


def test_frozen_semantic_threshold_boundaries() -> None:
    thresholds = SemanticThresholds(0.75, 0.75, 0.82)

    assert thresholds.classify(0.749999) is ContaminationOutcome.PASS
    assert thresholds.classify(0.75) is ContaminationOutcome.MANUAL_REVIEW
    assert thresholds.classify(0.819999) is ContaminationOutcome.MANUAL_REVIEW
    assert thresholds.classify(0.82) is ContaminationOutcome.REJECT


def test_threshold_changes_are_rejected() -> None:
    with pytest.raises(ValueError, match="frozen"):
        SemanticThresholds(0.70, 0.70, 0.82)


def test_selected_semantic_config_is_exactly_pinned() -> None:
    config = load_semantic_artifact_config(Path("configs/synthesis/semantic_all_minilm_l6_v2.yaml"))

    assert config.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert config.embedding_dimension == 384
    assert config.trust_remote_code is False
    assert config.local_files_only_after_download is True
    assert config.expected_download_bytes == 91_577_897


def test_semantic_config_forbids_remote_code() -> None:
    thresholds = SemanticThresholds(0.75, 0.75, 0.82)

    with pytest.raises(ValueError, match="remote code"):
        SemanticArtifactConfig(
            artifact_id="fixture",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            license_id="apache-2.0",
            expected_download_bytes=91_577_897,
            embedding_dimension=384,
            pooling="attention_mask_weighted_mean",
            normalization="l2",
            max_length=256,
            device="cpu",
            dtype="float32",
            batch_size=32,
            trust_remote_code=True,
            local_files_only_after_download=True,
            required_files=("model.safetensors",),
            cache_root=Path("ignored"),
            thresholds=thresholds,
        )
