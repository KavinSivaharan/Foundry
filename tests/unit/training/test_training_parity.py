from __future__ import annotations

from foundry.training.parity import compare_training_summaries


def _summary(tokens: int) -> dict[str, object]:
    return {
        "base_model_id": "base",
        "base_revision": "revision",
        "recipe_sha256": "recipe",
        "requirements_lock_sha256": "lock",
        "sft_format_sha256": "format",
        "software_versions": {"torch": "version"},
        "cuda_runtime": "12.1",
        "optimizer_steps": 200,
        "examples_processed": 1600,
        "padded_training_tokens": 819200,
        "nonpadding_training_tokens": tokens,
        "training_source_records": 450,
        "training_truncated_records": 0,
        "trainable_parameters": 10,
        "total_parameters": 100,
    }


def test_parity_passes_at_two_percent_boundary() -> None:
    result = compare_training_summaries(_summary(98_000), _summary(100_000))
    assert result["metadata_parity_passed"] is True
    assert result["nonpadding_token_parity_passed"] is True


def test_parity_rejects_material_loss_token_difference() -> None:
    result = compare_training_summaries(_summary(88_000), _summary(100_000))
    assert result["padded_token_parity_passed"] is True
    assert result["nonpadding_token_parity_passed"] is False
