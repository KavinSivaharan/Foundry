from foundry.training.token_matched_parity import compare_token_matched_runs


def _summary(tokens: int) -> dict[str, object]:
    return {
        "recipe_id": "foundry-token-matched-qlora-v2",
        "recipe_sha256": "recipe",
        "selected_method": "whole_example_token_budgeted_gradient_accumulation",
        "base_recipe_sha256": "base-recipe",
        "base_model_id": "model",
        "base_revision": "revision",
        "requirements_lock_sha256": "lock",
        "sft_format_sha256": "format",
        "software_versions": {"torch": "2.5.1"},
        "cuda_runtime": "12.1",
        "gpu_name": "gpu",
        "optimizer_steps": 4,
        "scheduler_steps": 4,
        "training_source_records": 450,
        "training_truncated_records": 0,
        "validation_source_records": 50,
        "validation_truncated_records": 0,
        "trainable_parameters": 1,
        "total_parameters": 2,
        "only_lora_trainable": True,
        "base_loaded_in_4bit": True,
        "actual_loss_bearing_tokens": tokens,
        "token_count_matches_schedule": True,
        "losses_all_finite": True,
        "gradients_all_finite": True,
        "adapter_offline_reload_ok": True,
        "development_benchmark_exposure_during_training": False,
    }


def test_token_matched_parity_passes_at_smoke_measurements() -> None:
    result = compare_token_matched_runs(_summary(5464), _summary(5440))
    assert result["relative_token_difference"] == 24 / 5464
    assert result["parity_gate_passed"] is True


def test_token_matched_parity_fails_above_half_percent() -> None:
    result = compare_token_matched_runs(_summary(5464), _summary(5400))
    assert result["parity_gate_passed"] is False
