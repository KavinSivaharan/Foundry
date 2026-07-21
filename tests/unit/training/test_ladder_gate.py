from foundry.training.ladder_gate import evaluate_checkpoint


def _base() -> dict[str, object]:
    return {
        "section_metrics": {
            "arithmetic": {"accuracy": 1.0},
            "format": {"accuracy": 14 / 15},
            "instruction": {"accuracy": 14 / 15},
        }
    }


def _training() -> dict[str, object]:
    return {
        "losses_all_finite": True,
        "token_count_matches_schedule": True,
        "checkpoints": {
            "8": {
                "synthetic_validation_loss": 0.5,
                "cumulative_actual_loss_bearing_tokens": 3600,
            }
        },
    }


def _retention(instruction: float = 0.9) -> dict[str, object]:
    return {
        "adapter_sha256": "a" * 64,
        "section_metrics": {
            "arithmetic": {"accuracy": 0.9},
            "format": {"accuracy": 0.9},
            "instruction": {"accuracy": instruction},
        },
        "extractability": 0.9,
        "prompt_echo_rate": 0.05,
        "question_generation": 0,
        "backend_failures": 0,
    }


def test_checkpoint_gate_accepts_exact_boundaries() -> None:
    result = evaluate_checkpoint(
        base=_base(), training=_training(), retention=_retention(), checkpoint=8
    )
    assert result["passed"] is True
    assert result["failed_checks"] == []


def test_checkpoint_gate_rejects_instruction_below_boundary() -> None:
    result = evaluate_checkpoint(
        base=_base(), training=_training(), retention=_retention(0.899), checkpoint=8
    )
    assert result["passed"] is False
    assert result["failed_checks"] == ["instruction_following"]
