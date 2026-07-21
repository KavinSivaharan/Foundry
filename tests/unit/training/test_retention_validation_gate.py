from foundry.training.retention_validation_gate import evaluate_validation_arm


def _summary(arithmetic: float, instruction: float) -> dict[str, object]:
    return {
        "adapter_sha256": "a" * 64,
        "section_metrics": {
            "arithmetic": {"accuracy": arithmetic},
            "format": {"accuracy": 1.0},
            "instruction": {"accuracy": instruction},
        },
        "extractability": 1.0,
        "prompt_echo_rate": 0.0,
        "question_generation": 0,
        "backend_failures": 0,
    }


def test_validation_gate_accepts_thresholds() -> None:
    result = evaluate_validation_arm(base=_summary(1.0, 0.92), retention=_summary(0.9, 0.9))
    assert result["passed"] is True


def test_validation_gate_rejects_instruction_failure() -> None:
    result = evaluate_validation_arm(base=_summary(1.0, 0.92), retention=_summary(1.0, 0.84))
    assert result["passed"] is False
    assert result["failed_checks"] == ["instruction_following"]
