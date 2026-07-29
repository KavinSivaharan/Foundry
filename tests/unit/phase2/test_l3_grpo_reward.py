from __future__ import annotations

from foundry.phase2.l3_grpo_reward import (
    REWARD_CONFIGURATION,
    TaskRewardMetadata,
    calibrate_reward_contract,
    reward_configuration_sha256,
    reward_contract_sha256,
    reward_fixture_sha256,
    reward_implementation_sha256,
    score_reward,
)


def _metadata() -> TaskRewardMetadata:
    return TaskRewardMetadata(
        source_id="fixture",
        prompt="Report the completed workshop count.",
        canonical_answer="7",
        answer_type="integer",
        family="fixture",
        difficulty="easy",
        verifier_metadata_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )


def test_reward_safety_fixture_replays_deterministically() -> None:
    first = calibrate_reward_contract()
    second = calibrate_reward_contract()
    assert first == second
    assert first["fixture_count"] == 8
    assert first["correct_greater_than_incorrect"] is True
    assert first["unsafe_or_malformed_cannot_outscore_correct"] is True
    for value in (
        reward_configuration_sha256(),
        reward_fixture_sha256(),
        reward_implementation_sha256(),
        reward_contract_sha256(),
        first["calibration_sha256"],
    ):
        assert isinstance(value, str)
        assert len(value) == 64


def test_task_reward_orders_correct_incorrect_malformed_and_unsafe() -> None:
    metadata = _metadata()
    correct = score_reward(metadata, "Work gives seven.\nFinal answer: 7")
    incorrect = score_reward(metadata, "Final answer: 8")
    malformed = score_reward(metadata, "No terminal number is available.")
    unsafe = score_reward(metadata, "Could it be seven?\nFinal answer: 7")
    backend = score_reward(metadata, "", backend_status="failure")
    assert correct.total > incorrect.total
    assert correct.total > malformed.total
    assert correct.total > unsafe.total
    assert correct.total > backend.total
    assert correct.answer_correct and correct.extractable and correct.exact_format
    assert unsafe.question_generation
    assert backend.backend_failure


def test_reward_has_no_learned_or_benchmark_scoring_path() -> None:
    assert REWARD_CONFIGURATION["learned_reward_model"] is False
    assert REWARD_CONFIGURATION["llm_judge"] is False
    assert REWARD_CONFIGURATION["human_in_the_loop_during_training"] is False
    assert REWARD_CONFIGURATION["benchmark_reward"] is False
    assert REWARD_CONFIGURATION["response_length_reward"] is False
