from __future__ import annotations

import inspect

import pytest

import foundry.training.grpo_reward as subject
from foundry.training.retention import RetentionItem


def _synthetic(
    *, prompt: str = "Compute the original fixture total.", answer: str = "7"
) -> subject.SyntheticRewardMetadata:
    return subject.SyntheticRewardMetadata(
        synthetic_id="synthetic-fixture-1",
        prompt=prompt,
        canonical_answer=answer,
        family="fixture_bookkeeping",
        submode="fixture_total",
        difficulty="medium",
        output_contract_enabled=True,
        verifier_metadata_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )


def _replay(
    *,
    prompt: str = "Return the material name only.",
    kind: str = "exact_text",
    expected: str = "cedar",
) -> subject.ReplayRewardMetadata:
    item = RetentionItem(
        item_id="replay-fixture-1",
        section="instruction",
        skill="fixture_exact_text",
        kind=kind,  # type: ignore[arg-type]
        prompt=prompt,
        expected=expected,
    )
    return subject.ReplayRewardMetadata(
        replay_id=item.item_id,
        prompt=prompt,
        retention_item=item,
        scorer_metadata_sha256="c" * 64,
        provenance_sha256="d" * 64,
    )


def test_correct_canonical_answer_receives_all_synthetic_positive_components() -> None:
    result = subject.score_reward(_synthetic(), "Work gives seven.\nFinal answer: 7")

    assert result.correctness == 1.0
    assert result.extractability == 0.10
    assert result.exact_contract == 0.05
    assert result.total == 1.15


def test_wrong_but_extractable_answer_does_not_receive_correctness() -> None:
    result = subject.score_reward(_synthetic(), "Final answer: 8")

    assert result.correctness == 0.0
    assert result.extractability == 0.10
    assert result.exact_contract == 0.0
    assert result.total == 0.10


def test_correct_answer_in_noncontract_format_keeps_correctness_only() -> None:
    result = subject.score_reward(_synthetic(), "Therefore, the answer is 7.")

    assert result.correctness == 1.0
    assert result.extractability == 0.10
    assert result.exact_contract == 0.0
    assert result.total == 1.10


def test_conflicting_answers_receive_penalty_and_no_positive_reward() -> None:
    result = subject.score_reward(_synthetic(), "Final answer: 7\nFinal answer: 8")

    assert result.conflicting_answers is True
    assert result.conflicting_answers_penalty == -0.25
    assert result.correctness == 0.0
    assert result.extractability == 0.0
    assert result.total == -0.25


def test_prompt_echo_and_question_generation_share_one_safety_penalty() -> None:
    prompt = "Return the total for the described original workshop bundle."
    echoed = subject.score_reward(_synthetic(prompt=prompt), f"{prompt}\nFinal answer: 7")
    questioned = subject.score_reward(
        _synthetic(prompt="Report the total."),
        "Could the result be seven?\nFinal answer: 7",
    )

    assert echoed.prompt_echo is True
    assert echoed.echo_or_question_penalty == -0.25
    assert questioned.question_generation is True
    assert questioned.echo_or_question_penalty == -0.25
    assert echoed.total == questioned.total == 0.90


def test_truncation_is_additive_to_otherwise_earned_components() -> None:
    result = subject.score_reward(_synthetic(), "Final answer: 7", generation_truncated=True)

    assert result.generation_truncated is True
    assert result.truncation_penalty == -0.10
    assert result.correctness == 1.0
    assert result.extractability == 0.10
    assert result.exact_contract == 0.05
    assert result.total == 1.05


def test_fraction_answers_are_compared_exactly() -> None:
    result = subject.score_reward(
        _synthetic(answer="3/2"), "The ratio is exact.\nFinal answer: 3/2"
    )
    assert result.total == 1.15


def test_exact_contract_rejects_wrong_case_duplicate_line_and_trailing_text() -> None:
    wrong_case = subject.score_reward(_synthetic(), "final answer: 7")
    duplicate = subject.score_reward(_synthetic(), "Final answer: 7\nFinal answer: 7")
    trailing = subject.score_reward(_synthetic(), "Final answer: 7\nAdditional explanation.")

    assert wrong_case.exact_contract == 0.0
    assert duplicate.exact_contract == 0.0
    assert trailing.exact_contract == 0.0


def test_replay_uses_frozen_prompt_specific_scorer_and_format_contract() -> None:
    correct = subject.score_reward(_replay(), "cedar")
    wrong = subject.score_reward(_replay(), "The answer is cedar.")

    assert correct.replay_task_correctness == 1.0
    assert correct.replay_format_contract == 0.05
    assert correct.total == 1.05
    assert wrong.replay_task_correctness == 0.0
    assert wrong.replay_format_contract == 0.0


def test_replay_scorer_is_called_only_for_replay_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = subject.score_response
    calls: list[str] = []

    def recording_score(item: RetentionItem, response: str) -> dict[str, object]:
        calls.append(item.item_id)
        return original(item, response)

    monkeypatch.setattr(subject, "score_response", recording_score)
    subject.score_synthetic_reward(_synthetic(), "Final answer: 7")
    assert calls == []

    calls.clear()
    subject.score_replay_reward(_replay(), "cedar")
    assert calls == ["replay-fixture-1"]


def test_synthetic_reward_never_uses_replay_scorer_for_correctness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deliberately_wrong_score(_item: RetentionItem, _response: str) -> dict[str, object]:
        return {
            "correct": False,
            "extractable": False,
            "malformed": True,
            "prompt_echo": False,
            "question_generation": False,
            "exact_format": False,
            "extracted_hash": None,
        }

    monkeypatch.setattr(subject, "score_response", deliberately_wrong_score)
    result = subject.score_synthetic_reward(_synthetic(), "Final answer: 7")
    assert result.correctness == 1.0
    assert result.total == 1.15


def test_model_visible_interface_cannot_leak_labels_or_verifier_metadata() -> None:
    metadata = _synthetic(prompt="Report the original fixture count.", answer="731")

    visible = subject.model_visible_prompt(metadata)

    assert visible == metadata.prompt
    assert metadata.canonical_answer not in visible
    assert metadata.verifier_metadata_sha256 not in visible
    assert metadata.provenance_sha256 not in visible


def test_replay_metadata_requires_exact_frozen_item_identity() -> None:
    item = RetentionItem(
        item_id="frozen-replay-id",
        section="format",
        skill="fixture",
        kind="exact_text",
        prompt="Return OK only.",
        expected="OK",
    )
    with pytest.raises(ValueError, match="replay_id"):
        subject.ReplayRewardMetadata(
            replay_id="different-id",
            prompt=item.prompt,
            retention_item=item,
            scorer_metadata_sha256="c" * 64,
            provenance_sha256="d" * 64,
        )


@pytest.mark.parametrize("answer", [" 7", "7.0", "6/2", "not-a-number", "1/0"])
def test_noncanonical_or_invalid_answer_metadata_fails_closed(answer: str) -> None:
    with pytest.raises(ValueError, match="canonical_answer"):
        _synthetic(answer=answer)


def test_reward_is_deterministic_and_calibration_is_frozen() -> None:
    metadata = _synthetic()
    completion = "The exact total is seven.\nFinal answer: 7"

    first = subject.score_reward(metadata, completion).as_dict()
    second = subject.score_reward(metadata, completion).as_dict()
    calibration_one = subject.calibrate_reward_contract()
    calibration_two = subject.calibrate_reward_contract()

    assert first == second
    assert calibration_one == calibration_two
    assert calibration_one["all_passed"] is True
    assert calibration_one["fixture_count"] == 8


def test_configuration_forbids_unapproved_rewards_and_external_judges() -> None:
    config = subject.REWARD_CONFIGURATION
    assert config["response_length_reward"] is False
    assert config["reasoning_length_reward"] is False
    assert config["style_reward"] is False
    assert config["verbosity_reward"] is False
    assert config["benchmark_reward"] is False
    assert config["category_multipliers"] is False
    assert config["learned_reward_model"] is False
    assert config["llm_judge"] is False


def test_reward_module_has_no_benchmark_model_or_network_access() -> None:
    source = inspect.getsource(subject).lower()
    forbidden = (
        "gsm1k",
        "benchmark.py",
        "requests.",
        "http://",
        "https://",
        "automodelfor",
        "from_pretrained",
    )
    assert all(value not in source for value in forbidden)


def test_reward_hashes_are_stable_sha256_values() -> None:
    calibration = subject.calibrate_reward_contract()
    values = (
        subject.reward_implementation_sha256(),
        subject.reward_configuration_sha256(),
        subject.reward_fixture_sha256(),
        calibration["calibration_sha256"],
    )
    assert all(isinstance(value, str) and len(value) == 64 for value in values)
