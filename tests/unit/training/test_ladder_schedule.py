from foundry.training.ladder_schedule import (
    BLOCK_TOKENS,
    CHECKPOINTS,
    build_ladder_schedule,
    select_exact_block,
)
from foundry.training.token_matching import TokenCensusEntry


def _entry(index: int, tokens: int = 40) -> TokenCensusEntry:
    return TokenCensusEntry(
        synthetic_id=f"item-{index:03d}",
        dataset_arm="generic_control",
        family=("bookkeeping", "rate", "discrete")[index % 3],
        submode="fixture",
        difficulty=("easy", "medium", "hard")[index % 3],
        output_contract_enabled=index % 5 == 0,
        formatted_input_tokens=100,
        loss_bearing_tokens=tokens,
        truncated_tokens=0,
        labels_entirely_masked=False,
        formatted_example_sha256=f"{index:064x}",
    )


def test_ladder_schedule_is_exact_and_deterministic() -> None:
    entries = tuple(_entry(index) for index in range(100))
    first = build_ladder_schedule(entries, format_id="v4", arm="generic_control")
    second = build_ladder_schedule(entries, format_id="v4", arm="generic_control")
    assert first == second
    assert len(first) == 32
    assert all(step.occurrences for step in first)
    for checkpoint in CHECKPOINTS:
        assert sum(step.loss_bearing_tokens for step in first[:checkpoint]) == (
            BLOCK_TOKENS * checkpoint // 8
        )
    pairs = [
        (item.synthetic_id, item.occurrence_index) for step in first for item in step.occurrences
    ]
    assert len(pairs) == len(set(pairs))


def test_exact_block_fails_when_whole_examples_cannot_reach_budget() -> None:
    entries = tuple(_entry(index, tokens=41) for index in range(20))
    try:
        select_exact_block(entries, format_id="v3", arm="targeted", block=0)
    except ValueError as error:
        assert "no exact token subset" in str(error)
    else:
        raise AssertionError("insufficient ladder block should fail closed")
