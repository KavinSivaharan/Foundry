from foundry.training.assistant_only_token_matching import (
    _stable_occurrences,
)
from foundry.training.token_matching import TokenCensusEntry


def _entry(index: int, tokens: int) -> TokenCensusEntry:
    return TokenCensusEntry(
        synthetic_id=f"fixture-{index:03d}",
        dataset_arm="generic_control",
        family="fixture_family",
        submode="fixture_mode",
        difficulty="easy",
        output_contract_enabled=False,
        formatted_input_tokens=100,
        loss_bearing_tokens=tokens,
        truncated_tokens=0,
        labels_entirely_masked=False,
        formatted_example_sha256=f"{index:064x}",
    )


def test_stable_occurrences_cover_every_example_and_balance_repeats() -> None:
    entries = tuple(_entry(index, 20 + index % 3) for index in range(450))
    first = _stable_occurrences(entries, 1375, "generic_control")
    second = _stable_occurrences(entries, 1375, "generic_control")
    assert first == second
    assert len(first) == 1375
    counts = {entry.synthetic_id: 0 for entry in entries}
    for occurrence in first:
        counts[occurrence.synthetic_id] += 1
    assert set(counts.values()) == {3, 4}
