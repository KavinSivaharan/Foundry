from collections import Counter

from foundry.phase2.vetted_schedule import Entry, _exact_fill, _segment_occurrences


def test_exact_fill_is_deterministic_and_exact() -> None:
    entries = (
        Entry("a", 7, "vetted", "f", "d"),
        Entry("b", 11, "vetted", "f", "d"),
        Entry("c", 13, "vetted", "f", "d"),
    )
    first = _exact_fill(entries, 55)
    assert first == _exact_fill(entries, 55)
    assert sum(item.tokens for item in first) == 55


def test_segment_occurrences_are_unique_and_exact() -> None:
    entries = tuple(
        Entry(f"id-{index:03d}", 5 + index % 3, "vetted", "family", "difficulty")
        for index in range(12)
    )
    counters: Counter[str] = Counter()
    occurrences = _segment_occurrences(entries, 100, 0, counters)
    assert sum(item.tokens for item in occurrences) == 100
    assert len({(item.record_id, item.occurrence_index) for item in occurrences}) == len(
        occurrences
    )
