from foundry.phase2.contamination import (
    _cross_source_duplicate_reasons,
    _duplicate_reasons,
    _reference,
    contiguous_ngrams,
    operation_cues,
)


def _row(source_id: str, question: str, program: str = "program") -> dict[str, object]:
    return {
        "source_id": source_id,
        "combined_question": question,
        "program_sha256": program,
        "program_structure_sha256": program,
    }


def test_twelve_token_matching_is_contiguous_and_normalized() -> None:
    text = "One two three four five six seven eight nine ten eleven twelve thirteen."
    ngrams = contiguous_ngrams(text)

    assert len(ngrams) == 2
    assert tuple("one two three four five six seven eight nine ten eleven twelve".split()) in ngrams
    assert not contiguous_ngrams("only eleven tokens appear in this deliberately short text now")


def test_reference_freezes_exact_number_neutral_and_structure_hashes() -> None:
    first = _reference("a", "A shelf has 12 boxes in 3 rows.")
    second = _reference("b", "A shelf has 20 boxes in 5 rows.")

    assert first.normalized_sha256 != second.normalized_sha256
    assert first.number_neutral_sha256 == second.number_neutral_sha256
    assert first.structure_sha256 == second.structure_sha256


def test_operation_cues_are_deterministic_and_broad_topics_do_not_reject_alone() -> None:
    assert operation_cues("Find the total when each group has four items.") == (
        "addition",
        "multiplication",
    )
    assert operation_cues("Find an unrelated value.") == ()


def test_candidate_duplicate_screening_keeps_smallest_stable_id() -> None:
    rows = [
        _row("b", "A box has 4 items."),
        _row("a", "A box has 4 items."),
        _row("c", "A different box has 8 items."),
    ]

    reasons = _duplicate_reasons(rows)

    assert reasons == {"b": "candidate_exact_duplicate"}


def test_formula_template_duplicate_requires_matching_formula_and_text_template() -> None:
    rows = [
        _row("a", "A box has 4 items.", "same"),
        _row("b", "A box has 8 items.", "same"),
        _row("c", "A crate has 8 items.", "same"),
        _row("d", "A bag has 8 items.", "different"),
    ]

    reasons = _duplicate_reasons(rows)

    assert reasons["b"] == "candidate_formula_text_duplicate"
    assert "c" not in reasons
    assert "d" not in reasons


def test_cross_source_duplicates_are_rejected_content_independently() -> None:
    reference = [_row("asdiv-a", "A box has 4 items.", "structure")]
    candidates = [
        _row("mathqa-a", "A box has 4 items.", "other"),
        _row("mathqa-b", "A box has 9 items.", "structure"),
        _row("mathqa-c", "A crate has 9 items.", "structure"),
    ]

    reasons = _cross_source_duplicate_reasons(candidates, reference)

    assert reasons == {
        "mathqa-a": "cross_source_exact_duplicate",
        "mathqa-b": "cross_source_formula_text_duplicate",
    }
