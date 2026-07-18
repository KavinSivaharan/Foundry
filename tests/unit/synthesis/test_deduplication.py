import re
from dataclasses import replace

from foundry.synthesis.contamination import DevelopmentQuestion
from foundry.synthesis.deduplication import DeduplicationIndex
from foundry.synthesis.generators.bookkeeping import generate_bookkeeping
from foundry.synthesis.schema import DifficultyLevel


def _draft():
    return generate_bookkeeping(
        seed=15,
        difficulty=DifficultyLevel.MEDIUM,
        variant=4,
        output_contract_enabled=False,
    )


def test_exact_development_copy_is_rejected() -> None:
    draft = _draft()
    index = DeduplicationIndex((DevelopmentQuestion("a" * 64, 0, draft.rendered_question),))

    assert index.screen(draft).rejection_reason == "exact_normalized_text"


def test_number_swap_is_rejected_before_semantics() -> None:
    draft = _draft()
    changed = re.sub(r"\d+", "999", draft.rendered_question, count=1)
    index = DeduplicationIndex((DevelopmentQuestion("b" * 64, 1, changed),))

    assert index.screen(draft).rejection_reason == "numeric_template_copy"


def test_generated_structural_copy_is_rejected() -> None:
    draft = _draft()
    index = DeduplicationIndex(())
    index.add_candidate(draft)
    rewritten_text = "Entirely rewritten wording."
    rewritten = replace(
        draft,
        candidate_id="syn-other",
        rendered_question=rewritten_text,
        realization=replace(draft.realization, text=rewritten_text),
    )

    assert index.screen(rewritten).rejection_reason == "latent_structure_copy"


def test_unrelated_original_draft_passes_lexical_stages() -> None:
    draft = _draft()
    index = DeduplicationIndex(
        (DevelopmentQuestion("c" * 64, 2, "A violinist tunes a quiet melody before sunrise."),)
    )

    assert index.screen(draft).rejection_reason is None
