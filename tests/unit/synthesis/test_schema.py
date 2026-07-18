from dataclasses import replace

import pytest

from foundry.synthesis.schema import (
    DifficultyLevel,
    ExactValue,
    LatentProgramSpec,
    ProgramParameter,
    ProgramStep,
    ProvenanceMetadata,
    ReviewStatus,
    SyntheticExample,
    ValidityStatus,
    VerificationEvidence,
)

_HASH = "a" * 64


def _evidence(verifier: str, method: str, answer: ExactValue) -> VerificationEvidence:
    return VerificationEvidence(verifier, "1", method, answer, _HASH)


def _example() -> SyntheticExample:
    answer = ExactValue(5, 2)
    program = LatentProgramSpec(
        program_family="original_fixture_reservoir_delta",
        parameters=(
            ProgramParameter("start", ExactValue(1)),
            ProgramParameter("inflow", ExactValue(3, 2)),
        ),
        steps=(ProgramStep("total", "add", ("start", "inflow"), answer),),
        constraints=("all quantities are exact",),
        answer_symbol="total",
    )
    return SyntheticExample(
        synthetic_example_id="syn-v1-000001",
        generator_version="test-v1",
        random_seed=7,
        target_failure_category="rate_ratio_percentage_or_average",
        secondary_skill_tags=("exact_fraction",),
        difficulty_level=DifficultyLevel.EASY,
        latent_program=program,
        rendered_question="An original fixture asks for an exact reservoir change.",
        deterministic_solution_trace=("Add the two exact quantities.",),
        canonical_final_answer=answer,
        required_final_answer_format="Final answer: <canonical-number>",
        primary_verification_evidence=_evidence("dag-executor", "symbolic-dag", answer),
        independent_verification_evidence=_evidence("inverse-check", "inverse", answer),
        validity_status=ValidityStatus.ACCEPTED,
        rejection_reason=None,
        normalized_text_hash=_HASH,
        latent_program_hash="b" * 64,
        provenance=ProvenanceMetadata(
            source_kind="independent_procedural",
            generator_config_sha256=_HASH,
            taxonomy_sha256="b" * 64,
            benchmark_content_used_as_generator_input=False,
        ),
        contamination_check_status=ReviewStatus.PASSED,
        deduplication_status=ReviewStatus.PASSED,
    )


def test_complete_example_accepts_exact_fraction_and_independent_evidence() -> None:
    example = _example()

    assert example.canonical_final_answer.render() == "5/2"
    assert example.canonical_final_answer.fraction.numerator == 5


def test_example_rejects_same_verification_method_family() -> None:
    example = _example()
    duplicate_method = _evidence("other-id", "symbolic-dag", ExactValue(5, 2))

    with pytest.raises(ValueError, match="different method family"):
        replace(example, independent_verification_evidence=duplicate_method)


def test_provenance_rejects_benchmark_generator_input() -> None:
    with pytest.raises(ValueError, match="benchmark content cannot be generator input"):
        ProvenanceMetadata("independent_procedural", _HASH, _HASH, True)


def test_example_rejects_unnormalized_canonical_answer() -> None:
    example = _example()
    answer = ExactValue(10, 4)

    with pytest.raises(ValueError, match="must be normalized"):
        replace(
            example,
            canonical_final_answer=answer,
            primary_verification_evidence=_evidence("dag-executor", "symbolic-dag", answer),
            independent_verification_evidence=_evidence("inverse-check", "inverse", answer),
        )
