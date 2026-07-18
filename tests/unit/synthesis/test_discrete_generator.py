import inspect

from foundry.synthesis.generators.discrete import (
    generate_discrete,
    validate_discrete_constraints,
    verify_discrete_constructive,
    verify_discrete_enumeration,
)
from foundry.synthesis.schema import DifficultyLevel


def test_all_discrete_modes_are_unique_and_dually_verified() -> None:
    for variant in range(4):
        draft = generate_discrete(
            seed=70 + variant,
            difficulty=DifficultyLevel.MEDIUM,
            variant=variant,
            output_contract_enabled=variant == 0,
        )
        replay = generate_discrete(
            seed=70 + variant,
            difficulty=DifficultyLevel.MEDIUM,
            variant=variant,
            output_contract_enabled=variant == 0,
        )
        primary = verify_discrete_constructive(draft)
        independent = verify_discrete_enumeration(draft)

        assert draft == replay
        assert primary.success and independent.success
        assert primary.method_family != independent.method_family
        assert primary.answer == independent.answer == draft.canonical_final_answer
        assert not validate_discrete_constraints(draft)


def test_hard_discrete_candidate_expands_the_domain() -> None:
    easy = generate_discrete(
        seed=90,
        difficulty=DifficultyLevel.EASY,
        variant=0,
        output_contract_enabled=False,
    )
    hard = generate_discrete(
        seed=90,
        difficulty=DifficultyLevel.HARD,
        variant=0,
        output_contract_enabled=False,
    )

    assert hard.verifier_payload["total"] > easy.verifier_payload["total"]  # type: ignore[operator]


def test_every_discrete_mode_meets_documented_difficulty_ranges() -> None:
    expected = {
        DifficultyLevel.EASY: (9, 35),
        DifficultyLevel.MEDIUM: (36, 80),
        DifficultyLevel.HARD: (81, 200),
    }
    for difficulty, (low, high) in expected.items():
        for variant in range(4):
            draft = generate_discrete(
                seed=400 + variant,
                difficulty=difficulty,
                variant=variant,
                output_contract_enabled=False,
            )
            evidence = draft.verifier_payload["difficulty_evidence"]
            assert isinstance(evidence, dict)
            assert low <= evidence["domain_size"] <= high  # type: ignore[operator]
            assert evidence["independent_constraints"] in {1, 2}
            assert evidence["dependency_depth"] >= 1  # type: ignore[operator]


def test_dual_capacity_constraints_are_not_tied() -> None:
    for difficulty in DifficultyLevel:
        draft = generate_discrete(
            seed=515,
            difficulty=difficulty,
            variant=3,
            output_contract_enabled=False,
        )
        payload = draft.verifier_payload
        first = payload["first_resource"] // payload["first_per"]  # type: ignore[operator]
        second = payload["second_resource"] // payload["second_per"]  # type: ignore[operator]
        assert first != second


def test_output_track_has_one_canonical_terminal_line() -> None:
    draft = generate_discrete(
        seed=111,
        difficulty=DifficultyLevel.HARD,
        variant=3,
        output_contract_enabled=True,
    )

    final_lines = [
        line for line in draft.training_completion.splitlines() if line.startswith("Final answer:")
    ]
    assert final_lines == [f"Final answer: {draft.canonical_final_answer.render()}"]


def test_discrete_generator_public_interface_is_benchmark_blind() -> None:
    parameters = set(inspect.signature(generate_discrete).parameters)

    assert parameters == {"seed", "difficulty", "variant", "output_contract_enabled"}
    assert not {"question", "answer", "benchmark", "examples"} & parameters
