import inspect
from fractions import Fraction

from foundry.synthesis.generators.rates import (
    generate_rates,
    validate_rate_constraints,
    verify_rate_equation,
    verify_rate_inverse,
)
from foundry.synthesis.schema import DifficultyLevel


def test_all_rate_modes_are_deterministic_and_dually_verified() -> None:
    for variant in range(5):
        first = generate_rates(
            seed=100 + variant,
            difficulty=DifficultyLevel.MEDIUM,
            variant=variant,
            output_contract_enabled=variant % 2 == 0,
        )
        second = generate_rates(
            seed=100 + variant,
            difficulty=DifficultyLevel.MEDIUM,
            variant=variant,
            output_contract_enabled=variant % 2 == 0,
        )
        primary = verify_rate_equation(first)
        independent = verify_rate_inverse(first)

        assert first == second
        assert primary.success and independent.success
        assert primary.method_family != independent.method_family
        assert primary.answer == independent.answer == first.canonical_final_answer
        assert not validate_rate_constraints(first)


def test_hard_weighted_average_adds_a_third_group() -> None:
    medium = generate_rates(
        seed=31,
        difficulty=DifficultyLevel.MEDIUM,
        variant=3,
        output_contract_enabled=False,
    )
    hard = generate_rates(
        seed=31,
        difficulty=DifficultyLevel.HARD,
        variant=3,
        output_contract_enabled=False,
    )

    assert len(medium.verifier_payload["weights"]) == 2  # type: ignore[arg-type]
    assert len(hard.verifier_payload["weights"]) == 3  # type: ignore[arg-type]
    assert verify_rate_equation(hard).answer == verify_rate_inverse(hard).answer


def test_percentage_uses_exact_rational_arithmetic() -> None:
    draft = generate_rates(
        seed=47,
        difficulty=DifficultyLevel.EASY,
        variant=2,
        output_contract_enabled=True,
    )

    assert verify_rate_equation(draft).answer == draft.canonical_final_answer
    base = draft.verifier_payload["base"]
    percent = draft.verifier_payload["percent"]
    assert isinstance(base, int) and isinstance(percent, int)
    assert draft.canonical_final_answer.fraction == Fraction(base * percent, 100)


def test_rate_generator_public_interface_is_benchmark_blind() -> None:
    parameters = set(inspect.signature(generate_rates).parameters)

    assert parameters == {"seed", "difficulty", "variant", "output_contract_enabled"}
    assert not {"question", "answer", "benchmark", "examples"} & parameters
