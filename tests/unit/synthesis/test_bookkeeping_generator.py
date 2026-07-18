import inspect

from foundry.synthesis.generators.bookkeeping import (
    generate_bookkeeping,
    validate_bookkeeping_constraints,
    verify_bookkeeping_dag,
    verify_bookkeeping_ledger,
)
from foundry.synthesis.schema import DifficultyLevel


def test_bookkeeping_is_deterministic_and_verifiers_agree() -> None:
    first = generate_bookkeeping(
        seed=17,
        difficulty=DifficultyLevel.MEDIUM,
        variant=5,
        output_contract_enabled=True,
    )
    second = generate_bookkeeping(
        seed=17,
        difficulty=DifficultyLevel.MEDIUM,
        variant=5,
        output_contract_enabled=True,
    )

    assert first == second
    assert first.structure_sha256 == second.structure_sha256
    assert verify_bookkeeping_dag(first).answer == verify_bookkeeping_ledger(first).answer
    assert verify_bookkeeping_dag(first).answer == first.canonical_final_answer
    assert not validate_bookkeeping_constraints(first)


def test_bookkeeping_difficulty_controls_dependency_depth() -> None:
    easy = generate_bookkeeping(
        seed=3,
        difficulty=DifficultyLevel.EASY,
        variant=0,
        output_contract_enabled=False,
    )
    hard = generate_bookkeeping(
        seed=3,
        difficulty=DifficultyLevel.HARD,
        variant=0,
        output_contract_enabled=False,
    )

    assert len(easy.latent_program.steps) == 2
    assert len(hard.latent_program.steps) == 4


def test_bookkeeping_grouping_is_exact_and_uses_distinct_verifiers() -> None:
    draft = generate_bookkeeping(
        seed=29,
        difficulty=DifficultyLevel.HARD,
        variant=4,
        output_contract_enabled=True,
    )
    primary = verify_bookkeeping_dag(draft)
    independent = verify_bookkeeping_ledger(draft)

    assert draft.latent_program.steps[-1].operation == "divide"
    assert primary.success and independent.success
    assert primary.method_family != independent.method_family
    assert primary.answer == independent.answer == draft.canonical_final_answer


def test_generator_public_interface_cannot_receive_benchmark_content() -> None:
    parameters = set(inspect.signature(generate_bookkeeping).parameters)

    assert parameters == {"seed", "difficulty", "variant", "output_contract_enabled"}
    assert not {"question", "answer", "benchmark", "examples"} & parameters
