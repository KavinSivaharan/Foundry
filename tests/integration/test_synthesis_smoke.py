from foundry.synthesis.generators.bookkeeping import (
    generate_bookkeeping,
    verify_bookkeeping_dag,
    verify_bookkeeping_ledger,
)
from foundry.synthesis.generators.discrete import (
    generate_discrete,
    verify_discrete_constructive,
    verify_discrete_enumeration,
)
from foundry.synthesis.generators.rates import (
    generate_rates,
    verify_rate_equation,
    verify_rate_inverse,
)
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.verification import validate_final_answer_contract


def test_all_three_families_share_exact_output_contract_and_dual_verification() -> None:
    cases = (
        (generate_bookkeeping, verify_bookkeeping_dag, verify_bookkeeping_ledger),
        (generate_rates, verify_rate_equation, verify_rate_inverse),
        (generate_discrete, verify_discrete_constructive, verify_discrete_enumeration),
    )
    for index, (generate, primary_verify, independent_verify) in enumerate(cases):
        draft = generate(
            seed=500 + index,
            difficulty=DifficultyLevel.HARD,
            variant=index,
            output_contract_enabled=True,
        )
        primary = primary_verify(draft)
        independent = independent_verify(draft)

        assert primary.answer == independent.answer == draft.canonical_final_answer
        assert primary.method_family != independent.method_family
        assert validate_final_answer_contract(
            draft.training_completion, draft.canonical_final_answer
        )
