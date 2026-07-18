"""Typed natural-language realization for procedural synthesis."""

from foundry.synthesis.realization.compiler import compile_problem, select_plan
from foundry.synthesis.realization.contracts import validate_realization
from foundry.synthesis.realization.ir import (
    BookkeepingProblemIR,
    CompiledRealization,
    DiscreteProblemIR,
    ProblemIR,
    RateProblemIR,
)

__all__ = [
    "BookkeepingProblemIR",
    "CompiledRealization",
    "DiscreteProblemIR",
    "ProblemIR",
    "RateProblemIR",
    "compile_problem",
    "select_plan",
    "validate_realization",
]
