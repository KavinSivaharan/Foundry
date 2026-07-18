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
from foundry.synthesis.realization.model_contracts import (
    RealizationRequest,
    RealizationResponse,
)
from foundry.synthesis.realization.validation import (
    fill_validated_template,
    parse_realization_response,
    validate_realization_response,
)

__all__ = [
    "BookkeepingProblemIR",
    "CompiledRealization",
    "DiscreteProblemIR",
    "ProblemIR",
    "RateProblemIR",
    "RealizationRequest",
    "RealizationResponse",
    "compile_problem",
    "fill_validated_template",
    "parse_realization_response",
    "select_plan",
    "validate_realization",
    "validate_realization_response",
]
