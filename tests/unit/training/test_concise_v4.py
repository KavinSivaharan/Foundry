from typing import Any

import pytest

from foundry.training.concise_v4 import concise_completion, concise_reasoning
from foundry.training.config import concise_assistant_v4_format_contract_sha256


def _record(mode: str, trace: list[str], answer: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "deterministic_solution_trace": trace,
        "canonical_final_answer": answer,
        "primary_verifier_success": True,
        "independent_verifier_success": True,
        "verifier_agreement": True,
    }


@pytest.mark.parametrize(
    ("mode", "trace", "answer", "expected"),
    [
        (
            "inventory",
            [
                "The typed ledger starts with 10 items.",
                "Update 1 changes the item balance by +5, giving 15.",
                "Update 2 changes the item balance by -3, giving 12.",
            ],
            "12",
            "10 + 5 - 3 = 12.",
        ),
        (
            "ratio_scale",
            [
                "Divide 42 by the first ratio part 6 to obtain scale 7.",
                "Multiply scale 7 by the second part 3.",
            ],
            "21",
            "42 / 6 = 7.\n7 × 3 = 21.",
        ),
        (
            "complete_packages",
            ["Use exact integer division: 21 divided by 5 gives 4."],
            "4",
            "21 = 5 × 4 + 1.",
        ),
        (
            "weighted_average",
            [
                "The exact weighted total is 114.",
                "Divide by the total weight 11 to obtain the weighted average.",
            ],
            "114/11",
            "114 / 11 = 114/11.",
        ),
    ],
)
def test_concise_reasoning_replays_supported_modes(
    mode: str, trace: list[str], answer: str, expected: str
) -> None:
    record = _record(mode, trace, answer)
    lines, replay = concise_reasoning(record)
    assert "\n".join(lines) == expected
    assert str(replay) == answer
    assert concise_completion(record).endswith(f"Final answer: {answer}")


def test_v4_fails_closed_on_trace_state_mismatch() -> None:
    record = _record(
        "inventory",
        [
            "The typed ledger starts with 10 items.",
            "Update 1 changes the item balance by +5, giving 14.",
        ],
        "14",
    )
    with pytest.raises(ValueError, match="state does not replay"):
        concise_reasoning(record)


def test_v4_contract_hash_is_stable_length() -> None:
    assert len(concise_assistant_v4_format_contract_sha256()) == 64
