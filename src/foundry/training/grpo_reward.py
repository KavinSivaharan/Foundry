"""Deterministic verifier rewards for Foundry's bounded GRPO experiment.

The policy model sees only :func:`model_visible_prompt`.  Canonical answers,
retention references, verifier metadata, and provenance hashes stay on the
trusted reward side of the interface.  This module deliberately has no model,
benchmark, network, or learned-reward dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from foundry.evaluation.answer_extraction import (
    CANONICAL_EXTRACTOR_ID,
    CanonicalExtractionError,
    extract_canonical_number,
)
from foundry.training.retention import RetentionItem
from foundry.training.retention import score_response as score_response

REWARD_CONTRACT_ID = "foundry-verifier-grpo-reward-v1"
type SourceKind = Literal["synthetic", "replay"]

CORRECTNESS_REWARD = 1.0
EXTRACTABILITY_REWARD = 0.10
EXACT_CONTRACT_REWARD = 0.05
TRUNCATION_PENALTY = -0.10
ECHO_OR_QUESTION_PENALTY = -0.25
CONFLICTING_ANSWERS_PENALTY = -0.25

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_FINAL_ANSWER_PREFIX = "Final answer:"


def canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value using Foundry's stable representation."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _canonical_answer(value: str) -> tuple[str, Fraction]:
    _require_text(value, "canonical_answer")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError("canonical_answer must not contain whitespace")
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("canonical_answer must be an exact number") from error
    rendered = (
        str(fraction.numerator)
        if fraction.denominator == 1
        else f"{fraction.numerator}/{fraction.denominator}"
    )
    if value != rendered:
        raise ValueError("canonical_answer must use reduced canonical notation")
    return rendered, fraction


@dataclass(frozen=True)
class SyntheticRewardMetadata:
    """Trusted reward-side metadata for one synthetic arithmetic prompt."""

    synthetic_id: str
    prompt: str
    canonical_answer: str
    family: str
    submode: str
    difficulty: str
    output_contract_enabled: bool
    verifier_metadata_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        for field, value in (
            ("synthetic_id", self.synthetic_id),
            ("prompt", self.prompt),
            ("family", self.family),
            ("submode", self.submode),
        ):
            _require_text(value, field)
        _canonical_answer(self.canonical_answer)
        if self.difficulty not in _DIFFICULTIES:
            raise ValueError("difficulty must be easy, medium, or hard")
        if not isinstance(self.output_contract_enabled, bool):
            raise ValueError("output_contract_enabled must be boolean")
        _require_sha256(self.verifier_metadata_sha256, "verifier_metadata_sha256")
        _require_sha256(self.provenance_sha256, "provenance_sha256")

    @property
    def item_id(self) -> str:
        """Return the stable item identifier without exposing reward metadata."""

        return self.synthetic_id


@dataclass(frozen=True)
class ReplayRewardMetadata:
    """Trusted reward-side metadata for one frozen shared-replay prompt."""

    replay_id: str
    prompt: str
    retention_item: RetentionItem
    scorer_metadata_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.replay_id, "replay_id")
        _require_text(self.prompt, "prompt")
        if self.replay_id != self.retention_item.item_id:
            raise ValueError("replay_id must match the frozen retention item")
        if self.prompt != self.retention_item.prompt:
            raise ValueError("replay prompt must match the frozen retention item")
        _require_sha256(self.scorer_metadata_sha256, "scorer_metadata_sha256")
        _require_sha256(self.provenance_sha256, "provenance_sha256")

    @property
    def item_id(self) -> str:
        """Return the stable item identifier without exposing reward metadata."""

        return self.replay_id


type RewardMetadata = SyntheticRewardMetadata | ReplayRewardMetadata


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable additive components for one completion reward."""

    contract_id: str
    source_kind: SourceKind
    item_id: str
    correctness: float
    extractability: float
    exact_contract: float
    replay_task_correctness: float
    replay_format_contract: float
    truncation_penalty: float
    echo_or_question_penalty: float
    conflicting_answers_penalty: float
    extracted_answer_sha256: str | None
    prompt_echo: bool
    question_generation: bool
    conflicting_answers: bool
    generation_truncated: bool
    total: float

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible audit record."""

        return asdict(self)


def model_visible_prompt(metadata: RewardMetadata) -> str:
    """Return the only text supplied to the policy model.

    In particular, this interface cannot expose the canonical answer, expected
    replay response, verifier metadata, provenance hashes, or reward weights.
    """

    return metadata.prompt


def _synthetic_safety_score(
    metadata: SyntheticRewardMetadata, completion: str
) -> dict[str, object]:
    normalized_prompt = " ".join(metadata.prompt.lower().split())
    normalized_completion = " ".join(completion.lower().split())
    return {
        "prompt_echo": len(normalized_prompt) >= 24 and normalized_prompt in normalized_completion,
        "question_generation": "?" in completion
        or bool(
            re.search(
                r"(?:^|\n)\s*(?:question|problem)\s*:",
                completion,
                re.IGNORECASE,
            )
        ),
    }


def _conflicting_answers(completion: str) -> bool:
    try:
        extract_canonical_number(completion)
    except CanonicalExtractionError as error:
        return error.category == "conflicting_answers"
    return False


def _exact_final_answer_contract(completion: str, canonical_answer: str) -> bool:
    lines = completion.strip().splitlines()
    if not lines:
        return False
    answer_lines = [line for line in lines if line.startswith(_FINAL_ANSWER_PREFIX)]
    return answer_lines == [f"{_FINAL_ANSWER_PREFIX} {canonical_answer}"] and (
        lines[-1] == answer_lines[0]
    )


def _penalties(
    *,
    completion: str,
    prompt_echo: bool,
    question_generation: bool,
    generation_truncated: bool,
) -> tuple[float, float, float, bool]:
    conflicting = _conflicting_answers(completion)
    return (
        TRUNCATION_PENALTY if generation_truncated else 0.0,
        ECHO_OR_QUESTION_PENALTY if prompt_echo or question_generation else 0.0,
        CONFLICTING_ANSWERS_PENALTY if conflicting else 0.0,
        conflicting,
    )


def score_synthetic_reward(
    metadata: SyntheticRewardMetadata,
    completion: str,
    *,
    generation_truncated: bool = False,
) -> RewardBreakdown:
    """Score one synthetic completion with exact executable components."""

    _require_text(completion, "completion")
    _, expected = _canonical_answer(metadata.canonical_answer)
    safety = _synthetic_safety_score(metadata, completion)
    extracted: Fraction | None = None
    try:
        extracted = extract_canonical_number(completion)
    except CanonicalExtractionError:
        pass
    extractability = EXTRACTABILITY_REWARD if extracted is not None else 0.0
    correctness = CORRECTNESS_REWARD if extracted == expected else 0.0
    exact_contract = (
        EXACT_CONTRACT_REWARD
        if _exact_final_answer_contract(completion, metadata.canonical_answer)
        else 0.0
    )
    prompt_echo = bool(safety["prompt_echo"])
    question_generation = bool(safety["question_generation"])
    truncation, echo_or_question, conflicting_penalty, conflicting = _penalties(
        completion=completion,
        prompt_echo=prompt_echo,
        question_generation=question_generation,
        generation_truncated=generation_truncated,
    )
    total = round(
        correctness
        + extractability
        + exact_contract
        + truncation
        + echo_or_question
        + conflicting_penalty,
        10,
    )
    return RewardBreakdown(
        contract_id=REWARD_CONTRACT_ID,
        source_kind="synthetic",
        item_id=metadata.synthetic_id,
        correctness=correctness,
        extractability=extractability,
        exact_contract=exact_contract,
        replay_task_correctness=0.0,
        replay_format_contract=0.0,
        truncation_penalty=truncation,
        echo_or_question_penalty=echo_or_question,
        conflicting_answers_penalty=conflicting_penalty,
        extracted_answer_sha256=(None if extracted is None else canonical_sha256(str(extracted))),
        prompt_echo=prompt_echo,
        question_generation=question_generation,
        conflicting_answers=conflicting,
        generation_truncated=generation_truncated,
        total=total,
    )


def score_replay_reward(
    metadata: ReplayRewardMetadata,
    completion: str,
    *,
    generation_truncated: bool = False,
) -> RewardBreakdown:
    """Score one replay completion only with its frozen prompt-specific scorer."""

    _require_text(completion, "completion")
    score = score_response(metadata.retention_item, completion)
    task_correctness = CORRECTNESS_REWARD if bool(score["correct"]) else 0.0
    format_contract = EXACT_CONTRACT_REWARD if bool(score["exact_format"]) else 0.0
    prompt_echo = bool(score["prompt_echo"])
    question_generation = bool(score["question_generation"])
    truncation, echo_or_question, conflicting_penalty, conflicting = _penalties(
        completion=completion,
        prompt_echo=prompt_echo,
        question_generation=question_generation,
        generation_truncated=generation_truncated,
    )
    total = round(
        task_correctness + format_contract + truncation + echo_or_question + conflicting_penalty,
        10,
    )
    return RewardBreakdown(
        contract_id=REWARD_CONTRACT_ID,
        source_kind="replay",
        item_id=metadata.replay_id,
        correctness=0.0,
        extractability=0.0,
        exact_contract=0.0,
        replay_task_correctness=task_correctness,
        replay_format_contract=format_contract,
        truncation_penalty=truncation,
        echo_or_question_penalty=echo_or_question,
        conflicting_answers_penalty=conflicting_penalty,
        extracted_answer_sha256=(
            score["extracted_hash"] if isinstance(score.get("extracted_hash"), str) else None
        ),
        prompt_echo=prompt_echo,
        question_generation=question_generation,
        conflicting_answers=conflicting,
        generation_truncated=generation_truncated,
        total=total,
    )


def score_reward(
    metadata: RewardMetadata,
    completion: str,
    *,
    generation_truncated: bool = False,
) -> RewardBreakdown:
    """Dispatch one completion to exactly one deterministic reward path."""

    if isinstance(metadata, SyntheticRewardMetadata):
        return score_synthetic_reward(
            metadata, completion, generation_truncated=generation_truncated
        )
    return score_replay_reward(metadata, completion, generation_truncated=generation_truncated)


REWARD_CONFIGURATION: dict[str, object] = {
    "schema_version": 1,
    "contract_id": REWARD_CONTRACT_ID,
    "canonical_extractor_id": CANONICAL_EXTRACTOR_ID,
    "synthetic": {
        "correctness": CORRECTNESS_REWARD,
        "extractability": EXTRACTABILITY_REWARD,
        "exact_contract": EXACT_CONTRACT_REWARD,
    },
    "replay": {
        "task_correctness": CORRECTNESS_REWARD,
        "format_contract": EXACT_CONTRACT_REWARD,
        "scorer": "foundry.training.retention.score_response",
    },
    "safety_penalties": {
        "truncated": TRUNCATION_PENALTY,
        "prompt_echo_or_question_generation": ECHO_OR_QUESTION_PENALTY,
        "conflicting_answers": CONFLICTING_ANSWERS_PENALTY,
    },
    "response_length_reward": False,
    "reasoning_length_reward": False,
    "style_reward": False,
    "verbosity_reward": False,
    "benchmark_reward": False,
    "category_multipliers": False,
    "learned_reward_model": False,
    "llm_judge": False,
}


def reward_configuration_sha256() -> str:
    """Return the frozen reward-configuration identity."""

    return canonical_sha256(REWARD_CONFIGURATION)


def reward_implementation_sha256() -> str:
    """Hash this implementation with checkout-independent line endings."""

    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RewardCalibrationFixture:
    """One original, non-benchmark reward-contract calibration case."""

    fixture_id: str
    source_kind: SourceKind
    prompt: str
    completion: str
    expected_total: float
    generation_truncated: bool = False
    canonical_answer: str | None = None
    replay_kind: str | None = None
    replay_expected: str | None = None


CALIBRATION_FIXTURES: tuple[RewardCalibrationFixture, ...] = (
    RewardCalibrationFixture(
        "synthetic-correct-exact",
        "synthetic",
        "A workshop records a completed count.",
        "The ledger total is seven.\nFinal answer: 7",
        1.15,
        canonical_answer="7",
    ),
    RewardCalibrationFixture(
        "synthetic-wrong-extractable",
        "synthetic",
        "A depot records a completed count.",
        "Final answer: 8",
        0.10,
        canonical_answer="7",
    ),
    RewardCalibrationFixture(
        "synthetic-correct-noncontract",
        "synthetic",
        "A studio records a completed count.",
        "Therefore, the answer is 7.",
        1.10,
        canonical_answer="7",
    ),
    RewardCalibrationFixture(
        "synthetic-conflicting",
        "synthetic",
        "A laboratory records a completed count.",
        "Final answer: 7\nFinal answer: 8",
        -0.25,
        canonical_answer="7",
    ),
    RewardCalibrationFixture(
        "synthetic-prompt-echo",
        "synthetic",
        "Return the total for the described workshop bundle.",
        "Return the total for the described workshop bundle.\nFinal answer: 7",
        0.90,
        canonical_answer="7",
    ),
    RewardCalibrationFixture(
        "synthetic-question-generation",
        "synthetic",
        "Report the depot total.",
        "Could the result be seven?\nFinal answer: 7",
        0.90,
        canonical_answer="7",
    ),
    RewardCalibrationFixture(
        "synthetic-truncated",
        "synthetic",
        "Report the studio total.",
        "Final answer: 7",
        1.05,
        generation_truncated=True,
        canonical_answer="7",
    ),
    RewardCalibrationFixture(
        "replay-exact-text",
        "replay",
        "Return the material name only.",
        "cedar",
        1.05,
        replay_kind="exact_text",
        replay_expected="cedar",
    ),
)


def reward_fixture_sha256() -> str:
    """Return the frozen original-fixture identity."""

    return canonical_sha256([asdict(fixture) for fixture in CALIBRATION_FIXTURES])


def _fixture_metadata(fixture: RewardCalibrationFixture) -> RewardMetadata:
    if fixture.source_kind == "synthetic":
        if fixture.canonical_answer is None:
            raise ValueError("synthetic fixture requires canonical_answer")
        return SyntheticRewardMetadata(
            synthetic_id=fixture.fixture_id,
            prompt=fixture.prompt,
            canonical_answer=fixture.canonical_answer,
            family="original_fixture_arithmetic",
            submode="original_fixture_count",
            difficulty="easy",
            output_contract_enabled=True,
            verifier_metadata_sha256="a" * 64,
            provenance_sha256="b" * 64,
        )
    if fixture.replay_kind not in {"numeric_terminal", "exact_text", "json_exact"}:
        raise ValueError("replay fixture requires a supported scorer kind")
    if fixture.replay_expected is None:
        raise ValueError("replay fixture requires replay_expected")
    item = RetentionItem(
        item_id=fixture.fixture_id,
        section="instruction",
        skill="original_fixture_instruction",
        kind=fixture.replay_kind,  # type: ignore[arg-type]
        prompt=fixture.prompt,
        expected=fixture.replay_expected,
    )
    return ReplayRewardMetadata(
        replay_id=fixture.fixture_id,
        prompt=fixture.prompt,
        retention_item=item,
        scorer_metadata_sha256="c" * 64,
        provenance_sha256="d" * 64,
    )


def calibrate_reward_contract() -> dict[str, object]:
    """Run the frozen fixtures and fail closed on any component drift."""

    rows: list[dict[str, object]] = []
    for fixture in CALIBRATION_FIXTURES:
        result = score_reward(
            _fixture_metadata(fixture),
            fixture.completion,
            generation_truncated=fixture.generation_truncated,
        )
        if result.total != fixture.expected_total:
            raise RuntimeError(f"reward calibration drifted for {fixture.fixture_id}")
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "source_kind": fixture.source_kind,
                "expected_total": fixture.expected_total,
                "actual": result.as_dict(),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_id": REWARD_CONTRACT_ID,
        "configuration_sha256": reward_configuration_sha256(),
        "fixture_sha256": reward_fixture_sha256(),
        "fixture_count": len(rows),
        "all_passed": True,
        "results": rows,
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    return payload
