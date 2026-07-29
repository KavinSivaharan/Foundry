"""Deterministic verifier reward for Milestone 14A L3 GRPO."""

from __future__ import annotations

import hashlib
import json
import math
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
from foundry.training.config import canonical_sha256
from foundry.training.retention import (
    RetentionItem,
    question_generation_evidence,
    score_response,
)

REWARD_CONTRACT_ID = "foundry-l3-verifier-grpo-reward-v1"
SourceKind = Literal["task", "base_replay"]
BackendStatus = Literal["ok", "failure"]

TASK_CORRECTNESS_REWARD = 1.0
REPLAY_SCORER_REWARD = 1.0
EXTRACTION_REWARD = 0.10
FORMAT_REWARD = 0.05
INSTRUCTION_COMPLIANCE_REWARD = 0.05
TRUNCATION_PENALTY = -0.10
PROMPT_ECHO_PENALTY = -0.25
QUESTION_GENERATION_PENALTY = -0.25
MALFORMED_OUTPUT_PENALTY = -0.50
BACKEND_FAILURE_PENALTY = -1.0
_FINAL_ANSWER_PREFIX = "Final answer:"
_FINAL_ANSWER_LINE = re.compile(r"Final answer: (?P<answer>[+-]?(?:\d+(?:\.\d+)?|\d+/\d+))")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _fraction(value: str) -> Fraction:
    _require_text(value, "canonical_answer")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError("canonical_answer must not contain whitespace")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("canonical_answer must be an exact number") from error


@dataclass(frozen=True)
class TaskRewardMetadata:
    """Trusted answer-side metadata for one vetted human-written task."""

    source_id: str
    prompt: str
    canonical_answer: str
    answer_type: str
    family: str
    difficulty: str
    verifier_metadata_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source_id", self.source_id),
            ("prompt", self.prompt),
            ("answer_type", self.answer_type),
            ("family", self.family),
            ("difficulty", self.difficulty),
        ):
            _require_text(value, name)
        _fraction(self.canonical_answer)
        _require_sha256(self.verifier_metadata_sha256, "verifier_metadata_sha256")
        _require_sha256(self.provenance_sha256, "provenance_sha256")


@dataclass(frozen=True)
class ReplayRewardMetadata:
    """Trusted scorer-side metadata for one frozen replay record."""

    replay_id: str
    prompt: str
    retention_item: RetentionItem
    scorer_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.replay_id, "replay_id")
        _require_text(self.prompt, "prompt")
        if self.retention_item.item_id != self.replay_id:
            raise ValueError("replay ID differs from its frozen scorer item")
        if self.retention_item.prompt != self.prompt:
            raise ValueError("replay prompt differs from its frozen scorer item")
        _require_sha256(self.scorer_sha256, "scorer_sha256")
        _require_sha256(self.provenance_sha256, "provenance_sha256")


RewardMetadata = TaskRewardMetadata | ReplayRewardMetadata


@dataclass(frozen=True)
class RewardBreakdown:
    """Complete additive deterministic reward evidence for one completion."""

    contract_id: str
    source_kind: SourceKind
    source_id: str
    task_answer_correctness: float
    replay_scorer_correctness: float
    extraction: float
    canonical_or_required_format: float
    instruction_compliance: float
    truncation_penalty: float
    prompt_echo_penalty: float
    question_generation_penalty: float
    malformed_output_penalty: float
    backend_failure_penalty: float
    answer_correct: bool
    extractable: bool
    exact_format: bool
    instruction_compliant: bool
    generation_truncated: bool
    prompt_echo: bool
    question_generation: bool
    malformed_output: bool
    backend_failure: bool
    extracted_answer_sha256: str | None
    total: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _prompt_echo(prompt: str, completion: str) -> bool:
    normalized_prompt = " ".join(prompt.lower().split())
    normalized_completion = " ".join(completion.lower().split())
    return len(normalized_prompt) >= 24 and normalized_prompt in normalized_completion


def _exact_task_format(completion: str) -> bool:
    lines = completion.strip().splitlines()
    answer_lines = [line for line in lines if line.startswith(_FINAL_ANSWER_PREFIX)]
    if not lines or len(answer_lines) != 1 or lines[-1] != answer_lines[0]:
        return False
    match = _FINAL_ANSWER_LINE.fullmatch(answer_lines[0])
    if match is None:
        return False
    try:
        _fraction(match.group("answer"))
    except ValueError:
        return False
    return True


def _task_score(
    metadata: TaskRewardMetadata,
    completion: str,
    *,
    generation_truncated: bool,
    backend_status: BackendStatus,
) -> RewardBreakdown:
    expected = _fraction(metadata.canonical_answer)
    extracted: Fraction | None = None
    malformed = False
    if backend_status == "ok":
        try:
            extracted = extract_canonical_number(completion)
        except CanonicalExtractionError:
            malformed = True
    else:
        malformed = True
    answer_correct = extracted == expected
    extractable = extracted is not None
    exact_format = _exact_task_format(completion)
    prompt_echo = _prompt_echo(metadata.prompt, completion)
    question_generation = bool(
        question_generation_evidence(metadata.prompt, completion)["decision"]
    )
    backend_failure = backend_status == "failure"
    components = {
        "task_answer_correctness": TASK_CORRECTNESS_REWARD if answer_correct else 0.0,
        "replay_scorer_correctness": 0.0,
        "extraction": EXTRACTION_REWARD if extractable else 0.0,
        "canonical_or_required_format": FORMAT_REWARD if exact_format else 0.0,
        "instruction_compliance": 0.0,
        "truncation_penalty": TRUNCATION_PENALTY if generation_truncated else 0.0,
        "prompt_echo_penalty": PROMPT_ECHO_PENALTY if prompt_echo else 0.0,
        "question_generation_penalty": (
            QUESTION_GENERATION_PENALTY if question_generation else 0.0
        ),
        "malformed_output_penalty": MALFORMED_OUTPUT_PENALTY if malformed else 0.0,
        "backend_failure_penalty": (BACKEND_FAILURE_PENALTY if backend_failure else 0.0),
    }
    total = round(math.fsum(components.values()), 10)
    return RewardBreakdown(
        contract_id=REWARD_CONTRACT_ID,
        source_kind="task",
        source_id=metadata.source_id,
        **components,
        answer_correct=answer_correct,
        extractable=extractable,
        exact_format=exact_format,
        instruction_compliant=False,
        generation_truncated=generation_truncated,
        prompt_echo=prompt_echo,
        question_generation=question_generation,
        malformed_output=malformed,
        backend_failure=backend_failure,
        extracted_answer_sha256=(
            None
            if extracted is None
            else hashlib.sha256(str(extracted).encode("utf-8")).hexdigest()
        ),
        total=total,
    )


def _replay_score(
    metadata: ReplayRewardMetadata,
    completion: str,
    *,
    generation_truncated: bool,
    backend_status: BackendStatus,
) -> RewardBreakdown:
    score = score_response(metadata.retention_item, completion)
    backend_failure = backend_status == "failure"
    answer_correct = bool(score["correct"]) and not backend_failure
    extractable = bool(score["extractable"]) and not backend_failure
    exact_format = bool(score["exact_format"]) and not backend_failure
    prompt_echo = bool(score["prompt_echo"])
    question_generation = bool(score["question_generation"])
    malformed = bool(score["malformed"]) or backend_failure
    instruction_compliant = (
        answer_correct and exact_format and not prompt_echo and not question_generation
    )
    components = {
        "task_answer_correctness": 0.0,
        "replay_scorer_correctness": REPLAY_SCORER_REWARD if answer_correct else 0.0,
        "extraction": EXTRACTION_REWARD if extractable else 0.0,
        "canonical_or_required_format": FORMAT_REWARD if exact_format else 0.0,
        "instruction_compliance": (INSTRUCTION_COMPLIANCE_REWARD if instruction_compliant else 0.0),
        "truncation_penalty": TRUNCATION_PENALTY if generation_truncated else 0.0,
        "prompt_echo_penalty": PROMPT_ECHO_PENALTY if prompt_echo else 0.0,
        "question_generation_penalty": (
            QUESTION_GENERATION_PENALTY if question_generation else 0.0
        ),
        "malformed_output_penalty": MALFORMED_OUTPUT_PENALTY if malformed else 0.0,
        "backend_failure_penalty": (BACKEND_FAILURE_PENALTY if backend_failure else 0.0),
    }
    total = round(math.fsum(components.values()), 10)
    extracted_hash = score.get("extracted_hash")
    return RewardBreakdown(
        contract_id=REWARD_CONTRACT_ID,
        source_kind="base_replay",
        source_id=metadata.replay_id,
        **components,
        answer_correct=answer_correct,
        extractable=extractable,
        exact_format=exact_format,
        instruction_compliant=instruction_compliant,
        generation_truncated=generation_truncated,
        prompt_echo=prompt_echo,
        question_generation=question_generation,
        malformed_output=malformed,
        backend_failure=backend_failure,
        extracted_answer_sha256=(extracted_hash if isinstance(extracted_hash, str) else None),
        total=total,
    )


def score_reward(
    metadata: RewardMetadata,
    completion: str,
    *,
    generation_truncated: bool = False,
    backend_status: BackendStatus = "ok",
) -> RewardBreakdown:
    """Score one completion with no learned or model-generated judgment."""

    if backend_status not in {"ok", "failure"}:
        raise ValueError("backend status is outside the frozen two-value contract")
    if not isinstance(completion, str) or (backend_status == "ok" and not completion.strip()):
        raise ValueError("successful completion must contain non-empty text")
    if isinstance(metadata, TaskRewardMetadata):
        return _task_score(
            metadata,
            completion,
            generation_truncated=generation_truncated,
            backend_status=backend_status,
        )
    return _replay_score(
        metadata,
        completion,
        generation_truncated=generation_truncated,
        backend_status=backend_status,
    )


REWARD_CONFIGURATION: dict[str, object] = {
    "schema_version": 1,
    "contract_id": REWARD_CONTRACT_ID,
    "canonical_extractor_id": CANONICAL_EXTRACTOR_ID,
    "task": {
        "exact_answer_correctness": TASK_CORRECTNESS_REWARD,
        "deterministic_extraction": EXTRACTION_REWARD,
        "canonical_final_answer_format": FORMAT_REWARD,
    },
    "replay": {
        "frozen_scorer_correctness": REPLAY_SCORER_REWARD,
        "deterministic_extraction": EXTRACTION_REWARD,
        "required_format": FORMAT_REWARD,
        "instruction_compliance": INSTRUCTION_COMPLIANCE_REWARD,
        "scorer": "foundry.training.retention.score_response",
    },
    "penalties": {
        "truncation": TRUNCATION_PENALTY,
        "prompt_echo": PROMPT_ECHO_PENALTY,
        "question_generation": QUESTION_GENERATION_PENALTY,
        "malformed_output": MALFORMED_OUTPUT_PENALTY,
        "backend_failure": BACKEND_FAILURE_PENALTY,
    },
    "learned_reward_model": False,
    "llm_judge": False,
    "human_in_the_loop_during_training": False,
    "benchmark_reward": False,
    "response_length_reward": False,
    "category_multiplier": False,
}


def reward_configuration_sha256() -> str:
    return canonical_sha256(REWARD_CONFIGURATION)


def reward_implementation_sha256() -> str:
    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RewardFixture:
    """One original non-benchmark reward safety fixture."""

    fixture_id: str
    kind: Literal["task", "replay"]
    prompt: str
    completion: str
    expected_total: float
    canonical_answer: str | None = None
    replay_kind: Literal["numeric_terminal", "exact_text", "json_exact"] | None = None
    replay_expected: str | None = None
    truncated: bool = False
    backend_status: BackendStatus = "ok"


REWARD_FIXTURES: tuple[RewardFixture, ...] = (
    RewardFixture(
        "task-correct-compliant",
        "task",
        "A workshop records a completed count.",
        "The ledger contains seven.\nFinal answer: 7",
        1.15,
        canonical_answer="7",
    ),
    RewardFixture(
        "task-incorrect-extractable",
        "task",
        "A depot records a completed count.",
        "Final answer: 8",
        0.15,
        canonical_answer="7",
    ),
    RewardFixture(
        "task-malformed",
        "task",
        "A studio records a completed count.",
        "No numeric result is available.",
        -0.5,
        canonical_answer="7",
    ),
    RewardFixture(
        "task-unsafe-question",
        "task",
        "Report the final workshop count.",
        "Could the answer be seven?\nFinal answer: 7",
        0.9,
        canonical_answer="7",
    ),
    RewardFixture(
        "task-backend-failure",
        "task",
        "Report the final depot count.",
        "",
        -1.5,
        canonical_answer="7",
        backend_status="failure",
    ),
    RewardFixture(
        "replay-correct-compliant",
        "replay",
        "Return the material name only.",
        "cedar",
        1.2,
        replay_kind="exact_text",
        replay_expected="cedar",
    ),
    RewardFixture(
        "replay-incorrect",
        "replay",
        "Return the material name only.",
        "maple",
        0.1,
        replay_kind="exact_text",
        replay_expected="cedar",
    ),
    RewardFixture(
        "task-truncated",
        "task",
        "Report the final studio count.",
        "Final answer: 7",
        1.05,
        canonical_answer="7",
        truncated=True,
    ),
)


def reward_fixture_sha256() -> str:
    return canonical_sha256([asdict(fixture) for fixture in REWARD_FIXTURES])


def _fixture_metadata(fixture: RewardFixture) -> RewardMetadata:
    if fixture.kind == "task":
        if fixture.canonical_answer is None:
            raise ValueError("task fixture lacks canonical answer")
        return TaskRewardMetadata(
            source_id=fixture.fixture_id,
            prompt=fixture.prompt,
            canonical_answer=fixture.canonical_answer,
            answer_type="integer",
            family="original_fixture",
            difficulty="easy",
            verifier_metadata_sha256="a" * 64,
            provenance_sha256="b" * 64,
        )
    if fixture.replay_kind is None or fixture.replay_expected is None:
        raise ValueError("replay fixture lacks scorer metadata")
    item = RetentionItem(
        item_id=fixture.fixture_id,
        section="instruction",
        skill="original_fixture",
        kind=fixture.replay_kind,
        prompt=fixture.prompt,
        expected=fixture.replay_expected,
    )
    return ReplayRewardMetadata(
        replay_id=fixture.fixture_id,
        prompt=fixture.prompt,
        retention_item=item,
        scorer_sha256="c" * 64,
        provenance_sha256="d" * 64,
    )


def calibrate_reward_contract() -> dict[str, object]:
    """Replay every fixture and freeze the complete safety ordering."""

    rows: list[dict[str, object]] = []
    results: dict[str, RewardBreakdown] = {}
    for fixture in REWARD_FIXTURES:
        result = score_reward(
            _fixture_metadata(fixture),
            fixture.completion,
            generation_truncated=fixture.truncated,
            backend_status=fixture.backend_status,
        )
        if result.total != fixture.expected_total:
            raise RuntimeError(
                f"reward fixture {fixture.fixture_id} drifted: "
                f"{result.total} != {fixture.expected_total}"
            )
        results[fixture.fixture_id] = result
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "source_kind": fixture.kind,
                "expected_total": fixture.expected_total,
                "result": result.as_dict(),
            }
        )
    correct = results["task-correct-compliant"].total
    if not (
        correct > results["task-incorrect-extractable"].total
        and correct > results["task-malformed"].total
        and correct > results["task-unsafe-question"].total
        and correct > results["task-backend-failure"].total
    ):
        raise RuntimeError("reward safety ordering differs")
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_id": REWARD_CONTRACT_ID,
        "configuration_sha256": reward_configuration_sha256(),
        "implementation_sha256": reward_implementation_sha256(),
        "fixture_sha256": reward_fixture_sha256(),
        "fixture_count": len(rows),
        "true_positive_and_true_negative_fixtures": True,
        "correct_greater_than_incorrect": True,
        "unsafe_or_malformed_cannot_outscore_correct": True,
        "deterministic_replay": True,
        "model_generated_score": False,
        "human_training_score": False,
        "results": rows,
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    return payload


def reward_contract_sha256() -> str:
    calibration = calibrate_reward_contract()
    return canonical_sha256(
        {
            "contract_id": REWARD_CONTRACT_ID,
            "configuration_sha256": reward_configuration_sha256(),
            "implementation_sha256": reward_implementation_sha256(),
            "fixture_sha256": reward_fixture_sha256(),
            "calibration_sha256": calibration["calibration_sha256"],
        }
    )


def serialize_reward_contract() -> str:
    """Return a stable content-free contract projection for evidence writing."""

    value = {
        "schema_version": 1,
        "contract_id": REWARD_CONTRACT_ID,
        "configuration": REWARD_CONFIGURATION,
        "configuration_sha256": reward_configuration_sha256(),
        "implementation_sha256": reward_implementation_sha256(),
        "fixture_sha256": reward_fixture_sha256(),
        "calibration": calibrate_reward_contract(),
        "reward_contract_sha256": reward_contract_sha256(),
    }
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
