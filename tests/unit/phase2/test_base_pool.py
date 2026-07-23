import json
from pathlib import Path

import pytest

from foundry.config import GenerationConfig
from foundry.evaluation.backends import FakeModelBackend, GenerationResult
from foundry.evaluation.prompting import ChatMessage
from foundry.phase2 import base_pool
from foundry.phase2.base_pool import (
    PoolEvaluationConfig,
    _exact_format,
    load_pool_config,
    replay_sample_ids,
    run_base_pool_evaluation,
)


class RecordingBackend(FakeModelBackend):
    def __init__(self, responses: dict[str, str]) -> None:
        super().__init__(responses)
        self.messages: list[tuple[ChatMessage, ChatMessage]] = []

    def generate(
        self,
        stable_id: str,
        messages: tuple[ChatMessage, ChatMessage],
        generation: GenerationConfig,
    ) -> GenerationResult:
        self.messages.append(messages)
        return super().generate(stable_id, messages, generation)


def _config(path: Path, *, mathqa: bool = False) -> None:
    payload = {
        "schema_version": 1,
        "candidate_corpus": (
            "MathQA verified uncontaminated train subset"
            if mathqa
            else "ASDiv V1.0 verified uncontaminated pool"
        ),
        "candidate_source_commit": (base_pool.MATHQA_COMMIT if mathqa else base_pool.ASDIV_COMMIT),
        "canonical_extractor_id": base_pool.CANONICAL_EXTRACTOR_ID,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "model_id": base_pool.MODEL_ID,
        "model_revision": base_pool.MODEL_REVISION,
        "replay_sample_seed": (base_pool.MATHQA_REPLAY_SEED if mathqa else base_pool.REPLAY_SEED),
        "replay_sample_size": base_pool.REPLAY_SIZE,
        "seed": base_pool.SEED,
        "system_instruction": base_pool.SYSTEM_INSTRUCTION,
        "user_message": "untouched normalized human-written question only",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(source_id: str, expected: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "family": "multi_step_bookkeeping_or_omission",
        "grade": "2",
        "operation_count": 1,
        "formula_depth": 1,
        "answer_type": "integer",
        "combined_question": f"An original human question for {source_id}.",
        "canonical_answer": expected,
    }


def test_frozen_config_loads_exactly(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _config(path)

    config = load_pool_config(path)

    assert isinstance(config, PoolEvaluationConfig)
    assert config.max_new_tokens == 512
    assert config.do_sample is False


def test_frozen_mathqa_config_loads_exactly(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _config(path, mathqa=True)

    config = load_pool_config(path)

    assert config.candidate_source_commit == base_pool.MATHQA_COMMIT
    assert config.replay_sample_seed == base_pool.MATHQA_REPLAY_SEED


def test_replay_sample_is_stable_and_order_independent() -> None:
    rows = [_candidate("c", "3"), _candidate("a", "1"), _candidate("b", "2")]

    first = replay_sample_ids(rows, "seed", 2)
    second = replay_sample_ids(list(reversed(rows)), "seed", 2)

    assert first == second
    assert len(first) == 2


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Reasoning.\nFinal answer: 4", True),
        ("Final answer: 3/4", True),
        ("Final answer: 4\nMore text", False),
        ("The answer is 4.", False),
    ],
)
def test_exact_format_contract(response: str, expected: bool) -> None:
    assert _exact_format(response) is expected


def test_pool_runner_uses_untouched_question_and_exact_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    _config(config_path)
    candidate_path = tmp_path / "candidates.jsonl"
    rows = [_candidate(f"id-{index:02d}", "4") for index in range(base_pool.REPLAY_SIZE)]
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    backend = RecordingBackend(
        {str(row["source_id"]): "Reasoning.\nFinal answer: 4" for row in rows}
    )
    monkeypatch.setattr(base_pool, "initialize_determinism", lambda seed: None)

    summary = run_base_pool_evaluation(
        config_path=config_path,
        candidates_path=candidate_path,
        output_dir=tmp_path / "output",
        backend_factory=lambda config: backend,
    )

    assert summary["correct"] == base_pool.REPLAY_SIZE
    assert summary["backend_failures"] == 0
    assert summary["replay_exact"] is True
    assert len(backend.messages) == base_pool.REPLAY_SIZE * 2
    assert backend.messages[0][0]["content"] == base_pool.SYSTEM_INSTRUCTION
    assert backend.messages[0][1]["content"] == rows[0]["combined_question"]
    assert summary["gsm1k_exposed_to_selection"] is False


def test_existing_prediction_file_fails_closed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _config(config_path)
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(json.dumps(_candidate("a", "4")) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "predictions.jsonl").write_text("existing", encoding="utf-8")

    def backend_factory(config: object) -> FakeModelBackend:
        del config
        return FakeModelBackend({"a": "Final answer: 4"})

    with pytest.raises(FileExistsError):
        run_base_pool_evaluation(
            config_path=config_path,
            candidates_path=candidate_path,
            output_dir=output,
            backend_factory=backend_factory,
        )
