from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foundry.cycle.contract import load_cycle_config
from foundry.cycle.corpus import (
    ScheduleUnit,
    VariantOccurrence,
    _bounded_exact_fill,
    build_continuation_corpus,
    build_smoke_corpus,
)
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/cycles/cycle1_verifier_filtered.yaml"


class FixedTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self, completion_tokens: dict[str, int] | None = None) -> None:
        self.completion_tokens = completion_tokens or {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> tuple[str, str | None]:
        assert tokenize is False
        completion = None if add_generation_prompt else messages[-1]["content"]
        return ("prefix" if add_generation_prompt else "full", completion)

    def __call__(
        self,
        rendered: tuple[str, str | None],
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        kind, completion = rendered
        if kind == "prefix":
            return {"input_ids": [1] * 10}
        assert completion is not None
        tokens = self.completion_tokens.get(completion, max(2, len(completion.split()) + 1))
        return {"input_ids": [1] * 10 + [2] * (tokens - 1) + [self.eos_token_id]}


def _record(index: int) -> dict[str, Any]:
    completion = f"Reasoning {index}.\nFinal answer: {index}"
    return {
        "source_id": f"source-{index}",
        "question": f"What is {index}?",
        "assistant_completion": completion,
        "assistant_completion_sha256": f"original-{index}",
        "family": "arithmetic",
    }


def test_exact_fill_is_deterministic_and_bounded() -> None:
    units = (
        ScheduleUnit("a", (VariantOccurrence("a", "original", 2),)),
        ScheduleUnit("b", (VariantOccurrence("b", "original", 3),)),
    )

    first = _bounded_exact_fill(units, 7)
    second = _bounded_exact_fill(units, 7)

    assert sum(item.tokens for item in first) == 7
    assert first == second


def test_smoke_corpus_has_four_records_and_two_steps(tmp_path: Path) -> None:
    records = [_record(index) for index in range(4)]
    selected = {
        "source-0": {
            "completion": "Selected reasoning.\nFinal answer: 0",
            "completion_index": 3,
        }
    }

    result = build_smoke_corpus(
        records=records,
        selected_traces=selected,
        tokenizer=FixedTokenizer(),
        output_directory=tmp_path / "smoke",
    )

    assert result["unique_records"] == 4
    assert result["optimizer_steps"] == 2
    assert len(result["records"]) == 4
    schedule = json.loads((tmp_path / "smoke/schedule.json").read_text(encoding="utf-8"))
    assert len(schedule) == 2
    assert canonical_sha256(schedule) == result["schedule_sha256"]


def test_full_corpus_reconstructs_exact_l3_replay_and_32k_schedule(
    tmp_path: Path,
) -> None:
    config = load_cycle_config(CONFIG)
    dataset = config.section("dataset")
    corpus = config.section("corpus")
    targeted_path = config.resolve_artifact(str(dataset["targeted_training_relative_path"]))
    records = [
        json.loads(line) for line in targeted_path.read_text(encoding="utf-8").splitlines() if line
    ]
    completion_tokens = {
        str(record["assistant_completion"]): int(record["assistant_tokens_including_eos"])
        for record in records
    }
    traces = tmp_path / "selected.jsonl"
    traces.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": record["source_id"],
                    "completion": record["assistant_completion"],
                    "completion_index": 0,
                },
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    result = build_continuation_corpus(
        targeted_training_path=targeted_path,
        selected_trace_path=traces,
        source_schedule_path=config.resolve_artifact(str(corpus["source_schedule_relative_path"])),
        expected_source_schedule_sha256=str(corpus["source_schedule_sha256"]),
        replay_path=config.resolve_artifact(str(corpus["replay_relative_source_path"])),
        expected_replay_file_sha256=str(corpus["replay_file_sha256"]),
        tokenizer=FixedTokenizer(completion_tokens),
        output_directory=tmp_path / "full",
    )

    assert result["task_tokens"] == 24_000
    assert result["replay_tokens"] == 8_000
    assert result["total_tokens"] == 32_000
    assert result["optimizer_steps"] == 32
    assert result["selected_trace_prompts"] == 180
    assert result["all_prompt_mixtures_exact"] is True
    assert result["no_holdout_or_gsm_records"] is True
