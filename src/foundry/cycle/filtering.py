"""Apply the frozen verifier and persist raw selected traces only in runtime storage."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from foundry.cycle.contract import CycleConfig, verified_payload
from foundry.cycle.generation import select_smoke_records
from foundry.cycle.selection import (
    CandidateDecision,
    build_selection_summary,
    select_candidate,
    verify_candidate,
)
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def filter_candidates(
    *,
    config: CycleConfig,
    candidates_path: Path,
    output_directory: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Score all attempted completions and deterministically select at most one."""

    if output_directory.exists():
        raise FileExistsError("verifier-selection output must be fresh")
    rows = _jsonl(candidates_path)
    if not rows:
        raise ValueError("candidate generation produced no rows")
    generation_summary = verified_payload(
        candidates_path.parent / "summary.json",
        "generation_sha256",
    )
    if (
        file_sha256(candidates_path) != generation_summary["raw_file_sha256"]
        or generation_summary["smoke"] is not smoke
        or generation_summary["attempted_completions"] != len(rows)
        or generation_summary["completions_per_prompt"] != 8
        or generation_summary["backend_failures"]
        != sum(row.get("backend_error_type") is not None for row in rows)
    ):
        raise ValueError("candidate-generation artifact identity differs")
    dataset = config.section("dataset")
    source_records = _jsonl(
        config.resolve_artifact(str(dataset["targeted_training_relative_path"]))
    )
    expected_records = (
        select_smoke_records(source_records)
        if smoke
        else sorted(source_records, key=lambda item: str(item["source_id"]))
    )
    expected_by_id = {str(item["source_id"]): item for item in expected_records}
    expected_source_ids = set(expected_by_id)
    if len(expected_by_id) != (4 if smoke else int(dataset["training_records"])):
        raise ValueError("verifier expected-source reconstruction differs")
    decisions: list[CandidateDecision] = []
    raw_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    grouped: defaultdict[str, list[CandidateDecision]] = defaultdict(list)
    for row in rows:
        source_id = str(row["source_id"])
        completion_index = int(row["completion_index"])
        source = expected_by_id.get(source_id)
        if (
            source is None
            or str(row["family"]) != str(source["family"])
            or str(row["question"]) != str(source["question"])
            or str(row["question_sha256"]) != str(source["question_sha256"])
            or str(row["canonical_answer"]) != str(source["canonical_answer"])
            or str(row["original_completion_sha256"]) != str(source["assistant_completion_sha256"])
            or (source_id, completion_index) in raw_by_key
        ):
            raise ValueError("candidate source identity or attempt identity differs")
        decision = verify_candidate(
            source_id=source_id,
            family=str(row["family"]),
            prompt=str(row["question"]),
            canonical_answer=str(row["canonical_answer"]),
            completion_index=completion_index,
            completion=str(row["completion"]),
            completion_tokens=int(row["completion_tokens"]),
            truncated=bool(row["truncated"]),
            backend_error_type=(
                None if row.get("backend_error_type") is None else str(row["backend_error_type"])
            ),
        )
        decisions.append(decision)
        grouped[source_id].append(decision)
        raw_by_key[(source_id, completion_index)] = row
    summary = build_selection_summary(
        decisions=decisions,
        expected_source_ids=expected_source_ids,
        family_by_source_id={
            source_id: str(item["family"]) for source_id, item in expected_by_id.items()
        },
        original_target_sha256={
            source_id: str(item["assistant_completion_sha256"])
            for source_id, item in expected_by_id.items()
        },
    )
    selected_decisions = {
        source_id: selected
        for source_id, values in grouped.items()
        if (selected := select_candidate(values)) is not None
    }
    selected_rows = []
    for source_id, decision in sorted(selected_decisions.items()):
        raw = raw_by_key[(source_id, decision.completion_index)]
        selected_rows.append(
            {
                "schema_version": 1,
                "source_id": source_id,
                "family": decision.family,
                "completion_index": decision.completion_index,
                "completion_tokens": decision.completion_tokens,
                "completion": str(raw["completion"]),
                "raw_sha256": decision.raw_sha256,
                "normalized_sha256": decision.normalized_sha256,
                "original_completion_sha256": str(raw["original_completion_sha256"]),
                "question_sha256": str(raw["question_sha256"]),
            }
        )
    output_directory.mkdir(parents=True, exist_ok=False)
    decisions_path = output_directory / "component_decisions.jsonl"
    selected_path = output_directory / "selected_traces.jsonl"
    _write_jsonl(decisions_path, [item.as_dict() for item in decisions])
    _write_jsonl(selected_path, selected_rows)
    summary["component_decisions_sha256"] = canonical_sha256([item.as_dict() for item in decisions])
    summary["component_decisions_file_sha256"] = file_sha256(decisions_path)
    summary["selected_trace_file_sha256"] = file_sha256(selected_path)
    summary["selected_trace_manifest_sha256"] = canonical_sha256(
        [
            {key: value for key, value in item.items() if key != "completion"}
            for item in selected_rows
        ]
    )
    summary["selection_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "selection_sha256"}
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
