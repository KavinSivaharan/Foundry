"""Offline deterministic rescoring of frozen vetted-corpus retention outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256
from foundry.training.retention import (
    RetentionItem,
    load_suite,
    question_generation_configuration_sha256,
    score_response,
)

VARIANTS = ("v1", "v2")
ARMS = ("generic", "targeted")
CHECKPOINTS = (16, 32, 64)
SUITES = ("adjudication", "anchor")


def wilson_lower_bound(correct: int, total: int) -> float:
    """Return the two-sided 95% Wilson lower confidence bound."""

    if not 0 <= correct <= total or total <= 0:
        raise ValueError("invalid Wilson inputs")
    z = 1.959963984540054
    probability = correct / total
    numerator = (
        probability
        + z * z / (2 * total)
        - z * math.sqrt((probability * (1 - probability) + z * z / (4 * total)) / total)
    )
    return numerator / (1 + z * z / total)


def _response_payload_sha256(rows: list[dict[str, Any]]) -> str:
    return canonical_sha256(
        [
            {
                "id": row["id"],
                "response": row["response"],
                "response_sha256": row["response_sha256"],
            }
            for row in rows
        ]
    )


def _score_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["score"]["correct"]) for row in rows)
    sections: dict[str, dict[str, int | float]] = {}
    for section in ("arithmetic", "format", "instruction"):
        selected = [row for row in rows if row["section"] == section]
        section_correct = sum(bool(row["score"]["correct"]) for row in selected)
        sections[section] = {
            "total": len(selected),
            "correct": section_correct,
            "accuracy": section_correct / len(selected),
        }
    failure_families = Counter(
        str(row["skill"]) for row in rows if not bool(row["score"]["correct"])
    )
    metrics: dict[str, Any] = {
        "total": total,
        "correct": correct,
        "overall_preservation": correct / total,
        "wilson_lower_95": wilson_lower_bound(correct, total),
        "section_metrics": sections,
        "extractable": sum(bool(row["score"]["extractable"]) for row in rows),
        "prompt_echo": sum(bool(row["score"]["prompt_echo"]) for row in rows),
        "question_generation": sum(bool(row["score"]["question_generation"]) for row in rows),
        "malformed_outputs": sum(bool(row["score"]["malformed"]) for row in rows),
        "backend_failures": sum("backend_error_type" in row["score"] for row in rows),
        "maximum_failure_family": max(failure_families.values(), default=0),
    }
    metrics["extractability"] = int(metrics["extractable"]) / total
    metrics["prompt_echo_rate"] = int(metrics["prompt_echo"]) / total
    metrics["passed"] = bool(
        float(metrics["overall_preservation"]) >= 0.90
        and all(
            float(cast(dict[str, Any], sections[section])["accuracy"]) >= 0.90
            for section in sections
        )
        and float(metrics["wilson_lower_95"]) >= 0.85
        and float(metrics["prompt_echo_rate"]) <= 0.02
        and int(metrics["question_generation"]) == 0
        and int(metrics["backend_failures"]) == 0
        and int(metrics["maximum_failure_family"]) <= 3
    )
    return metrics


def rescore_packet(
    *,
    suite_path: Path,
    raw_path: Path,
    summary_path: Path,
    output_raw_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    """Rescore one existing packet without changing response bytes or unrelated scores."""

    suite = load_suite(suite_path)
    suite_index = {item.item_id: item for item in suite.items}
    raw = cast(list[dict[str, Any]], json.loads(raw_path.read_text(encoding="utf-8")))
    old_summary = cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))
    corrected: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for row in raw:
        response = str(row["response"])
        if hashlib.sha256(response.encode("utf-8")).hexdigest() != row["response_sha256"]:
            raise ValueError("stored response hash differs")
        item = suite_index[str(row["id"])]
        score = score_response(
            RetentionItem(
                item.item_id,
                item.section,
                item.skill,
                item.kind,
                item.prompt,
                item.expected,
            ),
            response,
        )
        old_score = cast(dict[str, Any], row["score"])
        changed_fields = sorted(
            key for key in set(old_score) | set(score) if old_score.get(key) != score.get(key)
        )
        if changed_fields and changed_fields != ["question_generation"]:
            raise ValueError("offline rescore changed a non-question score")
        if changed_fields:
            changed.append(
                {
                    "id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest(),
                    "response_sha256": row["response_sha256"],
                    "changed_fields": changed_fields,
                    "old_question_generation": old_score["question_generation"],
                    "corrected_question_generation": score["question_generation"],
                }
            )
        corrected.append({**row, "score": score})
    if _response_payload_sha256(raw) != _response_payload_sha256(corrected):
        raise RuntimeError("offline rescore changed response bytes or hashes")
    metrics = _score_metrics(corrected)
    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    output_raw_path.write_text(
        json.dumps(corrected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "suite_sha256": suite.suite_sha256,
        "adapter_sha256": old_summary["adapter_sha256"],
        "base_conditioned_subset_sha256": old_summary["base_conditioned_subset_sha256"],
        "source_raw_file_sha256": file_sha256(raw_path),
        "source_summary_sha256": old_summary["summary_sha256"],
        "response_payload_sha256": _response_payload_sha256(raw),
        "corrected_raw_file_sha256": file_sha256(output_raw_path),
        "question_generation_configuration_sha256": (question_generation_configuration_sha256()),
        "changed_decisions": len(changed),
        "changed_decision_sha256": canonical_sha256(changed),
        "metrics": metrics,
    }
    result["summary_sha256"] = canonical_sha256(result)
    output_summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_matrix(
    *,
    repository_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Rescore the complete V1/V2 checkpoint matrix and select by frozen hierarchy."""

    raw_root = repository_root / "results/raw/phase2_vetted_corpus"
    suite_paths = {
        "adjudication": repository_root
        / "results/raw/training/retention_powered_adjudication/retention_adjudication_v2.json",
        "anchor": repository_root
        / "results/raw/training/retention_powered_adjudication/retention_anchor_holdout_v1.json",
    }
    cells: dict[str, Any] = {}
    for variant in VARIANTS:
        cells[variant] = {}
        for arm in ARMS:
            cells[variant][arm] = {}
            for checkpoint in CHECKPOINTS:
                cells[variant][arm][str(checkpoint)] = {}
                source = raw_root / f"{variant}_retention/{arm}/step-{checkpoint}"
                target = output_root / f"{variant}/{arm}/step-{checkpoint}"
                for suite in SUITES:
                    cells[variant][arm][str(checkpoint)][suite] = rescore_packet(
                        suite_path=suite_paths[suite],
                        raw_path=source / f"{suite}_raw.json",
                        summary_path=source / f"{suite}_summary.json",
                        output_raw_path=target / f"{suite}_raw.json",
                        output_summary_path=target / f"{suite}_summary.json",
                    )
    selected_variant: str | None = None
    selected_checkpoint: int | None = None
    for variant in VARIANTS:
        passing = [
            checkpoint
            for checkpoint in CHECKPOINTS
            if all(
                bool(cells[variant][arm][str(checkpoint)][suite]["metrics"]["passed"])
                for arm in ARMS
                for suite in SUITES
            )
        ]
        if passing:
            selected_variant = variant
            selected_checkpoint = max(passing)
            break
    summary: dict[str, Any] = {
        "schema_version": 1,
        "rescore_id": "foundry-vetted-corpus-retention-offline-rescore-v1",
        "model_inference_runs": 0,
        "cells": cells,
        "selected_variant": selected_variant,
        "selected_checkpoint": selected_checkpoint,
        "selection_hierarchy": ["v1_latest_common", "v2_latest_common"],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_matrix(
                repository_root=args.repository_root.resolve(),
                output_root=args.output_root.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
