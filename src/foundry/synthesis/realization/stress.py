"""Bounded, non-dataset stress exercise for the typed realization compiler."""

# ruff: noqa: E501  # content-free evidence keys remain visually aligned

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch

from foundry.synthesis.contamination import normalized_text_sha256, numeric_template_sha256
from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.generators.bookkeeping import generate_bookkeeping
from foundry.synthesis.generators.discrete import generate_discrete
from foundry.synthesis.generators.rates import generate_rates
from foundry.synthesis.realization.compiler import compiler_capacity
from foundry.synthesis.realization.contracts import validate_realization
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.semantic import PinnedSentenceEncoder, load_semantic_artifact_config

STRESS_MASTER_SEED = "foundry-m4.2-typed-renderer-stress-v1"
Generator = Callable[..., CandidateDraft]
_FAMILIES: tuple[tuple[str, Generator], ...] = (
    ("bookkeeping", generate_bookkeeping),
    ("rates", generate_rates),
    ("discrete", generate_discrete),
)


def _seed(family: str, index: int) -> int:
    material = f"{STRESS_MASTER_SEED}:{family}:{index}"
    return int(hashlib.sha256(material.encode()).hexdigest()[:16], 16)


def construct_stress_drafts(attempts_per_family: int = 300) -> tuple[CandidateDraft, ...]:
    """Construct at most 900 in-memory candidates without persistence or benchmark access."""

    if attempts_per_family < 1 or attempts_per_family > 300:
        raise ValueError("renderer stress is bounded to 300 attempts per family")
    difficulties = (DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD)
    drafts: list[CandidateDraft] = []
    for family, generator in _FAMILIES:
        for index in range(attempts_per_family):
            drafts.append(
                generator(
                    seed=_seed(family, index),
                    difficulty=difficulties[index % len(difficulties)],
                    variant=index,
                    output_contract_enabled=index % 5 == 0,
                )
            )
    return tuple(drafts)


def _neighbor_distribution(embeddings: torch.Tensor) -> dict[str, object]:
    matrix = embeddings @ embeddings.T
    matrix.fill_diagonal_(-1.0)
    maxima = torch.max(matrix, dim=1).values
    bins = {
        "below_0_50": int((maxima < 0.50).sum().item()),
        "0_50_to_0_60": int(((maxima >= 0.50) & (maxima < 0.60)).sum().item()),
        "0_60_to_0_70": int(((maxima >= 0.60) & (maxima < 0.70)).sum().item()),
        "0_70_to_0_75": int(((maxima >= 0.70) & (maxima < 0.75)).sum().item()),
        "0_75_to_0_82": int(((maxima >= 0.75) & (maxima < 0.82)).sum().item()),
        "at_or_above_0_82": int((maxima >= 0.82).sum().item()),
    }
    return {
        "bins": bins,
        "minimum": round(float(torch.min(maxima).item()), 6),
        "median": round(float(torch.quantile(maxima, 0.5).item()), 6),
        "p90": round(float(torch.quantile(maxima, 0.9).item()), 6),
        "maximum": round(float(torch.max(maxima).item()), 6),
    }


def run_renderer_stress(
    repository_root: Path,
) -> tuple[dict[str, object], tuple[CandidateDraft, ...]]:
    """Run the complete 900-attempt in-memory stress and return content-free evidence."""

    started = time.perf_counter()
    drafts = construct_stress_drafts(300)
    failures: Counter[str] = Counter()
    for draft in drafts:
        failures.update(
            validate_realization(
                problem=draft.problem_ir,
                realization=draft.realization,
                answer=draft.canonical_final_answer,
            )
        )
    exact = [normalized_text_sha256(draft.rendered_question) for draft in drafts]
    numeric = [numeric_template_sha256(draft.rendered_question) for draft in drafts]
    structural = [draft.structure_sha256 for draft in drafts]
    signatures = [draft.render_signature_sha256 for draft in drafts]
    config = load_semantic_artifact_config(
        repository_root / "configs/synthesis/semantic_all_minilm_l6_v2.yaml"
    )
    encoder = PinnedSentenceEncoder(config, repository_root)
    embeddings = encoder.encode([draft.rendered_question for draft in drafts])
    family_counts = Counter(draft.target_failure_category for draft in drafts)
    summary: dict[str, object] = {
        "schema_version": 1,
        "run_id": "typed-realizer-stress-v1",
        "master_seed": STRESS_MASTER_SEED,
        "attempted": len(drafts),
        "attempted_by_family": {str(key): value for key, value in sorted(family_counts.items())},
        "successful_renders": len(drafts) - sum(failures.values()),
        "typed_validation_failures": dict(sorted(failures.items())),
        "morphology_failures": failures["morphology_or_agreement_failure"],
        "target_type_failures": failures["target_type_mismatch"],
        "missing_node_failures": failures["missing_semantic_node"],
        "duplicated_node_failures": failures["duplicated_semantic_node"],
        "grammar_metadata_failures": failures["grammar_metadata_failure"],
        "exact_duplicate_count": len(exact) - len(set(exact)),
        "number_neutral_template_duplicate_count": len(numeric) - len(set(numeric)),
        "structural_duplicate_count": len(structural) - len(set(structural)),
        "distinct_render_signatures": len(set(signatures)),
        "compiler_capacity": compiler_capacity(),
        "semantic_neighbor_distribution": _neighbor_distribution(embeddings),
        "semantic_model_id": config.model_id,
        "semantic_model_revision": config.revision,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "persisted_question_corpus": False,
        "benchmark_comparison_performed": False,
    }
    return summary, drafts


def deterministic_manual_sample(
    drafts: tuple[CandidateDraft, ...], per_family: int = 20
) -> tuple[CandidateDraft, ...]:
    """Select a stable 60-render audit sample from the in-memory stress candidates."""

    if per_family < 20:
        raise ValueError("the stress audit requires at least 20 samples per family")
    selected: list[CandidateDraft] = []
    for family, _ in _FAMILIES:
        matching = [
            draft
            for draft in drafts
            if family in draft.generator_id
            or (family == "rates" and draft.generator_id == "exact-rate-ratio-relations")
        ]
        if family == "bookkeeping":
            matching = [
                draft for draft in drafts if draft.generator_id == "bookkeeping-state-transitions"
            ]
        elif family == "discrete":
            matching = [
                draft for draft in drafts if draft.generator_id == "bounded-discrete-allocation"
            ]
        step = max(1, len(matching) // per_family)
        selected.extend(matching[index] for index in range(0, len(matching), step)[:per_family])
    if len(selected) != per_family * len(_FAMILIES):
        raise ValueError("stress audit sample construction failed")
    return tuple(selected)
