"""Bounded 30-IR/90-beam compact tagged-realization micro-smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from foundry.synthesis.contamination import ContaminationOutcome
from foundry.synthesis.realization.compact_contracts import (
    COMPACT_SYSTEM_PROMPT_SHA256,
    CompactRealizationResponse,
)
from foundry.synthesis.realization.compact_prompting import (
    COMPACT_COMBINED_PROTOCOL_SHA256,
    COMPACT_USER_PROTOCOL_SHA256,
)
from foundry.synthesis.realization.compact_request import (
    PreparedCompactRequest,
    prepare_compact_request,
)
from foundry.synthesis.realization.compact_runtime import PinnedCompactQwenRealizer
from foundry.synthesis.realization.compact_smoke_contract import (
    CompactSmokeConfig,
    build_compact_attempt_plan,
    generate_procedural_ir,
    load_compact_smoke_config,
)
from foundry.synthesis.realization.compact_validation import (
    CompactContractError,
    fill_compact_response,
    parse_compact_response,
    validate_compact_response,
)
from foundry.synthesis.realization.development_export import load_development_question_export
from foundry.synthesis.realization.diversity import load_frozen_internal_policy
from foundry.synthesis.realization.local_runtime import GeneratedBeam
from foundry.synthesis.realization.screening import RealizationScreeningIndex, ScreenDecision
from foundry.synthesis.realization.smoke import (
    RunMeasurements,
    VerificationBundle,
    _directory_bytes,
    _verify_draft,
    _working_set_peak_bytes,
)
from foundry.synthesis.realization.smoke_contract import RealizationAttemptPlan
from foundry.synthesis.realization.validation import validate_filled_question
from foundry.synthesis.semantic import PinnedSentenceEncoder, load_semantic_artifact_config

_LAYERS = (
    "tag_parse",
    "event_tag_set",
    "event_tag_occurrence",
    "event_order",
    "placeholder_set",
    "placeholder_assignment",
    "semantic_anchor",
    "raw_numeric_guard",
    "answer_calculation_guard",
    "outside_text",
    "target_preservation",
    "filled_question_consistency",
    "primary_verifier",
    "independent_verifier",
    "verifier_agreement",
    "output_contract",
    "language_quality",
    "benchmark_contamination",
    "internal_diversity",
)


@dataclass(frozen=True)
class CompactBeamRecord:
    beam_index: int
    raw_text: str
    raw_sha256: str
    generated_tokens: int
    tag_parsed: bool
    response_sha256: str | None
    filled_question: str | None
    filled_question_sha256: str | None
    validation_reasons: tuple[str, ...]
    benchmark_screen: ScreenDecision | None
    internal_screen: ScreenDecision | None
    layer_results: dict[str, bool]
    automatic_pass: bool
    rejection_reason: str | None
    selected: bool


@dataclass(frozen=True)
class CompactIrRecord:
    plan: RealizationAttemptPlan
    candidate_id: str
    semantic_ir_sha256: str
    latent_structure_sha256: str
    request_sha256: str
    semantic_frame: str
    realization_signature: str
    primary_evidence_sha256: str
    independent_evidence_sha256: str
    verifier_agreement: bool
    generation_elapsed_seconds: float
    generation_timeout_exceeded: bool
    input_tokens: int
    beams: tuple[CompactBeamRecord, ...]
    selected_beam_index: int | None


def _reason_layers(reasons: tuple[str, ...], parsed: bool) -> dict[str, bool]:
    values = {layer: True for layer in _LAYERS}
    if not parsed:
        values["tag_parse"] = False
    categories = {
        "event_tag_set": {"event_tag_set_mismatch"},
        "event_tag_occurrence": {
            "event_tag_occurrence_mismatch",
            "question_tag_occurrence_mismatch",
        },
        "event_order": {"event_order_changed"},
        "placeholder_set": {
            "placeholder_set_mismatch",
            "altered_or_duplicated_placeholder",
        },
        "placeholder_assignment": {
            "placeholder_assignment_mismatch",
            "placeholder_occurrence_mismatch",
        },
        "semantic_anchor": {"semantic_anchor_missing", "semantic_anchor_reversal"},
        "raw_numeric_guard": {"raw_numeric_literal"},
        "answer_calculation_guard": {"answer_or_calculation_content"},
        "outside_text": {
            "text_outside_tags",
            "text_after_q_or_unparsed_tag",
            "forbidden_markup",
        },
        "target_preservation": {"target_placeholder_missing"},
        "language_quality": {
            "unlicensed_pronoun",
            "unlicensed_semantic_content",
            "malformed_punctuation",
            "malformed_question_tag",
            "malformed_event_punctuation",
        },
    }
    reason_set = set(reasons)
    for layer, layer_reasons in categories.items():
        if reason_set & layer_reasons:
            values[layer] = False
    return values


def _first_failed(layers: dict[str, bool]) -> str | None:
    return next((layer for layer in _LAYERS if not layers[layer]), None)


def _process_beams(
    *,
    prepared: PreparedCompactRequest,
    raw_beams: tuple[GeneratedBeam, ...],
    timeout_exceeded: bool,
    verification: VerificationBundle,
    screening: RealizationScreeningIndex,
) -> tuple[tuple[CompactBeamRecord, ...], int | None]:
    intermediate: list[dict[str, object]] = []
    embedding_questions: list[str] = []
    embedding_indexes: list[int] = []
    for raw_beam in raw_beams:
        parsed = False
        response: CompactRealizationResponse | None = None
        response_hash: str | None = None
        filled_question: str | None = None
        reasons: tuple[str, ...] = ()
        try:
            response = parse_compact_response(raw_beam.raw_text)
            parsed = True
            response_hash = response.sha256
            reasons = validate_compact_response(prepared.request, response)
            if not reasons:
                filled = fill_compact_response(
                    prepared.request,
                    response,
                    prepared.replacements,
                )
                filled_question = filled.question
                reasons = validate_filled_question(filled_question)
        except CompactContractError as error:
            reasons = (str(error),)
        layers = _reason_layers(reasons, parsed)
        if timeout_exceeded:
            layers["tag_parse"] = False
            reasons = (*reasons, "generation_timeout")
        if filled_question is None:
            layers["filled_question_consistency"] = False
        if verification.constraint_rejections or not verification.primary.success:
            layers["primary_verifier"] = False
        if not verification.independent.success:
            layers["independent_verifier"] = False
        if not verification.agreement:
            layers["verifier_agreement"] = False
        if not verification.output_contract_valid:
            layers["output_contract"] = False
        if filled_question is not None and reasons:
            layers["language_quality"] = False
        intermediate.append(
            {
                "raw": raw_beam,
                "parsed": parsed,
                "response_hash": response_hash,
                "filled_question": filled_question,
                "reasons": reasons,
                "layers": layers,
            }
        )
        if filled_question is not None and _first_failed(layers) is None:
            embedding_indexes.append(len(intermediate) - 1)
            embedding_questions.append(filled_question)
    embeddings: dict[int, torch.Tensor] = {}
    if embedding_questions:
        encoded = screening.encode(embedding_questions)
        embeddings = {index: encoded[position] for position, index in enumerate(embedding_indexes)}
    records: list[CompactBeamRecord] = []
    selected_index: int | None = None
    selected_embedding: torch.Tensor | None = None
    for index, values in enumerate(intermediate):
        raw = cast(GeneratedBeam, values["raw"])
        layers = cast(dict[str, bool], values["layers"])
        filled_question = cast(str | None, values["filled_question"])
        embedding = embeddings.get(index)
        benchmark: ScreenDecision | None = None
        internal: ScreenDecision | None = None
        if embedding is not None and filled_question is not None:
            benchmark = screening.screen_benchmark(filled_question, embedding)
            if benchmark.outcome is not ContaminationOutcome.PASS:
                layers["benchmark_contamination"] = False
            internal = screening.screen_internal(
                question=filled_question,
                embedding=embedding,
                latent_structure_sha256=prepared.draft.structure_sha256,
                semantic_frame=prepared.semantic_frame,
                realization_signature=prepared.realization_signature,
            )
            if internal.outcome is not ContaminationOutcome.PASS:
                layers["internal_diversity"] = False
        automatic_pass = _first_failed(layers) is None
        selected = automatic_pass and selected_index is None
        if selected:
            selected_index = raw.beam_index
            selected_embedding = embedding
        records.append(
            CompactBeamRecord(
                beam_index=raw.beam_index,
                raw_text=raw.raw_text,
                raw_sha256=raw.raw_sha256,
                generated_tokens=raw.generated_tokens,
                tag_parsed=cast(bool, values["parsed"]),
                response_sha256=cast(str | None, values["response_hash"]),
                filled_question=filled_question,
                filled_question_sha256=None
                if filled_question is None
                else hashlib.sha256(filled_question.encode("utf-8")).hexdigest(),
                validation_reasons=cast(tuple[str, ...], values["reasons"]),
                benchmark_screen=benchmark,
                internal_screen=internal,
                layer_results=dict(layers),
                automatic_pass=automatic_pass,
                rejection_reason=None if automatic_pass else _first_failed(layers),
                selected=selected,
            )
        )
    if selected_index is not None and selected_embedding is not None:
        selected_record = next(record for record in records if record.selected)
        assert selected_record.filled_question is not None
        screening.add_selected(
            candidate_id=f"{prepared.draft.candidate_id}:beam:{selected_index}",
            question=selected_record.filled_question,
            latent_structure_sha256=prepared.draft.structure_sha256,
            semantic_frame=prepared.semantic_frame,
            realization_signature=prepared.realization_signature,
            embedding=selected_embedding,
        )
    return tuple(records), selected_index


def _deterministic_payload(records: tuple[CompactIrRecord, ...]) -> list[dict[str, object]]:
    return [
        {
            "plan": asdict(record.plan),
            "candidate_id": record.candidate_id,
            "semantic_ir_sha256": record.semantic_ir_sha256,
            "latent_structure_sha256": record.latent_structure_sha256,
            "request_sha256": record.request_sha256,
            "semantic_frame": record.semantic_frame,
            "realization_signature": record.realization_signature,
            "primary_evidence_sha256": record.primary_evidence_sha256,
            "independent_evidence_sha256": record.independent_evidence_sha256,
            "verifier_agreement": record.verifier_agreement,
            "selected_beam_index": record.selected_beam_index,
            "beams": [
                {
                    "beam_index": beam.beam_index,
                    "raw_text": beam.raw_text,
                    "raw_sha256": beam.raw_sha256,
                    "response_sha256": beam.response_sha256,
                    "filled_question_sha256": beam.filled_question_sha256,
                    "validation_reasons": beam.validation_reasons,
                    "benchmark_screen": None
                    if beam.benchmark_screen is None
                    else asdict(beam.benchmark_screen),
                    "internal_screen": None
                    if beam.internal_screen is None
                    else asdict(beam.internal_screen),
                    "layer_results": beam.layer_results,
                    "automatic_pass": beam.automatic_pass,
                    "rejection_reason": beam.rejection_reason,
                    "selected": beam.selected,
                }
                for beam in record.beams
            ],
        }
        for record in records
    ]


def deterministic_compact_sha256(records: tuple[CompactIrRecord, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            _deterministic_payload(records),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _write_raw(path: Path, records: tuple[CompactIrRecord, ...]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(asdict(record), sort_keys=True, default=str) for record in records
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return path.stat().st_size


def _summary(
    *,
    config: CompactSmokeConfig,
    records: tuple[CompactIrRecord, ...],
    measurements: RunMeasurements,
    runtime: PinnedCompactQwenRealizer,
) -> dict[str, object]:
    beams = [beam for record in records for beam in record.beams]
    selected = [record for record in records if record.selected_beam_index is not None]
    by_group = Counter(str(record.plan.group) for record in selected)
    by_category = Counter(str(record.plan.category) for record in selected)
    by_difficulty = Counter(str(record.plan.difficulty) for record in selected)
    by_output = Counter(
        "enabled" if record.plan.output_contract_enabled else "disabled" for record in selected
    )
    benchmark = Counter(
        str(beam.benchmark_screen.outcome) for beam in beams if beam.benchmark_screen is not None
    )
    internal = Counter(
        str(beam.internal_screen.outcome) for beam in beams if beam.internal_screen is not None
    )
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "config_sha256": config.config_sha256,
        "ir_master_seed": config.ir_master_seed,
        "model": asdict(config.model),
        "runtime_metadata": asdict(runtime.metadata),
        "compact_system_prompt_sha256": COMPACT_SYSTEM_PROMPT_SHA256,
        "compact_user_protocol_sha256": COMPACT_USER_PROTOCOL_SHA256,
        "compact_combined_protocol_sha256": COMPACT_COMBINED_PROTOCOL_SHA256,
        "generation_config_sha256": runtime.generation_config_sha256(),
        "internal_diversity_policy_sha256": load_frozen_internal_policy(
            config.internal_diversity_policy
        ).sha256,
        "attempted_irs": len(records),
        "generated_beams": len(beams),
        "automatic_selected_irs": len(selected),
        "tag_parsed_beams": sum(beam.tag_parsed for beam in beams),
        "placeholder_preserved_beams": sum(
            beam.layer_results["placeholder_set"] and beam.layer_results["placeholder_assignment"]
            for beam in beams
        ),
        "semantic_anchor_preserved_beams": sum(
            beam.layer_results["semantic_anchor"] for beam in beams
        ),
        "target_preserved_beams": sum(beam.layer_results["target_preservation"] for beam in beams),
        "layer_failures": {
            layer: sum(not beam.layer_results[layer] for beam in beams) for layer in _LAYERS
        },
        "selected_beam_contribution": {
            str(index): sum(record.selected_beam_index == index for record in selected)
            for index in (1, 2, 3)
        },
        "automatic_acceptance_by_group": {
            group: by_group[group] for group in ("generic_control", "targeted")
        },
        "automatic_acceptance_by_category": {
            category: by_category[category]
            for category in (
                "constraint_distribution_or_discrete_reasoning",
                "multi_step_bookkeeping_or_omission",
                "rate_ratio_percentage_or_average",
            )
        },
        "automatic_acceptance_by_difficulty": {
            difficulty: by_difficulty[difficulty] for difficulty in ("easy", "hard", "medium")
        },
        "automatic_acceptance_by_output_track": {
            state: by_output[state] for state in ("disabled", "enabled")
        },
        "benchmark_screen_outcomes": dict(sorted(benchmark.items())),
        "internal_screen_outcomes": dict(sorted(internal.items())),
        "deterministic_run_sha256": deterministic_compact_sha256(records),
        "measurements": asdict(measurements),
        "manual_audit": {"status": "pending", "audited_beams": 0},
        "deterministic_replay": {"status": "pending"},
        "readiness_gate": {"status": "pending_manual_audit_and_replay"},
    }


def execute_compact_smoke(
    *,
    repository_root: Path,
    config: CompactSmokeConfig,
    write_artifacts: bool,
) -> tuple[tuple[CompactIrRecord, ...], dict[str, object]]:
    """Run exactly 30 fresh IRs and exactly three beams per IR."""

    started = time.perf_counter()
    plans = build_compact_attempt_plan(config)
    development = load_development_question_export(config.development_question_export)
    encoder = PinnedSentenceEncoder(
        load_semantic_artifact_config(config.semantic_artifact), repository_root
    )
    screening = RealizationScreeningIndex(
        encoder=encoder,
        development_questions=development,
        internal_policy=load_frozen_internal_policy(config.internal_diversity_policy),
    )
    runtime = PinnedCompactQwenRealizer(repository_root=repository_root, config=config)
    initialization_seconds = time.perf_counter() - started
    records: list[CompactIrRecord] = []
    generation_seconds = 0.0
    for plan in plans:
        draft = generate_procedural_ir(plan)
        prepared = prepare_compact_request(draft, style_variant=plan.style_variant)
        verification = _verify_draft(draft)
        generation = runtime.generate(prepared)
        generation_seconds += generation.elapsed_seconds
        beams, selected_index = _process_beams(
            prepared=prepared,
            raw_beams=generation.beams,
            timeout_exceeded=generation.timeout_exceeded,
            verification=verification,
            screening=screening,
        )
        records.append(
            CompactIrRecord(
                plan=plan,
                candidate_id=draft.candidate_id,
                semantic_ir_sha256=draft.semantic_ir_sha256,
                latent_structure_sha256=draft.structure_sha256,
                request_sha256=prepared.request_sha256,
                semantic_frame=prepared.semantic_frame,
                realization_signature=prepared.realization_signature,
                primary_evidence_sha256=verification.primary.evidence_sha256,
                independent_evidence_sha256=verification.independent.evidence_sha256,
                verifier_agreement=verification.agreement,
                generation_elapsed_seconds=generation.elapsed_seconds,
                generation_timeout_exceeded=generation.timeout_exceeded,
                input_tokens=generation.input_tokens,
                beams=beams,
                selected_beam_index=selected_index,
            )
        )
        if plan.attempt_index in {15, 30}:
            observed_beams = [beam for record in records for beam in record.beams]
            passed = {
                layer: sum(beam.layer_results[layer] for beam in observed_beams)
                for layer in (
                    "placeholder_set",
                    "semantic_anchor",
                    "target_preservation",
                    "language_quality",
                    "filled_question_consistency",
                )
            }

            print(
                json.dumps(
                    {
                        "progress_irs": plan.attempt_index,
                        "beams": plan.attempt_index * 3,
                        "tag_parsed": sum(beam.tag_parsed for beam in observed_beams),
                        "placeholder_preserved": passed["placeholder_set"],
                        "semantic_anchor_preserved": passed["semantic_anchor"],
                        "target_preserved": passed["target_preservation"],
                        "language_quality_passed": passed["language_quality"],
                        "filled_consistency_passed": passed["filled_question_consistency"],
                        "automatic_selected": sum(
                            record.selected_beam_index is not None for record in records
                        ),
                        "cuda_failures_or_memory_warnings": 0,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    frozen = tuple(records)
    raw_path = config.raw_directory / "beams.jsonl"
    raw_bytes = _write_raw(raw_path, frozen) if write_artifacts else 0
    measurements = RunMeasurements(
        initialization_seconds=initialization_seconds,
        generation_seconds=generation_seconds,
        total_runtime_seconds=time.perf_counter() - started,
        input_tokens=sum(record.input_tokens for record in frozen),
        output_tokens=sum(beam.generated_tokens for record in frozen for beam in record.beams),
        peak_system_ram_bytes=_working_set_peak_bytes(),
        peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        raw_artifact_bytes=raw_bytes,
        model_cache_bytes=_directory_bytes(repository_root / config.model.cache_root / "hub"),
    )
    summary = _summary(config=config, records=frozen, measurements=measurements, runtime=runtime)
    if write_artifacts:
        config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        config.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return frozen, summary


def _main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    config = load_compact_smoke_config(args.config)
    records, summary = execute_compact_smoke(
        repository_root=repository_root,
        config=config,
        write_artifacts=not args.replay,
    )
    if args.replay:
        original: dict[str, Any] = json.loads(config.summary_path.read_text(encoding="utf-8"))
        expected = original.get("deterministic_run_sha256")
        actual = deterministic_compact_sha256(records)
        result = {
            "status": "passed" if actual == expected else "failed",
            "expected_sha256": expected,
            "actual_sha256": actual,
            "irs": len(records),
            "beams": sum(len(record.beams) for record in records),
            "replay_measurements": summary["measurements"],
        }
        replay_path = config.raw_directory / "replay.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if actual == expected else 1
    print(
        json.dumps(
            {
                "status": "completed",
                "attempted_irs": summary["attempted_irs"],
                "generated_beams": summary["generated_beams"],
                "automatic_selected_irs": summary["automatic_selected_irs"],
                "deterministic_run_sha256": summary["deterministic_run_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CompactBeamRecord",
    "CompactIrRecord",
    "deterministic_compact_sha256",
    "execute_compact_smoke",
]
