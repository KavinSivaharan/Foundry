"""Bounded Qwen3 realization smoke orchestration and replay evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from foundry.synthesis.contamination import ContaminationOutcome, normalized_text_sha256
from foundry.synthesis.generators import CandidateDraft, GeneratorVerification
from foundry.synthesis.generators.bookkeeping import (
    validate_bookkeeping_constraints,
    verify_bookkeeping_dag,
    verify_bookkeeping_ledger,
)
from foundry.synthesis.generators.discrete import (
    validate_discrete_constraints,
    verify_discrete_constructive,
    verify_discrete_enumeration,
)
from foundry.synthesis.generators.rates import (
    validate_rate_constraints,
    verify_rate_equation,
    verify_rate_inverse,
)
from foundry.synthesis.realization.development_export import (
    load_development_question_export,
)
from foundry.synthesis.realization.diversity import load_frozen_internal_policy
from foundry.synthesis.realization.local_runtime import PinnedQwenRealizer
from foundry.synthesis.realization.model_contracts import SYSTEM_PROMPT_SHA256
from foundry.synthesis.realization.prompting import (
    COMBINED_PROMPT_PROTOCOL_SHA256,
    USER_PROMPT_PROTOCOL_SHA256,
)
from foundry.synthesis.realization.request_builder import (
    PreparedRealizationRequest,
    prepare_realization_request,
)
from foundry.synthesis.realization.screening import RealizationScreeningIndex, ScreenDecision
from foundry.synthesis.realization.smoke_contract import (
    RealizationAttemptPlan,
    RealizationSmokeConfig,
    build_realization_attempt_plan,
    generate_procedural_ir,
    load_realization_smoke_config,
)
from foundry.synthesis.realization.validation import (
    RealizationContractError,
    fill_validated_template,
    parse_realization_response,
    response_sha256,
    validate_filled_question,
    validate_realization_response,
)
from foundry.synthesis.semantic import PinnedSentenceEncoder, load_semantic_artifact_config
from foundry.synthesis.taxonomy import FailureCategory
from foundry.synthesis.verification import validate_final_answer_contract


@dataclass(frozen=True)
class VerificationBundle:
    primary: GeneratorVerification
    independent: GeneratorVerification
    constraint_rejections: tuple[str, ...]
    agreement: bool
    output_contract_valid: bool


@dataclass(frozen=True)
class BeamRecord:
    beam_index: int
    raw_text: str
    raw_sha256: str
    generated_tokens: int
    parsed: bool
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
class IrRecord:
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
    beams: tuple[BeamRecord, ...]
    selected_beam_index: int | None


@dataclass(frozen=True)
class RunMeasurements:
    initialization_seconds: float
    generation_seconds: float
    total_runtime_seconds: float
    input_tokens: int
    output_tokens: int
    peak_system_ram_bytes: int
    peak_gpu_allocated_bytes: int
    peak_gpu_reserved_bytes: int
    raw_artifact_bytes: int
    model_cache_bytes: int


_LAYER_NAMES = (
    "strict_json",
    "schema",
    "placeholder_set",
    "placeholder_integrity",
    "raw_numeric_guard",
    "answer_calculation_guard",
    "semantic_node_coverage",
    "target_intent_preservation",
    "unit_interval_preservation",
    "entity_clause_discourse",
    "filled_question_consistency",
    "primary_verifier",
    "independent_verifier",
    "verifier_agreement",
    "output_contract",
    "language_quality",
    "benchmark_contamination",
    "internal_diversity",
)


def _working_set_peak_bytes() -> int:
    """Return the Windows process peak working set without an extra dependency."""

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_memory_info = psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_memory_info.restype = ctypes.c_int
    if not get_memory_info(get_current_process(), ctypes.byref(counters), ctypes.sizeof(counters)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _verify_draft(draft: CandidateDraft) -> VerificationBundle:
    if draft.target_failure_category == FailureCategory.MULTI_STEP_BOOKKEEPING:
        primary = verify_bookkeeping_dag(draft)
        independent = verify_bookkeeping_ledger(draft)
        constraints = validate_bookkeeping_constraints(draft)
    elif draft.target_failure_category == FailureCategory.RATE_RATIO_PERCENTAGE:
        primary = verify_rate_equation(draft)
        independent = verify_rate_inverse(draft)
        constraints = validate_rate_constraints(draft)
    elif draft.target_failure_category == FailureCategory.CONSTRAINT_DISCRETE:
        primary = verify_discrete_constructive(draft)
        independent = verify_discrete_enumeration(draft)
        constraints = validate_discrete_constraints(draft)
    else:
        raise ValueError("draft has an unapproved generator category")
    agreement = (
        primary.success
        and independent.success
        and primary.answer is not None
        and primary.answer == independent.answer == draft.canonical_final_answer
        and primary.verifier_id != independent.verifier_id
        and primary.method_family != independent.method_family
    )
    output_contract_valid = True
    if draft.output_contract_enabled:
        output_contract_valid = validate_final_answer_contract(
            draft.training_completion, draft.canonical_final_answer
        )
    return VerificationBundle(
        primary,
        independent,
        constraints,
        agreement,
        output_contract_valid,
    )


def _reason_layers(reasons: tuple[str, ...], parsed: bool) -> dict[str, bool]:
    values = {name: True for name in _LAYER_NAMES}
    if not parsed:
        values["strict_json"] = False
        values["schema"] = False
        return values
    categories = {
        "placeholder_set": {
            "duplicated_placeholder_inventory",
            "placeholder_set_mismatch",
            "template_placeholder_set_mismatch",
        },
        "placeholder_integrity": {"placeholder_occurrence_mismatch"},
        "raw_numeric_guard": {"invented_numeric_literal"},
        "answer_calculation_guard": {
            "answer_content_forbidden",
            "calculation_content_forbidden",
        },
        "semantic_node_coverage": {
            "invented_semantic_node",
            "semantic_node_placeholder_mismatch",
            "missing_semantic_node",
            "duplicated_semantic_node",
        },
        "target_intent_preservation": {
            "target_type_changed",
            "question_intent_changed",
            "conflicting_question_intent",
        },
        "unit_interval_preservation": {"rate_denominator_missing"},
        "entity_clause_discourse": {
            "invalid_clause_map_index",
            "discourse_order_changed",
            "style_id_changed",
            "unlicensed_pronoun",
        },
        "language_quality": {
            "malformed_question_template",
            "malformed_punctuation",
            "duplicated_clause",
        },
    }
    reason_set = set(reasons)
    for layer, layer_reasons in categories.items():
        if reason_set & layer_reasons:
            values[layer] = False
    return values


def _first_failed_layer(layers: dict[str, bool]) -> str | None:
    return next((name for name in _LAYER_NAMES if not layers[name]), None)


def _process_beams(
    *,
    prepared: PreparedRealizationRequest,
    raw_beams: tuple[Any, ...],
    timeout_exceeded: bool,
    verification: VerificationBundle,
    screening: RealizationScreeningIndex,
) -> tuple[tuple[BeamRecord, ...], int | None, torch.Tensor | None]:
    intermediate: list[dict[str, object]] = []
    filled_for_embedding: list[str] = []
    embedding_indexes: list[int] = []
    for raw_beam in raw_beams:
        parsed = False
        response_hash: str | None = None
        filled_question: str | None = None
        reasons: tuple[str, ...] = ()
        try:
            response = parse_realization_response(raw_beam.raw_text)
            parsed = True
            response_hash = response_sha256(response)
            reasons = validate_realization_response(prepared.request, response)
            if not reasons:
                filled = fill_validated_template(prepared.request, response, prepared.replacements)
                filled_question = filled.question
                reasons = validate_filled_question(filled_question)
        except RealizationContractError as error:
            reasons = (f"contract_error:{error}",)
        layers = _reason_layers(reasons, parsed)
        if timeout_exceeded:
            layers["strict_json"] = False
            reasons = (*reasons, "generation_timeout")
        if filled_question is None:
            layers["filled_question_consistency"] = False
        if verification.constraint_rejections:
            layers["primary_verifier"] = False
        if not verification.primary.success:
            layers["primary_verifier"] = False
        if not verification.independent.success:
            layers["independent_verifier"] = False
        if not verification.agreement:
            layers["verifier_agreement"] = False
        if not verification.output_contract_valid:
            layers["output_contract"] = False
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
        if filled_question is not None and _first_failed_layer(layers) is None:
            embedding_indexes.append(len(intermediate) - 1)
            filled_for_embedding.append(filled_question)
    embeddings: dict[int, torch.Tensor] = {}
    if filled_for_embedding:
        encoded = screening.encode(filled_for_embedding)
        embeddings = {
            record_index: encoded[position]
            for position, record_index in enumerate(embedding_indexes)
        }
    records: list[BeamRecord] = []
    selected_index: int | None = None
    selected_embedding: torch.Tensor | None = None
    for record_index, values in enumerate(intermediate):
        raw_beam = values["raw"]
        layers = cast(dict[str, bool], values["layers"])
        filled_question = cast(str | None, values["filled_question"])
        benchmark: ScreenDecision | None = None
        internal: ScreenDecision | None = None
        embedding = embeddings.get(record_index)
        if embedding is not None and filled_question is not None:
            benchmark = screening.screen_benchmark(filled_question, embedding)
            if benchmark.outcome is ContaminationOutcome.REJECT:
                layers["benchmark_contamination"] = False
            internal = screening.screen_internal(
                question=filled_question,
                embedding=embedding,
                latent_structure_sha256=prepared.draft.structure_sha256,
                semantic_frame=prepared.semantic_frame,
                realization_signature=prepared.realization_signature,
            )
            if internal.outcome is ContaminationOutcome.REJECT:
                layers["internal_diversity"] = False
        automatic_pass = _first_failed_layer(layers) is None
        selected = automatic_pass and selected_index is None
        if selected:
            selected_index = raw_beam.beam_index
            selected_embedding = embedding
        rejection_reason = None if automatic_pass else _first_failed_layer(layers)
        reasons = cast(tuple[str, ...], values["reasons"])
        records.append(
            BeamRecord(
                beam_index=raw_beam.beam_index,
                raw_text=raw_beam.raw_text,
                raw_sha256=raw_beam.raw_sha256,
                generated_tokens=raw_beam.generated_tokens,
                parsed=cast(bool, values["parsed"]),
                response_sha256=cast(str | None, values["response_hash"]),
                filled_question=filled_question,
                filled_question_sha256=None
                if filled_question is None
                else normalized_text_sha256(filled_question),
                validation_reasons=reasons,
                benchmark_screen=benchmark,
                internal_screen=internal,
                layer_results=dict(layers),
                automatic_pass=automatic_pass,
                rejection_reason=rejection_reason,
                selected=selected,
            )
        )
    if selected_index is not None and selected_embedding is not None:
        selected_record = records[selected_index - 1]
        if selected_record.filled_question is None:
            raise RuntimeError("selected beam has no filled question")
        screening.add_selected(
            candidate_id=f"{prepared.draft.candidate_id}:beam:{selected_index}",
            question=selected_record.filled_question,
            latent_structure_sha256=prepared.draft.structure_sha256,
            semantic_frame=prepared.semantic_frame,
            realization_signature=prepared.realization_signature,
            embedding=selected_embedding,
        )
    return tuple(records), selected_index, selected_embedding


def _raw_record(record: IrRecord) -> dict[str, object]:
    return cast(dict[str, object], asdict(record))


def _deterministic_payload(records: tuple[IrRecord, ...]) -> list[dict[str, object]]:
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


def deterministic_run_sha256(records: tuple[IrRecord, ...]) -> str:
    """Hash every replay-required field, including byte-identical beam text."""

    return hashlib.sha256(
        json.dumps(
            _deterministic_payload(records),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _write_raw_records(path: Path, records: tuple[IrRecord, ...]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(_raw_record(record), sort_keys=True, default=str) for record in records
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return path.stat().st_size


def _summary(
    *,
    config: RealizationSmokeConfig,
    records: tuple[IrRecord, ...],
    measurements: RunMeasurements,
    runtime: PinnedQwenRealizer,
) -> dict[str, object]:
    beams = [beam for record in records for beam in record.beams]
    parsed_beams = [beam for beam in beams if beam.parsed]
    selected = [record for record in records if record.selected_beam_index is not None]
    layer_failures = {
        layer: sum(not beam.layer_results[layer] for beam in beams) for layer in _LAYER_NAMES
    }
    selected_beams = Counter(record.selected_beam_index for record in selected)
    by_group = Counter(str(record.plan.group) for record in selected)
    by_category = Counter(str(record.plan.category) for record in selected)
    by_difficulty = Counter(str(record.plan.difficulty) for record in selected)
    output_track = Counter(
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
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "user_prompt_protocol_sha256": USER_PROMPT_PROTOCOL_SHA256,
        "combined_prompt_protocol_sha256": COMBINED_PROMPT_PROTOCOL_SHA256,
        "generation_config_sha256": runtime.generation_config_sha256(),
        "internal_diversity_policy_sha256": load_frozen_internal_policy(
            config.internal_diversity_policy
        ).sha256,
        "attempted_irs": len(records),
        "generated_beams": len(beams),
        "automatic_selected_irs": len(selected),
        "json_parsed_beams": len(parsed_beams),
        "placeholder_preserved_beams": sum(
            beam.layer_results["placeholder_set"] and beam.layer_results["placeholder_integrity"]
            for beam in parsed_beams
        ),
        "semantic_node_preserved_beams": sum(
            beam.layer_results["semantic_node_coverage"] for beam in parsed_beams
        ),
        "target_preserved_beams": sum(
            beam.layer_results["target_intent_preservation"] for beam in parsed_beams
        ),
        "layer_failures": layer_failures,
        "selected_beam_contribution": {str(index): selected_beams[index] for index in (1, 2, 3)},
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
            state: output_track[state] for state in ("disabled", "enabled")
        },
        "benchmark_screen_outcomes": dict(sorted(benchmark.items())),
        "internal_screen_outcomes": dict(sorted(internal.items())),
        "deterministic_run_sha256": deterministic_run_sha256(records),
        "measurements": asdict(measurements),
        "manual_audit": {"status": "pending", "audited_beams": 0},
        "deterministic_replay": {"status": "pending"},
        "readiness_gate": {"status": "pending_manual_audit_and_replay"},
    }


def execute_smoke(
    *,
    repository_root: Path,
    config: RealizationSmokeConfig,
    write_artifacts: bool,
) -> tuple[tuple[IrRecord, ...], dict[str, object]]:
    """Run exactly 120 IRs and three beams each under frozen controls."""

    started = time.perf_counter()
    plans = build_realization_attempt_plan(config)
    development = load_development_question_export(config.development_question_export)
    semantic_config = load_semantic_artifact_config(config.semantic_artifact)
    encoder = PinnedSentenceEncoder(semantic_config, repository_root)
    screening = RealizationScreeningIndex(
        encoder=encoder,
        development_questions=development,
        internal_policy=load_frozen_internal_policy(config.internal_diversity_policy),
    )
    runtime = PinnedQwenRealizer(repository_root=repository_root, config=config)
    initialization_seconds = time.perf_counter() - started
    records: list[IrRecord] = []
    generation_seconds = 0.0
    for plan in plans:
        draft = generate_procedural_ir(plan)
        prepared = prepare_realization_request(draft, style_variant=plan.style_variant)
        verification = _verify_draft(draft)
        generation = runtime.generate(prepared)
        generation_seconds += generation.elapsed_seconds
        beams, selected_index, _ = _process_beams(
            prepared=prepared,
            raw_beams=generation.beams,
            timeout_exceeded=generation.timeout_exceeded,
            verification=verification,
            screening=screening,
        )
        records.append(
            IrRecord(
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
        if plan.attempt_index in {30, 60, 90, 120}:
            print(
                json.dumps(
                    {
                        "progress_irs": plan.attempt_index,
                        "beams": plan.attempt_index * 3,
                        "automatic_selected": sum(
                            item.selected_beam_index is not None for item in records
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    frozen_records = tuple(records)
    total_runtime = time.perf_counter() - started
    raw_path = config.raw_directory / "beams.jsonl"
    raw_bytes = _write_raw_records(raw_path, frozen_records) if write_artifacts else 0
    measurements = RunMeasurements(
        initialization_seconds=initialization_seconds,
        generation_seconds=generation_seconds,
        total_runtime_seconds=total_runtime,
        input_tokens=sum(record.input_tokens for record in frozen_records),
        output_tokens=sum(
            beam.generated_tokens for record in frozen_records for beam in record.beams
        ),
        peak_system_ram_bytes=_working_set_peak_bytes(),
        peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        raw_artifact_bytes=raw_bytes,
        model_cache_bytes=_directory_bytes(repository_root / config.model.cache_root / "hub"),
    )
    summary = _summary(
        config=config,
        records=frozen_records,
        measurements=measurements,
        runtime=runtime,
    )
    if write_artifacts:
        config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        config.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return frozen_records, summary


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
    config = load_realization_smoke_config(args.config)
    records, summary = execute_smoke(
        repository_root=repository_root,
        config=config,
        write_artifacts=not args.replay,
    )
    if args.replay:
        original = json.loads(config.summary_path.read_text(encoding="utf-8"))
        expected = original.get("deterministic_run_sha256")
        actual = deterministic_run_sha256(records)
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
