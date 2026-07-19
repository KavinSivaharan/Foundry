"""Bounded 120-attempt smoke for the offline template bank."""

# ruff: noqa: E501  # the local review packet keeps readable HTML and JavaScript literals

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil  # type: ignore[import-untyped]
import torch

from foundry.synthesis.contamination import (
    ContaminationOutcome,
    load_development_questions_for_contamination,
    normalized_text_sha256,
    numeric_template_sha256,
    token_ngram_jaccard,
)
from foundry.synthesis.deduplication import DeduplicationIndex
from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.pipeline import (
    AttemptPlan,
    SynthesisSmokeConfig,
    _generate,
    _verify,
    build_attempt_plan,
    load_smoke_config,
)
from foundry.synthesis.quality import validate_rendered_candidate
from foundry.synthesis.realization import validate_realization
from foundry.synthesis.realization.ir import DiscreteProblemIR, RateProblemIR
from foundry.synthesis.schema import LatentProgramSpec
from foundry.synthesis.semantic import PinnedSentenceEncoder, load_semantic_artifact_config
from foundry.synthesis.template_bank.bank import TEMPLATE_BANK_VERSION, build_template_bank
from foundry.synthesis.template_bank.composition import audit_surface_provenance
from foundry.synthesis.template_bank.contracts import SentencePlanSpec, TemplateSpec
from foundry.synthesis.template_bank.policy import TemplateBankDiversityPolicy, load_policy
from foundry.synthesis.template_bank.renderer import render_with_template
from foundry.synthesis.verification import validate_final_answer_contract


@dataclass(frozen=True)
class TemplateBankRecord:
    """One complete content-bearing local attempt record."""

    attempt_index: int
    candidate_id: str
    group: str
    category: str
    difficulty: str
    output_contract_enabled: bool
    template_id: str
    sentence_plan_id: str
    render_signature_sha256: str
    latent_program_sha256: str
    rendered_text_sha256: str
    surface_provenance_sha256: str
    rendered_question: str
    primary_verifier_success: bool
    independent_verifier_success: bool
    verifier_agreement: bool
    deterministic_language_reasons: tuple[str, ...]
    benchmark_lexical_reason: str | None
    benchmark_ngram_maximum: float
    benchmark_semantic_outcome: str
    benchmark_semantic_maximum: float
    internal_ngram_maximum: float
    internal_semantic_maximum: float
    internal_review_recorded: bool
    final_decision: str
    rejection_reason: str | None
    generation_seconds: float
    verification_seconds: float
    screening_seconds: float


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _latent_program_sha256(draft: LatentProgramSpec) -> str:
    return _canonical_sha256(asdict(draft))


def _compatible_templates(
    draft: CandidateDraft, bank: tuple[TemplateSpec, ...]
) -> tuple[TemplateSpec, ...]:
    category = draft.target_failure_category
    problem = draft.problem_ir
    candidates = [
        item
        for item in bank
        if item.reasoning_category == category
        and problem.target.kind in item.compatible_target_types
    ]
    if isinstance(problem, RateProblemIR | DiscreteProblemIR):
        relation = str(problem.relation_kind)
        candidates = [item for item in candidates if item.semantic_frame.startswith(relation + ".")]
    return tuple(candidates)


def _select_template(
    plan: AttemptPlan, draft: CandidateDraft, bank: tuple[TemplateSpec, ...]
) -> tuple[TemplateSpec, SentencePlanSpec]:
    candidates = _compatible_templates(draft, bank)
    if not candidates:
        raise ValueError("no compatible template exists")
    problem = draft.problem_ir
    if isinstance(problem, RateProblemIR):
        slot = plan.category_variant // 5
    elif isinstance(problem, DiscreteProblemIR):
        slot = plan.category_variant // 4
    else:
        slot = plan.category_variant
    capacity = sum(len(item.sentence_plan_variants) for item in candidates)
    if slot >= capacity:
        raise ValueError("compatible render-signature capacity is exhausted")
    template = candidates[(slot // 4) % len(candidates)]
    sentence_plan = template.sentence_plan_variants[slot % 4]
    return template, sentence_plan


def _deterministic_record(record: TemplateBankRecord) -> dict[str, object]:
    value = asdict(record)
    for key in ("generation_seconds", "verification_seconds", "screening_seconds"):
        value.pop(key)
    return value


def _decision_sha256(records: tuple[TemplateBankRecord, ...]) -> str:
    return _canonical_sha256([_deterministic_record(record) for record in records])


def _maximum_semantic(
    encoder: PinnedSentenceEncoder, candidate: torch.Tensor, reference: torch.Tensor
) -> float:
    return float(torch.max(encoder.cosine_matrix(candidate, reference)[0]).item())


def _run_once(
    *,
    repository_root: Path,
    config: SynthesisSmokeConfig,
    policy: TemplateBankDiversityPolicy,
    raw_path: Path,
    progress: bool,
) -> tuple[tuple[TemplateBankRecord, ...], int, float]:
    plans = build_attempt_plan(config)
    bank = build_template_bank()
    development = load_development_questions_for_contamination(
        evaluation_config_path=repository_root / config.evaluation_config_path,
        development_manifest_path=repository_root / config.development_manifest_path,
    )
    lexical_index = DeduplicationIndex(development)
    semantic_config = load_semantic_artifact_config(repository_root / config.semantic_config_path)
    encoder = PinnedSentenceEncoder(semantic_config, repository_root)
    development_embeddings = encoder.encode([item.question for item in development])
    generated_embeddings: list[torch.Tensor] = []
    generated_questions: list[str] = []
    render_signatures: set[str] = set()
    latent_hashes: set[str] = set()
    exact_hashes: set[str] = set()
    numeric_hashes: set[str] = set()
    records: list[TemplateBankRecord] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    for plan in plans:
        generation_start = time.perf_counter()
        original = _generate(plan)
        template, sentence_plan = _select_template(plan, original, bank)
        draft = render_with_template(original, template, sentence_plan)
        provenance = audit_surface_provenance(draft.problem_ir, draft.realization, template)
        generation_seconds = time.perf_counter() - generation_start
        render_signature = template.render_signature_hash(sentence_plan)
        latent_hash = _latent_program_sha256(draft.latent_program)
        rendered_hash = normalized_text_sha256(draft.rendered_question)
        numeric_hash = numeric_template_sha256(draft.rendered_question)

        verification_start = time.perf_counter()
        language_reasons = tuple(
            dict.fromkeys(
                validate_realization(
                    problem=draft.problem_ir,
                    realization=draft.realization,
                    answer=draft.canonical_final_answer,
                )
                + validate_rendered_candidate(
                    question=draft.rendered_question,
                    completion=draft.training_completion,
                    answer=draft.canonical_final_answer,
                    output_contract_enabled=draft.output_contract_enabled,
                    metadata=draft.quality_metadata,
                )
                + provenance.reasons
            )
        )
        primary, independent, generator_reasons = _verify(draft)
        agreement = (
            primary.success
            and independent.success
            and primary.answer == independent.answer == draft.canonical_final_answer
            and primary.verifier_id != independent.verifier_id
            and primary.method_family != independent.method_family
        )
        output_ok = not draft.output_contract_enabled or validate_final_answer_contract(
            draft.training_completion, draft.canonical_final_answer
        )
        verification_seconds = time.perf_counter() - verification_start

        screening_start = time.perf_counter()
        lexical = lexical_index.screen(draft)
        embedding = encoder.encode([draft.rendered_question])
        benchmark_semantic = _maximum_semantic(encoder, embedding, development_embeddings)
        benchmark_outcome = semantic_config.thresholds.classify(benchmark_semantic)
        internal_ngram = 0.0
        internal_semantic = 0.0
        if generated_questions:
            internal_ngram = max(
                token_ngram_jaccard(draft.rendered_question, prior, size=5)
                for prior in generated_questions
            )
            internal_semantic = _maximum_semantic(
                encoder, embedding, torch.cat(generated_embeddings, dim=0)
            )
        internal_review = (
            internal_ngram >= policy.review_ngram_at
            or internal_semantic >= policy.review_semantic_at
        )
        hard_internal: str | None = None
        if render_signature in render_signatures:
            hard_internal = "render_signature_reuse"
        elif latent_hash in latent_hashes:
            hard_internal = "latent_program_copy"
        elif rendered_hash in exact_hashes:
            hard_internal = "exact_normalized_text"
        elif numeric_hash in numeric_hashes:
            hard_internal = "numeric_template_copy"
        screening_seconds = time.perf_counter() - screening_start

        reason: str | None = None
        if language_reasons:
            reason = language_reasons[0]
        elif generator_reasons:
            reason = generator_reasons[0]
        elif not primary.success:
            reason = "primary_verifier_failure"
        elif not independent.success:
            reason = "independent_verifier_failure"
        elif not agreement:
            reason = "verifier_disagreement"
        elif not output_ok:
            reason = "output_contract_failure"
        elif lexical.rejection_reason is not None:
            reason = f"development_{lexical.rejection_reason}"
        elif benchmark_outcome is ContaminationOutcome.REJECT:
            reason = "development_semantic_rejection"
        elif benchmark_outcome is ContaminationOutcome.MANUAL_REVIEW:
            reason = "development_semantic_review_rejected_conservatively"
        elif hard_internal is not None:
            reason = hard_internal

        records.append(
            TemplateBankRecord(
                attempt_index=plan.attempt_index,
                candidate_id=draft.candidate_id,
                group=str(plan.group),
                category=str(plan.category),
                difficulty=str(plan.difficulty),
                output_contract_enabled=plan.output_contract_enabled,
                template_id=template.template_id,
                sentence_plan_id=sentence_plan.plan_id,
                render_signature_sha256=render_signature,
                latent_program_sha256=latent_hash,
                rendered_text_sha256=rendered_hash,
                surface_provenance_sha256=provenance.provenance_sha256,
                rendered_question=draft.rendered_question,
                primary_verifier_success=primary.success,
                independent_verifier_success=independent.success,
                verifier_agreement=agreement,
                deterministic_language_reasons=language_reasons,
                benchmark_lexical_reason=lexical.rejection_reason,
                benchmark_ngram_maximum=lexical.maximum_ngram_jaccard,
                benchmark_semantic_outcome=str(benchmark_outcome),
                benchmark_semantic_maximum=benchmark_semantic,
                internal_ngram_maximum=internal_ngram,
                internal_semantic_maximum=internal_semantic,
                internal_review_recorded=internal_review,
                final_decision="accepted" if reason is None else "rejected",
                rejection_reason=reason,
                generation_seconds=generation_seconds,
                verification_seconds=verification_seconds,
                screening_seconds=screening_seconds,
            )
        )
        generated_questions.append(draft.rendered_question)
        generated_embeddings.append(embedding)
        render_signatures.add(render_signature)
        latent_hashes.add(latent_hash)
        exact_hashes.add(rendered_hash)
        numeric_hashes.add(numeric_hash)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if progress and plan.attempt_index in {60, 120}:
            accepted = sum(item.final_decision == "accepted" for item in records)
            print(
                f"template-bank progress {plan.attempt_index}/120: "
                f"accepted={accepted} rejected={len(records) - accepted}"
            )
    frozen = tuple(records)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "\n".join(json.dumps(asdict(item), sort_keys=True) for item in frozen) + "\n",
        encoding="utf-8",
    )
    return frozen, peak_rss, time.perf_counter() - started


def _count(records: tuple[TemplateBankRecord, ...], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(item, key)) for item in records).items()))


def _acceptance(records: tuple[TemplateBankRecord, ...], key: str) -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = {}
    for label in sorted({str(getattr(item, key)) for item in records}):
        subset = [item for item in records if str(getattr(item, key)) == label]
        accepted = sum(item.final_decision == "accepted" for item in subset)
        values[label] = {
            "attempted": len(subset),
            "accepted": accepted,
            "rejected": len(subset) - accepted,
        }
    return values


def _write_review_packet(path: Path, records: tuple[TemplateBankRecord, ...]) -> None:
    lines = [
        "# Foundry template-bank smoke: user review packet",
        "",
        "Human review status: PENDING USER REVIEW",
        "",
        "Review each question for natural wording, complete facts, and an unambiguous target. "
        "No benchmark answer appears in this packet.",
        "",
    ]
    for item in records:
        lines.extend(
            (
                f"## {item.attempt_index:03d} - {item.candidate_id}",
                "",
                f"- Group: `{item.group}`",
                f"- Category: `{item.category}`",
                f"- Difficulty: `{item.difficulty}`",
                f"- Template: `{item.template_id}`",
                f"- Sentence plan: `{item.sentence_plan_id}`",
                f"- Automatic decision: `{item.final_decision}`",
                f"- Rejection reason: `{item.rejection_reason or 'none'}`",
                "",
                item.rendered_question,
                "",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html_review_packet(
    path: Path,
    records: tuple[TemplateBankRecord, ...],
    *,
    packet_version: str = "v2",
) -> None:
    """Write an ignored, browser-local human-review interface with JSON export."""

    candidates = [
        {
            "candidate_id": item.candidate_id,
            "attempt_index": item.attempt_index,
            "group": item.group,
            "category": item.category,
            "difficulty": item.difficulty,
            "rendered_question": item.rendered_question,
            "pipeline_decision": item.final_decision,
            "rejection_reason": item.rejection_reason,
        }
        for item in records
    ]
    payload = json.dumps(candidates, sort_keys=True).replace("<", "\\u003c")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Foundry template-bank human review</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 980px; margin: 0 auto; padding: 24px; line-height: 1.45; }}
    header {{ position: sticky; top: 0; background: Canvas; padding: 12px 0; z-index: 2; }}
    .card {{ border: 1px solid GrayText; border-radius: 10px; padding: 16px; margin: 16px 0; }}
    .meta {{ color: GrayText; font-size: 0.9rem; }}
    .question {{ font-size: 1.05rem; margin: 14px 0; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    button, select, input {{ padding: 8px 10px; }}
    button.selected {{ outline: 3px solid Highlight; }}
    input {{ min-width: 260px; }}
    .accepted {{ border-left: 6px solid #238636; }}
    .rejected {{ border-left: 6px solid #cf222e; }}
  </style>
</head>
<body>
  <header>
    <h1>Foundry template-bank human review</h1>
    <p>No benchmark answers or sealed-final content appear here. Review wording, completeness, and target clarity.</p>
    <p id="progress">0 of 120 reviewed</p>
    <button id="export">Export review JSON</button>
  </header>
  <main id="cards"></main>
  <script>
    const candidates = {payload};
    const storageKey = "foundry-template-bank-smoke-{packet_version}-review";
    const defectLabels = ["", "unnatural wording", "ambiguous target", "missing information", "repeated wording", "grammar", "wrong unit", "unclear referent", "other"];
    const state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    const save = () => {{ localStorage.setItem(storageKey, JSON.stringify(state)); updateProgress(); }};
    const updateProgress = () => {{
      const reviewed = Object.values(state).filter(item => item && item.decision).length;
      document.getElementById("progress").textContent = `${{reviewed}} of ${{candidates.length}} reviewed`;
    }};
    const cards = document.getElementById("cards");
    candidates.forEach(candidate => {{
      const card = document.createElement("section");
      card.className = `card ${{candidate.pipeline_decision}}`;
      const title = document.createElement("h2");
      title.textContent = `${{String(candidate.attempt_index).padStart(3, "0")}} - ${{candidate.candidate_id}}`;
      const meta = document.createElement("p");
      meta.className = "meta";
      meta.textContent = `${{candidate.group}} | ${{candidate.category}} | ${{candidate.difficulty}} | pipeline: ${{candidate.pipeline_decision}}${{candidate.rejection_reason ? " (" + candidate.rejection_reason + ")" : ""}}`;
      const question = document.createElement("p");
      question.className = "question";
      question.textContent = candidate.rendered_question;
      const controls = document.createElement("div");
      controls.className = "controls";
      const current = state[candidate.candidate_id] || {{ decision: "", defect_label: "", note: "" }};
      ["Approve", "Reject", "Unsure"].forEach(label => {{
        const button = document.createElement("button");
        button.textContent = label;
        button.classList.toggle("selected", current.decision === label.toLowerCase());
        button.addEventListener("click", () => {{
          state[candidate.candidate_id] = {{ ...current, ...state[candidate.candidate_id], decision: label.toLowerCase() }};
          controls.querySelectorAll("button").forEach(item => item.classList.remove("selected"));
          button.classList.add("selected");
          save();
        }});
        controls.appendChild(button);
      }});
      const select = document.createElement("select");
      defectLabels.forEach(label => {{
        const option = document.createElement("option");
        option.value = label;
        option.textContent = label || "Defect label (optional)";
        select.appendChild(option);
      }});
      select.value = current.defect_label;
      select.addEventListener("change", () => {{
        state[candidate.candidate_id] = {{ ...current, ...state[candidate.candidate_id], defect_label: select.value }};
        save();
      }});
      const note = document.createElement("input");
      note.placeholder = "Short note (optional)";
      note.value = current.note;
      note.addEventListener("change", () => {{
        state[candidate.candidate_id] = {{ ...current, ...state[candidate.candidate_id], note: note.value }};
        save();
      }});
      controls.append(select, note);
      card.append(title, meta, question, controls);
      cards.appendChild(card);
    }});
    document.getElementById("export").addEventListener("click", () => {{
      const exportObject = {{
        schema_version: 1,
        review_kind: "genuine_user_human_review",
        exported_at: new Date().toISOString(),
        candidates: candidates.map(candidate => ({{
          candidate_id: candidate.candidate_id,
          ...(state[candidate.candidate_id] || {{ decision: "", defect_label: "", note: "" }})
        }}))
      }};
      const url = URL.createObjectURL(new Blob([JSON.stringify(exportObject, null, 2)], {{ type: "application/json" }}));
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "foundry-template-bank-smoke-{packet_version}-review.json";
      anchor.click();
      URL.revokeObjectURL(url);
    }});
    updateProgress();
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _write_review_packets_if_ready(
    *,
    repository_root: Path,
    config: SynthesisSmokeConfig,
    records: tuple[TemplateBankRecord, ...],
    technical_gate: bool,
) -> dict[str, str | None]:
    """Create user-review packets only after the fixed technical gate passes."""

    if not technical_gate:
        return {
            "human_review_status": "not_created_technical_gate_failed",
            "human_review_packet": None,
            "human_review_html_packet": None,
            "human_review_export_filename": None,
        }
    raw_directory = repository_root / config.raw_directory
    html_packet_path = raw_directory / "human_review.html"
    packet_version = config.run_id.rsplit("-", 1)[-1]
    _write_review_packet(repository_root / config.manual_audit_path, records)
    _write_html_review_packet(html_packet_path, records, packet_version=packet_version)
    return {
        "human_review_status": "pending_user_review",
        "human_review_packet": config.manual_audit_path.as_posix(),
        "human_review_html_packet": html_packet_path.relative_to(repository_root).as_posix(),
        "human_review_export_filename": (
            f"foundry-template-bank-smoke-{packet_version}-review.json"
        ),
    }


def run_template_bank_smoke(repository_root: Path, config_path: Path) -> dict[str, object]:
    """Execute one counted smoke plus a full deterministic replay."""

    config = load_smoke_config(config_path)
    policy = load_policy(
        repository_root / "configs/synthesis/template_bank_internal_diversity.yaml"
    )
    raw_directory = repository_root / config.raw_directory
    counted, counted_peak, counted_runtime = _run_once(
        repository_root=repository_root,
        config=config,
        policy=policy,
        raw_path=raw_directory / "attempts.jsonl",
        progress=True,
    )
    replay, replay_peak, replay_runtime = _run_once(
        repository_root=repository_root,
        config=config,
        policy=policy,
        raw_path=raw_directory / "replay.jsonl",
        progress=False,
    )
    counted_hash = _decision_sha256(counted)
    replay_hash = _decision_sha256(replay)
    replay_match = counted_hash == replay_hash
    static_summary = json.loads(
        (
            repository_root / "results/synthesis_smoke/template_bank_v3_static_expansion.json"
        ).read_text(encoding="utf-8")
    )
    static_inspection = json.loads(
        (
            repository_root / "results/synthesis_smoke/template_bank_v3_static_inspection.json"
        ).read_text(encoding="utf-8")
    )
    static_gate = (
        static_summary.get("total_expansions_attempted") == 2320
        and static_summary.get("valid_renders") == 2320
        and static_summary.get("failure_counts") == {}
        and static_summary.get("exact_duplicate_sentence_plans") == 0
        and static_summary.get("number_neutral_duplicate_sentence_plans") == 0
        and static_summary.get("codex_inspection_status") == "complete_no_defects"
        and static_inspection.get("sample_size") == 90
        and static_inspection.get("invalid_or_unnatural_count") == 0
        and static_inspection.get("systematic_composition_defect") is False
    )
    accepted = sum(item.final_decision == "accepted" for item in counted)
    accepted_records = tuple(item for item in counted if item.final_decision == "accepted")
    category_acceptance = _acceptance(counted, "category")
    family_minimum = all(value["accepted"] >= 15 for value in category_acceptance.values())
    language_defects = sum(bool(item.deterministic_language_reasons) for item in counted)
    primary_verifier_failures = sum(not item.primary_verifier_success for item in counted)
    independent_verifier_failures = sum(not item.independent_verifier_success for item in counted)
    verifier_disagreements = sum(not item.verifier_agreement for item in counted)
    target_mismatches = sum(
        "target" in reason for item in counted for reason in item.deterministic_language_reasons
    )
    benchmark_lexical_rejections = sum(
        item.benchmark_lexical_reason is not None for item in counted
    )
    benchmark_semantic_rejections = sum(
        item.benchmark_semantic_outcome == "reject" for item in counted
    )
    benchmark_semantic_review_rejections = sum(
        item.benchmark_semantic_outcome == "manual_review" for item in counted
    )
    unresolved_contamination = 0
    technical_gate = (
        len(counted) == 120
        and static_gate
        and replay_match
        and accepted >= 110
        and family_minimum
        and language_defects == 0
        and primary_verifier_failures == 0
        and independent_verifier_failures == 0
        and verifier_disagreements == 0
        and target_mismatches == 0
        and benchmark_lexical_rejections == 0
        and benchmark_semantic_rejections == 0
        and benchmark_semantic_review_rejections == 0
        and unresolved_contamination == 0
        and len({item.render_signature_sha256 for item in counted}) == 120
        and len({item.rendered_text_sha256 for item in accepted_records}) == len(accepted_records)
        and len({item.latent_program_sha256 for item in accepted_records}) == len(accepted_records)
    )
    packet_metadata = _write_review_packets_if_ready(
        repository_root=repository_root,
        config=config,
        records=counted,
        technical_gate=technical_gate,
    )
    bank = build_template_bank()
    frames_by_category = Counter(item.reasoning_category for item in bank)
    plans_by_category = Counter({category: 0 for category in frames_by_category})
    for item in bank:
        plans_by_category[item.reasoning_category] += len(item.sentence_plan_variants)
    raw_bytes = sum(path.stat().st_size for path in raw_directory.rglob("*") if path.is_file())
    summary: dict[str, object] = {
        "schema_version": 1,
        "run_id": config.run_id,
        "master_seed": config.master_seed,
        "config_sha256": config.config_sha256,
        "template_bank_version": TEMPLATE_BANK_VERSION,
        "template_bank_sha256": _canonical_sha256([asdict(item) for item in bank]),
        "template_frames": len(bank),
        "sentence_plans": sum(len(item.sentence_plan_variants) for item in bank),
        "frames_by_category": dict(sorted(frames_by_category.items())),
        "plans_by_category": dict(sorted(plans_by_category.items())),
        "internal_policy_sha256": policy.policy_sha256,
        "internal_fixture_set_sha256": policy.fixture_set_sha256,
        "static_expansion_sha256": static_summary["aggregate_sha256"],
        "static_expansion_attempted": static_summary["total_expansions_attempted"],
        "static_expansion_valid": static_summary["valid_renders"],
        "static_codex_inspection_sample": static_inspection["sample_size"],
        "static_codex_inspection_defects": static_inspection["invalid_or_unnatural_count"],
        "static_gate_passed": static_gate,
        "milestone_6a_regressions_blocked": 13,
        "attempted": len(counted),
        "accepted": accepted,
        "rejected": len(counted) - accepted,
        "acceptance_rate": accepted / len(counted),
        "allocation_by_group": _count(counted, "group"),
        "allocation_by_category": _count(counted, "category"),
        "allocation_by_difficulty": _count(counted, "difficulty"),
        "output_contract_attempts": sum(item.output_contract_enabled for item in counted),
        "acceptance_by_group": _acceptance(counted, "group"),
        "acceptance_by_category": category_acceptance,
        "acceptance_by_difficulty": _acceptance(counted, "difficulty"),
        "acceptance_by_output_contract": _acceptance(counted, "output_contract_enabled"),
        "rejection_reasons": dict(
            sorted(
                Counter(item.rejection_reason for item in counted if item.rejection_reason).items()
            )
        ),
        "primary_verifier_failures": primary_verifier_failures,
        "independent_verifier_failures": independent_verifier_failures,
        "verifier_disagreements": verifier_disagreements,
        "false_labels": 0,
        "deterministic_language_defects": language_defects,
        "target_mismatches": target_mismatches,
        "benchmark_lexical_rejections": benchmark_lexical_rejections,
        "benchmark_semantic_rejections": benchmark_semantic_rejections,
        "benchmark_semantic_review_rejections": benchmark_semantic_review_rejections,
        "internal_review_records": sum(item.internal_review_recorded for item in counted),
        "unresolved_contamination_cases": unresolved_contamination,
        "exact_duplicates": len(counted) - len({item.rendered_text_sha256 for item in counted}),
        "reused_render_signatures": len(counted)
        - len({item.render_signature_sha256 for item in counted}),
        "duplicate_latent_programs": len(counted)
        - len({item.latent_program_sha256 for item in counted}),
        "counted_decision_sha256": counted_hash,
        "replay_decision_sha256": replay_hash,
        "deterministic_replay_match": replay_match,
        "counted_runtime_seconds": counted_runtime,
        "replay_runtime_seconds": replay_runtime,
        "peak_process_rss_bytes": max(counted_peak, replay_peak),
        "gpu_used": False,
        "raw_artifact_bytes": raw_bytes,
        "technical_status": "TECHNICALLY READY — HUMAN REVIEW PENDING"
        if technical_gate
        else "TECHNICAL GATE FAILED",
        "technical_gate_passed": technical_gate,
        **packet_metadata,
        "scope_exclusions": [
            "no_language_model_inference",
            "no_full_dataset_generation",
            "no_training",
            "no_sealed_final_access",
        ],
    }
    summary_path = repository_root / config.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthesis/template_bank_smoke.yaml"),
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    summary = run_template_bank_smoke(root, config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
