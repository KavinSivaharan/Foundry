"""Bounded, non-dataset static expansion of every offline sentence plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from foundry.synthesis.contamination import normalized_text_sha256, numeric_template_sha256
from foundry.synthesis.pipeline import AttemptPlan, GroupName, _generate
from foundry.synthesis.quality import validate_rendered_candidate
from foundry.synthesis.realization import validate_realization
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.taxonomy import FailureCategory
from foundry.synthesis.template_bank.bank import build_template_bank
from foundry.synthesis.template_bank.composition import audit_surface_provenance
from foundry.synthesis.template_bank.contracts import SentencePlanSpec, TemplateSpec
from foundry.synthesis.template_bank.renderer import render_with_template

EXPANSION_FIXTURES_PER_PLAN = 10
EXPANSION_VERSION = "foundry-template-bank-static-expansion-v3"


@dataclass(frozen=True)
class ExpansionRecord:
    """One content-free static expansion decision."""

    attempt_index: int
    template_id: str
    sentence_plan_id: str
    category: str
    fixture_index: int
    difficulty: str
    output_contract_enabled: bool
    rendered_text_sha256: str
    numeric_template_sha256: str
    render_signature_sha256: str
    surface_provenance_sha256: str
    reasons: tuple[str, ...]


def _seed(template_id: str, plan_id: str, fixture_index: int) -> int:
    material = f"{EXPANSION_VERSION}:{template_id}:{plan_id}:{fixture_index}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _category(template: TemplateSpec) -> FailureCategory:
    return FailureCategory(template.reasoning_category)


def _variant(template: TemplateSpec, template_index: int, fixture_index: int) -> int:
    occurrence = template_index + fixture_index * len(build_template_bank())
    if template.reasoning_category == str(FailureCategory.MULTI_STEP_BOOKKEEPING):
        return occurrence
    relation = template.semantic_frame.split(".", 1)[0]
    if template.reasoning_category == str(FailureCategory.RATE_RATIO_PERCENTAGE):
        base = {
            "rate_total": 0,
            "ratio_scale": 1,
            "percentage": 2,
            "weighted_average": 3,
            "combined_rate": 4,
        }[relation]
        return base + 5 * occurrence
    base = {
        "two_type_allocation": 0,
        "complete_packages": 1,
        "equal_distribution": 2,
        "dual_capacity": 3,
    }[relation]
    return base + 4 * occurrence


def _plan(
    template: TemplateSpec,
    sentence_plan: SentencePlanSpec,
    template_index: int,
    fixture_index: int,
    attempt_index: int,
) -> AttemptPlan:
    return AttemptPlan(
        attempt_index=attempt_index,
        group=GroupName.GENERIC_CONTROL,
        group_index=attempt_index,
        category=_category(template),
        category_variant=_variant(template, template_index, fixture_index),
        difficulty=(DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD)[
            fixture_index % 3
        ],
        output_contract_enabled=fixture_index % 2 == 1,
        random_seed=_seed(template.template_id, sentence_plan.plan_id, fixture_index),
    )


def _write_sample(path: Path, rows: list[tuple[ExpansionRecord, str]]) -> None:
    lines = [
        "# Foundry template-bank static expansion: Codex inspection sample",
        "",
        "This is an ignored, synthetic-only inspection artifact. It is not human review.",
        "",
    ]
    for record, question in rows:
        lines.extend(
            (
                f"## {record.attempt_index:04d} - {record.template_id}/{record.sentence_plan_id}",
                "",
                f"- Category: `{record.category}`",
                f"- Difficulty: `{record.difficulty}`",
                "",
                question,
                "",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_static_expansion(repository_root: Path) -> dict[str, object]:
    """Render ten fixtures per plan without creating a training dataset."""

    bank = build_template_bank()
    records: list[ExpansionRecord] = []
    failure_rows: list[tuple[ExpansionRecord, str]] = []
    sample_candidates_by_category: dict[str, list[tuple[ExpansionRecord, str]]] = {
        str(FailureCategory.MULTI_STEP_BOOKKEEPING): [],
        str(FailureCategory.RATE_RATIO_PERCENTAGE): [],
        str(FailureCategory.CONSTRAINT_DISCRETE): [],
    }
    attempt_index = 0
    for template_index, template in enumerate(bank):
        for sentence_plan in template.sentence_plan_variants:
            for fixture_index in range(EXPANSION_FIXTURES_PER_PLAN):
                attempt_index += 1
                plan = _plan(
                    template,
                    sentence_plan,
                    template_index,
                    fixture_index,
                    attempt_index,
                )
                original = _generate(plan)
                rendered = render_with_template(original, template, sentence_plan)
                provenance = audit_surface_provenance(
                    rendered.problem_ir, rendered.realization, template
                )
                reasons = tuple(
                    dict.fromkeys(
                        validate_realization(
                            problem=rendered.problem_ir,
                            realization=rendered.realization,
                            answer=rendered.canonical_final_answer,
                        )
                        + validate_rendered_candidate(
                            question=rendered.rendered_question,
                            completion=rendered.training_completion,
                            answer=rendered.canonical_final_answer,
                            output_contract_enabled=rendered.output_contract_enabled,
                            metadata=rendered.quality_metadata,
                        )
                        + provenance.reasons
                    )
                )
                record = ExpansionRecord(
                    attempt_index=attempt_index,
                    template_id=template.template_id,
                    sentence_plan_id=sentence_plan.plan_id,
                    category=template.reasoning_category,
                    fixture_index=fixture_index,
                    difficulty=str(plan.difficulty),
                    output_contract_enabled=plan.output_contract_enabled,
                    rendered_text_sha256=normalized_text_sha256(rendered.rendered_question),
                    numeric_template_sha256=numeric_template_sha256(rendered.rendered_question),
                    render_signature_sha256=template.render_signature_hash(sentence_plan),
                    surface_provenance_sha256=provenance.provenance_sha256,
                    reasons=reasons,
                )
                records.append(record)
                if reasons:
                    failure_rows.append((record, rendered.rendered_question))
                if not reasons:
                    sample_candidates_by_category[template.reasoning_category].append(
                        (record, rendered.rendered_question)
                    )

    expected = 232 * EXPANSION_FIXTURES_PER_PLAN
    if len(records) != expected:
        raise AssertionError("static expansion did not account for every fixture")
    failures = Counter(reason for record in records for reason in record.reasons)
    rendered_hashes = [record.rendered_text_sha256 for record in records]
    numeric_hashes = [record.numeric_template_sha256 for record in records]
    signatures = {record.render_signature_sha256 for record in records}
    sentence_plan_signatures = [
        template.render_signature_hash(plan)
        for template in bank
        for plan in template.sentence_plan_variants
    ]
    number_neutral_plan_keys = [
        hashlib.sha256(
            json.dumps(
                {
                    "semantic_frame": template.semantic_frame,
                    "clause_order": plan.clause_order,
                    "opening_form": plan.opening_form,
                    "event_form": plan.event_form,
                    "question_form": plan.question_form,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for template in bank
        for plan in template.sentence_plan_variants
    ]
    sample_by_category = {
        category: tuple(rows[index * len(rows) // 30] for index in range(30))
        for category, rows in sample_candidates_by_category.items()
    }
    sample_rows = [row for category in sample_by_category.values() for row in category]
    if len(sample_rows) != 90:
        raise AssertionError("static inspection sample must contain 30 renders per family")
    sample_path = repository_root / "results/raw/template_bank_static_v3/codex_sample.md"
    _write_sample(sample_path, sample_rows)
    failure_path = repository_root / "results/raw/template_bank_static_v3/failures.md"
    _write_sample(failure_path, failure_rows)
    deterministic_records = [asdict(record) for record in records]
    aggregate_sha256 = hashlib.sha256(
        json.dumps(deterministic_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    inspection_path = (
        repository_root / "results/synthesis_smoke/template_bank_v3_static_inspection.json"
    )
    inspection_status = "pending"
    if inspection_path.exists():
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        if (
            inspection.get("expansion_aggregate_sha256") == aggregate_sha256
            and inspection.get("sample_size") == 90
            and inspection.get("invalid_or_unnatural_count") == 0
            and inspection.get("systematic_composition_defect") is False
        ):
            inspection_status = "complete_no_defects"
    summary: dict[str, object] = {
        "schema_version": 2,
        "expansion_version": EXPANSION_VERSION,
        "sentence_plans": 232,
        "fixtures_per_plan": EXPANSION_FIXTURES_PER_PLAN,
        "total_expansions_attempted": len(records),
        "valid_renders": sum(not record.reasons for record in records),
        "expected_incompatibility_rejections": 0,
        "failure_counts": dict(sorted(failures.items())),
        "duplicate_noun_detections": failures["adjacent_duplicate_noun"],
        "internal_label_detections": failures["internal_frame_label_leak"]
        + failures["internal_identifier_leak"],
        "ordinal_failures": failures["invalid_ordinal_morphology"],
        "morphology_failures": failures["morphology_or_agreement_failure"]
        + failures["unsupported_morphology"],
        "target_mismatches": sum(count for reason, count in failures.items() if "target" in reason),
        "semantic_coverage_failures": failures["semantic_node_realization_count"]
        + failures["semantic_node_coverage"],
        "exact_duplicate_expansions": len(rendered_hashes) - len(set(rendered_hashes)),
        "number_neutral_duplicate_expansions": len(numeric_hashes) - len(set(numeric_hashes)),
        "distinct_render_signatures": len(signatures),
        "exact_duplicate_sentence_plans": len(sentence_plan_signatures)
        - len(set(sentence_plan_signatures)),
        "number_neutral_duplicate_sentence_plans": len(number_neutral_plan_keys)
        - len(set(number_neutral_plan_keys)),
        "codex_inspection_sample_size": len(sample_rows),
        "codex_inspection_sample_by_category": {
            key: len(value) for key, value in sorted(sample_by_category.items())
        },
        "codex_inspection_status": inspection_status,
        "aggregate_sha256": aggregate_sha256,
        "raw_sample_path": sample_path.relative_to(repository_root).as_posix(),
        "raw_failure_path": failure_path.relative_to(repository_root).as_posix(),
        "training_dataset_created": False,
    }
    summary_path = (
        repository_root / "results/synthesis_smoke/template_bank_v3_static_expansion.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    summary = run_static_expansion(args.repository_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
