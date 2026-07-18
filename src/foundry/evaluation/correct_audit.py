"""Label-blind extraction evidence for auditing correct-scored responses."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from foundry.evaluation import answer_extraction as canonical
from foundry.evaluation.answer_extraction import serialize_canonical_number

AUDIT_SCHEMA_VERSION = 1
EXPECTED_BASELINE_RECORDS = 814
EXPECTED_CORRECT_SCORED = 521
EXPECTED_RAW_SHA256 = "73d52dace0f27577b1177bdfa81dfbb4c88252107c9b04e2ff49dbbd93da6cc0"
AuditClassificationName = Literal[
    "confirmed_intended_answer",
    "confirmed_false_acceptance",
    "ambiguous_requires_review",
]
_AUDIT_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        "confirmed_intended_answer",
        "confirmed_false_acceptance",
        "ambiguous_requires_review",
    }
)

_RULE_NAMES = (
    "literal_final_answer_line",
    "standalone_decorated_value",
    "standalone_plain_value",
    "explicit_answer_cue",
    "named_result_statement",
    "direct_conclusion_statement",
    "conclusion_verb_prose",
    "conclusion_decorated_value",
    "conclusion_leading_value",
    "conclusion_terminal_prose",
    "embedded_final_answer_line",
    "latex_final_answer_line",
    "terminal_assignment",
    "terminal_equation",
)
_NEGATIVE_INTENT = re.compile(
    r"\b(deficit|loss|lost|negative|short(?:fall)?|under(?:\s+budget)?)\b",
    re.IGNORECASE,
)


class CorrectAuditError(ValueError):
    """Raised when a label-blind audit view cannot be built safely."""


@dataclass(frozen=True)
class LabelBlindAuditView:
    """One correct-scored response with scoring labels deliberately omitted."""

    schema_version: int
    stable_id: str
    response: str
    strict_parser_accepted: bool
    canonical_extractor_accepted: bool
    extraction_rule: str
    extracted_value: int | str
    source_span_start: int
    source_span_end: int
    source_span_text: str
    rule_context: str
    competing_terminal_values: bool
    terminal_context_values: tuple[int | str, ...]
    output_complete: bool
    output_tokens: int | None
    suspicious_patterns: tuple[str, ...]


@dataclass(frozen=True)
class AuditViewSummary:
    """Content-free provenance and counts for a label-blind audit view."""

    schema_version: int
    raw_predictions_sha256: str
    audit_configuration_sha256: str
    label_blind_views_sha256: str
    audited_population: int
    strict_parser_accepted: int
    canonical_only: int
    extraction_rule_counts: dict[str, int]
    suspicious_pattern_counts: dict[str, int]


@dataclass(frozen=True)
class FrozenAuditClassification:
    """One label-blind intent decision, stored only in ignored raw results."""

    schema_version: int
    stable_id: str
    strict_parser_accepted: bool
    canonical_extractor_accepted: bool
    extraction_rule: str
    extracted_value: int | str
    competing_terminal_values: bool
    output_complete: bool
    output_tokens: int | None
    audit_classification: AuditClassificationName
    sanitized_rationale: str


@dataclass(frozen=True)
class FrozenClassificationSummary:
    """Content-free evidence that all label-blind decisions were frozen."""

    schema_version: int
    freeze_status: str
    audited_population: int
    classification_counts: dict[str, int]
    audit_configuration_sha256: str
    label_blind_views_sha256: str
    frozen_classifications_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonl_bytes(views: tuple[LabelBlindAuditView, ...]) -> bytes:
    rendered = "".join(
        json.dumps(asdict(view), sort_keys=True, separators=(",", ":")) + "\n" for view in views
    )
    return rendered.encode("utf-8")


def _classification_jsonl_bytes(
    classifications: tuple[FrozenAuditClassification, ...],
) -> bytes:
    rendered = "".join(
        json.dumps(asdict(classification), sort_keys=True, separators=(",", ":")) + "\n"
        for classification in classifications
    )
    return rendered.encode("utf-8")


def _read_jsonl_objects(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise CorrectAuditError(f"{path.name} line {line_number} is invalid JSON") from error
        if not isinstance(raw, dict):
            raise CorrectAuditError(f"{path.name} line {line_number} is not an object")
        records.append(cast(dict[str, object], raw))
    return tuple(records)


def audit_configuration_sha256() -> str:
    """Hash the immutable view construction and suspicion rules."""

    payload = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "canonical_extractor_id": canonical.CANONICAL_EXTRACTOR_ID,
        "canonical_extractor_sha256": canonical.canonical_extractor_sha256(),
        "correct_population_selection": "raw_correct_true_label_not_exported",
        "negative_intent_pattern": _NEGATIVE_INTENT.pattern,
        "rule_names": list(_RULE_NAMES),
        "suspicious_rules": {
            "canonical_only": True,
            "long_output_threshold": 700,
            "loose_conclusion_rule_indices": [5, 6, 7, 8, 9],
            "multiple_terminal_context_values": True,
            "negative_intent_without_negative_value": True,
            "percentage_currency_mixture": True,
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(serialized.encode("utf-8"))


def _captured_group(match: re.Match[str]) -> tuple[str, tuple[int, int]]:
    for name in ("boxed", "bold", "latex", "plain"):
        value = match.groupdict().get(name)
        if value is not None:
            return value, match.span(name)
    raise CorrectAuditError("terminal extraction rule did not capture a numeric source span")


def _context_values(context: str) -> tuple[int | str, ...]:
    values: set[int | str] = set()
    for match in canonical._NUMBER_TOKEN.finditer(context):
        try:
            value = canonical._normalize_value(match.group())
        except canonical.CanonicalExtractionError:
            continue
        values.add(serialize_canonical_number(value))
    return tuple(sorted(values, key=str))


def _trace_extraction(
    response: str, extracted_value: int | str
) -> tuple[
    str,
    int,
    int,
    str,
    str,
    tuple[int | str, ...],
]:
    stripped = response.strip()
    offset = response.find(stripped)
    expected = str(extracted_value)
    matches: list[tuple[int, re.Match[str], str, tuple[int, int], tuple[int | str, ...]]] = []
    for index, pattern in enumerate(canonical._TERMINAL_PATTERNS):
        match = pattern.search(stripped)
        if match is None:
            continue
        raw_value, relative_span = _captured_group(match)
        try:
            normalized = serialize_canonical_number(canonical._normalize_value(raw_value))
        except canonical.CanonicalExtractionError:
            continue
        if str(normalized) != expected:
            continue
        matches.append((index, match, raw_value, relative_span, _context_values(match.group())))
    if not matches:
        raise CorrectAuditError("canonical answer could not be traced to a terminal rule")
    index, match, raw_value, relative_span, context_values = matches[0]
    start = offset + relative_span[0]
    end = offset + relative_span[1]
    return (
        _RULE_NAMES[index],
        start,
        end,
        raw_value,
        match.group().strip(),
        context_values,
    )


def _suspicious_patterns(
    *,
    strict_parser_accepted: bool,
    rule_name: str,
    extracted_value: int | str,
    rule_context: str,
    context_values: tuple[int | str, ...],
    output_tokens: int | None,
) -> tuple[str, ...]:
    patterns: list[str] = []
    if not strict_parser_accepted:
        patterns.append("canonical_only")
    if rule_name in {_RULE_NAMES[index] for index in (5, 6, 7, 8, 9)}:
        patterns.append("loose_conclusion_rule")
    if len(context_values) > 1:
        patterns.append("multiple_values_in_terminal_context")
    if "%" in rule_context and "$" in rule_context and len(context_values) > 1:
        patterns.append("percentage_currency_mixture")
    numeric_value = str(extracted_value)
    if not numeric_value.startswith("-") and _NEGATIVE_INTENT.search(rule_context):
        patterns.append("negative_intent_without_negative_value")
    if output_tokens is not None and output_tokens >= 700:
        patterns.append("long_output_near_limit")
    return tuple(patterns)


def _parse_raw_records(path: Path) -> tuple[dict[str, object], ...]:
    raw_bytes = path.read_bytes()
    actual_hash = _sha256_bytes(raw_bytes)
    if actual_hash != EXPECTED_RAW_SHA256:
        raise CorrectAuditError(
            f"raw prediction SHA-256 differs: expected {EXPECTED_RAW_SHA256}, got {actual_hash}"
        )
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise CorrectAuditError(f"raw line {line_number} is invalid JSON") from error
        if not isinstance(raw, dict):
            raise CorrectAuditError(f"raw line {line_number} is not an object")
        records.append(cast(dict[str, object], raw))
    if len(records) != EXPECTED_BASELINE_RECORDS:
        raise CorrectAuditError(
            f"expected {EXPECTED_BASELINE_RECORDS} raw records, found {len(records)}"
        )
    return tuple(records)


def build_label_blind_views(path: Path) -> tuple[LabelBlindAuditView, ...]:
    """Build correct-scored audit views while omitting questions and reference labels."""

    views: list[LabelBlindAuditView] = []
    for record in _parse_raw_records(path):
        if record.get("correct") is not True:
            continue
        stable_id = record.get("stable_id")
        response = record.get("response")
        extracted_value = record.get("predicted_answer")
        strict_parser_accepted = record.get("exact_format_compliant")
        generation_truncated = record.get("generation_truncated")
        output_tokens = record.get("output_tokens")
        if not isinstance(stable_id, str) or len(stable_id) != 64:
            raise CorrectAuditError("correct-scored record has an invalid stable identifier")
        if not isinstance(response, str):
            raise CorrectAuditError(f"correct-scored record {stable_id} has no response")
        if isinstance(extracted_value, bool) or not isinstance(extracted_value, int | str):
            raise CorrectAuditError(f"correct-scored record {stable_id} has no extracted value")
        if not isinstance(strict_parser_accepted, bool):
            raise CorrectAuditError(f"correct-scored record {stable_id} has invalid strict status")
        if generation_truncated is not False:
            raise CorrectAuditError(f"correct-scored record {stable_id} is unexpectedly truncated")
        if output_tokens is not None and (
            isinstance(output_tokens, bool) or not isinstance(output_tokens, int)
        ):
            raise CorrectAuditError(f"correct-scored record {stable_id} has invalid token metadata")
        recomputed = serialize_canonical_number(canonical.extract_canonical_number(response))
        if recomputed != extracted_value:
            raise CorrectAuditError(f"correct-scored record {stable_id} extraction changed")
        rule, start, end, span_text, context, context_values = _trace_extraction(
            response,
            extracted_value,
        )
        suspicious = _suspicious_patterns(
            strict_parser_accepted=strict_parser_accepted,
            rule_name=rule,
            extracted_value=extracted_value,
            rule_context=context,
            context_values=context_values,
            output_tokens=output_tokens,
        )
        views.append(
            LabelBlindAuditView(
                schema_version=AUDIT_SCHEMA_VERSION,
                stable_id=stable_id,
                response=response,
                strict_parser_accepted=strict_parser_accepted,
                canonical_extractor_accepted=True,
                extraction_rule=rule,
                extracted_value=extracted_value,
                source_span_start=start,
                source_span_end=end,
                source_span_text=span_text,
                rule_context=context,
                competing_terminal_values=len(context_values) > 1,
                terminal_context_values=context_values,
                output_complete=True,
                output_tokens=output_tokens,
                suspicious_patterns=suspicious,
            )
        )
    if len(views) != EXPECTED_CORRECT_SCORED:
        raise CorrectAuditError(
            f"expected {EXPECTED_CORRECT_SCORED} correct-scored views, found {len(views)}"
        )
    if len({view.stable_id for view in views}) != len(views):
        raise CorrectAuditError("correct-scored audit views contain duplicate identifiers")
    return tuple(views)


def save_label_blind_views(
    views: tuple[LabelBlindAuditView, ...],
    *,
    raw_path: Path,
    output_dir: Path,
) -> AuditViewSummary:
    """Save detailed views under an ignored path and return content-free provenance."""

    if len(views) != EXPECTED_CORRECT_SCORED:
        raise CorrectAuditError("label-blind view set is incomplete")
    rendered = _jsonl_bytes(views)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_blind_views.jsonl").write_bytes(rendered)
    rule_counts = Counter(view.extraction_rule for view in views)
    pattern_counts = Counter(pattern for view in views for pattern in view.suspicious_patterns)
    summary = AuditViewSummary(
        schema_version=AUDIT_SCHEMA_VERSION,
        raw_predictions_sha256=_sha256_bytes(raw_path.read_bytes()),
        audit_configuration_sha256=audit_configuration_sha256(),
        label_blind_views_sha256=_sha256_bytes(rendered),
        audited_population=len(views),
        strict_parser_accepted=sum(view.strict_parser_accepted for view in views),
        canonical_only=sum(not view.strict_parser_accepted for view in views),
        extraction_rule_counts=dict(sorted(rule_counts.items())),
        suspicious_pattern_counts=dict(sorted(pattern_counts.items())),
    )
    (output_dir / "view_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _normalize_classifications(
    views: tuple[LabelBlindAuditView, ...],
    working_records: tuple[dict[str, object], ...],
) -> tuple[FrozenAuditClassification, ...]:
    if len(working_records) != len(views):
        raise CorrectAuditError(
            f"classification count differs from label-blind views: "
            f"{len(working_records)} != {len(views)}"
        )
    normalized: list[FrozenAuditClassification] = []
    for index, (view, record) in enumerate(zip(views, working_records, strict=True), start=1):
        stable_id = record.get("stable_id")
        classification = record.get("audit_classification")
        rationale = record.get("sanitized_rationale")
        if stable_id != view.stable_id:
            raise CorrectAuditError(f"classification {index} does not match label-blind view order")
        if not isinstance(classification, str) or classification not in _AUDIT_CLASSIFICATIONS:
            raise CorrectAuditError(f"classification {index} has an invalid audit decision")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 300:
            raise CorrectAuditError(f"classification {index} needs a short sanitized rationale")
        evidence_fields = {
            "strict_parser_accepted": view.strict_parser_accepted,
            "canonical_extractor_accepted": view.canonical_extractor_accepted,
            "extraction_rule": view.extraction_rule,
            "extracted_value": view.extracted_value,
            "competing_terminal_values": view.competing_terminal_values,
            "output_complete": view.output_complete,
            "output_tokens": view.output_tokens,
        }
        for field, expected in evidence_fields.items():
            if record.get(field) != expected:
                raise CorrectAuditError(f"classification {index} changed evidence field {field}")
        normalized.append(
            FrozenAuditClassification(
                schema_version=AUDIT_SCHEMA_VERSION,
                stable_id=view.stable_id,
                strict_parser_accepted=view.strict_parser_accepted,
                canonical_extractor_accepted=view.canonical_extractor_accepted,
                extraction_rule=view.extraction_rule,
                extracted_value=view.extracted_value,
                competing_terminal_values=view.competing_terminal_values,
                output_complete=view.output_complete,
                output_tokens=view.output_tokens,
                audit_classification=cast(AuditClassificationName, classification),
                sanitized_rationale=rationale.strip(),
            )
        )
    if len({record.stable_id for record in normalized}) != len(normalized):
        raise CorrectAuditError("frozen classifications contain duplicate identifiers")
    return tuple(normalized)


def freeze_label_blind_classifications(
    views: tuple[LabelBlindAuditView, ...],
    *,
    working_path: Path,
    output_dir: Path,
) -> FrozenClassificationSummary:
    """Validate and freeze every intent decision before score metadata is consulted."""

    if len(views) != EXPECTED_CORRECT_SCORED:
        raise CorrectAuditError("cannot freeze an incomplete correct-response audit")
    working_records = _read_jsonl_objects(working_path)
    classifications = _normalize_classifications(views, working_records)
    rendered = _classification_jsonl_bytes(classifications)
    counts = Counter(record.audit_classification for record in classifications)
    if sum(counts.values()) != EXPECTED_CORRECT_SCORED:
        raise CorrectAuditError("frozen classification counts are internally inconsistent")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classifications.jsonl").write_bytes(rendered)
    summary = FrozenClassificationSummary(
        schema_version=AUDIT_SCHEMA_VERSION,
        freeze_status="frozen_before_score_join",
        audited_population=len(classifications),
        classification_counts={
            "ambiguous_requires_review": counts["ambiguous_requires_review"],
            "confirmed_false_acceptance": counts["confirmed_false_acceptance"],
            "confirmed_intended_answer": counts["confirmed_intended_answer"],
        },
        audit_configuration_sha256=audit_configuration_sha256(),
        label_blind_views_sha256=_sha256_bytes(_jsonl_bytes(views)),
        frozen_classifications_sha256=_sha256_bytes(rendered),
    )
    (output_dir / "freeze_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_frozen_classifications(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[FrozenAuditClassification, ...]:
    """Load classifications only when their frozen byte hash still matches."""

    raw_bytes = path.read_bytes()
    actual_sha256 = _sha256_bytes(raw_bytes)
    if actual_sha256 != expected_sha256:
        raise CorrectAuditError(
            f"frozen classification SHA-256 differs: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    records = _read_jsonl_objects(path)
    classifications: list[FrozenAuditClassification] = []
    for index, record in enumerate(records, start=1):
        schema_version = record.get("schema_version")
        stable_id = record.get("stable_id")
        strict_accepted = record.get("strict_parser_accepted")
        canonical_accepted = record.get("canonical_extractor_accepted")
        extraction_rule = record.get("extraction_rule")
        extracted_value = record.get("extracted_value")
        competing_values = record.get("competing_terminal_values")
        output_complete = record.get("output_complete")
        output_tokens = record.get("output_tokens")
        decision = record.get("audit_classification")
        rationale = record.get("sanitized_rationale")
        if schema_version != AUDIT_SCHEMA_VERSION or not isinstance(stable_id, str):
            raise CorrectAuditError(f"frozen classification {index} has an invalid identity")
        if not isinstance(strict_accepted, bool) or not isinstance(canonical_accepted, bool):
            raise CorrectAuditError(f"frozen classification {index} has invalid parser status")
        if not isinstance(extraction_rule, str):
            raise CorrectAuditError(f"frozen classification {index} has an invalid rule")
        if isinstance(extracted_value, bool) or not isinstance(extracted_value, int | str):
            raise CorrectAuditError(f"frozen classification {index} has an invalid value")
        if not isinstance(competing_values, bool) or not isinstance(output_complete, bool):
            raise CorrectAuditError(
                f"frozen classification {index} has invalid completion evidence"
            )
        if output_tokens is not None and (
            isinstance(output_tokens, bool) or not isinstance(output_tokens, int)
        ):
            raise CorrectAuditError(f"frozen classification {index} has invalid token metadata")
        if not isinstance(decision, str) or decision not in _AUDIT_CLASSIFICATIONS:
            raise CorrectAuditError(f"frozen classification {index} has an invalid decision")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 300:
            raise CorrectAuditError(f"frozen classification {index} has an invalid rationale")
        classifications.append(
            FrozenAuditClassification(
                schema_version=AUDIT_SCHEMA_VERSION,
                stable_id=stable_id,
                strict_parser_accepted=strict_accepted,
                canonical_extractor_accepted=canonical_accepted,
                extraction_rule=extraction_rule,
                extracted_value=extracted_value,
                competing_terminal_values=competing_values,
                output_complete=output_complete,
                output_tokens=output_tokens,
                audit_classification=cast(AuditClassificationName, decision),
                sanitized_rationale=rationale,
            )
        )
    if len(classifications) != EXPECTED_CORRECT_SCORED:
        raise CorrectAuditError("frozen classification set is incomplete")
    if len({record.stable_id for record in classifications}) != len(classifications):
        raise CorrectAuditError("frozen classifications contain duplicate identifiers")
    return tuple(classifications)


def build_content_free_audit_summary(
    views: tuple[LabelBlindAuditView, ...],
    classifications: tuple[FrozenAuditClassification, ...],
    *,
    raw_path: Path,
    baseline_summary_path: Path,
    frozen_classifications_sha256: str,
) -> dict[str, object]:
    """Join frozen decisions with scores and return aggregate-only audit evidence."""

    if len(views) != EXPECTED_CORRECT_SCORED or len(classifications) != len(views):
        raise CorrectAuditError("score join requires all 521 frozen classifications")
    classification_by_id = {record.stable_id: record for record in classifications}
    if set(classification_by_id) != {view.stable_id for view in views}:
        raise CorrectAuditError("frozen classifications do not match the label-blind population")
    correct_ids = {
        cast(str, record["stable_id"])
        for record in _parse_raw_records(raw_path)
        if record.get("correct") is True
    }
    if correct_ids != set(classification_by_id):
        raise CorrectAuditError("frozen classifications do not match correct-scored records")
    try:
        baseline_raw: Any = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CorrectAuditError("baseline summary is invalid JSON") from error
    if not isinstance(baseline_raw, dict):
        raise CorrectAuditError("baseline summary is not an object")
    baseline = cast(dict[str, object], baseline_raw)
    if baseline.get("processed_examples") != EXPECTED_BASELINE_RECORDS:
        raise CorrectAuditError("baseline population differs from the frozen 814 examples")
    if baseline.get("correct_examples") != EXPECTED_CORRECT_SCORED:
        raise CorrectAuditError("baseline correct count differs from the frozen 521 examples")

    classification_counts = Counter(record.audit_classification for record in classifications)
    intended = classification_counts["confirmed_intended_answer"]
    false_acceptances = classification_counts["confirmed_false_acceptance"]
    ambiguous = classification_counts["ambiguous_requires_review"]
    strict_correct = sum(view.strict_parser_accepted for view in views)
    canonical_only_correct = len(views) - strict_correct
    lower_bound = intended / EXPECTED_BASELINE_RECORDS
    upper_bound = (intended + ambiguous) / EXPECTED_BASELINE_RECORDS
    adjusted_accuracy: float | None = lower_bound if ambiguous == 0 else None
    if false_acceptances == 0 and ambiguous == 0:
        decision = "BASELINE TRUSTED"
    elif ambiguous == 0:
        decision = "BASELINE USABLE WITH AUDITED ADJUSTMENT"
    else:
        decision = "BASELINE EVALUATOR REQUIRES RECONSIDERATION"
    rule_counts = Counter(view.extraction_rule for view in views)
    pattern_counts = Counter(pattern for view in views for pattern in view.suspicious_patterns)
    false_ids = {
        record.stable_id
        for record in classifications
        if record.audit_classification == "confirmed_false_acceptance"
    }
    percentage_views = {
        view.stable_id
        for view in views
        if "percentage_currency_mixture" in view.suspicious_patterns
    }
    sign_views = {
        view.stable_id
        for view in views
        if "negative_intent_without_negative_value" in view.suspicious_patterns
    }
    return {
        "schema_version": 1,
        "decision": decision,
        "counts": {
            "development_population": EXPECTED_BASELINE_RECORDS,
            "total_audited": len(classifications),
            "strict_parser_correct": strict_correct,
            "canonical_only_correct": canonical_only_correct,
            "confirmed_intended_correct": intended,
            "confirmed_false_acceptance": false_acceptances,
            "ambiguous_requires_review": ambiguous,
        },
        "accuracy": {
            "frozen_evaluator_correct": EXPECTED_CORRECT_SCORED,
            "frozen_evaluator_accuracy": EXPECTED_CORRECT_SCORED / EXPECTED_BASELINE_RECORDS,
            "confirmed_false_positive_rate": false_acceptances / len(classifications),
            "audited_lower_bound_accuracy": lower_bound,
            "audited_upper_bound_accuracy": upper_bound,
            "adjusted_exact_accuracy": adjusted_accuracy,
        },
        "extraction_rule_counts": dict(sorted(rule_counts.items())),
        "suspicious_pattern_counts": dict(sorted(pattern_counts.items())),
        "previous_false_acceptance_pattern_review": {
            "percentage_currency_terminal_collision": {
                "correct_scored_occurrences": len(percentage_views),
                "confirmed_false_acceptances": len(percentage_views & false_ids),
            },
            "unsigned_magnitude_in_negative_intent_language": {
                "correct_scored_occurrences": len(sign_views),
                "confirmed_false_acceptances": len(sign_views & false_ids),
            },
        },
        "failure_taxonomy_assessment": {
            "extractable_but_incorrect_population": baseline.get("extractable_incorrect_examples"),
            "unextractable_population": baseline.get("unextractable_examples"),
            "remains_useful": decision != "BASELINE EVALUATOR REQUIRES RECONSIDERATION",
            "scope": "provisional development-only taxonomy with overlapping secondary causes",
        },
        "provenance": {
            "raw_predictions_sha256": _sha256_bytes(raw_path.read_bytes()),
            "audit_configuration_sha256": audit_configuration_sha256(),
            "label_blind_views_sha256": _sha256_bytes(_jsonl_bytes(views)),
            "frozen_classifications_sha256": frozen_classifications_sha256,
            "model_id": baseline.get("model_id"),
            "model_revision": baseline.get("model_revision"),
            "dataset_id": baseline.get("dataset_id"),
            "dataset_revision": baseline.get("dataset_revision"),
            "prompt_sha256": baseline.get("prompt_sha256"),
            "canonical_extractor_id": baseline.get("canonical_extractor_id"),
            "canonical_extractor_sha256": baseline.get("canonical_extractor_sha256"),
            "config_sha256": baseline.get("config_sha256"),
            "manifest_sha256": baseline.get("manifest_sha256"),
        },
        "limitations": [
            "Intent was audited from completions and extraction evidence, not by independently "
            "solving benchmark questions.",
            "The audit establishes precision only for the 521 development responses scored "
            "correct by the frozen evaluator.",
            "The mathematical failure taxonomy remains provisional and is not a claim about "
            "sealed-final performance.",
        ],
    }
