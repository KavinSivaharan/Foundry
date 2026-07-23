from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast
from urllib.parse import urlsplit

import torch

from foundry.phase2.asdiv import canonical_sha256, file_sha256
from foundry.synthesis.contamination import (
    canonical_number_neutral_identity,
    load_development_questions_for_contamination,
    normalize_text,
    normalized_text_sha256,
)
from foundry.synthesis.semantic import (
    PinnedSentenceEncoder,
    load_semantic_artifact_config,
    verify_local_artifact,
)

DEVELOPMENT_SEMANTIC_REJECT: Final = 0.75
PHASE1_SYNTHETIC_SEMANTIC_REJECT: Final = 0.82
CONTIGUOUS_TOKEN_COUNT: Final = 12

_OPERATION_CUES: Final = {
    "addition": frozenset({"altogether", "combined", "sum", "total"}),
    "subtraction": frozenset({"difference", "fewer", "left", "less", "remain", "remaining"}),
    "multiplication": frozenset({"each", "every", "groups", "times"}),
    "division": frozenset({"equally", "per", "share", "shared"}),
    "rate": frozenset({"average", "percent", "percentage", "rate", "ratio", "speed"}),
}
_FORMULA_OPERATION_CATEGORIES: Final = {
    "+": "addition",
    "add": "addition",
    "-": "subtraction",
    "subtract": "subtraction",
    "*": "multiplication",
    "multiply": "multiplication",
    "/": "division",
    "divide": "division",
    "inverse": "division",
    "percent": "rate",
    "speed": "rate",
    "speed_in_still_water": "rate",
    "stream_speed": "rate",
}


@dataclass(frozen=True)
class ReferenceText:
    stable_id: str
    text: str
    normalized_sha256: str
    number_neutral_sha256: str
    ngrams: frozenset[tuple[str, ...]]
    operation_cues: tuple[str, ...]
    structure_sha256: str


@dataclass(frozen=True)
class SyntheticReference:
    stable_id: str
    text: str
    latent_program_sha256: str


def _string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def _integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"field {key!r} must be an integer")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: object = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(cast(dict[str, object], raw))
    return rows


def contiguous_ngrams(text: str, size: int = CONTIGUOUS_TOKEN_COUNT) -> frozenset[tuple[str, ...]]:
    if size <= 0:
        raise ValueError("n-gram size must be positive")
    tokens = normalize_text(text, replace_numbers=False).split()
    if len(tokens) < size:
        return frozenset()
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def operation_cues(text: str) -> tuple[str, ...]:
    tokens = set(normalize_text(text, replace_numbers=False).split())
    return tuple(
        category for category, cues in sorted(_OPERATION_CUES.items()) if tokens & set(cues)
    )


def _structure_sha256(text: str) -> str:
    payload = {
        "number_neutral_sha256": canonical_number_neutral_identity(text).sha256,
        "operation_cues": operation_cues(text),
    }
    return canonical_sha256(payload)


def _reference(stable_id: str, text: str) -> ReferenceText:
    return ReferenceText(
        stable_id=stable_id,
        text=text,
        normalized_sha256=normalized_text_sha256(text),
        number_neutral_sha256=canonical_number_neutral_identity(text).sha256,
        ngrams=contiguous_ngrams(text),
        operation_cues=operation_cues(text),
        structure_sha256=_structure_sha256(text),
    )


def _load_synthetic(paths: Sequence[Path]) -> tuple[SyntheticReference, ...]:
    references: list[SyntheticReference] = []
    for path in paths:
        for row in _load_jsonl(path):
            references.append(
                SyntheticReference(
                    stable_id=_string(row, "synthetic_id"),
                    text=_string(row, "rendered_question"),
                    latent_program_sha256=_string(row, "latent_program_sha256"),
                )
            )
    if len(references) != 1000:
        raise ValueError(f"expected 1000 Phase 1 synthetic questions, found {len(references)}")
    if len({item.stable_id for item in references}) != len(references):
        raise ValueError("Phase 1 synthetic IDs are not unique")
    return tuple(references)


def _source_fingerprint(source_url: str) -> str:
    parsed = urlsplit(source_url)
    host = parsed.hostname or ""
    path = parsed.path.strip("/")
    return normalize_text(f"{host} {path}", replace_numbers=True)


def _formula_categories(row: dict[str, object]) -> tuple[str, ...]:
    raw = row.get("operation_sequence")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("operation_sequence must be a string list")
    categories = {
        category
        for item in cast(list[str], raw)
        if (category := _FORMULA_OPERATION_CATEGORIES.get(item)) is not None
    }
    return tuple(sorted(categories))


def _duplicate_reasons(rows: Sequence[dict[str, object]]) -> dict[str, str]:
    by_exact: defaultdict[str, list[str]] = defaultdict(list)
    by_normalized: defaultdict[str, list[str]] = defaultdict(list)
    by_formula_template: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    source_ids: list[str] = []
    for row in rows:
        source_id = _string(row, "source_id")
        text = _string(row, "combined_question")
        source_ids.append(source_id)
        by_exact[hashlib.sha256(text.encode("utf-8")).hexdigest()].append(source_id)
        normalized = normalized_text_sha256(text)
        by_normalized[normalized].append(source_id)
        by_formula_template[
            (_string(row, "program_sha256"), canonical_number_neutral_identity(text).sha256)
        ].append(source_id)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("supported candidate source IDs are not unique")

    reasons: dict[str, str] = {}
    for label, groups in (
        ("candidate_exact_duplicate", by_exact),
        ("candidate_normalized_duplicate", by_normalized),
        ("candidate_formula_text_duplicate", by_formula_template),
    ):
        for group in groups.values():
            if len(group) <= 1:
                continue
            for source_id in sorted(group)[1:]:
                reasons.setdefault(source_id, label)
    return reasons


def _cross_source_duplicate_reasons(
    rows: Sequence[dict[str, object]], cross_source_rows: Sequence[dict[str, object]]
) -> dict[str, str]:
    if not cross_source_rows:
        return {}
    exact = {
        hashlib.sha256(_string(row, "combined_question").encode("utf-8")).hexdigest()
        for row in cross_source_rows
    }
    normalized = {
        normalized_text_sha256(_string(row, "combined_question")) for row in cross_source_rows
    }
    formula_text = {
        (
            _string(row, "program_structure_sha256"),
            canonical_number_neutral_identity(_string(row, "combined_question")).sha256,
        )
        for row in cross_source_rows
    }
    reasons: dict[str, str] = {}
    for row in rows:
        source_id = _string(row, "source_id")
        text = _string(row, "combined_question")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() in exact:
            reasons[source_id] = "cross_source_exact_duplicate"
        elif normalized_text_sha256(text) in normalized:
            reasons[source_id] = "cross_source_normalized_duplicate"
        elif (
            _string(row, "program_structure_sha256"),
            canonical_number_neutral_identity(text).sha256,
        ) in formula_text:
            reasons[source_id] = "cross_source_formula_text_duplicate"
    return reasons


def _closest(
    encoder: PinnedSentenceEncoder,
    candidate_texts: Sequence[str],
    reference_texts: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates = encoder.encode(candidate_texts)
    references = encoder.encode(reference_texts)
    similarities = encoder.cosine_matrix(candidates, references)
    maxima, indices = similarities.max(dim=1)
    return candidates, maxima, indices


def _json_line(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def _write_lines(path: Path, rows: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = _json_line(row)
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _float_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
    return {
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


def screen_contamination(
    *,
    supported_path: Path,
    evaluation_config_path: Path,
    development_manifest_path: Path,
    synthetic_paths: Sequence[Path],
    semantic_config_path: Path,
    repository_root: Path,
    output_dir: Path,
    cross_source_paths: Sequence[Path] = (),
    output_filename: str = "clean_asdiv.jsonl",
    minimum_candidates: int = 1000,
) -> dict[str, object]:
    candidates = _load_jsonl(supported_path)
    if len(candidates) < minimum_candidates:
        raise ValueError("supported candidate input is below the configured verified gate")
    if Path(output_filename).name != output_filename or not output_filename.endswith(".jsonl"):
        raise ValueError("output filename must be a plain JSONL filename")
    development = load_development_questions_for_contamination(
        evaluation_config_path=evaluation_config_path,
        development_manifest_path=development_manifest_path,
    )
    synthetic = _load_synthetic(synthetic_paths)
    development_references = tuple(
        _reference(item.stable_id, item.question) for item in development
    )
    synthetic_references = tuple(_reference(item.stable_id, item.text) for item in synthetic)

    development_exact = {item.normalized_sha256 for item in development_references}
    development_templates = {item.number_neutral_sha256 for item in development_references}
    development_ngrams = frozenset(
        ngram for reference in development_references for ngram in reference.ngrams
    )
    development_structures = {item.structure_sha256 for item in development_references}
    synthetic_exact = {item.normalized_sha256 for item in synthetic_references}
    synthetic_templates = {item.number_neutral_sha256 for item in synthetic_references}
    synthetic_ngrams = frozenset(
        ngram for reference in synthetic_references for ngram in reference.ngrams
    )
    synthetic_latent_hashes = {item.latent_program_sha256 for item in synthetic}
    normalized_reference_text = " ".join(
        item.text for item in (*development_references, *synthetic_references)
    ).casefold()
    duplicate_reasons = _duplicate_reasons(candidates)
    cross_source_rows = [row for path in cross_source_paths for row in _load_jsonl(path)]
    cross_source_reasons = _cross_source_duplicate_reasons(candidates, cross_source_rows)

    semantic_config = load_semantic_artifact_config(semantic_config_path)
    artifact_evidence = verify_local_artifact(semantic_config, repository_root)
    encoder = PinnedSentenceEncoder(semantic_config, repository_root)
    candidate_texts = [_string(row, "combined_question") for row in candidates]
    candidate_embeddings, development_maxima, development_indices = _closest(
        encoder,
        candidate_texts,
        [item.question for item in development],
    )
    _, synthetic_maxima, synthetic_indices = _closest(
        encoder,
        candidate_texts,
        [item.text for item in synthetic],
    )
    replay_size = min(30, len(candidate_texts))
    replay = encoder.encode(candidate_texts[:replay_size])
    if not torch.equal(candidate_embeddings[:replay_size], replay):
        raise RuntimeError("fixed semantic replay sample is not deterministic")

    development_scores = cast(list[float], development_maxima.tolist())
    development_closest = cast(list[int], development_indices.tolist())
    synthetic_scores = cast(list[float], synthetic_maxima.tolist())
    synthetic_closest = cast(list[int], synthetic_indices.tolist())
    evidence_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    for index, row in enumerate(candidates):
        source_id = _string(row, "source_id")
        text = _string(row, "combined_question")
        normalized_hash = normalized_text_sha256(text)
        template_hash = canonical_number_neutral_identity(text).sha256
        ngrams = contiguous_ngrams(text)
        structure_hash = _structure_sha256(text)
        source_fingerprint = _source_fingerprint(_string(row, "source_url"))

        checks = {
            "development_exact": normalized_hash in development_exact,
            "development_twelve_token": bool(ngrams & development_ngrams),
            "development_number_neutral": template_hash in development_templates,
            "development_operation_structure": structure_hash in development_structures,
            "phase1_exact": normalized_hash in synthetic_exact,
            "phase1_twelve_token": bool(ngrams & synthetic_ngrams),
            "phase1_number_neutral": template_hash in synthetic_templates,
            "phase1_program_structure": _string(row, "program_structure_sha256")
            in synthetic_latent_hashes,
            "source_reference": bool(source_fingerprint)
            and source_fingerprint in normalized_reference_text,
        }
        for key, matched in checks.items():
            if matched:
                metric_counts[key] += 1

        reason: str | None = None
        for key in (
            "development_exact",
            "development_twelve_token",
            "development_number_neutral",
            "development_operation_structure",
            "phase1_exact",
            "phase1_twelve_token",
            "phase1_number_neutral",
            "phase1_program_structure",
            "source_reference",
        ):
            if checks[key]:
                reason = key
                break
        reason = reason or duplicate_reasons.get(source_id)
        reason = reason or cross_source_reasons.get(source_id)
        if reason is None and development_scores[index] >= DEVELOPMENT_SEMANTIC_REJECT:
            reason = "development_semantic"
        if reason is None and synthetic_scores[index] >= PHASE1_SYNTHETIC_SEMANTIC_REJECT:
            reason = "phase1_semantic"

        evidence: dict[str, object] = {
            "source_id": source_id,
            "question_sha256": _string(row, "question_sha256"),
            "normalized_sha256": normalized_hash,
            "number_neutral_sha256": template_hash,
            "operation_structure_sha256": structure_hash,
            "formula_categories": _formula_categories(row),
            "development_closest_id": development[development_closest[index]].stable_id,
            "development_semantic_maximum": development_scores[index],
            "phase1_closest_id": synthetic[synthetic_closest[index]].stable_id,
            "phase1_semantic_maximum": synthetic_scores[index],
            "checks": checks,
            "decision": "reject" if reason is not None else "pass",
            "rejection_reason": reason,
        }
        evidence_rows.append(evidence)
        if reason is not None:
            rejection_counts[reason] += 1
            continue
        clean = dict(row)
        clean["contamination"] = evidence
        clean_rows.append(clean)
        family_counts[_string(row, "family")] += 1

    evidence_rows.sort(key=lambda row: str(row["source_id"]))
    clean_rows.sort(key=lambda row: str(row["source_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_hash = _write_lines(output_dir / "contamination_evidence.jsonl", evidence_rows)
    clean_hash = _write_lines(output_dir / output_filename, clean_rows)
    source_hashes = {str(path): file_sha256(path) for path in synthetic_paths}
    cross_source_hashes = {str(path): file_sha256(path) for path in cross_source_paths}
    screening_config = {
        "development_semantic_reject_at_or_above": DEVELOPMENT_SEMANTIC_REJECT,
        "phase1_synthetic_semantic_reject_at_or_above": PHASE1_SYNTHETIC_SEMANTIC_REJECT,
        "contiguous_token_count": CONTIGUOUS_TOKEN_COUNT,
        "manual_review_band": None,
        "semantic_artifact_id": semantic_config.artifact_id,
        "semantic_revision": semantic_config.revision,
        "number_neutral_contract": "foundry-number-neutral-v1",
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "input": {
            "supported_path_sha256": file_sha256(supported_path),
            "candidate_count": len(candidates),
            "development_count": len(development),
            "development_manifest_sha256": file_sha256(development_manifest_path),
            "phase1_synthetic_count": len(synthetic),
            "phase1_synthetic_file_sha256": source_hashes,
            "cross_source_count": len(cross_source_rows),
            "cross_source_file_sha256": cross_source_hashes,
        },
        "screening_config": screening_config,
        "screening_config_sha256": canonical_sha256(screening_config),
        "semantic_artifact_files": [asdict(item) for item in artifact_evidence],
        "semantic_replay_size": replay_size,
        "semantic_replay_exact": True,
        "metric_match_counts": dict(sorted(metric_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejected_count": len(candidates) - len(clean_rows),
        "clean_count": len(clean_rows),
        "clean_family_counts": dict(sorted(family_counts.items())),
        "unresolved_semantic_candidates": 0,
        "development_similarity": _float_summary(development_scores),
        "phase1_similarity": _float_summary(synthetic_scores),
        "evidence_sha256": evidence_hash,
        "clean_rows_sha256": clean_hash,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_dir / "contamination_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen verified ASDiv candidates")
    parser.add_argument("--supported", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--phase1-synthetic", type=Path, action="append", required=True)
    parser.add_argument("--semantic-config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cross-source", type=Path, action="append", default=[])
    parser.add_argument("--output-filename", default="clean_asdiv.jsonl")
    parser.add_argument("--minimum-candidates", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = screen_contamination(
        supported_path=args.supported,
        evaluation_config_path=args.evaluation_config,
        development_manifest_path=args.development_manifest,
        synthetic_paths=args.phase1_synthetic,
        semantic_config_path=args.semantic_config,
        repository_root=args.repository_root,
        output_dir=args.output_dir,
        cross_source_paths=args.cross_source,
        output_filename=args.output_filename,
        minimum_candidates=args.minimum_candidates,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
