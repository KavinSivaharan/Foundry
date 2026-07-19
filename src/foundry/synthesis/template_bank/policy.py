"""Frozen generated-to-generated diversity policy for the offline bank."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TemplateBankDiversityPolicy:
    """Hard identity controls plus non-destructive topical-neighbor review."""

    policy_id: str
    hard_reject: tuple[str, ...]
    review_ngram_at: float
    review_semantic_at: float
    topical_similarity_auto_reject: bool
    benchmark_policy_unchanged: bool
    fixture_set_sha256: str
    policy_sha256: str


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_policy(path: Path) -> TemplateBankDiversityPolicy:
    """Load and validate the pre-smoke internal policy and original fixture inventory."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("template-bank internal policy must be a mapping")
    fixture_path = Path(str(raw["calibration_fixture_path"]))
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or len(fixtures) != 6:
        raise ValueError("template-bank policy requires six original fixture classes")
    expected = ["reject", "reject", "reject", "review", "pass", "pass"]
    if [item.get("expected") for item in fixtures if isinstance(item, dict)] != expected:
        raise ValueError("template-bank fixture outcomes differ from the frozen calibration")
    hard = raw.get("hard_reject")
    review = raw.get("record_for_review")
    alternatives = raw.get("alternatives_rejected")
    if not isinstance(hard, list) or not all(isinstance(item, str) for item in hard):
        raise ValueError("hard-reject controls must be strings")
    if not isinstance(review, dict) or not isinstance(alternatives, list):
        raise ValueError("review and alternative policy records are required")
    canonical = dict(raw)
    policy_hash = _canonical_hash(canonical)
    return TemplateBankDiversityPolicy(
        policy_id=str(raw["policy_id"]),
        hard_reject=tuple(hard),
        review_ngram_at=float(review["five_token_jaccard_at_or_above"]),
        review_semantic_at=float(review["semantic_similarity_at_or_above"]),
        topical_similarity_auto_reject=bool(
            raw["distinct_signature_topical_similarity_is_automatic_reject"]
        ),
        benchmark_policy_unchanged=bool(raw["benchmark_contamination_policy_unchanged"]),
        fixture_set_sha256=_canonical_hash(fixtures),
        policy_sha256=policy_hash,
    )
