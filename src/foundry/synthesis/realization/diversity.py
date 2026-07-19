"""Original-fixture calibration for generated-to-generated diversity screening."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import torch
import yaml

from foundry.synthesis.contamination import (
    ContaminationOutcome,
    normalize_text,
    normalized_text_sha256,
    numeric_template_sha256,
)
from foundry.synthesis.deduplication import token_ngrams
from foundry.synthesis.semantic import PinnedSentenceEncoder


class PolicyMode(StrEnum):
    """The three policy families compared before any Qwen realization output."""

    LEGACY_BANDS = "legacy_bands"
    EVIDENCE_GATED = "evidence_gated"


@dataclass(frozen=True)
class InternalDiversityFixture:
    """One original, benchmark-independent pair with content-free structural evidence."""

    fixture_id: str
    relationship: str
    left: str
    right: str
    expected_outcome: ContaminationOutcome
    same_semantic_frame: bool
    same_realization_signature: bool
    same_latent_structure: bool
    ambiguity_note: str | None


@dataclass(frozen=True)
class InternalDiversityPolicy:
    """One predeclared candidate layered after frozen structural controls."""

    policy_id: str
    mode: PolicyMode
    semantic_review_at: float
    semantic_reject_at: float
    reject_support_ngram_at: float
    review_support_ngram_at: float
    same_frame_review_at: float
    signature_is_support: bool
    exact_reject: bool = True
    numeric_template_reject: bool = True
    latent_structure_reject: bool = True
    ngram_reject_at: float = 0.35

    def __post_init__(self) -> None:
        if not 0 <= self.semantic_review_at < self.semantic_reject_at <= 1:
            raise ValueError("semantic review/reject thresholds are invalid")
        if self.ngram_reject_at != 0.35:
            raise ValueError("five-token Jaccard rejection must remain 0.35")
        if not self.exact_reject or not self.numeric_template_reject:
            raise ValueError("exact and number-neutral controls cannot be disabled")
        if not self.latent_structure_reject:
            raise ValueError("latent-structure rejection cannot be disabled")

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairEvidence:
    """Auditable lexical, structural, and semantic evidence for one pair."""

    exact_match: bool
    numeric_template_match: bool
    latent_structure_match: bool
    five_token_jaccard: float
    semantic_similarity: float
    same_semantic_frame: bool
    same_realization_signature: bool


@dataclass(frozen=True)
class InternalPolicyDecision:
    """One policy outcome and its first decisive content-free reason."""

    outcome: ContaminationOutcome
    reason: str


@dataclass(frozen=True)
class FixturePolicyResult:
    """One content-free fixture decision under one candidate policy."""

    fixture_id: str
    relationship: str
    expected_outcome: ContaminationOutcome
    actual_outcome: ContaminationOutcome
    reason: str
    five_token_jaccard: float
    semantic_similarity: float


@dataclass(frozen=True)
class CandidatePolicySummary:
    """Aggregate calibration outcome for one policy candidate."""

    policy_id: str
    policy_sha256: str
    exact_matches: int
    duplicate_escapes: int
    distinct_auto_rejections: int
    ambiguous_misclassifications: int
    review_count: int
    reject_count: int
    pass_count: int


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _string(data: dict[str, object], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a nonempty string")
    return value


def _boolean(data: dict[str, object], key: str, location: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{location}.{key} must be a boolean")
    return value


def _number(data: dict[str, object], key: str, location: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{location}.{key} must be numeric")
    return float(value)


def load_internal_diversity_fixtures(path: Path) -> tuple[InternalDiversityFixture, ...]:
    """Load the original fixture set and reject schema or identifier drift."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("internal-diversity fixtures must be a nonempty array")
    fixtures: list[InternalDiversityFixture] = []
    expected_keys = {
        "fixture_id",
        "relationship",
        "left",
        "right",
        "expected_outcome",
        "same_semantic_frame",
        "same_realization_signature",
        "same_latent_structure",
        "ambiguity_note",
    }
    for index, value in enumerate(raw):
        data = _mapping(value, f"fixtures[{index}]")
        if set(data) != expected_keys:
            raise ValueError(f"fixtures[{index}] fields differ from the frozen schema")
        note = data["ambiguity_note"]
        if note is not None and (not isinstance(note, str) or not note.strip()):
            raise ValueError(f"fixtures[{index}].ambiguity_note is invalid")
        fixture = InternalDiversityFixture(
            fixture_id=_string(data, "fixture_id", f"fixtures[{index}]"),
            relationship=_string(data, "relationship", f"fixtures[{index}]"),
            left=_string(data, "left", f"fixtures[{index}]"),
            right=_string(data, "right", f"fixtures[{index}]"),
            expected_outcome=ContaminationOutcome(
                _string(data, "expected_outcome", f"fixtures[{index}]")
            ),
            same_semantic_frame=_boolean(data, "same_semantic_frame", f"fixtures[{index}]"),
            same_realization_signature=_boolean(
                data, "same_realization_signature", f"fixtures[{index}]"
            ),
            same_latent_structure=_boolean(data, "same_latent_structure", f"fixtures[{index}]"),
            ambiguity_note=note,
        )
        if fixture.expected_outcome is ContaminationOutcome.MANUAL_REVIEW:
            if fixture.ambiguity_note is None:
                raise ValueError("manual-review fixtures require an ambiguity note")
        elif fixture.ambiguity_note is not None:
            raise ValueError("only manual-review fixtures may carry ambiguity notes")
        fixtures.append(fixture)
    if len({fixture.fixture_id for fixture in fixtures}) != len(fixtures):
        raise ValueError("fixture identifiers must be unique")
    return tuple(fixtures)


def fixture_set_sha256(fixtures: tuple[InternalDiversityFixture, ...]) -> str:
    """Hash normalized fixture semantics independently of line endings."""

    payload = json.dumps(
        [asdict(fixture) for fixture in fixtures],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_candidate_policies(path: Path) -> tuple[InternalDiversityPolicy, ...]:
    """Load exactly three predeclared policy candidates."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "policy candidates")
    candidates_raw = root.get("candidate_policies")
    if not isinstance(candidates_raw, list) or len(candidates_raw) != 3:
        raise ValueError("exactly three internal-diversity policies must be compared")
    policies: list[InternalDiversityPolicy] = []
    expected_keys = {
        "policy_id",
        "mode",
        "semantic_review_at",
        "semantic_reject_at",
        "reject_support_ngram_at",
        "review_support_ngram_at",
        "same_frame_review_at",
        "signature_is_support",
    }
    for index, value in enumerate(candidates_raw):
        data = _mapping(value, f"candidate_policies[{index}]")
        if set(data) != expected_keys:
            raise ValueError(f"candidate_policies[{index}] fields differ from the schema")
        policies.append(
            InternalDiversityPolicy(
                policy_id=_string(data, "policy_id", f"candidate_policies[{index}]"),
                mode=PolicyMode(_string(data, "mode", f"candidate_policies[{index}]")),
                semantic_review_at=_number(
                    data, "semantic_review_at", f"candidate_policies[{index}]"
                ),
                semantic_reject_at=_number(
                    data, "semantic_reject_at", f"candidate_policies[{index}]"
                ),
                reject_support_ngram_at=_number(
                    data, "reject_support_ngram_at", f"candidate_policies[{index}]"
                ),
                review_support_ngram_at=_number(
                    data, "review_support_ngram_at", f"candidate_policies[{index}]"
                ),
                same_frame_review_at=_number(
                    data, "same_frame_review_at", f"candidate_policies[{index}]"
                ),
                signature_is_support=_boolean(
                    data, "signature_is_support", f"candidate_policies[{index}]"
                ),
            )
        )
    if len({policy.policy_id for policy in policies}) != 3:
        raise ValueError("candidate policy IDs must be unique")
    return tuple(policies)


def load_frozen_internal_policy(path: Path) -> InternalDiversityPolicy:
    """Load the selected policy and verify its calibration and benchmark firewall."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "frozen internal-diversity policy")
    if root.get("fixture_set_sha256") != (
        "e5ba09dc45c6afd58c2c6f9435a33756cb0bb5c20ab73d41313bc49e09c17b89"
    ):
        raise ValueError("internal-diversity fixture hash differs from calibration")
    if root.get("calibration_sha256") != (
        "e855e29a953cbb6b0563e73def3ee4bceb3bbbf20a6f0c5dc7f18573d43063ab"
    ):
        raise ValueError("internal-diversity calibration hash differs from the frozen run")
    benchmark = _mapping(root.get("benchmark_contamination"), "benchmark_contamination")
    expected_benchmark = {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "manual_review_at": 0.75,
        "automatic_reject_at": 0.82,
        "trust_remote_code": False,
        "unchanged": True,
    }
    if benchmark != expected_benchmark:
        raise ValueError("generated-to-development contamination policy changed")
    selected = _mapping(root.get("selected_policy"), "selected_policy")
    policy_hash = selected.pop("policy_sha256", None)
    expected_keys = {
        "policy_id",
        "mode",
        "semantic_review_at",
        "semantic_reject_at",
        "reject_support_ngram_at",
        "review_support_ngram_at",
        "same_frame_review_at",
        "signature_is_support",
    }
    if set(selected) != expected_keys:
        raise ValueError("selected policy fields differ from the frozen schema")
    policy = InternalDiversityPolicy(
        policy_id=_string(selected, "policy_id", "selected_policy"),
        mode=PolicyMode(_string(selected, "mode", "selected_policy")),
        semantic_review_at=_number(selected, "semantic_review_at", "selected_policy"),
        semantic_reject_at=_number(selected, "semantic_reject_at", "selected_policy"),
        reject_support_ngram_at=_number(selected, "reject_support_ngram_at", "selected_policy"),
        review_support_ngram_at=_number(selected, "review_support_ngram_at", "selected_policy"),
        same_frame_review_at=_number(selected, "same_frame_review_at", "selected_policy"),
        signature_is_support=_boolean(selected, "signature_is_support", "selected_policy"),
    )
    if policy_hash != policy.sha256:
        raise ValueError("selected internal-diversity policy hash is invalid")
    return policy


def five_token_jaccard(left: str, right: str) -> float:
    """Compute frozen number-neutral five-token Jaccard overlap."""

    left_ngrams = token_ngrams(left)
    right_ngrams = token_ngrams(right)
    union = left_ngrams | right_ngrams
    return 1.0 if not union else len(left_ngrams & right_ngrams) / len(union)


def build_pair_evidence(
    fixture: InternalDiversityFixture, semantic_similarity: float
) -> PairEvidence:
    """Build every allowed fixture signal without benchmark information."""

    return PairEvidence(
        exact_match=normalized_text_sha256(fixture.left) == normalized_text_sha256(fixture.right),
        numeric_template_match=numeric_template_sha256(fixture.left)
        == numeric_template_sha256(fixture.right),
        latent_structure_match=fixture.same_latent_structure,
        five_token_jaccard=five_token_jaccard(fixture.left, fixture.right),
        semantic_similarity=semantic_similarity,
        same_semantic_frame=fixture.same_semantic_frame,
        same_realization_signature=fixture.same_realization_signature,
    )


def classify_internal_pair(
    policy: InternalDiversityPolicy, evidence: PairEvidence
) -> InternalPolicyDecision:
    """Apply frozen hard controls, then the candidate semantic policy."""

    if evidence.exact_match:
        return InternalPolicyDecision(ContaminationOutcome.REJECT, "exact_normalized_text")
    if evidence.numeric_template_match:
        return InternalPolicyDecision(ContaminationOutcome.REJECT, "numeric_template_copy")
    if evidence.latent_structure_match:
        return InternalPolicyDecision(ContaminationOutcome.REJECT, "latent_structure_copy")
    if evidence.five_token_jaccard >= policy.ngram_reject_at:
        return InternalPolicyDecision(ContaminationOutcome.REJECT, "token_ngram_overlap")
    if policy.mode is PolicyMode.LEGACY_BANDS:
        if evidence.semantic_similarity >= policy.semantic_reject_at:
            return InternalPolicyDecision(ContaminationOutcome.REJECT, "semantic_similarity")
        if evidence.semantic_similarity >= policy.semantic_review_at:
            return InternalPolicyDecision(
                ContaminationOutcome.MANUAL_REVIEW, "semantic_similarity_band"
            )
        return InternalPolicyDecision(ContaminationOutcome.PASS, "all_checks_passed")

    signature_support = policy.signature_is_support and evidence.same_realization_signature
    reject_support = (
        evidence.five_token_jaccard >= policy.reject_support_ngram_at or signature_support
    )
    if evidence.semantic_similarity >= policy.semantic_reject_at and reject_support:
        return InternalPolicyDecision(ContaminationOutcome.REJECT, "supported_semantic_duplicate")
    review_support = (
        evidence.five_token_jaccard >= policy.review_support_ngram_at
        or signature_support
        or (
            evidence.same_semantic_frame
            and evidence.semantic_similarity >= policy.same_frame_review_at
        )
    )
    if evidence.semantic_similarity >= policy.semantic_review_at and review_support:
        return InternalPolicyDecision(
            ContaminationOutcome.MANUAL_REVIEW, "supported_semantic_review"
        )
    return InternalPolicyDecision(ContaminationOutcome.PASS, "all_checks_passed")


def evaluate_candidate_policies(
    *,
    encoder: PinnedSentenceEncoder,
    fixtures: tuple[InternalDiversityFixture, ...],
    policies: tuple[InternalDiversityPolicy, ...],
) -> tuple[dict[str, tuple[FixturePolicyResult, ...]], tuple[CandidatePolicySummary, ...], str]:
    """Run all fixtures once and evaluate the three predeclared policies."""

    texts = [text for fixture in fixtures for text in (fixture.left, fixture.right)]
    embeddings = encoder.encode(texts)
    by_policy: dict[str, tuple[FixturePolicyResult, ...]] = {}
    summaries: list[CandidatePolicySummary] = []
    for policy in policies:
        results: list[FixturePolicyResult] = []
        for index, fixture in enumerate(fixtures):
            similarity = float(torch.dot(embeddings[index * 2], embeddings[index * 2 + 1]).item())
            evidence = build_pair_evidence(fixture, similarity)
            decision = classify_internal_pair(policy, evidence)
            results.append(
                FixturePolicyResult(
                    fixture_id=fixture.fixture_id,
                    relationship=fixture.relationship,
                    expected_outcome=fixture.expected_outcome,
                    actual_outcome=decision.outcome,
                    reason=decision.reason,
                    five_token_jaccard=evidence.five_token_jaccard,
                    semantic_similarity=similarity,
                )
            )
        frozen = tuple(results)
        by_policy[policy.policy_id] = frozen
        summaries.append(
            CandidatePolicySummary(
                policy_id=policy.policy_id,
                policy_sha256=policy.sha256,
                exact_matches=sum(
                    result.actual_outcome is result.expected_outcome for result in frozen
                ),
                duplicate_escapes=sum(
                    result.expected_outcome is ContaminationOutcome.REJECT
                    and result.actual_outcome is not ContaminationOutcome.REJECT
                    for result in frozen
                ),
                distinct_auto_rejections=sum(
                    result.expected_outcome is ContaminationOutcome.PASS
                    and result.actual_outcome is ContaminationOutcome.REJECT
                    for result in frozen
                ),
                ambiguous_misclassifications=sum(
                    result.expected_outcome is ContaminationOutcome.MANUAL_REVIEW
                    and result.actual_outcome is not ContaminationOutcome.MANUAL_REVIEW
                    for result in frozen
                ),
                review_count=sum(
                    result.actual_outcome is ContaminationOutcome.MANUAL_REVIEW for result in frozen
                ),
                reject_count=sum(
                    result.actual_outcome is ContaminationOutcome.REJECT for result in frozen
                ),
                pass_count=sum(
                    result.actual_outcome is ContaminationOutcome.PASS for result in frozen
                ),
            )
        )
    payload = {
        "fixture_set_sha256": fixture_set_sha256(fixtures),
        "embedding_sha256": hashlib.sha256(embeddings.numpy().tobytes()).hexdigest(),
        "policy_summaries": [asdict(summary) for summary in summaries],
        "results": {
            policy_id: [asdict(result) for result in results]
            for policy_id, results in by_policy.items()
        },
    }
    calibration_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return by_policy, tuple(summaries), calibration_sha256


def text_normalization_signature(text: str) -> str:
    """Expose normalized content only for original-fixture test assertions."""

    return normalize_text(text, replace_numbers=True)
