"""Separated benchmark-contamination and generated-diversity screening."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from foundry.synthesis.contamination import (
    ContaminationOutcome,
    DevelopmentQuestion,
    normalized_text_sha256,
    numeric_template_sha256,
)
from foundry.synthesis.realization.diversity import (
    InternalDiversityPolicy,
    PairEvidence,
    classify_internal_pair,
    five_token_jaccard,
)
from foundry.synthesis.semantic import PinnedSentenceEncoder


@dataclass(frozen=True)
class ScreenDecision:
    """Content-free result from one complete comparison scope."""

    outcome: ContaminationOutcome
    reason: str
    maximum_similarity: float
    maximum_five_token_jaccard: float
    matched_identifier_prefix: str | None


@dataclass(frozen=True)
class GeneratedReference:
    candidate_id: str
    question: str
    latent_structure_sha256: str
    semantic_frame: str
    realization_signature: str
    embedding: torch.Tensor


class RealizationScreeningIndex:
    """Frozen benchmark firewall plus separately calibrated internal policy."""

    def __init__(
        self,
        *,
        encoder: PinnedSentenceEncoder,
        development_questions: tuple[DevelopmentQuestion, ...],
        internal_policy: InternalDiversityPolicy,
    ) -> None:
        if len(development_questions) != 904:
            raise ValueError("benchmark contamination index requires 904 development questions")
        self.encoder = encoder
        self.internal_policy = internal_policy
        self.development = development_questions
        self.development_embeddings = encoder.encode(
            [question.question for question in development_questions]
        )
        self.generated: list[GeneratedReference] = []

    def encode(self, questions: list[str]) -> torch.Tensor:
        """Embed a stable batch of filled candidate questions."""

        return self.encoder.encode(questions)

    def screen_benchmark(self, question: str, embedding: torch.Tensor) -> ScreenDecision:
        """Apply unchanged 0.75/0.82 generated-to-development controls."""

        exact_hash = normalized_text_sha256(question)
        template_hash = numeric_template_sha256(question)
        maximum_jaccard = 0.0
        jaccard_identifier: str | None = None
        for reference in self.development:
            if normalized_text_sha256(reference.question) == exact_hash:
                return ScreenDecision(
                    ContaminationOutcome.REJECT,
                    "exact_normalized_text",
                    1.0,
                    1.0,
                    reference.stable_id[:12],
                )
            if numeric_template_sha256(reference.question) == template_hash:
                return ScreenDecision(
                    ContaminationOutcome.REJECT,
                    "numeric_template_copy",
                    1.0,
                    1.0,
                    reference.stable_id[:12],
                )
            jaccard = five_token_jaccard(question, reference.question)
            if jaccard > maximum_jaccard:
                maximum_jaccard = jaccard
                jaccard_identifier = reference.stable_id[:12]
        if maximum_jaccard >= 0.35:
            return ScreenDecision(
                ContaminationOutcome.REJECT,
                "token_ngram_overlap",
                0.0,
                maximum_jaccard,
                jaccard_identifier,
            )
        scores = self.encoder.cosine_matrix(embedding.unsqueeze(0), self.development_embeddings)[0]
        maximum, index = torch.max(scores, dim=0)
        similarity = float(maximum.item())
        outcome = self.encoder.config.thresholds.classify(similarity)
        reason = {
            ContaminationOutcome.PASS: "benchmark_all_checks_passed",
            ContaminationOutcome.MANUAL_REVIEW: "benchmark_semantic_manual_review",
            ContaminationOutcome.REJECT: "benchmark_semantic_similarity",
        }[outcome]
        return ScreenDecision(
            outcome,
            reason,
            similarity,
            maximum_jaccard,
            self.development[int(index.item())].stable_id[:12],
        )

    def screen_internal(
        self,
        *,
        question: str,
        embedding: torch.Tensor,
        latent_structure_sha256: str,
        semantic_frame: str,
        realization_signature: str,
    ) -> ScreenDecision:
        """Compare one beam only to candidates selected for earlier IRs."""

        if not self.generated:
            return ScreenDecision(
                ContaminationOutcome.PASS,
                "internal_first_candidate",
                0.0,
                0.0,
                None,
            )
        best_outcome = ContaminationOutcome.PASS
        best_reason = "internal_all_checks_passed"
        best_identifier: str | None = None
        maximum_similarity = 0.0
        maximum_jaccard = 0.0
        for reference in self.generated:
            similarity = float(torch.dot(embedding, reference.embedding).item())
            jaccard = five_token_jaccard(question, reference.question)
            evidence = PairEvidence(
                exact_match=normalized_text_sha256(question)
                == normalized_text_sha256(reference.question),
                numeric_template_match=numeric_template_sha256(question)
                == numeric_template_sha256(reference.question),
                latent_structure_match=latent_structure_sha256 == reference.latent_structure_sha256,
                five_token_jaccard=jaccard,
                semantic_similarity=similarity,
                same_semantic_frame=semantic_frame == reference.semantic_frame,
                same_realization_signature=realization_signature == reference.realization_signature,
            )
            decision = classify_internal_pair(self.internal_policy, evidence)
            maximum_similarity = max(maximum_similarity, similarity)
            maximum_jaccard = max(maximum_jaccard, jaccard)
            if decision.outcome is ContaminationOutcome.REJECT:
                return ScreenDecision(
                    decision.outcome,
                    decision.reason,
                    maximum_similarity,
                    maximum_jaccard,
                    reference.candidate_id[:12],
                )
            if decision.outcome is ContaminationOutcome.MANUAL_REVIEW:
                best_outcome = decision.outcome
                best_reason = decision.reason
                best_identifier = reference.candidate_id[:12]
        return ScreenDecision(
            best_outcome,
            best_reason,
            maximum_similarity,
            maximum_jaccard,
            best_identifier,
        )

    def add_selected(
        self,
        *,
        candidate_id: str,
        question: str,
        latent_structure_sha256: str,
        semantic_frame: str,
        realization_signature: str,
        embedding: torch.Tensor,
    ) -> None:
        """Add only the automatically selected beam for the completed IR."""

        self.generated.append(
            GeneratedReference(
                candidate_id,
                question,
                latent_structure_sha256,
                semantic_frame,
                realization_signature,
                embedding.detach().cpu(),
            )
        )


__all__ = ["RealizationScreeningIndex", "ScreenDecision"]
