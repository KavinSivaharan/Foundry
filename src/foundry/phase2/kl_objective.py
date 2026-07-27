"""Pure token-level forward-KL objective helpers for replay-ce-token-kl-v1."""

from __future__ import annotations

from typing import Any


def assistant_shift_mask(labels: Any) -> Any:
    """Select logits that predict supervised assistant tokens, including final EOS."""

    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must have shape [batch, sequence]")
    mask = labels[:, 1:] != -100
    if not bool(mask.any().item()):
        raise ValueError("assistant mask selects no tokens")
    return mask


def forward_token_kl(
    reference_logits: Any,
    policy_logits: Any,
    mask: Any,
    torch: Any,
) -> Any:
    """Compute mean KL(reference || policy) in float32 on selected token positions."""

    if reference_logits.shape != policy_logits.shape:
        raise ValueError("reference and policy logit shapes differ")
    if reference_logits.ndim != 3 or mask.shape != reference_logits.shape[:2]:
        raise ValueError("KL mask shape differs from shifted logits")
    reference_selected = reference_logits[mask].float()
    policy_selected = policy_logits[mask].float()
    reference_log_prob = torch.nn.functional.log_softmax(reference_selected, dim=-1)
    policy_log_prob = torch.nn.functional.log_softmax(policy_selected, dim=-1)
    reference_prob = reference_log_prob.exp()
    per_token = torch.sum(
        reference_prob * (reference_log_prob - policy_log_prob),
        dim=-1,
    )
    return per_token.mean()


def replay_total_loss(cross_entropy: Any, token_kl: Any, coefficient: float) -> Any:
    """Combine replay-target CE with the frozen nonnegative KL coefficient."""

    if coefficient <= 0.0:
        raise ValueError("KL training coefficient must be positive")
    return cross_entropy + coefficient * token_kl
