"""Versioned offline sentence-plan bank for deterministic synthesis."""

from foundry.synthesis.template_bank.bank import TEMPLATE_BANK_VERSION, build_template_bank
from foundry.synthesis.template_bank.contracts import SentencePlanSpec, TemplateSpec

__all__ = [
    "TEMPLATE_BANK_VERSION",
    "SentencePlanSpec",
    "TemplateSpec",
    "build_template_bank",
]
