"""Frozen architecture, curriculum, pilot-size, and success-gate contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from foundry.synthesis.taxonomy import (
    OUTPUT_CONTRACT_TRACK_ID,
    SELECTED_REASONING_CATEGORIES,
    taxonomy_contract_sha256,
)

SYNTHESIS_CONTRACT_VERSION = "foundry-synthesis-design-v1"


@dataclass(frozen=True)
class ArchitectureOption:
    """One considered method for producing synthetic questions."""

    option_id: str
    description: str
    label_source: str
    advantages: tuple[str, ...]
    risks: tuple[str, ...]
    pilot_decision: str


ARCHITECTURE_OPTIONS: tuple[ArchitectureOption, ...] = (
    ArchitectureOption(
        "A",
        "Fully procedural latent programs with controlled natural-language templates.",
        "Exact execution of the sampled latent program.",
        (
            "No model is trusted for labels.",
            "Deterministic reproduction from version and seed.",
            "Lowest dependency, cost, and contamination risk.",
        ),
        (
            "Template language may be less varied.",
            "Template diversity must be measured and bounded explicitly.",
        ),
        "selected",
    ),
    ArchitectureOption(
        "B",
        "Procedural programs followed by local-model paraphrasing and semantic verification.",
        "Exact execution of the original latent program.",
        ("Potentially greater surface-language diversity.",),
        (
            "Paraphrasing can change constraints or targets.",
            "Requires a new local model path and substantially stronger semantic checks.",
        ),
        "deferred",
    ),
    ArchitectureOption(
        "C",
        "Frontier-model question generation followed by executable verification.",
        "Independent executable reconstruction, never the model response.",
        ("Potentially broad linguistic diversity.",),
        (
            "Paid/cloud dependency is outside scope.",
            "Higher provenance, reproducibility, and benchmark-memorization risk.",
        ),
        "rejected_for_first_pilot",
    ),
)


@dataclass(frozen=True)
class DatasetContract:
    """Matched accepted-example budget for one future curriculum."""

    accepted_examples: int
    training_examples: int
    synthetic_validation_examples: int
    output_contract_examples: int
    expected_tokens: int


@dataclass(frozen=True)
class PilotSizeContract:
    """Staged generation and future training limits for the RTX pilot."""

    generator_smoke_examples: int
    targeted: DatasetContract
    generic_control: DatasetContract
    expected_generation_minutes_per_dataset: tuple[int, int]
    expected_verification_minutes_per_dataset: tuple[int, int]
    expected_combined_disk_megabytes: tuple[int, int]
    estimated_qlora_hours_per_run: tuple[float, float]
    estimated_qlora_peak_vram_gib: tuple[float, float]


PILOT_SIZE = PilotSizeContract(
    generator_smoke_examples=120,
    targeted=DatasetContract(4000, 3600, 400, 800, 1_400_000),
    generic_control=DatasetContract(4000, 3600, 400, 800, 1_400_000),
    expected_generation_minutes_per_dataset=(5, 15),
    expected_verification_minutes_per_dataset=(10, 30),
    expected_combined_disk_megabytes=(40, 100),
    estimated_qlora_hours_per_run=(2.0, 4.0),
    estimated_qlora_peak_vram_gib=(7.5, 9.5),
)


@dataclass(frozen=True)
class SuccessGate:
    """One predeclared go/no-go condition for generation or later training."""

    gate_id: str
    requirement: str
    stage: str


SUCCESS_GATES: tuple[SuccessGate, ...] = (
    SuccessGate(
        "dual-verification", "100% accepted examples have two agreeing verifiers.", "generation"
    ),
    SuccessGate(
        "contamination",
        "Zero unresolved benchmark-similarity candidates are accepted.",
        "generation",
    ),
    SuccessGate(
        "deduplication",
        "Zero exact, numeric-template, or latent-program duplicates remain.",
        "generation",
    ),
    SuccessGate(
        "human-audit",
        "At least 30 accepted and 30 rejected candidates per track pass audit.",
        "generation",
    ),
    SuccessGate(
        "one-seed-base",
        "Targeted training exceeds the 64.0049% base score by at least 1 point.",
        "signal_check",
    ),
    SuccessGate(
        "one-seed-control",
        "Targeted training exceeds matched generic control by at least 0.5 points.",
        "signal_check",
    ),
    SuccessGate(
        "two-seed-overall",
        "Mean targeted gain over base is at least 2 points across two seeds.",
        "confirmation",
    ),
    SuccessGate(
        "two-seed-control",
        "Mean targeted gain over generic control is at least 1.5 points.",
        "confirmation",
    ),
    SuccessGate(
        "category-gain",
        "At least two targeted categories improve by 5 points or more.",
        "confirmation",
    ),
    SuccessGate(
        "no-collapse",
        "Untargeted development accuracy falls by no more than 2 points.",
        "confirmation",
    ),
    SuccessGate("extractability", "Extractability remains at least 91.38%.", "confirmation"),
    SuccessGate(
        "sealed-final",
        "Sealed-final remains untouched until the recipe and checkpoint are frozen.",
        "all",
    ),
)


def synthesis_contract_sha256() -> str:
    """Hash all design choices that must precede pilot generation."""

    payload = {
        "version": SYNTHESIS_CONTRACT_VERSION,
        "taxonomy_sha256": taxonomy_contract_sha256(),
        "selected_categories": list(SELECTED_REASONING_CATEGORIES),
        "output_contract_track": OUTPUT_CONTRACT_TRACK_ID,
        "architecture_options": [asdict(option) for option in ARCHITECTURE_OPTIONS],
        "pilot_size": asdict(PILOT_SIZE),
        "success_gates": [asdict(gate) for gate in SUCCESS_GATES],
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
