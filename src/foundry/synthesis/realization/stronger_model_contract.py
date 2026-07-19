"""Frozen one-variable Qwen3-4B comparison against the Milestone 5C micro-smoke."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import yaml

from foundry.synthesis.realization.compact_request import prepare_compact_request
from foundry.synthesis.realization.compact_smoke_contract import (
    CompactSmokeConfig,
    build_compact_attempt_plan,
    generate_procedural_ir,
    load_compact_smoke_config,
)
from foundry.synthesis.realization.smoke_contract import ModelContract, RealizationAttemptPlan


@dataclass(frozen=True)
class StrongerModelArtifact:
    repo_id: str
    revision: str
    license_id: str
    parameter_count: int
    repository_bytes: int
    safetensors_bytes: int
    model_config_sha256: str
    tokenizer_json_sha256: str
    chat_template_sha256: str


@dataclass(frozen=True)
class MemoryProbeContract:
    random_seed: int
    style_variant: int
    peak_reserved_vram_limit_bytes: int
    minimum_free_vram_bytes: int
    raw_path: Path


@dataclass(frozen=True)
class StrongerModelComparisonConfig:
    run_id: str
    base_compact_config: Path
    m5c_raw_path: Path
    m5c_raw_sha256: str
    m5c_control_manifest_sha256: str
    m5c_deterministic_run_sha256: str
    artifact: StrongerModelArtifact
    combined_experiment_sha256: str
    raw_directory: Path
    summary_path: Path
    artifact_summary_path: Path
    memory_probe: MemoryProbeContract
    compact: CompactSmokeConfig
    config_sha256: str


@dataclass(frozen=True)
class ControlledIrVerification:
    ir_count: int
    semantic_ir_hash_matches: int
    latent_hash_matches: int
    request_hash_matches: int
    plan_matches: int
    control_manifest_sha256: str


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _string(data: dict[str, object], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a nonempty string")
    return value


def _integer(data: dict[str, object], key: str, location: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}.{key} must be an integer")
    return value


def _boolean(data: dict[str, object], key: str, location: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{location}.{key} must be boolean")
    return value


def _sha256(value: str, location: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{location} must be a lowercase SHA-256")
    return value


def _experiment_hash(
    base: CompactSmokeConfig, artifact: StrongerModelArtifact, raw_sha: str, deterministic_sha: str
) -> str:
    payload = {
        "base_compact_config_sha256": base.config_sha256,
        "m5c_deterministic_run_sha256": deterministic_sha,
        "m5c_raw_sha256": raw_sha,
        "model_id": artifact.repo_id,
        "model_revision": artifact.revision,
        "model_config_sha256": artifact.model_config_sha256,
        "tokenizer_sha256": artifact.tokenizer_json_sha256,
        "chat_template_sha256": artifact.chat_template_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_stronger_model_config(path: Path) -> StrongerModelComparisonConfig:
    """Load the immutable one-model substitution while reusing the frozen M5C contract."""

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "stronger-model comparison")
    if root.get("schema_version") != 1:
        raise ValueError("stronger-model schema version changed")
    model = _mapping(root.get("model"), "model")
    probe = _mapping(root.get("memory_probe"), "memory_probe")
    rules = _mapping(root.get("rules"), "rules")
    base_path = Path(_string(root, "base_compact_config", "root"))
    base = load_compact_smoke_config(base_path)
    artifact = StrongerModelArtifact(
        repo_id=_string(model, "repo_id", "model"),
        revision=_string(model, "revision", "model"),
        license_id=_string(model, "license_id", "model"),
        parameter_count=_integer(model, "parameter_count", "model"),
        repository_bytes=_integer(model, "repository_bytes", "model"),
        safetensors_bytes=_integer(model, "safetensors_bytes", "model"),
        model_config_sha256=_sha256(
            _string(model, "model_config_sha256", "model"), "model.model_config_sha256"
        ),
        tokenizer_json_sha256=_sha256(
            _string(model, "tokenizer_json_sha256", "model"),
            "model.tokenizer_json_sha256",
        ),
        chat_template_sha256=_sha256(
            _string(model, "chat_template_sha256", "model"),
            "model.chat_template_sha256",
        ),
    )
    raw_sha = _sha256(_string(root, "m5c_raw_sha256", "root"), "root.m5c_raw_sha256")
    deterministic_sha = _sha256(
        _string(root, "m5c_deterministic_run_sha256", "root"),
        "root.m5c_deterministic_run_sha256",
    )
    expected_experiment = _sha256(
        _string(root, "combined_experiment_sha256", "root"),
        "root.combined_experiment_sha256",
    )
    if _experiment_hash(base, artifact, raw_sha, deterministic_sha) != expected_experiment:
        raise ValueError("combined stronger-model experiment hash changed")
    expected_rules = {
        "only_model_changes": True,
        "fresh_or_replacement_irs": False,
        "prompt_changes": False,
        "validator_changes": False,
        "dependency_changes": False,
        "fallback_inference": False,
        "cpu_offload": False,
        "quantization": False,
        "sealed_final_allowed": False,
    }
    if rules != expected_rules:
        raise ValueError("stronger-model safety rules changed")
    if artifact != StrongerModelArtifact(
        "Qwen/Qwen3-4B-Instruct-2507",
        "cdbee75f17c01a7cc42f958dc650907174af0554",
        "Apache-2.0",
        4_022_468_096,
        8_060_917_568,
        8_044_982_000,
        "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326",
    ):
        raise ValueError("approved stronger-model artifact changed")
    config_sha = hashlib.sha256(
        json.dumps(root, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    raw_directory = Path(_string(root, "raw_directory", "root"))
    summary_path = Path(_string(root, "summary_path", "root"))
    compact = replace(
        base,
        run_id=_string(root, "run_id", "root"),
        model=ModelContract(
            artifact.repo_id,
            artifact.revision,
            artifact.license_id,
            _boolean(model, "trust_remote_code", "model"),
            _string(model, "dtype", "model"),
            _boolean(model, "local_files_only", "model"),
            Path(_string(model, "cache_root", "model")),
        ),
        chat_template_sha256=artifact.chat_template_sha256,
        raw_directory=raw_directory,
        summary_path=summary_path,
        config_sha256=config_sha,
    )
    comparison = StrongerModelComparisonConfig(
        run_id=compact.run_id,
        base_compact_config=base_path,
        m5c_raw_path=Path(_string(root, "m5c_raw_path", "root")),
        m5c_raw_sha256=raw_sha,
        m5c_control_manifest_sha256=_sha256(
            _string(root, "m5c_control_manifest_sha256", "root"),
            "root.m5c_control_manifest_sha256",
        ),
        m5c_deterministic_run_sha256=deterministic_sha,
        artifact=artifact,
        combined_experiment_sha256=expected_experiment,
        raw_directory=raw_directory,
        summary_path=summary_path,
        artifact_summary_path=Path(_string(root, "artifact_summary_path", "root")),
        memory_probe=MemoryProbeContract(
            random_seed=_integer(probe, "random_seed", "memory_probe"),
            style_variant=_integer(probe, "style_variant", "memory_probe"),
            peak_reserved_vram_limit_bytes=_integer(
                probe, "peak_reserved_vram_limit_bytes", "memory_probe"
            ),
            minimum_free_vram_bytes=_integer(probe, "minimum_free_vram_bytes", "memory_probe"),
            raw_path=Path(_string(probe, "raw_path", "memory_probe")),
        ),
        compact=compact,
        config_sha256=config_sha,
    )
    if comparison.memory_probe.peak_reserved_vram_limit_bytes != 9_932_111_872:
        raise ValueError("memory-probe reserved VRAM limit changed")
    if comparison.memory_probe.minimum_free_vram_bytes != 536_870_912:
        raise ValueError("memory-probe free VRAM minimum changed")
    return comparison


def _plan_payload(plan: RealizationAttemptPlan) -> dict[str, object]:
    return {
        "attempt_index": plan.attempt_index,
        "category": str(plan.category),
        "category_variant": plan.category_variant,
        "difficulty": str(plan.difficulty),
        "group": str(plan.group),
        "group_index": plan.group_index,
        "output_contract_enabled": plan.output_contract_enabled,
        "random_seed": plan.random_seed,
        "style_variant": plan.style_variant,
    }


def _control_payload(record: dict[str, object]) -> dict[str, object]:
    plan = cast(dict[str, object], record["plan"])
    return {
        "attempt_index": plan["attempt_index"],
        "group": plan["group"],
        "category": plan["category"],
        "difficulty": plan["difficulty"],
        "output_contract_enabled": plan["output_contract_enabled"],
        "candidate_id": record["candidate_id"],
        "semantic_ir_sha256": record["semantic_ir_sha256"],
        "latent_structure_sha256": record["latent_structure_sha256"],
        "request_sha256": record["request_sha256"],
        "semantic_frame": record["semantic_frame"],
        "realization_signature": record["realization_signature"],
    }


def verify_controlled_irs(
    comparison: StrongerModelComparisonConfig,
) -> ControlledIrVerification:
    """Prove all plans, mathematical IRs, and compact requests match M5C exactly."""

    raw_bytes = comparison.m5c_raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != comparison.m5c_raw_sha256:
        raise ValueError("Milestone 5C raw artifact hash changed")
    records = [
        cast(dict[str, object], json.loads(line)) for line in raw_bytes.decode("utf-8").splitlines()
    ]
    plans = build_compact_attempt_plan(comparison.compact)
    if len(records) != 30 or len(plans) != 30:
        raise ValueError("controlled comparison requires exactly 30 prior IR records")
    semantic_matches = latent_matches = request_matches = plan_matches = 0
    for plan, record in zip(plans, records, strict=True):
        if cast(dict[str, object], record["plan"]) == _plan_payload(plan):
            plan_matches += 1
        draft = generate_procedural_ir(plan)
        prepared = prepare_compact_request(draft, style_variant=plan.style_variant)
        semantic_matches += draft.semantic_ir_sha256 == record["semantic_ir_sha256"]
        latent_matches += draft.structure_sha256 == record["latent_structure_sha256"]
        request_matches += prepared.request_sha256 == record["request_sha256"]
        if draft.candidate_id != record["candidate_id"]:
            raise ValueError("controlled comparison candidate ID changed")
    manifest = [_control_payload(record) for record in records]
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = ControlledIrVerification(
        ir_count=len(records),
        semantic_ir_hash_matches=semantic_matches,
        latent_hash_matches=latent_matches,
        request_hash_matches=request_matches,
        plan_matches=plan_matches,
        control_manifest_sha256=manifest_sha,
    )
    if result != ControlledIrVerification(
        30, 30, 30, 30, 30, comparison.m5c_control_manifest_sha256
    ):
        raise ValueError("Milestone 5D differs from the exact M5C IR/control manifest")
    return result


__all__ = [
    "ControlledIrVerification",
    "MemoryProbeContract",
    "StrongerModelArtifact",
    "StrongerModelComparisonConfig",
    "load_stronger_model_config",
    "verify_controlled_irs",
]
