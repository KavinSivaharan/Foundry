"""One original-fixture memory and compatibility probe for the approved Qwen3-4B model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from foundry.synthesis.realization.compact_request import prepare_compact_request
from foundry.synthesis.realization.compact_runtime import PinnedCompactQwenRealizer
from foundry.synthesis.realization.compact_smoke_contract import generate_procedural_ir
from foundry.synthesis.realization.compact_validation import parse_compact_response
from foundry.synthesis.realization.smoke import _working_set_peak_bytes
from foundry.synthesis.realization.smoke_contract import (
    RealizationAttemptPlan,
    RealizationGroup,
)
from foundry.synthesis.realization.stronger_model_contract import (
    StrongerModelComparisonConfig,
    load_stronger_model_config,
)
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.taxonomy import FailureCategory


def run_memory_probe(
    *, repository_root: Path, comparison: StrongerModelComparisonConfig
) -> dict[str, object]:
    """Load Qwen3-4B and return exactly three beams for one original procedural fixture."""

    probe = comparison.memory_probe
    plan = RealizationAttemptPlan(
        attempt_index=1,
        group=RealizationGroup.TARGETED,
        group_index=0,
        category=FailureCategory.MULTI_STEP_BOOKKEEPING,
        category_variant=2,
        difficulty=DifficultyLevel.MEDIUM,
        output_contract_enabled=False,
        random_seed=probe.random_seed,
        style_variant=probe.style_variant,
    )
    draft = generate_procedural_ir(plan)
    prepared = prepare_compact_request(draft, style_variant=plan.style_variant)
    runtime = PinnedCompactQwenRealizer(repository_root=repository_root, config=comparison.compact)
    parameter_devices = sorted({str(parameter.device) for parameter in runtime.model.parameters()})
    if parameter_devices != ["cuda:0"]:
        raise RuntimeError("memory probe detected non-CUDA model weights")
    device_map = getattr(runtime.model, "hf_device_map", None)
    if isinstance(device_map, dict) and any(
        str(device) != "cuda:0" for device in device_map.values()
    ):
        raise RuntimeError("memory probe detected CPU or nonapproved device offload")
    torch.cuda.reset_peak_memory_stats()
    generation = runtime.generate(prepared)
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    parsed = 0
    parser_status: list[dict[str, object]] = []
    for beam in generation.beams:
        try:
            parse_compact_response(beam.raw_text)
            status = "parsed"
            parsed += 1
        except ValueError as error:
            status = f"rejected:{type(error).__name__}"
        parser_status.append(
            {
                "beam_index": beam.beam_index,
                "raw_text": beam.raw_text,
                "raw_sha256": beam.raw_sha256,
                "generated_tokens": beam.generated_tokens,
                "parser_status": status,
            }
        )
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    checks = {
        "exactly_three_beams": len(generation.beams) == 3,
        "all_weights_on_cuda": parameter_devices == ["cuda:0"],
        "no_cpu_offload": not isinstance(device_map, dict)
        or all(str(device) == "cuda:0" for device in device_map.values()),
        "no_timeout": not generation.timeout_exceeded,
        "all_beams_decoded": all(bool(beam.raw_text) for beam in generation.beams),
        "all_beams_tag_parsed": parsed == 3,
        "peak_reserved_within_limit": peak_reserved <= probe.peak_reserved_vram_limit_bytes,
        "free_vram_meets_minimum": free_bytes >= probe.minimum_free_vram_bytes,
    }
    result: dict[str, object] = {
        "status": "passed" if all(checks.values()) else "failed",
        "fixture": {
            "kind": "original_nonbenchmark_procedural_fixture",
            "semantic_ir_sha256": draft.semantic_ir_sha256,
            "latent_structure_sha256": draft.structure_sha256,
            "request_sha256": prepared.request_sha256,
        },
        "model": asdict(comparison.artifact),
        "runtime_metadata": asdict(runtime.metadata),
        "generation": {
            "beams": len(generation.beams),
            "input_tokens": generation.input_tokens,
            "output_tokens": sum(beam.generated_tokens for beam in generation.beams),
            "elapsed_seconds": generation.elapsed_seconds,
            "timeout_exceeded": generation.timeout_exceeded,
            "tag_parsed_beams": parsed,
        },
        "resources": {
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "free_gpu_after_generation_bytes": free_bytes,
            "total_gpu_bytes": total_bytes,
            "peak_system_ram_bytes": _working_set_peak_bytes(),
        },
        "checks": checks,
        "beams": parser_status,
    }
    probe.raw_path.parent.mkdir(parents=True, exist_ok=True)
    probe.raw_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    comparison = load_stronger_model_config(args.config)
    result = run_memory_probe(repository_root=args.repository_root.resolve(), comparison=comparison)
    visible = {key: value for key, value in result.items() if key != "beams"}
    print(json.dumps(visible, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["run_memory_probe"]
