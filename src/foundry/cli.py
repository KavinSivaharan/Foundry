"""Command-line interface for the approved reproducible evaluation foundation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from foundry.config import ConfigError, load_config
from foundry.evaluation.backends import (
    BackendError,
    FakeModelBackend,
    HuggingFaceCudaBackend,
)
from foundry.evaluation.benchmark import (
    BenchmarkError,
    load_fixture_examples,
    load_huggingface_examples,
)
from foundry.evaluation.calibration import (
    PROMPT_FORMAT_CALIBRATION,
    CalibrationError,
    as_development_benchmark_manifest,
    assert_prompt_only_variant,
    build_format_calibration_manifests,
    load_development_subset,
    save_development_subset,
    validate_format_calibration_pair,
)
from foundry.evaluation.manifests import (
    ManifestError,
    build_manifests,
    load_manifest,
    save_manifest,
    validate_manifest_pair,
)
from foundry.evaluation.runner import run_evaluation


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foundry")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate and hash a YAML config")
    validate.add_argument("--config", required=True, type=_path)

    manifests = subparsers.add_parser(
        "build-manifests",
        help="build deterministic development and sealed-final identifier manifests",
    )
    manifests.add_argument("--config", required=True, type=_path)

    fixture = subparsers.add_parser(
        "evaluate-fixture",
        help="run the full evaluation path with a local synthetic fixture",
    )
    fixture.add_argument("--config", required=True, type=_path)
    fixture.add_argument("--manifest", required=True, type=_path)
    fixture.add_argument("--fixture", required=True, type=_path)
    fixture.add_argument("--output-dir", required=True, type=_path)
    fixture.add_argument("--limit", type=int)
    fixture.add_argument("--allow-sealed-final", action="store_true")

    smoke = subparsers.add_parser(
        "smoke",
        help="run at most 10 pinned development examples on a CUDA GPU",
    )
    smoke.add_argument("--config", required=True, type=_path)
    smoke.add_argument("--manifest", required=True, type=_path)
    smoke.add_argument("--output-dir", required=True, type=_path)
    smoke.add_argument("--limit", required=True, type=int)

    calibration_manifests = subparsers.add_parser(
        "build-format-calibration",
        help="build deterministic prompt-calibration and future-baseline ID manifests",
    )
    calibration_manifests.add_argument("--config", required=True, type=_path)
    calibration_manifests.add_argument("--development-manifest", required=True, type=_path)
    calibration_manifests.add_argument("--calibration-manifest", required=True, type=_path)
    calibration_manifests.add_argument("--baseline-manifest", required=True, type=_path)
    calibration_manifests.add_argument("--size", required=True, type=int)
    calibration_manifests.add_argument("--seed", required=True)

    calibrate = subparsers.add_parser(
        "format-calibrate",
        help="evaluate exactly 30 prompt-format calibration development identifiers",
    )
    calibrate.add_argument("--base-config", required=True, type=_path)
    calibrate.add_argument("--config", required=True, type=_path)
    calibrate.add_argument("--development-manifest", required=True, type=_path)
    calibrate.add_argument("--calibration-manifest", required=True, type=_path)
    calibrate.add_argument("--output-dir", required=True, type=_path)
    return parser


def _resolve_config_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path.cwd() / path


def _run_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(json.dumps({"config_sha256": config.sha256}, sort_keys=True))
    return 0


def _run_build_manifests(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    development, sealed_final = build_manifests(config)
    development_path = _resolve_config_path(config.partition.development_manifest)
    final_path = _resolve_config_path(config.partition.sealed_final_manifest)
    save_manifest(development, development_path)
    save_manifest(sealed_final, final_path)
    reloaded_development = load_manifest(development_path, config)
    reloaded_final = load_manifest(final_path, config)
    validate_manifest_pair(reloaded_development, reloaded_final, config)
    print(
        json.dumps(
            {
                "development_examples": len(development.entries),
                "development_manifest": str(development_path),
                "sealed_final_examples": len(sealed_final.entries),
                "sealed_final_manifest": str(final_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _run_fixture(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest = load_manifest(args.manifest, config)
    fixture_examples = load_fixture_examples(
        args.fixture,
        manifest,
        allow_sealed_final=args.allow_sealed_final,
        limit=args.limit,
    )
    examples = tuple(item.example for item in fixture_examples)
    backend = FakeModelBackend(
        {item.example.stable_id: item.fake_response for item in fixture_examples}
    )
    summary = run_evaluation(
        config=config,
        manifest=manifest,
        examples=examples,
        backend=backend,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


def _run_smoke(args: argparse.Namespace) -> int:
    if not 1 <= args.limit <= 10:
        raise ValueError("smoke --limit must be between 1 and 10")
    config = load_config(args.config)
    manifest = load_manifest(args.manifest, config)
    if manifest.partition != "development":
        raise ManifestError("the approved smoke run may use only the development manifest")

    backend = HuggingFaceCudaBackend(config)
    examples = load_huggingface_examples(config, manifest, limit=args.limit)
    summary = run_evaluation(
        config=config,
        manifest=manifest,
        examples=examples,
        backend=backend,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


def _run_build_format_calibration(args: argparse.Namespace) -> int:
    if args.size != 30:
        raise CalibrationError("the approved prompt-format calibration size is exactly 30")
    config = load_config(args.config)
    development = load_manifest(args.development_manifest, config)
    calibration, baseline = build_format_calibration_manifests(
        development,
        calibration_size=args.size,
        selection_seed=args.seed,
    )
    save_development_subset(calibration, args.calibration_manifest)
    save_development_subset(baseline, args.baseline_manifest)
    reloaded_calibration = load_development_subset(args.calibration_manifest, development)
    reloaded_baseline = load_development_subset(args.baseline_manifest, development)
    validate_format_calibration_pair(reloaded_calibration, reloaded_baseline, development)
    print(
        json.dumps(
            {
                "baseline_examples": len(baseline.entries),
                "baseline_manifest_sha256": baseline.manifest_sha256,
                "calibration_examples": len(calibration.entries),
                "calibration_manifest_sha256": calibration.manifest_sha256,
                "selection_seed": args.seed,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_format_calibrate(args: argparse.Namespace) -> int:
    base_config = load_config(args.base_config)
    variant_config = load_config(args.config)
    assert_prompt_only_variant(base_config, variant_config)
    development = load_manifest(args.development_manifest, base_config)
    calibration = load_development_subset(args.calibration_manifest, development)
    if calibration.purpose != PROMPT_FORMAT_CALIBRATION:
        raise CalibrationError("format-calibrate requires the prompt-format calibration manifest")
    if len(calibration.entries) != 30:
        raise CalibrationError("the approved prompt-format calibration requires exactly 30 IDs")
    manifest = as_development_benchmark_manifest(calibration, development, variant_config)
    backend = HuggingFaceCudaBackend(variant_config)
    examples = load_huggingface_examples(variant_config, manifest)
    summary = run_evaluation(
        config=variant_config,
        manifest=manifest,
        examples=examples,
        backend=backend,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one Foundry evaluation-foundation command."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            return _run_validate(args)
        if args.command == "build-manifests":
            return _run_build_manifests(args)
        if args.command == "evaluate-fixture":
            return _run_fixture(args)
        if args.command == "smoke":
            return _run_smoke(args)
        if args.command == "build-format-calibration":
            return _run_build_format_calibration(args)
        if args.command == "format-calibrate":
            return _run_format_calibrate(args)
        parser.error(f"unsupported command {args.command}")
    except (
        BackendError,
        BenchmarkError,
        CalibrationError,
        ConfigError,
        ManifestError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
