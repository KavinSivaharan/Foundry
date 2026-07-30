"""Fresh-process runtime entrypoints used by the Cycle 1 controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from foundry.cycle.contract import (
    CycleContractError,
    load_cycle_config,
    validate_process_environment,
)
from foundry.cycle.corpus_runtime import (
    freeze_production_corpus,
    freeze_smoke_corpus,
)
from foundry.cycle.evaluation import (
    run_benchmark,
    run_development_retention,
    run_holdout_retention,
)
from foundry.cycle.filtering import filter_candidates
from foundry.cycle.generation import generate_candidates, run_registry_sanity
from foundry.cycle.training import train_candidate
from foundry.training.qlora import file_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "generate",
            "filter",
            "freeze-corpus",
            "train",
            "development-retention",
            "holdout-retention",
            "benchmark",
            "registry-sanity",
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--training-directory", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_cycle_config(args.config)
    validate_process_environment(config=config)
    if (
        Path(sys.executable).resolve()
        != Path(str(config.section("environment")["training_interpreter"])).resolve()
        or file_sha256(Path(sys.executable)) != config.section("environment")["interpreter_sha256"]
    ):
        raise CycleContractError("Cycle runtime is not using the authorized training interpreter")
    if args.operation == "generate":
        result = generate_candidates(
            config=config,
            output_directory=args.output_directory,
            smoke=args.smoke,
        )
    elif args.operation == "filter":
        if args.input_path is None:
            raise ValueError("filter requires --input-path")
        result = filter_candidates(
            config=config,
            candidates_path=args.input_path,
            output_directory=args.output_directory,
            smoke=args.smoke,
        )
    elif args.operation == "freeze-corpus":
        if args.input_path is None:
            raise ValueError("freeze-corpus requires --input-path")
        result = (
            freeze_smoke_corpus(
                config=config,
                selected_trace_path=args.input_path,
                output_directory=args.output_directory,
            )
            if args.smoke
            else freeze_production_corpus(
                config=config,
                selected_trace_path=args.input_path,
                output_directory=args.output_directory,
            )
        )
    elif args.operation == "train":
        if args.input_path is None:
            raise ValueError("train requires the corpus directory as --input-path")
        result = train_candidate(
            config=config,
            corpus_directory=args.input_path,
            output_directory=args.output_directory,
            smoke=args.smoke,
        )
    elif args.operation == "development-retention":
        if args.training_directory is None:
            raise ValueError("development-retention requires --training-directory")
        result = run_development_retention(
            config=config,
            training_directory=args.training_directory,
            output_directory=args.output_directory,
        )
    elif args.operation == "holdout-retention":
        if args.adapter_path is None or args.expected_adapter_sha256 is None:
            raise ValueError(
                "holdout-retention requires --adapter-path and --expected-adapter-sha256"
            )
        result = run_holdout_retention(
            config=config,
            adapter_path=args.adapter_path,
            expected_adapter_sha256=args.expected_adapter_sha256,
            output_directory=args.output_directory,
        )
    elif args.operation == "benchmark":
        if args.adapter_path is None or args.expected_adapter_sha256 is None:
            raise ValueError("benchmark requires --adapter-path and --expected-adapter-sha256")
        result = run_benchmark(
            config=config,
            adapter_path=args.adapter_path,
            expected_adapter_sha256=args.expected_adapter_sha256,
            output_directory=args.output_directory,
        )
    else:
        if args.adapter_path is None or args.expected_adapter_sha256 is None:
            raise ValueError(
                "registry-sanity requires --adapter-path and --expected-adapter-sha256"
            )
        result = run_registry_sanity(
            config=config,
            adapter_path=args.adapter_path,
            expected_adapter_sha256=args.expected_adapter_sha256,
            output_directory=args.output_directory,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
