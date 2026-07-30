"""Fresh-process runtime entrypoints used by the Cycle 1 controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType

from foundry.cycle.contract import (
    CYCLE_ID,
    RECOVERY_EXECUTION_ID,
    CycleContractError,
    bind_cycle_execution,
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
from foundry.cycle.generation_observability import (
    ensure_recovery_runtime,
    recovery_identity,
    verify_source_identity,
)
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
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--execution-id")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    return parser


def main() -> int:
    args = _parser().parse_args()
    base_config = load_cycle_config(args.config)
    execution_id = args.execution_id or CYCLE_ID
    if args.diagnostic and args.operation != "generate":
        raise CycleContractError("diagnostic mode is authorized only for generation")
    config = base_config if args.diagnostic else bind_cycle_execution(base_config, execution_id)
    validate_process_environment(config=config)
    if (
        Path(sys.executable).resolve()
        != Path(str(config.section("environment")["training_interpreter"])).resolve()
        or file_sha256(Path(sys.executable)) != config.section("environment")["interpreter_sha256"]
    ):
        raise CycleContractError("Cycle runtime is not using the authorized training interpreter")
    source_identity = None
    if not args.diagnostic:
        required_source = (args.source_root, args.source_commit, args.source_tree)
        if any(value is None for value in required_source):
            raise CycleContractError("cycle runtime requires frozen source identity")
        if args.source_root.resolve() != config.source_root.resolve():
            raise CycleContractError("cycle runtime source root differs from execution binding")
        module_file = Path(__file__)
        source_identity = verify_source_identity(
            source_root=args.source_root,
            expected_commit=str(args.source_commit),
            expected_tree=str(args.source_tree),
            imported_file=module_file,
        )
        try:
            args.output_directory.resolve().relative_to(config.runtime_root.resolve())
        except ValueError as error:
            raise CycleContractError("cycle runtime output escapes the execution root") from error
    if args.operation == "generate":
        if args.diagnostic:
            required = (
                args.execution_id,
                args.source_root,
                args.source_commit,
                args.source_tree,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "diagnostic generation requires execution ID and frozen source identity"
                )
            expected_runtime = Path(r"C:\Users\Admin\Projects\Foundry-cycle1-r1-runtime").resolve()
            expected_output = expected_runtime / "diagnostic" / "generation"
            if (
                execution_id != RECOVERY_EXECUTION_ID
                or args.source_root.resolve()
                != Path(r"C:\Users\Admin\Projects\Foundry-cycle1-r1-observe-frozen").resolve()
                or args.output_directory.resolve() != expected_output
            ):
                raise CycleContractError("diagnostic recovery path or identity differs")
            module = sys.modules[generate_candidates.__module__]
            if not isinstance(module, ModuleType) or module.__file__ is None:
                raise CycleContractError("generation module source path is unavailable")
            source_identity = verify_source_identity(
                source_root=args.source_root,
                expected_commit=str(args.source_commit),
                expected_tree=str(args.source_tree),
                imported_file=Path(module.__file__),
            )
            ensure_recovery_runtime(
                expected_runtime,
                recovery_identity(
                    config_sha256=config.sha256,
                    model_revision=str(config.section("model")["revision"]),
                    dataset_sha256=str(config.section("dataset")["identity_sha256"]),
                    starting_adapter_sha256=str(config.section("warm_start")["adapter_sha256"]),
                    prior_runtime_tree_sha256=(
                        "5086deab27d522e939874e65c5d8d74b7d5c43a082c73e6b51398ff04b317d05"
                    ),
                    prior_rejection_sha256=(
                        "8ceefd73523aac59a95d6ccac30f2f2fb19d3272328cda51c253dbe55114a000"
                    ),
                ),
            )
        result = generate_candidates(
            config=config,
            output_directory=args.output_directory,
            smoke=args.smoke,
            diagnostic=args.diagnostic,
            execution_id=execution_id,
            source_identity=source_identity,
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
