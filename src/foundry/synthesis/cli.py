"""Command line entry point for the bounded synthesis smoke and replay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from foundry.synthesis.pipeline import load_smoke_config, run_smoke


def _manual_decisions(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, bool) for key, value in raw.items()
    ):
        raise ValueError("manual semantic decisions must be a candidate-ID-to-boolean mapping")
    return dict(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foundry bounded procedural synthesis smoke")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pause-at", type=int)
    parser.add_argument("--manual-decisions", type=Path)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="dry deterministic replay against an existing counted-run summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path.cwd()
    config = load_smoke_config(args.config)
    decisions = {} if args.manual_decisions is None else _manual_decisions(args.manual_decisions)
    if args.replay and args.pause_at is not None:
        raise ValueError("a deterministic replay cannot pause")
    expected: dict[str, object] | None = None
    if args.replay:
        summary_path = repository_root / config.summary_path
        raw_expected = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(raw_expected, dict):
            raise ValueError("the counted-run summary must be a JSON object")
        expected = raw_expected
        replay_directory = config.raw_directory / "deterministic_replay"
        config = replace(
            config,
            raw_directory=replay_directory,
            summary_path=replay_directory / "summary.json",
            manual_audit_path=replay_directory / "manual_audit.jsonl",
        )
    _, summary = run_smoke(
        repository_root=repository_root,
        config=config,
        manual_decisions=decisions,
        pause_at=args.pause_at,
    )
    if expected is not None:
        for key in ("deterministic_decision_sha256", "deterministic_aggregate_sha256"):
            if summary.get(key) != expected.get(key):
                raise RuntimeError(f"deterministic replay mismatch for {key}")
        proof = {
            "counted_attempts": expected.get("attempted"),
            "replay_attempts_not_counted": summary.get("attempted"),
            "deterministic_decision_sha256": summary["deterministic_decision_sha256"],
            "deterministic_aggregate_sha256": summary["deterministic_aggregate_sha256"],
            "matched": True,
        }
        proof_path = repository_root / config.raw_directory / "replay_proof.json"
        proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "REPLAY MATCHED "
            f"decision_sha256={summary['deterministic_decision_sha256']} "
            f"aggregate_sha256={summary['deterministic_aggregate_sha256']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
