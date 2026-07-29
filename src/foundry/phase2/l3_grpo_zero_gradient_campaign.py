"""Run the two authorized Milestone 14A-R1 official smokes sequentially."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from foundry.phase2.l3_grpo_campaign import (
    _environment,
    _require_clean_synchronized_main,
    _run,
    _runtime_command,
)
from foundry.phase2.l3_grpo_zero_gradient_compatibility import (
    write_compatibility_result,
)


def run_campaign(root: Path) -> dict[str, object]:
    """Run exactly two fresh processes, with no retry path."""

    root = root.resolve()
    environment = _environment(root)
    _require_clean_synchronized_main(root)
    raw = root / "results/raw/phase2_vetted_corpus/milestone14a_r1"
    for index in (1, 2):
        _require_clean_synchronized_main(root)
        run_root = raw / f"compatibility/run-{index}"
        _run(
            _runtime_command(
                root,
                arm="generic",
                mode="compatibility",
                output_dir=run_root / "artifacts",
                raw_evidence=run_root / "raw_evidence.json",
                summary=run_root / "summary.json",
            ),
            root=root,
            environment=environment,
            stdout=raw / f"compatibility/logs/run-{index}.stdout.txt",
            stderr=raw / f"compatibility/logs/run-{index}.stderr.txt",
        )
    return write_compatibility_result(root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_campaign(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
