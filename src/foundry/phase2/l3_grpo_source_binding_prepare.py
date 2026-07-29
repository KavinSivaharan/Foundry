"""Freeze the Milestone 14B-R3 two-layer L3 GRPO source-binding contract."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from foundry.phase2.l3_grpo_source_binding import (
    CONTRACT_OUTPUT,
    FIXTURE_OUTPUT,
    LAYER1_OUTPUT,
    LAYER2_OUTPUT,
    REPRODUCTION_OUTPUT,
    STARTING_COMMIT,
    build_combined_contract,
    build_fixture_record,
    build_layer1_manifest,
    build_layer2_manifest,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout.strip()


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite source-binding artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def freeze(root: Path) -> tuple[dict[str, object], ...]:
    """Freeze both layers after the source fix is synchronized and clean."""

    root = root.resolve()
    if root != Path(r"C:\Users\Admin\Projects\Foundry").resolve():
        raise ValueError("Milestone 14B-R3 is attached to the wrong repository")
    head = _git(root, "rev-parse", "HEAD")
    if (
        head == STARTING_COMMIT
        or _git(root, "branch", "--show-current") != "main"
        or head != _git(root, "rev-parse", "origin/main")
        or _git(root, "rev-list", "--left-right", "--count", "main...origin/main").split()
        != ["0", "0"]
        or _git(root, "status", "--porcelain")
    ):
        raise RuntimeError("source-binding freeze requires the synchronized clean fix commit")
    reproduction = root / "results/phase2_vetted_corpus" / REPRODUCTION_OUTPUT
    if not reproduction.is_file():
        raise FileNotFoundError("source-binding defect reproduction is absent")
    layer1 = build_layer1_manifest(root)
    layer2 = build_layer2_manifest(root, source_commit=head)
    fixtures = build_fixture_record()
    contract = build_combined_contract(layer1=layer1, layer2=layer2, fixtures=fixtures)
    return layer1, layer2, fixtures, contract


def write_freeze(root: Path) -> tuple[dict[str, object], ...]:
    values = freeze(root)
    output = root.resolve() / "results/phase2_vetted_corpus"
    for name, value in zip(
        (LAYER1_OUTPUT, LAYER2_OUTPUT, FIXTURE_OUTPUT, CONTRACT_OUTPUT),
        values,
        strict=True,
    ):
        _write_new(output / name, value)
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    layer1, layer2, fixtures, contract = write_freeze(args.root)
    print(
        json.dumps(
            {
                "layer1_manifest_sha256": layer1["layer1_manifest_sha256"],
                "layer2_manifest_sha256": layer2["layer2_manifest_sha256"],
                "fixture_sha256": fixtures["fixture_sha256"],
                "source_binding_contract_sha256": contract["source_binding_contract_sha256"],
                "source_commit": layer2["source_commit"],
                "source_tree": layer2["source_tree"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
