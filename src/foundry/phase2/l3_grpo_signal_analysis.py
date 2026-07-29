"""Publish the content-free Milestone 14B signal-density aggregate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from foundry.phase2.l3_grpo_signal_audit import ARMS, build_signal_summary
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

OUTPUT_NAME = "milestone14b_signal_density_summary.json"


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify(value: Mapping[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def build_publication(
    *,
    raw_by_arm: Mapping[str, Mapping[str, Any]],
    runtime_by_arm: Mapping[str, Mapping[str, Any]],
    raw_file_sha256_by_arm: Mapping[str, str],
) -> dict[str, object]:
    """Combine verified raw audits and runtime summaries without prompt content."""

    for arm in ARMS:
        if arm not in raw_by_arm or arm not in runtime_by_arm or arm not in raw_file_sha256_by_arm:
            raise ValueError(f"{arm} signal-audit evidence is missing")
        _verify(raw_by_arm[arm], "raw_audit_sha256")
        _verify(runtime_by_arm[arm], "summary_sha256")
        runtime = runtime_by_arm[arm]
        raw = raw_by_arm[arm]
        if (
            runtime.get("arm") != arm
            or raw.get("arm") != arm
            or runtime.get("raw_evidence_file_sha256") != raw_file_sha256_by_arm[arm]
            or runtime.get("source_commit") != raw.get("source_commit")
            or runtime.get("signal_audit_contract_sha256")
            != raw.get("signal_audit_contract_sha256")
        ):
            raise ValueError(f"{arm} signal-audit runtime evidence differs")

    density = build_signal_summary(
        raw_by_arm["generic"],
        raw_by_arm["targeted"],
    )
    source_commits = {cast(str, raw_by_arm[arm]["source_commit"]) for arm in ARMS}
    contract_hashes = {cast(str, raw_by_arm[arm]["signal_audit_contract_sha256"]) for arm in ARMS}
    if len(source_commits) != 1 or len(contract_hashes) != 1:
        raise ValueError("signal-audit arm provenance differs")
    publication: dict[str, object] = {
        "schema_version": 1,
        "publication_id": "foundry-l3-grpo-signal-density-publication-v1",
        "source_commit": next(iter(source_commits)),
        "signal_audit_contract_sha256": next(iter(contract_hashes)),
        "signal_density": density,
        "arm_evidence": {
            arm: {
                "raw_audit_sha256": raw_by_arm[arm]["raw_audit_sha256"],
                "raw_evidence_file_sha256": raw_file_sha256_by_arm[arm],
                "runtime_summary_sha256": runtime_by_arm[arm]["summary_sha256"],
            }
            for arm in ARMS
        },
        "resource_usage": {
            "runtime_seconds": sum(
                float(cast(float, runtime_by_arm[arm]["runtime_seconds"])) for arm in ARMS
            ),
            "model_load_seconds": sum(
                float(cast(float, runtime_by_arm[arm]["model_load_seconds"])) for arm in ARMS
            ),
            "generation_seconds": sum(
                float(cast(float, runtime_by_arm[arm]["generation_seconds"])) for arm in ARMS
            ),
            "peak_allocated_vram_bytes": max(
                cast(int, runtime_by_arm[arm]["peak_allocated_vram_bytes"]) for arm in ARMS
            ),
            "peak_reserved_vram_bytes": max(
                cast(int, runtime_by_arm[arm]["peak_reserved_vram_bytes"]) for arm in ARMS
            ),
            "peak_process_rss_bytes": max(
                cast(int, runtime_by_arm[arm]["peak_process_rss_bytes"]) for arm in ARMS
            ),
            "completion_tokens": sum(
                cast(int, runtime_by_arm[arm]["completion_tokens"]) for arm in ARMS
            ),
            "raw_output_disk_bytes": sum(
                cast(int, runtime_by_arm[arm]["output_disk_bytes"]) for arm in ARMS
            ),
        },
        "optimizer_created": False,
        "backward_calls": 0,
        "scheduler_created": False,
        "adapter_saved": False,
        "counted_training_status": "not_run",
        "holdout_v2_status": "not_run",
        "gsm1k_status": "not_run",
        "sealed_content_use": 0,
    }
    publication["publication_sha256"] = canonical_sha256(publication)
    return publication


def aggregate_signal_audit(root: Path) -> dict[str, object]:
    """Read both ignored audits and build their tracked, content-free publication."""

    root = root.resolve()
    raw_root = root / "results/raw/phase2_vetted_corpus/milestone14b/signal_audit"
    raw_by_arm: dict[str, dict[str, Any]] = {}
    runtime_by_arm: dict[str, dict[str, Any]] = {}
    raw_file_sha256_by_arm: dict[str, str] = {}
    for arm in ARMS:
        raw_path = raw_root / arm / "raw_evidence.json"
        raw_by_arm[arm] = _read(raw_path)
        raw_file_sha256_by_arm[arm] = file_sha256(raw_path)
        runtime_by_arm[arm] = _read(raw_root / arm / "summary.json")
    return build_publication(
        raw_by_arm=raw_by_arm,
        runtime_by_arm=runtime_by_arm,
        raw_file_sha256_by_arm=raw_file_sha256_by_arm,
    )


def _write_new_or_identical(path: Path, value: object) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"existing signal-density publication differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    publication = aggregate_signal_audit(args.root)
    output = args.root / "results/phase2_vetted_corpus" / OUTPUT_NAME
    _write_new_or_identical(output, publication)
    print(json.dumps(publication, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
