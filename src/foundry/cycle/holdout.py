"""Exact runtime registration for the frozen independent holdout-v2 suite."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from foundry.cycle.contract import (
    load_cycle_config,
    validate_process_environment,
)
from foundry.phase2.launch_contract import (
    FORBIDDEN_PREIMPORT_MODULES,
    validate_preimport,
)

SUITE_ID = "foundry-kl-independent-retention-v2"
EVALUATION_ID = "foundry-kl-independent-retention-evaluation-v2"
SUITE_LAYOUT = {"arithmetic": 120, "format": 120, "instruction": 300}
SUITE_SHA256 = "b8b978ba69b501187b984d4631e8d7f5a41f39efeb292cb391a845c3e61e1b18"
SUBSET_ID = "foundry-kl-independent-retention-base-correct-v2"
SUBSET_LAYOUT = {"arithmetic": 79, "format": 89, "instruction": 149}
SUBSET_SHA256 = "a23b1014d92e9f98b74da3b29913a430bdaebf8e07a16b31b4c3dcc831f1f420"
EXISTING_SUBSET_SHA256 = "4ea32e5cbde0addfbf448291f92d40f1bb40e33dbe4706f7770a3e4c7a0047b7"
SUPPLEMENT_BASE_SUMMARY_SHA256 = "08e9fb7227d5cdabd70a281e6b65d20786011973d4e49653d1066f78afd2ba56"


def _load_combined_subset(path: Path, suite: Any) -> Any:
    from foundry.training.config import canonical_sha256
    from foundry.training.retention import BaseConditionedSubset, Section

    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("combined base-correct subset must be an object")
    expected_hash = value.get("subset_sha256")
    payload = {key: item for key, item in value.items() if key != "subset_sha256"}
    raw_items = value.get("items")
    if (
        value.get("schema_version") != 1
        or value.get("instrument_id") != SUITE_ID
        or value.get("subset_id") != SUBSET_ID
        or value.get("combined_suite_sha256") != suite.suite_sha256
        or value.get("existing_subset_sha256") != EXISTING_SUBSET_SHA256
        or value.get("supplement_base_summary_sha256") != SUPPLEMENT_BASE_SUMMARY_SHA256
        or expected_hash != SUBSET_SHA256
        or expected_hash != canonical_sha256(payload)
        or not isinstance(raw_items, list)
    ):
        raise ValueError("combined base-correct subset identity or hash differs")
    suite_index = {item.item_id: item for item in suite.items}
    items: list[tuple[str, Section, str]] = []
    family_counts: Counter[str] = Counter()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("combined subset item must be an object")
        item_id = str(raw.get("id"))
        suite_item = suite_index.get(item_id)
        if (
            suite_item is None
            or raw.get("section") != suite_item.section
            or raw.get("skill") != suite_item.skill
        ):
            raise ValueError("combined subset item differs from the suite")
        items.append((item_id, suite_item.section, suite_item.skill))
        if "family" in raw:
            family_counts[str(raw["family"])] += 1
    counts = Counter(section for _, section, _ in items)
    if (
        len({item[0] for item in items}) != len(items)
        or value.get("total") != len(items)
        or dict(counts) != SUBSET_LAYOUT
        or value.get("section_counts") != SUBSET_LAYOUT
        or value.get("supplement_family_correct_counts") != dict(family_counts)
        or value.get("all_base_correct_items_included") is not True
        or value.get("calibration_exposure_before_freeze") is not False
        or value.get("adapter_exposure_before_freeze") is not False
        or value.get("prompts_or_references_in_manifest") is not False
    ):
        raise ValueError("combined subset content differs")
    return BaseConditionedSubset(
        instrument_id=SUITE_ID,
        subset_id=SUBSET_ID,
        suite_sha256=suite.suite_sha256,
        base_summary_sha256=SUPPLEMENT_BASE_SUMMARY_SHA256,
        items=tuple(items),
        subset_sha256=SUBSET_SHA256,
    )


def _register(config_path: Path) -> Any:
    config = load_cycle_config(config_path)
    validate_process_environment(config=config)
    validate_preimport()
    if FORBIDDEN_PREIMPORT_MODULES.intersection(sys.modules):
        raise RuntimeError("model stack entered before holdout-v2 registration")
    from foundry.training import retention

    if SUITE_ID in retention.SUITE_LAYOUTS or SUITE_ID in retention.EVALUATION_IDS:
        raise RuntimeError("holdout-v2 suite is unexpectedly pre-registered")
    retention.SUITE_LAYOUTS[SUITE_ID] = dict(SUITE_LAYOUT)
    retention.EVALUATION_IDS[SUITE_ID] = EVALUATION_ID
    retention.load_base_conditioned_subset = _load_combined_subset
    return retention


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("mode", choices=("evaluate", "assess"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    retention = _register(args.config)
    if args.mode == "evaluate":
        sys.argv = [sys.argv[0], *args.arguments]
        retention.main()
        return 0
    from foundry.training import base_conditioned_retention

    sys.argv = [sys.argv[0], "assess", *args.arguments]
    base_conditioned_retention.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
