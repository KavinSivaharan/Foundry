"""Contracts for the fixed 500-by-2 matched-template candidate pool."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest

from foundry.synthesis.template_bank.matched_dataset import (
    CATEGORY_ORDER,
    FAMILY_ACCEPTED,
    FAMILY_ATTEMPTS,
    GROUP_ORDER,
    build_quota_contract,
    build_schedule,
    load_matched_dataset_config,
    load_schedule,
)

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs/synthesis/matched_signal_dataset.yaml"


def test_matched_quota_contract_is_exact() -> None:
    quota = build_quota_contract(CONFIG)
    groups = cast(dict[str, dict[str, object]], quota["groups"])

    assert quota["accepted_total"] == 1000
    assert quota["attempt_total"] == 1100
    for group in GROUP_ORDER:
        families = cast(dict[str, dict[str, object]], groups[group]["families"])
        for family in CATEGORY_ORDER:
            cells = cast(list[dict[str, object]], families[family]["cells"])
            assert (
                sum(cast(int, cell["accepted"]) for cell in cells)
                == (FAMILY_ACCEPTED[group][family])
            )
            assert (
                sum(cast(int, cell["attempts"]) for cell in cells)
                == (FAMILY_ATTEMPTS[group][family])
            )


def test_frozen_matched_schedule_is_unique_and_reconstructible() -> None:
    config = load_matched_dataset_config(CONFIG)
    frozen = load_schedule(config)
    rebuilt = build_schedule(CONFIG)

    assert [asdict(item) for item in rebuilt] == [asdict(item) for item in frozen]
    assert len(frozen) == 1100
    assert len({item.synthetic_id for item in frozen}) == 1100
    assert len({item.latent_program_sha256 for item in frozen}) == 1100
    assert len({item.exact_text_sha256 for item in frozen}) == 1100
    assert Counter((item.group, item.family) for item in frozen) == Counter(
        {
            (group, family): FAMILY_ATTEMPTS[group][family]
            for group in GROUP_ORDER
            for family in CATEGORY_ORDER
        }
    )


def test_matched_schedule_tampering_fails_closed(tmp_path: Path) -> None:
    config = load_matched_dataset_config(CONFIG)
    raw = json.loads(config.schedule_manifest.read_text(encoding="utf-8"))
    raw["records"][0]["latent_seed"] += 1
    tampered = tmp_path / "schedule.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        load_schedule(replace(config, schedule_manifest=tampered))
