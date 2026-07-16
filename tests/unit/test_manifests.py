import json
from pathlib import Path

import pytest

from foundry.config import load_config
from foundry.evaluation.manifests import (
    ManifestError,
    build_manifests,
    load_manifest,
    require_partition_access,
    save_manifest,
    validate_manifest_pair,
)

CONFIG_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml")


def test_manifests_are_deterministic_and_disjoint() -> None:
    config = load_config(CONFIG_PATH)

    first_development, first_final = build_manifests(config)
    second_development, second_final = build_manifests(config)

    assert first_development == second_development
    assert first_final == second_final
    assert len(first_development.entries) == 904
    assert len(first_final.entries) == 301

    development_rows = {entry.row_index for entry in first_development.entries}
    final_rows = {entry.row_index for entry in first_final.entries}
    assert development_rows.isdisjoint(final_rows)
    assert development_rows | final_rows == set(range(1205))
    validate_manifest_pair(first_development, first_final, config)


def test_manifest_round_trip_preserves_digest(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    development, _ = build_manifests(config)
    path = tmp_path / "development.json"

    save_manifest(development, path)

    assert load_manifest(path, config) == development


def test_modified_manifest_is_rejected(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    development, _ = build_manifests(config)
    path = tmp_path / "development.json"
    save_manifest(development, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["row_index"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_manifest(path, config)


def test_sealed_final_requires_explicit_access() -> None:
    config = load_config(CONFIG_PATH)
    _, sealed_final = build_manifests(config)

    with pytest.raises(ManifestError, match="sealed-final access denied"):
        require_partition_access(sealed_final, allow_sealed_final=False)

    require_partition_access(sealed_final, allow_sealed_final=True)
