from __future__ import annotations

from pathlib import Path

from foundry.phase2 import l3_grpo_zero_gradient_prepare as prepare
from foundry.training.config import canonical_sha256


def test_prediagnostic_source_manifest_is_complete_and_reconstructs() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = prepare._source_manifest(root)
    expected = {
        *prepare.IMPLEMENTATION_FILES,
        *prepare.FROZEN_DEPENDENCY_FILES,
    }
    assert {row["path"] for row in manifest["files"]} == expected
    manifest_sha256 = manifest["source_manifest_sha256"]
    projected = dict(manifest)
    projected.pop("source_manifest_sha256")
    assert manifest_sha256 == canonical_sha256(projected)


def test_freeze_constants_bind_the_published_stop() -> None:
    assert prepare.STARTING_COMMIT == "b0635a7c0f551dfb8efd846da5cfe83b28f7af18"
    assert (
        prepare.ORIGINAL_BLOCKER_SHA256
        == "d4b23d898ef3c53db46882a4a218c2a43cd85298ebdfa75139eaf3a7c08e8752"
    )
    assert prepare.OUTPUT_NAME == "milestone14a_r1_zero_gradient_freeze.json"
