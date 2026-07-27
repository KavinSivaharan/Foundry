from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from foundry.phase2 import kl_gradient_scale
from foundry.training.config import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/phase2_vetted_corpus"


def _read(name: str) -> dict[str, Any]:
    value: object = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


def test_raw_scale_analysis_reconstructs_original_ladder() -> None:
    value = kl_gradient_scale.raw_scale_analysis(ROOT)
    supplied = value.pop("raw_scale_analysis_sha256")
    assert supplied == canonical_sha256(value)
    generic = value["arms"]["generic"]
    targeted = value["arms"]["targeted"]
    assert generic["historical_replay_ce"] == pytest.approx(0.22957696406246395)
    assert targeted["historical_replay_token_kl"] == pytest.approx(0.00012543418646708914)
    assert generic["previous_coefficients"][-1][
        "historical_weighted_kl_percent_of_replay_ce"
    ] == pytest.approx(0.017164471969871118)
    assert value["selection_from_raw_loss_scale"] is False


def _audit(ce_generic: float, kl_generic: float, ce_targeted: float, kl_targeted: float):
    value = {
        "schema_version": 1,
        "arms": {
            "generic": {
                "measurement": {
                    "ce_global_l2_norm": ce_generic,
                    "kl_global_l2_norm": kl_generic,
                    "kl_to_ce_gradient_norm_ratio": kl_generic / ce_generic,
                }
            },
            "targeted": {
                "measurement": {
                    "ce_global_l2_norm": ce_targeted,
                    "kl_global_l2_norm": kl_targeted,
                    "kl_to_ce_gradient_norm_ratio": kl_targeted / ce_targeted,
                }
            },
        },
    }
    value["gradient_audit_sha256"] = canonical_sha256(value)
    return value


def test_common_ladder_uses_larger_arm_requirement() -> None:
    value = kl_gradient_scale.derive_common_ladder(_audit(10.0, 2.0, 9.0, 1.0))
    assert [row["rho_exact"] for row in value["ladder"]] == [
        "0.10",
        "0.30",
        "1.00",
        "3.00",
    ]
    assert [row["lambda_common_exact"] for row in value["ladder"]] == [
        "0.90",
        "2.70",
        "9.00",
        "27.00",
    ]


def test_previous_ladder_interpretation_uses_weighted_gradient_norm() -> None:
    value = kl_gradient_scale.previous_ladder_interpretation(_audit(1.0, 0.1, 1.0, 2.0))
    generic = value["arms"]["generic"]["coefficients"]
    targeted = value["arms"]["targeted"]["coefficients"]
    assert generic[0]["classification"] == "less_than_1_percent"
    assert targeted[-1]["weighted_kl_to_ce_gradient_ratio"] == pytest.approx(0.6)
    assert targeted[-1]["classification"] == "30_to_100_percent"


def test_ladder_rejects_unbounded_coefficient() -> None:
    with pytest.raises(ValueError, match="1,000,000"):
        kl_gradient_scale.derive_common_ladder(_audit(1.0, 1e-9, 1.0, 1e-9))


def test_published_raw_scale_and_gradient_freeze_reconstruct() -> None:
    assert _read("milestone13d_raw_scale_analysis.json") == kl_gradient_scale.raw_scale_analysis(
        ROOT
    )
    values = kl_gradient_scale.gradient_audit_record(ROOT)
    manifest = values["manifest"]
    audit = values["audit"]
    assert _read("milestone13d_gradient_measurement_manifest.json") == manifest
    assert _read("milestone13d_historical_gradient_audit.json") == audit
    assert _read(
        "milestone13d_objective_graph_integrity.json"
    ) == kl_gradient_scale.objective_graph_integrity(ROOT, audit)
    assert _read(
        "milestone13d_previous_ladder_gradient_scale.json"
    ) == kl_gradient_scale.previous_ladder_interpretation(audit)
    assert _read("milestone13d_gradient_ladder.json") == kl_gradient_scale.derive_common_ladder(
        audit
    )
