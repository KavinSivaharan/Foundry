from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from foundry.phase2 import kl_objective_contract

ROOT = Path(__file__).resolve().parents[3]
RECORD_PATH = ROOT / "results" / "phase2_vetted_corpus" / "milestone13c_r3_kl_objective.json"


def _record() -> dict[str, Any]:
    value: object = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


def test_objective_configuration_freezes_only_token_kl_intervention() -> None:
    value = kl_objective_contract._configuration_contract()
    assert value["candidate_coefficients_in_order"] == [0.01, 0.03, 0.10, 0.30]
    assert value["kl_direction"] == ("adapter_disabled_frozen_base||active_adapter_policy")
    assert value["reference_model_count"] == 1
    assert value["vetted_kl"] is False


def test_masking_and_reference_contracts_are_exact() -> None:
    masking = kl_objective_contract._masking_contract()
    reference = kl_objective_contract._reference_contract()
    assert masking["selection"] == "labels[:, 1:] != -100"
    assert masking["included"] == ["assistant_content", "final_assistant_eos"]
    assert reference["reference_context"] == "torch.no_grad"
    assert reference["second_full_model_instantiated"] is False


def test_published_objective_contract_validates() -> None:
    value = _record()
    kl_objective_contract.validate(ROOT, value)
    assert value["fixture"]["finite_gradients"]
    assert value["fixture"]["post_update_replay_kl"] > 0.0
    assert value["holdout_v2_adapter_exposure"] is False
