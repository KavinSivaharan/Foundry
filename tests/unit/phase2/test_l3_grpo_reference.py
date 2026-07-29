from __future__ import annotations

import pytest

from foundry.phase2.l3_grpo_reference import (
    EXPECTED_ADAPTER_PARAMETERS,
    EXPECTED_ADAPTER_TENSORS,
    EXPECTED_LAYERS,
    EXPECTED_PROJECTIONS,
    reference_mechanism_contract,
    trl_per_token_kl,
)


def test_reference_contract_freezes_one_base_two_l3_adapters() -> None:
    contract = reference_mechanism_contract()
    assert contract["second_full_base_model"] is False
    assert contract["reference_trainable"] is False
    assert contract["reference_no_grad"] is True
    assert contract["reference_optimizer_owned"] is False
    assert contract["policy_only_optimizer_owned"] is True
    assert contract["reference_kl_orientation"] == ("frozen_l3_reference_to_active_l3_policy")
    assert contract["expected_adapter_tensors_each"] == EXPECTED_ADAPTER_TENSORS == 112
    assert contract["expected_adapter_parameters_each"] == EXPECTED_ADAPTER_PARAMETERS
    assert contract["expected_layers"] == list(EXPECTED_LAYERS) == list(range(14, 28))
    assert contract["expected_projections"] == list(EXPECTED_PROJECTIONS)
    assert len(str(contract["reference_mechanism_sha256"])) == 64


def test_trl_kl_is_zero_at_identity_and_positive_after_perturbation() -> None:
    assert trl_per_token_kl(-2.0, -2.0) == 0.0
    assert trl_per_token_kl(-2.0, -2.25) > 0.0
    with pytest.raises(ValueError, match="finite"):
        trl_per_token_kl(float("nan"), -2.0)
