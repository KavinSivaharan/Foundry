from __future__ import annotations

from types import SimpleNamespace

import pytest

from foundry.phase2 import l3_grpo_zero_gradient_diagnostic as diagnostic
from foundry.training.config import canonical_sha256

torch = pytest.importorskip("torch")


def test_diagnostic_self_hash_validation() -> None:
    value: dict[str, object] = {"schema_version": 1, "value": "fixture"}
    value["sha256"] = canonical_sha256(value)
    diagnostic._verify_self_hash(value, "sha256")
    value["value"] = "changed"
    with pytest.raises(ValueError, match="does not reconstruct"):
        diagnostic._verify_self_hash(value, "sha256")


def test_diagnostic_partitions_exact_policy_reference_and_base_ownership() -> None:
    values: list[tuple[str, object]] = []
    for index in range(112):
        values.append(
            (
                f"base.layers.{index}.lora_A.default.weight",
                SimpleNamespace(requires_grad=True),
            )
        )
        values.append(
            (
                f"base.layers.{index}.lora_A.reference.weight",
                SimpleNamespace(requires_grad=False),
            )
        )
    values.append(("base.layers.0.weight", SimpleNamespace(requires_grad=False)))
    model = SimpleNamespace(named_parameters=lambda: values)
    policy, reference, base = diagnostic._parameter_partitions(model)
    assert len(policy) == 112
    assert len(reference) == 112
    assert len(base) == 1


def test_diagnostic_tensor_list_requires_a_row_list() -> None:
    assert diagnostic._tensor_list(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    with pytest.raises(TypeError, match="did not produce a list"):
        diagnostic._tensor_list(torch.tensor(1))
