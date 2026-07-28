from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from foundry.phase2 import vetted_qlora_layer_restricted
from foundry.phase2.layer_restricted import (
    LAYER_SCOPES,
    build_contract,
    scope_for_label,
    select_largest_passing,
)
from foundry.phase2.vetted_qlora_layer_restricted import _inventory


def test_layer_ladder_is_exact_and_immutable() -> None:
    assert [
        (scope.label, scope.top_layer_count, scope.layer_indices) for scope in LAYER_SCOPES
    ] == [
        ("L1", 4, (24, 25, 26, 27)),
        ("L2", 8, (20, 21, 22, 23, 24, 25, 26, 27)),
        ("L3", 14, tuple(range(14, 28))),
    ]


def test_tracked_layer_scope_contract_reconstructs() -> None:
    path = Path("results/phase2_vetted_corpus/milestone13e_layer_scope_contract.json")
    tracked = json.loads(path.read_text(encoding="utf-8"))
    assert build_contract(Path(".")) == tracked
    assert [
        (
            scope.adapted_module_count,
            scope.trainable_tensor_count,
            scope.trainable_parameter_count,
        )
        for scope in LAYER_SCOPES
    ] == [
        (16, 32, 311_296),
        (32, 64, 622_592),
        (56, 112, 1_089_536),
    ]


def test_only_predeclared_layer_labels_are_accepted() -> None:
    assert scope_for_label("L3").layer_indices == tuple(range(14, 28))
    with pytest.raises(ValueError, match="not predeclared"):
        scope_for_label("top-20")


@pytest.mark.parametrize(
    ("passes", "expected"),
    [
        ({"L1": True, "L2": True, "L3": True}, "L3"),
        ({"L1": True, "L2": True, "L3": False}, "L2"),
        ({"L1": True, "L2": False, "L3": False}, "L1"),
        ({"L1": False, "L2": False, "L3": False}, None),
    ],
)
def test_selection_always_chooses_largest_common_pass(
    passes: dict[str, bool],
    expected: str | None,
) -> None:
    assert select_largest_passing(passes) == expected


def test_selection_rejects_an_incomplete_or_expanded_ladder() -> None:
    with pytest.raises(ValueError, match="exact frozen ladder"):
        select_largest_passing({"L1": True, "L2": True})
    with pytest.raises(ValueError, match="exact frozen ladder"):
        select_largest_passing({"L1": True, "L2": True, "L3": True, "L4": True})


@dataclass
class _Parameter:
    shape: tuple[int, int]


class _Model:
    def __init__(self, layers: tuple[int, ...]) -> None:
        self._parameters: list[tuple[str, _Parameter]] = []
        dimensions = {
            "q_proj": ((8, 1536), (1536, 8)),
            "k_proj": ((8, 1536), (256, 8)),
            "v_proj": ((8, 1536), (256, 8)),
            "o_proj": ((8, 1536), (1536, 8)),
        }
        for layer in layers:
            for projection, (shape_a, shape_b) in dimensions.items():
                prefix = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
                self._parameters.extend(
                    [
                        (f"{prefix}.lora_A.default.weight", _Parameter(shape_a)),
                        (f"{prefix}.lora_B.default.weight", _Parameter(shape_b)),
                    ]
                )

    def named_parameters(self) -> list[tuple[str, _Parameter]]:
        return self._parameters


@pytest.mark.parametrize("label", ["L1", "L2", "L3"])
def test_runtime_inventory_enforces_exact_selected_layers(label: str) -> None:
    scope = scope_for_label(label)
    inventory = _inventory(_Model(scope.layer_indices), scope)
    assert inventory["layer_indices"] == list(scope.layer_indices)
    assert inventory["trainable_tensor_count"] == scope.trainable_tensor_count
    assert inventory["trainable_parameter_count"] == scope.trainable_parameter_count
    assert inventory["adapted_module_count"] == scope.adapted_module_count


def test_training_source_keeps_kl_diagnostic_outside_optimizer_loop() -> None:
    source = inspect.getsource(vetted_qlora_layer_restricted.train)
    loop = source.index("for step in schedule:")
    diagnostic = source.index("kl_runner._measure")
    assert "coefficient" not in source
    assert "forward_token_kl" not in source
    assert diagnostic > loop
    assert '"post_training_diagnostic_only"' in source


def test_prepare_preserves_v1_lora_fields_and_adds_only_layer_selection() -> None:
    source = inspect.getsource(vetted_qlora_layer_restricted._prepare)
    for fragment in (
        "r=8",
        "lora_alpha=16",
        "lora_dropout=0.05",
        'bias="none"',
        "target_modules=list(PROJECTIONS)",
        "layers_to_transform=list(scope.layer_indices)",
        'layers_pattern="layers"',
        'task_type="CAUSAL_LM"',
    ):
        assert fragment in source
