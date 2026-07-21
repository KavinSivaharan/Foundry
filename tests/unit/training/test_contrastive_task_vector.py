from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from foundry.training.config import TARGET_MODULES, canonical_sha256
from foundry.training.contrastive_task_vector import (
    CONTRASTIVE_NAME,
    GENERIC_NAME,
    TARGETED_NAME,
    _materialize_exact_fp32_cat,
    _paired_modules,
    _verify_saved_dense_equivalence,
    analyze_dense_updates,
    construct_task_vector,
    inspect_adapter,
    load_protocol,
    validate_source_compatibility,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PROTOCOL = PROJECT_ROOT / "configs/training/contrastive_task_vector_v1.json"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
Q_PROJ = "base_model.model.model.layers.0.self_attn.q_proj"


def _write_adapter(
    path: Path,
    *,
    tensors: dict[str, torch.Tensor],
    rank: int,
    alpha: int,
    target_modules: list[str],
    base_revision: str = BASE_REVISION,
) -> None:
    path.mkdir(parents=True)
    config = {
        "base_model_name_or_path": f"snapshots/{base_revision}",
        "bias": "none",
        "fan_in_fan_out": False,
        "lora_alpha": alpha,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "r": rank,
        "target_modules": target_modules,
        "use_dora": False,
        "use_rslora": False,
    }
    (path / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_file(
        {name: tensor.detach().clone().contiguous() for name, tensor in tensors.items()},
        path / "adapter_model.safetensors",
    )


def _one_module_tensors(*, sign: float) -> dict[str, torch.Tensor]:
    return {
        f"{Q_PROJ}.lora_A.weight": torch.eye(2, dtype=torch.float32),
        f"{Q_PROJ}.lora_B.weight": sign * torch.eye(2, dtype=torch.float32),
    }


def _write_protocol(
    path: Path,
    *,
    generic: dict[str, Any],
    targeted: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "foundry-targeted-minus-generic-task-vector-v1",
        "base_model": {"revision": BASE_REVISION},
        "source_adapters": {
            GENERIC_NAME: {"directory_sha256": generic["directory_sha256"]},
            TARGETED_NAME: {"directory_sha256": targeted["directory_sha256"]},
        },
        "source_contract": {
            "rank": generic["rank"],
            "alpha": generic["alpha"],
            "dropout": generic["dropout"],
            "bias": generic["bias"],
            "modules_to_save": None,
            "target_modules": list(reversed(generic["target_modules"])),
            "lora_module_count": generic["lora_module_count"],
            "saved_tensor_count": generic["saved_tensor_count"],
            "inventory_sha256": generic["tensor_inventory_sha256"],
        },
        "composition": {
            "adapter_name": CONTRASTIVE_NAME,
            "adapters": [TARGETED_NAME, GENERIC_NAME],
            "weights": [1.0, -1.0],
            "combination_type": "cat",
            "factor_dtype": "float32",
            "merge_allowed": False,
            "compression_allowed": False,
        },
    }
    protocol = {**payload, "config_sha256": canonical_sha256(payload)}
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol


def _tiny_sources(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    generic_path = tmp_path / "generic"
    targeted_path = tmp_path / "targeted"
    _write_adapter(
        generic_path,
        tensors=_one_module_tensors(sign=1.0),
        rank=2,
        alpha=4,
        target_modules=["q_proj"],
    )
    _write_adapter(
        targeted_path,
        tensors=_one_module_tensors(sign=-1.0),
        rank=2,
        alpha=4,
        target_modules=["q_proj"],
    )
    generic = inspect_adapter(generic_path)
    targeted = inspect_adapter(targeted_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(
        protocol_path,
        generic=generic,
        targeted=targeted,
    )
    return generic_path, targeted_path, protocol_path, protocol


def test_frozen_protocol_hash_and_contract_load() -> None:
    protocol = load_protocol(FROZEN_PROTOCOL)

    assert protocol["config_sha256"] == (
        "b4914d5a95bb46a52374b9a390038634f01df99f69a4ef6f79c5bfe4f8d983fa"
    )
    assert protocol["source_contract"]["inventory_sha256"] == (
        "c3334a15154beae94d6b4bd043d24742cf522b86f48efd044e3a9fe7e1f978b9"
    )
    assert protocol["composition"] == {
        "adapter_name": CONTRASTIVE_NAME,
        "adapters": [TARGETED_NAME, GENERIC_NAME],
        "weights": [1.0, -1.0],
        "combination_type": "cat",
        "expected_rank": 32,
        "expected_alpha": 32,
        "expected_scaling": 1.0,
        "factor_dtype": "float32",
        "merge_allowed": False,
        "compression_allowed": False,
    }


def test_protocol_rejects_hash_and_composition_changes(tmp_path: Path) -> None:
    protocol = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
    protocol["composition"]["weights"] = [1.0, 1.0]
    stale_hash = tmp_path / "stale-hash.json"
    stale_hash.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration hash"):
        load_protocol(stale_hash)

    payload = {key: value for key, value in protocol.items() if key != "config_sha256"}
    protocol["config_sha256"] = canonical_sha256(payload)
    invalid_composition = tmp_path / "invalid-composition.json"
    invalid_composition.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="composition definition"):
        load_protocol(invalid_composition)


def test_source_compatibility_normalizes_target_module_order(tmp_path: Path) -> None:
    modules = [
        Q_PROJ,
        "base_model.model.model.layers.1.mlp.down_proj",
    ]
    generic_tensors: dict[str, torch.Tensor] = {}
    targeted_tensors: dict[str, torch.Tensor] = {}
    for index, module in enumerate(modules, start=1):
        generic_tensors[f"{module}.lora_A.weight"] = torch.full((2, 3), float(index))
        generic_tensors[f"{module}.lora_B.weight"] = torch.full((4, 2), float(index))
        targeted_tensors[f"{module}.lora_A.weight"] = torch.full((2, 3), -float(index))
        targeted_tensors[f"{module}.lora_B.weight"] = torch.full((4, 2), float(index + 1))

    generic_path = tmp_path / "generic"
    targeted_path = tmp_path / "targeted"
    _write_adapter(
        generic_path,
        tensors=generic_tensors,
        rank=2,
        alpha=4,
        target_modules=["q_proj", "down_proj"],
    )
    _write_adapter(
        targeted_path,
        tensors=targeted_tensors,
        rank=2,
        alpha=4,
        target_modules=["down_proj", "q_proj"],
    )
    generic = inspect_adapter(generic_path)
    targeted = inspect_adapter(targeted_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path, generic=generic, targeted=targeted)

    result = validate_source_compatibility(
        generic=generic,
        targeted=targeted,
        protocol=load_protocol(protocol_path),
    )

    assert result["gate_passed"] is True
    assert generic["target_modules"] == targeted["target_modules"] == ["down_proj", "q_proj"]
    assert result["source_inventory_sha256"] == protocol["source_contract"]["inventory_sha256"]


def test_source_compatibility_fails_closed_on_contract_mismatches(tmp_path: Path) -> None:
    generic_path, targeted_path, _, protocol = _tiny_sources(tmp_path)
    generic = inspect_adapter(generic_path)
    targeted = inspect_adapter(targeted_path)
    mismatches = {
        "directory_sha256": "0" * 64,
        "rank": 3,
        "alpha": 8,
        "dropout": 0.0,
        "bias": "all",
        "modules_to_save": ["lm_head"],
        "target_modules": ["v_proj"],
        "base_model_name_or_path": "snapshots/wrong-revision",
        "saved_tensor_count": 4,
        "lora_module_count": 2,
        "use_dora": True,
    }

    for field, value in mismatches.items():
        incompatible = dict(targeted)
        incompatible[field] = value
        with pytest.raises(ValueError, match="compatibility failed"):
            validate_source_compatibility(
                generic=generic,
                targeted=incompatible,
                protocol=protocol,
            )

    wrong_inventory = copy.deepcopy(protocol)
    wrong_inventory["source_contract"]["inventory_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="inventory hash"):
        validate_source_compatibility(
            generic=generic,
            targeted=targeted,
            protocol=wrong_inventory,
        )


def test_tensor_inventory_requires_exact_a_b_pairing() -> None:
    a = {"name": f"{Q_PROJ}.lora_A.weight", "shape": [2, 3], "dtype": "torch.float32"}
    b = {"name": f"{Q_PROJ}.lora_B.weight", "shape": [4, 2], "dtype": "torch.float32"}

    assert set(_paired_modules([a, b])[Q_PROJ]) == {"A", "B"}
    with pytest.raises(ValueError, match="exactly one A and one B"):
        _paired_modules([a])
    with pytest.raises(ValueError, match="duplicated LoRA A"):
        _paired_modules([a, a, b])
    with pytest.raises(ValueError, match="unexpected adapter tensor key"):
        _paired_modules([{"name": "base.weight", "shape": [1], "dtype": "torch.float32"}])


def test_dense_analysis_uses_exact_targeted_minus_generic_direction(tmp_path: Path) -> None:
    generic_path, targeted_path, protocol_path, _ = _tiny_sources(tmp_path)

    result = analyze_dense_updates(
        generic_path=generic_path,
        targeted_path=targeted_path,
        protocol_path=protocol_path,
    )

    aggregate = result["aggregate"]
    assert aggregate["module_count"] == 1
    assert aggregate["generic_update_frobenius_norm"] == pytest.approx(math.sqrt(8.0))
    assert aggregate["targeted_update_frobenius_norm"] == pytest.approx(math.sqrt(8.0))
    assert aggregate["contrastive_update_frobenius_norm"] == pytest.approx(math.sqrt(32.0))
    assert aggregate["generic_targeted_cosine_similarity"] == pytest.approx(-1.0)
    assert aggregate["contrastive_to_targeted_norm_ratio"] == pytest.approx(2.0)
    assert aggregate["contrastive_to_generic_norm_ratio"] == pytest.approx(2.0)
    assert aggregate["maximum_absolute_contrastive_value"] == pytest.approx(4.0)
    assert result["contrastive_definition"] == {
        "formula": "delta_targeted_minus_delta_generic",
        "targeted_weight": 1.0,
        "generic_weight": -1.0,
        "source_scaling": 2.0,
    }
    assert result["per_module"][0]["generic_targeted_cosine_similarity"] == pytest.approx(-1.0)
    assert result["all_values_finite"] is True
    assert result["sealed_final_accessed"] is False


def test_output_path_must_not_equal_contain_or_be_within_sources(tmp_path: Path) -> None:
    generic_path, targeted_path, protocol_path, _ = _tiny_sources(tmp_path)
    raw_path = tmp_path / "raw.json"
    prohibited = (generic_path, generic_path / "nested-output", tmp_path)

    for output_parent in prohibited:
        with pytest.raises(ValueError, match="overlaps protected"):
            construct_task_vector(
                generic_path=generic_path,
                targeted_path=targeted_path,
                model_path=tmp_path / "unused-model",
                protocol_path=protocol_path,
                output_parent=output_parent,
                raw_path=raw_path,
                evidence_path=tmp_path / "evidence.json",
            )


@pytest.mark.parametrize("target_name", ["raw", "evidence"])
def test_every_construct_write_target_is_protected(tmp_path: Path, target_name: str) -> None:
    generic_path, targeted_path, protocol_path, _ = _tiny_sources(tmp_path)
    targets = {
        "raw": generic_path / "raw.json",
        "evidence": targeted_path / "evidence.json",
    }

    with pytest.raises(ValueError, match="overlaps protected"):
        construct_task_vector(
            generic_path=generic_path,
            targeted_path=targeted_path,
            model_path=tmp_path / "base",
            protocol_path=protocol_path,
            output_parent=tmp_path / "safe-output",
            raw_path=targets[target_name] if target_name == "raw" else tmp_path / "raw.json",
            evidence_path=(
                targets[target_name] if target_name == "evidence" else tmp_path / "evidence.json"
            ),
        )


def test_construct_write_targets_cannot_overlap_each_other(tmp_path: Path) -> None:
    generic_path, targeted_path, protocol_path, _ = _tiny_sources(tmp_path)
    output_parent = tmp_path / "composed"

    with pytest.raises(ValueError, match="write targets overlap"):
        construct_task_vector(
            generic_path=generic_path,
            targeted_path=targeted_path,
            model_path=tmp_path / "base",
            protocol_path=protocol_path,
            output_parent=output_parent,
            raw_path=output_parent / "raw.json",
            evidence_path=tmp_path / "evidence.json",
        )


def _full_module_names() -> list[str]:
    return [
        (
            f"base_model.model.model.layers.{layer}."
            f"{'self_attn' if projection in {'q_proj', 'k_proj', 'v_proj', 'o_proj'} else 'mlp'}."
            f"{projection}"
        )
        for layer in range(28)
        for projection in TARGET_MODULES
    ]


def test_saved_rank_32_cat_factors_are_dense_equivalent(tmp_path: Path) -> None:
    generic_tensors: dict[str, torch.Tensor] = {}
    targeted_tensors: dict[str, torch.Tensor] = {}
    contrastive_tensors: dict[str, torch.Tensor] = {}
    for index, module in enumerate(_full_module_names(), start=1):
        generic_a = torch.zeros((16, 1), dtype=torch.float32)
        generic_b = torch.zeros((1, 16), dtype=torch.float32)
        targeted_a = torch.zeros((16, 1), dtype=torch.float32)
        targeted_b = torch.zeros((1, 16), dtype=torch.float32)
        generic_a[0, 0] = 1.0
        generic_b[0, 0] = float(index) / 1000.0
        targeted_a[0, 0] = 1.0
        targeted_b[0, 0] = float(index + 1) / 500.0
        generic_tensors[f"{module}.lora_A.weight"] = generic_a
        generic_tensors[f"{module}.lora_B.weight"] = generic_b
        targeted_tensors[f"{module}.lora_A.weight"] = targeted_a
        targeted_tensors[f"{module}.lora_B.weight"] = targeted_b
        contrastive_tensors[f"{module}.lora_A.weight"] = torch.cat(
            (2.0 * targeted_a, -2.0 * generic_a), dim=0
        )
        contrastive_tensors[f"{module}.lora_B.weight"] = torch.cat((targeted_b, generic_b), dim=1)

    generic_path = tmp_path / "generic"
    targeted_path = tmp_path / "targeted"
    contrastive_path = tmp_path / "contrastive"
    targets = list(TARGET_MODULES)
    _write_adapter(
        generic_path,
        tensors=generic_tensors,
        rank=16,
        alpha=32,
        target_modules=targets,
    )
    _write_adapter(
        targeted_path,
        tensors=targeted_tensors,
        rank=16,
        alpha=32,
        target_modules=list(reversed(targets)),
    )
    _write_adapter(
        contrastive_path,
        tensors=contrastive_tensors,
        rank=32,
        alpha=32,
        target_modules=targets,
    )

    result = _verify_saved_dense_equivalence(
        generic_path=generic_path,
        targeted_path=targeted_path,
        contrastive_path=contrastive_path,
        maximum_tolerance=1e-5,
        relative_tolerance=1e-5,
    )

    assert result["gate_passed"] is True
    assert result["module_count"] == 196
    assert result["contrastive_rank"] == 32
    assert result["contrastive_alpha"] == 32
    assert result["maximum_absolute_error"] <= 1e-5
    assert result["relative_frobenius_error"] <= 1e-5
    assert all(item["passed"] for item in result["per_module"])


def test_materialization_uses_exact_disk_factors_not_rounded_loaded_sources(
    tmp_path: Path,
) -> None:
    class FakeModel:
        def __init__(self, module: torch.nn.Module) -> None:
            self.module = module

        def named_modules(self) -> list[tuple[str, torch.nn.Module]]:
            return [(Q_PROJ, self.module)]

    generic_path = tmp_path / "generic"
    targeted_path = tmp_path / "targeted"
    exact_generic_a = torch.tensor([[1.0003, -0.5003]], dtype=torch.float32)
    exact_generic_b = torch.tensor([[0.2503], [-0.7503]], dtype=torch.float32)
    exact_targeted_a = torch.tensor([[-1.0003, 0.1253]], dtype=torch.float32)
    exact_targeted_b = torch.tensor([[0.5003], [0.8753]], dtype=torch.float32)
    _write_adapter(
        generic_path,
        tensors={
            f"{Q_PROJ}.lora_A.weight": exact_generic_a,
            f"{Q_PROJ}.lora_B.weight": exact_generic_b,
        },
        rank=1,
        alpha=2,
        target_modules=["q_proj"],
    )
    _write_adapter(
        targeted_path,
        tensors={
            f"{Q_PROJ}.lora_A.weight": exact_targeted_a,
            f"{Q_PROJ}.lora_B.weight": exact_targeted_b,
        },
        rank=1,
        alpha=2,
        target_modules=["q_proj"],
    )

    module = torch.nn.Module()
    module.lora_A = torch.nn.ModuleDict(
        {
            GENERIC_NAME: torch.nn.Linear(2, 1, bias=False),
            TARGETED_NAME: torch.nn.Linear(2, 1, bias=False),
            CONTRASTIVE_NAME: torch.nn.Linear(2, 2, bias=False),
        }
    )
    module.lora_B = torch.nn.ModuleDict(
        {
            GENERIC_NAME: torch.nn.Linear(1, 2, bias=False),
            TARGETED_NAME: torch.nn.Linear(1, 2, bias=False),
            CONTRASTIVE_NAME: torch.nn.Linear(2, 2, bias=False),
        }
    )
    module.scaling = {GENERIC_NAME: 2.0, TARGETED_NAME: 2.0, CONTRASTIVE_NAME: 1.0}
    module.merged = False
    with torch.no_grad():
        module.lora_A[GENERIC_NAME].weight.copy_(exact_generic_a.half().float())
        module.lora_B[GENERIC_NAME].weight.copy_(exact_generic_b.half().float())
        module.lora_A[TARGETED_NAME].weight.copy_(exact_targeted_a.half().float())
        module.lora_B[TARGETED_NAME].weight.copy_(exact_targeted_b.half().float())

    modules = _materialize_exact_fp32_cat(
        FakeModel(module),
        torch,
        generic_path=generic_path,
        targeted_path=targeted_path,
    )

    assert [name for name, _ in modules] == [Q_PROJ]
    assert torch.equal(
        module.lora_A[CONTRASTIVE_NAME].weight,
        torch.cat((2.0 * exact_targeted_a, -2.0 * exact_generic_a), dim=0),
    )
    assert torch.equal(
        module.lora_B[CONTRASTIVE_NAME].weight,
        torch.cat((exact_targeted_b, exact_generic_b), dim=1),
    )
