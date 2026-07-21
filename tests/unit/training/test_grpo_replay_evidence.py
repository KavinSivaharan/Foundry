from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy
import pytest
import torch

from foundry.training import grpo_replay_evidence as evidence


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _generation(
    *,
    completion_suffix: str = "",
    group_id: str = "compat-group-001",
    source_kind: Literal["synthetic", "base_replay"] = "synthetic",
) -> evidence.GenerationEvidence:
    return evidence.capture_generation_evidence(
        group_id=group_id,
        source_kind=source_kind,
        prompt_sha256=_hash(f"prompt-{group_id}"),
        generated_token_ids=torch.tensor([[10, 11, 2], [20, 21, 2], [30, 31, 2], [40, 41, 2]]),
        decoded_completions=[f"alpha{completion_suffix}", "beta", "gamma", "delta"],
        completion_token_lengths=[3, 3, 3, 3],
        truncation_flags=[False, True, False, False],
        rng_before_sha256=_hash(f"rng-before-{group_id}"),
        rng_after_sha256=_hash(f"rng-after-{group_id}"),
        reward_components=[
            {"correctness": 1.0, "truncated": False, "total": 1.15},
            {"correctness": 0.0, "truncated": True, "total": -0.1},
            {"correctness": 1.0, "truncated": False, "total": 1.1},
            {"correctness": 0.0, "truncated": False, "total": 0.0},
        ],
        warning_messages=["deterministic warning"],
        reference_logprobs=torch.tensor([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [-7.0, -8.0]]),
        policy_logprobs=torch.tensor([[-0.9, -1.9], [-2.9, -3.9], [-4.9, -5.9], [-6.9, -7.9]]),
        per_token_kl=torch.tensor([[0.1, 0.1]] * 4),
    )


def _state(label: str) -> dict[str, object]:
    value = evidence.state_tree_evidence(
        {"label": label, "tensor": torch.tensor([1.0, 2.0])}, label=label
    )
    return value


def _gradient_state(label: str) -> dict[str, object]:
    return {
        **_state(label),
        "frozen_gradient_count": 0,
        "lora_gradient_count": 1,
    }


def _step(step: int, *, completion_suffix: str = "") -> evidence.CompatibilityStepEvidence:
    return evidence.build_compatibility_step_evidence(
        step=step,
        generation=_generation(
            completion_suffix=completion_suffix,
            group_id=f"synthetic-{step}",
        ),
        loss=0.25 / step,
        loss_tensor=torch.tensor(0.25 / step),
        mean_kl=0.01 * step,
        mean_kl_tensor=torch.tensor(0.01 * step),
        rng_before=_state(f"rng-before-{step}"),
        rng_after=_state(f"rng-after-{step}"),
        lora_before=_state(f"lora-before-{step}"),
        lora_after=_state(f"lora-after-{step}"),
        gradients_after_backward=_gradient_state(f"gradients-{step}"),
        gradients_after_clipping=_gradient_state(f"clipped-gradients-{step}"),
        optimizer_before=_state(f"optimizer-before-{step}"),
        optimizer_after=_state(f"optimizer-after-{step}"),
        scheduler_before=_state(f"scheduler-before-{step}"),
        scheduler_after=_state(f"scheduler-after-{step}"),
        strict_mode_evidence={
            "after_backward": True,
            "before_optimizer": True,
            "after_optimizer": True,
            "after_scheduler": True,
        },
    )


def _two_step_packet(*, completion_suffix: str = "") -> dict[str, object]:
    return evidence.build_two_step_packet(
        run_contract={
            "seed": 20260720,
            "execution_sha256": _hash("execution"),
            "trained_model_released_before_reload": True,
            "pre_reload_memory_gate_passed": True,
        },
        steps=[_step(1, completion_suffix=completion_suffix), _step(2)],
        replay_generation=_generation(group_id="replay", source_kind="base_replay"),
        initial_lora={
            **_state("initial-lora"),
            "lora_state_sha256": _hash("initial-lora"),
            "lora_tensor_state_sha256": _hash("initial-lora-tensors"),
        },
        final_lora={
            **_state("final-lora"),
            "lora_state_sha256": _hash("final-lora"),
            "lora_tensor_state_sha256": _hash("final-lora-tensors"),
        },
        reloaded_lora={
            **_state("reloaded-lora"),
            "lora_state_sha256": _hash("reloaded-lora"),
            "lora_tensor_state_sha256": _hash("final-lora-tensors"),
        },
        base_before={**_state("base-before"), "base_parameter_state_sha256": _hash("base")},
        base_after={**_state("base-after"), "base_parameter_state_sha256": _hash("base")},
        reloaded_base={**_state("base-reload"), "base_parameter_state_sha256": _hash("base")},
        final_optimizer=_state("final-optimizer"),
        final_scheduler=_state("final-scheduler"),
        adapter_artifact_sha256=_hash("adapter"),
        adapter_directory_sha256=_hash("adapter-directory"),
    )


def test_tensor_and_state_tree_evidence_are_exact_and_order_independent() -> None:
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    first = evidence.tensor_evidence(tensor)
    second = evidence.tensor_evidence(tensor.clone())
    changed = evidence.tensor_evidence(torch.tensor([[1.0, 3.0]], dtype=torch.float32))

    assert first.sha256 == second.sha256
    assert first.sha256 != changed.sha256
    assert first.shape == (1, 2)
    left = evidence.state_tree_evidence({"b": tensor, "a": 1.0}, label="state")
    right = evidence.state_tree_evidence({"a": 1.0, "b": tensor.clone()}, label="state")
    assert left == right
    assert left["tensor_count"] == 1


def test_state_tree_rejects_nonfinite_and_unordered_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        evidence.state_tree_evidence({"loss": float("nan")}, label="bad")
    with pytest.raises(TypeError, match="unsupported replay-state"):
        evidence.state_tree_evidence({"values": {1, 2}}, label="bad")


def test_rng_capture_reconstructs_after_state_restoration() -> None:
    python_state = __import__("random").getstate()
    torch_state = torch.get_rng_state()
    numpy_state = numpy.random.get_state()
    first = evidence.capture_rng_state(torch, numpy_random=numpy.random)
    __import__("random").random()
    torch.rand(2)
    numpy.random.random()
    __import__("random").setstate(python_state)
    torch.set_rng_state(torch_state)
    numpy.random.set_state(numpy_state)
    second = evidence.capture_rng_state(torch, numpy_random=numpy.random)
    assert first["rng_sha256"] == second["rng_sha256"]


class _TinyLoRAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.base_weight = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)


def test_lora_and_gradient_capture_records_frozen_gradient_violation() -> None:
    model = _TinyLoRAModel()
    before = evidence.capture_lora_state(model)
    model.lora_a.grad = torch.tensor([0.5, 0.25])
    gradients = evidence.capture_gradient_state(model)
    assert before["parameter_count"] == 1
    assert gradients["lora_gradient_count"] == 1
    assert gradients["frozen_gradient_count"] == 0

    model.base_weight.grad = torch.tensor([0.75])
    violated = evidence.capture_gradient_state(model)
    assert violated["frozen_gradient_names"] == ["base_weight"]


def test_lora_tensor_identity_ignores_reload_trainability_but_not_bytes() -> None:
    trainable = _TinyLoRAModel()
    reloaded = _TinyLoRAModel()
    reloaded.lora_a.requires_grad_(False)
    first = evidence.capture_lora_state(trainable)
    second = evidence.capture_lora_state(reloaded)
    assert first["lora_state_sha256"] != second["lora_state_sha256"]
    assert first["lora_tensor_state_sha256"] == second["lora_tensor_state_sha256"]
    reloaded.lora_a.data.add_(1.0)
    assert (
        evidence.capture_lora_state(reloaded)["lora_tensor_state_sha256"]
        != first["lora_tensor_state_sha256"]
    )


def test_base_parameter_capture_is_byte_exact_stable_and_excludes_lora() -> None:
    model = _TinyLoRAModel()
    first = evidence.capture_base_parameter_state(model)
    second = evidence.capture_base_parameter_state(model)

    assert first == second
    assert first["parameter_count"] == 1
    assert first["total_numel"] == 1
    assert first["total_bytes"] == 4
    rows = first["parameters"]
    assert isinstance(rows, list)
    assert [row["name"] for row in rows] == ["base_weight"]
    assert rows[0]["dtype"] == "torch.float32"
    assert rows[0]["shape"] == [1]
    assert "lora_a" not in json.dumps(first, sort_keys=True)

    model.lora_a.data.add_(10.0)
    assert evidence.capture_base_parameter_state(model) == first
    model.base_weight.data.add_(1.0)
    changed = evidence.capture_base_parameter_state(model)
    assert changed["base_parameter_state_sha256"] != first["base_parameter_state_sha256"]


class _OrderedBaseModel:
    def __init__(self, rows: list[tuple[object, object]]) -> None:
        self.rows = rows

    def named_parameters(self, *, remove_duplicate: bool = True) -> list[tuple[object, object]]:
        del remove_duplicate
        return self.rows


class _TensorWithoutDevice:
    def __init__(self) -> None:
        self.value = torch.tensor([1.0])
        self.requires_grad = False

    @property
    def shape(self) -> torch.Size:
        return self.value.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.value.dtype

    def detach(self) -> _TensorWithoutDevice:
        return self

    def cpu(self) -> _TensorWithoutDevice:
        return self

    def contiguous(self) -> _TensorWithoutDevice:
        return self

    def numpy(self) -> numpy.ndarray[Any, Any]:
        return self.value.numpy()

    def numel(self) -> int:
        return self.value.numel()


def test_base_parameter_capture_sorts_names_and_binds_name_dtype_shape_and_bytes() -> None:
    alpha = torch.nn.Parameter(torch.tensor([1.0, 2.0]), requires_grad=False)
    zeta = torch.nn.Parameter(torch.tensor([[3]], dtype=torch.int64), requires_grad=False)
    left = evidence.capture_base_parameter_state(
        _OrderedBaseModel([("zeta", zeta), ("alpha", alpha)])
    )
    right = evidence.capture_base_parameter_state(
        _OrderedBaseModel([("alpha", alpha.clone().detach()), ("zeta", zeta.clone().detach())])
    )
    rows = left["parameters"]
    assert isinstance(rows, list)
    assert [row["name"] for row in rows] == ["alpha", "zeta"]
    assert left["base_parameter_state_sha256"] == right["base_parameter_state_sha256"]

    renamed = evidence.capture_base_parameter_state(_OrderedBaseModel([("beta", alpha)]))
    reshaped = evidence.capture_base_parameter_state(
        _OrderedBaseModel([("alpha", alpha.reshape(1, 2))])
    )
    retyped = evidence.capture_base_parameter_state(
        _OrderedBaseModel([("alpha", alpha.detach().to(torch.float64))])
    )
    assert renamed["base_parameter_state_sha256"] != left["base_parameter_state_sha256"]
    assert reshaped["base_parameter_state_sha256"] != left["base_parameter_state_sha256"]
    assert retyped["base_parameter_state_sha256"] != left["base_parameter_state_sha256"]


@pytest.mark.parametrize(
    ("model", "error", "message"),
    [
        (object(), TypeError, "named_parameters"),
        (
            _OrderedBaseModel([("base", torch.tensor([1.0])), ("base", torch.tensor([2.0]))]),
            ValueError,
            "duplicate",
        ),
        (_OrderedBaseModel([(1, torch.tensor([1.0]))]), TypeError, "non-text"),
        (_OrderedBaseModel([("", torch.tensor([1.0]))]), ValueError, "non-empty"),
        (_OrderedBaseModel([("base", object())]), TypeError, "detach"),
        (_OrderedBaseModel([("base", _TensorWithoutDevice())]), TypeError, "device metadata"),
        (_OrderedBaseModel([("lora_A.weight", torch.tensor([1.0]))]), ValueError, "no non-LoRA"),
    ],
)
def test_base_parameter_capture_fails_closed_on_duplicate_or_unsupported_state(
    model: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        evidence.capture_base_parameter_state(model)


def test_optimizer_and_scheduler_capture_full_state_changes() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    optimizer_before = evidence.capture_optimizer_state(optimizer)
    scheduler_before = evidence.capture_scheduler_state(scheduler)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    scheduler.step()
    optimizer_after = evidence.capture_optimizer_state(optimizer)
    scheduler_after = evidence.capture_scheduler_state(scheduler)
    assert optimizer_before["state_sha256"] != optimizer_after["state_sha256"]
    assert scheduler_before["state_sha256"] != scheduler_after["state_sha256"]
    assert isinstance(optimizer_after["tensor_count"], int)
    assert optimizer_after["tensor_count"] > 0


def test_generation_evidence_hashes_content_without_retaining_it() -> None:
    value = _generation()
    record = value.as_dict()
    serialized = json.dumps(record, sort_keys=True)
    assert "alpha" not in serialized
    assert "beta" not in serialized
    assert record["completion_count"] == 4
    assert record["completion_token_lengths"] == [3, 3, 3, 3]
    assert record["truncation_flags"] == [False, True, False, False]
    assert record["warning_count"] == 1
    assert value.evidence_sha256 == _generation().evidence_sha256
    assert value.evidence_sha256 != _generation(completion_suffix=" changed").evidence_sha256


def test_generation_evidence_fails_on_column_or_logprob_mismatch() -> None:
    arguments: dict[str, Any] = {
        "group_id": "group",
        "source_kind": "synthetic",
        "prompt_sha256": _hash("prompt"),
        "generated_token_ids": [[1, 2]],
        "decoded_completions": ["answer"],
        "completion_token_lengths": [2],
        "truncation_flags": [False],
        "rng_before_sha256": _hash("rng-before"),
        "rng_after_sha256": _hash("rng-after"),
        "reward_components": [{"total": 1.0}],
    }
    with pytest.raises(ValueError, match="equal counts"):
        evidence.capture_generation_evidence(**{**arguments, "decoded_completions": []})
    with pytest.raises(ValueError, match="supplied together"):
        evidence.capture_generation_evidence(**arguments, reference_logprobs=torch.tensor([1.0]))


def test_generation_evidence_accepts_only_one_warning_representation() -> None:
    warning_hash = _hash("known deterministic warning")
    arguments: dict[str, Any] = {
        "group_id": "group",
        "source_kind": "synthetic",
        "prompt_sha256": _hash("prompt"),
        "generated_token_ids": [[1, 2]],
        "decoded_completions": ["answer"],
        "completion_token_lengths": [2],
        "truncation_flags": [False],
        "rng_before_sha256": _hash("rng-before"),
        "rng_after_sha256": _hash("rng-after"),
        "reward_components": [{"total": 1.0}],
    }
    value = evidence.capture_generation_evidence(**arguments, warning_sha256s=[warning_hash])
    assert value.warning_sha256s == (warning_hash,)
    with pytest.raises(ValueError, match="mutually exclusive"):
        evidence.capture_generation_evidence(
            **arguments,
            warning_messages=["known deterministic warning"],
            warning_sha256s=[warning_hash],
        )


def test_generation_only_packet_is_self_hashed_and_tamper_evident() -> None:
    generations = [
        _generation(group_id="synthetic-1"),
        _generation(group_id="synthetic-2"),
        _generation(group_id="replay-1", source_kind="base_replay"),
    ]
    packet = evidence.build_generation_only_packet(
        run_contract={"seed": 20260720, "model_sha256": _hash("model")},
        generations=generations,
        rng_before=_state("rng-before"),
        rng_after=_state("rng-after"),
        lora_state=_state("lora"),
    )
    assert (
        evidence.validate_replay_packet(packet, expected_kind="generation_only")
        == packet["packet_sha256"]
    )
    tampered = dict(packet)
    tampered["rng_after"] = _state("changed")
    with pytest.raises(ValueError, match="self-hash"):
        evidence.validate_replay_packet(tampered, expected_kind="generation_only")


def test_generation_only_packet_enforces_frozen_three_group_composition() -> None:
    common: dict[str, Any] = {
        "run_contract": {"seed": 20260720},
        "rng_before": _state("rng-before"),
        "rng_after": _state("rng-after"),
        "lora_state": _state("lora"),
    }
    with pytest.raises(ValueError, match="exactly three"):
        evidence.build_generation_only_packet(**common, generations=[_generation(group_id="one")])
    with pytest.raises(ValueError, match="ordered as two synthetic"):
        evidence.build_generation_only_packet(
            **common,
            generations=[
                _generation(group_id="replay", source_kind="base_replay"),
                _generation(group_id="synthetic-1"),
                _generation(group_id="synthetic-2"),
            ],
        )


def test_two_step_packet_requires_exact_order_and_detects_any_difference() -> None:
    packet = _two_step_packet()
    same = _two_step_packet()
    assert (
        evidence.assert_exact_replay([packet, same], expected_kind="two_step_compatibility")
        == packet["packet_sha256"]
    )

    with pytest.raises(ValueError, match="ordered steps"):
        evidence.build_two_step_packet(
            run_contract={"seed": 1},
            steps=[_step(2), _step(1)],
            replay_generation=_generation(group_id="replay", source_kind="base_replay"),
            initial_lora={
                **_state("initial"),
                "lora_state_sha256": _hash("initial"),
                "lora_tensor_state_sha256": _hash("initial-tensors"),
            },
            final_lora={
                **_state("final"),
                "lora_state_sha256": _hash("final"),
                "lora_tensor_state_sha256": _hash("final-tensors"),
            },
            reloaded_lora={
                **_state("reload"),
                "lora_state_sha256": _hash("reload"),
                "lora_tensor_state_sha256": _hash("final-tensors"),
            },
            base_before={**_state("base-before"), "base_parameter_state_sha256": _hash("base")},
            base_after={**_state("base-after"), "base_parameter_state_sha256": _hash("base")},
            reloaded_base={**_state("base-reload"), "base_parameter_state_sha256": _hash("base")},
            final_optimizer=_state("optimizer"),
            final_scheduler=_state("scheduler"),
            adapter_artifact_sha256=_hash("adapter"),
            adapter_directory_sha256=_hash("directory"),
        )

    changed = _two_step_packet(completion_suffix=" changed")
    with pytest.raises(RuntimeError, match="decoded_completion_sha256"):
        evidence.assert_exact_replay([packet, changed], expected_kind="two_step_compatibility")

    missing_release_proof = dict(packet)
    missing_release_proof["run_contract"] = {
        "seed": 20260720,
        "execution_sha256": _hash("execution"),
    }
    missing_release_proof.pop("packet_sha256")
    missing_release_proof["packet_sha256"] = evidence.canonical_sha256(missing_release_proof)
    with pytest.raises(ValueError, match="trained_model_released_before_reload"):
        evidence.validate_replay_packet(
            missing_release_proof,
            expected_kind="two_step_compatibility",
        )


def test_step_rejects_nonfinite_loss_and_out_of_range_index() -> None:
    values: dict[str, Any] = {
        "step": 1,
        "generation": _generation(),
        "loss": 0.2,
        "loss_tensor": torch.tensor(0.2),
        "mean_kl": 0.01,
        "mean_kl_tensor": torch.tensor(0.01),
        "rng_before": _state("a"),
        "rng_after": _state("b"),
        "lora_before": _state("c"),
        "lora_after": _state("d"),
        "gradients_after_backward": _gradient_state("e"),
        "gradients_after_clipping": _gradient_state("e-clipped"),
        "optimizer_before": _state("f"),
        "optimizer_after": _state("g"),
        "scheduler_before": _state("h"),
        "scheduler_after": _state("i"),
        "strict_mode_evidence": {
            "after_backward": True,
            "before_optimizer": True,
            "after_optimizer": True,
            "after_scheduler": True,
        },
    }
    with pytest.raises(ValueError, match="finite"):
        evidence.build_compatibility_step_evidence(**{**values, "loss": float("inf")})
    with pytest.raises(ValueError, match="1 or 2"):
        evidence.build_compatibility_step_evidence(**{**values, "step": 3})


def test_fresh_process_packets_round_trip_and_compare(tmp_path: Path) -> None:
    packet = _two_step_packet()
    paths = [tmp_path / f"process-{index}.json" for index in range(3)]
    for path in paths:
        evidence.write_replay_packet_new(path, packet, kind="two_step_compatibility")
    assert (
        evidence.compare_fresh_process_packets(paths, expected_kind="two_step_compatibility")
        == packet["packet_sha256"]
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        evidence.write_replay_packet_new(paths[0], packet, kind="two_step_compatibility")
    with pytest.raises(ValueError, match="distinct"):
        evidence.compare_fresh_process_packets(
            [paths[0], paths[0]], expected_kind="two_step_compatibility"
        )


def test_fresh_process_comparison_rejects_tampering(tmp_path: Path) -> None:
    first = _two_step_packet()
    second = _two_step_packet(completion_suffix=" changed")
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    evidence.write_replay_packet_new(paths[0], first, kind="two_step_compatibility")
    evidence.write_replay_packet_new(paths[1], second, kind="two_step_compatibility")
    with pytest.raises(RuntimeError, match="differs"):
        evidence.compare_fresh_process_packets(paths, expected_kind="two_step_compatibility")
