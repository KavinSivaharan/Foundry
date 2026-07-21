from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from foundry.training import grpo_two_step_runtime as runtime
from foundry.training.config import canonical_sha256


class _StrictTorch:
    def __init__(self, *, strict: bool) -> None:
        self.strict = strict

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.strict

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return not self.strict


def test_strict_guard_and_frozen_group_contract_fail_closed() -> None:
    runtime._require_strict(_StrictTorch(strict=True), "backward")
    with pytest.raises(RuntimeError, match="backward"):
        runtime._require_strict(_StrictTorch(strict=False), "backward")

    groups = [
        SimpleNamespace(group_id=value, source_kind=kind)
        for value, kind in zip(
            runtime.replay_runtime.FROZEN_REPLAY_GROUP_IDS,
            ("synthetic", "synthetic", "base_replay"),
            strict=True,
        )
    ]
    runtime._require_frozen_group_ids(groups)
    groups[1].group_id = "changed"
    with pytest.raises(ValueError, match="frozen G1 generic"):
        runtime._require_frozen_group_ids(groups)


def test_scheduler_position_uses_wrapper_state_dict_and_requires_integer() -> None:
    scheduler = SimpleNamespace(state_dict=lambda: {"last_epoch": 2})
    assert runtime._scheduler_last_epoch(scheduler) == 2
    with pytest.raises(RuntimeError, match="integer last_epoch"):
        runtime._scheduler_last_epoch(SimpleNamespace(state_dict=lambda: {"last_epoch": 2.0}))


def test_pre_reload_release_requires_collected_model_and_bounded_cuda_memory() -> None:
    class _Model:
        pass

    model = _Model()
    model_reference = weakref.ref(model)
    del model
    gc.collect()
    evidence = runtime._validate_pre_reload_release(
        model_reference=model_reference,
        before={
            "allocated_vram_bytes": 2 * 1024**3,
            "reserved_vram_bytes": 3 * 1024**3,
        },
        after={"allocated_vram_bytes": 64, "reserved_vram_bytes": 128},
    )
    assert evidence["trained_model_collected"] is True
    assert evidence["pre_reload_memory_gate_passed"] is True
    assert evidence["allocated_vram_drop_bytes"] == 2 * 1024**3 - 64


def test_pre_reload_release_fails_for_live_model_or_allocator_residue() -> None:
    class _Model:
        pass

    model = _Model()
    model_reference = weakref.ref(model)
    snapshot = {"allocated_vram_bytes": 0, "reserved_vram_bytes": 0}
    with pytest.raises(RuntimeError, match="strongly referenced"):
        runtime._validate_pre_reload_release(
            model_reference=model_reference,
            before=snapshot,
            after=snapshot,
        )
    del model
    gc.collect()
    with pytest.raises(RuntimeError, match="allocated VRAM"):
        runtime._validate_pre_reload_release(
            model_reference=model_reference,
            before={
                "allocated_vram_bytes": runtime.MAX_PRE_RELOAD_ALLOCATED_VRAM_BYTES + 1,
                "reserved_vram_bytes": runtime.MAX_PRE_RELOAD_RESERVED_VRAM_BYTES + 1,
            },
            after={
                "allocated_vram_bytes": runtime.MAX_PRE_RELOAD_ALLOCATED_VRAM_BYTES + 1,
                "reserved_vram_bytes": runtime.MAX_PRE_RELOAD_RESERVED_VRAM_BYTES + 1,
            },
        )


class _RecorderProbe:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.policy: torch.Tensor | None = None
        self.loss_record: dict[str, Any] | None = None

    def start_generation(self) -> None:
        self.events.append("generation-start")

    def finish_generation(self, result: Any) -> None:
        assert result == {"completion_ids": "tokens"}
        self.events.append("generation-finish")

    def begin_policy_capture(self) -> None:
        self.events.append("policy-start")

    def capture_policy(self, value: torch.Tensor) -> None:
        self.policy = value

    def end_policy_capture(self) -> torch.Tensor:
        self.events.append("policy-finish")
        assert self.policy is not None
        value = self.policy
        self.policy = None
        return value

    def abort_policy_capture(self) -> None:
        self.events.append("policy-abort")

    def record_loss(self, **kwargs: Any) -> None:
        self.loss_record = kwargs

    def after_backward(self, model: Any) -> None:
        del model
        self.events.append("after-backward")


class _StockTrainerProbe:
    def _generate_and_score_completions(self, inputs: Any) -> dict[str, str]:
        assert inputs == "prompt"
        return {"completion_ids": "tokens"}

    def _get_per_token_logps(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        return torch.tensor([[-1.0, -1.2]], dtype=torch.float32)

    def compute_loss(
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> torch.Tensor:
        del return_outputs, num_items_in_batch
        policy = self._get_per_token_logps(model, inputs)
        assert policy.shape == (1, 2)
        return torch.tensor(0.25, dtype=torch.float32)

    def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
        del model, inputs, num_items_in_batch
        return torch.tensor(0.25)


def test_trainer_wrapper_observes_stock_generation_loss_and_backward() -> None:
    recorder = _RecorderProbe()
    trainer_type = runtime.make_two_step_evidence_trainer(  # type: ignore[arg-type]
        _StockTrainerProbe, recorder
    )
    trainer_type._foundry_torch = torch
    trainer = trainer_type()

    assert trainer._generate_and_score_completions("prompt") == {"completion_ids": "tokens"}
    reference = torch.tensor([[-0.8, -1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    loss = trainer.compute_loss(
        object(),
        {"ref_per_token_logps": reference, "completion_mask": mask},
    )
    trainer.training_step(object(), {})

    assert torch.equal(loss, torch.tensor(0.25))
    assert recorder.events == [
        "generation-start",
        "generation-finish",
        "policy-start",
        "policy-finish",
        "after-backward",
    ]
    assert recorder.loss_record is not None
    expected_policy = torch.tensor([[-1.0, -1.2]])
    expected_delta = reference - expected_policy
    expected_kl = torch.exp(expected_delta) - expected_delta - 1
    assert torch.equal(recorder.loss_record["policy"], expected_policy)
    assert torch.allclose(recorder.loss_record["per_token_kl"], expected_kl)
    assert torch.allclose(recorder.loss_record["mean_kl_tensor"], expected_kl.mean())


class _CallbackRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def step_begin(self, **kwargs: Any) -> None:
        assert set(kwargs) == {"state", "model", "optimizer", "scheduler"}
        self.events.append("begin")

    def pre_optimizer(self, **kwargs: Any) -> None:
        assert set(kwargs) == {"model"}
        self.events.append("pre")

    def post_optimizer(self, **kwargs: Any) -> None:
        assert set(kwargs) == {"model", "optimizer"}
        self.events.append("post")

    def step_end(self, **kwargs: Any) -> None:
        assert set(kwargs) == {"state", "scheduler"}
        self.events.append("end")


def test_callback_maps_stock_lifecycle_boundaries_without_replacing_operations() -> None:
    recorder = _CallbackRecorder()
    callback = runtime.make_two_step_capture_callback(  # type: ignore[arg-type]
        object, recorder
    )
    state = SimpleNamespace(global_step=0)
    kwargs = {"model": object(), "optimizer": object(), "lr_scheduler": object()}
    control = object()
    assert callback.on_step_begin(object(), state, control, **kwargs) is control
    assert callback.on_pre_optimizer_step(object(), state, control, **kwargs) is control
    assert callback.on_optimizer_step(object(), state, control, **kwargs) is control
    assert callback.on_step_end(object(), state, control, **kwargs) is control
    assert recorder.events == ["begin", "pre", "post", "end"]


def _write_metadata(
    path: Path,
    *,
    packet_hash: str,
    process_hash: str,
) -> None:
    value: dict[str, object] = {
        "schema_version": runtime.TWO_STEP_RUNTIME_SCHEMA_VERSION,
        "runtime_id": runtime.TWO_STEP_RUNTIME_ID,
        "packet_sha256": packet_hash,
        "process_instance_sha256": process_hash,
        "process_command_sha256": canonical_sha256([process_hash]),
        "resource_measurement": {
            "reserved_vram_gate_passed": True,
            "peak_reserved_vram_bytes": 123,
            "trained_model_release": {
                "trained_model_collected": True,
                "pre_reload_memory_gate_passed": True,
            },
        },
    }
    value["metadata_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_combine_requires_two_distinct_processes_and_writes_content_free_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet_hash = "a" * 64
    packet_paths = [tmp_path / "one.packet", tmp_path / "two.packet"]
    metadata_paths = [tmp_path / "one.meta", tmp_path / "two.meta"]
    for path in packet_paths:
        path.write_bytes(b"same-packet")
    _write_metadata(metadata_paths[0], packet_hash=packet_hash, process_hash="b" * 64)
    _write_metadata(metadata_paths[1], packet_hash=packet_hash, process_hash="c" * 64)
    monkeypatch.setattr(
        runtime,
        "compare_fresh_process_packets",
        lambda paths, expected_kind: packet_hash,
    )

    summary_path = tmp_path / "summary.json"
    summary = runtime.combine_fresh_process_runs(
        packet_paths=packet_paths,
        metadata_paths=metadata_paths,
        summary_path=summary_path,
    )
    assert summary["processes"] == 2
    assert summary["optimizer_steps_per_process"] == 2
    assert summary["exact_replay_passed"] is True
    assert summary["compatibility_gate_passed"] is True
    assert summary["trained_model_release_passed"] is True
    assert summary["prompts_completions_answers_or_tensor_values_in_summary"] is False
    assert (
        json.loads(summary_path.read_text(encoding="utf-8"))["summary_sha256"]
        == (summary["summary_sha256"])
    )

    _write_metadata(
        tmp_path / "same.meta",
        packet_hash=packet_hash,
        process_hash="b" * 64,
    )
    with pytest.raises(ValueError, match="distinct process identities"):
        runtime.combine_fresh_process_runs(
            packet_paths=packet_paths,
            metadata_paths=[metadata_paths[0], tmp_path / "same.meta"],
            summary_path=tmp_path / "not-written.json",
        )


def test_metadata_is_self_hashed_and_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    _write_metadata(path, packet_hash="a" * 64, process_hash="b" * 64)
    assert runtime._load_metadata(path)["packet_sha256"] == "a" * 64
    value = json.loads(path.read_text(encoding="utf-8"))
    value["packet_sha256"] = "c" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        runtime._load_metadata(path)
