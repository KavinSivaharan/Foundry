from __future__ import annotations

import json
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from foundry.training import grpo_runtime as runtime
from foundry.training import grpo_schedule as schedule
from foundry.training.config import canonical_sha256


def _hash(label: str) -> str:
    return canonical_sha256({"fixture": label})


def _messages(label: str) -> tuple[schedule.PromptMessage, ...]:
    return (
        schedule.PromptMessage("system", "Answer the original fixture."),
        schedule.PromptMessage("user", f"Original fixture prompt {label}."),
    )


def _metadata(source_kind: str, label: str) -> dict[str, object]:
    if source_kind == "synthetic":
        return {
            "reward_kind": "synthetic",
            "canonical_final_answer": "1",
            "family": "multi_step_bookkeeping_or_omission",
            "mode": "fixture_mode",
            "difficulty": "easy",
            "output_contract_enabled": True,
            "verifier_metadata_sha256": _hash(f"verifier-{label}"),
            "provenance_sha256": _hash(f"provenance-{label}"),
        }
    return {
        "reward_kind": "base_replay",
        "section": "arithmetic",
        "skill": "fixture_skill",
        "kind": "exact_text",
        "expected": "OK",
        "scorer_sha256": _hash(f"scorer-{label}"),
        "provenance_sha256": _hash(f"provenance-{label}"),
    }


def _runtime_group(
    label: str,
    *,
    source_kind: str = "synthetic",
    position: int = 1,
) -> runtime.RuntimePromptGroup:
    messages = _messages(label)
    return runtime.RuntimePromptGroup(
        group_id=f"group-{label}",
        arm="generic_control",
        position=position,
        source_kind=source_kind,  # type: ignore[arg-type]
        source_id=f"source-{label}",
        category=(
            "multi_step_bookkeeping_or_omission" if source_kind == "synthetic" else "arithmetic"
        ),
        messages=messages,
        prompt_sha256=canonical_sha256([message.as_dict() for message in messages]),
        prompt_tokens=10,
        reward_metadata_json=json.dumps(
            _metadata(source_kind, label), sort_keys=True, separators=(",", ":")
        ),
    )


def _packet_and_manifest(arm: schedule.Arm) -> tuple[dict[str, object], dict[str, object]]:
    family_remaining = dict(schedule.SYNTHETIC_QUOTAS[arm])
    replay_index = 0
    groups: list[dict[str, object]] = []
    manifest_groups: list[dict[str, object]] = []
    for position in range(1, schedule.GROUPS_PER_ARM + 1):
        if position in schedule.REPLAY_POSITIONS:
            section = schedule.REPLAY_SECTION_ORDER[replay_index % 3]
            replay_index += 1
            source_kind = "base_replay"
            category = section
            metadata = _metadata(source_kind, f"{arm}-{position}")
            metadata["section"] = section
        else:
            family = next(name for name in schedule.FAMILY_ORDER if family_remaining[name] > 0)
            family_remaining[family] -= 1
            source_kind = "synthetic"
            category = family
            metadata = _metadata(source_kind, f"{arm}-{position}")
            metadata["family"] = family
        messages = _messages(f"{arm}-{position}")
        row: dict[str, object] = {
            "group_id": f"grpo-{arm}-g{position:03d}",
            "position": position,
            "source_kind": source_kind,
            "source_id": f"source-{arm}-{position:03d}",
            "category": category,
            "prompt_sha256": canonical_sha256([item.as_dict() for item in messages]),
            "prompt_tokens": 20 + position,
            "completions_per_group": 4,
            "messages": [item.as_dict() for item in messages],
            "reward_metadata": metadata,
        }
        groups.append(row)
        manifest_groups.append(
            {key: value for key, value in row.items() if key not in {"messages", "reward_metadata"}}
        )
    packet: dict[str, object] = {
        "schema_version": schedule.SCHEDULE_SCHEMA_VERSION,
        "schedule_id": schedule.SCHEDULE_ID,
        "seed": schedule.SCHEDULE_SEED,
        "arm": arm,
        "groups": groups,
    }
    packet_hash = canonical_sha256(packet)
    manifest: dict[str, object] = {
        "schema_version": schedule.SCHEDULE_SCHEMA_VERSION,
        "schedule_id": schedule.SCHEDULE_ID,
        "seed": schedule.SCHEDULE_SEED,
        "arm": arm,
        "groups_per_arm": 64,
        "synthetic_groups": 52,
        "replay_groups": 12,
        "completions_per_group": 4,
        "total_completions": 256,
        "prompt_packet_sha256": packet_hash,
        "prompts_or_outputs_in_manifest": False,
        "groups": manifest_groups,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return packet, manifest


def _write_pair(tmp_path: Path, arm: schedule.Arm) -> tuple[Path, Path]:
    packet, manifest = _packet_and_manifest(arm)
    tmp_path.mkdir(parents=True, exist_ok=True)
    packet_path = tmp_path / "packet.json"
    manifest_path = tmp_path / "manifest.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return packet_path, manifest_path


def test_runtime_schedule_is_strict_prompt_only_and_manifest_bound(tmp_path: Path) -> None:
    packet_path, manifest_path = _write_pair(tmp_path, "generic_control")
    loaded = runtime.load_runtime_schedule(
        packet_path, manifest_path, expected_arm="generic_control"
    )
    assert len(loaded.groups) == 64
    assert Counter(group.source_kind for group in loaded.groups) == {
        "synthetic": 52,
        "base_replay": 12,
    }
    assert [
        group.position for group in loaded.groups if group.source_kind == "base_replay"
    ] == list(schedule.REPLAY_POSITIONS)
    assert all(
        [item["role"] for item in group.policy_row()["prompt"]] == ["system", "user"]
        for group in loaded.groups
    )  # type: ignore[index]
    visible = json.dumps([group.policy_row()["prompt"] for group in loaded.groups])
    hidden = json.dumps([group.reward_metadata_json for group in loaded.groups])
    assert "canonical_final_answer" not in visible
    assert '"expected"' not in visible
    assert "canonical_final_answer" in hidden
    assert "expected" in hidden

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["groups"][0]["training_completion"] = "forbidden"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        runtime.load_runtime_schedule(packet_path, manifest_path, expected_arm="generic_control")


def test_runtime_schedule_rejects_prompt_or_manifest_tampering(tmp_path: Path) -> None:
    packet_path, manifest_path = _write_pair(tmp_path, "targeted")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["groups"][0]["messages"][1]["content"] += " Changed."
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(ValueError, match="packet or manifest identity"):
        runtime.load_runtime_schedule(packet_path, manifest_path, expected_arm="targeted")

    packet_path, manifest_path = _write_pair(tmp_path / "second", "targeted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["groups"][0]["position"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        runtime.load_runtime_schedule(packet_path, manifest_path, expected_arm="targeted")


def test_reward_callback_uses_hidden_metadata_and_exact_truncation_flags() -> None:
    synthetic = _runtime_group("synthetic")
    replay = _runtime_group("replay", source_kind="base_replay", position=2)
    groups = (synthetic, replay)
    flags = (False, True, False, False, False, False, False, False)
    callback = runtime.VerifierRewardCallback(
        groups,
        truncation_provider=lambda *, expected_count: flags[:expected_count],
        completion_token_counter=lambda text: len(text.split()),
    )
    rows = [synthetic.policy_row()] * 4 + [replay.policy_row()] * 4
    completions: list[object] = [
        [{"role": "assistant", "content": "Final answer: 1"}],
        [{"role": "assistant", "content": "Final answer: 2"}],
        [{"role": "assistant", "content": "Final answer: 1"}],
        [{"role": "assistant", "content": "Final answer: 1"}],
        [{"role": "assistant", "content": "OK"}],
        [{"role": "assistant", "content": "NO"}],
        [{"role": "assistant", "content": "OK"}],
        [{"role": "assistant", "content": "OK"}],
    ]
    rewards = callback(
        prompts=[row["prompt"] for row in rows],
        completions=completions,
        group_id=[row["group_id"] for row in rows],
        source_kind=[row["source_kind"] for row in rows],
        source_id=[row["source_id"] for row in rows],
        prompt_sha256=[row["prompt_sha256"] for row in rows],
        reward_metadata_json=[row["reward_metadata_json"] for row in rows],
    )
    assert len(rewards) == 8
    assert callback.records[0].reward.correctness == 1.0
    assert callback.records[1].reward.correctness == 0.0
    assert callback.records[1].reward.generation_truncated is True
    assert callback.records[4].reward.replay_task_correctness == 1.0
    summary = runtime.summarize_reward_records(
        callback.records, groups, require_nonzero_variance=True
    )
    assert summary["groups"] == 2
    assert summary["completions"] == 8
    assert summary["synthetic_groups"] == 1
    assert summary["replay_groups"] == 1
    assert summary["scheduled_prompt_tokens"] == 20
    assert summary["generation_input_prompt_tokens"] == 80
    assert summary["nonzero_variance_groups"] == 2
    assert summary["reward_event_counts"] == {
        "synthetic_correct": 3,
        "replay_correct": 3,
        "task_correct_total": 6,
        "synthetic_extractable": 4,
        "synthetic_exact_contract": 3,
        "replay_exact_format": 3,
        "truncated": 1,
        "prompt_echo_or_question": 0,
        "conflicting_answers": 0,
    }
    assert all(
        "completion" not in record
        for record in [item.content_free_record() for item in callback.records]
    )


def test_reward_callback_fails_closed_on_metadata_prompt_and_group_drift() -> None:
    group = _runtime_group("strict")
    callback = runtime.VerifierRewardCallback(
        (group,), truncation_provider=lambda *, expected_count: (False,) * expected_count
    )
    row = group.policy_row()
    with pytest.raises(ValueError, match="outside"):
        callback(
            prompts=[row["prompt"]],
            completions=["Final answer: 1"],
            group_id=["other"],
            source_kind=[row["source_kind"]],
            source_id=[row["source_id"]],
            prompt_sha256=[row["prompt_sha256"]],
            reward_metadata_json=[row["reward_metadata_json"]],
        )
    with pytest.raises(ValueError, match="prompt"):
        callback(
            prompts=[[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]],
            completions=["Final answer: 1"],
            group_id=[row["group_id"]],
            source_kind=[row["source_kind"]],
            source_id=[row["source_id"]],
            prompt_sha256=[row["prompt_sha256"]],
            reward_metadata_json=[row["reward_metadata_json"]],
        )


def test_compatibility_group_selection_and_accounting_are_exact() -> None:
    groups = tuple(
        [
            _runtime_group("synthetic-a", position=1),
            _runtime_group("synthetic-b", position=2),
        ]
        + [
            _runtime_group("replay-a", source_kind="base_replay", position=3),
        ]
    )
    value = runtime.RuntimeSchedule("generic_control", groups, _hash("packet"), _hash("manifest"))
    updates, generation_only = runtime.select_compatibility_groups(value)
    assert len(updates) == 2
    assert all(group.source_kind == "synthetic" for group in updates)
    assert generation_only.source_kind == "base_replay"
    assert len((*updates, generation_only)) * schedule.COMPLETIONS_PER_GROUP == 12


def test_frozen_arguments_preserve_all_decision_values() -> None:
    config = runtime.load_grpo_config(Path("configs/training/verifier_grpo_v1.json"))
    values = runtime.frozen_grpo_argument_values(
        config,
        variant_id="G1",
        output_dir=Path("ignored-output"),
        mode="train",
    )
    assert values["max_steps"] == 64
    assert values["num_generations"] == 4
    assert values["per_device_train_batch_size"] == 4
    assert values["max_completion_length"] == 256
    assert values["beta"] == 0.04
    assert values["shuffle_dataset"] is False
    assert values["mask_truncated_completions"] is True
    assert values["disable_dropout"] is True
    assert values["save_strategy"] == "no"
    smoke = runtime.frozen_grpo_argument_values(
        config,
        variant_id="G2",
        output_dir=Path("ignored-output"),
        mode="compatibility",
    )
    assert smoke["max_steps"] == 2
    assert smoke["beta"] == 0.10

    arguments = SimpleNamespace(**values)
    runtime.assert_frozen_grpo_arguments(
        arguments,
        config,
        variant_id="G1",
        output_dir=Path("ignored-output"),
        mode="train",
    )
    arguments.num_generations = 3
    with pytest.raises(ValueError, match="num_generations"):
        runtime.assert_frozen_grpo_arguments(
            arguments,
            config,
            variant_id="G1",
            output_dir=Path("ignored-output"),
            mode="train",
        )


class _Parameter:
    def __init__(
        self, device: str, *, requires_grad: bool = False, grad: object | None = None
    ) -> None:
        self.device = device
        self.requires_grad = requires_grad
        self.grad = grad


class _DeviceModel:
    def __init__(
        self, devices: tuple[str, ...], device_map: dict[str, object] | None = None
    ) -> None:
        self._parameters = tuple(_Parameter(device) for device in devices)
        if device_map is not None:
            self.hf_device_map = device_map

    def named_parameters(self) -> list[tuple[str, _Parameter]]:
        return [(f"parameter_{index}", value) for index, value in enumerate(self._parameters)]


def test_cuda_audit_rejects_cpu_offload_and_invalid_device_maps() -> None:
    assert runtime.assert_cuda_only_model(_DeviceModel(("cuda:0",))) == ("cuda:0",)
    with pytest.raises(RuntimeError, match="offloading"):
        runtime.assert_cuda_only_model(_DeviceModel(("cuda:0", "cpu")))
    with pytest.raises(RuntimeError, match="device map"):
        runtime.assert_cuda_only_model(_DeviceModel(("cuda:0",), {"layer": "disk"}))


def test_frozen_base_gradient_audit_fails_closed() -> None:
    clean = _DeviceModel(("cuda:0",))
    runtime.assert_frozen_base_has_no_gradients(clean)
    dirty = _DeviceModel(("cuda:0",))
    dirty._parameters = (_Parameter("cuda:0", grad=object()),)
    with pytest.raises(RuntimeError, match="received gradients"):
        runtime.assert_frozen_base_has_no_gradients(dirty)


class _SavingModel:
    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"adapter")


class _SavingTokenizer:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_final_adapter_save_allows_trainer_created_parent_but_never_overwrites(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "trainer_state").mkdir(parents=True)
    path, digest = runtime.save_final_adapter(output_dir, _SavingModel(), _SavingTokenizer())
    assert path == output_dir / "adapter"
    assert len(digest) == 64
    with pytest.raises(FileExistsError, match="overwrite"):
        runtime.save_final_adapter(output_dir, _SavingModel(), _SavingTokenizer())


def test_reward_summary_rejects_missing_wrong_order_and_zero_variance() -> None:
    first = _runtime_group("first")
    second = _runtime_group("second", position=2)
    breakdown = runtime.score_reward(
        runtime._reward_metadata(first, first.reward_metadata_json, first.policy_row()["prompt"]),
        "Final answer: 1",
    )
    records = [
        runtime.CompletionRewardAudit(
            sequence=index,
            group_id=group_id,
            source_kind="synthetic",
            source_id="fixture",
            completion="Final answer: 1",
            completion_sha256=_hash(f"completion-{index}"),
            completion_tokens=3,
            reward=breakdown,
        )
        for index, group_id in enumerate([first.group_id] * 4 + [second.group_id] * 4)
    ]
    with pytest.raises(RuntimeError, match="variance"):
        runtime.summarize_reward_records(records, (first, second), require_nonzero_variance=True)
    swapped = records[4:] + records[:4]
    with pytest.raises(RuntimeError, match="order"):
        runtime.summarize_reward_records(swapped, (first, second), require_nonzero_variance=False)
    with pytest.raises(RuntimeError, match="completion count"):
        runtime.summarize_reward_records(
            records[:-1], (first, second), require_nonzero_variance=False
        )


def test_finite_metric_gate_requires_loss_and_kl() -> None:
    assert runtime._finite_history_metrics([{"loss": 1.0, "kl": 0.1}]) == {
        "loss": [1.0],
        "kl": [0.1],
    }
    with pytest.raises(RuntimeError, match="KL"):
        runtime._finite_history_metrics([{"loss": 1.0}])
    with pytest.raises(RuntimeError, match="NaN"):
        runtime._finite_history_metrics([{"loss": float("nan"), "kl": 0.1}])


def test_runtime_group_is_immutable() -> None:
    group = _runtime_group("immutable")
    assert replace(group, prompt_tokens=11).prompt_tokens == 11
    with pytest.raises(FrozenInstanceError):
        group.prompt_tokens = 12  # type: ignore[misc]


def test_tracked_compatibility_failure_evidence_reconstructs() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "results/training/verifier_grpo_v1_compatibility_failure.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    declared = value.pop("summary_sha256")
    assert declared == canonical_sha256(value)
    assert value["status"] == "failed_before_first_completion"
    assert value["gate_passed"] is False
    assert value["stop_rule_enforced"] is True
    assert value["completions_generated"] == 0
    assert value["optimizer_steps_completed"] == 0
    assert value["training_authorized_after_smoke"] is False
