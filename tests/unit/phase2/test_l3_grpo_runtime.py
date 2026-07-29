from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from foundry.phase2 import l3_grpo_runtime
from foundry.phase2.l3_grpo_schedule import (
    PromptMessage,
)
from foundry.training.config import canonical_sha256


class _FakeConfig:
    def __init__(self, **values: object) -> None:
        vars(self).update(values)


class _FakeTrl:
    GRPOConfig = _FakeConfig


def test_trainer_arguments_match_the_frozen_recipe(tmp_path: Path) -> None:
    arguments = l3_grpo_runtime._trainer_arguments(
        _FakeTrl,
        output_dir=tmp_path,
        max_steps=32,
    )
    assert arguments.max_steps == 32
    assert arguments.num_generations == 4
    assert arguments.beta == 0.04
    assert arguments.learning_rate == 0.000001
    assert arguments.optim == "paged_adamw_8bit"
    assert arguments.loss_type == "dr_grpo"
    assert arguments.scale_rewards is False
    assert arguments.mask_truncated_completions is True
    assert arguments.gradient_checkpointing is False
    assert arguments.use_vllm is False
    assert arguments.full_determinism is False


def _runtime_group() -> l3_grpo_runtime.RuntimeGroup:
    messages = (
        PromptMessage("system", "fixture system"),
        PromptMessage("user", "Report the final fixture count."),
    )
    return l3_grpo_runtime.RuntimeGroup(
        group_id="fixture-group",
        arm="generic",
        position=1,
        source_kind="task",
        source_id="fixture-task",
        category="multi_step_bookkeeping_or_omission",
        messages=messages,
        prompt_sha256=canonical_sha256([message.as_dict() for message in messages]),
        prompt_tokens=24,
        reward_metadata_json=json.dumps(
            {
                "reward_kind": "task",
                "canonical_answer": "7",
                "answer_type": "integer",
                "family": "multi_step_bookkeeping_or_omission",
                "difficulty": "easy",
                "verifier_metadata_sha256": "a" * 64,
                "provenance_sha256": "b" * 64,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def test_runtime_reward_callback_binds_hidden_metadata_and_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _runtime_group()
    callback = l3_grpo_runtime.VerifierRewardCallback(
        [group],
        completion_token_counter=lambda completion: len(completion.split()),
    )
    monkeypatch.setattr(
        l3_grpo_runtime,
        "get_active_truncation_flags",
        lambda *, expected_count: (False,) * expected_count,
    )
    row = group.policy_row()
    values = callback(
        prompts=[row["prompt"]] * 4,
        completions=[
            [{"role": "assistant", "content": "Final answer: 7"}],
            [{"role": "assistant", "content": "Final answer: 8"}],
            [{"role": "assistant", "content": "No result."}],
            [{"role": "assistant", "content": "Could it be seven?\nFinal answer: 7"}],
        ],
        group_id=[row["group_id"]] * 4,
        source_kind=[row["source_kind"]] * 4,
        source_id=[row["source_id"]] * 4,
        prompt_sha256=[row["prompt_sha256"]] * 4,
        reward_metadata_json=[row["reward_metadata_json"]] * 4,
    )
    assert values[0] > values[1]
    assert values[0] > values[2]
    assert values[0] > values[3]
    assert len(callback.records) == 4
    summary = l3_grpo_runtime.summarize_rewards(
        callback.records,
        [group],
        require_nonzero_variance=True,
    )
    assert summary["groups"] == 1
    assert summary["completions"] == 4
    assert summary["nonzero_variance_groups"] == 1
    assert summary["backend_failures"] == 0


def test_task_generation_evidence_uses_content_free_task_label() -> None:
    torch = pytest.importorskip("torch")
    group = _runtime_group()
    evidence = l3_grpo_runtime._capture_l3_generation_evidence(
        group=group,
        generated_token_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        decoded_completions=("Final answer: 7", "Final answer: 8"),
        completion_token_lengths=(2, 2),
        truncation_flags=(False, False),
        reward_components=(
            {"total": 1.2, "backend_failure": False},
            {"total": 0.2, "backend_failure": False},
        ),
        rng_before_sha256="c" * 64,
        rng_after_sha256="d" * 64,
        warning_sha256s=("e" * 64,),
        reference_logprobs=torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]),
        policy_logprobs=torch.tensor([[-1.1, -2.1], [-3.1, -4.1]]),
        per_token_kl=torch.tensor([[0.1, 0.1], [0.1, 0.1]]),
    )
    payload = evidence.as_dict()
    assert payload["source_kind"] == "task"
    evidence_sha256 = payload.pop("evidence_sha256")
    payload.pop("warning_count")
    assert evidence_sha256 == canonical_sha256(payload)


def test_optimizer_contract_rejects_non_policy_ownership() -> None:
    parameter = SimpleNamespace(requires_grad=True, numel=lambda: 1)
    trainer = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                ("base_model.weight", parameter),
            ]
        ),
        optimizer=SimpleNamespace(param_groups=[{"params": [parameter]}]),
    )
    with pytest.raises(RuntimeError, match="policy-only"):
        l3_grpo_runtime._optimizer_ownership(trainer)
