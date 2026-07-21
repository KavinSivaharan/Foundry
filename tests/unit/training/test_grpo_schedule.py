from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from foundry.training import grpo_schedule as gs
from foundry.training.base_replay import REPLAY_CORPUS_ID, REPLAY_FORMAT_ID
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256


def _synthetic(
    arm: gs.Arm, family: gs.Family, index: int, *, marker: str | None = None
) -> gs.SyntheticPrompt:
    text = marker or f"Original {arm} {family} prompt {index}."
    messages = (
        gs.PromptMessage("system", "Fixture system."),
        gs.PromptMessage("user", text),
    )
    return gs.SyntheticPrompt(
        synthetic_id=f"{arm}-{family}-{index:03d}",
        arm=arm,
        family=family,
        messages=messages,
        prompt_sha256=canonical_sha256([item.as_dict() for item in messages]),
        canonical_final_answer="1",
        mode="fixture_mode",
        difficulty="easy",
        output_contract_enabled=False,
        verifier_metadata_sha256=canonical_sha256({"verified": True}),
        provenance_sha256=canonical_sha256({"id": f"{arm}-{family}-{index:03d}"}),
    )


def _synthetic_pool(arm: gs.Arm, count_per_family: int = 36) -> tuple[gs.SyntheticPrompt, ...]:
    return tuple(
        _synthetic(arm, family, index)
        for family in gs.FAMILY_ORDER
        for index in range(count_per_family)
    )


def _replay_pool(count_per_section: int = 5) -> tuple[gs.ReplayPrompt, ...]:
    prompts: list[gs.ReplayPrompt] = []
    for section in gs.REPLAY_SECTION_ORDER:
        for index in range(count_per_section):
            messages = (
                gs.PromptMessage("system", "Replay fixture system."),
                gs.PromptMessage("user", f"Original replay {section} prompt {index}."),
            )
            prompts.append(
                gs.ReplayPrompt(
                    replay_id=f"replay-{section}-{index:02d}",
                    section=section,
                    skill=f"fixture-{section}",
                    messages=messages,
                    prompt_sha256=canonical_sha256([item.as_dict() for item in messages]),
                    kind="numeric_terminal",
                    expected="1",
                    scorer_sha256=canonical_sha256({"scorer": "fixture"}),
                    provenance_sha256=canonical_sha256({"id": f"replay-{section}-{index:02d}"}),
                )
            )
    return tuple(prompts)


def _word_counter(messages: tuple[gs.PromptMessage, ...]) -> int:
    return sum(len(message.content.split()) for message in messages) + 3


def _bundle() -> gs.GRPOScheduleBundle:
    return gs.build_grpo_schedules(
        generic_prompts=_synthetic_pool("generic_control"),
        targeted_prompts=_synthetic_pool("targeted"),
        replay_prompts=_replay_pool(),
        prompt_token_counter=_word_counter,
    )


def test_schedule_is_exact_balanced_prompt_only_and_deterministic() -> None:
    first = _bundle()
    second = _bundle()
    assert first == second
    summary_without_hash = dict(first.summary)
    summary_without_hash.pop("summary_sha256")
    assert canonical_sha256(summary_without_hash) == first.summary["summary_sha256"]
    assert first.summary["prompt_token_parity_passed"] is True
    assert float(first.summary["prompt_token_parity_ratio"]) <= 0.01

    all_group_ids: list[str] = []
    selected_synthetic: dict[str, set[str]] = {}
    replay_ids: list[list[str]] = []
    for arm_schedule in (first.generic_control, first.targeted):
        assert len(arm_schedule.groups) == 64
        assert sum(item.completions_per_group for item in arm_schedule.groups) == 256
        synthetic = [item for item in arm_schedule.groups if item.source_kind == "synthetic"]
        replay = [item for item in arm_schedule.groups if item.source_kind == "base_replay"]
        assert len(synthetic) == 52
        assert len(replay) == 12
        assert [item.position for item in replay] == list(gs.REPLAY_POSITIONS)
        expected = gs.SYNTHETIC_QUOTAS[arm_schedule.arm]
        assert Counter(item.category for item in synthetic) == expected
        assert Counter(item.category for item in replay) == {
            "arithmetic": 4,
            "format": 4,
            "instruction": 4,
        }
        assert len({item.source_id for item in synthetic}) == 52
        assert len({item.prompt_sha256 for item in arm_schedule.groups}) == 64
        assert arm_schedule.manifest["retry_until_success"] is False
        assert arm_schedule.manifest["model_output_selection"] is False
        assert arm_schedule.manifest["model_outputs_observed_during_scheduling"] is False
        manifest_text = json.dumps(arm_schedule.manifest)
        packet_text = json.dumps(arm_schedule.prompt_packet())
        assert "Original " not in manifest_text
        assert "reward_metadata" not in manifest_text
        assert "canonical_final_answer" in packet_text
        assert '"expected"' in packet_text
        assert "training_completion" not in packet_text
        assert "deterministic_solution_trace" not in packet_text
        assert "assistant_response" not in packet_text
        all_group_ids.extend(item.group_id for item in arm_schedule.groups)
        selected_synthetic[arm_schedule.arm] = {item.source_id for item in synthetic}
        replay_ids.append([item.source_id for item in replay])
    assert len(set(all_group_ids)) == 128
    assert selected_synthetic["generic_control"].isdisjoint(selected_synthetic["targeted"])
    assert replay_ids[0] == replay_ids[1]


def test_schedule_rejects_disjointness_quota_replay_and_token_parity_failures() -> None:
    generic = _synthetic_pool("generic_control")
    targeted = _synthetic_pool("targeted")
    replay = _replay_pool()
    duplicate = gs.SyntheticPrompt(
        synthetic_id=targeted[0].synthetic_id,
        arm="targeted",
        family=targeted[0].family,
        messages=generic[0].messages,
        prompt_sha256=generic[0].prompt_sha256,
        canonical_final_answer="1",
        mode="fixture_mode",
        difficulty="easy",
        output_contract_enabled=False,
        verifier_metadata_sha256=canonical_sha256({"verified": True}),
        provenance_sha256=canonical_sha256({"id": "duplicate"}),
    )
    with pytest.raises(ValueError, match="disjoint"):
        gs.build_grpo_schedules(
            generic_prompts=generic,
            targeted_prompts=(duplicate, *targeted[1:]),
            replay_prompts=replay,
            prompt_token_counter=_word_counter,
        )
    with pytest.raises(ValueError, match="insufficient"):
        gs.build_grpo_schedules(
            generic_prompts=generic[:20],
            targeted_prompts=targeted,
            replay_prompts=replay,
            prompt_token_counter=_word_counter,
        )
    with pytest.raises(ValueError, match="fewer than four"):
        gs.build_grpo_schedules(
            generic_prompts=generic,
            targeted_prompts=targeted,
            replay_prompts=tuple(item for item in replay if item.section != "format")[:10],
            prompt_token_counter=_word_counter,
        )

    def incompatible_counter(messages: tuple[gs.PromptMessage, ...]) -> int:
        text = messages[-1].content
        if "generic_control" in text:
            return 100
        if "targeted" in text:
            return 200
        return 1

    with pytest.raises(ValueError, match="parity exceeds"):
        gs.build_grpo_schedules(
            generic_prompts=generic,
            targeted_prompts=targeted,
            replay_prompts=replay,
            prompt_token_counter=incompatible_counter,
        )


def test_synthetic_loader_projects_away_answers_and_rejects_unverified_rows(
    tmp_path: Path,
) -> None:
    row: dict[str, object] = {
        "synthetic_id": "fixture-001",
        "group": "targeted",
        "future_split": "training",
        "final_decision": "accepted",
        "family": gs.FAMILY_ORDER[0],
        "mode": "fixture_mode",
        "difficulty": "easy",
        "output_contract_enabled": False,
        "rendered_question": "An original fixture asks for a total.",
        "rendered_text_sha256": "a" * 64,
        "latent_program_sha256": "b" * 64,
        "semantic_ir_sha256": "c" * 64,
        "canonical_final_answer": "999",
        "training_completion": "This must not survive prompt projection.",
        "deterministic_solution_trace": "Private reasoning trace.",
        "primary_evidence_sha256": "d" * 64,
        "independent_evidence_sha256": "e" * 64,
        "primary_verifier_success": True,
        "independent_verifier_success": True,
        "verifier_agreement": True,
    }
    path = tmp_path / "synthetic.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    prompts = gs.load_synthetic_prompts(path, "targeted")
    assert len(prompts) == 1
    assert prompts[0].canonical_final_answer == "999"
    assert not hasattr(prompts[0], "training_completion")
    assert "999" not in json.dumps([message.as_dict() for message in prompts[0].messages])
    assert "Private reasoning trace" not in repr(prompts[0])
    row["verifier_agreement"] = False
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verified accepted"):
        gs.load_synthetic_prompts(path, "targeted")


def test_replay_loader_validates_manifest_then_projects_away_base_output(tmp_path: Path) -> None:
    response = "A private frozen base response."
    anchor: dict[str, object] = {
        "schema_version": 1,
        "anchor_id": "foundry-shared-retention-anchor-v1",
        "chat_template": "fixture",
        "masking": "fixture",
        "system_prompt": "Original fixture system.",
        "items": [
            {
                "id": "replay-1",
                "section": "arithmetic",
                "skill": "addition",
                "kind": "numeric_terminal",
                "prompt": "Original fixture prompt.",
                "expected": "7",
                "gold_response": "Private predefined gold.",
            }
        ],
    }
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
    raw: dict[str, object] = {
        "schema_version": 1,
        "replay_corpus_id": REPLAY_CORPUS_ID,
        "replay_format_id": REPLAY_FORMAT_ID,
        "source_anchor_sha256": canonical_sha256(anchor),
        "base_result_sha256": "b" * 64,
        "items": [
            {
                "id": "replay-1",
                "section": "arithmetic",
                "skill": "addition",
                "system_prompt": "Original fixture system.",
                "prompt": "Original fixture prompt.",
                "assistant_response": response,
                "assistant_response_sha256": gs._sha256_text(response),
            }
        ],
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "replay_corpus_id": REPLAY_CORPUS_ID,
        "replay_format_id": REPLAY_FORMAT_ID,
        "gate_passed": True,
        "prompts_or_outputs_in_manifest": False,
        "anchor_sha256": canonical_sha256(anchor),
        "replay_corpus_sha256": canonical_sha256(raw),
        "raw_replay_packet_sha256": file_sha256(raw_path),
        "total": 1,
        "items": [
            {
                "id": "replay-1",
                "section": "arithmetic",
                "skill": "addition",
                "base_output_sha256": gs._sha256_text(response),
            }
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prompts = gs.load_replay_prompts(raw_path, manifest_path, anchor_path)
    assert len(prompts) == 1
    assert not hasattr(prompts[0], "assistant_response")
    assert prompts[0].kind == "numeric_terminal"
    assert prompts[0].expected == "7"
    assert response not in json.dumps([message.as_dict() for message in prompts[0].messages])
    manifest["total"] = 2
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest_without_hash)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="counts differ"):
        gs.load_replay_prompts(raw_path, manifest_path, anchor_path)


def test_transformers_counter_and_ignored_packet_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeTokenizer:
        def apply_chat_template(
            self,
            conversation: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> list[int]:
            assert tokenize is True
            assert add_generation_prompt is True
            assert conversation[0]["role"] == "system"
            return [1, 2, 3, 4]

    messages = (gs.PromptMessage("system", "x"), gs.PromptMessage("user", "y"))
    assert gs.count_transformers_prompt_tokens(FakeTokenizer(), messages) == 4
    bundle = _bundle()
    monkeypatch.setattr(gs, "_git_ignored", lambda _root, path: "raw" in path.parts)
    root = tmp_path / "repo"
    generic_packet = root / "results" / "raw" / "generic.json"
    targeted_packet = root / "results" / "raw" / "targeted.json"
    gs.write_grpo_schedule_bundle(
        bundle,
        repository_root=root,
        generic_packet_path=generic_packet,
        targeted_packet_path=targeted_packet,
        generic_manifest_path=root / "results" / "generic_manifest.json",
        targeted_manifest_path=root / "results" / "targeted_manifest.json",
        summary_path=root / "results" / "summary.json",
    )
    assert "messages" in generic_packet.read_text(encoding="utf-8")
    assert "messages" not in (root / "results" / "generic_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "reward_metadata" not in (root / "results" / "generic_manifest.json").read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not Git ignored"):
        gs.write_grpo_schedule_bundle(
            bundle,
            repository_root=root,
            generic_packet_path=root / "tracked" / "generic.json",
            targeted_packet_path=targeted_packet,
            generic_manifest_path=root / "results" / "generic_manifest.json",
            targeted_manifest_path=root / "results" / "targeted_manifest.json",
            summary_path=root / "results" / "summary.json",
        )


def test_production_prompt_sources_build_when_local_raw_artifacts_exist() -> None:
    root = Path(__file__).resolve().parents[3]
    generic_path = root / "results/raw/foundry_500x2_signal_data/generic_control_training.jsonl"
    targeted_path = root / "results/raw/foundry_500x2_signal_data/targeted_training.jsonl"
    replay_path = root / "results/raw/training/base_replay_kl/replay_corpus.json"
    replay_manifest = root / "results/training/base_replay_corpus.json"
    anchor_path = (
        root / "results/raw/training/retention_powered_adjudication/shared_retention_anchor_v1.json"
    )
    if not all(
        path.exists()
        for path in (generic_path, targeted_path, replay_path, replay_manifest, anchor_path)
    ):
        pytest.skip("ignored production prompt sources are not present")
    bundle = gs.build_grpo_schedules(
        generic_prompts=gs.load_synthetic_prompts(generic_path, "generic_control"),
        targeted_prompts=gs.load_synthetic_prompts(targeted_path, "targeted"),
        replay_prompts=gs.load_replay_prompts(replay_path, replay_manifest, anchor_path),
        prompt_token_counter=_word_counter,
    )
    assert bundle.summary["prompt_token_parity_passed"] is True
    assert bundle.generic_control.manifest["total_completions"] == 256
    assert bundle.targeted.manifest["total_completions"] == 256
