import inspect
import json
from pathlib import Path

import pytest

from foundry.training import base_replay as br
from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

FROZEN_ANCHOR = Path(
    "results/raw/training/retention_powered_adjudication/shared_retention_anchor_v1.json"
)


def _fixture_root() -> dict[str, object]:
    items: list[dict[str, str]] = []
    for section in br.SECTION_ORDER:
        for index in range(40):
            item_id = f"original-{section}-{index:03d}"
            if section == "arithmetic":
                kind = "numeric_terminal"
                expected = str(index + 2)
                prompt = f"Add one to {index + 1}."
                gold = f"One more is {expected}.\nFinal answer: {expected}"
            else:
                kind = "exact_text"
                expected = f"token-{section}-{index}"
                prompt = f"Return the original token for row {index}."
                gold = expected
            items.append(
                {
                    "id": item_id,
                    "section": section,
                    "skill": f"original-{section}-skill",
                    "kind": kind,
                    "prompt": prompt,
                    "expected": expected,
                    "gold_response": gold,
                }
            )
    return {
        "schema_version": 1,
        "anchor_id": br.ANCHOR_ID,
        "chat_template": br.CHAT_TEMPLATE_ID,
        "masking": br.MASKING_ID,
        "system_prompt": "Follow this original deterministic fixture instruction.",
        "items": items,
    }


def _fixture_anchor() -> br.ReplayAnchor:
    root = _fixture_root()
    items = root["items"]
    assert isinstance(items, list)
    identity = br._AnchorIdentity(
        anchor_sha256=canonical_sha256(root),
        prompt_sha256=canonical_sha256(
            [{"id": item["id"], "prompt": item["prompt"]} for item in items]
        ),
        gold_response_sha256=canonical_sha256(
            [{"id": item["id"], "gold_response": item["gold_response"]} for item in items]
        ),
        answer_sha256=canonical_sha256(
            [{"id": item["id"], "expected": item["expected"]} for item in items]
        ),
    )
    return br._parse_anchor(root, identity)


def _correct_generations(anchor: br.ReplayAnchor) -> list[br.BaseGenerationResult]:
    generations: list[br.BaseGenerationResult] = []
    for item in anchor.items:
        if item.section == "arithmetic":
            response = f"The original calculation is complete.\nFinal answer: {item.expected}"
        else:
            response = item.expected
        generations.append(br.BaseGenerationResult(item.item_id, response))
    return generations


def _write_evaluation(
    tmp_path: Path,
    anchor: br.ReplayAnchor,
    generations: list[br.BaseGenerationResult] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    raw = tmp_path / "base_raw.json"
    summary = tmp_path / "base_summary.json"
    value = br.write_base_anchor_evaluation(
        anchor=anchor,
        generations=_correct_generations(anchor) if generations is None else generations,
        raw_path=raw,
        summary_path=summary,
    )
    return raw, summary, value


def _write_fixture_anchor(tmp_path: Path) -> Path:
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(_fixture_root()), encoding="utf-8")
    return path


def _freeze_fixture(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    anchor: br.ReplayAnchor,
    raw: Path,
    summary: Path,
    defect_ids: tuple[str, ...] = (),
) -> tuple[dict[str, object], Path, Path]:
    anchor_path = _write_fixture_anchor(tmp_path)
    monkeypatch.setattr(br, "load_frozen_replay_anchor", lambda _: anchor)
    replay_raw = tmp_path / "ignored" / "replay.json"
    manifest_path = tmp_path / "manifest.json"
    manifest = br.freeze_base_correct_replay_corpus(
        anchor_path=anchor_path,
        base_summary_path=summary,
        base_raw_path=raw,
        replay_raw_path=replay_raw,
        manifest_path=manifest_path,
        objectively_defective_item_ids=defect_ids,
    )
    return manifest, replay_raw, manifest_path


def test_frozen_anchor_has_exact_identity_and_balance() -> None:
    anchor = br.load_frozen_replay_anchor(FROZEN_ANCHOR)
    assert anchor.anchor_sha256 == br.ANCHOR_SHA256
    assert anchor.prompt_sha256 == br.ANCHOR_PROMPT_SHA256
    assert anchor.gold_response_sha256 == br.ANCHOR_GOLD_RESPONSE_SHA256
    assert anchor.answer_sha256 == br.ANCHOR_ANSWER_SHA256
    assert len(anchor.items) == 120
    assert {
        section: sum(item.section == section for item in anchor.items)
        for section in br.SECTION_ORDER
    } == br.EXPECTED_SECTION_COUNTS


def test_anchor_tampering_fails_closed() -> None:
    root = _fixture_root()
    anchor = _fixture_anchor()
    items = root["items"]
    assert isinstance(items, list)
    items[0]["prompt"] = "Changed original fixture prompt."
    identity = br._AnchorIdentity(
        anchor.anchor_sha256,
        anchor.prompt_sha256,
        anchor.gold_response_sha256,
        anchor.answer_sha256,
    )
    with pytest.raises(ValueError, match="hash differs"):
        br._parse_anchor(root, identity)


def test_generation_contract_is_frozen_and_base_api_has_no_adapter() -> None:
    assert br.GENERATION_CONFIG == {
        "do_sample": False,
        "max_new_tokens": 384,
        "seed": 20260720,
    }
    assert canonical_sha256(br.GENERATION_CONFIG) == br.GENERATION_CONFIG_SHA256
    assert "adapter" not in inspect.signature(br.evaluate_untouched_base_anchor).parameters


def test_base_evaluation_requires_complete_ordered_anchor(tmp_path: Path) -> None:
    anchor = _fixture_anchor()
    generations = _correct_generations(anchor)
    generations[0], generations[1] = generations[1], generations[0]
    with pytest.raises(ValueError, match="order or IDs"):
        br.write_base_anchor_evaluation(
            anchor=anchor,
            generations=generations,
            raw_path=tmp_path / "raw.json",
            summary_path=tmp_path / "summary.json",
        )


def test_replay_freeze_uses_only_correct_actual_base_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = _fixture_anchor()
    generations = _correct_generations(anchor)
    first = anchor.items[0]
    assert generations[0].response != first.gold_response
    generations[1] = br.BaseGenerationResult(anchor.items[1].item_id, "Final answer: -999")
    raw, summary, _ = _write_evaluation(tmp_path, anchor, generations)
    manifest, replay_raw, manifest_path = _freeze_fixture(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        anchor=anchor,
        raw=raw,
        summary=summary,
    )
    packet = json.loads(replay_raw.read_text(encoding="utf-8"))
    tracked = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["total"] == 119
    assert manifest["section_counts"] == {
        "arithmetic": 39,
        "format": 40,
        "instruction": 40,
    }
    assert anchor.items[1].item_id not in {item["id"] for item in packet["items"]}
    assert packet["items"][0]["assistant_response"] == generations[0].response
    assert packet["items"][0]["assistant_response"] != first.gold_response
    assert all("prompt" not in item for item in tracked["items"])
    assert all("assistant_response" not in item for item in tracked["items"])
    assert manifest["predefined_gold_used_as_replay_target"] is False
    assert manifest["gate_passed"] is True


def test_replay_gate_rejects_insufficient_total(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = _fixture_anchor()
    generations = _correct_generations(anchor)
    for index in range(20, 40):
        item = anchor.items[index]
        generations[index] = br.BaseGenerationResult(item.item_id, "incorrect")
    for index in range(60, 80):
        item = anchor.items[index]
        generations[index] = br.BaseGenerationResult(item.item_id, "incorrect")
    for index in range(100, 120):
        item = anchor.items[index]
        generations[index] = br.BaseGenerationResult(item.item_id, "incorrect")
    raw, summary, _ = _write_evaluation(tmp_path, anchor, generations)
    with pytest.raises(ValueError, match="overall_at_least_75"):
        _freeze_fixture(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            anchor=anchor,
            raw=raw,
            summary=summary,
        )


def test_replay_gate_rejects_insufficient_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = _fixture_anchor()
    generations = _correct_generations(anchor)
    for index in range(19, 40):
        item = anchor.items[index]
        generations[index] = br.BaseGenerationResult(item.item_id, "Final answer: -999")
    raw, summary, _ = _write_evaluation(tmp_path, anchor, generations)
    with pytest.raises(ValueError, match="arithmetic_at_least_20"):
        _freeze_fixture(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            anchor=anchor,
            raw=raw,
            summary=summary,
        )


def test_replay_gate_rejects_backend_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = _fixture_anchor()
    generations = _correct_generations(anchor)
    generations[0] = br.BaseGenerationResult(anchor.items[0].item_id, "", "RuntimeError")
    raw, summary, _ = _write_evaluation(tmp_path, anchor, generations)
    with pytest.raises(ValueError, match="zero_backend_failures"):
        _freeze_fixture(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            anchor=anchor,
            raw=raw,
            summary=summary,
        )


def test_replay_gate_rejects_confirmed_objective_defect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = _fixture_anchor()
    raw, summary, _ = _write_evaluation(tmp_path, anchor)
    with pytest.raises(ValueError, match="zero_objective_prompt_or_scorer_defects"):
        _freeze_fixture(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            anchor=anchor,
            raw=raw,
            summary=summary,
            defect_ids=(anchor.items[0].item_id,),
        )


def test_replay_freeze_rejects_raw_response_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = _fixture_anchor()
    raw, summary, _ = _write_evaluation(tmp_path, anchor)
    rows = json.loads(raw.read_text(encoding="utf-8"))
    rows[0]["response"] = "tampered"
    raw.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_value = json.loads(summary.read_text(encoding="utf-8"))
    summary_value["raw_packet_sha256"] = file_sha256(raw)
    summary_value.pop("summary_sha256")
    summary_value["summary_sha256"] = canonical_sha256(summary_value)
    summary.write_text(json.dumps(summary_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="response hash differs"):
        _freeze_fixture(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            anchor=anchor,
            raw=raw,
            summary=summary,
        )


def test_replay_freeze_rejects_adapter_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anchor = _fixture_anchor()
    raw, summary, _ = _write_evaluation(tmp_path, anchor)
    summary_value = json.loads(summary.read_text(encoding="utf-8"))
    summary_value["adapter_loaded"] = True
    summary_value["adapter_sha256"] = "a" * 64
    summary_value.pop("summary_sha256")
    summary_value["summary_sha256"] = canonical_sha256(summary_value)
    summary.write_text(json.dumps(summary_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter state differs"):
        _freeze_fixture(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            anchor=anchor,
            raw=raw,
            summary=summary,
        )


def test_base_evaluator_refuses_adapter_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="refuses an adapter"):
        br.evaluate_untouched_base_anchor(
            anchor_path=FROZEN_ANCHOR,
            model_path=model_path,
            raw_path=tmp_path / "raw.json",
            summary_path=tmp_path / "summary.json",
        )
