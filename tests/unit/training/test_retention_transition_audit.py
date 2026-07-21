import json
from pathlib import Path

from foundry.training.retention_transition_audit import build_instruction_transition_audit


def _row(item_id: str, correct: bool, response: str) -> dict[str, object]:
    return {
        "id": item_id,
        "response": response,
        "response_sha256": "a" * 64,
        "score": {"correct": correct},
    }


def test_transition_audit_counts_shared_regression(tmp_path: Path) -> None:
    instruction_items = [
        {
            "id": f"item-{index}",
            "section": "instruction",
            "skill": "copy",
            "kind": "exact_text",
            "prompt": f"Return token {index}.",
            "expected": str(index),
        }
        for index in range(25)
    ]
    items = (
        [
            {
                "id": f"arithmetic-{index}",
                "section": "arithmetic",
                "skill": "addition",
                "kind": "numeric_terminal",
                "prompt": f"Add {index} and 1.",
                "expected": str(index + 1),
            }
            for index in range(45)
        ]
        + [
            {
                "id": f"format-{index}",
                "section": "format",
                "skill": "copy",
                "kind": "exact_text",
                "prompt": f"Copy format token {index}.",
                "expected": str(index),
            }
            for index in range(20)
        ]
        + instruction_items
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "foundry-retention-validation-v1",
                "system_prompt": "Follow the instruction.",
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": 32,
                    "seed": 20260720,
                },
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    paths = {name: tmp_path / f"{name}.json" for name in ("base", "generic", "targeted")}
    for name, path in paths.items():
        path.write_text(
            json.dumps(
                [
                    _row(item["id"], not (name != "base" and index == 0), str(index))
                    for index, item in enumerate(instruction_items)
                ]
            ),
            encoding="utf-8",
        )
    classifications = tmp_path / "classifications.json"
    classifications.write_text(
        json.dumps({"classifications": {"item-0": "genuine_instruction_noncompliance"}}),
        encoding="utf-8",
    )
    summary, detailed = build_instruction_transition_audit(
        suite_path=suite,
        base_path=paths["base"],
        generic_path=paths["generic"],
        targeted_path=paths["targeted"],
        classifications_path=classifications,
    )
    assert len(detailed) == 25
    assert summary["correct"] == {"base": 25, "generic_control": 24, "targeted": 24}
    assert summary["shared_adapter_failures"] == 1
    assert summary["genuine_behavior_regressions"] == 1
    assert summary["confirmed_prompt_or_scorer_defects"] == 0
