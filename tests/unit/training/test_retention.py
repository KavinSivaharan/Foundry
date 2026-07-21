from pathlib import Path

from foundry.training.retention import SUITE_LAYOUTS, RetentionItem, load_suite, score_response

SUITE = Path("configs/training/assistant_only_v3_retention_suite.json")


def test_frozen_retention_suite_has_required_balance() -> None:
    suite = load_suite(SUITE)
    assert len(suite.items) == 60
    assert sum(item.section == "arithmetic" for item in suite.items) == 30
    assert sum(item.section == "format" for item in suite.items) == 15
    assert sum(item.section == "instruction" for item in suite.items) == 15
    assert len(suite.suite_sha256) == 64
    assert len(suite.prompt_sha256) == 64


def test_unknown_retention_suite_identity_rejects(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_text('{"schema_version":1,"suite_id":"unknown"}', encoding="utf-8")
    try:
        load_suite(path)
    except ValueError as error:
        assert "identity differs" in str(error)
    else:
        raise AssertionError("unknown suite identity should reject")


def test_disjoint_retention_suite_layouts_are_frozen() -> None:
    assert SUITE_LAYOUTS == {
        "foundry-original-retention-suite-v1": {
            "arithmetic": 30,
            "format": 15,
            "instruction": 15,
        },
        "foundry-retention-validation-v1": {
            "arithmetic": 45,
            "format": 20,
            "instruction": 25,
        },
        "foundry-retention-final-holdout-v1": {
            "arithmetic": 45,
            "format": 20,
            "instruction": 25,
        },
    }


def test_numeric_terminal_scoring_uses_exact_extractor() -> None:
    item = RetentionItem("fixture", "arithmetic", "addition", "numeric_terminal", "original", "42")
    result = score_response(item, "14 + 28 = 42.\nFinal answer: 42")
    assert result["correct"] is True
    assert result["extractable"] is True
    assert result["exact_format"] is True


def test_exact_text_scoring_rejects_extra_prose() -> None:
    item = RetentionItem("fixture", "format", "word", "exact_text", "original", "cedar")
    assert score_response(item, "cedar")["correct"] is True
    assert score_response(item, "The answer is cedar.")["correct"] is False


def test_json_scoring_allows_whitespace_but_not_extra_keys() -> None:
    item = RetentionItem("fixture", "format", "json", "json_exact", "original", '{"ready":true}')
    assert score_response(item, '{ "ready": true }')["correct"] is True
    assert score_response(item, '{"ready":true,"extra":1}')["correct"] is False
