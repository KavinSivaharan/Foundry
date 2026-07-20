from pathlib import Path

from foundry.training.retention import RetentionItem, load_suite, score_response

SUITE = Path("configs/training/assistant_only_v3_retention_suite.json")


def test_frozen_retention_suite_has_required_balance() -> None:
    suite = load_suite(SUITE)
    assert len(suite.items) == 60
    assert sum(item.section == "arithmetic" for item in suite.items) == 30
    assert sum(item.section == "format" for item in suite.items) == 15
    assert sum(item.section == "instruction" for item in suite.items) == 15
    assert len(suite.suite_sha256) == 64
    assert len(suite.prompt_sha256) == 64


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
