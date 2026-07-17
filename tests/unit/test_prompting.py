from pathlib import Path

import pytest

from foundry.config import load_config
from foundry.evaluation.prompting import prompt_sha256, render_messages

CONFIG_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml")
FORMAT_V1_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_format_v1.yaml")
FORMAT_V2_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_format_v2.yaml")


def test_prompt_rendering_is_stable() -> None:
    config = load_config(CONFIG_PATH)
    question = "A box has 3 rows of 4 pencils. How many pencils are there?"

    first = render_messages(config.prompt, question)
    second = render_messages(config.prompt, question)

    assert first == second
    assert first[0]["role"] == "system"
    assert first[1]["role"] == "user"
    assert first[1]["content"].count(question) == 1
    assert first[1]["content"].endswith("Final answer: <integer>")
    assert (
        prompt_sha256(config.prompt)
        == "738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350"
    )


@pytest.mark.parametrize(
    ("path", "expected_hash", "expected_user_content"),
    [
        (
            FORMAT_V1_PATH,
            "de85fb299156e284d34f51f74983a19b564fd5725d000bc0dac10186e274fcbc",
            (
                "Solve this problem.\n\n"
                "[QUESTION]\n\n"
                "You may show your reasoning. Your final line must contain only:\n"
                "Final answer: <integer>"
            ),
        ),
        (
            FORMAT_V2_PATH,
            "a17f10b85f491b865b2c9cc8e4b0b9f2550eae13259f308502591023f1fa9324",
            (
                "Solve this problem.\n\n"
                "[QUESTION]\n\n"
                "You may show concise reasoning.\n"
                "Before you finish, output exactly one final line in this exact format:\n"
                "Final answer: <integer>\n"
                "Replace <integer> with the integer answer.\n"
                "Do not use a boxed answer, units, currency symbols, Markdown, or any other "
                "words on the final line.\n"
                "Do not write anything after the final line."
            ),
        ),
    ],
)
def test_calibration_prompt_variants_have_exact_rendering_and_hash(
    path: Path,
    expected_hash: str,
    expected_user_content: str,
) -> None:
    config = load_config(path)
    messages = render_messages(config.prompt, "[QUESTION]")

    assert messages[1]["content"] == expected_user_content
    assert prompt_sha256(config.prompt) == expected_hash
