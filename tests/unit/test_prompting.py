from pathlib import Path

from foundry.config import load_config
from foundry.evaluation.prompting import prompt_sha256, render_messages

CONFIG_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml")


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
    assert prompt_sha256(config.prompt) == prompt_sha256(config.prompt)
