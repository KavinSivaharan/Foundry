from foundry.training.adapter_sanity import DIAGNOSTIC_PROMPTS, diagnostic_prompt_sha256


def test_diagnostic_prompts_are_original_and_bounded() -> None:
    assert len(DIAGNOSTIC_PROMPTS) == 6
    assert len(set(DIAGNOSTIC_PROMPTS)) == 6
    assert all(prompt.strip() == prompt for prompt in DIAGNOSTIC_PROMPTS)
    assert all(
        "GSM" not in prompt and "benchmark" not in prompt.lower() for prompt in DIAGNOSTIC_PROMPTS
    )


def test_diagnostic_prompt_hash_is_stable() -> None:
    assert diagnostic_prompt_sha256() == diagnostic_prompt_sha256()
    assert len(diagnostic_prompt_sha256()) == 64
