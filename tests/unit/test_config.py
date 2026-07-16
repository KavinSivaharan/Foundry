from pathlib import Path

import pytest
import yaml

from foundry.config import ConfigError, load_config

CONFIG_PATH = Path("configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml")


def test_approved_config_is_fully_pinned() -> None:
    config = load_config(CONFIG_PATH)

    assert config.model.repo_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.model.revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert config.dataset.repo_id == "ScaleAI/gsm1k"
    assert config.dataset.revision == "bc09569d09a614b9b530edc7f076fb214ac10493"
    assert len(config.sha256) == 64


def test_unpinned_model_revision_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"]["revision"] = "main"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="immutable 40-character"):
        load_config(path)


def test_sampling_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["generation"]["do_sample"] = True
    path = tmp_path / "sampling.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="do_sample must be false"):
        load_config(path)


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"]["trust_remote_code"] = True
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)
