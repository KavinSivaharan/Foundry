from pathlib import Path

import pytest

from foundry.config import load_config
from foundry.training.adapter_evaluation import PeftCudaBackend


def test_adapter_backend_rejects_hash_before_model_load(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = load_config(Path("configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml"))
    with pytest.raises(Exception, match="adapter directory hash differs"):
        PeftCudaBackend(
            config=config,
            model_path=tmp_path / "model",
            adapter_path=adapter,
            expected_adapter_sha256="0" * 64,
        )
