from pathlib import Path

import pytest

from foundry.training.assistant_only_qlora import run_assistant_only_training


def test_runtime_rejects_unapproved_group_before_hardware_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="group"):
        run_assistant_only_training(
            recipe_path=Path("configs/training/assistant_only_sft_v3.yaml"),
            learning_rate=0.0002,
            model_path=tmp_path,
            lock_path=tmp_path / "lock",
            train_path=tmp_path / "train",
            validation_path=tmp_path / "validation",
            group="other",
            output_dir=tmp_path / "output",
            summary_path=tmp_path / "summary.json",
            max_steps=32,
        )
