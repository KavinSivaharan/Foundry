from pathlib import Path

from foundry.synthesis.realization.compact_smoke_contract import load_compact_smoke_config
from foundry.synthesis.realization.stronger_model_contract import (
    load_stronger_model_config,
)


def test_stronger_model_config_changes_only_the_model_and_artifact_paths() -> None:
    comparison = load_stronger_model_config(
        Path("configs/synthesis/local_realization_stronger_model_micro.yaml")
    )
    base = load_compact_smoke_config(Path("configs/synthesis/local_realization_compact_micro.yaml"))
    compact = comparison.compact
    assert compact.model.repo_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert compact.model.revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert compact.generation.num_beams == 3
    assert compact.generation.num_return_sequences == 3
    assert compact.generation.max_new_tokens == 384
    assert compact.generation.seed == 5172026
    assert compact.ir_master_seed == "foundry-m5c-compact-ir-master-20260718-v1"
    assert compact.compact_system_prompt_sha256 == base.compact_system_prompt_sha256
    assert compact.compact_user_protocol_sha256 == base.compact_user_protocol_sha256
    assert compact.compact_combined_protocol_sha256 == base.compact_combined_protocol_sha256
    assert comparison.memory_probe.peak_reserved_vram_limit_bytes == 9_932_111_872
    assert comparison.memory_probe.minimum_free_vram_bytes == 536_870_912
    assert (
        comparison.combined_experiment_sha256
        == "e09e1acab6d6ee2c2f0d04032fb3808326a14fc443c43979757356702c2fb377"
    )
