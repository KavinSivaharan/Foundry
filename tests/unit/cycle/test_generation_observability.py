from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from foundry.cycle.generation_observability import (
    FAILURE_PHASES,
    OBSERVABILITY_CONTRACT_ID,
    RECOVERY_EXECUTION_ID,
    SCIENTIFIC_CYCLE_ID,
    AttemptIdentity,
    GenerationEvidenceError,
    attempt_manifest,
    content_free_attempt_projection,
    exception_evidence,
    generation_state,
    inspect_prior_runtime,
    map_failure_phase,
    observability_contract,
    observability_fixture,
    persist_attempt_failure,
    persist_attempt_success,
    recovery_identity,
    rng_state_sha256,
    warning_evidence,
)
from foundry.training.config import canonical_sha256

REPO = Path(__file__).parents[3]


def _identity() -> AttemptIdentity:
    return AttemptIdentity(
        recovery_execution_id=RECOVERY_EXECUTION_ID,
        scientific_cycle_id=SCIENTIFIC_CYCLE_ID,
        process_role="diagnostic_generation",
        prompt_position_index=0,
        source_id_sha256="1" * 64,
        prompt_sha256="2" * 64,
        completion_index=0,
        prompt_subseed=20260720,
        model_revision="3" * 40,
        starting_adapter_sha256="4" * 64,
        controller_source_commit="5" * 40,
        controller_source_tree="6" * 40,
        python_import_root="C:/frozen/src",
        interpreter_sha256="7" * 64,
        environment_sha256="8" * 64,
    )


def _raised(message: str = "exact backend failure") -> RuntimeError:
    try:
        raise RuntimeError(message)
    except RuntimeError as error:
        return error


def _state() -> dict[str, Any]:
    return {
        "generation_arguments_sha256": "9" * 64,
        "generation_config_sha256": "a" * 64,
        "input_id_shape": [1, 12],
        "input_token_count": 12,
        "cuda_available": True,
    }


def _parameter_state() -> dict[str, str]:
    return {
        "base_state_sha256": "b" * 64,
        "adapter_state_sha256": "c" * 64,
        "model_state_sha256": "d" * 64,
    }


def _warning_payload() -> dict[str, Any]:
    return {"count": 0, "warnings": [], "warning_projection_sha256": "e" * 64}


def _persist_failure(tmp_path: Path, error: RuntimeError | None = None) -> dict[str, Any]:
    return persist_attempt_failure(
        evidence_root=tmp_path,
        identity=_identity(),
        state=_state(),
        parameter_state_before=_parameter_state(),
        parameter_state_after=_parameter_state(),
        rng_before_sha256="f" * 64,
        rng_after_sha256="0" * 64,
        warnings_payload=_warning_payload(),
        error=error or _raised(),
        active_phase="generation_forward",
        source_root=REPO / "src",
    )


def test_failure_phase_ids_are_exact_and_complete() -> None:
    assert FAILURE_PHASES == (
        "model_load",
        "adapter_load",
        "tokenizer_encode",
        "generation_prepare",
        "generation_forward",
        "sampling",
        "decode",
        "output_validation",
        "verifier",
        "persistence",
        "invalid_or_ambiguous",
    )


@pytest.mark.parametrize(
    "phase",
    [value for value in FAILURE_PHASES if value != "invalid_or_ambiguous"],
)
def test_each_explicit_generation_phase_maps_to_itself(phase: str) -> None:
    if phase == "generation_forward":
        assert map_failure_phase(phase, []) == "generation_forward"
    else:
        assert map_failure_phase(phase, []) == phase


def test_sampling_and_cache_stack_frames_refine_forward_phase() -> None:
    assert map_failure_phase("generation_forward", [{"function": "multinomial"}]) == "sampling"
    assert (
        map_failure_phase(
            "generation_forward",
            [{"function": "prepare_inputs_for_generation"}],
        )
        == "generation_prepare"
    )


def test_runtime_error_message_and_traceback_are_persisted_exactly(tmp_path: Path) -> None:
    packet = _persist_failure(tmp_path)
    exception = packet["exception"]
    assert exception["exception_class"] == "RuntimeError"
    assert exception["exception_message"] == "exact backend failure"
    assert "RuntimeError: exact backend failure" in exception["traceback"]
    path = next(tmp_path.glob("*.json"))
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == packet


def test_failure_packet_is_written_before_the_original_error_is_rethrown(
    tmp_path: Path,
) -> None:
    error = _raised("rethrow me")
    with pytest.raises(RuntimeError, match="rethrow me"):
        _persist_failure(tmp_path, error)
        raise error
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_chained_exception_cause_and_context_are_preserved() -> None:
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as error:
        evidence = exception_evidence(
            error,
            active_phase="generation_forward",
            source_root=REPO / "src",
        )
    assert evidence["chained_cause"]["class"] == "ValueError"
    assert evidence["chained_context"]["class"] == "ValueError"


def test_failed_attempt_cannot_become_empty_success_or_enter_downstream(
    tmp_path: Path,
) -> None:
    packet = _persist_failure(tmp_path)
    assert packet["outcome"] == "failure"
    assert packet["output_token_count"] == 0
    assert packet["output_token_ids_sha256"] is None
    assert packet["verifier_called"] is False
    assert packet["training_record_created"] is False


def test_success_packet_has_output_tokens_and_empty_exception(tmp_path: Path) -> None:
    packet = persist_attempt_success(
        evidence_root=tmp_path,
        identity=_identity(),
        state=_state(),
        rng_before_sha256="f" * 64,
        rng_after_sha256="0" * 64,
        warnings_payload=_warning_payload(),
        token_ids=[10, 11],
    )
    assert packet["outcome"] == "success"
    assert packet["exception"] is None
    assert packet["output_token_count"] == 2
    assert packet["output_token_ids_sha256"] == canonical_sha256([10, 11])


def test_prompt_text_is_absent_and_tracked_projection_is_content_free(
    tmp_path: Path,
) -> None:
    packet = _persist_failure(tmp_path)
    serialized = json.dumps(packet)
    assert '"prompt"' not in serialized
    projection = content_free_attempt_projection(packet)
    projected = json.dumps(projection)
    assert "exact backend failure" not in projected
    assert '"traceback"' not in projected
    assert '"source_path"' not in projected
    assert projection["exception"]["exception_message_sha256"]


def test_attempt_manifest_reconstructs_packet_self_hashes(tmp_path: Path) -> None:
    packet = _persist_failure(tmp_path)
    manifest = attempt_manifest(tmp_path)
    assert manifest["attempt_count"] == 1
    assert manifest["failures"] == 1
    assert manifest["attempts"][0]["attempt_evidence_sha256"] == packet["attempt_evidence_sha256"]


def test_generation_arguments_are_not_mutated_by_state_capture() -> None:
    arguments = {"temperature": 0.8, "top_p": 0.95, "use_cache": True}
    before = json.loads(json.dumps(arguments))
    generation_state(
        torch=None,
        psutil=None,
        model=None,
        input_ids=None,
        attention_mask=None,
        generation_arguments=arguments,
        generation_config_sha256="a" * 64,
    )
    assert arguments == before


def test_failure_integrity_requires_unchanged_model_adapter_and_base(
    tmp_path: Path,
) -> None:
    packet = _persist_failure(tmp_path)
    assert packet["post_failure_integrity"] == {
        "model_state_hash_unchanged": True,
        "adapter_state_hash_unchanged": True,
        "base_state_hash_unchanged": True,
    }


def test_changed_adapter_state_fails_closed(tmp_path: Path) -> None:
    changed = {**_parameter_state(), "adapter_state_sha256": "1" * 64}
    with pytest.raises(GenerationEvidenceError, match="changed model or adapter"):
        persist_attempt_failure(
            evidence_root=tmp_path,
            identity=_identity(),
            state=_state(),
            parameter_state_before=_parameter_state(),
            parameter_state_after=changed,
            rng_before_sha256="f" * 64,
            rng_after_sha256="0" * 64,
            warnings_payload=_warning_payload(),
            error=_raised(),
            active_phase="generation_forward",
            source_root=REPO / "src",
        )


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_rng_state_all() -> list[Any]:
        return []

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "Fake CUDA"

    @staticmethod
    def memory_allocated() -> int:
        return 101

    @staticmethod
    def memory_reserved() -> int:
        return 202

    @staticmethod
    def max_memory_allocated() -> int:
        return 303

    @staticmethod
    def max_memory_reserved() -> int:
        return 404


class _FakeState:
    def cpu(self) -> _FakeState:
        return self

    def tolist(self) -> list[int]:
        return [1, 2, 3]


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def get_rng_state() -> _FakeState:
        return _FakeState()

    @staticmethod
    def are_deterministic_algorithms_enabled() -> bool:
        return True

    @staticmethod
    def is_autocast_enabled(device: str | None = None) -> bool:
        assert device in {None, "cuda"}
        return False

    @staticmethod
    def is_grad_enabled() -> bool:
        return False


class _FakeTensor:
    shape = (1, 12)
    device = "cuda:0"
    dtype = "torch.int64"

    @staticmethod
    def numel() -> int:
        return 12

    def count_nonzero(self) -> Any:
        return SimpleNamespace(item=lambda: 12)


class _FakeModel:
    training = False
    _disable_adapters = False
    hf_device_map = {"": 0}
    config = SimpleNamespace(quantization_config={"load_in_4bit": True})
    generation_config = SimpleNamespace(cache_implementation=None)

    def parameters(self) -> Any:
        return iter([SimpleNamespace(device="cuda:0", dtype="torch.float16")])

    @staticmethod
    def active_adapters() -> list[str]:
        return ["default"]


def test_rng_cuda_device_and_memory_evidence_are_recorded() -> None:
    assert rng_state_sha256(_FakeTorch()) is not None
    state = generation_state(
        torch=_FakeTorch(),
        psutil=SimpleNamespace(
            Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=505))
        ),
        model=_FakeModel(),
        input_ids=_FakeTensor(),
        attention_mask=_FakeTensor(),
        generation_arguments={"use_cache": True},
        generation_config_sha256="a" * 64,
    )
    assert state["cuda_device_name"] == "Fake CUDA"
    assert state["allocated_vram_bytes"] == 101
    assert state["reserved_vram_bytes"] == 202
    assert state["process_rss_bytes"] == 505
    assert state["deterministic_algorithms"] is True


def test_warning_evidence_records_raw_and_hashed_state() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warnings.warn("observable warning", RuntimeWarning, stacklevel=1)
    evidence = warning_evidence(captured, source_root=REPO / "src")
    assert evidence["count"] == 1
    assert evidence["warnings"][0]["warning_message"] == "observable warning"
    projection = content_free_attempt_projection(evidence)
    assert "observable warning" not in json.dumps(projection)


def test_missing_traceback_evidence_fails_closed() -> None:
    with pytest.raises(GenerationEvidenceError, match="no complete traceback"):
        exception_evidence(
            RuntimeError("never raised"),
            active_phase="generation_forward",
            source_root=REPO / "src",
        )


def test_unknown_failure_phase_is_invalid_or_ambiguous() -> None:
    assert map_failure_phase("unknown", []) == "invalid_or_ambiguous"


def test_no_retry_and_no_base_fallback_are_frozen_contract_terms() -> None:
    contract = observability_contract()
    semantics = contract["failure_semantics"]
    assert semantics["automatic_retry"] is False
    assert semantics["base_model_fallback"] is False
    assert semantics["empty_completion_on_failure"] is False


def test_observability_contract_self_hash_reconstructs() -> None:
    contract = observability_contract()
    supplied = contract.pop("contract_sha256")
    assert canonical_sha256(contract) == supplied
    assert contract["contract_id"] == OBSERVABILITY_CONTRACT_ID


def test_observability_fixture_self_hash_reconstructs() -> None:
    fixture = observability_fixture()
    supplied = fixture.pop("fixture_sha256")
    assert canonical_sha256(fixture) == supplied
    assert fixture["integrity"]["retry_count"] == 0


def test_prior_rejection_inspection_detects_class_only(tmp_path: Path) -> None:
    generation = tmp_path / "compatibility/trial-1/generation"
    generation.mkdir(parents=True)
    rows = [{"backend_error_type": "RuntimeError"} for _ in range(32)]
    (generation / "candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "generation.stderr.txt").write_text("warning only\n", encoding="utf-8")
    result = inspect_prior_runtime(tmp_path)
    assert result["files_inspected"] == 2
    assert result["complete_trustworthy_traceback_exists"] is False
    candidate = next(row for row in result["files"] if row["path"].endswith("candidates.jsonl"))
    assert candidate["contains_exception_class"] is True
    assert candidate["contains_exception_message"] is False


def test_published_rejection_and_old_runtime_identity_remain_frozen() -> None:
    path = REPO / "results/phase2_vetted_corpus/milestone15a_cycle1_rejection.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    supplied = value.pop("publication_sha256")
    assert canonical_sha256(value) == supplied
    assert value["compatibility_trial"]["generation"]["backend_failures"] == 32
    assert value["resource_evidence"]["external_runtime_tree_sha256"] == (
        "5086deab27d522e939874e65c5d8d74b7d5c43a082c73e6b51398ff04b317d05"
    )


def test_observability_freeze_self_hash_and_class_only_inspection() -> None:
    path = REPO / "results/phase2_vetted_corpus/milestone15a_r1_observability_freeze.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    supplied = value.pop("observability_freeze_sha256")
    assert canonical_sha256(value) == supplied
    assert value["files_inspected"] == 37
    evidence_rows = [row for row in value["files"] if row["evidence"]]
    assert evidence_rows == [
        {
            "bytes": 35261,
            "evidence": ["exception_class"],
            "path": "compatibility/trial-1/generation/candidates.jsonl",
            "sha256": ("d6c7776b2cd89105b03aea9f6b39e52bead87394735cb0db8f3c79e8bf712187"),
        }
    ]


def test_backend_adjudication_reconstructs_and_selects_only_case_one() -> None:
    path = REPO / "results/phase2_vetted_corpus/milestone15a_r1_backend_adjudication.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    supplied = value.pop("adjudication_sha256")
    decision = dict(value["decision"])
    decision_sha256 = decision.pop("decision_sha256")

    assert canonical_sha256(value) == supplied
    assert canonical_sha256(decision) == decision_sha256
    assert value["classification"] == "general_generation_integration_defect"
    assert value["decision"]["defect_subtype"] == "stochastic_generation_context_defect"
    assert value["case_requirements"]["case_2_required"] is False
    assert value["case_requirements"]["case_3_required"] is False
    assert value["diagnostic"]["attempt_count"] == 1


def test_generation_correction_freeze_reconstructs_without_scientific_drift() -> None:
    path = REPO / "results/phase2_vetted_corpus/milestone15a_r1_generation_correction.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    supplied = value.pop("correction_sha256")

    assert canonical_sha256(value) == supplied
    assert canonical_sha256(value["correction_contract"]) == value["correction_contract_sha256"]
    assert canonical_sha256(value["regression_fixture"]) == value["regression_fixture_sha256"]
    assert value["frozen_scientific_identity"]["generation_settings_changed"] is False
    assert value["execution_binding"]["config_hash_changed"] is False
    assert (
        value["old_implementation"]["cycle_generation_source_sha256"]
        != value["corrected_implementation"]["cycle_generation_source_sha256"]
    )


def test_scientific_configuration_hash_is_unchanged_by_instrumentation() -> None:
    path = REPO / "configs/cycles/cycle1_verifier_filtered.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert canonical_sha256(payload) == (
        "7fc85c4cbb957cd425fded14eba5507d706a7eeb76fa1107c3a50c4b335bd064"
    )


def test_recovery_identity_links_the_published_parent_without_source_binding() -> None:
    value = recovery_identity(
        config_sha256="1" * 64,
        model_revision="2" * 40,
        dataset_sha256="3" * 64,
        starting_adapter_sha256="4" * 64,
        prior_runtime_tree_sha256="5" * 64,
        prior_rejection_sha256="6" * 64,
    )
    supplied = value.pop("recovery_identity_sha256")
    assert canonical_sha256(value) == supplied
    assert value["recovery_execution_id"] == RECOVERY_EXECUTION_ID
    assert "source_commit" not in value


def test_attempt_identity_contains_hashes_but_no_source_or_prompt_text() -> None:
    value = _identity().as_dict()
    assert value["source_id_sha256"] == "1" * 64
    assert value["prompt_sha256"] == "2" * 64
    assert "source_id" not in value
    assert "prompt" not in value
    assert len(value["generation_attempt_id"]) == hashlib.sha256().digest_size * 2
