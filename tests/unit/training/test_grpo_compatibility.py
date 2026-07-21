from __future__ import annotations

import random
import warnings
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from foundry.training import grpo_compatibility as compatibility
from foundry.training import grpo_trainer


class _FakeCuda:
    def __init__(self) -> None:
        self.state = b"cuda-before"

    def is_available(self) -> bool:
        return True

    def get_rng_state_all(self) -> list[bytes]:
        return [self.state]


class _FakeTorch:
    def __init__(self) -> None:
        self.enabled = True
        self.warn_only = False
        self.cpu_state = b"cpu-before"
        self.cuda = _FakeCuda()
        self.transitions: list[tuple[bool, bool]] = []

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self.enabled = enabled
        self.warn_only = warn_only
        self.transitions.append((enabled, warn_only))

    def get_rng_state(self) -> bytes:
        return self.cpu_state


class _FakeNumpyRandom:
    def __init__(self) -> None:
        self.keys = b"numpy-before"

    def get_state(self) -> tuple[str, bytes, int, int, float]:
        return ("MT19937", self.keys, 11, 0, 0.0)


class _GenerationOwner:
    def generate(self, inputs: object, *, generation_config: object) -> object:
        """Fixture generation_config and logits_processor generation path."""

        del inputs, generation_config
        return object()


class _TopP:
    def __call__(self, scores: object) -> object:
        """Fixture sorted_logits.softmax(dim=-1).cumsum(dim=-1)."""

        # cumulative_probs <= (1 - self.top_p)
        return scores


def _hash(value: Any) -> str:
    return compatibility.callable_source_sha256(value)


def _sampling() -> SimpleNamespace:
    return SimpleNamespace(do_sample=True, temperature=0.8, top_p=0.95, top_k=50)


def _state() -> compatibility.ModelAdapterState:
    return compatibility.ModelAdapterState(model_sha256="a" * 64, adapter_sha256="b" * 64)


_CUMSUM_WARNING = (
    "cumsum_cuda_kernel does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at "
    "https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic "
    "support for this operation. (Triggered internally at C:\\build\\Context.cpp:95.)"
)


def _contract(
    torch: _FakeTorch,
    *,
    state_probe: Any = _state,
    numpy_random: Any | None = None,
) -> compatibility.TopPWarningOnlyGenerationContract:
    return compatibility.TopPWarningOnlyGenerationContract(
        torch_module=torch,
        generation_owner=_GenerationOwner,
        top_p_call=_TopP.__call__,
        state_probe=state_probe,
        numpy_random=_FakeNumpyRandom() if numpy_random is None else numpy_random,
        expected_generation_sha256=_hash(_GenerationOwner.generate),
        expected_top_p_sha256=_hash(_TopP.__call__),
        generation_fragments=("generation_config", "logits_processor"),
        top_p_fragments=("softmax(dim=-1).cumsum(dim=-1)", "1 - self.top_p"),
    )


def _install_generate(
    contract: compatibility.TopPWarningOnlyGenerationContract,
    torch: _FakeTorch,
    *,
    warning: str | tuple[str, ...] | None = _CUMSUM_WARNING,
    warning_category: type[Warning] = UserWarning,
    mutate_warning_filters: bool = False,
    numpy_random: _FakeNumpyRandom | None = None,
    advance_torch_rng: bool = True,
    error: BaseException | None = None,
) -> tuple[object, object]:
    original = _GenerationOwner.generate

    def fixture_generate(
        self: _GenerationOwner, inputs: object, *, generation_config: object
    ) -> object:
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert generation_config is not None
        if advance_torch_rng:
            torch.cpu_state = b"cpu-after"
            torch.cuda.state = b"cuda-after"
        if numpy_random is not None:
            numpy_random.keys = b"numpy-after"
        if mutate_warning_filters:
            warnings.filterwarnings("ignore", message="fixture-only-warning")
        emitted = (warning,) if isinstance(warning, str) else (() if warning is None else warning)
        for message in emitted:
            warnings.warn(message, category=warning_category, stacklevel=2)
        if error is not None:
            raise error
        return inputs

    type.__setattr__(_GenerationOwner, "generate", fixture_generate)
    contract.expected_generation_sha256 = _hash(fixture_generate)
    contract.generation_fragments = ("generation_config",)
    token = object()
    try:
        with contract.install():
            result = _GenerationOwner().generate(token, generation_config=_sampling())
        return token, result
    finally:
        type.__setattr__(_GenerationOwner, "generate", original)


def test_warning_only_is_limited_to_generate_and_preserves_sampling_and_result() -> None:
    torch = _FakeTorch()
    contract = _contract(torch)
    token, result = _install_generate(contract, torch)
    assert result is token
    assert torch.enabled is True
    assert torch.warn_only is False
    assert torch.transitions == [(True, True), (True, False)]
    evidence = contract.evidence()
    assert evidence["generation_calls"] == 1
    assert evidence["warning_count"] == 1
    assert evidence["all_warnings_whitelisted"] is True
    assert evidence["all_expected_warnings_present"] is True
    assert evidence["all_single_warning_class"] is True
    assert evidence["all_warning_filters_restored"] is True
    assert evidence["all_state_unchanged"] is True
    assert evidence["all_adapter_runtime_unchanged"] is True
    assert evidence["all_scaling_unchanged"] is True
    assert evidence["all_training_state_unchanged"] is True
    assert evidence["all_strict_restorations"] is True
    assert evidence["rng_restored"] is False
    assert evidence["rng_state_contract_sha256"] == compatibility.RNG_STATE_CONTRACT_SHA256
    assert len(str(evidence["evidence_sha256"])) == 64
    record = contract.call_records()[0]
    assert record.call_index == 1
    assert record.rng_advanced is True
    assert record.warning_class_ids == (compatibility.CANONICAL_WARNING_CLASS_ID,)
    assert record.distinct_warning_class_count == 1
    assert record.warning_filters_restored is True
    assert record.warning_filters_before_sha256 == record.warning_filters_after_sha256
    assert isinstance(contract.call_records(), tuple)


def test_state_probe_release_is_explicit_and_generation_bound() -> None:
    torch = _FakeTorch()
    contract = _contract(torch)
    with pytest.raises(RuntimeError, match="before generation"):
        contract.release_state_probe()
    _install_generate(contract, torch)
    contract.release_state_probe()
    assert contract.evidence()["generation_calls"] == 1
    with pytest.raises(RuntimeError, match="not bound"):
        contract.release_state_probe()


def test_exception_restores_strict_mode_method_and_records_rng_and_state() -> None:
    torch = _FakeTorch()
    contract = _contract(torch)
    original = _GenerationOwner.generate
    with pytest.raises(RuntimeError, match="fixture failure"):
        _install_generate(contract, torch, error=RuntimeError("fixture failure"))
    assert _GenerationOwner.generate is original
    assert torch.enabled is True and torch.warn_only is False
    evidence = contract.evidence()
    assert evidence["generation_calls"] == 1
    assert evidence["all_state_unchanged"] is True
    assert evidence["all_warning_filters_restored"] is True


@pytest.mark.parametrize(
    ("warning", "message", "whitelisted", "present"),
    [
        (
            "another kernel does not have a deterministic implementation",
            "non-whitelisted",
            False,
            False,
        ),
        (None, "no deterministic cumsum warning", True, False),
    ],
)
def test_warning_whitelist_and_presence_fail_closed(
    warning: str | None,
    message: str,
    whitelisted: bool,
    present: bool,
) -> None:
    torch = _FakeTorch()
    contract = _contract(torch)
    with pytest.raises(RuntimeError, match=message):
        _install_generate(contract, torch, warning=warning)
    assert torch.enabled is True and torch.warn_only is False
    assert contract.evidence()["all_warnings_whitelisted"] is whitelisted
    assert contract.evidence()["all_expected_warnings_present"] is present


def test_canonical_warning_identity_rejects_wrong_category_and_appended_kernel() -> None:
    torch = _FakeTorch()
    contract = _contract(torch)
    with pytest.raises(RuntimeError, match="non-whitelisted"):
        _install_generate(
            contract,
            torch,
            warning=_CUMSUM_WARNING,
            warning_category=RuntimeWarning,
        )
    assert contract.call_records()[0].warning_class_ids[0].startswith("unclassified:")

    torch = _FakeTorch()
    contract = _contract(torch)
    with pytest.raises(RuntimeError, match="non-whitelisted"):
        _install_generate(
            contract,
            torch,
            warning=(
                compatibility.CANONICAL_WARNING_NORMALIZED_TEXT
                + " another_cuda_kernel also lacks deterministic support."
            ),
        )


def test_multiple_distinct_normalized_warning_classes_reject_but_repetition_passes() -> None:
    torch = _FakeTorch()
    contract = _contract(torch)
    with pytest.raises(RuntimeError, match="multiple distinct normalized warning classes"):
        _install_generate(
            contract,
            torch,
            warning=(_CUMSUM_WARNING, "another kernel warning"),
        )
    assert contract.call_records()[0].distinct_warning_class_count == 2

    torch = _FakeTorch()
    contract = _contract(torch)
    _install_generate(contract, torch, warning=(_CUMSUM_WARNING, _CUMSUM_WARNING))
    record = contract.call_records()[0]
    assert record.warning_count == 2
    assert record.distinct_warning_class_count == 1
    assert record.warnings_whitelisted is True


@pytest.mark.parametrize("error", [None, RuntimeError("fixture failure")])
def test_warning_filters_are_restored_after_success_and_exception(
    error: BaseException | None,
) -> None:
    before = compatibility.warning_filters_sha256()
    torch = _FakeTorch()
    contract = _contract(torch)
    if error is None:
        _install_generate(contract, torch, mutate_warning_filters=True)
    else:
        with pytest.raises(RuntimeError, match="fixture failure"):
            _install_generate(
                contract,
                torch,
                mutate_warning_filters=True,
                error=error,
            )
    assert compatibility.warning_filters_sha256() == before
    record = contract.call_records()[0]
    assert record.warning_filters_before_sha256 == before
    assert record.warning_filters_after_sha256 == before
    assert record.warning_filters_restored is True


def test_model_or_adapter_mutation_fails_closed() -> None:
    torch = _FakeTorch()
    states = iter(
        [
            compatibility.ModelAdapterState("a" * 64, "b" * 64),
            compatibility.ModelAdapterState("a" * 64, "c" * 64),
        ]
    )
    with pytest.raises(RuntimeError, match="state changed"):
        _install_generate(_contract(torch, state_probe=lambda: next(states)), torch)


def test_disabled_adapter_rejects_before_warning_only_mode() -> None:
    torch = _FakeTorch()
    state = compatibility.ModelAdapterState(
        "a" * 64,
        "b" * 64,
        active_adapters=("default",),
        adapter_enabled=False,
    )
    contract = _contract(torch, state_probe=lambda: state)
    original = _GenerationOwner.generate
    contract.expected_generation_sha256 = _hash(original)
    with contract.install():
        with pytest.raises(RuntimeError, match="uniformly enabled"):
            _GenerationOwner().generate(object(), generation_config=_sampling())
    assert torch.transitions == []


def test_strict_entry_nested_context_and_operation_labels_reject() -> None:
    torch = _FakeTorch()
    contract = _contract(torch)
    torch.warn_only = True
    with pytest.raises(RuntimeError, match="strict deterministic entry"):
        with contract.install():
            pass
    torch.warn_only = False
    for label in ("backward", "optimizer", "loss"):
        with pytest.raises(ValueError, match="only for generation"):
            with contract.install(label):
                pass
    with contract.install():
        with pytest.raises(RuntimeError, match="nested"):
            with contract.install():
                pass


def test_source_and_sampling_drift_reject_before_policy_work() -> None:
    torch = _FakeTorch()
    contract = _contract(torch)
    contract.expected_top_p_sha256 = "0" * 64
    with pytest.raises(ValueError, match="top-p warper source hash differs"):
        with contract.install():
            pass
    contract = _contract(torch)
    original = _GenerationOwner.generate

    def bad_sampling_generate(
        self: _GenerationOwner, inputs: object, *, generation_config: object
    ) -> object:
        del self, generation_config
        warnings.warn(
            "cumsum_cuda_kernel does not have a deterministic implementation", stacklevel=2
        )
        return inputs

    type.__setattr__(_GenerationOwner, "generate", bad_sampling_generate)
    contract.expected_generation_sha256 = _hash(bad_sampling_generate)
    contract.generation_fragments = ("generation_config",)
    try:
        with contract.install():
            with pytest.raises(ValueError, match="sampling values differ"):
                _GenerationOwner().generate(
                    object(),
                    generation_config=SimpleNamespace(
                        do_sample=True, temperature=0.8, top_p=1.0, top_k=50
                    ),
                )
    finally:
        type.__setattr__(_GenerationOwner, "generate", original)


def test_frozen_hashes_and_evidence_are_deterministic() -> None:
    assert len(compatibility.WARNING_NORMALIZATION_SHA256) == 64
    assert len(compatibility.WARNING_CLASSIFICATION_SHA256) == 64
    assert len(compatibility.WARNING_WHITELIST_SHA256) == 64
    assert len(compatibility.RNG_STATE_CONTRACT_SHA256) == 64
    assert len(compatibility.SAMPLING_CONTRACT_SHA256) == 64
    assert len(compatibility.FIXTURE_SHA256) == 64
    assert compatibility.source_contract_sha256() == compatibility.source_contract_sha256()
    assert compatibility.normalize_warning_message("  CUMSUM_CUDA_KERNEL\n warning  ") == (
        "cumsum_cuda_kernel warning"
    )
    assert (
        compatibility.normalize_warning_message(
            "cumsum_cuda_kernel does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'. "
            "You can file an issue at https://example.invalid. "
            "(Triggered internally at C:\\build\\Context.cpp:95.)"
        )
        == compatibility.CANONICAL_WARNING_NORMALIZED_TEXT
    )


class _BoolRows:
    def __init__(self, values: list[bool]) -> None:
        self.values = values

    def detach(self) -> _BoolRows:
        return self

    def cpu(self) -> _BoolRows:
        return self

    def tolist(self) -> list[bool]:
        return self.values


class _EqualityRows:
    def any(self, *, dim: int) -> _BoolRows:
        assert dim == 1
        return _BoolRows([True])


class _CompletionIds:
    def __eq__(self, other: object) -> Any:
        assert other == 2
        return _EqualityRows()

    def __len__(self) -> int:
        return 1

    def size(self, dim: int) -> int:
        assert dim == 0
        return 1


class _Processing:
    eos_token_id = 2

    def batch_decode(self, values: object, *, skip_special_tokens: bool) -> list[str]:
        del values
        assert skip_special_tokens is True
        return ["ok"]


class _BaseTrainer:
    def __init__(self) -> None:
        self.processing_class = _Processing()

    def _generate_and_score_completions(self, inputs: object) -> object:
        del inputs
        self.processing_class.batch_decode(_CompletionIds(), skip_special_tokens=True)
        return object()


def test_trainer_optional_generation_scope_wraps_stock_path_and_restores() -> None:
    events: list[str] = []

    class Scope:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, *args: object) -> None:
            del args
            events.append("exit")

    method_hash = grpo_trainer.callable_source_sha256(_BaseTrainer._generate_and_score_completions)
    trainer_type = grpo_trainer.make_truncation_aware_grpo_trainer(
        _BaseTrainer,
        expected_method_sha256=method_hash,
        required_fragments=("batch_decode",),
        generation_scope_factory=lambda: Scope(),
    )
    trainer_type()._generate_and_score_completions({})
    assert events == ["enter", "exit"]


def test_unconfigured_trainer_scope_remains_a_noop() -> None:
    method_hash = grpo_trainer.callable_source_sha256(_BaseTrainer._generate_and_score_completions)
    trainer_type = grpo_trainer.make_truncation_aware_grpo_trainer(
        _BaseTrainer,
        expected_method_sha256=method_hash,
        required_fragments=("batch_decode",),
        generation_scope_factory=lambda: nullcontext(),
    )
    assert trainer_type()._generate_and_score_completions({}) is not None


def test_rng_hash_captures_python_numpy_and_torch_states_without_restoring() -> None:
    torch = _FakeTorch()
    numpy_random = _FakeNumpyRandom()
    prior_python_state = random.getstate()
    try:
        random.seed(7)
        before_python = compatibility.rng_state_sha256(
            torch,
            numpy_random=numpy_random,
        )
        random.random()
        assert compatibility.rng_state_sha256(torch, numpy_random=numpy_random) != before_python

        random.seed(7)
        before_numpy = compatibility.rng_state_sha256(
            torch,
            numpy_random=numpy_random,
        )
        numpy_random.keys = b"numpy-after"
        assert compatibility.rng_state_sha256(torch, numpy_random=numpy_random) != before_numpy
        assert numpy_random.keys == b"numpy-after"
    finally:
        random.setstate(prior_python_state)


def test_per_generation_rng_transition_includes_numpy_without_restoring() -> None:
    torch = _FakeTorch()
    numpy_random = _FakeNumpyRandom()
    contract = _contract(torch, numpy_random=numpy_random)
    _install_generate(
        contract,
        torch,
        numpy_random=numpy_random,
        advance_torch_rng=False,
    )
    record = contract.call_records()[0]
    assert record.rng_advanced is True
    assert record.rng_before_sha256 != record.rng_after_sha256
    assert numpy_random.keys == b"numpy-after"


def test_model_adapter_probe_captures_runtime_scaling_and_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = SimpleNamespace(training=True)
    lora = SimpleNamespace(training=False, scaling={"default": 2.0})

    class Model:
        def get_model_status(self) -> SimpleNamespace:
            return SimpleNamespace(enabled=True, active_adapters=["default"])

        def named_modules(self) -> list[tuple[str, object]]:
            return [("", root), ("layer", lora)]

    monkeypatch.setattr(compatibility, "base_parameter_signature_sha256", lambda model: "a" * 64)
    monkeypatch.setattr(
        compatibility,
        "adapter_state_sha256",
        lambda model, adapter_name: "b" * 64,
    )
    state = compatibility.model_adapter_state(Model())
    assert state.active_adapters == ("default",)
    assert state.adapter_enabled is True
    assert len(state.scaling_sha256) == 64
    assert len(state.training_state_sha256) == 64


def test_pinned_cuda_cumsum_fails_strictly_and_executes_warning_only() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the pinned cumsum compatibility fixture")
    prior_enabled = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    values = torch.tensor([0.1, 0.2, 0.7], device="cuda", dtype=torch.float32)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        with pytest.raises(RuntimeError, match="cumsum_cuda_kernel.*deterministic"):
            torch.cumsum(values, dim=-1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="cumsum_cuda_kernel.*deterministic"):
            result = torch.cumsum(values, dim=-1)
        assert result.shape == values.shape
    finally:
        torch.use_deterministic_algorithms(prior_enabled, warn_only=prior_warn_only)
