"""Single-base dual-adapter reference mechanism for Milestone 14A."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foundry.training.config import canonical_sha256
from foundry.training.grpo_replay_evidence import tensor_evidence

REFERENCE_MECHANISM_ID = "foundry-l3-starting-policy-reference-v1"
POLICY_ADAPTER_NAME = "default"
REFERENCE_ADAPTER_NAME = "reference"
EXPECTED_ADAPTER_TENSORS = 112
EXPECTED_ADAPTER_PARAMETERS = 1_089_536
EXPECTED_LAYERS = tuple(range(14, 28))
EXPECTED_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _adapter_marker(adapter_name: str) -> str:
    return f".{adapter_name}."


def _is_adapter_parameter(name: str, adapter_name: str) -> bool:
    return "lora_" in name and _adapter_marker(adapter_name) in name


def _normalized_name(name: str, adapter_name: str) -> str:
    marker = _adapter_marker(adapter_name)
    if marker not in name:
        raise ValueError("LoRA tensor name does not contain its adapter name")
    return name.replace(marker, ".<adapter>.")


def capture_adapter_state(model: object, adapter_name: str) -> dict[str, object]:
    """Hash one adapter independently of its PEFT runtime name."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model does not expose named_parameters()")
    rows: list[dict[str, object]] = []
    parameter_count = 0
    trainable_count = 0
    for raw_name, parameter in sorted(named_parameters(), key=lambda item: str(item[0])):
        name = str(raw_name)
        if not _is_adapter_parameter(name, adapter_name):
            continue
        evidence = tensor_evidence(parameter).as_dict()
        numel = int(parameter.numel())
        parameter_count += numel
        trainable_count += int(bool(parameter.requires_grad))
        rows.append(
            {
                "name": _normalized_name(name, adapter_name),
                "dtype": evidence["dtype"],
                "shape": evidence["shape"],
                "sha256": evidence["sha256"],
                "numel": numel,
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    if len(rows) != EXPECTED_ADAPTER_TENSORS or parameter_count != EXPECTED_ADAPTER_PARAMETERS:
        raise RuntimeError("dual-adapter tensor inventory differs from frozen L3")
    tensor_rows = [
        {
            "name": row["name"],
            "dtype": row["dtype"],
            "shape": row["shape"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    return {
        "adapter_name": adapter_name,
        "tensor_count": len(rows),
        "parameter_count": parameter_count,
        "trainable_tensor_count": trainable_count,
        "tensors": rows,
        "normalized_tensor_state_sha256": canonical_sha256(tensor_rows),
        "runtime_state_sha256": canonical_sha256(rows),
    }


def assert_policy_reference_identity(
    model: object,
    *,
    require_policy_trainable: bool,
) -> dict[str, object]:
    """Require byte-identical adapters with ownership only on the policy."""

    policy = capture_adapter_state(model, POLICY_ADAPTER_NAME)
    reference = capture_adapter_state(model, REFERENCE_ADAPTER_NAME)
    if policy["normalized_tensor_state_sha256"] != reference["normalized_tensor_state_sha256"]:
        raise RuntimeError("policy and reference adapters are not byte-identical")
    expected_policy_trainable = EXPECTED_ADAPTER_TENSORS if require_policy_trainable else 0
    if policy["trainable_tensor_count"] != expected_policy_trainable:
        raise RuntimeError("policy trainability differs from the reference contract")
    if reference["trainable_tensor_count"] != 0:
        raise RuntimeError("reference adapter unexpectedly owns gradients")
    return {
        "policy": policy,
        "reference": reference,
        "byte_identical": True,
        "reference_frozen": True,
        "policy_only_trainable": require_policy_trainable,
        "identity_sha256": policy["normalized_tensor_state_sha256"],
    }


def set_policy_active(model: Any) -> None:
    """Activate the policy adapter and reassert exact trainability ownership."""

    model.set_adapter(POLICY_ADAPTER_NAME)
    for name, parameter in model.named_parameters():
        if _is_adapter_parameter(str(name), REFERENCE_ADAPTER_NAME):
            parameter.requires_grad_(False)
        elif _is_adapter_parameter(str(name), POLICY_ADAPTER_NAME):
            parameter.requires_grad_(True)
        elif "lora_" not in str(name):
            parameter.requires_grad_(False)


def set_reference_active_frozen(model: Any) -> None:
    """Activate the reference adapter while keeping every tensor frozen."""

    model.set_adapter(REFERENCE_ADAPTER_NAME)
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(False)


def active_adapter_name(model: object) -> str:
    """Return one active PEFT adapter name and reject mixed activation."""

    value = getattr(model, "active_adapter", None)
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise RuntimeError("model does not expose exactly one active adapter")


@dataclass
class SharedStartingPolicyReference:
    """Callable reference proxy sharing the policy model's frozen full base."""

    model: Any
    torch_module: Any
    call_count: int = 0
    reference_state_sha256: str | None = None
    last_policy_state_sha256: str | None = None

    def __post_init__(self) -> None:
        if active_adapter_name(self.model) != POLICY_ADAPTER_NAME:
            raise RuntimeError("reference proxy requires the policy adapter to be active")
        identity = assert_policy_reference_identity(self.model, require_policy_trainable=True)
        self.reference_state_sha256 = str(
            cast_dict(identity["reference"])["normalized_tensor_state_sha256"]
        )
        self.last_policy_state_sha256 = str(
            cast_dict(identity["policy"])["normalized_tensor_state_sha256"]
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if active_adapter_name(self.model) != POLICY_ADAPTER_NAME:
            raise RuntimeError("reference call did not begin from the active policy")
        policy_before = capture_adapter_state(self.model, POLICY_ADAPTER_NAME)
        reference_before = capture_adapter_state(self.model, REFERENCE_ADAPTER_NAME)
        if reference_before["normalized_tensor_state_sha256"] != self.reference_state_sha256:
            raise RuntimeError("reference adapter changed before its forward pass")
        try:
            set_reference_active_frozen(self.model)
            if active_adapter_name(self.model) != REFERENCE_ADAPTER_NAME:
                raise RuntimeError("reference adapter switch did not take effect")
            with self.torch_module.no_grad():
                output = self.model(*args, **kwargs)
            if _requires_grad(output):
                raise RuntimeError("reference forward retained an autograd graph")
        finally:
            set_policy_active(self.model)
        if active_adapter_name(self.model) != POLICY_ADAPTER_NAME:
            raise RuntimeError("policy adapter was not restored after reference scoring")
        reference_after = capture_adapter_state(self.model, REFERENCE_ADAPTER_NAME)
        if reference_after["normalized_tensor_state_sha256"] != self.reference_state_sha256:
            raise RuntimeError("reference adapter changed during reference scoring")
        self.call_count += 1
        self.last_policy_state_sha256 = str(policy_before["normalized_tensor_state_sha256"])
        return output

    def evidence(self) -> dict[str, object]:
        if self.call_count <= 0:
            raise RuntimeError("reference proxy was never used")
        payload: dict[str, object] = {
            "mechanism_id": REFERENCE_MECHANISM_ID,
            "implementation": "one_quantized_base_with_frozen_and_policy_l3_adapters",
            "policy_adapter_name": POLICY_ADAPTER_NAME,
            "reference_adapter_name": REFERENCE_ADAPTER_NAME,
            "reference_forward_no_grad": True,
            "reference_optimizer_owned": False,
            "second_full_base_model": False,
            "adapter_switch_restores_policy": True,
            "call_count": self.call_count,
            "reference_state_sha256": self.reference_state_sha256,
            "last_policy_state_sha256": self.last_policy_state_sha256,
        }
        payload["runtime_evidence_sha256"] = canonical_sha256(payload)
        return payload


def cast_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("reference evidence must be an object")
    return value


def _requires_grad(value: object) -> bool:
    requires_grad = getattr(value, "requires_grad", None)
    if isinstance(requires_grad, bool):
        return requires_grad
    if isinstance(value, dict):
        return any(_requires_grad(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_requires_grad(item) for item in value)
    for attribute in ("logits", "loss", "hidden_states"):
        if hasattr(value, attribute) and _requires_grad(getattr(value, attribute)):
            return True
    return False


def trl_per_token_kl(reference_logp: float, policy_logp: float) -> float:
    """Return TRL 0.17's non-negative per-token KL estimator."""

    if not math.isfinite(reference_logp) or not math.isfinite(policy_logp):
        raise ValueError("reference and policy log probabilities must be finite")
    delta = reference_logp - policy_logp
    value = math.exp(delta) - delta - 1.0
    if value < -1e-12 or not math.isfinite(value):
        raise RuntimeError("per-token KL estimator is invalid")
    return max(0.0, value)


def reference_mechanism_contract() -> dict[str, object]:
    """Freeze model-independent semantics and source identities."""

    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    zero = trl_per_token_kl(-2.0, -2.0)
    positive = trl_per_token_kl(-2.0, -2.25)
    if zero != 0.0 or positive <= 0.0:
        raise RuntimeError("reference KL calibration differs")
    payload: dict[str, object] = {
        "schema_version": 1,
        "mechanism_id": REFERENCE_MECHANISM_ID,
        "implementation": "one_quantized_base_with_two_l3_lora_adapters",
        "policy_adapter_name": POLICY_ADAPTER_NAME,
        "reference_adapter_name": REFERENCE_ADAPTER_NAME,
        "adapters_initialize_byte_identically": True,
        "reference_trainable": False,
        "reference_no_grad": True,
        "reference_optimizer_owned": False,
        "policy_only_optimizer_owned": True,
        "reference_kl_orientation": "frozen_l3_reference_to_active_l3_policy",
        "second_full_base_model": False,
        "expected_adapter_tensors_each": EXPECTED_ADAPTER_TENSORS,
        "expected_adapter_parameters_each": EXPECTED_ADAPTER_PARAMETERS,
        "expected_layers": list(EXPECTED_LAYERS),
        "expected_projections": list(EXPECTED_PROJECTIONS),
        "zero_identity_kl": zero,
        "positive_controlled_perturbation_kl": positive,
        "adapter_switch_exception_safe": True,
        "offline_policy_save_reload_required": True,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "shared_reference_call_source_sha256": hashlib.sha256(
            inspect.getsource(SharedStartingPolicyReference.__call__).encode("utf-8")
        ).hexdigest(),
    }
    payload["reference_mechanism_sha256"] = canonical_sha256(payload)
    return payload
