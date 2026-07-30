"""Fail-closed observability for Cycle 1 generation attempts.

Raw exception messages, tracebacks, warnings, and source paths are written only
below the ignored external runtime root.  Publication code consumes the
content-free projection returned by :func:`content_free_attempt_projection`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import traceback
import unicodedata
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

OBSERVABILITY_CONTRACT_ID = "foundry-generation-backend-failure-evidence-v1"
RECOVERY_EXECUTION_ID = "foundry-cycle1-vfbon-sft-v1-r1"
SCIENTIFIC_CYCLE_ID = "foundry-cycle1-vfbon-sft-v1"
PRIOR_REJECTION_CLASSIFICATION = "pre_recovery_fail_closed_generation_rejection"

FAILURE_PHASES = (
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

_RAW_ONLY_KEYS = {
    "exception_message",
    "traceback",
    "source_path",
    "warning_message",
}
_CLASS_KEYS = {"backend_error_type", "exception_class", "error_type"}
_MESSAGE_KEYS = {"backend_error_message", "exception_message", "error_message"}
_TRACEBACK_KEYS = {
    "traceback",
    "full_traceback",
    "traceback_text",
    "traceback_sha256",
    "normalized_traceback_sha256",
}
_LOCATION_KEYS = {
    "source_location",
    "source_path",
    "stack_frames",
    "frames",
    "line_number",
}
_PHASE_KEYS = {"failure_phase", "exception_phase", "generation_phase"}
_EXIT_KEYS = {"return_code", "exit_code", "child_process_exit_code"}


class GenerationEvidenceError(RuntimeError):
    """Raised when fail-closed generation evidence is missing or ambiguous."""


@dataclass(frozen=True)
class AttemptIdentity:
    """Content-free identity for one deterministic generation attempt."""

    recovery_execution_id: str
    scientific_cycle_id: str
    process_role: str
    prompt_position_index: int
    source_id_sha256: str
    prompt_sha256: str
    completion_index: int
    prompt_subseed: int
    model_revision: str
    starting_adapter_sha256: str
    controller_source_commit: str
    controller_source_tree: str
    python_import_root: str
    interpreter_sha256: str
    environment_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation and deterministic attempt ID."""

        value = {
            "recovery_execution_id": self.recovery_execution_id,
            "scientific_cycle_id": self.scientific_cycle_id,
            "process_role": self.process_role,
            "prompt_position_index": self.prompt_position_index,
            "source_id_sha256": self.source_id_sha256,
            "prompt_sha256": self.prompt_sha256,
            "completion_index": self.completion_index,
            "prompt_subseed": self.prompt_subseed,
            "model_revision": self.model_revision,
            "starting_adapter_sha256": self.starting_adapter_sha256,
            "controller_source_commit": self.controller_source_commit,
            "controller_source_tree": self.controller_source_tree,
            "python_import_root": self.python_import_root,
            "interpreter_sha256": self.interpreter_sha256,
            "environment_sha256": self.environment_sha256,
        }
        value["generation_attempt_id"] = canonical_sha256(value)
        return value


def observability_contract() -> dict[str, Any]:
    """Return the frozen, model-free observability contract."""

    value: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": OBSERVABILITY_CONTRACT_ID,
        "failure_phases": list(FAILURE_PHASES),
        "raw_only_fields": sorted(_RAW_ONLY_KEYS),
        "failure_semantics": {
            "exception_propagates_after_persistence": True,
            "automatic_retry": False,
            "base_model_fallback": False,
            "empty_completion_on_failure": False,
            "verifier_after_generation_failure": False,
            "training_record_from_failed_attempt": False,
        },
        "normalization": {
            "exception_message": "NFKC_LF_COLLAPSED_WHITESPACE",
            "traceback": "NFKC_LF_WINDOWS_PATHS_REDACTED_COLLAPSED_HORIZONTAL_WHITESPACE",
        },
        "post_failure_integrity": {
            "model_parameter_state_unchanged": True,
            "adapter_parameter_state_unchanged": True,
            "base_parameter_state_unchanged": True,
        },
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def observability_fixture() -> dict[str, Any]:
    """Return the frozen content-free failure fixture used by unit tests."""

    value: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": OBSERVABILITY_CONTRACT_ID,
        "identity": {
            "recovery_execution_id": RECOVERY_EXECUTION_ID,
            "scientific_cycle_id": SCIENTIFIC_CYCLE_ID,
            "process_role": "diagnostic_generation",
            "prompt_position_index": 0,
            "source_id_sha256": "1" * 64,
            "prompt_sha256": "2" * 64,
            "completion_index": 0,
            "prompt_subseed": 20260720,
        },
        "failure": {
            "active_phase": "generation_forward",
            "exception_class": "RuntimeError",
            "normalized_exception_message_sha256": "3" * 64,
            "normalized_traceback_sha256": "4" * 64,
        },
        "integrity": {
            "base_state_hash_unchanged": True,
            "adapter_state_hash_unchanged": True,
            "model_state_hash_unchanged": True,
            "retry_count": 0,
            "base_model_fallback": False,
            "verifier_called": False,
            "training_record_created": False,
        },
    }
    value["fixture_sha256"] = canonical_sha256(value)
    return value


def _normalize_message(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(normalized.split())


def _normalize_traceback(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[A-Za-z]:\\[^\n\",]+", "<WINDOWS_PATH>", normalized)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_relative(path: str, source_root: Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return None


def _traceback_frames(
    traceback_value: TracebackType | None,
    source_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current = traceback_value
    while current is not None:
        frame = current.tb_frame
        source_path = frame.f_code.co_filename
        rows.append(
            {
                "module": str(frame.f_globals.get("__name__", "<unknown>")),
                "function": frame.f_code.co_name,
                "source_path": source_path,
                "source_path_sha256": _sha256_text(source_path),
                "authorized_source_relative_path": _source_relative(source_path, source_root),
                "line_number": current.tb_lineno,
            }
        )
        current = current.tb_next
    return rows


def map_failure_phase(active_phase: str, frames: Sequence[Mapping[str, Any]]) -> str:
    """Map one failure to exactly one frozen phase identifier."""

    if active_phase not in FAILURE_PHASES or active_phase == "invalid_or_ambiguous":
        return "invalid_or_ambiguous"
    if active_phase != "generation_forward":
        return active_phase
    stack = " ".join(
        f"{frame.get('module', '')} {frame.get('function', '')} {frame.get('source_path', '')}"
        for frame in frames
    ).casefold()
    if any(
        marker in stack
        for marker in (
            "multinomial",
            "_sample",
            "logits_processor",
            "logits_warper",
            "softmax",
        )
    ):
        return "sampling"
    if any(
        marker in stack
        for marker in (
            "prepare_inputs_for_generation",
            "past_key_values",
            "cache_utils",
            "_update_model_kwargs_for_generation",
        )
    ):
        return "generation_prepare"
    return "generation_forward"


def _chain_identity(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    message = str(error)
    return {
        "class": type(error).__name__,
        "message_sha256": _sha256_text(message),
        "normalized_message_sha256": _sha256_text(_normalize_message(message)),
    }


def exception_evidence(
    error: BaseException,
    *,
    active_phase: str,
    source_root: Path,
) -> dict[str, Any]:
    """Capture exact raw and normalized exception evidence."""

    message = str(error)
    traceback_text = "".join(traceback.format_exception(error))
    frames = _traceback_frames(error.__traceback__, source_root)
    phase = map_failure_phase(active_phase, frames)
    if not traceback_text or not frames:
        raise GenerationEvidenceError("generation failure has no complete traceback evidence")
    return {
        "exception_class": type(error).__name__,
        "exception_message": message,
        "exception_message_sha256": _sha256_text(message),
        "normalized_exception_message_sha256": _sha256_text(_normalize_message(message)),
        "traceback": traceback_text,
        "traceback_sha256": _sha256_text(traceback_text),
        "normalized_traceback_sha256": _sha256_text(_normalize_traceback(traceback_text)),
        "stack_frames": [
            {
                **frame,
                "exception_phase": phase,
            }
            for frame in frames
        ],
        "failure_phase": phase,
        "chained_cause": _chain_identity(error.__cause__),
        "chained_context": _chain_identity(error.__context__),
        "child_process_exit_code": None,
    }


def _safe_call(function: Any, default: Any = None) -> Any:
    try:
        return function()
    except Exception:
        return default


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return repr(value)


def _first_parameter(model: Any) -> Any | None:
    return next(iter(model.parameters()), None) if model is not None else None


def rng_state_sha256(torch: Any | None) -> str | None:
    """Hash Python, torch CPU, and all CUDA RNG states without mutating them."""

    if torch is None:
        return None
    value: dict[str, Any] = {
        "python": repr(random.getstate()),
        "torch_cpu": _safe_call(lambda: torch.get_rng_state().cpu().tolist()),
        "torch_cuda": _safe_call(
            lambda: [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()],
            [],
        ),
    }
    return canonical_sha256(value)


def parameter_state_hashes(model: Any, torch: Any) -> dict[str, str]:
    """Hash complete base and LoRA parameter bytes in deterministic name order."""

    base = hashlib.sha256()
    adapter = hashlib.sha256()
    base_count = 0
    adapter_count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        tensor = parameter.detach().contiguous()
        target = adapter if "lora_" in name else base
        target.update(name.encode("utf-8"))
        target.update(str(tuple(tensor.shape)).encode("utf-8"))
        target.update(str(tensor.dtype).encode("utf-8"))
        target.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
        if "lora_" in name:
            adapter_count += 1
        else:
            base_count += 1
    if base_count == 0 or adapter_count == 0:
        raise GenerationEvidenceError(
            "parameter-state hashing did not find base and adapter tensors"
        )
    value = {
        "base_state_sha256": base.hexdigest(),
        "adapter_state_sha256": adapter.hexdigest(),
    }
    value["model_state_sha256"] = canonical_sha256(value)
    return value


def warning_evidence(
    values: Iterable[warnings.WarningMessage],
    *,
    source_root: Path,
) -> dict[str, Any]:
    """Capture raw warnings and a stable content-free projection."""

    rows = []
    for value in values:
        message = str(value.message)
        filename = str(value.filename)
        rows.append(
            {
                "category": value.category.__name__,
                "warning_message": message,
                "message_sha256": _sha256_text(message),
                "normalized_message_sha256": _sha256_text(_normalize_message(message)),
                "source_path": filename,
                "source_path_sha256": _sha256_text(filename),
                "authorized_source_relative_path": _source_relative(filename, source_root),
                "line_number": value.lineno,
            }
        )
    projection = content_free_attempt_projection(rows)
    return {
        "count": len(rows),
        "warnings": rows,
        "warning_projection_sha256": canonical_sha256(projection),
    }


def generation_state(
    *,
    torch: Any | None,
    psutil: Any | None,
    model: Any | None,
    input_ids: Any | None,
    attention_mask: Any | None,
    generation_arguments: Mapping[str, Any],
    generation_config_sha256: str,
) -> dict[str, Any]:
    """Capture device, dtype, cache, CUDA, memory, and call-state evidence."""

    parameter = _first_parameter(model)
    model_config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    quantization = getattr(model_config, "quantization_config", None)
    model_value = cast(Any, model)
    active_adapter = (
        _safe_call(lambda: model_value.active_adapters()) if model is not None else None
    )
    if active_adapter is None:
        active_adapter = getattr(model, "active_adapter", None)
    torch_value = cast(Any, torch)
    cuda_available = bool(
        _safe_call(lambda: torch_value.cuda.is_available(), False) if torch is not None else False
    )
    memory = _safe_call(lambda: psutil.Process().memory_info()) if psutil is not None else None
    cache_value = getattr(generation_config, "cache_implementation", None)
    cache_object = getattr(model, "_cache", None)
    is_autocast = (
        bool(
            _safe_call(
                lambda: torch_value.is_autocast_enabled("cuda"),
                _safe_call(lambda: torch_value.is_autocast_enabled(), False),
            )
        )
        if torch is not None
        else False
    )
    return {
        "generation_arguments_sha256": canonical_sha256(dict(generation_arguments)),
        "generation_config_sha256": generation_config_sha256,
        "input_id_shape": list(getattr(input_ids, "shape", [])),
        "input_token_count": int(input_ids.numel()) if input_ids is not None else None,
        "attention_mask_shape": list(getattr(attention_mask, "shape", [])),
        "attention_mask_nonzero_count": (
            int(attention_mask.count_nonzero().item()) if attention_mask is not None else None
        ),
        "use_cache": bool(generation_arguments.get("use_cache")),
        "cache_implementation": _jsonable(cache_value),
        "cache_type": type(cache_object).__name__ if cache_object is not None else None,
        "model_training": getattr(model, "training", None),
        "model_eval": (not bool(model_value.training) if hasattr(model, "training") else None),
        "active_adapter": _jsonable(active_adapter),
        "adapters_enabled": (
            not bool(getattr(model, "_disable_adapters", False)) if model is not None else None
        ),
        "model_device": str(getattr(parameter, "device", "")) if parameter is not None else None,
        "input_device": str(getattr(input_ids, "device", "")) if input_ids is not None else None,
        "model_dtype": str(getattr(parameter, "dtype", "")) if parameter is not None else None,
        "input_dtype": str(getattr(input_ids, "dtype", "")) if input_ids is not None else None,
        "quantization_configuration_sha256": (
            canonical_sha256(_jsonable(quantization)) if quantization is not None else None
        ),
        "device_map": _jsonable(getattr(model, "hf_device_map", None)),
        "cuda_available": cuda_available,
        "cuda_device_name": (
            _safe_call(lambda: str(torch_value.cuda.get_device_name(0))) if cuda_available else None
        ),
        "allocated_vram_bytes": (
            int(_safe_call(lambda: torch_value.cuda.memory_allocated(), 0)) if cuda_available else 0
        ),
        "reserved_vram_bytes": (
            int(_safe_call(lambda: torch_value.cuda.memory_reserved(), 0)) if cuda_available else 0
        ),
        "peak_allocated_vram_bytes": (
            int(_safe_call(lambda: torch_value.cuda.max_memory_allocated(), 0))
            if cuda_available
            else 0
        ),
        "peak_reserved_vram_bytes": (
            int(_safe_call(lambda: torch_value.cuda.max_memory_reserved(), 0))
            if cuda_available
            else 0
        ),
        "process_rss_bytes": int(getattr(memory, "rss", 0)) if memory is not None else None,
        "deterministic_algorithms": (
            bool(
                _safe_call(
                    lambda: torch_value.are_deterministic_algorithms_enabled(),
                    False,
                )
            )
            if torch is not None
            else None
        ),
        "autocast_enabled": is_autocast if torch is not None else None,
        "gradient_enabled": (
            bool(_safe_call(lambda: torch_value.is_grad_enabled(), False))
            if torch is not None
            else None
        ),
    }


def content_free_attempt_projection(value: Any) -> Any:
    """Remove raw messages, tracebacks, warning text, and absolute source paths."""

    if isinstance(value, Mapping):
        return {
            str(key): content_free_attempt_projection(item)
            for key, item in value.items()
            if str(key) not in _RAW_ONLY_KEYS
        }
    if isinstance(value, list | tuple):
        return [content_free_attempt_projection(item) for item in value]
    return value


def _write_packet(path: Path, packet: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(packet, indent=2, sort_keys=True) + "\n")


def persist_attempt_success(
    *,
    evidence_root: Path,
    identity: AttemptIdentity,
    state: dict[str, Any],
    rng_before_sha256: str | None,
    rng_after_sha256: str | None,
    warnings_payload: dict[str, Any],
    token_ids: Sequence[int],
) -> dict[str, Any]:
    """Persist one successful attempt with an explicitly empty exception packet."""

    identity_value = identity.as_dict()
    packet: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": OBSERVABILITY_CONTRACT_ID,
        "identity": identity_value,
        "outcome": "success",
        "failure_phase": None,
        "generation_state": state,
        "exception": None,
        "rng_before_sha256": rng_before_sha256,
        "rng_after_sha256": rng_after_sha256,
        "warning_evidence": warnings_payload,
        "output_token_count": len(token_ids),
        "output_token_ids_sha256": canonical_sha256(list(token_ids)),
        "retry_count": 0,
        "base_model_fallback": False,
        "verifier_called": False,
        "training_record_created": False,
    }
    packet["attempt_evidence_sha256"] = canonical_sha256(packet)
    path = evidence_root / f"{identity_value['generation_attempt_id']}.json"
    _write_packet(path, packet)
    return packet


def persist_attempt_failure(
    *,
    evidence_root: Path,
    identity: AttemptIdentity,
    state: dict[str, Any],
    parameter_state_before: Mapping[str, str] | None,
    parameter_state_after: Mapping[str, str] | None,
    rng_before_sha256: str | None,
    rng_after_sha256: str | None,
    warnings_payload: dict[str, Any],
    error: BaseException,
    active_phase: str,
    source_root: Path,
) -> dict[str, Any]:
    """Persist complete failure evidence before the caller rethrows the error."""

    exception = exception_evidence(error, active_phase=active_phase, source_root=source_root)
    if parameter_state_before is None or parameter_state_after is None:
        integrity: dict[str, bool | None] = {
            "model_state_hash_unchanged": None,
            "adapter_state_hash_unchanged": None,
            "base_state_hash_unchanged": None,
        }
    else:
        integrity = {
            "model_state_hash_unchanged": (
                parameter_state_before["model_state_sha256"]
                == parameter_state_after["model_state_sha256"]
            ),
            "adapter_state_hash_unchanged": (
                parameter_state_before["adapter_state_sha256"]
                == parameter_state_after["adapter_state_sha256"]
            ),
            "base_state_hash_unchanged": (
                parameter_state_before["base_state_sha256"]
                == parameter_state_after["base_state_sha256"]
            ),
        }
    identity_value = identity.as_dict()
    packet: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": OBSERVABILITY_CONTRACT_ID,
        "identity": identity_value,
        "outcome": "failure",
        "failure_phase": exception["failure_phase"],
        "generation_state": state,
        "exception": exception,
        "parameter_state_before": dict(parameter_state_before or {}),
        "parameter_state_after": dict(parameter_state_after or {}),
        "post_failure_integrity": integrity,
        "rng_before_sha256": rng_before_sha256,
        "rng_after_sha256": rng_after_sha256,
        "warning_evidence": warnings_payload,
        "output_token_count": 0,
        "output_token_ids_sha256": None,
        "retry_count": 0,
        "base_model_fallback": False,
        "verifier_called": False,
        "training_record_created": False,
    }
    packet["attempt_evidence_sha256"] = canonical_sha256(packet)
    path = evidence_root / f"{identity_value['generation_attempt_id']}.json"
    _write_packet(path, packet)
    if not all(value is True for value in integrity.values() if value is not None):
        raise GenerationEvidenceError("generation failure changed model or adapter state")
    return packet


def attempt_manifest(evidence_root: Path) -> dict[str, Any]:
    """Reconstruct all attempt packets and return a content-free manifest."""

    rows = []
    for path in sorted(evidence_root.glob("*.json")):
        packet = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        supplied = packet.get("attempt_evidence_sha256")
        projected = {key: item for key, item in packet.items() if key != "attempt_evidence_sha256"}
        if supplied != canonical_sha256(projected):
            raise GenerationEvidenceError(f"attempt packet does not reconstruct: {path.name}")
        content_free = content_free_attempt_projection(packet)
        rows.append(
            {
                "file": path.name,
                "file_sha256": file_sha256(path),
                "attempt_evidence_sha256": supplied,
                "content_free_sha256": canonical_sha256(content_free),
                "outcome": packet["outcome"],
                "failure_phase": packet["failure_phase"],
            }
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": OBSERVABILITY_CONTRACT_ID,
        "attempts": rows,
        "attempt_count": len(rows),
        "failures": sum(row["outcome"] == "failure" for row in rows),
    }
    value["attempt_manifest_sha256"] = canonical_sha256(value)
    return value


def _keys_with_values(value: Any, target: set[str]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in target and item not in (None, "", [], {}):
                found.append((str(key), item))
            found.extend(_keys_with_values(item, target))
    elif isinstance(value, list):
        for item in value:
            found.extend(_keys_with_values(item, target))
    return found


def _structured_text(path: Path, text: str) -> Any | None:
    try:
        if path.suffix.casefold() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if path.suffix.casefold() == ".json":
            return json.loads(text)
    except json.JSONDecodeError:
        return None
    return None


def inspect_prior_runtime(root: Path) -> dict[str, Any]:
    """Inventory only the allowlisted immutable Milestone 15A runtime root."""

    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        is_evidence_text = (
            path.suffix.casefold() in {".json", ".jsonl", ".txt"}
            and "/adapter/" not in f"/{relative}"
        )
        text = data.decode("utf-8", errors="replace") if is_evidence_text else ""
        structured = _structured_text(path, text) if is_evidence_text else None
        if structured is not None:
            flags = [
                bool(_keys_with_values(structured, keys))
                for keys in (
                    _CLASS_KEYS,
                    _MESSAGE_KEYS,
                    _TRACEBACK_KEYS,
                    _LOCATION_KEYS,
                    _PHASE_KEYS,
                    _EXIT_KEYS,
                )
            ]
        elif is_evidence_text and path.suffix.casefold() == ".txt":
            has_class = bool(
                re.search(
                    r"(?:^|\n)(?:[\w.]+)?(?:RuntimeError|[A-Za-z]+Error):",
                    text,
                )
            )
            has_traceback = "Traceback (most recent call last)" in text
            flags = [
                has_class,
                has_class,
                has_traceback,
                has_traceback and bool(re.search(r'File "[^"]+\.py", line \d+', text)),
                False,
                bool(
                    re.search(
                        r"(?:return|exit)[ _-]?code\s*[:=]\s*-?\d+",
                        text,
                        re.IGNORECASE,
                    )
                ),
            ]
        else:
            flags = [False] * 6
        rows.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                **dict(
                    zip(
                        (
                            "contains_exception_class",
                            "contains_exception_message",
                            "contains_traceback",
                            "contains_source_location",
                            "contains_failing_phase",
                            "contains_process_exit_code",
                        ),
                        flags,
                        strict=True,
                    )
                ),
            }
        )
    value: dict[str, Any] = {
        "schema_version": 1,
        "inspection_id": "foundry-cycle1-prior-generation-error-inspection-v1",
        "prior_classification": PRIOR_REJECTION_CLASSIFICATION,
        "files_inspected": len(rows),
        "files": rows,
        "complete_trustworthy_traceback_exists": any(
            row["contains_traceback"]
            and row["contains_source_location"]
            and row["contains_exception_message"]
            for row in rows
        ),
    }
    value["inspection_manifest_sha256"] = canonical_sha256(rows)
    value["inspection_sha256"] = canonical_sha256(value)
    return value


def verify_source_identity(
    *,
    source_root: Path,
    expected_commit: str,
    expected_tree: str,
    imported_file: Path,
) -> dict[str, str]:
    """Validate one clean detached source and its active Python import root."""

    import subprocess

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return result.stdout.strip()

    if git("branch", "--show-current"):
        raise GenerationEvidenceError("observability source is not detached")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise GenerationEvidenceError("observability source worktree is not clean")
    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    if commit != expected_commit or tree != expected_tree:
        raise GenerationEvidenceError("observability source commit or tree differs")
    import_root = (source_root / "src").resolve()
    try:
        imported_file.resolve().relative_to(import_root)
    except ValueError as error:
        raise GenerationEvidenceError("Foundry was not imported from the frozen source") from error
    return {
        "commit": commit,
        "tree": tree,
        "import_root": str(import_root),
        "status": "clean",
    }


def recovery_identity(
    *,
    config_sha256: str,
    model_revision: str,
    dataset_sha256: str,
    starting_adapter_sha256: str,
    prior_runtime_tree_sha256: str,
    prior_rejection_sha256: str,
) -> dict[str, Any]:
    """Build the source-independent identity for the single R1 runtime root."""

    value: dict[str, Any] = {
        "schema_version": 1,
        "recovery_execution_id": RECOVERY_EXECUTION_ID,
        "scientific_cycle_id": SCIENTIFIC_CYCLE_ID,
        "parent_classification": PRIOR_REJECTION_CLASSIFICATION,
        "parent_runtime_tree_sha256": prior_runtime_tree_sha256,
        "parent_rejection_record_sha256": prior_rejection_sha256,
        "config_sha256": config_sha256,
        "model_revision": model_revision,
        "dataset_sha256": dataset_sha256,
        "starting_adapter_sha256": starting_adapter_sha256,
    }
    value["recovery_identity_sha256"] = canonical_sha256(value)
    return value


def ensure_recovery_runtime(root: Path, identity: dict[str, Any]) -> None:
    """Create or exactly verify the single external R1 runtime root."""

    path = root / "recovery_identity.json"
    if root.exists():
        if not path.is_file():
            raise GenerationEvidenceError("existing R1 runtime has no recovery identity")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            raise GenerationEvidenceError("existing R1 runtime belongs to another experiment")
        return
    root.mkdir(parents=False, exist_ok=False)
    _write_packet(path, identity)


def environment_sha256(environment: Mapping[str, str] | None = None) -> str:
    """Hash the exact case-normalized process environment."""

    actual = os.environ if environment is None else environment
    return canonical_sha256({str(key).upper(): value for key, value in actual.items()})
