from collections.abc import Mapping
from typing import Any

DIAGNOSTIC_ID: str
FREEZE_FILE: str

def _verify_self_hash(value: Mapping[str, object], key: str) -> None: ...
def _parameter_partitions(
    model: Any,
) -> tuple[
    tuple[tuple[str, Any], ...],
    tuple[tuple[str, Any], ...],
    tuple[tuple[str, Any], ...],
]: ...
def _tensor_list(value: Any) -> list[Any]: ...
def _policy_logprobs(trainer: Any, result: Mapping[str, Any], torch: Any) -> Any: ...
def main() -> int: ...
