import json

import pytest

from foundry.phase2.retention_offline_rescore import (
    _response_payload_sha256,
    wilson_lower_bound,
)


def test_wilson_lower_bound_is_stable() -> None:
    assert wilson_lower_bound(184, 187) == pytest.approx(0.9539019690235603)
    assert wilson_lower_bound(206, 210) == pytest.approx(0.952056618075792)


def test_response_payload_hash_includes_exact_response_bytes() -> None:
    rows = [{"id": "x", "response": "answer", "response_sha256": "a" * 64}]
    assert _response_payload_sha256(rows) == _response_payload_sha256(json.loads(json.dumps(rows)))
    changed = [{**rows[0], "response": "answer "}]
    assert _response_payload_sha256(rows) != _response_payload_sha256(changed)


def test_wilson_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid Wilson"):
        wilson_lower_bound(2, 1)
    with pytest.raises(ValueError, match="invalid Wilson"):
        wilson_lower_bound(0, 0)
