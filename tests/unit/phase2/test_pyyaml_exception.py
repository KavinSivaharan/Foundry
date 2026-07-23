from __future__ import annotations

import pytest

from foundry.phase2.pyyaml_exception import (
    EXCEPTION_ID,
    EXPECTED_DISCREPANCY,
    validate_evidence,
    validate_narrow_exception,
)
from foundry.training.config import canonical_sha256


def _validate(**overrides: object) -> None:
    values: dict[str, object] = {
        "discrepancies": [EXPECTED_DISCREPANCY],
        "installed_version": "6.0.3",
        "metadata_present": True,
        "source_hashes_equal": True,
        "parsed_structures_equal": True,
        "typed_configs_equal": True,
        "replay_equal": True,
    }
    values.update(overrides)
    validate_narrow_exception(**values)  # type: ignore[arg-type]


def test_exact_known_mismatch_passes() -> None:
    _validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("discrepancies", [EXPECTED_DISCREPANCY, "second problem"]),
        ("installed_version", "6.0.4"),
        ("metadata_present", False),
        ("source_hashes_equal", False),
        ("parsed_structures_equal", False),
        ("typed_configs_equal", False),
        ("replay_equal", False),
    ],
)
def test_any_exception_expansion_or_equivalence_drift_fails(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _validate(**{field: value})


def test_no_broad_pip_check_waiver() -> None:
    with pytest.raises(ValueError):
        _validate(discrepancies=["some other package is broken"])


def test_tampered_evidence_fails() -> None:
    evidence = {
        "exception_id": EXCEPTION_ID,
        "exception_decision": "pass",
        "pip_check_discrepancies": [EXPECTED_DISCREPANCY],
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    validate_evidence(evidence)
    evidence["exception_decision"] = "failed"
    with pytest.raises(ValueError):
        validate_evidence(evidence)
