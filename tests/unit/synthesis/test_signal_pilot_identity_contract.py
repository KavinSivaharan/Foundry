"""Schedule/runtime number-neutral identity integration tests."""

from __future__ import annotations

from foundry.synthesis.contamination import canonical_number_neutral_identity
from foundry.synthesis.template_bank import signal_allocator, signal_smoke


def test_schedulers_and_runtime_import_the_same_canonical_identity_function() -> None:
    assert signal_allocator.canonical_number_neutral_identity is (canonical_number_neutral_identity)
    assert signal_smoke.canonical_number_neutral_identity is canonical_number_neutral_identity
