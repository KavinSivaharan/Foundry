from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundry.cycle.contract import STAGES, CycleContractError, load_cycle_config
from foundry.cycle.controller import CycleController, StageOutcome
from foundry.cycle.state import StateStore, _state_hash

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/cycles/cycle1_verifier_filtered.yaml"
SOURCE = {
    "commit": "controller-commit",
    "tree": "controller-tree",
    "status": "clean",
    "import_root": "C:/frozen/src",
}


class FakeBackend:
    def __init__(
        self,
        *,
        failing_stage: str | None = None,
        reject_at_decision: bool = False,
        raising_stage: str | None = None,
    ) -> None:
        self.failing_stage = failing_stage
        self.reject_at_decision = reject_at_decision
        self.raising_stage = raising_stage
        self.calls: list[str] = []
        self.finalized = False

    def execute(self, stage: str, state: dict[str, Any]) -> StageOutcome:
        self.calls.append(stage)
        if stage == self.raising_stage:
            raise RuntimeError("synthetic backend failure")
        if stage == self.failing_stage:
            return StageOutcome(False, {"stage": stage}, f"{stage}_gate_failed")
        if stage == "decide":
            rejected = state["decision"] == "rejected" or self.reject_at_decision
            reason = (
                str(state["stop_reason"])
                if state["decision"] == "rejected"
                else "benchmark_gate_failed"
            )
            return StageOutcome(
                True,
                {"stage": stage, "decision": "rejected" if rejected else "promoted"},
                reason if rejected else None,
                "rejected" if rejected else "promoted",
            )
        return StageOutcome(True, {"stage": stage})

    def finalize(self, state: dict[str, Any]) -> None:
        assert state["decision"] in {"promoted", "rejected"}
        self.finalized = True


def _store(tmp_path: Path) -> StateStore:
    return StateStore(
        path=tmp_path / "state.json",
        config=load_cycle_config(CONFIG),
        source=SOURCE,
        interpreter_sha256="interpreter",
    )


def _run(tmp_path: Path, backend: FakeBackend) -> dict[str, Any]:
    controller = CycleController(
        config=load_cycle_config(CONFIG),
        store=_store(tmp_path),
        backend=backend,
    )
    return controller.run()


def test_complete_success_path_and_successful_promotion(tmp_path: Path) -> None:
    backend = FakeBackend()

    state = _run(tmp_path, backend)

    assert state["decision"] == "promoted"
    assert backend.calls == list(STAGES)
    assert all(record["status"] == "completed" for record in state["stages"].values())
    assert backend.finalized is True


@pytest.mark.parametrize(
    "stage",
    [
        "generate_candidates",
        "verify_and_select_traces",
        "train_candidate",
        "development_retention",
        "holdout_retention",
    ],
)
def test_failed_scientific_gate_rejects_and_skips_later_work(
    tmp_path: Path,
    stage: str,
) -> None:
    backend = FakeBackend(failing_stage=stage)

    state = _run(tmp_path, backend)

    assert state["decision"] == "rejected"
    assert state["stages"][stage]["status"] == "failed"
    assert state["stages"]["decide"]["status"] == "completed"
    assert state["stages"]["promote_or_reject"]["status"] == "completed"
    assert state["stages"]["publish_trace"]["status"] == "completed"
    assert backend.finalized is True


def test_training_exception_rejects_without_escaping_controller(tmp_path: Path) -> None:
    backend = FakeBackend(raising_stage="train_candidate")

    state = _run(tmp_path, backend)

    assert state["decision"] == "rejected"
    assert state["stop_reason"] == "train_candidate_exception:RuntimeError"
    assert state["stages"]["benchmark"]["status"] == "skipped"


def test_benchmark_gate_rejection_preserves_terminal_flow(tmp_path: Path) -> None:
    state = _run(tmp_path, FakeBackend(reject_at_decision=True))

    assert state["decision"] == "rejected"
    assert state["stop_reason"] == "benchmark_gate_failed"
    assert state["stages"]["benchmark"]["status"] == "completed"
    assert state["stages"]["publish_trace"]["status"] == "completed"


def test_resume_starts_at_first_unfinished_stage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    state = store.create_or_load()
    for stage in STAGES[:3]:
        store.begin(state, stage)
        store.complete(state, stage, {"stage": stage})
        state = store.load()
    backend = FakeBackend()

    resumed = CycleController(
        config=load_cycle_config(CONFIG),
        store=store,
        backend=backend,
    ).run()

    assert backend.calls[0] == "generate_candidates"
    assert resumed["decision"] == "promoted"


def test_identity_drift_is_rejected_even_when_state_hash_reconstructs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    state = store.create_or_load()
    state["config_sha256"] = "changed"
    state["state_sha256"] = _state_hash(state)
    store.path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CycleContractError, match="identity drift"):
        store.load()


def test_out_of_order_stage_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    state = store.create_or_load()

    with pytest.raises(CycleContractError, match="out-of-order"):
        store.begin(state, "diagnose")


def test_compatibility_preflight_failure_is_terminal_without_retry(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.reject_at_compatibility_preflight(
        "compatibility_smoke_mismatch",
        {"passed": False, "trial": 2},
    )
    backend = FakeBackend()

    state = CycleController(
        config=load_cycle_config(CONFIG),
        store=store,
        backend=backend,
    ).run()

    assert backend.calls == ["decide", "promote_or_reject", "publish_trace"]
    assert state["decision"] == "rejected"
    assert state["stages"]["compatibility_smoke"]["status"] == "failed"
    assert state["stop_reason"] == "compatibility_smoke_mismatch"
