"""Durable content-free state machine for Foundry Cycle 1."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from foundry.cycle.contract import (
    CONTROLLER_ID,
    CYCLE_ID,
    STAGES,
    CycleConfig,
    CycleContractError,
    content_free_projection,
    cycle_execution_metadata,
)
from foundry.training.config import canonical_sha256

TERMINAL_DECISIONS = {"promoted", "rejected"}
STAGE_STATUSES = {"pending", "running", "completed", "failed", "skipped"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _state_hash(value: dict[str, Any]) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "state_sha256"})


def initial_state(
    config: CycleConfig, source: dict[str, str], interpreter_sha256: str
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": 1,
        "controller_id": CONTROLLER_ID,
        "cycle_id": CYCLE_ID,
        "execution": cycle_execution_metadata(config),
        "config_sha256": config.sha256,
        "source": source,
        "interpreter_sha256": interpreter_sha256,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "current_stage": None,
        "decision": None,
        "stop_reason": None,
        "stages": {
            stage: {
                "index": index + 1,
                "status": "pending",
                "started_at_utc": None,
                "completed_at_utc": None,
                "evidence_sha256": None,
                "classification": None,
            }
            for index, stage in enumerate(STAGES)
        },
    }
    state["state_sha256"] = _state_hash(state)
    return state


@dataclass
class StateStore:
    """Atomic state persistence with strict ordering and identity drift checks."""

    path: Path
    config: CycleConfig
    source: dict[str, str]
    interpreter_sha256: str

    def create_or_load(self) -> dict[str, Any]:
        if self.path.exists():
            return self.load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = initial_state(self.config, self.source, self.interpreter_sha256)
        self.write(state)
        return state

    def load(self) -> dict[str, Any]:
        value: object = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CycleContractError("cycle state must be an object")
        state = cast(dict[str, Any], value)
        supplied = state.get("state_sha256")
        if supplied != _state_hash(state):
            raise CycleContractError("cycle state hash does not reconstruct")
        if (
            state.get("controller_id") != CONTROLLER_ID
            or state.get("cycle_id") != CYCLE_ID
            or state.get("execution") != cycle_execution_metadata(self.config)
            or state.get("config_sha256") != self.config.sha256
            or state.get("source") != self.source
            or state.get("interpreter_sha256") != self.interpreter_sha256
        ):
            raise CycleContractError("cycle state identity drift detected")
        stages = state.get("stages")
        if (
            not isinstance(stages, dict)
            or set(stages) != set(STAGES)
            or any(
                not isinstance(stages.get(name), dict) or stages[name].get("index") != index
                for index, name in enumerate(STAGES, start=1)
            )
        ):
            raise CycleContractError("cycle state stage order differs")
        for name, raw in stages.items():
            if not isinstance(raw, dict) or raw.get("status") not in STAGE_STATUSES:
                raise CycleContractError(f"invalid state for stage {name}")
        return state

    def write(self, state: dict[str, Any]) -> None:
        projected = content_free_projection(state)
        if projected != state:
            raise CycleContractError("prompt-bearing material cannot enter cycle state")
        state["updated_at_utc"] = utc_now()
        state["state_sha256"] = _state_hash(state)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def next_stage(self, state: dict[str, Any]) -> str | None:
        stages = cast(dict[str, dict[str, Any]], state["stages"])
        for name in STAGES:
            status = stages[name]["status"]
            if status in {"pending", "running"}:
                return name
        return None

    def begin(self, state: dict[str, Any], stage: str) -> None:
        expected = self.next_stage(state)
        if expected != stage:
            raise CycleContractError(
                f"out-of-order stage execution: expected {expected}, got {stage}"
            )
        record = cast(dict[str, Any], cast(dict[str, Any], state["stages"])[stage])
        if record["status"] == "pending":
            record["status"] = "running"
            record["started_at_utc"] = utc_now()
        state["current_stage"] = stage
        self.write(state)

    def complete(
        self,
        state: dict[str, Any],
        stage: str,
        evidence: dict[str, Any],
        *,
        classification: str | None = None,
    ) -> None:
        record = cast(dict[str, Any], cast(dict[str, Any], state["stages"])[stage])
        if record["status"] != "running" or state["current_stage"] != stage:
            raise CycleContractError(f"stage {stage} is not running")
        projected = content_free_projection(evidence)
        if projected != evidence:
            raise CycleContractError("stage evidence must be content-free")
        record["status"] = "completed"
        record["completed_at_utc"] = utc_now()
        record["evidence_sha256"] = canonical_sha256(evidence)
        record["classification"] = classification
        state["current_stage"] = None
        self.write(state)

    def fail(
        self,
        state: dict[str, Any],
        stage: str,
        reason: str,
        evidence: dict[str, Any],
    ) -> None:
        record = cast(dict[str, Any], cast(dict[str, Any], state["stages"])[stage])
        if record["status"] != "running":
            raise CycleContractError(f"stage {stage} is not running")
        record["status"] = "failed"
        record["completed_at_utc"] = utc_now()
        record["evidence_sha256"] = canonical_sha256(content_free_projection(evidence))
        record["classification"] = reason
        state["current_stage"] = None
        state["decision"] = "rejected"
        state["stop_reason"] = reason
        self.write(state)

    def skip_until_decision(self, state: dict[str, Any]) -> None:
        stages = cast(dict[str, dict[str, Any]], state["stages"])
        for name in STAGES:
            if name in {"decide", "promote_or_reject", "publish_trace"}:
                continue
            if stages[name]["status"] == "pending":
                stages[name]["status"] = "skipped"
                stages[name]["completed_at_utc"] = utc_now()
                stages[name]["classification"] = "not_reached_after_frozen_gate_failure"
        self.write(state)

    def set_decision(self, state: dict[str, Any], decision: str, reason: str | None) -> None:
        if decision not in TERMINAL_DECISIONS:
            raise CycleContractError("invalid terminal decision")
        state["decision"] = decision
        state["stop_reason"] = reason
        self.write(state)

    def reject_at_compatibility_preflight(
        self,
        reason: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Record a pre-production compatibility failure without a retry."""

        state = self.create_or_load()
        stages = cast(dict[str, dict[str, Any]], state["stages"])
        if any(record["status"] != "pending" for record in stages.values()):
            raise CycleContractError("compatibility rejection requires a pristine cycle state")
        compatibility_index = STAGES.index("compatibility_smoke")
        timestamp = utc_now()
        for name in STAGES[:compatibility_index]:
            stages[name]["status"] = "skipped"
            stages[name]["completed_at_utc"] = timestamp
            stages[name]["classification"] = "not_entered_before_compatibility_preflight"
        record = stages["compatibility_smoke"]
        record["status"] = "failed"
        record["started_at_utc"] = timestamp
        record["completed_at_utc"] = timestamp
        record["evidence_sha256"] = canonical_sha256(content_free_projection(evidence))
        record["classification"] = reason
        for name in STAGES[compatibility_index + 1 :]:
            if name in {"decide", "promote_or_reject", "publish_trace"}:
                continue
            stages[name]["status"] = "skipped"
            stages[name]["completed_at_utc"] = timestamp
            stages[name]["classification"] = "not_reached_after_frozen_gate_failure"
        state["decision"] = "rejected"
        state["stop_reason"] = reason
        state["current_stage"] = None
        self.write(state)
        return self.load()
