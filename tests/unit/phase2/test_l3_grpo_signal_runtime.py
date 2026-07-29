from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from foundry.phase2 import l3_grpo_signal_runtime as runtime
from foundry.phase2.l3_grpo_signal_audit import REWARD_COMPONENT_FIELDS
from foundry.training.config import canonical_sha256


def test_component_vectors_preserve_all_frozen_reward_components() -> None:
    records = [
        SimpleNamespace(
            reward=SimpleNamespace(**{name: float(index) for name in REWARD_COMPONENT_FIELDS})
        )
        for index in range(4)
    ]
    vectors = runtime._component_vectors(records)
    assert tuple(vectors) == REWARD_COMPONENT_FIELDS
    assert all(values == [0.0, 1.0, 2.0, 3.0] for values in vectors.values())


def test_partial_evidence_reconstructs_without_optimization() -> None:
    schedule = SimpleNamespace(packet_sha256="a" * 64, manifest_sha256="b" * 64)
    partial = runtime._partial(
        arm="generic",
        schedule=schedule,
        contract_sha256="c" * 64,
        source_commit="d" * 40,
        groups=[{"group_record_sha256": "e" * 64}],
    )
    supplied = partial["partial_audit_sha256"]
    payload = dict(partial)
    payload.pop("partial_audit_sha256")
    assert supplied == canonical_sha256(payload)
    assert partial["optimizer_created"] is False
    assert partial["backward_calls"] == 0
    assert partial["scheduler_created"] is False
    assert partial["adapter_saved"] is False


def test_runtime_has_no_forbidden_optimization_or_save_call() -> None:
    source_path = Path(runtime.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {"backward", "train", "save_pretrained", "create_optimizer"}
    invoked: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            invoked.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            invoked.add(node.func.id)
    assert forbidden.isdisjoint(invoked)
    source = source_path.read_text(encoding="utf-8")
    assert "trainer._generate_and_score_completions" in source
    assert "with torch.no_grad()" in source
    assert "_write_json_replace(" in source


def test_parser_requires_every_frozen_runtime_path() -> None:
    parsed = runtime._parser().parse_args(
        [
            "--root",
            "C:/Foundry",
            "--arm",
            "targeted",
            "--packet",
            "packet.json",
            "--manifest",
            "manifest.json",
            "--audit-contract",
            "contract.json",
            "--starting-adapter",
            "adapter",
            "--raw-evidence",
            "raw.json",
            "--summary",
            "summary.json",
        ]
    )
    assert parsed.arm == "targeted"
    assert parsed.packet == Path("packet.json")
