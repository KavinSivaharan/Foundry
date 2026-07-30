"""Model-free Milestone 14B-R4 adjudication of the prior GRPO attempt."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256

ADJUDICATION_ID = "foundry-l3-grpo-prior-attempt-adjudication-v1"
PRIOR_FREEZE_ID = "foundry-l3-grpo-prior-compatibility-attempt-freeze-v1"

CASE_1: Literal["invalidated_by_superseded_source_binding"] = (
    "invalidated_by_superseded_source_binding"
)
CASE_2: Literal["scientifically_counted_compatibility_failure"] = (
    "scientifically_counted_compatibility_failure"
)
CASE_3: Literal["prior_attempt_status_ambiguous"] = "prior_attempt_status_ambiguous"
NON_COUNTED_DIAGNOSTIC = "pre_reconciliation_non_counted_compatibility_diagnostic"

START_COMMIT = "8a5df768e5c376b2326db828e43df342de549a5e"
START_TREE = "602a02a7bee9c501ff2145fd3c2df72db5607187"
PRIOR_EXECUTION_COMMIT = "2c0ab0daa5ecc2003b69db091328460f099bd852"
PRIOR_EXECUTION_TREE = "8b8638eb389c157535147026cbd10538a7bc15ad"
PRIOR_SOURCE_FIX_COMMIT = "1bac985cb55ad8c528783e69f15b82ab54449e46"
PRIOR_SOURCE_FIX_TREE = "eaa94dc1119f47997aff0c67a06d8f1f178c4955"

TRACKED_ROOT = Path("results/phase2_vetted_corpus")
LAYER1_PATH = TRACKED_ROOT / "milestone14b_r3_layer1_scientific_qualification_manifest.json"
LAYER2_PATH = TRACKED_ROOT / "milestone14b_r3_layer2_compatibility_runtime_manifest.json"
BLOCKER_PATH = TRACKED_ROOT / "milestone14b_r3_compatibility_blocker.json"
PRIOR_FREEZE_PATH = TRACKED_ROOT / "milestone14b_r4_prior_compatibility_attempt_freeze.json"
ADJUDICATION_PATH = TRACKED_ROOT / "milestone14b_r4_prior_attempt_adjudication.json"

RAW_ROOT = Path("results/raw/phase2_vetted_corpus/milestone14b_r3/compatibility")
PARTIAL_PATH = RAW_ROOT / "generic/run-1/partial_evidence.json"
STDOUT_PATH = RAW_ROOT / "logs/generic-run-1.stdout.txt"
STDERR_PATH = RAW_ROOT / "logs/generic-run-1.stderr.txt"

EXPECTED_LAYER1_CANONICAL_SHA256 = (
    "6076832b5312183504bd2c94c41135930c461e801b56aecc0a349129043f964c"
)
EXPECTED_LAYER2_CANONICAL_SHA256 = (
    "df72fba61cf6e3785ccd40a1bbb2fb0c2a2c08c980f319bf9cc7e131148e4c50"
)
EXPECTED_BLOCKER_CANONICAL_SHA256 = (
    "bd59d40e0cb775bb4c3f0b751e6ec0f7c25a2a91b0a4fa87a74163691d57e797"
)
EXPECTED_PARTIAL_CANONICAL_SHA256 = (
    "c043de7b4cd5222b10cef7266367de982257ccff16734f22c599406cef48144a"
)
EXPECTED_PARTIAL_FILE_SHA256 = "257343185cc1a19b48fcfabf04f320c132af13b378aaa95c151fd7638e18858d"
EXPECTED_STDOUT_FILE_SHA256 = "da24ab8a7a48038e37b5269b2df893fa1151b9d7964d10c609298130e3133dcc"
EXPECTED_STDERR_FILE_SHA256 = "0c031cad7e1a8707b7e1561671af473c00e20d62d1875aee16162bae780018e7"
EXPECTED_FAILURE = "stock TRL advantages differ from the frozen reward projection"

EXPECTED_SCIENTIFIC_IDENTITIES: dict[str, object] = {
    "model": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "manifest_sha256": "02bff45c336c3650abe518a94accf4c321f0116678a99c2f56a131cf2eade34d",
    },
    "dataset_sha256": "ee18f7f9bd7625469d83e1c1df8dad4db0ecf8b3d8c870a07c60f2808e11dc31",
    "starting_adapters": {
        "generic": "67c6f1dd34c0fa1ddebb354dfe14c43e61c48fdd90c687ba1a9290d2401479cd",
        "targeted": "4e195ff2cb32c4faa6858915b95507862c911bb2eb853b060717416d825df91d",
    },
    "schedules": {
        "generic": "ff1005a1d7381acd52dd28b3d054b2979986c47595ed09c944880ea5fc5f5ff3",
        "targeted": "8326c1b91ba127c4734527abfed2f8bca41ecbb3a0bb7bc62a5bf940ac24f0c4",
        "shared_replay": "19e27fecde5349b6a7a9a24d8a0a8211a3b0da877282a51ece6b616688904181",
        "paired": "ed99aa38f77961fa1f669ba110cd86b3af092e027a60b8e92096a9a68bdfc8e3",
    },
    "reward": {
        "implementation_sha256": (
            "448574e61ff74c40b026dd493b9773023c92eb7f4dbaccb5dfbf511be1e68e66"
        ),
        "configuration_sha256": (
            "701ac381bf01337f706dfaf46ebfa91839147bd600e23dc21a3f1d00eb0f5df5"
        ),
        "fixture_sha256": ("ca7ea72ae288234eb769486f1a1dd0893f9c14b1c14487ae043336af30318199"),
        "calibration_sha256": ("e0952f0034424f7817998300207ec3eecf2ce4f8443a87405899decff3fb65e7"),
        "contract_sha256": ("441933982c2b51b49195763440c318893cea22af947c9efc50b732d05fee7b61"),
    },
    "reference_mechanism_sha256": (
        "674b368105f08b0e1eb00f54c6912f611da730ed070f60c63989de996ecb0316"
    ),
    "representatives": {
        "generic": {
            "gradient_projection_sha256": (
                "2a854c5cdb5696a29e94cb26003993da6a9da4a30aaeff029101c99b50ec0e97"
            ),
            "replay_group_id": "l3-grpo-generic-g004",
            "task_group_id": "l3-grpo-generic-g005",
            "task_group_record_sha256": (
                "14cd697162b9ace4809d715901821afd7120bfc9f3fdf86483a8d9d9785cc3f3"
            ),
        },
        "targeted": {
            "gradient_projection_sha256": (
                "de787cbb1687e5c86ab0d29cab0862af0df1d76c195eb7b557a6b4f8c01245dd"
            ),
            "replay_group_id": "l3-grpo-targeted-g004",
            "task_group_id": "l3-grpo-targeted-g001",
            "task_group_record_sha256": (
                "7128ecc43cdd6128e24c948823e2d9e1da249356534f413decd8c73f63c01ce3"
            ),
        },
    },
    "signal_decision": "schedule_viable",
    "signal_summary_sha256": ("fc7bad292f6c4b5acaa845df9b30cbc624de1ee524b57443ea09e634e2352ec4"),
    "selection_decision_sha256": (
        "0e809d1870ddef275017a11c4db5ffd766d9624a34dbf87a7ab417f8bed6a3cf"
    ),
    "qualification_contract_sha256": (
        "0ad3c6fc584b1dcc0221e6c29179f1f53c26c754297b6534222bf750e51a23bc"
    ),
}

CASE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    CASE_1: (
        "active_source_binding_structurally_invalid",
        "prohibited_manifest_self_reference_present",
        "active_import_root_not_detached",
        "execution_source_not_independently_certified",
        "terminal_result_not_scientifically_separable",
        "terminal_result_depended_on_invalid_binding_state",
    ),
    CASE_2: (
        "intended_runtime_imported",
        "scientific_identities_exact",
        "execution_source_independently_certified",
        "terminal_result_scientifically_separable",
        "terminal_result_valid_under_unchanged_contract",
        "manifest_structure_did_not_affect_numeric_result",
    ),
}

AdjudicationCase = Literal[
    "invalidated_by_superseded_source_binding",
    "scientifically_counted_compatibility_failure",
    "prior_attempt_status_ambiguous",
]


def _read(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _verify_canonical_self_hash(value: Mapping[str, Any], key: str, expected: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if supplied != expected or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def _git(root: Path, *args: str, capture_bytes: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=not capture_bytes,
        shell=False,
    )
    if capture_bytes:
        return cast(bytes, completed.stdout)
    return cast(str, completed.stdout).strip()


def _git_quiet(root: Path, *args: str) -> bool:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
    )
    return completed.returncode == 0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _hash_projection(value: object) -> str:
    return canonical_sha256(value)


def audit_historical_manifests(root: Path) -> dict[str, object]:
    """Verify the exact historical manifests and all 60 frozen source rows."""

    root = root.resolve()
    layer1_path = root / LAYER1_PATH
    layer2_path = root / LAYER2_PATH
    layer1 = _read(layer1_path)
    layer2 = _read(layer2_path)
    _verify_canonical_self_hash(
        layer1,
        "layer1_manifest_sha256",
        EXPECTED_LAYER1_CANONICAL_SHA256,
    )
    _verify_canonical_self_hash(
        layer2,
        "layer2_manifest_sha256",
        EXPECTED_LAYER2_CANONICAL_SHA256,
    )
    if layer1.get("scientific_identities") != EXPECTED_SCIENTIFIC_IDENTITIES:
        raise ValueError("historical Layer-1 scientific identities differ")
    evidence_rows = cast(list[dict[str, Any]], layer1.get("evidence_rows"))
    evidence_paths = cast(list[str], layer1.get("evidence_paths"))
    if (
        len(evidence_rows) != 14
        or [cast(str, row.get("path")) for row in evidence_rows] != evidence_paths
        or len(set(evidence_paths)) != 14
    ):
        raise ValueError("historical Layer-1 evidence allowlist differs")
    for row in evidence_rows:
        relative = cast(str, row["path"])
        evidence_path = root / relative
        evidence = _read(evidence_path)
        self_hash_key = cast(str, row["self_hash_key"])
        self_hash = cast(str, row["self_hash"])
        if evidence_path.stat().st_size != row.get("bytes") or file_sha256(
            evidence_path
        ) != row.get("sha256"):
            raise ValueError(f"historical Layer-1 evidence bytes differ: {relative}")
        _verify_canonical_self_hash(evidence, self_hash_key, self_hash)

    qualification_implementation = _read(
        root / "results/phase2_vetted_corpus/milestone14b_r1_qualification_implementation.json"
    )
    historical_rows = cast(list[dict[str, Any]], qualification_implementation.get("files"))
    historical_commit = cast(str, layer1.get("historical_source_commit"))
    if (
        qualification_implementation.get("implementation_sha256")
        != layer1.get("r1_implementation_sha256")
        or len(historical_rows) != layer1.get("historical_source_row_count")
        or canonical_sha256(historical_rows) != layer1.get("historical_source_rows_sha256")
        or canonical_sha256([row["path"] for row in historical_rows])
        != layer1.get("historical_source_paths_sha256")
        or cast(str, _git(root, "rev-parse", f"{historical_commit}^{{tree}}"))
        != layer1.get("historical_source_tree")
    ):
        raise ValueError("historical Layer-1 source aggregate differs")
    for row in historical_rows:
        relative = cast(str, row["path"])
        blob = cast(
            bytes,
            _git(
                root,
                "cat-file",
                "blob",
                f"{historical_commit}:{relative}",
                capture_bytes=True,
            ),
        )
        if len(blob) != row.get("bytes") or _sha256_bytes(blob) != row.get("sha256"):
            raise ValueError(f"historical Layer-1 source blob differs: {relative}")

    if (
        layer2.get("source_commit") != PRIOR_SOURCE_FIX_COMMIT
        or layer2.get("source_tree") != PRIOR_SOURCE_FIX_TREE
        or layer2.get("python_import_root") != str((root / "src").resolve())
    ):
        raise ValueError("historical Layer-2 source identity differs")

    ordered_paths = cast(list[str], layer2.get("ordered_paths"))
    rows = cast(list[dict[str, Any]], layer2.get("files"))
    if (
        len(ordered_paths) != 60
        or len(rows) != 60
        or len(set(ordered_paths)) != 60
        or [row.get("path") for row in rows] != ordered_paths
    ):
        raise ValueError("historical Layer-2 path allowlist differs")

    for row in rows:
        relative = cast(str, row["path"])
        execution_path = root / relative
        if execution_path.stat().st_size != row.get("execution_bytes") or file_sha256(
            execution_path
        ) != row.get("execution_sha256"):
            raise ValueError(f"historical Layer-2 execution bytes differ: {relative}")
        blob = cast(
            bytes,
            _git(
                root,
                "cat-file",
                "blob",
                f"{PRIOR_SOURCE_FIX_COMMIT}:{relative}",
                capture_bytes=True,
            ),
        )
        if len(blob) != row.get("git_blob_bytes") or _sha256_bytes(blob) != row.get(
            "git_blob_sha256"
        ):
            raise ValueError(f"historical Layer-2 Git blob differs: {relative}")
        source_blob = cast(
            str,
            _git(root, "rev-parse", f"{PRIOR_SOURCE_FIX_COMMIT}:{relative}"),
        )
        execution_blob = cast(
            str,
            _git(root, "rev-parse", f"{PRIOR_EXECUTION_COMMIT}:{relative}"),
        )
        starting_blob = cast(str, _git(root, "rev-parse", f"{START_COMMIT}:{relative}"))
        if source_blob != execution_blob or source_blob != starting_blob:
            raise ValueError(f"historical Layer-2 source changed after freeze: {relative}")

    for manifest_relative in (LAYER1_PATH, LAYER2_PATH):
        if not _git_quiet(
            root,
            "diff",
            "--quiet",
            START_COMMIT,
            "--",
            str(manifest_relative),
        ):
            raise ValueError(f"historical manifest working bytes changed: {manifest_relative}")
        tracked = cast(str, _git(root, "ls-files", "--", str(manifest_relative)))
        normalized = str(manifest_relative).replace("\\", "/")
        if tracked != normalized:
            raise ValueError(f"historical manifest is not tracked: {manifest_relative}")

    command_templates = cast(Mapping[str, Any], layer2.get("command_templates"))
    compatibility = cast(Mapping[str, Any], command_templates.get("compatibility"))
    return {
        "layer1": {
            "path": str(LAYER1_PATH).replace("\\", "/"),
            "external_file_sha256": file_sha256(layer1_path),
            "git_blob": cast(
                str,
                _git(
                    root,
                    "rev-parse",
                    f"{START_COMMIT}:{str(LAYER1_PATH).replace(chr(92), '/')}",
                ),
            ),
            "recorded_canonical_self_hash": layer1["layer1_manifest_sha256"],
            "self_hash_field_present": True,
            "tracked": True,
            "evidence_row_count": len(evidence_rows),
            "historical_source_commit": historical_commit,
            "historical_source_tree": layer1["historical_source_tree"],
            "historical_source_row_count": len(historical_rows),
            "historical_source_rows_sha256": layer1["historical_source_rows_sha256"],
        },
        "layer2": {
            "path": str(LAYER2_PATH).replace("\\", "/"),
            "external_file_sha256": file_sha256(layer2_path),
            "git_blob": cast(
                str,
                _git(
                    root,
                    "rev-parse",
                    f"{START_COMMIT}:{str(LAYER2_PATH).replace(chr(92), '/')}",
                ),
            ),
            "recorded_canonical_self_hash": layer2["layer2_manifest_sha256"],
            "self_hash_field_present": True,
            "command_template_fields_present": True,
            "argv_hash_fields_present": True,
            "tracked": True,
            "source_fix_commit": layer2["source_commit"],
            "source_fix_tree": layer2["source_tree"],
            "ordered_path_count": len(ordered_paths),
            "combined_execution_source_sha256": layer2["combined_execution_source_sha256"],
            "combined_git_blob_source_sha256": layer2["combined_git_blob_source_sha256"],
            "combined_source_sha256": layer2["combined_source_sha256"],
            "active_import_root": layer2["python_import_root"],
            "detached_import_root": False,
            "compatibility_command_template_sha256": compatibility["template_sha256"],
            "compatibility_template_argv_projection_sha256": compatibility[
                "argv_projection_sha256"
            ],
            "interpreter_sha256": layer2["interpreter_sha256"],
            "package_inventory_sha256": layer2["package_inventory_sha256"],
            "combined_child_environment_sha256": layer2["combined_child_environment_sha256"],
        },
        "violations": {
            "layer1_self_hash_field": True,
            "layer2_self_hash_field": True,
            "layer2_command_template_fields": True,
            "layer2_argv_hash_fields": True,
            "tracked_active_runtime_manifests": True,
            "primary_repository_import_root": True,
            "revised_non_circular_contract_passed": False,
        },
        "source_continuity": {
            "all_60_paths_match_source_fix_execution_and_start_commits": True,
            "source_fix_commit": PRIOR_SOURCE_FIX_COMMIT,
            "source_fix_tree": PRIOR_SOURCE_FIX_TREE,
            "execution_commit": PRIOR_EXECUTION_COMMIT,
            "execution_tree": PRIOR_EXECUTION_TREE,
            "starting_commit": START_COMMIT,
            "starting_tree": START_TREE,
        },
    }


def _advantage_evidence(pending: Mapping[str, Any]) -> dict[str, object]:
    generation = cast(Mapping[str, Any], pending["generation"])
    projection = cast(Mapping[str, Any], generation["reward_projection"])
    stock = [
        float(cast(float, value))
        for value in cast(Sequence[object], generation["stock_advantages"])
    ]
    frozen = [
        float(cast(float, value)) for value in cast(Sequence[object], projection["advantages"])
    ]
    if len(stock) != 4 or len(frozen) != 4:
        raise ValueError("positive-LR advantage vector length differs")
    deltas = [left - right for left, right in zip(stock, frozen, strict=True)]
    mismatch_count = sum(left != right for left, right in zip(stock, frozen, strict=True))
    if mismatch_count != 3 or max(abs(value) for value in deltas) != 5.960464477539063e-08:
        raise ValueError("positive-LR advantage mismatch differs")
    return {
        "component_count": len(stock),
        "exact_equal_component_count": len(stock) - mismatch_count,
        "mismatched_component_count": mismatch_count,
        "maximum_absolute_component_delta": max(abs(value) for value in deltas),
        "stock_advantage_vector_sha256": _hash_projection(stock),
        "frozen_advantage_vector_sha256": _hash_projection(frozen),
        "advantage_delta_vector_sha256": _hash_projection(deltas),
        "unprojected_reward_vector_sha256": _hash_projection(
            generation["reward_vector_unprojected"]
        ),
        "reward_projection_sha256": projection["reward_projection_sha256"],
        "reward_mean": projection["reward_mean"],
        "reward_variance": projection["reward_variance"],
    }


def build_prior_attempt_freeze(root: Path) -> dict[str, object]:
    """Reconstruct the interrupted attempt without exposing prompt or completion text."""

    root = root.resolve()
    manifests = audit_historical_manifests(root)
    blocker_path = root / BLOCKER_PATH
    partial_path = root / PARTIAL_PATH
    stdout_path = root / STDOUT_PATH
    stderr_path = root / STDERR_PATH
    blocker = _read(blocker_path)
    partial = _read(partial_path)
    _verify_canonical_self_hash(
        blocker,
        "compatibility_blocker_sha256",
        EXPECTED_BLOCKER_CANONICAL_SHA256,
    )
    _verify_canonical_self_hash(
        partial,
        "partial_evidence_sha256",
        EXPECTED_PARTIAL_CANONICAL_SHA256,
    )
    if (
        file_sha256(partial_path) != EXPECTED_PARTIAL_FILE_SHA256
        or file_sha256(stdout_path) != EXPECTED_STDOUT_FILE_SHA256
        or file_sha256(stderr_path) != EXPECTED_STDERR_FILE_SHA256
    ):
        raise ValueError("prior raw evidence file hash differs")
    if (
        blocker.get("source_commit") != PRIOR_EXECUTION_COMMIT
        or blocker.get("source_tree") != PRIOR_EXECUTION_TREE
        or partial.get("error") != EXPECTED_FAILURE
        or partial.get("stage") != "validation_failure"
        or blocker.get("compatibility_result") != "failed"
        or blocker.get("decision") != "stop"
    ):
        raise ValueError("prior terminal attempt identity differs")

    completed = cast(list[dict[str, Any]], partial["completed_classification_steps"])
    pending = cast(Mapping[str, Any], partial["pending_step"])
    if len(completed) != 1:
        raise ValueError("prior completed-step count differs")
    first = completed[0]
    failure = cast(Mapping[str, Any], blocker["failure"])
    model = cast(Mapping[str, Any], blocker["model_execution_accounting"])
    downstream = cast(Mapping[str, Any], blocker["downstream_accounting"])
    binding = cast(Mapping[str, Any], blocker["layered_source_binding"])
    publication = cast(Mapping[str, Any], blocker["publication_evidence"])
    resources = cast(Mapping[str, Any], blocker["resource_evidence"])

    if (
        first.get("group_id") != "l3-grpo-generic-g004"
        or first.get("classification") != "expected_zero_advantage_noop"
        or first.get("maximum_effective_learning_rate") != 0.0
        or first.get("policy_parameter_changed") is not False
        or first.get("reference_parameter_changed") is not False
        or first.get("base_parameter_changed") is not False
        or pending.get("group_id") != "l3-grpo-generic-g005"
        or pending.get("expected_effective_learning_rate") != 1.0e-6
        or failure.get("error_message") != EXPECTED_FAILURE
        or failure.get("source_binding_gate_passed") is not True
        or failure.get("no_retry_applied") is not True
        or model.get("model_loads") != 1
        or model.get("generated_groups") != 2
        or model.get("generated_completions") != 8
        or model.get("generated_completion_tokens") != 355
        or downstream.get("sealed_content_use") != 0
    ):
        raise ValueError("prior model-side accounting differs")

    stderr = stderr_path.read_text(encoding="utf-8")
    expected_runtime_path = str((root / "src/foundry/phase2/l3_grpo_runtime.py").resolve())
    expected_wrapper_path = str(
        (root / "src/foundry/phase2/l3_grpo_warmup_compatibility_runtime.py").resolve()
    )
    if (
        expected_runtime_path not in stderr
        or expected_wrapper_path not in stderr
        or f"RuntimeError: {EXPECTED_FAILURE}" not in stderr
    ):
        raise ValueError("prior traceback source identity differs")

    result: dict[str, object] = {
        "schema_version": 1,
        "freeze_id": PRIOR_FREEZE_ID,
        "basis": {
            "starting_commit": START_COMMIT,
            "starting_tree": START_TREE,
            "source_commit": blocker["source_commit"],
            "source_tree": blocker["source_tree"],
        },
        "historical_manifests": manifests,
        "process_identity": {
            "arm": blocker["arm"],
            "run_index": blocker["run_index"],
            "process_role": "generic_compatibility_smoke_a",
            "active_import_root": binding["python_import_root"],
            "wrapper_id": "foundry-l3-grpo-source-bound-warmup-compatibility-wrapper-v2",
            "wrapper_file_sha256": file_sha256(
                root / "src/foundry/phase2/l3_grpo_warmup_compatibility_runtime.py"
            ),
            "runtime_file_sha256": file_sha256(root / "src/foundry/phase2/l3_grpo_runtime.py"),
            "source_binding_file_sha256": file_sha256(
                root / "src/foundry/phase2/l3_grpo_source_binding.py"
            ),
            "argv_projection_sha256": binding["argv_projection_sha256"],
            "command_template_sha256": binding["command_template_sha256"],
            "clean_child_preflight_binding_evidence_sha256": binding[
                "clean_child_preflight_binding_evidence_sha256"
            ],
            "interpreter_sha256": cast(Mapping[str, Any], manifests["layer2"])[
                "interpreter_sha256"
            ],
            "package_inventory_sha256": cast(Mapping[str, Any], manifests["layer2"])[
                "package_inventory_sha256"
            ],
            "combined_child_environment_sha256": cast(Mapping[str, Any], manifests["layer2"])[
                "combined_child_environment_sha256"
            ],
            "source_binding_gate_passed": True,
        },
        "model_execution": {
            "model_loads": model["model_loads"],
            "generated_groups": model["generated_groups"],
            "generated_completions": model["generated_completions"],
            "generated_completion_tokens": model["generated_completion_tokens"],
            "optimizer_calls_completed": model["optimizer_calls_completed"],
            "scheduler_advances": model["scheduler_advances"],
            "trainer_global_steps": model["trainer_global_steps"],
            "policy_updates": model["policy_updates"],
            "reference_updates": model["reference_updates"],
            "adapter_or_checkpoint_writes": model["adapter_or_checkpoint_writes"],
        },
        "step_evidence": {
            "optimizer_call_1": {
                "group_id": first["group_id"],
                "source_kind": first["source_kind"],
                "completion_count": first["completion_count"],
                "valid_completion_tokens": sum(
                    cast(Sequence[int], first["valid_completion_token_counts"])
                ),
                "generated_token_ids_sha256": first["generated_token_ids_sha256"],
                "completion_sha256s_sha256": _hash_projection(first["completion_sha256s"]),
                "reward_vector_sha256": _hash_projection(first["reward_vector"]),
                "reward_variance": first["reward_variance"],
                "advantage_vector_sha256": _hash_projection(first["advantages"]),
                "nonzero_advantage_count": first["nonzero_advantage_count"],
                "effective_learning_rates": first["effective_learning_rates"],
                "classification": first["classification"],
                "policy_gradient_finite": first["policy_gradient_finite"],
                "policy_gradient_norm": first["policy_gradient_norm"],
                "changed_policy_tensor_count": first["changed_policy_tensor_count"],
                "policy_delta_norm": first["policy_delta_norm"],
                "changed_optimizer_state_tensor_count": first[
                    "changed_optimizer_state_tensor_count"
                ],
                "optimizer_call_completed": first["optimizer_call_completed"],
                "scheduler_step_completed": first["scheduler_step_completed"],
                "reference_parameter_changed": first["reference_parameter_changed"],
                "base_parameter_changed": first["base_parameter_changed"],
            },
            "optimizer_call_2": {
                "group_id": pending["group_id"],
                "source_kind": pending["source_kind"],
                "completion_count": pending["generation_record_count"],
                "valid_completion_tokens": sum(
                    cast(
                        Sequence[int],
                        cast(Mapping[str, Any], pending["generation"])[
                            "valid_completion_token_counts"
                        ],
                    )
                ),
                "generated_token_ids_sha256": _hash_projection(
                    cast(Mapping[str, Any], pending["generation"])["generated_token_ids"]
                ),
                "completion_sha256s_sha256": _hash_projection(
                    cast(Mapping[str, Any], pending["generation"])["completion_sha256s"]
                ),
                "effective_learning_rates": pending["effective_learning_rates"],
                "advantage_validation": _advantage_evidence(pending),
                "generation_completed": True,
                "backward_started": False,
                "policy_gradient_status": "not_reached",
                "policy_delta_status": "not_reached",
                "optimizer_call_completed": False,
                "scheduler_step_completed": False,
                "classification": "validation_failure_before_update_classification",
            },
        },
        "terminal": {
            "error_type": failure["error_type"],
            "error_message": failure["error_message"],
            "failure_classification": failure["failure_classification"],
            "failure_stage": failure["failure_stage"],
            "backend_failure_observed": failure["backend_failure_observed"],
            "oom_observed": failure["oom_observed"],
            "no_retry_applied": failure["no_retry_applied"],
            "compatibility_result": blocker["compatibility_result"],
            "decision": blocker["decision"],
            "blocker_canonical_sha256": blocker["compatibility_blocker_sha256"],
            "blocker_external_file_sha256": file_sha256(blocker_path),
        },
        "missing_interrupted_result_fields": [
            "adapter_directory_hash",
            "compatibility_envelope",
            "complete_smoke_gate",
            "cpu_offload_final_status",
            "final_policy_tensor_hash",
            "offline_reload_result",
            "optimizer_call_2_gradient_evidence",
            "optimizer_call_2_loss",
            "optimizer_call_2_policy_delta",
            "optimizer_call_2_policy_kl",
            "peak_allocated_vram_bytes",
            "peak_process_rss_bytes",
            "peak_reserved_vram_bytes",
            "qualification_envelope",
            "raw_summary",
            "runtime_seconds",
            "trainer_state",
        ],
        "publication_evidence": {
            "partial_evidence_canonical_sha256": partial["partial_evidence_sha256"],
            "partial_evidence_external_file_sha256": file_sha256(partial_path),
            "stdout_external_file_sha256": file_sha256(stdout_path),
            "stderr_external_file_sha256": file_sha256(stderr_path),
            "compatibility_tree_files": publication["compatibility_tree_files"],
            "compatibility_tree_disk_bytes": publication["compatibility_tree_disk_bytes"],
            "campaign_observed_wall_seconds": resources["campaign_observed_wall_seconds"],
            "summary_written": publication["summary_written"],
            "final_adapter_written": publication["final_adapter_written"],
            "trainer_state_files_written": publication["trainer_state_files_written"],
        },
        "scientific_settings_changed": False,
        "raw_prompt_or_completion_content_in_record": False,
        "sealed_content_use": 0,
    }
    result["prior_attempt_freeze_sha256"] = canonical_sha256(result)
    return result


def _function_source_evidence(root: Path) -> dict[str, object]:
    runtime_path = root / "src/foundry/phase2/l3_grpo_runtime.py"
    wrapper_path = root / "src/foundry/phase2/l3_grpo_warmup_compatibility_runtime.py"
    runtime_source = runtime_path.read_text(encoding="utf-8")
    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime_source)
    wrapper_tree = ast.parse(wrapper_source)

    finish: ast.FunctionDef | None = None
    for node in runtime_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SmokeRecorder":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "finish_generation":
                    finish = child
                    break
    wrapper_run = next(
        (
            node
            for node in wrapper_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        ),
        None,
    )
    if finish is None or wrapper_run is None:
        raise ValueError("prior runtime adjudication functions are missing")

    finish_names = {node.id for node in ast.walk(finish) if isinstance(node, ast.Name)}
    prohibited_manifest_names = {
        "binding",
        "layer1",
        "layer2",
        "manifest",
        "source_binding",
    }
    if finish_names.intersection(prohibited_manifest_names):
        raise ValueError("advantage validation unexpectedly consumes manifest state")

    calls: list[tuple[str, int]] = []
    for walked in ast.walk(wrapper_run):
        if not isinstance(walked, ast.Call):
            continue
        if isinstance(walked.func, ast.Name):
            calls.append((walked.func.id, walked.lineno))
        elif isinstance(walked.func, ast.Attribute) and isinstance(walked.func.value, ast.Name):
            calls.append(
                (
                    f"{walked.func.value.id}.{walked.func.attr}",
                    walked.lineno,
                )
            )
    binding_lines = [line for name, line in calls if name == "verify_layered_source_binding"]
    model_lines = [line for name, line in calls if name == "official.run"]
    if len(binding_lines) != 1 or len(model_lines) != 1 or binding_lines[0] >= model_lines[0]:
        raise ValueError("source binding did not precede official model execution")

    finish_segment = ast.get_source_segment(runtime_source, finish)
    if (
        finish_segment is None
        or 'advantages != projected["advantages"]' not in finish_segment
        or EXPECTED_FAILURE not in finish_segment
    ):
        raise ValueError("exact advantage comparison source differs")
    return {
        "runtime_file_sha256": file_sha256(runtime_path),
        "wrapper_file_sha256": file_sha256(wrapper_path),
        "binding_call_precedes_official_run": True,
        "binding_call_line": binding_lines[0],
        "official_run_call_line": model_lines[0],
        "advantage_validation_function": "SmokeRecorder.finish_generation",
        "advantage_validation_ast_sha256": _sha256_bytes(
            ast.dump(finish, include_attributes=False).encode("utf-8")
        ),
        "advantage_validation_source_sha256": hashlib.sha256(
            finish_segment.encode("utf-8")
        ).hexdigest(),
        "advantage_validation_manifest_name_intersection": [],
        "advantage_validation_uses_manifest_state": False,
        "exact_advantage_equality_gate_present": True,
    }


def classify_prior_attempt(evidence: Mapping[str, bool]) -> AdjudicationCase:
    """Select exactly one R4 case from explicit scientific evidence."""

    required_names = set(CASE_REQUIREMENTS[CASE_1]).union(CASE_REQUIREMENTS[CASE_2])
    if set(evidence) != required_names:
        raise ValueError("adjudication evidence fields differ")
    if (
        evidence["terminal_result_scientifically_separable"]
        == evidence["terminal_result_not_scientifically_separable"]
        or evidence["execution_source_independently_certified"]
        == evidence["execution_source_not_independently_certified"]
        or evidence["terminal_result_depended_on_invalid_binding_state"]
        == evidence["manifest_structure_did_not_affect_numeric_result"]
    ):
        raise ValueError("adjudication evidence is internally inconsistent")

    case_1 = all(evidence[name] for name in CASE_REQUIREMENTS[CASE_1])
    case_2 = all(evidence[name] for name in CASE_REQUIREMENTS[CASE_2])
    if case_1 and case_2:
        raise ValueError("prior attempt cannot satisfy two adjudication cases")
    if case_1:
        return CASE_1
    if case_2:
        return CASE_2
    return CASE_3


def build_prior_attempt_adjudication(
    root: Path,
    freeze: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Adjudicate whether the prior model-side result remains scientifically counted."""

    root = root.resolve()
    frozen = dict(build_prior_attempt_freeze(root) if freeze is None else freeze)
    supplied_freeze_hash = frozen.pop("prior_attempt_freeze_sha256", None)
    if supplied_freeze_hash != canonical_sha256(frozen):
        raise ValueError("prior-attempt freeze does not reconstruct")
    frozen["prior_attempt_freeze_sha256"] = supplied_freeze_hash

    manifests = cast(Mapping[str, Any], frozen["historical_manifests"])
    continuity = cast(Mapping[str, Any], manifests["source_continuity"])
    terminal = cast(Mapping[str, Any], frozen["terminal"])
    process = cast(Mapping[str, Any], frozen["process_identity"])
    source = _function_source_evidence(root)

    evidence: dict[str, bool] = {
        "active_source_binding_structurally_invalid": True,
        "prohibited_manifest_self_reference_present": True,
        "active_import_root_not_detached": True,
        "execution_source_not_independently_certified": False,
        "terminal_result_not_scientifically_separable": False,
        "terminal_result_depended_on_invalid_binding_state": False,
        "intended_runtime_imported": (
            process["runtime_file_sha256"] == source["runtime_file_sha256"]
        ),
        "scientific_identities_exact": (
            cast(Mapping[str, Any], manifests["layer1"])["recorded_canonical_self_hash"]
            == EXPECTED_LAYER1_CANONICAL_SHA256
        ),
        "execution_source_independently_certified": (
            continuity["all_60_paths_match_source_fix_execution_and_start_commits"] is True
            and process["source_binding_gate_passed"] is True
        ),
        "terminal_result_scientifically_separable": (
            source["advantage_validation_uses_manifest_state"] is False
            and source["binding_call_precedes_official_run"] is True
        ),
        "terminal_result_valid_under_unchanged_contract": (
            terminal["error_message"] == EXPECTED_FAILURE
            and terminal["failure_classification"]
            == "stock_trl_advantage_projection_exact_mismatch"
        ),
        "manifest_structure_did_not_affect_numeric_result": (
            source["advantage_validation_uses_manifest_state"] is False
        ),
    }
    classification = classify_prior_attempt(evidence)
    if classification != CASE_2:
        raise RuntimeError("published evidence no longer uniquely supports R4 Case 2")

    evidence_record: dict[str, object] = {
        "case_inputs": evidence,
        "historical_manifest_structural_invalidity_proven": True,
        "primary_import_root_was_not_detached": True,
        "all_60_runtime_paths_unchanged": True,
        "source_binding_gate_passed_before_model_execution": True,
        "source_analysis": source,
        "scientific_identity_projection_sha256": canonical_sha256(EXPECTED_SCIENTIFIC_IDENTITIES),
        "terminal_advantage_validation_sha256": canonical_sha256(
            cast(Mapping[str, Any], frozen["step_evidence"])["optimizer_call_2"]
        ),
    }
    decision: dict[str, object] = {
        "classification": classification,
        "scientifically_counted": True,
        "non_counted_diagnostic_classification": False,
        "new_compatibility_campaign_authorized": False,
        "source_binding_correction_authorized": False,
        "l3_verifier_grpo_compatibility_line": "closed",
        "reason": (
            "The prior child imported and hash-certified the intended unchanged runtime; "
            "the exact stock-TRL versus frozen-projection comparison consumed no manifest "
            "state, so its one-ULP model-side failure remains scientifically valid."
        ),
        "no_retry_applied_preserved": True,
        "next_action": "project-level GRPO interpretation",
    }
    implementation_path = Path(__file__).resolve()
    result: dict[str, object] = {
        "schema_version": 1,
        "adjudication_id": ADJUDICATION_ID,
        "basis": {
            "starting_commit": START_COMMIT,
            "starting_tree": START_TREE,
            "prior_attempt_freeze_sha256": frozen["prior_attempt_freeze_sha256"],
            "prior_terminal_blocker_sha256": terminal["blocker_canonical_sha256"],
        },
        "contract": {
            "case_1": CASE_1,
            "case_2": CASE_2,
            "case_3": CASE_3,
            "non_counted_diagnostic_label": NON_COUNTED_DIAGNOSTIC,
            "case_requirements_sha256": canonical_sha256(CASE_REQUIREMENTS),
            "exactly_one_case_required": True,
        },
        "implementation": {
            "path": str(implementation_path.relative_to(root)).replace("\\", "/"),
            "normalized_source_sha256": _normalized_text_sha256(implementation_path),
        },
        "evidence": evidence_record,
        "evidence_sha256": canonical_sha256(evidence_record),
        "decision": decision,
        "decision_sha256": canonical_sha256(decision),
        "counted_training_authorized": False,
        "development_retention_authorized": False,
        "holdout_v2_authorized": False,
        "gsm1k_authorized": False,
        "sealed_content_use": 0,
    }
    result["adjudication_sha256"] = canonical_sha256(result)
    return result


def verify_published_records(root: Path) -> dict[str, str]:
    """Rebuild both tracked records exactly."""

    root = root.resolve()
    published_freeze = _read(root / PRIOR_FREEZE_PATH)
    published_adjudication = _read(root / ADJUDICATION_PATH)
    expected_freeze = build_prior_attempt_freeze(root)
    if published_freeze != expected_freeze:
        raise ValueError("published prior-attempt freeze differs")
    expected_adjudication = build_prior_attempt_adjudication(root, expected_freeze)
    if published_adjudication != expected_adjudication:
        raise ValueError("published prior-attempt adjudication differs")
    return {
        "prior_attempt_freeze_sha256": cast(str, published_freeze["prior_attempt_freeze_sha256"]),
        "adjudication_sha256": cast(str, published_adjudication["adjudication_sha256"]),
        "decision_sha256": cast(str, published_adjudication["decision_sha256"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print deterministic records for review before apply-patch publication."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--record", choices=("freeze", "adjudication"), required=True)
    args = parser.parse_args(argv)
    freeze = build_prior_attempt_freeze(args.root)
    value = (
        freeze if args.record == "freeze" else build_prior_attempt_adjudication(args.root, freeze)
    )
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
