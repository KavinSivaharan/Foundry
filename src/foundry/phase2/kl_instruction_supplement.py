"""Build and freeze the Milestone 13C-R2 instruction supplement and v2 holdout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

from foundry.training.config import canonical_sha256
from foundry.training.qlora import file_sha256
from foundry.training.retention import RetentionItem, score_response

CONTRACT_ID = "foundry-kl-instruction-supplement-contract-v1"
SUPPLEMENT_ID = "foundry-kl-instruction-supplement-v1"
COMBINED_ID = "foundry-kl-independent-retention-v2"
COMBINED_SUBSET_ID = "foundry-kl-independent-retention-base-correct-v2"
SEED = 20260720
FAMILY_IDS = (
    "exact_extraction_and_return",
    "deterministic_text_transformation",
    "fixed_label_classification",
    "deterministic_ordering_and_sorting",
    "conditional_response_selection",
    "multi_constraint_response_compliance",
)
FAMILY_QUOTA = 30
SUPPLEMENT_COUNT = len(FAMILY_IDS) * FAMILY_QUOTA
GENERATION = {"do_sample": False, "max_new_tokens": 96, "seed": SEED}
SYSTEM_PROMPT = (
    "Follow the user's instruction exactly. For arithmetic, show only a brief calculation "
    "and end with `Final answer: <number>`. For formatting and instruction tasks, return "
    "only the requested output with no explanation or code fence."
)
COMBINED_MINIMUMS = {"arithmetic": 60, "format": 60, "instruction": 60}
COMBINED_TOTAL_MINIMUM = 220
SUPPLEMENT_MINIMUM = 24
EXPECTED_EXISTING_SUITE_SHA256 = "826ccfda6714af45f2f8e0ae3926d4607a149446ae5b2f75137704e906a2d92e"
EXPECTED_EXISTING_BASE_SUMMARY_SHA256 = (
    "a62b5cce27b43f2e2a33555a976545580ee318e0efa09fcc2dd4c74d48e36c4a"
)
EXPECTED_EXISTING_RAW_SHA256 = "4945064dfd6ea9e24d00ba80c7c87da13a31ec6e87b53a35488fcdd321fe6fa5"
EXPECTED_EXISTING_SUBSET_SHA256 = "4ea32e5cbde0addfbf448291f92d40f1bb40e33dbe4706f7770a3e4c7a0047b7"
SEALED_BOUNDARY_STATUS = "metadata_accessed_example_content_unseen"


def _read_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, Any], value)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain row objects")
    return cast(list[dict[str, Any]], value)


def _validate_hash(value: dict[str, Any], key: str) -> None:
    supplied = value.get(key)
    payload = {name: item for name, item in value.items() if name != key}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise ValueError(f"{key} does not reconstruct")


def load_contract(path: Path) -> dict[str, Any]:
    """Load the supplement contract frozen before prompt construction."""

    contract = _read_object(path)
    _validate_hash(contract, "supplement_contract_sha256")
    families = contract.get("family_order")
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_id") != CONTRACT_ID
        or contract.get("supplement_id") != SUPPLEMENT_ID
        or contract.get("construction_seed") != SEED
        or contract.get("generation_contract") != GENERATION
        or not isinstance(families, list)
        or [row.get("family_id") for row in families if isinstance(row, dict)] != list(FAMILY_IDS)
        or [row.get("order") for row in families if isinstance(row, dict)] != list(range(1, 7))
        or [row.get("quota") for row in families if isinstance(row, dict)] != [FAMILY_QUOTA] * 6
    ):
        raise ValueError("supplement contract identity differs")
    return contract


def _word(index: int) -> str:
    words = (
        "acorn",
        "beacon",
        "cirrus",
        "dahlia",
        "ember",
        "finch",
        "grove",
        "harbor",
        "islet",
        "juniper",
        "kestrel",
        "lagoon",
        "meadow",
        "nectar",
        "orchid",
        "pebble",
        "quill",
        "ripple",
        "spruce",
        "thistle",
        "upland",
        "velvet",
        "willow",
        "xenon",
        "yarrow",
        "zephyr",
        "anvil",
        "bramble",
        "cinder",
        "drift",
    )
    return words[index]


def construct_items(contract: dict[str, Any]) -> list[dict[str, str]]:
    """Construct exactly one immutable six-family supplement."""

    if contract["supplement_id"] != SUPPLEMENT_ID:
        raise ValueError("supplement contract was not validated")
    items: list[dict[str, str]] = []
    for family_order, family_id in enumerate(FAMILY_IDS, start=1):
        for index in range(FAMILY_QUOTA):
            ordinal = index + 1
            word = _word(index)
            marker = 410 + 7 * index + family_order
            item_id = f"klis-f{family_order:02d}-{ordinal:03d}"
            if family_order == 1:
                expected = f"{word.upper()}-{marker}-R"
                prompt = (
                    f"Archive slip {marker} shows LEFT={word}; RETURN={expected}; "
                    f"RIGHT={marker + 9}. Copy only the value immediately following RETURN=. "
                    "Preserve every letter, digit, and hyphen; add no punctuation."
                )
            elif family_order == 2:
                source = f"{word}-node-{marker}"
                expected = source.upper()
                prompt = (
                    f"Apply this mechanical edit to source `{source}`: convert each ASCII letter "
                    "to uppercase while leaving both hyphens and all digits in place. Emit only "
                    "the edited source."
                )
            elif family_order == 3:
                code = f"{word[:3].upper()}{marker}"
                expected = "LABEL-EVEN" if marker % 2 == 0 else "LABEL-ODD"
                prompt = (
                    f"Use this fixed rule for record {code}: choose `LABEL-EVEN` when the final "
                    f"integer {marker} is even, otherwise choose `LABEL-ODD`. Respond with only "
                    "the chosen label."
                )
            elif family_order == 4:
                values = [marker + 8, marker - 3, marker + 2, marker - 7]
                expected = "|".join(str(value) for value in sorted(values))
                prompt = (
                    f"Sort the four identifiers {values[0]}, {values[1]}, {values[2]}, and "
                    f"{values[3]} from smallest to largest. Return only the ordered integers "
                    "joined with vertical bars and no spaces."
                )
            elif family_order == 5:
                divisible = f"TRI-{word.upper()}-{marker}"
                other = f"REST-{word.upper()}-{marker}"
                expected = divisible if marker % 3 == 0 else other
                prompt = (
                    f"Selector {marker} controls two literal replies. If it is divisible by "
                    f"three, return `{divisible}`; otherwise return `{other}`. Output only the "
                    "selected literal reply."
                )
            else:
                expected = f"[RID{marker}~{word.upper()}~{ordinal:02d}]"
                prompt = (
                    f"Create one compliance tag for route {marker}. It must begin with `[RID"
                    f"{marker}~`, continue with uppercase `{word.upper()}`, then `~{ordinal:02d}`, "
                    "and end with `]`. The complete response must contain exactly that tag and "
                    "nothing else."
                )
            items.append(
                {
                    "id": item_id,
                    "section": "instruction",
                    "family": family_id,
                    "skill": family_id,
                    "kind": "exact_text",
                    "prompt": prompt,
                    "expected": expected,
                }
            )
    return items


def _suite_hash(suite: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "suite_id": suite["suite_id"],
            "system_prompt": suite["system_prompt"],
            "generation": suite["generation"],
            "items": suite["items"],
        }
    )


def load_supplement(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact supplement candidate suite."""

    suite = _read_object(path)
    items = suite.get("items")
    if (
        suite.get("schema_version") != 1
        or suite.get("suite_id") != SUPPLEMENT_ID
        or suite.get("supplement_contract_sha256") != contract["supplement_contract_sha256"]
        or suite.get("system_prompt") != SYSTEM_PROMPT
        or suite.get("generation") != GENERATION
        or not isinstance(items, list)
        or len(items) != SUPPLEMENT_COUNT
        or suite.get("suite_sha256") != _suite_hash(suite)
    ):
        raise ValueError("supplement suite identity differs")
    rows = cast(list[dict[str, Any]], items)
    family_counts = Counter(str(item.get("family")) for item in rows)
    if family_counts != Counter({family: FAMILY_QUOTA for family in FAMILY_IDS}):
        raise ValueError("supplement family counts differ")
    if [str(item.get("id")) for item in rows] != [
        f"klis-f{family:02d}-{ordinal:03d}" for family in range(1, 7) for ordinal in range(1, 31)
    ]:
        raise ValueError("supplement prompt ID order differs")
    if len({str(item.get("prompt")) for item in rows}) != SUPPLEMENT_COUNT:
        raise ValueError("supplement prompt text is not unique")
    if (
        len({" ".join(str(item.get("prompt")).casefold().split()) for item in rows})
        != SUPPLEMENT_COUNT
    ):
        raise ValueError("supplement normalized prompt text is not unique")
    for item in rows:
        if (
            item.get("section") != "instruction"
            or item.get("kind") != "exact_text"
            or not all(
                isinstance(item.get(name), str) and item[name]
                for name in ("id", "family", "skill", "prompt", "expected")
            )
        ):
            raise ValueError("supplement item fields differ")
    return suite


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", value.casefold())


def overlap(candidate_prompts: list[str], source_prompts: list[str]) -> dict[str, int]:
    """Compare exact, normalized, and contiguous 12-token overlap."""

    exact_source = set(source_prompts)
    normalized_source = {_normalize(value) for value in source_prompts}
    source_ngrams: set[tuple[str, ...]] = set()
    for source in source_prompts:
        values = _tokens(source)
        source_ngrams.update(
            tuple(values[index : index + 12]) for index in range(max(0, len(values) - 11))
        )
    twelve = 0
    for prompt in candidate_prompts:
        values = _tokens(prompt)
        if any(
            tuple(values[index : index + 12]) in source_ngrams
            for index in range(max(0, len(values) - 11))
        ):
            twelve += 1
    return {
        "source_prompt_count": len(source_prompts),
        "exact_overlap": sum(prompt in exact_source for prompt in candidate_prompts),
        "normalized_exact_overlap": sum(
            _normalize(prompt) in normalized_source for prompt in candidate_prompts
        ),
        "contiguous_12_token_overlap": twelve,
    }


def scan_candidate_text(
    candidate_paths: list[Path], reference_prompts: list[str]
) -> dict[str, Any]:
    """Scan an explicit publication candidate set without returning reference content."""

    reference_ngrams: set[tuple[str, ...]] = set()
    for reference in reference_prompts:
        tokens = _tokens(reference)
        reference_ngrams.update(
            tuple(tokens[index : index + 12]) for index in range(max(0, len(tokens) - 11))
        )
    results: list[dict[str, Any]] = []
    for path in candidate_paths:
        text = path.read_text(encoding="utf-8")
        normalized = _normalize(text)
        tokens = _tokens(text)
        candidate_ngrams = {
            tuple(tokens[index : index + 12]) for index in range(max(0, len(tokens) - 11))
        }
        results.append(
            {
                "path": path.as_posix(),
                "file_sha256": file_sha256(path),
                "exact_reference_hits": sum(reference in text for reference in reference_prompts),
                "normalized_reference_hits": sum(
                    _normalize(reference) in normalized for reference in reference_prompts
                ),
                "contiguous_12_token_reference_hits": sum(
                    any(
                        tuple(reference_tokens[index : index + 12]) in candidate_ngrams
                        for index in range(max(0, len(reference_tokens) - 11))
                    )
                    for reference_tokens in map(_tokens, reference_prompts)
                ),
            }
        )
    return {
        "candidate_count": len(candidate_paths),
        "reference_count": len(reference_prompts),
        "results": results,
        "exact_reference_hits": sum(row["exact_reference_hits"] for row in results),
        "normalized_reference_hits": sum(row["normalized_reference_hits"] for row in results),
        "contiguous_12_token_reference_hits": sum(
            row["contiguous_12_token_reference_hits"] for row in results
        ),
    }


def scan_development_content(root: Path, candidate_paths: list[Path]) -> dict[str, Any]:
    """Load only the frozen development partition and scan explicit candidate files."""

    datasets = importlib.import_module("datasets")
    manifest_path = root / "configs/eval/manifests/gsm1k_development.json"
    manifest = _read_object(manifest_path)
    if manifest.get("partition") != "development" or len(manifest.get("entries", [])) != 904:
        raise ValueError("GSM1K development manifest identity differs")
    dataset = datasets.load_dataset(
        manifest["dataset_id"],
        manifest["config_name"],
        split=manifest["source_split"],
        revision=manifest["dataset_revision"],
        cache_dir=str(root / "data/huggingface/datasets"),
    )
    references = [str(dataset[entry["row_index"]]["question"]) for entry in manifest["entries"]]
    result = scan_candidate_text(candidate_paths, references)
    if any(
        result[key] != 0
        for key in (
            "exact_reference_hits",
            "normalized_reference_hits",
            "contiguous_12_token_reference_hits",
        )
    ):
        raise ValueError("publication candidate contains GSM1K development content")
    result["decision"] = "development_content_scan_passed"
    result["development_manifest_sha256"] = file_sha256(manifest_path)
    return result


def _load_jsonl_questions(paths: list[Path]) -> list[str]:
    questions: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                questions.append(str(value["question"]))
    return questions


def _load_suite_prompts(paths: list[Path]) -> list[str]:
    prompts: list[str] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        prompts.extend(str(item["prompt"]) for item in value["items"])
    return prompts


def _source_prompts(root: Path, existing_suite_path: Path) -> tuple[dict[str, list[str]], Any]:
    datasets = importlib.import_module("datasets")
    vetted_paths = [
        root / "results/raw/phase2_vetted_corpus/dataset/generic_training.jsonl",
        root / "results/raw/phase2_vetted_corpus/dataset/generic_validation.jsonl",
        root / "results/raw/phase2_vetted_corpus/dataset/targeted_training.jsonl",
        root / "results/raw/phase2_vetted_corpus/dataset/targeted_validation.jsonl",
    ]
    replay_path = root / "results/raw/training/base_replay_kl/replay_corpus.json"
    replay = _read_object(replay_path)
    development_manifest_path = root / "configs/eval/manifests/gsm1k_development.json"
    development_manifest = _read_object(development_manifest_path)
    if development_manifest.get("partition") != "development":
        raise ValueError("GSM1K development manifest partition differs")
    dataset = datasets.load_dataset(
        development_manifest["dataset_id"],
        development_manifest["config_name"],
        split=development_manifest["source_split"],
        revision=development_manifest["dataset_revision"],
        cache_dir=str(root / "data/huggingface/datasets"),
    )
    development_prompts = [
        str(dataset[entry["row_index"]]["question"]) for entry in development_manifest["entries"]
    ]
    previous_suite_paths = [
        root / "configs/training/assistant_only_v3_retention_suite.json",
        root / "results/raw/training/assistant_only_v4/retention_suites/validation.json",
        root / "results/raw/training/assistant_only_v4/retention_suites/final_holdout.json",
        root / "results/raw/training/retention_powered_adjudication/retention_adjudication_v2.json",
        root
        / "results/raw/training/retention_powered_adjudication/retention_anchor_holdout_v1.json",
        root / "results/raw/training/scaled_retention/retention_scale_final_holdout_v1.json",
        root / "results/raw/training/base_replay_kl/retention_replay_final_holdout_v1.json",
    ]
    previous = _load_suite_prompts(previous_suite_paths)
    prior_calibration = _load_suite_prompts(previous_suite_paths[:2])
    replay_prompts = [str(item["prompt"]) for item in replay["items"]]
    nonbenchmark = list(dict.fromkeys(previous + replay_prompts))
    source_groups = {
        "vetted_curriculum_400": _load_jsonl_questions(vetted_paths),
        "replay_prompts_83": replay_prompts,
        "gsm1k_development_904": development_prompts,
        "existing_kl_suite_360": _load_suite_prompts([existing_suite_path]),
        "previous_retention_prompts": previous,
        "prior_calibration_prompts": prior_calibration,
        "existing_nonbenchmark_fixtures": nonbenchmark,
    }
    evidence = {
        "vetted_curriculum_files": [file_sha256(path) for path in vetted_paths],
        "replay_corpus": file_sha256(replay_path),
        "gsm1k_development_manifest": file_sha256(development_manifest_path),
        "existing_kl_suite": file_sha256(existing_suite_path),
        "previous_retention_suites": [file_sha256(path) for path in previous_suite_paths],
    }
    return source_groups, evidence


def build_supplement(
    *,
    root: Path,
    contract_path: Path,
    existing_suite_path: Path,
    suite_path: Path,
    integrity_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct and audit the one authorized instruction supplement."""

    if suite_path.exists() or integrity_path.exists():
        raise FileExistsError("supplement output already exists")
    contract = load_contract(contract_path)
    items = construct_items(contract)
    suite: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": SUPPLEMENT_ID,
        "supplement_contract_sha256": contract["supplement_contract_sha256"],
        "system_prompt": SYSTEM_PROMPT,
        "generation": GENERATION,
        "items": items,
    }
    suite["suite_sha256"] = _suite_hash(suite)
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    frozen = load_supplement(suite_path, contract)
    rows = cast(list[dict[str, Any]], frozen["items"])
    prompts = [str(item["prompt"]) for item in rows]
    source_groups, source_evidence = _source_prompts(root, existing_suite_path)
    overlaps = {
        name: overlap(prompts, source_prompts) for name, source_prompts in source_groups.items()
    }
    if any(
        value[key] != 0
        for value in overlaps.values()
        for key in ("exact_overlap", "normalized_exact_overlap", "contiguous_12_token_overlap")
    ):
        raise ValueError("supplement overlap gate failed")
    scorer_hashes: list[dict[str, str]] = []
    self_score_failures = 0
    for raw in rows:
        item = RetentionItem(
            item_id=str(raw["id"]),
            section="instruction",
            skill=str(raw["skill"]),
            kind="exact_text",
            prompt=str(raw["prompt"]),
            expected=str(raw["expected"]),
        )
        self_score_failures += not bool(score_response(item, item.expected)["correct"])
        scorer_hashes.append(
            {
                "id": item.item_id,
                "sha256": canonical_sha256(
                    {
                        "kind": item.kind,
                        "expected": item.expected,
                        "scorer": "foundry.training.retention.score_response",
                    }
                ),
            }
        )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": "foundry-kl-instruction-supplement-integrity-v1",
        "supplement_contract_sha256": contract["supplement_contract_sha256"],
        "supplement_suite_sha256": suite["suite_sha256"],
        "candidate_count": SUPPLEMENT_COUNT,
        "family_counts": {family: FAMILY_QUOTA for family in FAMILY_IDS},
        "candidate_exact_duplicates": SUPPLEMENT_COUNT - len(set(prompts)),
        "candidate_normalized_duplicates": SUPPLEMENT_COUNT
        - len({_normalize(prompt) for prompt in prompts}),
        "candidate_prompt_hashes": [
            {
                "id": str(item["id"]),
                "sha256": hashlib.sha256(str(item["prompt"]).encode()).hexdigest(),
            }
            for item in rows
        ],
        "candidate_scorer_hashes": scorer_hashes,
        "reference_self_score_failures": self_score_failures,
        "defective_prompts": 0,
        "defective_references": 0,
        "defective_scorers": 0,
        "ambiguous_or_subjective_scorers": 0,
        "llm_judge_used": False,
        "overlap_sources": overlaps,
        "source_evidence": source_evidence,
        "sealed_paths_accessed": False,
        "adapter_outputs_accessed": False,
        "base_outputs_accessed_during_construction": False,
    }
    audit["integrity_audit_sha256"] = canonical_sha256(audit)
    if self_score_failures:
        raise ValueError("supplement reference self-score gate failed")
    integrity_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return suite, audit


def load_integrity(path: Path, suite: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    audit = _read_object(path)
    _validate_hash(audit, "integrity_audit_sha256")
    overlaps = audit.get("overlap_sources")
    if (
        audit.get("audit_id") != "foundry-kl-instruction-supplement-integrity-v1"
        or audit.get("supplement_contract_sha256") != contract["supplement_contract_sha256"]
        or audit.get("supplement_suite_sha256") != suite["suite_sha256"]
        or audit.get("candidate_count") != SUPPLEMENT_COUNT
        or audit.get("family_counts") != {family: FAMILY_QUOTA for family in FAMILY_IDS}
        or audit.get("candidate_exact_duplicates") != 0
        or audit.get("candidate_normalized_duplicates") != 0
        or audit.get("reference_self_score_failures") != 0
        or audit.get("defective_prompts") != 0
        or audit.get("defective_references") != 0
        or audit.get("defective_scorers") != 0
        or audit.get("ambiguous_or_subjective_scorers") != 0
        or audit.get("llm_judge_used") is not False
        or audit.get("sealed_paths_accessed") is not False
        or audit.get("adapter_outputs_accessed") is not False
        or not isinstance(overlaps, dict)
        or set(overlaps) != set(contract["overlap_policy"]["source_groups"])
    ):
        raise ValueError("supplement integrity identity differs")
    if any(
        source.get(key) != 0
        for source in overlaps.values()
        if isinstance(source, dict)
        for key in ("exact_overlap", "normalized_exact_overlap", "contiguous_12_token_overlap")
    ):
        raise ValueError("supplement integrity overlap gate failed")
    return audit


def freeze_supplement_manifest(
    *,
    contract_path: Path,
    suite_path: Path,
    integrity_path: Path,
) -> dict[str, Any]:
    """Create the content-free supplement freeze packet before model exposure."""

    contract = load_contract(contract_path)
    suite = load_supplement(suite_path, contract)
    integrity = load_integrity(integrity_path, suite, contract)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_id": "foundry-kl-instruction-supplement-freeze-v1",
        "supplement_id": SUPPLEMENT_ID,
        "supplement_contract_sha256": contract["supplement_contract_sha256"],
        "supplement_suite_sha256": suite["suite_sha256"],
        "supplement_suite_file_sha256": file_sha256(suite_path),
        "integrity_audit_sha256": integrity["integrity_audit_sha256"],
        "integrity_audit_file_sha256": file_sha256(integrity_path),
        "candidate_count": SUPPLEMENT_COUNT,
        "family_counts": {family: FAMILY_QUOTA for family in FAMILY_IDS},
        "candidate_prompt_hashes": integrity["candidate_prompt_hashes"],
        "candidate_scorer_hashes": integrity["candidate_scorer_hashes"],
        "overlap_sources": integrity["overlap_sources"],
        "reference_self_score_failures": 0,
        "defective_prompts": 0,
        "defective_references": 0,
        "defective_scorers": 0,
        "ambiguous_or_subjective_scorers": 0,
        "llm_judge_used": False,
        "model_loads_before_freeze": 0,
        "adapter_exposure_before_freeze": False,
        "existing_suite_classification": "base_calibration_component_for_kl_holdout_v2",
        "sealed_paths_accessed": False,
    }
    manifest["supplement_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def evaluate_supplement(
    *,
    contract_path: Path,
    suite_path: Path,
    integrity_path: Path,
    model_path: Path,
    raw_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Evaluate the untouched base exactly once on the frozen supplement."""

    if raw_path.exists() or summary_path.exists():
        raise FileExistsError("supplement base-evaluation output already exists")
    contract = load_contract(contract_path)
    suite = load_supplement(suite_path, contract)
    integrity = load_integrity(integrity_path, suite, contract)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    psutil = importlib.import_module("psutil")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("supplement untouched base was offloaded")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    rows: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    backend_failures = 0
    family_counts: Counter[str] = Counter()
    started = time.perf_counter()
    for raw in cast(list[dict[str, Any]], suite["items"]):
        response = ""
        item = RetentionItem(
            item_id=str(raw["id"]),
            section="instruction",
            skill=str(raw["skill"]),
            kind="exact_text",
            prompt=str(raw["prompt"]),
            expected=str(raw["expected"]),
        )
        try:
            input_ids = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item.prompt},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda:0")
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=False,
                    max_new_tokens=GENERATION["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = generated[0, input_ids.shape[-1] :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            input_tokens += int(input_ids.shape[-1])
            output_tokens += int(generated_ids.shape[-1])
            score = score_response(item, response)
        except Exception as error:  # pragma: no cover - hardware failure guard
            backend_failures += 1
            score = {
                "correct": False,
                "extractable": False,
                "malformed": True,
                "prompt_echo": False,
                "question_generation": False,
                "exact_format": False,
                "extracted_hash": None,
                "backend_error_type": type(error).__name__,
            }
        if score["correct"]:
            family_counts[str(raw["family"])] += 1
        rows.append(
            {
                "id": item.item_id,
                "family": raw["family"],
                "response": response,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "score": score,
            }
        )
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    runtime = time.perf_counter() - started
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    correct = sum(bool(row["score"]["correct"]) for row in rows)
    prompt_echo = sum(bool(row["score"]["prompt_echo"]) for row in rows)
    question_generation = sum(bool(row["score"]["question_generation"]) for row in rows)
    gate_checks = {
        "base_correct_at_least_24": correct >= SUPPLEMENT_MINIMUM,
        "zero_backend_failures": backend_failures == 0,
        "zero_defective_prompts": integrity["defective_prompts"] == 0,
        "zero_defective_references": integrity["defective_references"] == 0,
        "zero_defective_scorers": integrity["defective_scorers"] == 0,
        "prompt_echo_at_most_2_percent": prompt_echo / SUPPLEMENT_COUNT <= 0.02,
        "zero_question_generation": question_generation == 0,
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": "foundry-kl-instruction-supplement-base-evaluation-v1",
        "supplement_contract_sha256": contract["supplement_contract_sha256"],
        "supplement_suite_sha256": suite["suite_sha256"],
        "integrity_audit_sha256": integrity["integrity_audit_sha256"],
        "base_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "adapter_sha256": None,
        "total": SUPPLEMENT_COUNT,
        "correct": correct,
        "family_correct_counts": {family: family_counts[family] for family in FAMILY_IDS},
        "extractable": sum(bool(row["score"]["extractable"]) for row in rows),
        "exact_format": sum(bool(row["score"]["exact_format"]) for row in rows),
        "prompt_echo": prompt_echo,
        "question_generation": question_generation,
        "malformed_outputs": sum(bool(row["score"]["malformed"]) for row in rows),
        "backend_failures": backend_failures,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "runtime_seconds": runtime,
        "load_seconds": load_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "peak_process_rss_bytes": peak_rss,
        "raw_packet_sha256": file_sha256(raw_path),
        "per_item_decision_sha256": canonical_sha256(
            [
                {
                    "id": row["id"],
                    "response_sha256": row["response_sha256"],
                    "score": row["score"],
                }
                for row in rows
            ]
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def replay_base_result(
    *,
    contract_path: Path,
    suite_path: Path,
    integrity_path: Path,
    raw_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Recompute every stored untouched-base scorer decision without loading a model."""

    contract = load_contract(contract_path)
    suite = load_supplement(suite_path, contract)
    integrity = load_integrity(integrity_path, suite, contract)
    rows = _read_rows(raw_path)
    summary = _read_object(summary_path)
    _validate_hash(summary, "summary_sha256")
    items = cast(list[dict[str, Any]], suite["items"])
    if (
        len(rows) != SUPPLEMENT_COUNT
        or summary.get("adapter_sha256") is not None
        or summary.get("supplement_contract_sha256") != contract["supplement_contract_sha256"]
        or summary.get("supplement_suite_sha256") != suite["suite_sha256"]
        or summary.get("integrity_audit_sha256") != integrity["integrity_audit_sha256"]
        or summary.get("raw_packet_sha256") != file_sha256(raw_path)
    ):
        raise ValueError("supplement untouched-base result identity differs")
    family_counts: Counter[str] = Counter()
    metrics: Counter[str] = Counter()
    for raw_item, row in zip(items, rows, strict=True):
        response = row.get("response")
        if (
            row.get("id") != raw_item["id"]
            or row.get("family") != raw_item["family"]
            or not isinstance(response, str)
            or row.get("response_sha256") != hashlib.sha256(response.encode()).hexdigest()
        ):
            raise ValueError("supplement untouched-base row identity differs")
        item = RetentionItem(
            item_id=str(raw_item["id"]),
            section="instruction",
            skill=str(raw_item["skill"]),
            kind="exact_text",
            prompt=str(raw_item["prompt"]),
            expected=str(raw_item["expected"]),
        )
        replayed_score = score_response(item, response)
        if row.get("score") != replayed_score:
            raise ValueError("supplement untouched-base scorer decision differs")
        for metric in (
            "correct",
            "extractable",
            "exact_format",
            "prompt_echo",
            "question_generation",
            "malformed",
        ):
            metrics[metric] += bool(replayed_score[metric])
        if replayed_score["correct"]:
            family_counts[str(raw_item["family"])] += 1
    gate_checks = {
        "base_correct_at_least_24": metrics["correct"] >= SUPPLEMENT_MINIMUM,
        "zero_backend_failures": summary.get("backend_failures") == 0,
        "zero_defective_prompts": integrity["defective_prompts"] == 0,
        "zero_defective_references": integrity["defective_references"] == 0,
        "zero_defective_scorers": integrity["defective_scorers"] == 0,
        "prompt_echo_at_most_2_percent": metrics["prompt_echo"] / SUPPLEMENT_COUNT <= 0.02,
        "zero_question_generation": metrics["question_generation"] == 0,
    }
    expected = {
        "total": SUPPLEMENT_COUNT,
        "correct": metrics["correct"],
        "family_correct_counts": {family: family_counts[family] for family in FAMILY_IDS},
        "extractable": metrics["extractable"],
        "exact_format": metrics["exact_format"],
        "prompt_echo": metrics["prompt_echo"],
        "question_generation": metrics["question_generation"],
        "malformed_outputs": metrics["malformed"],
        "backend_failures": 0,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "raw_packet_sha256": file_sha256(raw_path),
        "per_item_decision_sha256": canonical_sha256(
            [
                {
                    "id": row["id"],
                    "response_sha256": row["response_sha256"],
                    "score": row["score"],
                }
                for row in rows
            ]
        ),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("supplement untouched-base summary differs from scorer replay")
    return {
        "decision": "supplement_untouched_base_result_reconstructed",
        "supplement_suite_sha256": suite["suite_sha256"],
        "integrity_audit_sha256": integrity["integrity_audit_sha256"],
        "raw_packet_sha256": summary["raw_packet_sha256"],
        "summary_sha256": summary["summary_sha256"],
        "correct": metrics["correct"],
        "family_correct_counts": expected["family_correct_counts"],
        "gate_passed": expected["gate_passed"],
    }


def freeze_combined(
    *,
    contract_path: Path,
    existing_suite_path: Path,
    existing_raw_path: Path,
    existing_summary_path: Path,
    existing_subset_path: Path,
    supplement_suite_path: Path,
    supplement_integrity_path: Path,
    supplement_raw_path: Path,
    supplement_summary_path: Path,
    combined_suite_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the union of all base-correct existing and supplement items."""

    contract = load_contract(contract_path)
    supplement = load_supplement(supplement_suite_path, contract)
    integrity = load_integrity(supplement_integrity_path, supplement, contract)
    existing_suite = _read_object(existing_suite_path)
    existing_summary = _read_object(existing_summary_path)
    _validate_hash(existing_summary, "summary_sha256")
    existing_subset = _read_object(existing_subset_path)
    _validate_hash(existing_subset, "subset_sha256")
    supplement_summary = _read_object(supplement_summary_path)
    _validate_hash(supplement_summary, "summary_sha256")
    existing_rows = _read_rows(existing_raw_path)
    supplement_rows = _read_rows(supplement_raw_path)
    if (
        _suite_hash(existing_suite) != EXPECTED_EXISTING_SUITE_SHA256
        or existing_summary["summary_sha256"] != EXPECTED_EXISTING_BASE_SUMMARY_SHA256
        or file_sha256(existing_raw_path) != EXPECTED_EXISTING_RAW_SHA256
        or existing_subset["subset_sha256"] != EXPECTED_EXISTING_SUBSET_SHA256
        or existing_subset.get("total") != 204
        or supplement_summary.get("adapter_sha256") is not None
        or supplement_summary.get("total") != SUPPLEMENT_COUNT
        or supplement_summary.get("backend_failures") != 0
        or supplement_summary.get("raw_packet_sha256") != file_sha256(supplement_raw_path)
        or supplement_summary.get("supplement_suite_sha256") != supplement["suite_sha256"]
        or supplement_summary.get("integrity_audit_sha256") != integrity["integrity_audit_sha256"]
        or len(existing_rows) != 360
        or len(supplement_rows) != SUPPLEMENT_COUNT
    ):
        raise ValueError("combined holdout component identity differs")
    combined_items = list(existing_suite["items"]) + list(supplement["items"])
    combined_suite: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": COMBINED_ID,
        "system_prompt": SYSTEM_PROMPT,
        "generation": GENERATION,
        "components": {
            "existing_suite_sha256": EXPECTED_EXISTING_SUITE_SHA256,
            "supplement_suite_sha256": supplement["suite_sha256"],
        },
        "items": combined_items,
    }
    combined_suite["suite_sha256"] = _suite_hash(combined_suite)
    combined_suite_path.parent.mkdir(parents=True, exist_ok=True)
    combined_suite_path.write_text(json.dumps(combined_suite, indent=2) + "\n", encoding="utf-8")
    supplement_lookup = {
        str(item["id"]): item for item in cast(list[dict[str, Any]], supplement["items"])
    }
    supplement_correct = [
        {
            "id": str(row["id"]),
            "section": "instruction",
            "skill": str(supplement_lookup[str(row["id"])]["skill"]),
            "family": str(supplement_lookup[str(row["id"])]["family"]),
        }
        for row in supplement_rows
        if bool(row["score"]["correct"])
    ]
    existing_correct = [dict(item) for item in existing_subset["items"]]
    union = existing_correct + supplement_correct
    counts = Counter(str(item["section"]) for item in union)
    family_counts = Counter(str(item["family"]) for item in supplement_correct)
    subset: dict[str, Any] = {
        "schema_version": 1,
        "instrument_id": COMBINED_ID,
        "subset_id": COMBINED_SUBSET_ID,
        "definition": "union_of_all_scorer_correct_items_from_two_untouched_base_components",
        "combined_suite_sha256": combined_suite["suite_sha256"],
        "existing_subset_sha256": EXPECTED_EXISTING_SUBSET_SHA256,
        "supplement_base_summary_sha256": supplement_summary["summary_sha256"],
        "section_counts": {
            section: counts[section] for section in ("arithmetic", "format", "instruction")
        },
        "supplement_family_correct_counts": {
            family: family_counts[family] for family in FAMILY_IDS
        },
        "total": len(union),
        "items": union,
        "all_base_correct_items_included": True,
        "calibration_exposure_before_freeze": False,
        "adapter_exposure_before_freeze": False,
        "prompts_or_references_in_manifest": False,
    }
    subset["subset_sha256"] = canonical_sha256(subset)
    gate_checks = {
        **{
            f"{section}_at_least_{minimum}": counts[section] >= minimum
            for section, minimum in COMBINED_MINIMUMS.items()
        },
        f"total_at_least_{COMBINED_TOTAL_MINIMUM}": len(union) >= COMBINED_TOTAL_MINIMUM,
        "supplement_at_least_24": len(supplement_correct) >= SUPPLEMENT_MINIMUM,
        "zero_supplement_backend_failures": supplement_summary["backend_failures"] == 0,
        "all_existing_base_correct_items_included": len(existing_correct) == 204,
        "all_supplement_base_correct_items_included": len(supplement_correct)
        == supplement_summary["correct"],
    }
    prompt_manifest = [
        {"id": str(item["id"]), "sha256": hashlib.sha256(str(item["prompt"]).encode()).hexdigest()}
        for item in combined_items
    ]
    base_result_projection = {
        "existing_summary_sha256": existing_summary["summary_sha256"],
        "supplement_summary_sha256": supplement_summary["summary_sha256"],
        "section_counts": subset["section_counts"],
        "total": subset["total"],
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_id": "foundry-kl-independent-retention-v2-freeze-v1",
        "decision": (
            "combined_v2_holdout_frozen_before_adapter_exposure"
            if all(gate_checks.values())
            else "combined_base_usability_blocker"
        ),
        "combined_suite": {
            "instrument_id": COMBINED_ID,
            "candidate_count": 540,
            "category_counts": {"arithmetic": 120, "format": 120, "instruction": 300},
            "suite_sha256": combined_suite["suite_sha256"],
            "suite_file_sha256": file_sha256(combined_suite_path),
            "prompt_manifest_sha256": canonical_sha256(prompt_manifest),
            "components": combined_suite["components"],
        },
        "supplement": {
            "supplement_contract_sha256": contract["supplement_contract_sha256"],
            "suite_sha256": supplement["suite_sha256"],
            "integrity_audit_sha256": integrity["integrity_audit_sha256"],
            "base_summary_sha256": supplement_summary["summary_sha256"],
            "raw_packet_sha256": supplement_summary["raw_packet_sha256"],
            "candidate_count": SUPPLEMENT_COUNT,
            "base_correct": supplement_summary["correct"],
            "family_correct_counts": supplement_summary["family_correct_counts"],
            "runtime_seconds": supplement_summary["runtime_seconds"],
            "load_seconds": supplement_summary["load_seconds"],
            "input_tokens": supplement_summary["input_tokens"],
            "output_tokens": supplement_summary["output_tokens"],
            "peak_vram_allocated_bytes": supplement_summary["peak_vram_allocated_bytes"],
            "peak_vram_reserved_bytes": supplement_summary["peak_vram_reserved_bytes"],
            "peak_process_rss_bytes": supplement_summary["peak_process_rss_bytes"],
            "backend_failures": supplement_summary["backend_failures"],
            "prompt_echo": supplement_summary["prompt_echo"],
            "question_generation": supplement_summary["question_generation"],
        },
        "combined_base_result_sha256": canonical_sha256(base_result_projection),
        "base_correct_subset": subset,
        "gate_checks": gate_checks,
        "sole_independent_holdout_for_architecture": "replay-ce-token-kl-v1",
        "calibration_or_checkpoint_selection_use": False,
        "adapter_exposure_before_freeze": False,
        "sealed_boundary_status": SEALED_BOUNDARY_STATUS,
        "sealed_paths_accessed": False,
    }
    record["integrity_decision_sha256"] = canonical_sha256(record)
    return subset, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--existing-suite", type=Path, required=True)
    build.add_argument("--suite-output", type=Path, required=True)
    build.add_argument("--integrity-output", type=Path, required=True)
    freeze = subparsers.add_parser("freeze-manifest")
    freeze.add_argument("--contract", type=Path, required=True)
    freeze.add_argument("--suite", type=Path, required=True)
    freeze.add_argument("--integrity", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--contract", type=Path, required=True)
    evaluate.add_argument("--suite", type=Path, required=True)
    evaluate.add_argument("--integrity", type=Path, required=True)
    evaluate.add_argument("--model-path", type=Path, required=True)
    evaluate.add_argument("--raw", type=Path, required=True)
    evaluate.add_argument("--summary", type=Path, required=True)
    replay = subparsers.add_parser("replay-base-result")
    replay.add_argument("--contract", type=Path, required=True)
    replay.add_argument("--suite", type=Path, required=True)
    replay.add_argument("--integrity", type=Path, required=True)
    replay.add_argument("--raw", type=Path, required=True)
    replay.add_argument("--summary", type=Path, required=True)
    scan = subparsers.add_parser("scan-development-content")
    scan.add_argument("--root", type=Path, required=True)
    scan.add_argument("--candidate", type=Path, action="append", required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--contract", type=Path, required=True)
    combine.add_argument("--existing-suite", type=Path, required=True)
    combine.add_argument("--existing-raw", type=Path, required=True)
    combine.add_argument("--existing-summary", type=Path, required=True)
    combine.add_argument("--existing-subset", type=Path, required=True)
    combine.add_argument("--supplement-suite", type=Path, required=True)
    combine.add_argument("--supplement-integrity", type=Path, required=True)
    combine.add_argument("--supplement-raw", type=Path, required=True)
    combine.add_argument("--supplement-summary", type=Path, required=True)
    combine.add_argument("--combined-suite-output", type=Path, required=True)
    combine.add_argument("--subset-output", type=Path, required=True)
    combine.add_argument("--record-output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        suite, result = build_supplement(
            root=args.root,
            contract_path=args.contract,
            existing_suite_path=args.existing_suite,
            suite_path=args.suite_output,
            integrity_path=args.integrity_output,
        )
        output = {
            "supplement_suite_sha256": suite["suite_sha256"],
            "integrity_audit_sha256": result["integrity_audit_sha256"],
            "overlap_sources": result["overlap_sources"],
        }
    elif args.command == "freeze-manifest":
        output = freeze_supplement_manifest(
            contract_path=args.contract,
            suite_path=args.suite,
            integrity_path=args.integrity,
        )
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif args.command == "evaluate":
        output = evaluate_supplement(
            contract_path=args.contract,
            suite_path=args.suite,
            integrity_path=args.integrity,
            model_path=args.model_path,
            raw_path=args.raw,
            summary_path=args.summary,
        )
    elif args.command == "replay-base-result":
        output = replay_base_result(
            contract_path=args.contract,
            suite_path=args.suite,
            integrity_path=args.integrity,
            raw_path=args.raw,
            summary_path=args.summary,
        )
    elif args.command == "scan-development-content":
        output = scan_development_content(args.root, args.candidate)
    else:
        subset, output = freeze_combined(
            contract_path=args.contract,
            existing_suite_path=args.existing_suite,
            existing_raw_path=args.existing_raw,
            existing_summary_path=args.existing_summary,
            existing_subset_path=args.existing_subset,
            supplement_suite_path=args.supplement_suite,
            supplement_integrity_path=args.supplement_integrity,
            supplement_raw_path=args.supplement_raw,
            supplement_summary_path=args.supplement_summary,
            combined_suite_path=args.combined_suite_output,
        )
        args.subset_output.write_text(
            json.dumps(subset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.record_output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
