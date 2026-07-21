"""Build the original independent holdout for base-behavior replay experiments."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from foundry.config import load_config
from foundry.evaluation.manifests import load_manifest
from foundry.training.config import canonical_sha256
from foundry.training.retention import RetentionItem, score_response

SUITE_ID = "foundry-retention-replay-final-holdout-v1"
ARTIFACT_ID = "foundry-retention-replay-final-holdout-artifact-v1"
CONFIGURATION_ID = "foundry-retention-replay-final-holdout-config-v1"
SCORER_CONTRACT_ID = "foundry-retention-objective-scorer-v1"
PROMPT_AUDIT_ID = "foundry-retention-prompt-disjointness-audit-v1"
PRODUCTION_PRIOR_CORPUS_TOTAL = 3314
SYSTEM_PROMPT = (
    "Carry out the user's stated operation exactly and return only the requested response."
)
SECTION_COUNTS = {"arithmetic": 150, "format": 150, "instruction": 150}
GENERATION = {"do_sample": False, "max_new_tokens": 192, "seed": 20260721}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_WORDS_A = (
    "amber",
    "birch",
    "cobalt",
    "dahlia",
    "ember",
    "fern",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "lilac",
    "maple",
    "nectar",
    "opal",
    "pebble",
    "quartz",
    "reed",
    "saffron",
    "thistle",
    "umber",
    "violet",
    "willow",
    "xenon",
    "yarrow",
    "zinnia",
    "acorn",
    "basil",
    "coral",
    "drift",
)
_WORDS_B = (
    "elm",
    "frost",
    "grove",
    "heather",
    "iris",
    "jade",
    "knoll",
    "lagoon",
    "moss",
    "north",
    "oasis",
    "pine",
    "quiet",
    "ridge",
    "spruce",
    "tundra",
    "upland",
    "vale",
    "west",
    "yucca",
    "zephyr",
    "alder",
    "brook",
    "clover",
    "dune",
    "estuary",
    "flint",
    "glade",
    "hazel",
    "islet",
)
_WORDS_C = (
    "kite",
    "lantern",
    "meadow",
    "nutmeg",
    "orchid",
    "plume",
    "raven",
    "silver",
    "topaz",
    "valley",
    "wren",
    "azure",
    "beacon",
    "canyon",
    "delta",
    "echo",
    "forest",
    "granite",
    "horizon",
    "ivory",
    "jasmine",
    "lumen",
    "marsh",
    "navy",
    "orbit",
    "prairie",
    "ripple",
    "summit",
    "timber",
    "verdant",
)

_SCORER_CONTRACT: dict[str, object] = {
    "contract_id": SCORER_CONTRACT_ID,
    "numeric_terminal": "canonical Fraction equality from the deterministic extractor",
    "exact_text": "Unicode string equality after outer whitespace removal",
    "json_exact": "parsed JSON object equality with no extra keys",
    "subjective_judgment": False,
}

_BUILD_CONFIGURATION: dict[str, object] = {
    "configuration_id": CONFIGURATION_ID,
    "suite_id": SUITE_ID,
    "section_counts": SECTION_COUNTS,
    "generation": GENERATION,
    "arithmetic_families": [
        "sequential_integer_adjustment",
        "scaled_sum_with_offset",
        "exact_division_then_adjustment",
        "symmetric_three_value_mean",
        "integer_percentage",
    ],
    "format_families": [
        "spaced_vertical_bar",
        "double_colon_identifier",
        "three_field_json",
        "bracket_parenthesis_compound",
        "two_line_literal",
    ],
    "instruction_families": [
        "reverse_token_order",
        "alphabetical_token_order",
        "ordered_initial_letters",
        "left_rotation_by_two",
        "odd_position_selection",
    ],
    "items_per_family": 30,
    "prompt_audit": {
        "normalization": "lowercase ASCII alphanumeric tokens",
        "exact": True,
        "contiguous_token_window": 12,
    },
    "lexical_inventory_sha256": canonical_sha256([_WORDS_A, _WORDS_B, _WORDS_C]),
}


@dataclass(frozen=True)
class PriorPromptCorpus:
    """One named prior prompt collection used only for disjointness auditing."""

    corpus_id: str
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class ReplayHoldoutArtifacts:
    """JSON-ready holdout suite plus its content-free integrity evidence."""

    suite: dict[str, Any]
    evidence: dict[str, Any]


CorpusType = Literal[
    "retention_suite_json",
    "records_jsonl",
    "development_manifest_arrow",
]


@dataclass(frozen=True)
class PriorCorpusSpec:
    """Explicit source schema for one prior prompt corpus."""

    corpus_id: str
    corpus_type: CorpusType
    path: Path
    field: str
    source_path: Path | None = None
    config_path: Path | None = None


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def _normalized_text(text: str) -> str:
    return " ".join(_normalized_tokens(text))


def _token_windows(text: str, *, size: int = 12) -> set[tuple[str, ...]]:
    tokens = _normalized_tokens(text)
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def parse_prior_corpus_spec(value: str) -> PriorCorpusSpec:
    """Parse one unambiguous JSON CLI specification for a prior prompt corpus."""

    try:
        raw: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("prior corpus specification must be one JSON object") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("prior corpus specification must be one string-keyed object")
    root = cast(dict[str, object], raw)
    corpus_type = root.get("corpus_type")
    if corpus_type not in {
        "retention_suite_json",
        "records_jsonl",
        "development_manifest_arrow",
    }:
        raise ValueError("prior corpus specification has an unknown corpus_type")
    common = {"corpus_id", "corpus_type", "path", "field"}
    required = (
        common | {"source_path", "config_path"}
        if corpus_type == "development_manifest_arrow"
        else common
    )
    if set(root) != required:
        raise ValueError("prior corpus specification fields differ for its corpus_type")
    if any(not isinstance(root.get(key), str) or not str(root[key]).strip() for key in required):
        raise ValueError("prior corpus specification values must be non-empty strings")
    field = cast(str, root["field"])
    if corpus_type == "retention_suite_json" and field != "items.prompt":
        raise ValueError("retention_suite_json requires the explicit field items.prompt")
    if corpus_type == "development_manifest_arrow" and field != "question":
        raise ValueError("development_manifest_arrow requires the explicit field question")
    if corpus_type == "records_jsonl" and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", field):
        raise ValueError("records_jsonl field must be one explicit top-level identifier")
    return PriorCorpusSpec(
        corpus_id=cast(str, root["corpus_id"]),
        corpus_type=cast(CorpusType, corpus_type),
        path=Path(cast(str, root["path"])),
        field=field,
        source_path=(
            Path(cast(str, root["source_path"]))
            if corpus_type == "development_manifest_arrow"
            else None
        ),
        config_path=(
            Path(cast(str, root["config_path"]))
            if corpus_type == "development_manifest_arrow"
            else None
        ),
    )


def load_prior_corpus_specs_file(path: Path) -> tuple[PriorCorpusSpec, ...]:
    """Load a Windows-shell-safe JSON array of strict prior-corpus specifications."""

    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("prior corpus specification file must contain valid JSON") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError("prior corpus specification file must contain a non-empty array")
    return tuple(
        parse_prior_corpus_spec(json.dumps(item, sort_keys=True, separators=(",", ":")))
        for item in raw
    )


def _load_retention_suite_prompts(spec: PriorCorpusSpec) -> tuple[str, ...]:
    try:
        raw: object = json.loads(spec.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load retention corpus {spec.corpus_id}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise ValueError(f"retention corpus {spec.corpus_id} lacks items")
    raw_items = cast(list[object], raw["items"])
    prompts: list[str] = []
    for position, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"retention corpus {spec.corpus_id} item {position} is not an object")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"retention corpus {spec.corpus_id} item {position} lacks prompt text")
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"retention corpus {spec.corpus_id} is empty")
    return tuple(prompts)


def _load_jsonl_prompts(spec: PriorCorpusSpec) -> tuple[str, ...]:
    try:
        lines = spec.path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not load JSONL corpus {spec.corpus_id}") from error
    prompts: list[str] = []
    for position, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSONL corpus {spec.corpus_id} line {position} is invalid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(f"JSONL corpus {spec.corpus_id} line {position} is not an object")
        prompt = raw.get(spec.field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"JSONL corpus {spec.corpus_id} line {position} lacks field {spec.field}"
            )
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"JSONL corpus {spec.corpus_id} is empty")
    return tuple(prompts)


def _load_development_prompts(spec: PriorCorpusSpec) -> tuple[str, ...]:
    if "sealed" in spec.path.name.lower():
        raise ValueError("sealed-final manifests are forbidden as replay-holdout audit inputs")
    if spec.config_path is None or spec.source_path is None:
        raise ValueError("development manifest corpus requires config_path and source_path")
    config = load_config(spec.config_path)
    manifest = load_manifest(spec.path, config)
    if manifest.partition != "development":
        raise ValueError("only the canonical development partition may be audited")
    try:
        datasets = importlib.import_module("datasets")
        dataset: Any = datasets.Dataset.from_file(str(spec.source_path))
    except (ImportError, OSError) as error:
        raise ValueError(f"could not load pinned Arrow corpus {spec.corpus_id}") from error
    if len(dataset) != config.dataset.expected_examples:
        raise ValueError("pinned Arrow corpus length differs from the evaluation configuration")
    question_column: object = dataset[spec.field]
    if not isinstance(question_column, list) or len(question_column) != len(dataset):
        raise ValueError("pinned Arrow corpus question column has an invalid shape")
    questions = cast(list[object], question_column)
    prompts: list[str] = []
    for entry in manifest.entries:
        prompt = questions[entry.row_index]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"development row {entry.row_index} lacks question text")
        prompts.append(prompt)
    return tuple(prompts)


def load_prior_prompt_corpus(spec: PriorCorpusSpec) -> PriorPromptCorpus:
    """Load only the explicitly declared prompt field from one prior source."""

    if spec.corpus_type == "retention_suite_json":
        prompts = _load_retention_suite_prompts(spec)
    elif spec.corpus_type == "records_jsonl":
        prompts = _load_jsonl_prompts(spec)
    else:
        prompts = _load_development_prompts(spec)
    return PriorPromptCorpus(corpus_id=spec.corpus_id, prompts=prompts)


def validate_production_corpus_inventory(
    specs: Sequence[PriorCorpusSpec], corpora: Sequence[PriorPromptCorpus]
) -> None:
    """Require the predeclared 7+4+1 production audit inventory and 3,314 prompts."""

    if len(specs) != 12 or len(corpora) != 12:
        raise ValueError("production freeze requires exactly twelve prior prompt corpora")
    spec_ids = [spec.corpus_id for spec in specs]
    corpus_ids = [corpus.corpus_id for corpus in corpora]
    if spec_ids != corpus_ids or len(set(spec_ids)) != 12:
        raise ValueError("production prior corpus IDs or ordering differ")
    counts_by_type: dict[str, list[int]] = {
        "retention_suite_json": [],
        "records_jsonl": [],
        "development_manifest_arrow": [],
    }
    for spec, corpus in zip(specs, corpora, strict=True):
        counts_by_type[spec.corpus_type].append(len(corpus.prompts))
    if sorted(counts_by_type["retention_suite_json"]) != [60, 90, 90, 120, 300, 300, 450]:
        raise ValueError("production retention-suite inventory differs")
    if sorted(counts_by_type["records_jsonl"]) != [50, 50, 450, 450]:
        raise ValueError("production synthetic-split inventory differs")
    if counts_by_type["development_manifest_arrow"] != [904]:
        raise ValueError("production development-manifest inventory differs")
    if sum(len(corpus.prompts) for corpus in corpora) != PRODUCTION_PRIOR_CORPUS_TOTAL:
        raise ValueError("production prior prompt total must equal 3,314")


def _arithmetic_items() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for family in range(5):
        for offset in range(30):
            item_number = family * 30 + offset
            if family == 0:
                start = 37 + 4 * offset
                increase = 13 + (7 * offset) % 31
                decrease = 4 + (11 * offset) % 17
                expected = start + increase - decrease
                prompt = (
                    f"Begin with the integer {start}. Increase it by {increase}, then decrease "
                    f"the result by {decrease}. Put the resulting integer on a final line written "
                    "as `Final answer: <number>`."
                )
                skill = "sequential_integer_adjustment"
            elif family == 1:
                left = 8 + offset
                right = 3 + (5 * offset) % 19
                multiplier = 2 + offset % 6
                adjustment = 7 + (3 * offset) % 23
                expected = (left + right) * multiplier + adjustment
                prompt = (
                    f"Add {left} to {right}, multiply that sum by {multiplier}, and then add "
                    f"{adjustment}. Report the integer on a last line in the form "
                    "`Final answer: <number>`."
                )
                skill = "scaled_sum_with_offset"
            elif family == 2:
                divisor = 2 + offset % 8
                quotient = 19 + 3 * offset
                dividend = divisor * quotient
                adjustment = 2 + (7 * offset) % 13
                expected = quotient - adjustment
                prompt = (
                    f"Divide {dividend} by {divisor} exactly, then subtract {adjustment} from the "
                    "quotient. Finish with `Final answer: <number>` on its own line."
                )
                skill = "exact_division_then_adjustment"
            elif family == 3:
                centre = 24 + 3 * offset
                distance = 2 + offset % 11
                low = centre - distance
                high = centre + distance
                expected = centre
                prompt = (
                    f"Find the arithmetic mean of the three integers {low}, {centre}, and {high}. "
                    "Give the mean on a terminal line formatted `Final answer: <number>`."
                )
                skill = "symmetric_three_value_mean"
            else:
                percentage = (10, 20, 25, 50, 75)[offset % 5]
                base = 40 * (offset + 2)
                expected = base * percentage // 100
                prompt = (
                    f"Determine {percentage} percent of the integer {base}. End the response with "
                    "a line matching `Final answer: <number>`."
                )
                skill = "integer_percentage"
            items.append(
                {
                    "id": f"rrfh-arithmetic-{item_number:03d}",
                    "section": "arithmetic",
                    "skill": skill,
                    "kind": "numeric_terminal",
                    "prompt": prompt,
                    "expected": str(expected),
                }
            )
    return items


def _format_items() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for family in range(5):
        for offset in range(30):
            item_number = family * 30 + offset
            first = _WORDS_A[offset]
            second = _WORDS_B[(offset * 7 + family) % 30]
            third = _WORDS_C[(offset * 11 + 2 * family) % 30]
            if family == 0:
                expected = f"{first} | {second} | {third}"
                prompt = (
                    f"Produce one line from the literal parts `{first}`, `{second}`, and `{third}` "
                    "in that order. Join adjacent parts with a vertical bar having one space on "
                    "each side. Output only the finished line."
                )
                skill = "spaced_vertical_bar"
                kind = "exact_text"
            elif family == 1:
                expected = f"{first}::{second}::{offset:02d}"
                prompt = (
                    f"Form one lowercase identifier from prefix `{first}`, middle `{second}`, and "
                    f"two-digit suffix `{offset:02d}`. Separate the three fields with double "
                    "colons and return only that identifier."
                )
                skill = "double_colon_identifier"
                kind = "exact_text"
            elif family == 2:
                expected = json.dumps(
                    {"enabled": offset % 2 == 0, "label": first, "slot": offset},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                truth = "true" if offset % 2 == 0 else "false"
                prompt = (
                    "Output one JSON object and nothing else. It must contain exactly these "
                    f"fields: `enabled` set to {truth}, `label` set to the string `{first}`, and "
                    f"`slot` set to the integer {offset}."
                )
                skill = "three_field_json"
                kind = "json_exact"
            elif family == 3:
                expected = f"[{first}]-({second})"
                prompt = (
                    f"Place the literal token `{first}` inside square brackets and `{second}` "
                    "inside parentheses. Join those two groups with one hyphen and emit no other "
                    "text."
                )
                skill = "bracket_parenthesis_compound"
                kind = "exact_text"
            else:
                expected = f"{first} {second}\n{third} {offset:02d}"
                prompt = (
                    f"Write exactly two lines. The first line is `{first} {second}`. The second "
                    f"line is `{third} {offset:02d}`. Include neither labels nor bullets."
                )
                skill = "two_line_literal"
                kind = "exact_text"
            items.append(
                {
                    "id": f"rrfh-format-{item_number:03d}",
                    "section": "format",
                    "skill": skill,
                    "kind": kind,
                    "prompt": prompt,
                    "expected": expected,
                }
            )
    return items


def _instruction_items() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for family in range(5):
        for offset in range(30):
            item_number = family * 30 + offset
            tokens = [
                _WORDS_A[offset],
                _WORDS_B[(offset + 3) % 30],
                _WORDS_C[(offset + 7) % 30],
                _WORDS_A[(offset + 11) % 30],
                _WORDS_B[(offset + 17) % 30],
                _WORDS_C[(offset + 23) % 30],
            ]
            if family == 0:
                source = tokens[:5]
                expected = " ".join(reversed(source))
                prompt = (
                    f"Treat these five tokens as an ordered list: {', '.join(source)}. Emit the "
                    "same tokens in reverse order, separated by single spaces, with nothing else."
                )
                skill = "reverse_token_order"
            elif family == 1:
                source = tokens[:5]
                expected = " ".join(sorted(source))
                prompt = (
                    "Sort the following five lowercase tokens alphabetically: "
                    f"{', '.join(source)}. "
                    "Return only the sorted tokens separated by single spaces."
                )
                skill = "alphabetical_token_order"
            elif family == 2:
                source = tokens[:4]
                expected = "".join(token[0] for token in source)
                prompt = (
                    f"Read these labels from left to right: {', '.join(source)}. Concatenate their "
                    "first letters into one lowercase string and output only that string."
                )
                skill = "ordered_initial_letters"
            elif family == 3:
                source = tokens[:5]
                expected = " ".join(source[2:] + source[:2])
                prompt = (
                    f"For this token sequence, move its first two entries to the end without "
                    f"changing any other relative order: {', '.join(source)}. Return only the "
                    "resulting space-separated sequence."
                )
                skill = "left_rotation_by_two"
            else:
                source = tokens
                expected = " ".join(source[::2])
                prompt = (
                    "Number these six tokens from left to right starting at one: "
                    f"{', '.join(source)}. "
                    "Keep only the tokens in positions one, three, and five, preserving their "
                    "order. Output only those tokens separated by single spaces."
                )
                skill = "odd_position_selection"
            items.append(
                {
                    "id": f"rrfh-instruction-{item_number:03d}",
                    "section": "instruction",
                    "skill": skill,
                    "kind": "exact_text",
                    "prompt": prompt,
                    "expected": expected,
                }
            )
    return items


def _suite_payload(items: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "system_prompt": SYSTEM_PROMPT,
        "generation": dict(GENERATION),
        "items": items,
    }


def _suite_contract(suite: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "suite_id": suite["suite_id"],
        "system_prompt": suite["system_prompt"],
        "generation": suite["generation"],
        "items": suite["items"],
    }


def _audit_prior_prompts(
    items: Sequence[Mapping[str, str]], corpora: Sequence[PriorPromptCorpus]
) -> dict[str, Any]:
    corpus_ids = [corpus.corpus_id for corpus in corpora]
    if len(corpus_ids) != len(set(corpus_ids)) or any(not corpus_id for corpus_id in corpus_ids):
        raise ValueError("prior prompt corpus IDs must be non-empty and unique")
    prior_exact: set[str] = set()
    prior_windows: set[tuple[str, ...]] = set()
    corpus_evidence: list[dict[str, Any]] = []
    for corpus in corpora:
        if not corpus.prompts or any(not prompt.strip() for prompt in corpus.prompts):
            raise ValueError("each supplied prior prompt corpus must contain non-empty prompts")
        prior_exact.update(_normalized_text(prompt) for prompt in corpus.prompts)
        for prompt in corpus.prompts:
            prior_windows.update(_token_windows(prompt))
        corpus_evidence.append(
            {
                "corpus_id": corpus.corpus_id,
                "prompt_count": len(corpus.prompts),
                "corpus_sha256": canonical_sha256(list(corpus.prompts)),
            }
        )
    exact_collisions = 0
    window_collisions = 0
    for item in items:
        prompt = item["prompt"]
        exact_collisions += _normalized_text(prompt) in prior_exact
        window_collisions += bool(_token_windows(prompt) & prior_windows)
    if exact_collisions or window_collisions:
        raise ValueError(
            "holdout prompt audit found "
            f"{exact_collisions} exact and {window_collisions} twelve-token overlaps"
        )
    return {
        "audit_id": PROMPT_AUDIT_ID,
        "status": "passed" if corpora else "not_supplied",
        "normalizer": "lowercase_ascii_alphanumeric_tokens_v1",
        "window_size": 12,
        "corpora": corpus_evidence,
        "audited_prior_prompt_count": sum(len(corpus.prompts) for corpus in corpora),
        "exact_overlap_count": exact_collisions,
        "twelve_token_overlap_count": window_collisions,
    }


def _evidence_payload(suite: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    raw_items = cast(list[dict[str, str]], suite["items"])
    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "suite_id": SUITE_ID,
        "total": len(raw_items),
        "section_counts": dict(SECTION_COUNTS),
        "suite_sha256": canonical_sha256(_suite_contract(suite)),
        "answer_sha256": canonical_sha256(
            [{"id": item["id"], "expected": item["expected"]} for item in raw_items]
        ),
        "scorer_sha256": canonical_sha256(
            {
                "contract": _SCORER_CONTRACT,
                "assignments": [{"id": item["id"], "kind": item["kind"]} for item in raw_items],
            }
        ),
        "configuration_sha256": canonical_sha256(_BUILD_CONFIGURATION),
        "prompt_sha256": canonical_sha256(
            {"system": suite["system_prompt"], "prompts": [item["prompt"] for item in raw_items]}
        ),
        "generation_sha256": canonical_sha256(suite["generation"]),
        "item_id_sha256": canonical_sha256([item["id"] for item in raw_items]),
        "self_score_failures": 0,
        "ambiguous_reference_answers": 0,
        "prior_prompt_audit": dict(audit),
        "adapter_outputs_read": False,
        "benchmark_content_used": False,
        "synthetic_or_template_content_used": False,
        "sealed_final_accessed": False,
    }


def _validate_audit_evidence(audit: object) -> None:
    if not isinstance(audit, dict):
        raise ValueError("replay holdout prior-prompt audit must be an object")
    required_keys = {
        "audit_id",
        "status",
        "normalizer",
        "window_size",
        "corpora",
        "audited_prior_prompt_count",
        "exact_overlap_count",
        "twelve_token_overlap_count",
    }
    if set(audit) != required_keys:
        raise ValueError("replay holdout prior-prompt audit fields differ")
    raw_corpora = audit.get("corpora")
    if not isinstance(raw_corpora, list) or any(not isinstance(item, dict) for item in raw_corpora):
        raise ValueError("replay holdout prior-prompt corpus evidence differs")
    corpora = cast(list[dict[str, Any]], raw_corpora)
    corpus_ids = [str(item.get("corpus_id", "")) for item in corpora]
    if (
        audit.get("audit_id") != PROMPT_AUDIT_ID
        or audit.get("normalizer") != "lowercase_ascii_alphanumeric_tokens_v1"
        or audit.get("window_size") != 12
        or audit.get("exact_overlap_count") != 0
        or audit.get("twelve_token_overlap_count") != 0
        or len(corpus_ids) != len(set(corpus_ids))
        or any(not corpus_id for corpus_id in corpus_ids)
        or any(
            set(item) != {"corpus_id", "prompt_count", "corpus_sha256"}
            or not isinstance(item.get("prompt_count"), int)
            or int(item["prompt_count"]) <= 0
            or not isinstance(item.get("corpus_sha256"), str)
            or len(str(item["corpus_sha256"])) != 64
            for item in corpora
        )
        or audit.get("audited_prior_prompt_count")
        != sum(int(item["prompt_count"]) for item in corpora)
        or audit.get("status") != ("passed" if corpora else "not_supplied")
    ):
        raise ValueError("replay holdout prior-prompt audit evidence differs")


def validate_replay_holdout_artifacts(
    suite: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    """Strictly validate a built suite and all of its frozen content-free hashes."""

    if set(suite) != {"schema_version", "suite_id", "system_prompt", "generation", "items"}:
        raise ValueError("replay holdout suite fields differ")
    if suite.get("schema_version") != 1 or suite.get("suite_id") != SUITE_ID:
        raise ValueError("replay holdout suite identity differs")
    raw_items = suite.get("items")
    if not isinstance(raw_items, list) or any(not isinstance(item, dict) for item in raw_items):
        raise ValueError("replay holdout items must be objects")
    items = cast(list[dict[str, Any]], raw_items)
    if len(items) != 450:
        raise ValueError("replay holdout requires exactly 450 items")
    required_item_keys = {"id", "section", "skill", "kind", "prompt", "expected"}
    if any(set(item) != required_item_keys for item in items):
        raise ValueError("replay holdout item fields differ")
    counts = Counter(str(item["section"]) for item in items)
    if counts != SECTION_COUNTS:
        raise ValueError("replay holdout section counts differ")
    ids = [str(item["id"]) for item in items]
    prompts = [str(item["prompt"]) for item in items]
    if len(set(ids)) != 450:
        raise ValueError("replay holdout IDs must be unique")
    if len({_normalized_text(prompt) for prompt in prompts}) != 450:
        raise ValueError("replay holdout normalized prompts must be unique")
    self_score_failures = 0
    for raw in items:
        item = RetentionItem(
            item_id=str(raw["id"]),
            section=cast(Any, raw["section"]),
            skill=str(raw["skill"]),
            kind=cast(Any, raw["kind"]),
            prompt=str(raw["prompt"]),
            expected=str(raw["expected"]),
        )
        self_score_failures += not bool(score_response(item, item.expected)["correct"])
    if self_score_failures:
        raise ValueError("replay holdout contains a reference that does not self-score")

    _validate_audit_evidence(evidence.get("prior_prompt_audit"))
    expected_evidence = _evidence_payload(
        suite,
        cast(Mapping[str, Any], evidence.get("prior_prompt_audit", {})),
    )
    if evidence.get("self_score_failures") != 0:
        raise ValueError("replay holdout self-score evidence differs")
    evidence_without_hash = {
        key: value for key, value in evidence.items() if key != "summary_sha256"
    }
    if evidence_without_hash != expected_evidence:
        raise ValueError("replay holdout evidence or component hash differs")
    if evidence.get("summary_sha256") != canonical_sha256(expected_evidence):
        raise ValueError("replay holdout evidence summary hash differs")


def build_replay_final_holdout(
    *, prior_prompt_corpora: Sequence[PriorPromptCorpus] = ()
) -> ReplayHoldoutArtifacts:
    """Build all 450 prompts and reject any supplied prior-corpus collision."""

    items = _arithmetic_items() + _format_items() + _instruction_items()
    suite = _suite_payload(items)
    audit = _audit_prior_prompts(items, prior_prompt_corpora)
    evidence = _evidence_payload(suite, audit)
    evidence["summary_sha256"] = canonical_sha256(evidence)
    validate_replay_holdout_artifacts(suite, evidence)
    return ReplayHoldoutArtifacts(suite=suite, evidence=evidence)


def _repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _relative_to_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("freeze outputs must remain inside the repository") from error


def _git_ignored(path: Path, root: Path) -> bool:
    relative = _relative_to_root(path, root)
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", "--", str(relative)],
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError("git check-ignore failed while validating freeze outputs")
    return result.returncode == 0


def _validate_output_boundaries(suite_output: Path, evidence_output: Path) -> None:
    root = _repository_root()
    if suite_output.resolve() == evidence_output.resolve():
        raise ValueError("suite and evidence outputs must be different files")
    if not _git_ignored(suite_output, root):
        raise ValueError("content-bearing suite output must be ignored by Git")
    if _git_ignored(evidence_output, root):
        raise ValueError("content-free evidence output must be trackable by Git")


def write_replay_holdout_artifacts(
    artifacts: ReplayHoldoutArtifacts,
    *,
    suite_output: Path,
    evidence_output: Path,
    enforce_git_boundaries: bool = True,
) -> None:
    """Atomically write a content-bearing suite and content-free integrity evidence."""

    validate_replay_holdout_artifacts(artifacts.suite, artifacts.evidence)
    if enforce_git_boundaries:
        _validate_output_boundaries(suite_output, evidence_output)
    if suite_output.exists() or evidence_output.exists():
        raise FileExistsError("replay holdout freeze refuses to overwrite an existing artifact")
    suite_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    suite_temporary = suite_output.with_name(f".{suite_output.name}.tmp")
    evidence_temporary = evidence_output.with_name(f".{evidence_output.name}.tmp")
    if suite_temporary.exists() or evidence_temporary.exists():
        raise FileExistsError("replay holdout freeze temporary artifact already exists")
    suite_rendered = json.dumps(artifacts.suite, indent=2, sort_keys=True) + "\n"
    evidence_rendered = json.dumps(artifacts.evidence, indent=2, sort_keys=True) + "\n"
    if any(item["prompt"] in evidence_rendered for item in artifacts.suite["items"]):
        raise ValueError("content-free evidence unexpectedly contains holdout prompt text")
    try:
        suite_temporary.write_text(suite_rendered, encoding="utf-8")
        evidence_temporary.write_text(evidence_rendered, encoding="utf-8")
        os.replace(suite_temporary, suite_output)
        os.replace(evidence_temporary, evidence_output)
        written_suite: object = json.loads(suite_output.read_text(encoding="utf-8"))
        written_evidence: object = json.loads(evidence_output.read_text(encoding="utf-8"))
        if not isinstance(written_suite, dict) or not isinstance(written_evidence, dict):
            raise ValueError("written replay holdout artifacts have invalid roots")
        validate_replay_holdout_artifacts(written_suite, written_evidence)
    except Exception:
        for path in (suite_temporary, evidence_temporary, suite_output, evidence_output):
            if path.exists():
                path.unlink()
        raise


def freeze_replay_final_holdout_from_specs(
    specs: Sequence[PriorCorpusSpec],
    *,
    suite_output: Path,
    evidence_output: Path,
    enforce_git_boundaries: bool = True,
) -> ReplayHoldoutArtifacts:
    """Load the complete production audit inventory, freeze, validate, and write once."""

    corpora = tuple(load_prior_prompt_corpus(spec) for spec in specs)
    validate_production_corpus_inventory(specs, corpora)
    artifacts = build_replay_final_holdout(prior_prompt_corpora=corpora)
    write_replay_holdout_artifacts(
        artifacts,
        suite_output=suite_output,
        evidence_output=evidence_output,
        enforce_git_boundaries=enforce_git_boundaries,
    )
    return artifacts


def main() -> None:
    """Freeze the replay final holdout from explicit local corpus specifications."""

    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--prior-corpus",
        action="append",
        help=(
            "JSON object with corpus_id, corpus_type, path, and field; development Arrow "
            "sources also require source_path and config_path"
        ),
    )
    inputs.add_argument(
        "--prior-corpus-file",
        type=Path,
        help="JSON array containing the same strict prior-corpus specification objects",
    )
    parser.add_argument("--suite-output", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    args = parser.parse_args()
    specs = (
        load_prior_corpus_specs_file(args.prior_corpus_file)
        if args.prior_corpus_file is not None
        else tuple(parse_prior_corpus_spec(value) for value in args.prior_corpus)
    )
    artifacts = freeze_replay_final_holdout_from_specs(
        specs,
        suite_output=args.suite_output,
        evidence_output=args.evidence_output,
    )
    print(
        json.dumps(
            {
                "suite_id": artifacts.suite["suite_id"],
                "total": artifacts.evidence["total"],
                "suite_sha256": artifacts.evidence["suite_sha256"],
                "evidence_sha256": artifacts.evidence["summary_sha256"],
                "prior_corpora": len(
                    cast(dict[str, Any], artifacts.evidence["prior_prompt_audit"])["corpora"]
                ),
                "audited_prior_prompts": cast(
                    dict[str, Any], artifacts.evidence["prior_prompt_audit"]
                )["audited_prior_prompt_count"],
                "suite_output": str(args.suite_output),
                "evidence_output": str(args.evidence_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
