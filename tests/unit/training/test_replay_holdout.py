import copy
import json
import types
from collections import Counter
from pathlib import Path

import pytest
import yaml

from foundry.config import load_config
from foundry.evaluation.manifests import build_manifests, save_manifest
from foundry.training import replay_holdout
from foundry.training.base_conditioned_retention import assess_holdout_instrument_usability
from foundry.training.config import canonical_sha256
from foundry.training.replay_holdout import (
    PriorCorpusSpec,
    PriorPromptCorpus,
    build_replay_final_holdout,
    load_prior_corpus_specs_file,
    load_prior_prompt_corpus,
    parse_prior_corpus_spec,
    validate_production_corpus_inventory,
    validate_replay_holdout_artifacts,
    write_replay_holdout_artifacts,
)
from foundry.training.retention import load_suite, score_response


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_builder_has_exact_layout_unique_ids_and_prompts() -> None:
    artifacts = build_replay_final_holdout()
    items = artifacts.suite["items"]
    assert len(items) == 450
    assert Counter(item["section"] for item in items) == {
        "arithmetic": 150,
        "format": 150,
        "instruction": 150,
    }
    assert len({item["id"] for item in items}) == 450
    assert len({" ".join(item["prompt"].lower().split()) for item in items}) == 450
    assert artifacts.suite["suite_id"] == "foundry-retention-replay-final-holdout-v1"


def test_every_reference_self_scores_with_an_objective_scorer(tmp_path: Path) -> None:
    artifacts = build_replay_final_holdout()
    suite_path = tmp_path / "suite.json"
    _write_json(suite_path, artifacts.suite)
    suite = load_suite(suite_path)
    assert all(score_response(item, item.expected)["correct"] for item in suite.items)
    assert suite.suite_sha256 == artifacts.evidence["suite_sha256"]
    assert suite.prompt_sha256 == artifacts.evidence["prompt_sha256"]
    assert suite.generation_sha256 == artifacts.evidence["generation_sha256"]
    assert artifacts.evidence["self_score_failures"] == 0
    assert artifacts.evidence["ambiguous_reference_answers"] == 0
    assert all(
        len(artifacts.evidence[key]) == 64
        for key in (
            "suite_sha256",
            "answer_sha256",
            "scorer_sha256",
            "configuration_sha256",
            "summary_sha256",
        )
    )


def test_builder_is_exactly_deterministic() -> None:
    first = build_replay_final_holdout()
    second = build_replay_final_holdout()
    assert first == second
    assert canonical_sha256(first.suite) == canonical_sha256(second.suite)
    assert first.evidence["summary_sha256"] == second.evidence["summary_sha256"]


def test_prior_prompt_audit_accepts_disjoint_corpora_and_freezes_hashes() -> None:
    corpus = PriorPromptCorpus(
        corpus_id="original-prior-fixture-v1",
        prompts=(
            "Arrange seven wooden markers beside a quiet pond, then stop.",
            "State a color name after the bell rings once.",
        ),
    )
    artifacts = build_replay_final_holdout(prior_prompt_corpora=(corpus,))
    audit = artifacts.evidence["prior_prompt_audit"]
    assert audit["status"] == "passed"
    assert audit["audited_prior_prompt_count"] == 2
    assert audit["exact_overlap_count"] == 0
    assert audit["twelve_token_overlap_count"] == 0
    assert len(audit["corpora"][0]["corpus_sha256"]) == 64


def test_prior_prompt_audit_rejects_exact_and_twelve_token_overlap() -> None:
    prompt = build_replay_final_holdout().suite["items"][0]["prompt"]
    exact = PriorPromptCorpus(corpus_id="exact", prompts=(prompt,))
    with pytest.raises(ValueError, match="exact"):
        build_replay_final_holdout(prior_prompt_corpora=(exact,))

    overlapping = "unrelated prefix " + " ".join(prompt.split()[:12]) + " unrelated suffix"
    ngram = PriorPromptCorpus(corpus_id="ngram", prompts=(overlapping,))
    with pytest.raises(ValueError, match="twelve-token"):
        build_replay_final_holdout(prior_prompt_corpora=(ngram,))


def test_tampering_fails_strict_artifact_validation() -> None:
    artifacts = build_replay_final_holdout()
    tampered_suite = copy.deepcopy(artifacts.suite)
    tampered_suite["items"][0]["expected"] = "999999"
    with pytest.raises(ValueError, match="evidence|hash"):
        validate_replay_holdout_artifacts(tampered_suite, artifacts.evidence)

    tampered_evidence = copy.deepcopy(artifacts.evidence)
    tampered_evidence["answer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evidence|hash"):
        validate_replay_holdout_artifacts(artifacts.suite, tampered_evidence)


def test_new_holdout_routes_through_predeclared_usability_gate(tmp_path: Path) -> None:
    artifacts = build_replay_final_holdout(
        prior_prompt_corpora=(
            PriorPromptCorpus(
                corpus_id="complete-prefreeze-audit-fixture",
                prompts=(
                    "Copy the silver emblem after observing four quiet signals from the tower.",
                ),
            ),
        )
    )
    suite_path = tmp_path / "suite.json"
    evidence_path = tmp_path / "evidence.json"
    _write_json(suite_path, artifacts.suite)
    _write_json(evidence_path, artifacts.evidence)
    suite = load_suite(suite_path)
    summary: dict[str, object] = {
        "adapter_sha256": None,
        "suite_sha256": suite.suite_sha256,
        "total": 450,
        "section_metrics": {
            "arithmetic": {"correct": 60},
            "format": {"correct": 60},
            "instruction": {"correct": 130},
        },
        "extractable": 420,
        "prompt_echo": 0,
        "question_generation": 0,
        "malformed_outputs": 0,
        "backend_failures": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, summary)
    result = assess_holdout_instrument_usability(
        suite_path=suite_path,
        base_summary_path=summary_path,
        artifact_evidence_path=evidence_path,
    )
    assert result["gate_id"] == (
        "foundry-retention-replay-final-holdout-instrument-usability-gate-v1"
    )
    assert result["overall_correct"] == 250
    assert result["gate_passed"] is True


def test_usability_route_rejects_tampered_artifact_evidence(tmp_path: Path) -> None:
    artifacts = build_replay_final_holdout(
        prior_prompt_corpora=(
            PriorPromptCorpus(
                corpus_id="complete-prefreeze-audit-fixture",
                prompts=(
                    "Copy the silver emblem after observing four quiet signals from the tower.",
                ),
            ),
        )
    )
    suite_path = tmp_path / "suite.json"
    _write_json(suite_path, artifacts.suite)
    suite = load_suite(suite_path)
    summary: dict[str, object] = {
        "adapter_sha256": None,
        "suite_sha256": suite.suite_sha256,
        "total": 450,
        "section_metrics": {
            "arithmetic": {"correct": 100},
            "format": {"correct": 100},
            "instruction": {"correct": 100},
        },
        "backend_failures": 0,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, summary)
    tampered_evidence = copy.deepcopy(artifacts.evidence)
    tampered_evidence["configuration_sha256"] = "f" * 64
    tampered_evidence["summary_sha256"] = canonical_sha256(
        {key: value for key, value in tampered_evidence.items() if key != "summary_sha256"}
    )
    evidence_path = tmp_path / "evidence.json"
    _write_json(evidence_path, tampered_evidence)
    with pytest.raises(ValueError, match="evidence|component hash"):
        assess_holdout_instrument_usability(
            suite_path=suite_path,
            base_summary_path=summary_path,
            artifact_evidence_path=evidence_path,
        )


def test_explicit_specs_load_retention_json_and_jsonl(tmp_path: Path) -> None:
    suite_path = tmp_path / "prior-suite.json"
    _write_json(suite_path, {"items": [{"prompt": "Prior suite prompt."}]})
    suite_spec = parse_prior_corpus_spec(
        json.dumps(
            {
                "corpus_id": "prior-suite",
                "corpus_type": "retention_suite_json",
                "path": str(suite_path),
                "field": "items.prompt",
            }
        )
    )
    assert load_prior_prompt_corpus(suite_spec).prompts == ("Prior suite prompt.",)

    jsonl_path = tmp_path / "synthetic.jsonl"
    jsonl_path.write_text(
        json.dumps({"rendered_question": "First synthetic question."})
        + "\n"
        + json.dumps({"rendered_question": "Second synthetic question."})
        + "\n",
        encoding="utf-8",
    )
    jsonl_spec = parse_prior_corpus_spec(
        json.dumps(
            {
                "corpus_id": "synthetic-split",
                "corpus_type": "records_jsonl",
                "path": str(jsonl_path),
                "field": "rendered_question",
            }
        )
    )
    assert load_prior_prompt_corpus(jsonl_spec).prompts == (
        "First synthetic question.",
        "Second synthetic question.",
    )


def test_spec_parser_rejects_ambiguous_fields() -> None:
    with pytest.raises(ValueError, match="items.prompt"):
        parse_prior_corpus_spec(
            json.dumps(
                {
                    "corpus_id": "prior-suite",
                    "corpus_type": "retention_suite_json",
                    "path": "prior.json",
                    "field": "prompt",
                }
            )
        )
    with pytest.raises(ValueError, match="fields differ"):
        parse_prior_corpus_spec(
            json.dumps(
                {
                    "corpus_id": "records",
                    "corpus_type": "records_jsonl",
                    "path": "records.jsonl",
                    "field": "question",
                    "guessed_field": "prompt",
                }
            )
        )


def test_specification_file_is_windows_shell_safe_and_strict(tmp_path: Path) -> None:
    specs_path = tmp_path / "corpora.json"
    _write_json(
        specs_path,
        [
            {
                "corpus_id": "prior-suite",
                "corpus_type": "retention_suite_json",
                "path": r"C:\local cache\prior suite.json",
                "field": "items.prompt",
            },
            {
                "corpus_id": "synthetic",
                "corpus_type": "records_jsonl",
                "path": r"C:\local cache\synthetic.jsonl",
                "field": "rendered_question",
            },
        ],
    )
    specs = load_prior_corpus_specs_file(specs_path)
    assert [spec.corpus_id for spec in specs] == ["prior-suite", "synthetic"]
    assert str(specs[0].path) == r"C:\local cache\prior suite.json"
    empty_path = tmp_path / "empty.json"
    _write_json(empty_path, [])
    with pytest.raises(ValueError, match="non-empty array"):
        load_prior_corpus_specs_file(empty_path)


def _evaluation_config_payload(expected_examples: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": {
            "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "9" * 40,
            "dtype": "float16",
            "device": "cuda",
        },
        "dataset": {
            "repo_id": "ScaleAI/gsm1k",
            "revision": "b" * 40,
            "config_name": "default",
            "source_split": "test",
            "expected_examples": expected_examples,
        },
        "partition": {
            "seed": "original-development-fixture",
            "sealed_final_size": 1,
            "development_manifest": "development.json",
            "sealed_final_manifest": "sealed.json",
        },
        "prompt": {
            "system": "Follow the instruction.",
            "user_template": "{question}\nFinal answer: <integer>",
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 32,
        },
    }


def test_canonical_development_loader_reads_only_manifest_selected_question_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump(_evaluation_config_payload(4), sort_keys=False), encoding="utf-8"
    )
    config = load_config(config_path)
    development, sealed = build_manifests(config)
    development_path = tmp_path / "development.json"
    sealed_path = tmp_path / "sealed-final.json"
    save_manifest(development, development_path)
    save_manifest(sealed, sealed_path)
    rows = [
        {"question": f"Development question {index}.", "answer": "not requested"}
        for index in range(4)
    ]

    class FakeDataset:
        @classmethod
        def from_file(cls, path: str) -> "FakeDataset":
            assert path.endswith("source.arrow")
            return cls()

        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, key: str) -> list[str]:
            assert key == "question"
            return [str(row[key]) for row in rows]

    fake_datasets = types.SimpleNamespace(Dataset=FakeDataset)
    monkeypatch.setattr(replay_holdout.importlib, "import_module", lambda name: fake_datasets)
    spec = PriorCorpusSpec(
        corpus_id="development",
        corpus_type="development_manifest_arrow",
        path=development_path,
        field="question",
        source_path=tmp_path / "source.arrow",
        config_path=config_path,
    )
    corpus = load_prior_prompt_corpus(spec)
    assert len(corpus.prompts) == 3
    assert corpus.prompts == tuple(
        rows[entry.row_index]["question"] for entry in development.entries
    )

    sealed_spec = PriorCorpusSpec(
        corpus_id="forbidden",
        corpus_type="development_manifest_arrow",
        path=sealed_path,
        field="question",
        source_path=tmp_path / "source.arrow",
        config_path=config_path,
    )
    with pytest.raises(ValueError, match="sealed-final"):
        load_prior_prompt_corpus(sealed_spec)


def test_production_inventory_requires_seven_plus_four_plus_one_and_3314() -> None:
    specs: list[PriorCorpusSpec] = []
    corpora: list[PriorPromptCorpus] = []
    inventory = (
        [("retention_suite_json", count) for count in (60, 90, 90, 120, 300, 300, 450)]
        + [("records_jsonl", count) for count in (50, 50, 450, 450)]
        + [("development_manifest_arrow", 904)]
    )
    for index, (corpus_type, count) in enumerate(inventory):
        corpus_id = f"corpus-{index:02d}"
        specs.append(
            PriorCorpusSpec(
                corpus_id=corpus_id,
                corpus_type=corpus_type,
                path=Path(f"source-{index}.json"),
                field="question",
            )
        )
        corpora.append(
            PriorPromptCorpus(
                corpus_id=corpus_id,
                prompts=tuple(f"prior-{index}-{item}" for item in range(count)),
            )
        )
    validate_production_corpus_inventory(specs, corpora)
    assert sum(len(corpus.prompts) for corpus in corpora) == 3314
    corpora[-1] = PriorPromptCorpus(corpus_id=corpora[-1].corpus_id, prompts=("short",))
    with pytest.raises(ValueError, match="development-manifest"):
        validate_production_corpus_inventory(specs, corpora)


def test_writer_separates_content_bearing_suite_and_content_free_evidence(
    tmp_path: Path,
) -> None:
    artifacts = build_replay_final_holdout(
        prior_prompt_corpora=(
            PriorPromptCorpus(
                corpus_id="prior-fixture",
                prompts=("A distinct prior prompt used solely for overlap auditing.",),
            ),
        )
    )
    suite_path = tmp_path / "raw" / "suite.json"
    evidence_path = tmp_path / "tracked" / "evidence.json"
    write_replay_holdout_artifacts(
        artifacts,
        suite_output=suite_path,
        evidence_output=evidence_path,
        enforce_git_boundaries=False,
    )
    assert suite_path.exists()
    assert evidence_path.exists()
    evidence_text = evidence_path.read_text(encoding="utf-8")
    assert artifacts.suite["items"][0]["prompt"] not in evidence_text
    validate_replay_holdout_artifacts(
        json.loads(suite_path.read_text(encoding="utf-8")),
        json.loads(evidence_text),
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        write_replay_holdout_artifacts(
            artifacts,
            suite_output=suite_path,
            evidence_output=evidence_path,
            enforce_git_boundaries=False,
        )
