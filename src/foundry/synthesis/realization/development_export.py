"""One-way development-question export for the isolated contamination scanner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from foundry.synthesis.contamination import (
    DevelopmentQuestion,
    load_development_questions_for_contamination,
)


def write_development_question_export(
    *,
    output_path: Path,
    evaluation_config_path: Path,
    development_manifest_path: Path,
) -> str:
    """Write question-only development records to an ignored scanner artifact."""

    questions = load_development_questions_for_contamination(
        evaluation_config_path=evaluation_config_path,
        development_manifest_path=development_manifest_path,
    )
    lines = [
        json.dumps(
            {
                "stable_id": question.stable_id,
                "row_index": question.row_index,
                "question": question.question,
            },
            sort_keys=True,
        )
        for question in questions
    ]
    payload = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_development_question_export(path: Path) -> tuple[DevelopmentQuestion, ...]:
    """Load exactly 904 question-only rows without importing the dataset package."""

    questions: list[DevelopmentQuestion] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw: object = json.loads(line)
        if not isinstance(raw, dict) or set(raw) != {"stable_id", "row_index", "question"}:
            raise ValueError(f"development export line {line_number} has an invalid schema")
        stable_id = raw["stable_id"]
        row_index = raw["row_index"]
        question = raw["question"]
        if (
            not isinstance(stable_id, str)
            or isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(f"development export line {line_number} has invalid values")
        identity = (
            f"ScaleAI/gsm1k@bc09569d09a614b9b530edc7f076fb214ac10493:default:test:{row_index}"
        )
        if hashlib.sha256(identity.encode("utf-8")).hexdigest() != stable_id:
            raise ValueError(f"development export line {line_number} has an invalid identifier")
        questions.append(DevelopmentQuestion(stable_id, row_index, question))
    if len(questions) != 904:
        raise ValueError("development export must contain exactly 904 questions")
    if len({item.stable_id for item in questions}) != 904:
        raise ValueError("development export stable IDs must be unique")
    return tuple(questions)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    args = parser.parse_args()
    digest = write_development_question_export(
        output_path=args.output,
        evaluation_config_path=args.evaluation_config,
        development_manifest_path=args.development_manifest,
    )
    print(json.dumps({"questions": 904, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
