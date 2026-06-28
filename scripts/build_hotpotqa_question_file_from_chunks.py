from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    args = parse_args()

    chunks_path = resolve_project_path(args.chunks_path)
    output_path = resolve_project_path(args.output_path)

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            f"Project root resolved as: {PROJECT_ROOT}"
        )

    question_records = build_question_records_from_chunks(chunks_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as writer:
        for record in question_records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input chunks file: {chunks_path}")
    print(f"Created question file: {output_path}")
    print(f"Number of question records: {len(question_records)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HotPotQA question-level JSONL file from paragraph chunks."
    )

    parser.add_argument(
        "--chunks-path",
        default="data/processed/hotpotqa_paragraph_chunks.jsonl",
        help="Path to the processed HotPotQA paragraph chunk JSONL file.",
    )

    parser.add_argument(
        "--output-path",
        default="data/processed/hotpotqa_questions.jsonl",
        help="Output path for the generated question-level JSONL file.",
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def build_question_records_from_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()

    with chunks_path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue

            chunk = json.loads(line)

            question_id = get_first_present(
                chunk,
                "question_id",
                "source_question_id",
                "qid",
                "_id",
            )

            if question_id is None:
                raise ValueError(
                    f"Missing question id in chunk file at line {line_number}."
                )

            question_id = str(question_id)

            if question_id not in grouped:
                question = get_first_present(chunk, "question", "query")
                answer = get_first_present(chunk, "gold_answer", "answer")
                question_type = get_first_present(chunk, "question_type", "type")

                record = {
                    "question_id": question_id,
                    "question": question,
                    "gold_answer": answer,
                    "gold_supporting_titles": [],
                }

                if question_type is not None:
                    record["question_type"] = question_type

                grouped[question_id] = record

            if is_supporting_chunk(chunk):
                title = get_first_present(
                    chunk,
                    "doc_title",
                    "title",
                    "document_title",
                )

                if title is not None:
                    append_unique(
                        grouped[question_id]["gold_supporting_titles"],
                        str(title),
                    )

    records = list(grouped.values())
    validate_records(records)

    return records


def get_first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)

        if value is not None:
            return value

    return None


def is_supporting_chunk(chunk: dict[str, Any]) -> bool:
    if chunk.get("is_supporting_doc") is True:
        return True

    supporting_sentence_ids = chunk.get("supporting_sentence_ids")

    if isinstance(supporting_sentence_ids, list) and len(supporting_sentence_ids) > 0:
        return True

    return False


def append_unique(values: list[str], value: str) -> None:
    normalized_value = normalize_text(value)
    existing = {normalize_text(item) for item in values}

    if normalized_value and normalized_value not in existing:
        values.append(value)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No question records were created.")

    missing_question = [
        record["question_id"]
        for record in records
        if not record.get("question")
    ]

    missing_answer = [
        record["question_id"]
        for record in records
        if not record.get("gold_answer")
    ]

    if missing_question:
        raise ValueError(
            "Question text is missing from the chunk file for some records. "
            f"Example question_id: {missing_question[0]}"
        )

    if missing_answer:
        raise ValueError(
            "Gold answer is missing from the chunk file for some records. "
            f"Example question_id: {missing_answer[0]}"
        )


if __name__ == "__main__":
    main()