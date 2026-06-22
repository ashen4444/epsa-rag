from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Iterable

import yaml
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"

OUTPUT_WIDTH = 110
MAX_PARAGRAPH_CHARS: int | None = None
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inspections"

def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return config


def _project_path(relative_path: str) -> Path:
    path = Path(relative_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed corpus file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean_line = line.strip()

            if not clean_line:
                continue

            try:
                record = json.loads(clean_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} in {path}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in processed corpus: {path}")

    return records


def _get_first_available(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)

        if value is not None:
            return value

    return None


def _as_clean_string(value: Any, default: str = "") -> str:
    if value is None:
        return default

    return str(value).strip()


def _get_question_id(chunk: dict[str, Any]) -> str:
    question_id = _get_first_available(
        chunk,
        [
            "question_id",
            "source_question_id",
            "hotpotqa_id",
            "_id",
            "id",
        ],
    )

    question_id_text = _as_clean_string(question_id)

    if not question_id_text:
        raise ValueError(
            "Could not find question id in chunk. Expected one of: "
            "question_id, source_question_id, hotpotqa_id, _id, id."
        )

    return question_id_text


def _get_chunk_id(chunk: dict[str, Any]) -> str:
    chunk_id = _as_clean_string(chunk.get("chunk_id"))

    if not chunk_id:
        raise ValueError("Chunk record is missing non-empty chunk_id.")

    return chunk_id


def _get_doc_title(chunk: dict[str, Any]) -> str:
    title = _get_first_available(
        chunk,
        [
            "doc_title",
            "title",
            "document_title",
        ],
    )

    return _as_clean_string(title, default="UNKNOWN_TITLE")


def _get_paragraph_index(chunk: dict[str, Any]) -> int:
    value = chunk.get("paragraph_index", 0)

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_question_text(question_chunks: list[dict[str, Any]]) -> str:
    for chunk in question_chunks:
        question = _get_first_available(
            chunk,
            [
                "question",
                "query",
                "source_question",
                "question_text",
            ],
        )

        question_text = _as_clean_string(question)

        if question_text:
            return question_text

    return "QUESTION_TEXT_NOT_FOUND_IN_PROCESSED_CORPUS"


def _get_gold_answer(question_chunks: list[dict[str, Any]]) -> str:
    for chunk in question_chunks:
        answer = _get_first_available(
            chunk,
            [
                "answer",
                "gold_answer",
                "final_answer",
            ],
        )

        answer_text = _as_clean_string(answer)

        if answer_text:
            return answer_text

    return "ANSWER_NOT_FOUND_IN_PROCESSED_CORPUS"


def _get_supporting_sentence_ids(chunk: dict[str, Any]) -> list[int]:
    value = _get_first_available(
        chunk,
        [
            "supporting_sentence_ids",
            "supporting_sentences",
            "gold_supporting_sentence_ids",
        ],
    )

    if value is None or not isinstance(value, list):
        return []

    sentence_ids: list[int] = []

    for item in value:
        try:
            sentence_ids.append(int(item))
        except (TypeError, ValueError):
            continue

    return sentence_ids


def _is_supporting_doc(chunk: dict[str, Any]) -> bool:
    explicit_flag = chunk.get("is_supporting_doc")

    if isinstance(explicit_flag, bool):
        return explicit_flag

    return len(_get_supporting_sentence_ids(chunk)) > 0


def _get_sentences(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = chunk.get("sentences")

    if not isinstance(sentences, list):
        return []

    normalized: list[dict[str, Any]] = []

    for index, sentence in enumerate(sentences):
        if isinstance(sentence, dict):
            normalized.append(sentence)
        elif isinstance(sentence, str):
            normalized.append(
                {
                    "sentence_id": index,
                    "text": sentence,
                }
            )

    return normalized


def _get_sentence_text(sentence: dict[str, Any]) -> str:
    return _as_clean_string(sentence.get("text"))


def _get_sentence_id(sentence: dict[str, Any], fallback: int) -> int:
    value = sentence.get("sentence_id", fallback)

    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _get_paragraph_text(chunk: dict[str, Any]) -> str:
    text = _get_first_available(
        chunk,
        [
            "paragraph_text",
            "chunk_text",
            "text",
            "content",
        ],
    )

    clean_text = " ".join(_as_clean_string(text).split())

    if MAX_PARAGRAPH_CHARS is not None and len(clean_text) > MAX_PARAGRAPH_CHARS:
        return clean_text[: MAX_PARAGRAPH_CHARS - 3] + "..."

    return clean_text


def _print_wrapped_text(
    label: str,
    text: str,
    *,
    label_indent: str = "",
    text_indent: str = "  ",
    width: int | None = None,
) -> None:
    terminal_width = shutil.get_terminal_size((OUTPUT_WIDTH, 20)).columns
    effective_width = width or min(max(terminal_width, 80), 140)

    print(f"{label_indent}{label}:")

    if not text:
        print(text_indent)
        return

    wrapped_text = textwrap.fill(
        text,
        width=effective_width,
        initial_indent=text_indent,
        subsequent_indent=text_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )

    print(wrapped_text)


def _group_by_question_id(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for chunk in chunks:
        question_id = _get_question_id(chunk)
        grouped.setdefault(question_id, []).append(chunk)

    return grouped


def _print_available_question_id_examples(
    grouped_chunks: dict[str, list[dict[str, Any]]],
    limit: int = 10,
) -> None:
    print()
    print("Available question ID examples:")

    for index, question_id in enumerate(sorted(grouped_chunks.keys())[:limit], start=1):
        question = _get_question_text(grouped_chunks[question_id])
        print(f"  {index}. {question_id} | {question}")


def _get_question_id_from_user(
    grouped_chunks: dict[str, list[dict[str, Any]]],
) -> str:
    print("HotPotQA Question Context Inspector")
    print("=" * 100)
    print(f"Questions available in processed corpus: {len(grouped_chunks)}")
    _print_available_question_id_examples(grouped_chunks)
    print()

    question_id = input("Enter HotPotQA question ID to inspect: ").strip()

    if not question_id:
        raise ValueError("Question ID cannot be empty.")

    if question_id not in grouped_chunks:
        raise ValueError(
            f"Question ID not found in processed corpus: {question_id}"
        )

    return question_id


def _print_gold_supporting_summary(question_chunks: list[dict[str, Any]]) -> None:
    supporting_chunks = [
        chunk for chunk in question_chunks if _is_supporting_doc(chunk)
    ]

    print()
    print("Gold supporting facts summary:")

    if not supporting_chunks:
        print("  No supporting facts found in processed corpus for this question.")
        return

    for index, chunk in enumerate(
        sorted(
            supporting_chunks,
            key=lambda item: (_get_doc_title(item), _get_paragraph_index(item)),
        ),
        start=1,
    ):
        print(f"  {index}. Title: {_get_doc_title(chunk)}")
        print(f"     Chunk ID: {_get_chunk_id(chunk)}")
        print(f"     Supporting sentence IDs: {_get_supporting_sentence_ids(chunk)}")


def _print_context_document(
    index: int,
    total_count: int,
    chunk: dict[str, Any],
) -> None:
    title = _get_doc_title(chunk)
    chunk_id = _get_chunk_id(chunk)
    paragraph_index = _get_paragraph_index(chunk)
    supporting_sentence_ids = set(_get_supporting_sentence_ids(chunk))
    support_label = "GOLD SUPPORTING DOCUMENT" if _is_supporting_doc(chunk) else "distractor/context document"

    print()
    print("-" * 100)
    print(f"Context document {index}/{total_count}")
    print("-" * 100)
    print(f"Title: {title}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Paragraph index: {paragraph_index}")
    print(f"Label: {support_label}")

    if supporting_sentence_ids:
        print(f"Supporting sentence IDs: {sorted(supporting_sentence_ids)}")
    else:
        print("Supporting sentence IDs: []")

    sentences = _get_sentences(chunk)

    if sentences:
        print()
        print("Sentences:")

        for sentence_index, sentence in enumerate(sentences):
            sentence_id = _get_sentence_id(sentence, sentence_index)
            sentence_text = _get_sentence_text(sentence)
            marker = " <-- GOLD SUPPORTING SENTENCE" if sentence_id in supporting_sentence_ids else ""

            _print_wrapped_text(
                label=f"s{sentence_id}{marker}",
                text=sentence_text,
                label_indent="  ",
                text_indent="    ",
            )
    else:
        print()
        print("Sentences: NOT_FOUND_IN_PROCESSED_CHUNK")

    print()
    _print_wrapped_text(
        label="Full paragraph",
        text=_get_paragraph_text(chunk),
        label_indent="",
        text_indent="  ",
    )

def _safe_filename(value: str) -> str:
        safe_chars = []

        for char in value:
            if char.isalnum() or char in {"-", "_"}:
                safe_chars.append(char)
            else:
                safe_chars.append("_")

        return "".join(safe_chars).strip("_")

def _write_question_markdown_report(
            question_id: str,
            question_chunks: list[dict[str, Any]],
            output_path: Path,
    ) -> None:
        question = _get_question_text(question_chunks)
        answer = _get_gold_answer(question_chunks)

        sorted_chunks = sorted(
            question_chunks,
            key=lambda item: _get_paragraph_index(item)
        )

        supporting_chunks = [
            chunk for chunk in sorted_chunks if _is_supporting_doc(chunk)
        ]

        with output_path.open("w", encoding="utf-8") as file:
            file.write("# HotPotQA Question Context Inspection\n\n")

            file.write("## Question Metadata\n\n")
            file.write(f"- **Question ID:** `{question_id}`\n")
            file.write(f"- **Question:** {question}\n")
            file.write(f"- **Gold answer:** {answer}\n")
            file.write(f"- **Context documents/chunks found:** {len(sorted_chunks)}\n")
            file.write(f"- **Generated at:** {datetime.now().isoformat(timespec='seconds')}\n\n")

            file.write("## Gold Supporting Facts Summary\n\n")

            if not supporting_chunks:
                file.write("No supporting facts found in processed corpus for this question.\n\n")
            else:
                for index, chunk in enumerate(supporting_chunks, start=1):
                    file.write(f"### Supporting Document {index}: {_get_doc_title(chunk)}\n\n")
                    file.write(f"- **Chunk ID:** `{_get_chunk_id(chunk)}`\n")
                    file.write(f"- **Paragraph index:** {_get_paragraph_index(chunk)}\n")
                    file.write(
                        f"- **Supporting sentence IDs:** {_get_supporting_sentence_ids(chunk)}\n\n"
                    )

            file.write("## All Context Documents\n\n")

            for index, chunk in enumerate(sorted_chunks, start=1):
                title = _get_doc_title(chunk)
                chunk_id = _get_chunk_id(chunk)
                paragraph_index = _get_paragraph_index(chunk)
                supporting_sentence_ids = set(_get_supporting_sentence_ids(chunk))
                support_label = (
                    "GOLD SUPPORTING DOCUMENT"
                    if _is_supporting_doc(chunk)
                    else "distractor/context document"
                )

                file.write(f"### Context Document {index}: {title}\n\n")
                file.write(f"- **Chunk ID:** `{chunk_id}`\n")
                file.write(f"- **Paragraph index:** {paragraph_index}\n")
                file.write(f"- **Label:** {support_label}\n")
                file.write(f"- **Supporting sentence IDs:** {sorted(supporting_sentence_ids)}\n\n")

                sentences = _get_sentences(chunk)

                if sentences:
                    file.write("#### Sentences\n\n")

                    for sentence_index, sentence in enumerate(sentences):
                        sentence_id = _get_sentence_id(sentence, sentence_index)
                        sentence_text = _get_sentence_text(sentence)
                        marker = (
                            " **← GOLD SUPPORTING SENTENCE**"
                            if sentence_id in supporting_sentence_ids
                            else ""
                        )

                        file.write(f"- **s{sentence_id}**{marker}: {sentence_text}\n")

                    file.write("\n")
                else:
                    file.write("#### Sentences\n\n")
                    file.write("Sentences not found in processed chunk.\n\n")

                file.write("#### Full Paragraph\n\n")
                file.write(f"{_get_paragraph_text(chunk)}\n\n")
                file.write("---\n\n")


def main() -> None:
    config = _load_yaml_config(CONFIG_PATH)
    corpus_path = _project_path(config["paths"]["processed_corpus"])

    chunks = _load_jsonl(corpus_path)
    grouped_chunks = _group_by_question_id(chunks)

    question_id = _get_question_id_from_user(grouped_chunks)
    question_chunks = sorted(
        grouped_chunks[question_id],
        key=lambda item: _get_paragraph_index(item),
    )

    question = _get_question_text(question_chunks)
    answer = _get_gold_answer(question_chunks)

    print()
    print("=" * 100)
    print("Selected HotPotQA Question")
    print("=" * 100)
    print(f"Question ID: {question_id}")
    _print_wrapped_text(
        label="Question",
        text=question,
        label_indent="",
        text_indent="  ",
    )
    print(f"Gold answer: {answer}")
    print(f"Number of context documents/chunks found: {len(question_chunks)}")

    _print_gold_supporting_summary(question_chunks)

    print()
    print("=" * 100)
    print("All context documents for this question")
    print("=" * 100)

    for index, chunk in enumerate(question_chunks, start=1):
        _print_context_document(
            index=index,
            total_count=len(question_chunks),
            chunk=chunk,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"hotpotqa_question_{_safe_filename(question_id)}.md"
    )

    _write_question_markdown_report(
        question_id=question_id,
        question_chunks=question_chunks,
        output_path=output_path,
    )

    print()
    print("=" * 100)
    print(f"Markdown report saved to: {output_path}")


if __name__ == "__main__":
    main()