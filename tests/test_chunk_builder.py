from __future__ import annotations

import json
from pathlib import Path

from epsa_rag.datasets.chunk_builder import (
    build_chunks_for_example,
    build_paragraph_chunks,
    build_sentence_metadata,
    make_safe_doc_title,
)
from epsa_rag.datasets.schemas import HotPotQAExample
from epsa_rag.utils.jsonl import read_jsonl, write_jsonl


def make_example() -> HotPotQAExample:
    return HotPotQAExample(
        source_question_id="hotpot_train_000042",
        question="Where was Marie Curie born?",
        answer="Warsaw",
        question_type="bridge",
        level="medium",
        supporting_facts=[
            ("Marie Curie", 0),
            ("Marie Curie", 1),
        ],
        context=[
            (
                "Marie Curie",
                [
                    "Marie Curie was born in Warsaw.",
                    "She later moved to France.",
                ],
            ),
            (
                "Pierre Curie",
                [
                    "Pierre Curie was a French physicist.",
                ],
            ),
        ],
    )


def test_build_sentence_metadata_preserves_sentence_ids_and_offsets() -> None:
    paragraph_text, sentences = build_sentence_metadata(
        [
            "First sentence.",
            "Second sentence.",
        ]
    )

    assert paragraph_text == "First sentence. Second sentence."

    assert len(sentences) == 2

    assert sentences[0].sentence_id == 0
    assert sentences[0].text == "First sentence."
    assert sentences[0].start_char == 0
    assert sentences[0].end_char == 15

    assert sentences[1].sentence_id == 1
    assert sentences[1].text == "Second sentence."
    assert sentences[1].start_char == 16
    assert sentences[1].end_char == 32


def test_build_chunks_for_example_creates_one_chunk_per_context_document() -> None:
    example = make_example()

    chunks = build_chunks_for_example(example)

    assert len(chunks) == 2
    assert chunks[0].doc_title == "Marie Curie"
    assert chunks[1].doc_title == "Pierre Curie"


def test_chunk_text_and_paragraph_text_are_correct() -> None:
    example = make_example()

    chunks = build_chunks_for_example(example)
    first_chunk = chunks[0]

    assert (
        first_chunk.paragraph_text
        == "Marie Curie was born in Warsaw. She later moved to France."
    )
    assert (
        first_chunk.chunk_text
        == "Title: Marie Curie\nParagraph: Marie Curie was born in Warsaw. She later moved to France."
    )


def test_supporting_sentence_ids_are_mapped_correctly() -> None:
    example = make_example()

    chunks = build_chunks_for_example(example)

    marie_chunk = chunks[0]
    pierre_chunk = chunks[1]

    assert marie_chunk.supporting_sentence_ids == [0, 1]
    assert marie_chunk.is_supporting_doc is True

    assert pierre_chunk.supporting_sentence_ids == []
    assert pierre_chunk.is_supporting_doc is False


def test_chunk_id_is_stable_and_safe() -> None:
    example = make_example()

    chunks = build_chunks_for_example(example)

    assert chunks[0].chunk_id == "hotpot_train_000042::Marie_Curie::p0"
    assert chunks[1].chunk_id == "hotpot_train_000042::Pierre_Curie::p1"


def test_make_safe_doc_title_normalizes_problematic_characters() -> None:
    assert make_safe_doc_title("Marie Curie") == "Marie_Curie"
    assert make_safe_doc_title("A/B: Test!") == "AB_Test"
    assert make_safe_doc_title("   ") == "untitled"


def test_build_paragraph_chunks_handles_multiple_examples() -> None:
    examples = [make_example(), make_example()]

    chunks = build_paragraph_chunks(examples)

    assert len(chunks) == 4


def test_jsonl_writer_creates_valid_line_by_line_json(tmp_path: Path) -> None:
    example = make_example()
    chunks = build_chunks_for_example(example)

    output_path = tmp_path / "chunks.jsonl"

    written_count = write_jsonl(
        records=(chunk.to_json_dict() for chunk in chunks),
        output_path=output_path,
    )

    assert written_count == 2
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    first_record = json.loads(lines[0])
    second_record = json.loads(lines[1])

    assert first_record["doc_title"] == "Marie Curie"
    assert second_record["doc_title"] == "Pierre Curie"

    loaded_records = read_jsonl(output_path)

    assert len(loaded_records) == 2
    assert loaded_records[0]["chunk_id"] == "hotpot_train_000042::Marie_Curie::p0"