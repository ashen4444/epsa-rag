import json
from pathlib import Path
from typing import Any

import pytest

from epsa_rag.corpus import (
    ChunkNotFoundError,
    CorpusStore,
    CorpusStoreError,
    DuplicateChunkIdError,
    EmptyCorpusError,
)
from epsa_rag.datasets.schemas import ParagraphChunk, SentenceMetadata


def _model_fields(model: type) -> dict[str, Any]:
    """Return Pydantic v2/v1 model fields without triggering deprecation warnings."""

    model_fields = getattr(model, "model_fields", None)
    if model_fields is not None:
        return model_fields

    return getattr(model, "__fields__", {})


def _field_names(model: type) -> set[str]:
    return set(_model_fields(model).keys())


def _add_if_field(record: dict[str, Any], model: type, field_name: str, value: Any) -> None:
    if field_name in _field_names(model):
        record[field_name] = value


def _sentence_record(
    text: str,
    sentence_index: int = 0,
    is_supporting: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {}

    # Support both possible naming styles from schema evolution.
    _add_if_field(record, SentenceMetadata, "sentence_id", sentence_index)
    _add_if_field(record, SentenceMetadata, "sentence_index", sentence_index)
    _add_if_field(record, SentenceMetadata, "index", sentence_index)

    _add_if_field(record, SentenceMetadata, "text", text)
    _add_if_field(record, SentenceMetadata, "sentence_text", text)

    _add_if_field(record, SentenceMetadata, "start_char", 0)
    _add_if_field(record, SentenceMetadata, "end_char", len(text))

    _add_if_field(record, SentenceMetadata, "is_supporting_fact", is_supporting)
    _add_if_field(record, SentenceMetadata, "is_supporting_sentence", is_supporting)
    _add_if_field(record, SentenceMetadata, "is_supporting", is_supporting)

    return record


def _chunk_record(
    chunk_id: str,
    source_question_id: str,
    doc_title: str,
    paragraph_index: int,
    paragraph_text: str,
    is_supporting_doc: bool,
    supporting_sentence_ids: list[int] | None = None,
) -> dict[str, Any]:
    if supporting_sentence_ids is None:
        supporting_sentence_ids = []

    chunk_text = f"Title: {doc_title}\nParagraph: {paragraph_text}"

    record: dict[str, Any] = {}

    _add_if_field(record, ParagraphChunk, "chunk_id", chunk_id)
    _add_if_field(record, ParagraphChunk, "source_question_id", source_question_id)
    _add_if_field(record, ParagraphChunk, "question_id", source_question_id)

    _add_if_field(record, ParagraphChunk, "doc_title", doc_title)
    _add_if_field(record, ParagraphChunk, "title", doc_title)

    _add_if_field(record, ParagraphChunk, "paragraph_index", paragraph_index)
    _add_if_field(record, ParagraphChunk, "paragraph_text", paragraph_text)
    _add_if_field(record, ParagraphChunk, "chunk_text", chunk_text)

    _add_if_field(
        record,
        ParagraphChunk,
        "sentences",
        [
            _sentence_record(
                text=paragraph_text,
                sentence_index=0,
                is_supporting=0 in supporting_sentence_ids,
            )
        ],
    )

    _add_if_field(record, ParagraphChunk, "is_supporting_doc", is_supporting_doc)
    _add_if_field(record, ParagraphChunk, "is_supporting_document", is_supporting_doc)

    _add_if_field(record, ParagraphChunk, "supporting_sentence_ids", supporting_sentence_ids)
    _add_if_field(record, ParagraphChunk, "supporting_sentence_indices", supporting_sentence_ids)

    # Add common HotPotQA metadata fields only if your schema requires/includes them.
    _add_if_field(record, ParagraphChunk, "question", "Sample test question?")
    _add_if_field(record, ParagraphChunk, "answer", "Sample answer")
    _add_if_field(record, ParagraphChunk, "question_type", "bridge")
    _add_if_field(record, ParagraphChunk, "level", "easy")
    _add_if_field(record, ParagraphChunk, "split", "train")

    # Validate here so fixture errors are caught directly and clearly.
    if hasattr(ParagraphChunk, "model_validate"):
        chunk = ParagraphChunk.model_validate(record)
        return chunk.model_dump()

    chunk = ParagraphChunk.parse_obj(record)
    return chunk.dict()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    return [
        _chunk_record(
            chunk_id="hotpot_train_000001::Marie_Curie::p0",
            source_question_id="hotpot_train_000001",
            doc_title="Marie Curie",
            paragraph_index=0,
            paragraph_text="Marie Curie discovered radium.",
            is_supporting_doc=True,
            supporting_sentence_ids=[0],
        ),
        _chunk_record(
            chunk_id="hotpot_train_000001::Radium::p0",
            source_question_id="hotpot_train_000001",
            doc_title="Radium",
            paragraph_index=0,
            paragraph_text="Radium is a chemical element.",
            is_supporting_doc=False,
        ),
        _chunk_record(
            chunk_id="hotpot_train_000002::Christopher_Nolan::p0",
            source_question_id="hotpot_train_000002",
            doc_title="Christopher Nolan",
            paragraph_index=0,
            paragraph_text="Christopher Nolan was born in London.",
            is_supporting_doc=True,
            supporting_sentence_ids=[0],
        ),
    ]


def test_corpus_store_loads_valid_jsonl_records(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)

    assert len(store) == 3
    assert store.all_chunk_ids() == [
        "hotpot_train_000001::Marie_Curie::p0",
        "hotpot_train_000001::Radium::p0",
        "hotpot_train_000002::Christopher_Nolan::p0",
    ]


def test_corpus_store_preserves_chunk_order(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)

    assert [chunk.chunk_id for chunk in store.all_chunks()] == [
        record["chunk_id"] for record in sample_records
    ]


def test_get_chunk_returns_correct_paragraph_chunk(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)
    chunk = store.get_chunk("hotpot_train_000001::Marie_Curie::p0")

    assert chunk.chunk_id == "hotpot_train_000001::Marie_Curie::p0"
    assert chunk.source_question_id == "hotpot_train_000001"
    assert chunk.doc_title == "Marie Curie"
    assert chunk.is_supporting_doc is True


def test_get_chunk_raises_clear_error_for_missing_chunk_id(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)

    with pytest.raises(ChunkNotFoundError, match="Chunk not found"):
        store.get_chunk("missing_chunk_id")


def test_get_chunks_returns_chunks_in_requested_order(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)

    chunks = store.get_chunks(
        [
            "hotpot_train_000002::Christopher_Nolan::p0",
            "hotpot_train_000001::Marie_Curie::p0",
        ]
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "hotpot_train_000002::Christopher_Nolan::p0",
        "hotpot_train_000001::Marie_Curie::p0",
    ]


def test_duplicate_chunk_id_raises_error(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    duplicate = dict(sample_records[0])
    records = sample_records + [duplicate]

    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, records)

    with pytest.raises(DuplicateChunkIdError, match="Duplicate chunk_id"):
        CorpusStore.from_jsonl(corpus_path)


def test_empty_jsonl_file_raises_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "empty.jsonl"
    corpus_path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyCorpusError, match="empty"):
        CorpusStore.from_jsonl(corpus_path)


def test_get_by_question_id_returns_all_chunks_for_question(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)
    chunks = store.get_by_question_id("hotpot_train_000001")

    assert [chunk.chunk_id for chunk in chunks] == [
        "hotpot_train_000001::Marie_Curie::p0",
        "hotpot_train_000001::Radium::p0",
    ]


def test_get_by_doc_title_returns_matching_chunks(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)
    chunks = store.get_by_doc_title("Marie Curie")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "hotpot_train_000001::Marie_Curie::p0"


def test_get_supporting_chunks_returns_only_supporting_chunks(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)
    chunks = store.get_supporting_chunks()

    assert [chunk.chunk_id for chunk in chunks] == [
        "hotpot_train_000001::Marie_Curie::p0",
        "hotpot_train_000002::Christopher_Nolan::p0",
    ]
    assert all(chunk.is_supporting_doc for chunk in chunks)


def test_get_non_supporting_chunks_returns_only_non_supporting_chunks(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)
    chunks = store.get_non_supporting_chunks()

    assert [chunk.chunk_id for chunk in chunks] == [
        "hotpot_train_000001::Radium::p0",
    ]
    assert all(not chunk.is_supporting_doc for chunk in chunks)


def test_all_chunk_ids_and_all_chunk_texts_are_aligned(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)

    chunk_ids = store.all_chunk_ids()
    chunk_texts = store.all_chunk_texts()
    chunks = store.all_chunks()

    assert len(chunk_ids) == len(chunk_texts) == len(chunks)

    for index, chunk in enumerate(chunks):
        assert chunk_ids[index] == chunk.chunk_id
        assert chunk_texts[index] == chunk.chunk_text


def test_stats_returns_correct_counts(
    tmp_path: Path,
    sample_records: list[dict[str, Any]],
) -> None:
    corpus_path = tmp_path / "chunks.jsonl"
    _write_jsonl(corpus_path, sample_records)

    store = CorpusStore.from_jsonl(corpus_path)

    assert store.stats() == {
        "total_chunks": 3,
        "unique_questions": 2,
        "unique_doc_titles": 3,
        "supporting_chunks": 2,
        "non_supporting_chunks": 1,
    }


def test_missing_jsonl_path_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(CorpusStoreError, match="does not exist"):
        CorpusStore.from_jsonl(missing_path)


def test_invalid_json_record_raises_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "invalid.jsonl"
    corpus_path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(CorpusStoreError, match="Invalid JSON"):
        CorpusStore.from_jsonl(corpus_path)


def test_schema_mismatch_raises_error(tmp_path: Path) -> None:
    corpus_path = tmp_path / "schema_mismatch.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "chunk_id": "bad_chunk",
                "doc_title": "Missing Required Fields",
            }
        ],
    )

    with pytest.raises(CorpusStoreError, match="ParagraphChunk schema"):
        CorpusStore.from_jsonl(corpus_path)