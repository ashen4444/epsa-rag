import json
from pathlib import Path

import pytest

from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.bm25_retriever import BM25Retriever, tokenize_text
from epsa_rag.retrieval.retrieval_result import RetrievalResult


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_corpus_store(path: Path) -> CorpusStore:
    return CorpusStore.from_jsonl(path)


@pytest.fixture()
def tiny_corpus_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_corpus.jsonl"

    records = [
        {
            "chunk_id": "chunk_1",
            "source_question_id": "q1",
            "question": "Who discovered radium?",
            "answer": "Marie Curie",
            "doc_title": "Marie Curie",
            "paragraph_index": 0,
            "chunk_text": "Title: Marie Curie\nParagraph: Marie Curie discovered radium.",
            "paragraph_text": "Marie Curie discovered radium.",
            "sentences": [
                {
                    "sentence_id": 0,
                    "text": "Marie Curie discovered radium.",
                    "start_char": 0,
                    "end_char": 31,
                }
            ],
            "is_supporting_doc": True,
            "supporting_sentence_ids": [0],
        },
        {
            "chunk_id": "chunk_2",
            "source_question_id": "q1",
            "question": "Who discovered radium?",
            "answer": "Marie Curie",
            "doc_title": "Inception",
            "paragraph_index": 0,
            "chunk_text": "Title: Inception\nParagraph: Christopher Nolan directed Inception.",
            "paragraph_text": "Christopher Nolan directed Inception.",
            "sentences": [
                {
                    "sentence_id": 0,
                    "text": "Christopher Nolan directed Inception.",
                    "start_char": 0,
                    "end_char": 38,
                }
            ],
            "is_supporting_doc": False,
            "supporting_sentence_ids": [],
        },
        {
            "chunk_id": "chunk_3",
            "source_question_id": "q1",
            "question": "Who discovered radium?",
            "answer": "Marie Curie",
            "doc_title": "London",
            "paragraph_index": 0,
            "chunk_text": "Title: London\nParagraph: London is the capital of England.",
            "paragraph_text": "London is the capital of England.",
            "sentences": [
                {
                    "sentence_id": 0,
                    "text": "London is the capital of England.",
                    "start_char": 0,
                    "end_char": 33,
                }
            ],
            "is_supporting_doc": False,
            "supporting_sentence_ids": [],
        },
    ]

    _write_jsonl(path, records)
    return path


@pytest.fixture()
def tiny_corpus_store(tiny_corpus_path: Path) -> CorpusStore:
    return _load_corpus_store(tiny_corpus_path)


def test_tokenize_text_lowercases_and_extracts_words() -> None:
    tokens = tokenize_text("Marie Curie discovered Radium!")

    assert tokens == ["marie", "curie", "discovered", "radium"]


def test_bm25_retriever_builds_from_corpus_store(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    assert retriever.corpus_size == 3
    assert retriever.retriever_name == "bm25"


def test_search_returns_retrieval_result_objects(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    results = retriever.search("radium discovery", top_k=2)

    assert results
    assert all(isinstance(result, RetrievalResult) for result in results)


def test_search_returns_at_most_top_k_results(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    results = retriever.search("radium discovery", top_k=2)

    assert len(results) <= 2


def test_unique_term_query_ranks_expected_chunk_first(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    results = retriever.search("radium", top_k=3)

    assert results[0].chunk_id == "chunk_1"


def test_returned_ranks_start_at_one_and_increase_by_one(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    results = retriever.search("radium", top_k=3)

    assert [result.rank for result in results] == [1, 2, 3]


def test_returned_chunk_ids_exist_in_corpus(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    corpus_chunk_ids = set(tiny_corpus_store.all_chunk_ids())
    results = retriever.search("radium", top_k=3)

    assert all(result.chunk_id in corpus_chunk_ids for result in results)


def test_empty_query_raises_clear_error(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    with pytest.raises(ValueError, match="query must contain"):
        retriever.search("   ", top_k=3)


def test_query_without_searchable_tokens_raises_clear_error(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    with pytest.raises(ValueError, match="query must contain"):
        retriever.search("!!!", top_k=3)


def test_top_k_less_than_one_raises_clear_error(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("radium", top_k=0)


def test_all_returned_scores_are_floats(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    results = retriever.search("radium", top_k=3)

    assert all(isinstance(result.score, float) for result in results)


def test_tie_breaking_is_deterministic_by_corpus_order(
    tiny_corpus_store: CorpusStore,
) -> None:
    retriever = BM25Retriever.from_corpus_store(tiny_corpus_store)

    first_run = retriever.search("unknownterm", top_k=3)
    second_run = retriever.search("unknownterm", top_k=3)

    assert [result.chunk_id for result in first_run] == [
        result.chunk_id for result in second_run
    ]
    assert [result.chunk_id for result in first_run] == [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]


def test_building_with_empty_corpus_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        BM25Retriever(chunk_ids=[], chunk_texts=[])


def test_building_with_misaligned_corpus_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        BM25Retriever(chunk_ids=["chunk_1"], chunk_texts=["text one", "text two"])