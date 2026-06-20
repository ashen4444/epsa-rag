from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.dense_index import FaissDenseIndex
from epsa_rag.retrieval.dense_retriever import DenseRetriever
from epsa_rag.retrieval.retrieval_result import RetrievalResult


def write_controlled_corpus(path: Path) -> None:
    records = [
        {
            "chunk_id": "chunk_curie",
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
            "chunk_id": "chunk_nolan",
            "source_question_id": "q2",
            "question": "Who directed Inception?",
            "answer": "Christopher Nolan",
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
            "is_supporting_doc": True,
            "supporting_sentence_ids": [0],
        },
        {
            "chunk_id": "chunk_london",
            "source_question_id": "q3",
            "question": "What is the capital of England?",
            "answer": "London",
            "doc_title": "London",
            "paragraph_index": 0,
            "chunk_text": "Title: London\nParagraph: London is the capital of England.",
            "paragraph_text": "London is the capital of England.",
            "sentences": [
                {
                    "sentence_id": 0,
                    "text": "London is the capital of England.",
                    "start_char": 0,
                    "end_char": 34,
                }
            ],
            "is_supporting_doc": True,
            "supporting_sentence_ids": [0],
        },
    ]

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.fixture()
def corpus_store(tmp_path: Path) -> CorpusStore:
    corpus_path = tmp_path / "paragraph_chunks.jsonl"
    write_controlled_corpus(corpus_path)
    return CorpusStore.from_jsonl(corpus_path)


@pytest.fixture()
def dense_index() -> FaissDenseIndex:
    chunk_ids = ["chunk_curie", "chunk_nolan", "chunk_london"]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return FaissDenseIndex.from_embeddings(
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        normalize_embeddings=True,
        metadata={"purpose": "dense_retriever_unit_test"},
    )


def test_dense_retriever_returns_ranked_retrieval_results(
    corpus_store: CorpusStore,
    dense_index: FaissDenseIndex,
) -> None:
    retriever = DenseRetriever.from_index(
        corpus_store=corpus_store,
        dense_index=dense_index,
    )

    results = retriever.search_by_vector(
        query_embedding=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        top_k=2,
    )

    assert len(results) == 2
    assert all(isinstance(result, RetrievalResult) for result in results)
    assert [result.rank for result in results] == [1, 2]
    assert results[0].chunk_id == "chunk_nolan"
    assert results[0].retriever_name == "dense"
    assert isinstance(results[0].score, float)


def test_dense_retriever_returns_at_most_top_k(
    corpus_store: CorpusStore,
    dense_index: FaissDenseIndex,
) -> None:
    retriever = DenseRetriever.from_index(
        corpus_store=corpus_store,
        dense_index=dense_index,
    )

    results = retriever.search_by_vector(
        query_embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_curie"


def test_dense_retriever_result_chunk_ids_exist_in_corpus(
    corpus_store: CorpusStore,
    dense_index: FaissDenseIndex,
) -> None:
    retriever = DenseRetriever.from_index(
        corpus_store=corpus_store,
        dense_index=dense_index,
    )

    results = retriever.search_by_vector(
        query_embedding=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        top_k=3,
    )

    corpus_chunk_ids = set(corpus_store.all_chunk_ids())

    assert all(result.chunk_id in corpus_chunk_ids for result in results)


def test_dense_index_rejects_top_k_less_than_one(dense_index: FaissDenseIndex) -> None:
    with pytest.raises(ValueError, match="top_k"):
        dense_index.search(
            query_embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            top_k=0,
        )


def test_dense_index_rejects_dimension_mismatch(dense_index: FaissDenseIndex) -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        dense_index.search(
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=1,
        )


def test_dense_index_rejects_zero_query_vector(dense_index: FaissDenseIndex) -> None:
    with pytest.raises(ValueError, match="zero vector"):
        dense_index.search(
            query_embedding=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            top_k=1,
        )


def test_dense_index_rejects_invalid_embedding_values() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        FaissDenseIndex.from_embeddings(
            chunk_ids=["chunk_bad"],
            embeddings=np.asarray([[float("nan"), 0.0, 1.0]], dtype=np.float32),
            normalize_embeddings=True,
        )


def test_dense_index_rejects_zero_document_vector() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        FaissDenseIndex.from_embeddings(
            chunk_ids=["chunk_bad"],
            embeddings=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            normalize_embeddings=True,
        )


def test_dense_index_rejects_embedding_chunk_id_length_mismatch() -> None:
    with pytest.raises(ValueError, match="row count"):
        FaissDenseIndex.from_embeddings(
            chunk_ids=["chunk_a", "chunk_b"],
            embeddings=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            normalize_embeddings=True,
        )


def test_dense_index_tie_breaking_is_deterministic() -> None:
    index = FaissDenseIndex.from_embeddings(
        chunk_ids=["chunk_b", "chunk_a"],
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        normalize_embeddings=True,
    )

    results = index.search(
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=2,
    )

    assert [result.chunk_id for result in results] == ["chunk_a", "chunk_b"]


def test_dense_index_save_and_load_preserves_alignment(tmp_path: Path) -> None:
    original_index = FaissDenseIndex.from_embeddings(
        chunk_ids=["chunk_curie", "chunk_nolan", "chunk_london"],
        embeddings=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        normalize_embeddings=True,
        metadata={"embedding_model": "controlled_vectors"},
    )

    index_path = tmp_path / "faiss_index.bin"
    metadata_path = tmp_path / "dense_metadata.json"

    original_index.save(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    loaded_index = FaissDenseIndex.load(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    assert loaded_index.chunk_ids == original_index.chunk_ids
    assert loaded_index.dimension == original_index.dimension
    assert loaded_index.size == original_index.size

    results = loaded_index.search(
        query_embedding=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        top_k=1,
    )

    assert results[0].chunk_id == "chunk_london"


def test_dense_retriever_rejects_index_with_unknown_chunk_id(
    corpus_store: CorpusStore,
) -> None:
    index = FaissDenseIndex.from_embeddings(
        chunk_ids=["unknown_chunk"],
        embeddings=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        normalize_embeddings=True,
    )

    with pytest.raises(ValueError, match="do not exist in CorpusStore"):
        DenseRetriever.from_index(
            corpus_store=corpus_store,
            dense_index=index,
        )


def test_dense_retriever_search_requires_embedder(
    corpus_store: CorpusStore,
    dense_index: FaissDenseIndex,
) -> None:
    retriever = DenseRetriever.from_index(
        corpus_store=corpus_store,
        dense_index=dense_index,
    )

    with pytest.raises(RuntimeError, match="requires an embedder"):
        retriever.search(
            query="Who discovered radium?",
            top_k=1,
        )