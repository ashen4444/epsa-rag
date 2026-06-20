from __future__ import annotations

from pathlib import Path

import numpy as np

from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.dense_index import FaissDenseIndex
from epsa_rag.retrieval.embedding_backend import TextEmbedder
from epsa_rag.retrieval.retrieval_result import RetrievalResult


class DenseRetriever:
    """
    Dense semantic retriever over CorpusStore paragraph chunks.

    Output is intentionally minimal:
        rank, chunk_id, score, retriever_name

    Full chunk metadata must be fetched from CorpusStore using chunk_id.
    """

    retriever_name = "dense"

    def __init__(
        self,
        corpus_store: CorpusStore,
        dense_index: FaissDenseIndex,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self._corpus_store = corpus_store
        self._dense_index = dense_index
        self._embedder = embedder
        self._validate_index_against_corpus()

    @classmethod
    def from_corpus_store(
        cls,
        corpus_store: CorpusStore,
        embedder: TextEmbedder,
        normalize_embeddings: bool = True,
    ) -> DenseRetriever:
        chunk_ids = list(corpus_store.all_chunk_ids())
        chunk_texts = list(corpus_store.all_chunk_texts())

        if not chunk_ids:
            raise ValueError("CorpusStore is empty. Cannot build dense retriever.")

        if len(chunk_ids) != len(chunk_texts):
            raise ValueError(
                "CorpusStore chunk ID count and chunk text count do not match. "
                f"Got {len(chunk_ids)} chunk IDs and {len(chunk_texts)} chunk texts."
            )

        embeddings = embedder.embed_texts(chunk_texts)

        dense_index = FaissDenseIndex.from_embeddings(
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            normalize_embeddings=normalize_embeddings,
            metadata={
                "embedding_provider": "openai",
                "embedding_model": embedder.model_name,
                "source": "CorpusStore.all_chunk_texts",
            },
        )

        return cls(
            corpus_store=corpus_store,
            dense_index=dense_index,
            embedder=embedder,
        )

    @classmethod
    def from_index(
        cls,
        corpus_store: CorpusStore,
        dense_index: FaissDenseIndex,
        embedder: TextEmbedder | None = None,
    ) -> DenseRetriever:
        return cls(
            corpus_store=corpus_store,
            dense_index=dense_index,
            embedder=embedder,
        )

    @classmethod
    def load(
        cls,
        corpus_store: CorpusStore,
        index_path: str | Path,
        metadata_path: str | Path,
        embedder: TextEmbedder | None = None,
    ) -> DenseRetriever:
        dense_index = FaissDenseIndex.load(
            index_path=index_path,
            metadata_path=metadata_path,
        )

        return cls.from_index(
            corpus_store=corpus_store,
            dense_index=dense_index,
            embedder=embedder,
        )

    @property
    def index_size(self) -> int:
        return self._dense_index.size

    @property
    def embedding_dimension(self) -> int:
        return self._dense_index.dimension

    def save_index(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        self._dense_index.save(
            index_path=index_path,
            metadata_path=metadata_path,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._embedder is None:
            raise RuntimeError(
                "DenseRetriever.search() requires an embedder. "
                "Load or build the retriever with an embedder first."
            )

        cleaned_query = self._validate_query(query)
        query_embedding = self._embedder.embed_query(cleaned_query)

        return self.search_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def search_by_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        hits = self._dense_index.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return [
            RetrievalResult(
                rank=hit.rank,
                chunk_id=hit.chunk_id,
                score=hit.score,
                retriever_name=self.retriever_name,
            )
            for hit in hits
        ]

    def _validate_index_against_corpus(self) -> None:
        corpus_chunk_ids = set(self._corpus_store.all_chunk_ids())

        missing_chunk_ids = [
            chunk_id
            for chunk_id in self._dense_index.chunk_ids
            if chunk_id not in corpus_chunk_ids
        ]

        if missing_chunk_ids:
            preview = ", ".join(missing_chunk_ids[:5])
            raise ValueError(
                "Dense index contains chunk IDs that do not exist in CorpusStore. "
                f"Missing examples: {preview}"
            )

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query must not be empty.")

        return cleaned