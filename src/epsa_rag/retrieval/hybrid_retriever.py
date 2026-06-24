from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from epsa_rag.retrieval.retrieval_result import RetrievalResult

from epsa_rag.retrieval.fusion import (
    HybridFusionTrace,
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_with_trace,
)

@runtime_checkable
class RankedRetriever(Protocol):
    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        ...


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _read_nested_setting(settings: Any, *keys: str, default: Any = None) -> Any:
    """
    Read nested config values from either dict-like or attribute-like settings.

    Supports both:
        settings["retrieval"]["top_k"]
    and:
        settings.retrieval.top_k
    """

    current = settings

    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue

        if not hasattr(current, key):
            return default

        current = getattr(current, key)

    return current


class HybridRetriever:
    """
    Hybrid retrieval orchestrator.

    Responsibilities:
        1. call BM25Retriever
        2. call DenseRetriever
        3. fuse both ranked lists using Reciprocal Rank Fusion
        4. return list[RetrievalResult] with retriever_name='hybrid'

    It does not fetch full paragraph metadata and does not perform EPSA logic.
    """

    retriever_name = "hybrid"

    def __init__(
        self,
        bm25_retriever: RankedRetriever,
        dense_retriever: RankedRetriever,
        bm25_top_k: int,
        dense_top_k: int,
        final_top_k: int,
        rrf_k: int,
    ) -> None:
        if not isinstance(bm25_retriever, RankedRetriever):
            raise TypeError("bm25_retriever must provide a search(query, top_k) method.")

        if not isinstance(dense_retriever, RankedRetriever):
            raise TypeError("dense_retriever must provide a search(query, top_k) method.")

        _validate_positive_int("bm25_top_k", bm25_top_k)
        _validate_positive_int("dense_top_k", dense_top_k)
        _validate_positive_int("final_top_k", final_top_k)
        _validate_positive_int("rrf_k", rrf_k)

        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    @classmethod
    def from_settings(
        cls,
        bm25_retriever: RankedRetriever,
        dense_retriever: RankedRetriever,
        settings: Any,
    ) -> "HybridRetriever":
        """
        Build HybridRetriever from retrieval settings.

        Expected config shape:

            retrieval:
              top_k: 20
              bm25_top_k: 50
              dense_top_k: 50
              fusion_method: "rrf"
              rrf_k: 60
        """

        fusion_method = _read_nested_setting(
            settings,
            "retrieval",
            "fusion_method",
            default="rrf",
        )

        if str(fusion_method).lower() != "rrf":
            raise ValueError(
                "HybridRetriever currently supports only fusion_method='rrf'."
            )

        bm25_top_k = _read_nested_setting(settings, "retrieval", "bm25_top_k")
        dense_top_k = _read_nested_setting(settings, "retrieval", "dense_top_k")
        final_top_k = _read_nested_setting(settings, "retrieval", "top_k")
        rrf_k = _read_nested_setting(settings, "retrieval", "rrf_k")

        missing = [
            name
            for name, value in {
                "retrieval.bm25_top_k": bm25_top_k,
                "retrieval.dense_top_k": dense_top_k,
                "retrieval.top_k": final_top_k,
                "retrieval.rrf_k": rrf_k,
            }.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                "Missing required hybrid retrieval settings: "
                + ", ".join(missing)
            )

        return cls(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            bm25_top_k=int(bm25_top_k),
            dense_top_k=int(dense_top_k),
            final_top_k=int(final_top_k),
            rrf_k=int(rrf_k),
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        final_top_k = self.final_top_k if top_k is None else top_k
        _validate_positive_int("top_k", final_top_k)

        clean_query = query.strip()

        bm25_results = self.bm25_retriever.search(
            clean_query,
            top_k=self.bm25_top_k,
        )
        dense_results = self.dense_retriever.search(
            clean_query,
            top_k=self.dense_top_k,
        )

        return reciprocal_rank_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            final_top_k=final_top_k,
            rrf_k=self.rrf_k,
            retriever_name=self.retriever_name,
        )

    
    def search_with_trace(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[HybridFusionTrace]:
        """
        Search using BM25 + dense retrieval and return detailed RRF fusion traces.

        This method is intended for analysis/debugging only.
        The production public retrieval output remains search(), which returns
        list[RetrievalResult].
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        final_top_k = self.final_top_k if top_k is None else top_k
        _validate_positive_int("top_k", final_top_k)

        clean_query = query.strip()

        bm25_results = self.bm25_retriever.search(
            clean_query,
            top_k=self.bm25_top_k,
        )
        dense_results = self.dense_retriever.search(
            clean_query,
            top_k=self.dense_top_k,
        )

        return reciprocal_rank_fusion_with_trace(
            bm25_results=bm25_results,
            dense_results=dense_results,
            final_top_k=final_top_k,
            rrf_k=self.rrf_k,
        )