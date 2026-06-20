from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from epsa_rag.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class HybridFusionTrace:
    """
    Internal trace object for inspecting how BM25 and dense results contributed
    to the final hybrid RRF score.

    The public HybridRetriever output should still remain list[RetrievalResult].
    """

    chunk_id: str
    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None
    fusion_score: float


@dataclass
class _FusionState:
    chunk_id: str
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None
    fusion_score: float = 0.0


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def _get_chunk_id(result: RetrievalResult) -> str:
    chunk_id = getattr(result, "chunk_id", None)

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Every retrieval result must contain a non-empty chunk_id.")

    return chunk_id


def _get_score(result: RetrievalResult) -> float:
    score = getattr(result, "score", None)

    if score is None:
        return 0.0

    return float(score)


def reciprocal_rank_score(rank: int, rrf_k: int) -> float:
    """
    Compute the Reciprocal Rank Fusion contribution for one ranked item.

    Formula:
        1 / (rrf_k + rank)
    """

    _validate_positive_int("rank", rank)
    _validate_positive_int("rrf_k", rrf_k)

    return 1.0 / float(rrf_k + rank)


def reciprocal_rank_fusion_with_trace(
    bm25_results: Sequence[RetrievalResult],
    dense_results: Sequence[RetrievalResult],
    *,
    final_top_k: int,
    rrf_k: int,
) -> list[HybridFusionTrace]:
    """
    Fuse BM25 and dense ranked retrieval results using Reciprocal Rank Fusion.

    RRF uses the rank position within each source result list. Raw BM25 and dense
    scores are preserved only for trace/debugging, not for computing fusion.

    Tie-breaking is deterministic:
        1. higher fusion score
        2. better best source rank
        3. chunk_id alphabetically
    """

    _validate_positive_int("final_top_k", final_top_k)
    _validate_positive_int("rrf_k", rrf_k)

    states: dict[str, _FusionState] = {}

    def add_results(
        results: Sequence[RetrievalResult],
        source_name: str,
    ) -> None:
        seen_in_source: set[str] = set()

        for rank_position, result in enumerate(results, start=1):
            chunk_id = _get_chunk_id(result)

            # Ignore duplicates within the same retriever output. The first
            # occurrence is the highest-ranked occurrence and should be kept.
            if chunk_id in seen_in_source:
                continue

            seen_in_source.add(chunk_id)

            state = states.setdefault(chunk_id, _FusionState(chunk_id=chunk_id))
            contribution = reciprocal_rank_score(rank_position, rrf_k)

            if source_name == "bm25":
                state.bm25_rank = rank_position
                state.bm25_score = _get_score(result)
            elif source_name == "dense":
                state.dense_rank = rank_position
                state.dense_score = _get_score(result)
            else:
                raise ValueError(f"Unsupported fusion source: {source_name}")

            state.fusion_score += contribution

    add_results(bm25_results, "bm25")
    add_results(dense_results, "dense")

    traces = [
        HybridFusionTrace(
            chunk_id=state.chunk_id,
            bm25_rank=state.bm25_rank,
            dense_rank=state.dense_rank,
            bm25_score=state.bm25_score,
            dense_score=state.dense_score,
            fusion_score=state.fusion_score,
        )
        for state in states.values()
    ]

    def best_source_rank(trace: HybridFusionTrace) -> int:
        ranks = [
            rank
            for rank in (trace.bm25_rank, trace.dense_rank)
            if rank is not None
        ]
        return min(ranks) if ranks else 10**9

    traces.sort(
        key=lambda trace: (
            -trace.fusion_score,
            best_source_rank(trace),
            trace.chunk_id,
        )
    )

    return traces[:final_top_k]


def reciprocal_rank_fusion(
    bm25_results: Sequence[RetrievalResult],
    dense_results: Sequence[RetrievalResult],
    *,
    final_top_k: int,
    rrf_k: int,
    retriever_name: str = "hybrid",
) -> list[RetrievalResult]:
    """
    Fuse BM25 and dense results and return public RetrievalResult objects.

    The output score is the final RRF fusion score.
    """

    if not isinstance(retriever_name, str) or not retriever_name.strip():
        raise ValueError("retriever_name must be a non-empty string.")

    traces = reciprocal_rank_fusion_with_trace(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=final_top_k,
        rrf_k=rrf_k,
    )

    return [
        RetrievalResult(
            rank=rank,
            chunk_id=trace.chunk_id,
            score=trace.fusion_score,
            retriever_name=retriever_name,
        )
        for rank, trace in enumerate(traces, start=1)
    ]