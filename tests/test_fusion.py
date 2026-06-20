import pytest

from epsa_rag.retrieval.fusion import (
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_with_trace,
    reciprocal_rank_score,
)
from epsa_rag.retrieval.retrieval_result import RetrievalResult


def make_result(
    rank: int,
    chunk_id: str,
    score: float,
    retriever_name: str = "test",
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        chunk_id=chunk_id,
        score=score,
        retriever_name=retriever_name,
    )


def test_reciprocal_rank_score() -> None:
    assert reciprocal_rank_score(rank=1, rrf_k=60) == pytest.approx(1 / 61)
    assert reciprocal_rank_score(rank=5, rrf_k=60) == pytest.approx(1 / 65)


def test_rrf_combines_bm25_and_dense_ranked_lists() -> None:
    bm25_results = [
        make_result(1, "chunk-a", 12.0, "bm25"),
        make_result(2, "chunk-b", 8.0, "bm25"),
    ]
    dense_results = [
        make_result(1, "chunk-c", 0.91, "dense"),
        make_result(2, "chunk-a", 0.88, "dense"),
    ]

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=10,
        rrf_k=60,
    )

    by_chunk_id = {result.chunk_id: result for result in fused}

    assert set(by_chunk_id) == {"chunk-a", "chunk-b", "chunk-c"}
    assert by_chunk_id["chunk-a"].score == pytest.approx((1 / 61) + (1 / 62))
    assert by_chunk_id["chunk-b"].score == pytest.approx(1 / 62)
    assert by_chunk_id["chunk-c"].score == pytest.approx(1 / 61)
    assert fused[0].chunk_id == "chunk-a"


def test_rrf_includes_chunks_that_appear_in_only_one_source() -> None:
    bm25_results = [
        make_result(1, "chunk-a", 12.0, "bm25"),
    ]
    dense_results = [
        make_result(1, "chunk-b", 0.91, "dense"),
    ]

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=10,
        rrf_k=60,
    )

    assert {result.chunk_id for result in fused} == {"chunk-a", "chunk-b"}


def test_final_ranks_are_contiguous_and_start_at_one() -> None:
    bm25_results = [
        make_result(1, "chunk-a", 12.0, "bm25"),
        make_result(2, "chunk-b", 8.0, "bm25"),
    ]
    dense_results = [
        make_result(1, "chunk-c", 0.91, "dense"),
    ]

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=10,
        rrf_k=60,
    )

    assert [result.rank for result in fused] == [1, 2, 3]


def test_final_result_count_is_limited_by_top_k() -> None:
    bm25_results = [
        make_result(1, "chunk-a", 12.0, "bm25"),
        make_result(2, "chunk-b", 8.0, "bm25"),
    ]
    dense_results = [
        make_result(1, "chunk-c", 0.91, "dense"),
        make_result(2, "chunk-d", 0.89, "dense"),
    ]

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=2,
        rrf_k=60,
    )

    assert len(fused) == 2


def test_tie_breaking_is_deterministic() -> None:
    bm25_results = [
        make_result(1, "chunk-b", 12.0, "bm25"),
    ]
    dense_results = [
        make_result(1, "chunk-a", 0.91, "dense"),
    ]

    first_run = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=10,
        rrf_k=60,
    )
    second_run = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=10,
        rrf_k=60,
    )

    assert [result.chunk_id for result in first_run] == [
        result.chunk_id for result in second_run
    ]
    assert [result.chunk_id for result in first_run] == ["chunk-a", "chunk-b"]


def test_trace_preserves_source_ranks_and_scores() -> None:
    bm25_results = [
        make_result(1, "chunk-a", 12.0, "bm25"),
    ]
    dense_results = [
        make_result(1, "chunk-a", 0.91, "dense"),
    ]

    traces = reciprocal_rank_fusion_with_trace(
        bm25_results=bm25_results,
        dense_results=dense_results,
        final_top_k=10,
        rrf_k=60,
    )

    assert len(traces) == 1
    assert traces[0].chunk_id == "chunk-a"
    assert traces[0].bm25_rank == 1
    assert traces[0].dense_rank == 1
    assert traces[0].bm25_score == pytest.approx(12.0)
    assert traces[0].dense_score == pytest.approx(0.91)
    assert traces[0].fusion_score == pytest.approx((1 / 61) + (1 / 61))


@pytest.mark.parametrize(
    ("final_top_k", "rrf_k"),
    [
        (0, 60),
        (10, 0),
        (-1, 60),
        (10, -1),
    ],
)
def test_rrf_validates_positive_integer_settings(
    final_top_k: int,
    rrf_k: int,
) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            bm25_results=[],
            dense_results=[],
            final_top_k=final_top_k,
            rrf_k=rrf_k,
        )


def test_rrf_rejects_invalid_chunk_id() -> None:
    invalid_result = RetrievalResult(
        rank=1,
        chunk_id="",
        score=1.0,
        retriever_name="bm25",
    )

    with pytest.raises(ValueError, match="chunk_id"):
        reciprocal_rank_fusion(
            bm25_results=[invalid_result],
            dense_results=[],
            final_top_k=10,
            rrf_k=60,
        )