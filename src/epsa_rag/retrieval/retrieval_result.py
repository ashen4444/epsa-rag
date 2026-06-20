from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    Ranked retrieval result returned by a retrieval component.

    This schema is intentionally small. Full chunk metadata should be fetched
    from CorpusStore using chunk_id.
    """

    rank: int
    chunk_id: str
    score: float
    retriever_name: str = "bm25"