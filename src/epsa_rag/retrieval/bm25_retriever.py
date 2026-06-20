import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.retrieval_result import RetrievalResult


_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize_text(text: str) -> list[str]:
    """
    Deterministic lightweight tokenizer for BM25.

    Current preprocessing:
    - lowercase
    - basic word tokenization
    - empty-token removal

    No stemming, lemmatization, or stopword removal is used yet.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    return _WORD_PATTERN.findall(text.lower())


class BM25Retriever:
    """
    BM25 retriever over processed paragraph chunks loaded by CorpusStore.

    Important alignment rule:
        index i -> chunk_ids[i] -> chunk_texts[i]

    The retriever never parses the raw HotPotQA dataset directly.
    """

    def __init__(
        self,
        chunk_ids: Sequence[str],
        chunk_texts: Sequence[str],
        retriever_name: str = "bm25",
    ) -> None:
        self._validate_corpus_inputs(chunk_ids=chunk_ids, chunk_texts=chunk_texts)

        self._chunk_ids: tuple[str, ...] = tuple(chunk_ids)
        self._chunk_texts: tuple[str, ...] = tuple(chunk_texts)
        self._retriever_name = retriever_name

        tokenized_corpus = [tokenize_text(text) for text in self._chunk_texts]

        if not any(tokenized_corpus):
            raise ValueError("BM25 corpus must contain at least one tokenized term.")

        self._bm25 = BM25Okapi(tokenized_corpus)

    @classmethod
    def from_corpus_store(cls, corpus_store: CorpusStore) -> "BM25Retriever":
        """
        Build BM25 index from the existing CorpusStore.

        CorpusStore remains the source of truth for chunks and metadata.
        """

        chunk_ids = corpus_store.all_chunk_ids()
        chunk_texts = corpus_store.all_chunk_texts()

        return cls(chunk_ids=chunk_ids, chunk_texts=chunk_texts)

    @property
    def corpus_size(self) -> int:
        return len(self._chunk_ids)

    @property
    def retriever_name(self) -> str:
        return self._retriever_name

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Search the BM25 index and return ranked chunk IDs.

        Tie-breaking is deterministic:
            1. higher BM25 score first
            2. original corpus/index order second

        Returning chunk IDs only is intentional. Full paragraph text, title,
        and sentence metadata should be retrieved from CorpusStore.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        query_tokens = tokenize_text(query)

        if not query_tokens:
            raise ValueError("query must contain at least one searchable token.")

        scores = self._bm25.get_scores(query_tokens)

        ranked_items = [
            (index, float(score))
            for index, score in enumerate(scores)
        ]

        ranked_items.sort(key=lambda item: (-item[1], item[0]))

        limited_items = ranked_items[: min(top_k, len(ranked_items))]

        return [
            RetrievalResult(
                rank=rank,
                chunk_id=self._chunk_ids[index],
                score=score,
                retriever_name=self._retriever_name,
            )
            for rank, (index, score) in enumerate(limited_items, start=1)
        ]

    @staticmethod
    def _validate_corpus_inputs(
        chunk_ids: Sequence[str],
        chunk_texts: Sequence[str],
    ) -> None:
        if len(chunk_ids) == 0:
            raise ValueError("BM25 corpus cannot be empty.")

        if len(chunk_ids) != len(chunk_texts):
            raise ValueError(
                "BM25 corpus is misaligned: chunk_ids and chunk_texts must have the same length."
            )

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("BM25 corpus contains duplicate chunk_ids.")

        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Every chunk_id must be a non-empty string.")

        for chunk_text in chunk_texts:
            if not isinstance(chunk_text, str):
                raise ValueError("Every chunk_text must be a string.")