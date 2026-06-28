from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from epsa_rag.evaluation.answer_metrics import (
    answer_overlap_metrics,
    exact_match_score,
    partial_match_score,
)
from epsa_rag.rag.llm_client import ChatLLM
from epsa_rag.rag.prompt_templates import (
    build_final_answer_messages,
    build_hop2_query_messages,
)


@dataclass(frozen=True)
class TwoHopBaselineConfig:
    hop1_top_k: int = 5
    hop2_top_k: int = 5
    temperature: float = 0.0
    hop2_query_max_tokens: int = 48
    final_answer_max_tokens: int = 24


@dataclass(frozen=True)
class RAGDocument:
    chunk_id: str
    title: str
    text: str
    paragraph_index: int | None = None
    rank: int | None = None
    score: float | None = None


@dataclass(frozen=True)
class TwoHopBaselineResult:
    question_id: str
    question: str
    gold_answer: str | None

    hop1_query: str
    hop1_retrieved_chunk_ids: list[str]
    hop1_retrieved_titles: list[str]

    generated_hop2_query: str
    hop2_retrieved_chunk_ids: list[str]
    hop2_retrieved_titles: list[str]

    merged_context_chunk_ids: list[str]
    merged_context_titles: list[str]
    num_context_documents: int

    final_answer: str
    exact_match: float
    partial_match: float
    answer_precision: float
    answer_recall: float
    answer_f1: float

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_context_tokens: int
    latency_ms: float

    hop2_query_generation_error: str | None = None
    final_answer_generation_error: str | None = None
    retrieval_error: str | None = None
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TwoHopHybridRAGBaseline:
    def __init__(
        self,
        *,
        retriever: Any,
        corpus_store: Any,
        llm_client: ChatLLM,
        config: TwoHopBaselineConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.corpus_store = corpus_store
        self.llm_client = llm_client
        self.config = config or TwoHopBaselineConfig()

    def run(
        self,
        *,
        question_id: str,
        question: str,
        gold_answer: str | None = None,
        gold_supporting_titles: Sequence[str] | None = None,
    ) -> TwoHopBaselineResult:
        """
        Runs the fixed 2-hop Hybrid RAG baseline.

        gold_supporting_titles is accepted for compatibility with HotPotQA records,
        but supporting-document metrics are intentionally not logged in this
        end-to-end RAG baseline. Retrieval quality is evaluated separately.
        """
        _ = gold_supporting_titles

        started_at = time.perf_counter()

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        model_name: str | None = None

        try:
            hop1_docs = self._retrieve_documents(question, self.config.hop1_top_k)
        except Exception as exc:
            return self._error_result(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                latency_ms=self._elapsed_ms(started_at),
                retrieval_error=f"hop1_retrieval_error: {exc}",
            )

        hop1_context = format_documents_for_prompt(hop1_docs)

        try:
            hop2_response = self.llm_client.complete(
                build_hop2_query_messages(question, hop1_context),
                temperature=self.config.temperature,
                max_tokens=self.config.hop2_query_max_tokens,
            )

            prompt_tokens += hop2_response.prompt_tokens
            completion_tokens += hop2_response.completion_tokens
            total_tokens += hop2_response.total_tokens
            model_name = hop2_response.model_name or model_name

            hop2_query = sanitize_generated_query(hop2_response.content)
            if not hop2_query:
                raise ValueError("LLM returned an empty Hop-2 query.")

        except Exception as exc:
            return self._error_result(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                hop1_docs=hop1_docs,
                latency_ms=self._elapsed_ms(started_at),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                hop2_query_generation_error=str(exc),
                model_name=model_name,
            )

        try:
            hop2_docs = self._retrieve_documents(hop2_query, self.config.hop2_top_k)
        except Exception as exc:
            return self._error_result(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                hop1_docs=hop1_docs,
                generated_hop2_query=hop2_query,
                latency_ms=self._elapsed_ms(started_at),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                retrieval_error=f"hop2_retrieval_error: {exc}",
                model_name=model_name,
            )

        merged_docs = merge_unique_documents(hop1_docs, hop2_docs)
        merged_context = format_documents_for_prompt(merged_docs)
        estimated_context_tokens = estimate_token_count(merged_context)

        try:
            answer_response = self.llm_client.complete(
                build_final_answer_messages(question, merged_context),
                temperature=self.config.temperature,
                max_tokens=self.config.final_answer_max_tokens,
            )

            prompt_tokens += answer_response.prompt_tokens
            completion_tokens += answer_response.completion_tokens
            total_tokens += answer_response.total_tokens
            model_name = answer_response.model_name or model_name
            final_answer = answer_response.content.strip()

        except Exception as exc:
            return TwoHopBaselineResult(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                hop1_query=question,
                hop1_retrieved_chunk_ids=[doc.chunk_id for doc in hop1_docs],
                hop1_retrieved_titles=[doc.title for doc in hop1_docs],
                generated_hop2_query=hop2_query,
                hop2_retrieved_chunk_ids=[doc.chunk_id for doc in hop2_docs],
                hop2_retrieved_titles=[doc.title for doc in hop2_docs],
                merged_context_chunk_ids=[doc.chunk_id for doc in merged_docs],
                merged_context_titles=[doc.title for doc in merged_docs],
                num_context_documents=len(merged_docs),
                final_answer="",
                exact_match=0.0,
                partial_match=0.0,
                answer_precision=0.0,
                answer_recall=0.0,
                answer_f1=0.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_context_tokens=estimated_context_tokens,
                latency_ms=self._elapsed_ms(started_at),
                final_answer_generation_error=str(exc),
                model_name=model_name,
            )

        answer_overlap = answer_overlap_metrics(final_answer, gold_answer)

        return TwoHopBaselineResult(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            hop1_query=question,
            hop1_retrieved_chunk_ids=[doc.chunk_id for doc in hop1_docs],
            hop1_retrieved_titles=[doc.title for doc in hop1_docs],
            generated_hop2_query=hop2_query,
            hop2_retrieved_chunk_ids=[doc.chunk_id for doc in hop2_docs],
            hop2_retrieved_titles=[doc.title for doc in hop2_docs],
            merged_context_chunk_ids=[doc.chunk_id for doc in merged_docs],
            merged_context_titles=[doc.title for doc in merged_docs],
            num_context_documents=len(merged_docs),
            final_answer=final_answer,
            exact_match=exact_match_score(final_answer, gold_answer) if gold_answer else 0.0,
            partial_match=partial_match_score(final_answer, gold_answer) if gold_answer else 0.0,
            answer_precision=answer_overlap.precision if gold_answer else 0.0,
            answer_recall=answer_overlap.recall if gold_answer else 0.0,
            answer_f1=answer_overlap.f1 if gold_answer else 0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_context_tokens=estimated_context_tokens,
            latency_ms=self._elapsed_ms(started_at),
            model_name=model_name,
        )

    def _retrieve_documents(self, query: str, top_k: int) -> list[RAGDocument]:
        raw_results = self.retriever.search(query, top_k=top_k)

        if hasattr(raw_results, "results"):
            raw_results = raw_results.results

        documents: list[RAGDocument] = []

        for fallback_rank, result in enumerate(raw_results, start=1):
            chunk_id = extract_chunk_id(result)
            chunk = self.corpus_store.get_chunk(chunk_id)
            documents.append(document_from_chunk(chunk_id, chunk, result, fallback_rank))

        return documents

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000.0, 3)

    def _error_result(
        self,
        *,
        question_id: str,
        question: str,
        gold_answer: str | None,
        latency_ms: float,
        hop1_docs: Sequence[RAGDocument] | None = None,
        generated_hop2_query: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        hop2_query_generation_error: str | None = None,
        final_answer_generation_error: str | None = None,
        retrieval_error: str | None = None,
        model_name: str | None = None,
    ) -> TwoHopBaselineResult:
        hop1_docs = list(hop1_docs or [])
        context = format_documents_for_prompt(hop1_docs) if hop1_docs else ""

        return TwoHopBaselineResult(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            hop1_query=question,
            hop1_retrieved_chunk_ids=[doc.chunk_id for doc in hop1_docs],
            hop1_retrieved_titles=[doc.title for doc in hop1_docs],
            generated_hop2_query=generated_hop2_query,
            hop2_retrieved_chunk_ids=[],
            hop2_retrieved_titles=[],
            merged_context_chunk_ids=[doc.chunk_id for doc in hop1_docs],
            merged_context_titles=[doc.title for doc in hop1_docs],
            num_context_documents=len(hop1_docs),
            final_answer="",
            exact_match=0.0,
            partial_match=0.0,
            answer_precision=0.0,
            answer_recall=0.0,
            answer_f1=0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_context_tokens=estimate_token_count(context),
            latency_ms=latency_ms,
            hop2_query_generation_error=hop2_query_generation_error,
            final_answer_generation_error=final_answer_generation_error,
            retrieval_error=retrieval_error,
            model_name=model_name,
        )


def document_from_chunk(
    chunk_id: str,
    chunk: Any,
    retrieval_result: Any,
    fallback_rank: int,
) -> RAGDocument:
    title = str(read_field(chunk, "doc_title", "title", "document_title", default=""))

    chunk_text = read_field(chunk, "chunk_text", "text", default=None)
    paragraph_text = read_field(chunk, "paragraph_text", "paragraph", default=None)

    if chunk_text:
        text = str(chunk_text)
    elif paragraph_text:
        text = f"Title: {title}\nParagraph: {paragraph_text}" if title else str(paragraph_text)
    else:
        text = ""

    paragraph_index = read_field(chunk, "paragraph_index", "paragraph_id", default=None)

    rank = read_field(retrieval_result, "rank", default=fallback_rank)

    score = read_field(
        retrieval_result,
        "fusion_score",
        "score",
        "dense_score",
        "bm25_score",
        default=None,
    )

    return RAGDocument(
        chunk_id=chunk_id,
        title=title,
        text=text,
        paragraph_index=int(paragraph_index) if paragraph_index is not None else None,
        rank=int(rank) if rank is not None else fallback_rank,
        score=float(score) if score is not None else None,
    )


def extract_chunk_id(result: Any) -> str:
    chunk_id = read_field(result, "chunk_id", "id", "document_id", default=None)

    if chunk_id is None:
        raise ValueError(f"Retrieval result does not contain a chunk_id: {result!r}")

    return str(chunk_id)


def read_field(obj: Any, *field_names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in field_names:
            if name in obj:
                return obj[name]

    for name in field_names:
        if hasattr(obj, name):
            return getattr(obj, name)

    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        for name in field_names:
            if name in dumped:
                return dumped[name]

    return default


def merge_unique_documents(*document_groups: Sequence[RAGDocument]) -> list[RAGDocument]:
    seen: set[str] = set()
    merged: list[RAGDocument] = []

    for group in document_groups:
        for doc in group:
            if doc.chunk_id in seen:
                continue

            seen.add(doc.chunk_id)
            merged.append(doc)

    return merged


def format_documents_for_prompt(documents: Sequence[RAGDocument]) -> str:
    blocks: list[str] = []

    for index, doc in enumerate(documents, start=1):
        title_line = f"Title: {doc.title}" if doc.title else "Title: "

        blocks.append(
            f"[Document {index}]\n"
            f"Chunk ID: {doc.chunk_id}\n"
            f"{title_line}\n"
            f"Text:\n{doc.text.strip()}"
        )

    return "\n\n".join(blocks)


def sanitize_generated_query(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:text)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.strip('"').strip("'").strip()

    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")

    return first_line[:300]


def estimate_token_count(text: str) -> int:
    if not text:
        return 0

    return int(math.ceil(len(text) / 4.0))