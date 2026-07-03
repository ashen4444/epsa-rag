from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epsa_rag.epsa.epsa_controller import EPSAController
from epsa_rag.evaluation.answer_metrics import (
    answer_overlap_metrics,
    exact_match_score,
    partial_match_score,
)
from epsa_rag.evaluation.epsa_rag_metrics import summarize_epsa_rag_records
from epsa_rag.rag.llm_client import ChatLLM, OpenAIChatClient
from epsa_rag.rag.prompt_templates import build_final_answer_messages
from epsa_rag.rag.two_hop_baseline import (
    RAGDocument,
    document_from_chunk,
    estimate_token_count,
    extract_chunk_id,
    format_documents_for_prompt,
    read_field,
)


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    question: str
    gold_answer: str | None
    gold_supporting_titles: list[str]


@dataclass(frozen=True)
class EPSARAGConfig:
    hop1_top_k: int = 10
    hop2_top_k: int = 10
    temperature: float = 0.0
    final_answer_max_tokens: int = 24
    max_paths: int = 10
    insufficient_fallback_strategy: str = "fixed"
    insufficient_fallback_doc_limit: int = 8
    adaptive_fallback_high_confidence_threshold: float = 0.48
    adaptive_fallback_medium_confidence_threshold: float = 0.42
    adaptive_fallback_high_confidence_limit: int = 8
    adaptive_fallback_medium_confidence_limit: int = 10
    adaptive_fallback_low_confidence_limit: int = 12


@dataclass(frozen=True)
class RetrievedCandidate:
    document: RAGDocument
    epsa_chunk: dict[str, Any]


class EPSAControlledRAGRunner:
    """Run EPSA-controlled adaptive RAG over a fixed external retriever.

    The runner may call retrieval and final answer generation because it is an
    experiment pipeline. EPSA modules remain post-retrieval and deterministic.
    """

    def __init__(
        self,
        *,
        retriever: Any,
        corpus_store: Any,
        llm_client: ChatLLM,
        epsa_controller: EPSAController | None = None,
        config: EPSARAGConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.corpus_store = corpus_store
        self.llm_client = llm_client
        self.epsa_controller = epsa_controller or EPSAController()
        self.config = config or EPSARAGConfig()

    def run(
        self,
        *,
        question_id: str,
        question: str,
        gold_answer: str | None = None,
        gold_supporting_titles: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()

        try:
            hop1_candidates = retrieve_candidates(
                retriever=self.retriever,
                corpus_store=self.corpus_store,
                query=question,
                top_k=self.config.hop1_top_k,
            )
        except Exception as exc:
            return build_error_record(
                question_id=question_id,
                question=question,
                gold_answer=gold_answer,
                latency_ms=elapsed_ms(started_at),
                retrieval_error=f"hop1_retrieval_error: {exc}",
                gold_supporting_titles=gold_supporting_titles,
            )

        hop1_epsa_result = None
        final_epsa_result = None
        next_hop_query = None
        hop2_candidates: list[RetrievedCandidate] = []
        merged_candidates = list(hop1_candidates)
        epsa_error: str | None = None
        adaptive_stop_after_hop = "1_no_query"

        try:
            hop1_epsa_result = self.epsa_controller.run(
                question,
                [candidate.epsa_chunk for candidate in hop1_candidates],
                max_paths=self.config.max_paths,
                metadata={"epsa_pass": "hop1"},
            )

            next_hop_query = hop1_epsa_result.next_hop_query

            if hop1_epsa_result.sufficient:
                final_epsa_result = hop1_epsa_result
                adaptive_stop_after_hop = "1"
            elif next_hop_query is not None and next_hop_query.query:
                hop2_candidates = retrieve_candidates(
                    retriever=self.retriever,
                    corpus_store=self.corpus_store,
                    query=next_hop_query.query,
                    top_k=self.config.hop2_top_k,
                )
                merged_candidates = merge_retrieved_candidates(hop1_candidates, hop2_candidates)
                final_epsa_result = self.epsa_controller.run(
                    question,
                    [candidate.epsa_chunk for candidate in merged_candidates],
                    max_paths=self.config.max_paths,
                    metadata={"epsa_pass": "merged_hop1_hop2"},
                )
                adaptive_stop_after_hop = "2"
            else:
                final_epsa_result = hop1_epsa_result
                adaptive_stop_after_hop = "1_no_query"

        except Exception as exc:
            epsa_error = str(exc)
            final_epsa_result = hop1_epsa_result
            adaptive_stop_after_hop = "epsa_error_fallback"

        fallback_docs = [candidate.document for candidate in merged_candidates or hop1_candidates]
        resolved_fallback_doc_limit = resolve_insufficient_fallback_doc_limit(
            epsa_result=final_epsa_result,
            insufficient_fallback_strategy=self.config.insufficient_fallback_strategy,
            insufficient_fallback_doc_limit=self.config.insufficient_fallback_doc_limit,
            adaptive_fallback_high_confidence_threshold=(
                self.config.adaptive_fallback_high_confidence_threshold
            ),
            adaptive_fallback_medium_confidence_threshold=(
                self.config.adaptive_fallback_medium_confidence_threshold
            ),
            adaptive_fallback_high_confidence_limit=(
                self.config.adaptive_fallback_high_confidence_limit
            ),
            adaptive_fallback_medium_confidence_limit=(
                self.config.adaptive_fallback_medium_confidence_limit
            ),
            adaptive_fallback_low_confidence_limit=(
                self.config.adaptive_fallback_low_confidence_limit
            ),
        )
        final_fallback_docs = bound_fallback_documents_for_context(
            epsa_result=final_epsa_result,
            fallback_documents=fallback_docs,
            insufficient_fallback_doc_limit=self.config.insufficient_fallback_doc_limit,
            insufficient_fallback_strategy=self.config.insufficient_fallback_strategy,
            adaptive_fallback_high_confidence_threshold=(
                self.config.adaptive_fallback_high_confidence_threshold
            ),
            adaptive_fallback_medium_confidence_threshold=(
                self.config.adaptive_fallback_medium_confidence_threshold
            ),
            adaptive_fallback_high_confidence_limit=(
                self.config.adaptive_fallback_high_confidence_limit
            ),
            adaptive_fallback_medium_confidence_limit=(
                self.config.adaptive_fallback_medium_confidence_limit
            ),
            adaptive_fallback_low_confidence_limit=(
                self.config.adaptive_fallback_low_confidence_limit
            ),
        )
        final_context, context_source = choose_context_for_final_answer(
            epsa_result=final_epsa_result,
            fallback_documents=final_fallback_docs,
        )
        estimated_context_tokens = estimate_token_count(final_context)

        prompt_tokens = 0
        completion_tokens = 0
        total_llm_tokens = 0
        model_name: str | None = None
        predicted_answer = ""
        final_answer_error: str | None = None

        try:
            answer_response = self.llm_client.complete(
                build_final_answer_messages(question, final_context),
                temperature=self.config.temperature,
                max_tokens=self.config.final_answer_max_tokens,
            )
            predicted_answer = answer_response.content.strip()
            prompt_tokens = answer_response.prompt_tokens
            completion_tokens = answer_response.completion_tokens
            total_llm_tokens = answer_response.total_tokens
            model_name = answer_response.model_name
        except Exception as exc:
            final_answer_error = str(exc)

        answer_overlap = answer_overlap_metrics(predicted_answer, gold_answer)
        exact_match = exact_match_score(predicted_answer, gold_answer) if gold_answer else 0.0
        partial_match = partial_match_score(predicted_answer, gold_answer) if gold_answer else 0.0

        selected_chunk_ids = selected_chunk_ids_from_epsa(final_epsa_result)
        selected_evidence_unit_ids = selected_evidence_unit_ids_from_epsa(final_epsa_result)
        hop1_chunk_ids = [candidate.document.chunk_id for candidate in hop1_candidates]
        hop2_chunk_ids = [candidate.document.chunk_id for candidate in hop2_candidates]
        merged_chunk_ids = [candidate.document.chunk_id for candidate in merged_candidates]
        final_context_documents = documents_for_final_context_diagnostics(
            context_source=context_source,
            final_fallback_documents=final_fallback_docs,
            merged_candidates=merged_candidates,
            selected_chunk_ids=selected_chunk_ids,
        )
        gold_title_diagnostics = build_gold_title_diagnostics(
            gold_supporting_titles=gold_supporting_titles or [],
            hop1_candidates=hop1_candidates,
            hop2_candidates=hop2_candidates,
            merged_candidates=merged_candidates,
            selected_chunk_ids=selected_chunk_ids,
            final_context_documents=final_context_documents,
            context_source=context_source,
        )

        hop1_sufficient = bool(hop1_epsa_result.sufficient) if hop1_epsa_result is not None else False
        final_sufficient = bool(final_epsa_result.sufficient) if final_epsa_result is not None else False

        return {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "exact_match": exact_match,
            "partial_match": partial_match,
            "answer_precision": answer_overlap.precision if gold_answer else 0.0,
            "answer_recall": answer_overlap.recall if gold_answer else 0.0,
            "answer_f1": answer_overlap.f1 if gold_answer else 0.0,
            "hop1_retrieved_count": len(hop1_candidates),
            "hop2_retrieved_count": len(hop2_candidates),
            "merged_retrieved_count": len(merged_candidates),
            "epsa_hop1_sufficient": hop1_sufficient,
            "epsa_final_sufficient": final_sufficient,
            "adaptive_stop_after_hop": adaptive_stop_after_hop,
            "insufficient_fallback_strategy": normalize_insufficient_fallback_strategy(
                self.config.insufficient_fallback_strategy
            ),
            "resolved_insufficient_fallback_doc_limit": resolved_fallback_doc_limit,
            "selected_context_docs": count_selected_context_docs(
                final_epsa_result,
                final_fallback_docs,
                context_source,
            ),
            "selected_context_sentences": count_selected_context_sentences(final_epsa_result, context_source),
            "estimated_context_tokens": estimated_context_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_llm_tokens": total_llm_tokens,
            "latency_ms": elapsed_ms(started_at),
            "context_source": context_source,
            "next_hop_query": getattr(next_hop_query, "query", None),
            "next_hop_query_type": getattr(next_hop_query, "query_type", None),
            "next_hop_query_confidence": getattr(next_hop_query, "confidence", 0.0) or 0.0,
            "sufficiency_confidence": sufficiency_field(final_epsa_result, "confidence", 0.0),
            "missing_evidence": sufficiency_field(final_epsa_result, "missing_evidence", None),
            "decision_reason": sufficiency_field(final_epsa_result, "decision_reason", None),
            "answer_candidate": sufficiency_field(final_epsa_result, "answer_candidate", None),
            "answer_type": sufficiency_field(final_epsa_result, "answer_type", None),
            "selected_chunk_ids": serialize_list_for_csv(selected_chunk_ids),
            "selected_evidence_unit_ids": serialize_list_for_csv(selected_evidence_unit_ids),
            "hop1_retrieved_chunk_ids": serialize_list_for_csv(hop1_chunk_ids),
            "hop2_retrieved_chunk_ids": serialize_list_for_csv(hop2_chunk_ids),
            "merged_retrieved_chunk_ids": serialize_list_for_csv(merged_chunk_ids),
            **gold_title_diagnostics,
            "potential_false_sufficient_candidate": is_potential_false_sufficient(
                epsa_sufficient=final_sufficient,
                exact_match=exact_match,
                partial_match=partial_match,
                final_answer_generation_failed=bool(final_answer_error),
            ),
            "potential_false_insufficient_candidate": is_potential_false_insufficient(
                hop1_sufficient=hop1_sufficient,
                adaptive_stop_after_hop=adaptive_stop_after_hop,
                final_sufficient=final_sufficient,
                selected_chunk_ids=selected_chunk_ids,
                hop1_chunk_ids=hop1_chunk_ids,
            ),
            "retrieval_failed": False,
            "final_answer_generation_failed": bool(final_answer_error),
            "epsa_failed": bool(epsa_error),
            "error_message": join_error_messages(epsa_error, final_answer_error),
            "model_name": model_name,
        }


def retrieve_candidates(
    *,
    retriever: Any,
    corpus_store: Any,
    query: str,
    top_k: int,
) -> list[RetrievedCandidate]:
    raw_results = retriever.search(query, top_k=top_k)

    if hasattr(raw_results, "results"):
        raw_results = raw_results.results

    candidates: list[RetrievedCandidate] = []

    for fallback_rank, result in enumerate(raw_results, start=1):
        chunk_id = extract_chunk_id(result)
        chunk = corpus_store.get_chunk(chunk_id)
        document = document_from_chunk(chunk_id, chunk, result, fallback_rank)
        epsa_chunk = chunk_to_epsa_input(chunk, result, fallback_rank)
        candidates.append(RetrievedCandidate(document=document, epsa_chunk=epsa_chunk))

    return candidates


def chunk_to_epsa_input(chunk: Any, retrieval_result: Any, fallback_rank: int) -> dict[str, Any]:
    payload = object_to_dict(chunk)
    rank = read_field(retrieval_result, "rank", default=fallback_rank)
    score = read_field(
        retrieval_result,
        "fusion_score",
        "score",
        "dense_score",
        "bm25_score",
        default=None,
    )

    payload["rank"] = int(rank) if rank is not None else fallback_rank
    payload["retrieval_rank"] = int(rank) if rank is not None else fallback_rank

    if score is not None:
        payload["score"] = float(score)
        payload["retrieval_score"] = float(score)

    # EPSA analyzers accept both source_question_id and question_id.
    if "question_id" not in payload and "source_question_id" in payload:
        payload["question_id"] = payload["source_question_id"]

    return payload


def object_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)

    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)

    if hasattr(obj, "dict"):
        dumped = obj.dict()
        if isinstance(dumped, dict):
            return dict(dumped)

    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)

    raise TypeError(f"Cannot convert chunk object to dict: {type(obj)!r}")


def merge_retrieved_candidates(
    first: Sequence[RetrievedCandidate],
    second: Sequence[RetrievedCandidate],
) -> list[RetrievedCandidate]:
    seen: set[str] = set()
    merged: list[RetrievedCandidate] = []

    for group in (first, second):
        for candidate in group:
            chunk_id = candidate.document.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(candidate)

    return merged



def normalize_title_for_matching(title: Any) -> str:
    """Normalize document titles for deterministic gold-title matching."""

    return " ".join(str(title or "").casefold().strip().split())


def document_title(document: RAGDocument) -> str:
    """Return a stable title string from a RAG document."""

    return str(getattr(document, "title", "") or "")


def candidate_document_title(candidate: RetrievedCandidate) -> str:
    title = document_title(candidate.document)
    if title:
        return title

    return str(
        candidate.epsa_chunk.get("doc_title")
        or candidate.epsa_chunk.get("title")
        or ""
    )


def unique_titles_for_matching(titles: Sequence[str] | None) -> list[str]:
    return unique_preserve_order(str(title) for title in titles or [] if title is not None)


def matching_gold_titles(
    documents: Sequence[RAGDocument],
    gold_titles: Sequence[str] | None,
) -> list[str]:
    """Return gold titles whose normalized title appears in the given documents."""

    normalized_document_titles = {
        normalize_title_for_matching(document_title(document))
        for document in documents
    }
    normalized_document_titles.discard("")

    matches: list[str] = []
    for gold_title in unique_titles_for_matching(gold_titles):
        if normalize_title_for_matching(gold_title) in normalized_document_titles:
            matches.append(gold_title)

    return matches


def candidate_documents(candidates: Sequence[RetrievedCandidate]) -> list[RAGDocument]:
    return [candidate.document for candidate in candidates]


def documents_for_selected_chunk_ids(
    candidates: Sequence[RetrievedCandidate],
    selected_chunk_ids: Sequence[str],
) -> list[RAGDocument]:
    selected = {str(chunk_id) for chunk_id in selected_chunk_ids if chunk_id}
    if not selected:
        return []

    documents: list[RAGDocument] = []
    seen: set[str] = set()
    for candidate in candidates:
        chunk_id = str(candidate.document.chunk_id)
        if chunk_id not in selected or chunk_id in seen:
            continue
        documents.append(candidate.document)
        seen.add(chunk_id)

    return documents


def documents_for_final_context_diagnostics(
    *,
    context_source: str,
    final_fallback_documents: Sequence[RAGDocument],
    merged_candidates: Sequence[RetrievedCandidate],
    selected_chunk_ids: Sequence[str],
) -> list[RAGDocument]:
    if context_source == "epsa_pruned_context":
        return documents_for_selected_chunk_ids(merged_candidates, selected_chunk_ids)

    if context_source in {"epsa_insufficient_fallback_documents", "fallback_documents"}:
        return list(final_fallback_documents)

    return []


def best_gold_title_rank(
    candidates: Sequence[RetrievedCandidate],
    gold_titles: Sequence[str] | None,
) -> int | None:
    """Return the best 1-based candidate-pool rank containing any gold title."""

    normalized_gold_titles = {
        normalize_title_for_matching(title)
        for title in unique_titles_for_matching(gold_titles)
    }
    normalized_gold_titles.discard("")

    if not normalized_gold_titles:
        return None

    for fallback_rank, candidate in enumerate(candidates, start=1):
        if normalize_title_for_matching(candidate_document_title(candidate)) in normalized_gold_titles:
            return fallback_rank

    return None


def gold_title_coverage_status(
    *,
    gold_supporting_title_count: int,
    gold_titles_in_merged_count: int,
    gold_titles_selected_by_epsa_count: int,
    gold_titles_in_final_context_count: int,
    context_source: str,
) -> str:
    if gold_supporting_title_count <= 0:
        return "no_gold_titles_available"

    if gold_titles_in_merged_count <= 0:
        return "gold_not_retrieved"

    if (
        context_source == "epsa_insufficient_fallback_documents"
        and gold_titles_in_final_context_count > gold_titles_selected_by_epsa_count
    ):
        return "fallback_context_contains_gold"

    if gold_titles_selected_by_epsa_count >= gold_supporting_title_count:
        return "all_gold_selected"

    if gold_titles_selected_by_epsa_count > 0:
        return "partial_gold_selected"

    if gold_titles_in_merged_count >= gold_supporting_title_count:
        return "all_gold_retrieved_not_selected"

    if gold_titles_in_merged_count > 0:
        return "partial_gold_retrieved"

    return "unknown"


def build_gold_title_diagnostics(
    *,
    gold_supporting_titles: Sequence[str] | None,
    hop1_candidates: Sequence[RetrievedCandidate],
    hop2_candidates: Sequence[RetrievedCandidate],
    merged_candidates: Sequence[RetrievedCandidate],
    selected_chunk_ids: Sequence[str],
    final_context_documents: Sequence[RAGDocument],
    context_source: str,
) -> dict[str, Any]:
    gold_titles = unique_titles_for_matching(gold_supporting_titles)
    hop1_gold_titles = matching_gold_titles(candidate_documents(hop1_candidates), gold_titles)
    hop2_gold_titles = matching_gold_titles(candidate_documents(hop2_candidates), gold_titles)
    merged_gold_titles = matching_gold_titles(candidate_documents(merged_candidates), gold_titles)
    selected_documents = documents_for_selected_chunk_ids(merged_candidates, selected_chunk_ids)
    selected_gold_titles = matching_gold_titles(selected_documents, gold_titles)
    final_context_gold_titles = matching_gold_titles(final_context_documents, gold_titles)
    missing_from_merged = [
        title
        for title in gold_titles
        if normalize_title_for_matching(title)
        not in {normalize_title_for_matching(value) for value in merged_gold_titles}
    ]

    status = gold_title_coverage_status(
        gold_supporting_title_count=len(gold_titles),
        gold_titles_in_merged_count=len(merged_gold_titles),
        gold_titles_selected_by_epsa_count=len(selected_gold_titles),
        gold_titles_in_final_context_count=len(final_context_gold_titles),
        context_source=context_source,
    )

    return {
        "gold_supporting_title_count": len(gold_titles),
        "gold_titles_in_hop1_count": len(hop1_gold_titles),
        "gold_titles_in_hop2_count": len(hop2_gold_titles),
        "gold_titles_in_merged_count": len(merged_gold_titles),
        "gold_titles_selected_by_epsa_count": len(selected_gold_titles),
        "gold_titles_in_final_context_count": len(final_context_gold_titles),
        "gold_titles_missing_from_merged_count": len(missing_from_merged),
        "gold_titles_in_hop1": serialize_list_for_csv(hop1_gold_titles),
        "gold_titles_in_hop2": serialize_list_for_csv(hop2_gold_titles),
        "gold_titles_in_merged": serialize_list_for_csv(merged_gold_titles),
        "gold_titles_selected_by_epsa": serialize_list_for_csv(selected_gold_titles),
        "gold_titles_in_final_context": serialize_list_for_csv(final_context_gold_titles),
        "gold_titles_missing_from_merged": serialize_list_for_csv(missing_from_merged),
        "gold_title_best_rank": best_gold_title_rank(merged_candidates, gold_titles),
        "gold_title_coverage_status": status,
    }


def bound_fallback_documents_for_context(
    *,
    epsa_result: Any | None,
    fallback_documents: Sequence[RAGDocument],
    insufficient_fallback_doc_limit: int | None,
    insufficient_fallback_strategy: str = "fixed",
    adaptive_fallback_high_confidence_threshold: float = 0.48,
    adaptive_fallback_medium_confidence_threshold: float = 0.42,
    adaptive_fallback_high_confidence_limit: int = 8,
    adaptive_fallback_medium_confidence_limit: int = 10,
    adaptive_fallback_low_confidence_limit: int = 12,
) -> list[RAGDocument]:
    """Limit fallback context only when EPSA explicitly marks evidence insufficient.

    Fixed strategy preserves the Chat 17 bounded fallback behavior. Adaptive
    strategy keeps the conservative sufficient/insufficient decision unchanged,
    but varies the insufficient fallback size by deterministic EPSA confidence.
    """

    documents = list(fallback_documents)
    doc_limit = resolve_insufficient_fallback_doc_limit(
        epsa_result=epsa_result,
        insufficient_fallback_strategy=insufficient_fallback_strategy,
        insufficient_fallback_doc_limit=insufficient_fallback_doc_limit,
        adaptive_fallback_high_confidence_threshold=adaptive_fallback_high_confidence_threshold,
        adaptive_fallback_medium_confidence_threshold=adaptive_fallback_medium_confidence_threshold,
        adaptive_fallback_high_confidence_limit=adaptive_fallback_high_confidence_limit,
        adaptive_fallback_medium_confidence_limit=adaptive_fallback_medium_confidence_limit,
        adaptive_fallback_low_confidence_limit=adaptive_fallback_low_confidence_limit,
    )

    if doc_limit is None or doc_limit <= 0:
        return documents

    return documents[:doc_limit]


def resolve_insufficient_fallback_doc_limit(
    *,
    epsa_result: Any | None,
    insufficient_fallback_strategy: str = "fixed",
    insufficient_fallback_doc_limit: int | None = 8,
    adaptive_fallback_high_confidence_threshold: float = 0.48,
    adaptive_fallback_medium_confidence_threshold: float = 0.42,
    adaptive_fallback_high_confidence_limit: int = 8,
    adaptive_fallback_medium_confidence_limit: int = 10,
    adaptive_fallback_low_confidence_limit: int = 12,
) -> int | None:
    """Resolve the fallback document limit for the current EPSA result.

    Returns None when fallback bounding should not apply. That is intentional
    for EPSA-sufficient results, because sufficient cases must use EPSA-pruned
    context rather than an insufficient-fallback budget.
    """

    if not _epsa_result_explicitly_insufficient(epsa_result):
        return None

    strategy = normalize_insufficient_fallback_strategy(insufficient_fallback_strategy)

    if strategy == "fixed":
        return insufficient_fallback_doc_limit

    confidence = sufficiency_confidence_from_epsa(epsa_result)

    if confidence >= adaptive_fallback_high_confidence_threshold:
        return adaptive_fallback_high_confidence_limit
    if confidence >= adaptive_fallback_medium_confidence_threshold:
        return adaptive_fallback_medium_confidence_limit
    return adaptive_fallback_low_confidence_limit


def normalize_insufficient_fallback_strategy(value: str | None) -> str:
    strategy = (value or "fixed").strip().casefold()
    if strategy not in {"fixed", "adaptive"}:
        raise ValueError(
            "insufficient_fallback_strategy must be either 'fixed' or 'adaptive'."
        )
    return strategy


def sufficiency_confidence_from_epsa(epsa_result: Any | None) -> float:
    if epsa_result is None:
        return 0.0

    decision = getattr(epsa_result, "sufficiency_decision", None)
    if decision is not None:
        return safe_float(getattr(decision, "confidence", 0.0), default=0.0)

    return safe_float(getattr(epsa_result, "confidence", 0.0), default=0.0)


def safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_context_for_final_answer(
    *,
    epsa_result: Any | None,
    fallback_documents: Sequence[RAGDocument],
) -> tuple[str, str]:
    """Choose final-answer context with a conservative insufficient fallback.

    EPSA-pruned evidence is safe to send directly when EPSA is sufficient.
    When EPSA explicitly says the evidence is insufficient, sending only its
    partial pruned evidence can starve the final answer LLM. In that case, use
    the merged retrieved documents as a bounded fallback while preserving EPSA's
    sufficiency decision in the logged record.
    """

    if _epsa_result_explicitly_insufficient(epsa_result) and fallback_documents:
        return (
            format_documents_for_prompt(fallback_documents),
            "epsa_insufficient_fallback_documents",
        )

    pruned_context = getattr(epsa_result, "pruned_context", None)
    selected_context_text = getattr(pruned_context, "selected_context_text", "") or ""

    if selected_context_text.strip():
        return selected_context_text.strip(), "epsa_pruned_context"

    return format_documents_for_prompt(fallback_documents), "fallback_documents"


def _epsa_result_explicitly_insufficient(epsa_result: Any | None) -> bool:
    if epsa_result is None:
        return False

    if getattr(epsa_result, "sufficient", None) is False:
        return True

    decision = getattr(epsa_result, "sufficiency_decision", None)
    return getattr(decision, "sufficient", None) is False


def count_selected_context_docs(
    epsa_result: Any | None,
    fallback_documents: Sequence[RAGDocument],
    context_source: str,
) -> int:
    if context_source == "epsa_pruned_context":
        return len(selected_chunk_ids_from_epsa(epsa_result))

    return len(fallback_documents)


def count_selected_context_sentences(epsa_result: Any | None, context_source: str) -> int:
    if context_source != "epsa_pruned_context":
        return 0

    pruned_context = getattr(epsa_result, "pruned_context", None)
    return len(getattr(pruned_context, "selected_sentences", []) or [])


def selected_chunk_ids_from_epsa(epsa_result: Any | None) -> list[str]:
    if epsa_result is None:
        return []

    values = getattr(epsa_result, "selected_chunk_ids", None)
    if values is not None:
        return [str(value) for value in values if value]

    pruned_context = getattr(epsa_result, "pruned_context", None)
    return [str(value) for value in getattr(pruned_context, "selected_chunk_ids", []) or [] if value]


def selected_evidence_unit_ids_from_epsa(epsa_result: Any | None) -> list[str]:
    if epsa_result is None:
        return []

    values = getattr(epsa_result, "selected_evidence_unit_ids", None)
    if values is not None:
        return [str(value) for value in values if value]

    pruned_context = getattr(epsa_result, "pruned_context", None)
    return [
        str(value)
        for value in getattr(pruned_context, "selected_evidence_unit_ids", []) or []
        if value
    ]


def sufficiency_field(epsa_result: Any | None, field_name: str, default: Any = None) -> Any:
    decision = getattr(epsa_result, "sufficiency_decision", None)
    if decision is None:
        return default

    return getattr(decision, field_name, default)


def serialize_list_for_csv(values: Sequence[Any]) -> str:
    return json.dumps([str(value) for value in values], ensure_ascii=False)


def is_potential_false_sufficient(
    *,
    epsa_sufficient: bool,
    exact_match: float,
    partial_match: float,
    final_answer_generation_failed: bool = False,
) -> bool:
    if final_answer_generation_failed:
        return False

    return bool(epsa_sufficient and float(exact_match) == 0.0 and float(partial_match) == 0.0)


def is_potential_false_insufficient(
    *,
    hop1_sufficient: bool,
    adaptive_stop_after_hop: str,
    final_sufficient: bool,
    selected_chunk_ids: Sequence[str],
    hop1_chunk_ids: Sequence[str],
) -> bool:
    if hop1_sufficient or adaptive_stop_after_hop != "2" or not final_sufficient:
        return False

    selected = {str(value) for value in selected_chunk_ids if value}
    hop1 = {str(value) for value in hop1_chunk_ids if value}

    return bool(selected and selected.issubset(hop1))


def build_error_record(
    *,
    question_id: str,
    question: str,
    gold_answer: str | None,
    latency_ms: float,
    retrieval_error: str | None = None,
    epsa_error: str | None = None,
    final_answer_error: str | None = None,
    gold_supporting_titles: Sequence[str] | None = None,
) -> dict[str, Any]:
    gold_title_diagnostics = build_gold_title_diagnostics(
        gold_supporting_titles=gold_supporting_titles or [],
        hop1_candidates=[],
        hop2_candidates=[],
        merged_candidates=[],
        selected_chunk_ids=[],
        final_context_documents=[],
        context_source="none",
    )

    return {
        "question_id": question_id,
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": "",
        "exact_match": 0.0,
        "partial_match": 0.0,
        "answer_precision": 0.0,
        "answer_recall": 0.0,
        "answer_f1": 0.0,
        "hop1_retrieved_count": 0,
        "hop2_retrieved_count": 0,
        "merged_retrieved_count": 0,
        "epsa_hop1_sufficient": False,
        "epsa_final_sufficient": False,
        "adaptive_stop_after_hop": "error",
        "insufficient_fallback_strategy": "fixed",
        "resolved_insufficient_fallback_doc_limit": None,
        "selected_context_docs": 0,
        "selected_context_sentences": 0,
        "estimated_context_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_llm_tokens": 0,
        "latency_ms": latency_ms,
        "context_source": "none",
        "next_hop_query": None,
        "next_hop_query_type": None,
        "next_hop_query_confidence": 0.0,
        "sufficiency_confidence": 0.0,
        "missing_evidence": None,
        "decision_reason": None,
        "answer_candidate": None,
        "answer_type": None,
        "selected_chunk_ids": serialize_list_for_csv([]),
        "selected_evidence_unit_ids": serialize_list_for_csv([]),
        "hop1_retrieved_chunk_ids": serialize_list_for_csv([]),
        "hop2_retrieved_chunk_ids": serialize_list_for_csv([]),
        "merged_retrieved_chunk_ids": serialize_list_for_csv([]),
        **gold_title_diagnostics,
        "potential_false_sufficient_candidate": False,
        "potential_false_insufficient_candidate": False,
        "retrieval_failed": bool(retrieval_error),
        "final_answer_generation_failed": bool(final_answer_error),
        "epsa_failed": bool(epsa_error),
        "error_message": join_error_messages(retrieval_error, epsa_error, final_answer_error),
        "model_name": None,
    }


def join_error_messages(*messages: str | None) -> str:
    return " | ".join(message for message in messages if message) or ""


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_store, retriever = build_hybrid_retriever(args)
    questions = load_question_records(Path(args.questions_path), limit=args.limit)

    llm_client = OpenAIChatClient(
        model_name=args.llm_model,
        timeout=args.llm_timeout,
    )

    runner = EPSAControlledRAGRunner(
        retriever=retriever,
        corpus_store=corpus_store,
        llm_client=llm_client,
        epsa_controller=EPSAController(),
        config=EPSARAGConfig(
            hop1_top_k=args.hop1_top_k,
            hop2_top_k=args.hop2_top_k,
            temperature=args.temperature,
            final_answer_max_tokens=args.final_answer_max_tokens,
            max_paths=args.epsa_max_paths,
            insufficient_fallback_strategy=args.insufficient_fallback_strategy,
            insufficient_fallback_doc_limit=args.insufficient_fallback_doc_limit,
            adaptive_fallback_high_confidence_threshold=(
                args.adaptive_fallback_high_confidence_threshold
            ),
            adaptive_fallback_medium_confidence_threshold=(
                args.adaptive_fallback_medium_confidence_threshold
            ),
            adaptive_fallback_high_confidence_limit=args.adaptive_fallback_high_confidence_limit,
            adaptive_fallback_medium_confidence_limit=args.adaptive_fallback_medium_confidence_limit,
            adaptive_fallback_low_confidence_limit=args.adaptive_fallback_low_confidence_limit,
        ),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records_path = output_dir / f"epsa_rag_results_{timestamp}.csv"
    summary_json_path = output_dir / f"epsa_rag_summary_{timestamp}.json"
    summary_md_path = output_dir / f"epsa_rag_summary_{timestamp}.md"

    records: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        record = runner.run(
            question_id=item.question_id,
            question=item.question,
            gold_answer=item.gold_answer,
            gold_supporting_titles=item.gold_supporting_titles,
        )
        records.append(record)

        print(
            f"[{index}/{len(questions)}] "
            f"{item.question_id} | "
            f"stop={record['adaptive_stop_after_hop']} "
            f"EM={float(record['exact_match']):.0f} "
            f"F1={float(record['answer_f1']):.3f} "
            f"context_docs={record['selected_context_docs']} "
            f"tokens={record['total_llm_tokens']} "
            f"latency_ms={float(record['latency_ms']):.1f}"
        )

    write_csv_records(records_path, records)

    summary = summarize_epsa_rag_records(records)
    summary["config"] = {
        "questions_path": str(args.questions_path),
        "corpus_path": str(args.corpus_path),
        "retrieval_config": str(args.retrieval_config),
        "dense_index_path": str(args.dense_index_path),
        "dense_metadata_path": str(args.dense_metadata_path),
        "output_dir": str(args.output_dir),
        "limit": args.limit,
        "hop1_top_k": args.hop1_top_k,
        "hop2_top_k": args.hop2_top_k,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "temperature": args.temperature,
        "final_answer_max_tokens": args.final_answer_max_tokens,
        "epsa_max_paths": args.epsa_max_paths,
        "insufficient_fallback_strategy": args.insufficient_fallback_strategy,
        "insufficient_fallback_doc_limit": args.insufficient_fallback_doc_limit,
        "adaptive_fallback_high_confidence_threshold": (
            args.adaptive_fallback_high_confidence_threshold
        ),
        "adaptive_fallback_medium_confidence_threshold": (
            args.adaptive_fallback_medium_confidence_threshold
        ),
        "adaptive_fallback_high_confidence_limit": args.adaptive_fallback_high_confidence_limit,
        "adaptive_fallback_medium_confidence_limit": args.adaptive_fallback_medium_confidence_limit,
        "adaptive_fallback_low_confidence_limit": args.adaptive_fallback_low_confidence_limit,
    }

    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_md_path.write_text(build_markdown_summary(summary), encoding="utf-8")

    print("\nSaved EPSA RAG records:", records_path)
    print("Saved EPSA RAG summary JSON:", summary_json_path)
    print("Saved EPSA RAG summary Markdown:", summary_md_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EPSA-controlled adaptive Hybrid RAG evaluation."
    )

    parser.add_argument("--questions-path", default="data/processed/hotpotqa_questions.jsonl")
    parser.add_argument("--corpus-path", default="data/processed/hotpotqa_paragraph_chunks.jsonl")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")

    parser.add_argument("--dense-index-path", default="data/indexes/dense/faiss_index.bin")
    parser.add_argument("--dense-metadata-path", default="data/indexes/dense/dense_metadata.json")

    parser.add_argument("--output-dir", default="outputs/epsa_rag")
    parser.add_argument("--limit", type=int, default=100)

    parser.add_argument("--hop1-top-k", type=int, default=10)
    parser.add_argument("--hop2-top-k", type=int, default=10)
    parser.add_argument("--epsa-max-paths", type=int, default=10)
    parser.add_argument(
        "--insufficient-fallback-strategy",
        choices=("fixed", "adaptive"),
        default="fixed",
        help=(
            "Fallback strategy when EPSA marks evidence insufficient. "
            "'fixed' preserves Chat 17 behavior. 'adaptive' varies the bounded "
            "fallback size using deterministic EPSA sufficiency confidence."
        ),
    )
    parser.add_argument(
        "--insufficient-fallback-doc-limit",
        type=int,
        default=8,
        help=(
            "Maximum number of retrieved documents to send when EPSA marks evidence "
            "insufficient and --insufficient-fallback-strategy=fixed. Use 0 or a "
            "negative value to disable fixed-strategy bounding."
        ),
    )
    parser.add_argument(
        "--adaptive-fallback-high-confidence-threshold",
        type=float,
        default=0.48,
        help="Use the high-confidence adaptive fallback limit at or above this confidence.",
    )
    parser.add_argument(
        "--adaptive-fallback-medium-confidence-threshold",
        type=float,
        default=0.42,
        help="Use the medium-confidence adaptive fallback limit at or above this confidence.",
    )
    parser.add_argument(
        "--adaptive-fallback-high-confidence-limit",
        type=int,
        default=8,
        help="Fallback document limit for near-sufficient insufficient EPSA decisions.",
    )
    parser.add_argument(
        "--adaptive-fallback-medium-confidence-limit",
        type=int,
        default=10,
        help="Fallback document limit for medium-confidence insufficient EPSA decisions.",
    )
    parser.add_argument(
        "--adaptive-fallback-low-confidence-limit",
        type=int,
        default=12,
        help="Fallback document limit for low-confidence insufficient EPSA decisions.",
    )

    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-batch-size", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--final-answer-max-tokens", type=int, default=24)
    parser.add_argument("--llm-timeout", type=float, default=60.0)

    return parser.parse_args()


def build_hybrid_retriever(args: argparse.Namespace) -> tuple[Any, Any]:
    require_file(Path(args.questions_path), "questions path")
    require_file(Path(args.corpus_path), "corpus path")
    require_file(Path(args.retrieval_config), "retrieval config")
    require_file(Path(args.dense_index_path), "dense index path")
    require_file(Path(args.dense_metadata_path), "dense metadata path")

    load_retrieval_settings = resolve_symbol(
        "epsa_rag.config.retrieval_config",
        "load_retrieval_settings",
    )
    CorpusStore = resolve_symbol("epsa_rag.corpus.corpus_store", "CorpusStore")
    BM25Retriever = resolve_symbol("epsa_rag.retrieval.bm25_retriever", "BM25Retriever")
    DenseRetriever = resolve_symbol("epsa_rag.retrieval.dense_retriever", "DenseRetriever")
    HybridRetriever = resolve_symbol("epsa_rag.retrieval.hybrid_retriever", "HybridRetriever")
    OpenAITextEmbedder = resolve_symbol(
        "epsa_rag.retrieval.embedding_backend",
        "OpenAITextEmbedder",
    )

    settings = load_retrieval_settings(Path(args.retrieval_config))
    corpus_store = CorpusStore.from_jsonl(Path(args.corpus_path))
    bm25_retriever = BM25Retriever.from_corpus_store(corpus_store)

    embedder = OpenAITextEmbedder(
        model_name_or_path=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )

    dense_retriever = DenseRetriever.load(
        corpus_store=corpus_store,
        index_path=Path(args.dense_index_path),
        metadata_path=Path(args.dense_metadata_path),
        embedder=embedder,
    )

    hybrid_retriever = HybridRetriever.from_settings(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        settings=settings,
    )

    return corpus_store, hybrid_retriever


def resolve_symbol(module_name: str, symbol_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}. Pass the correct path with the matching CLI argument."
        )


def load_question_records(path: Path, *, limit: int | None) -> list[QuestionRecord]:
    records: list[QuestionRecord] = []

    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue

            raw = json.loads(line)
            records.append(parse_question_record(raw, line_number=line_number))

            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(f"No question records loaded from {path}")

    return records


def parse_question_record(raw: dict[str, Any], *, line_number: int) -> QuestionRecord:
    question_id = first_present(raw, "question_id", "id", "_id", default=f"line_{line_number}")
    question = first_present(raw, "question", "query", default=None)

    if not question:
        raise ValueError(f"Question text missing in record at line {line_number}")

    gold_answer = first_present(raw, "gold_answer", "answer", default=None)
    supporting_titles = extract_supporting_titles(raw)

    return QuestionRecord(
        question_id=str(question_id),
        question=str(question),
        gold_answer=str(gold_answer) if gold_answer is not None else None,
        gold_supporting_titles=supporting_titles,
    )


def first_present(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]

    return default


def extract_supporting_titles(raw: dict[str, Any]) -> list[str]:
    for key in ("gold_supporting_titles", "supporting_titles", "supporting_doc_titles"):
        value = raw.get(key)
        if isinstance(value, list):
            return unique_preserve_order(str(item) for item in value if item is not None)

    supporting_facts = raw.get("supporting_facts") or raw.get("gold_supporting_facts")
    titles: list[str] = []

    if isinstance(supporting_facts, list):
        for fact in supporting_facts:
            if isinstance(fact, (list, tuple)) and fact:
                titles.append(str(fact[0]))
            elif isinstance(fact, dict):
                title = fact.get("title") or fact.get("doc_title")
                if title is not None:
                    titles.append(str(title))

    return unique_preserve_order(titles)


def unique_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        key = " ".join(str(value).casefold().strip().split())
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(str(value))

    return output


def write_csv_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# EPSA RAG Evaluation Summary",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    skip_keys = {"baseline_reference", "config"}
    for key, value in summary.items():
        if key in skip_keys:
            continue
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        lines.append(f"| `{key}` | {rendered} |")

    lines.extend(
        [
            "",
            "## Baseline Reference",
            "",
            "| Baseline Metric | Value |",
            "|---|---:|",
        ]
    )

    for key, value in dict(summary.get("baseline_reference", {})).items():
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {rendered} |")

    lines.extend(
        [
            "",
            "## Method Note",
            "",
            "The fixed retriever is shared with the baseline. EPSA controls adaptive stopping, next-hop query generation, and context pruning after retrieval.",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
