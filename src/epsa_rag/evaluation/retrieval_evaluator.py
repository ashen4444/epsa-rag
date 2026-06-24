from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from epsa_rag.evaluation.retrieval_metrics import (
    approximate_token_count,
    both_supporting_documents_found_at_k,
    first_support_rank,
    gold_title_ranks,
    mean_found_gold_rank,
    missing_gold_document_rate,
    mrr_at_k,
    ndcg_at_k,
    supporting_doc_recall_at_k,
    top1_supporting_hit,
)


DEFAULT_K_VALUES: tuple[int, ...] = (1, 2, 5, 10, 20)


@dataclass(frozen=True)
class RetrievalQuestion:
    question_id: str
    question: str
    gold_answer: str | None
    gold_supporting_titles: list[str]
    gold_supporting_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalEvaluationRecord:
    question_id: str
    question: str
    gold_answer: str | None
    gold_supporting_titles: list[str]
    gold_supporting_chunk_ids: list[str]

    retrieved_titles_top20: list[str]
    retrieved_chunk_ids_top20: list[str]
    retrieved_scores_top20: list[float | None]

    gold_title_ranks: dict[str, int | None]

    top1_supporting_hit: bool

    both_supporting_found_at_2: bool
    both_supporting_found_at_5: bool
    both_supporting_found_at_10: bool
    both_supporting_found_at_20: bool

    supporting_doc_recall_at_2: float
    supporting_doc_recall_at_5: float
    supporting_doc_recall_at_10: float
    supporting_doc_recall_at_20: float

    first_support_rank: int | None
    first_support_mrr_at_10: float
    ndcg_at_10: float

    mean_found_gold_rank: float | None
    missing_gold_document_rate_at_20: float

    latency_ms: float | None
    retrieved_chunks_count: int
    estimated_context_tokens_at_2: int
    estimated_context_tokens_at_5: int
    estimated_context_tokens_at_10: int
    estimated_context_tokens_at_20: int

    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    evaluated_questions: int
    skipped_questions: int

    top1_supporting_hit_rate: float

    both_supporting_found_rate_at_2: float
    both_supporting_found_rate_at_5: float
    both_supporting_found_rate_at_10: float
    both_supporting_found_rate_at_20: float

    mean_supporting_doc_recall_at_2: float
    mean_supporting_doc_recall_at_5: float
    mean_supporting_doc_recall_at_10: float
    mean_supporting_doc_recall_at_20: float

    mean_first_support_mrr_at_10: float
    mean_ndcg_at_10: float

    mean_found_gold_rank: float | None
    mean_missing_gold_document_rate_at_20: float

    average_latency_ms: float | None
    average_retrieved_chunks: float

    average_estimated_context_tokens_at_2: float
    average_estimated_context_tokens_at_5: float
    average_estimated_context_tokens_at_10: float
    average_estimated_context_tokens_at_20: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrievalEvaluator:
    """Evaluate title-level HotPotQA supporting-document retrieval quality."""

    def __init__(
        self,
        *,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        strict_two_supporting_docs: bool = True,
    ) -> None:
        self.k_values = tuple(sorted(set(k_values)))
        self.strict_two_supporting_docs = strict_two_supporting_docs

        if 20 not in self.k_values:
            raise ValueError("k_values must include 20 because top-20 is the diagnostic pool.")

    def evaluate_question(
        self,
        *,
        question: RetrievalQuestion,
        retrieved_results: Sequence[Any],
        chunk_lookup: Mapping[str, Mapping[str, Any]],
        latency_ms: float | None = None,
        max_k: int = 20,
    ) -> RetrievalEvaluationRecord:
        if not question.question_id:
            return self._skipped_record(
                question=question,
                reason="missing_question_id",
                latency_ms=latency_ms,
            )

        if not question.question:
            return self._skipped_record(
                question=question,
                reason="missing_question_text",
                latency_ms=latency_ms,
            )

        if not question.gold_supporting_titles:
            return self._skipped_record(
                question=question,
                reason="missing_gold_supporting_titles",
                latency_ms=latency_ms,
            )

        if self.strict_two_supporting_docs and len(set(question.gold_supporting_titles)) != 2:
            return self._skipped_record(
                question=question,
                reason="expected_exactly_two_gold_supporting_titles",
                latency_ms=latency_ms,
            )

        top_results = list(retrieved_results[:max_k])

        retrieved_chunk_ids: list[str] = []
        retrieved_titles: list[str] = []
        retrieved_scores: list[float | None] = []
        retrieved_texts: list[str] = []

        for result in top_results:
            chunk_id = _get_result_field(result, ("chunk_id", "document_id", "doc_id", "id"))
            score = _get_optional_float_field(
                result,
                ("score", "fusion_score", "rrf_score", "dense_score", "bm25_score"),
            )

            chunk: Mapping[str, Any] = {}
            if chunk_id and chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]

            title = (
                _get_result_field(result, ("doc_title", "title"))
                or _get_mapping_field(chunk, ("doc_title", "title"))
                or ""
            )

            chunk_text = (
                _get_result_field(result, ("chunk_text", "text", "paragraph_text"))
                or _get_mapping_field(chunk, ("chunk_text", "text", "paragraph_text"))
                or ""
            )

            retrieved_chunk_ids.append(str(chunk_id) if chunk_id is not None else "")
            retrieved_titles.append(str(title))
            retrieved_scores.append(score)
            retrieved_texts.append(str(chunk_text))

        rank_map = gold_title_ranks(
            retrieved_titles=retrieved_titles,
            gold_supporting_titles=question.gold_supporting_titles,
        )

        return RetrievalEvaluationRecord(
            question_id=question.question_id,
            question=question.question,
            gold_answer=question.gold_answer,
            gold_supporting_titles=list(question.gold_supporting_titles),
            gold_supporting_chunk_ids=list(question.gold_supporting_chunk_ids),
            retrieved_titles_top20=retrieved_titles,
            retrieved_chunk_ids_top20=retrieved_chunk_ids,
            retrieved_scores_top20=retrieved_scores,
            gold_title_ranks=rank_map,
            top1_supporting_hit=top1_supporting_hit(
                retrieved_titles,
                question.gold_supporting_titles,
            ),
            both_supporting_found_at_2=both_supporting_documents_found_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                2,
            ),
            both_supporting_found_at_5=both_supporting_documents_found_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                5,
            ),
            both_supporting_found_at_10=both_supporting_documents_found_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                10,
            ),
            both_supporting_found_at_20=both_supporting_documents_found_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                20,
            ),
            supporting_doc_recall_at_2=supporting_doc_recall_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                2,
            ),
            supporting_doc_recall_at_5=supporting_doc_recall_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                5,
            ),
            supporting_doc_recall_at_10=supporting_doc_recall_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                10,
            ),
            supporting_doc_recall_at_20=supporting_doc_recall_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                20,
            ),
            first_support_rank=first_support_rank(
                retrieved_titles,
                question.gold_supporting_titles,
                k=20,
            ),
            first_support_mrr_at_10=mrr_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                10,
            ),
            ndcg_at_10=ndcg_at_k(
                retrieved_titles,
                question.gold_supporting_titles,
                10,
            ),
            mean_found_gold_rank=mean_found_gold_rank(
                retrieved_titles,
                question.gold_supporting_titles,
            ),
            missing_gold_document_rate_at_20=missing_gold_document_rate(
                retrieved_titles,
                question.gold_supporting_titles,
                k=20,
            ),
            latency_ms=latency_ms,
            retrieved_chunks_count=len(top_results),
            estimated_context_tokens_at_2=_estimated_tokens_at_k(retrieved_texts, 2),
            estimated_context_tokens_at_5=_estimated_tokens_at_k(retrieved_texts, 5),
            estimated_context_tokens_at_10=_estimated_tokens_at_k(retrieved_texts, 10),
            estimated_context_tokens_at_20=_estimated_tokens_at_k(retrieved_texts, 20),
        )

    def summarize(
        self,
        records: Sequence[RetrievalEvaluationRecord],
    ) -> RetrievalEvaluationSummary:
        valid_records = [record for record in records if not record.skipped]
        skipped_count = len(records) - len(valid_records)

        if not valid_records:
            return RetrievalEvaluationSummary(
                evaluated_questions=0,
                skipped_questions=skipped_count,
                top1_supporting_hit_rate=0.0,
                both_supporting_found_rate_at_2=0.0,
                both_supporting_found_rate_at_5=0.0,
                both_supporting_found_rate_at_10=0.0,
                both_supporting_found_rate_at_20=0.0,
                mean_supporting_doc_recall_at_2=0.0,
                mean_supporting_doc_recall_at_5=0.0,
                mean_supporting_doc_recall_at_10=0.0,
                mean_supporting_doc_recall_at_20=0.0,
                mean_first_support_mrr_at_10=0.0,
                mean_ndcg_at_10=0.0,
                mean_found_gold_rank=None,
                mean_missing_gold_document_rate_at_20=0.0,
                average_latency_ms=None,
                average_retrieved_chunks=0.0,
                average_estimated_context_tokens_at_2=0.0,
                average_estimated_context_tokens_at_5=0.0,
                average_estimated_context_tokens_at_10=0.0,
                average_estimated_context_tokens_at_20=0.0,
            )

        found_gold_ranks = [
            record.mean_found_gold_rank
            for record in valid_records
            if record.mean_found_gold_rank is not None
        ]

        latencies = [
            record.latency_ms for record in valid_records if record.latency_ms is not None
        ]

        return RetrievalEvaluationSummary(
            evaluated_questions=len(valid_records),
            skipped_questions=skipped_count,
            top1_supporting_hit_rate=_mean_bool(
                record.top1_supporting_hit for record in valid_records
            ),
            both_supporting_found_rate_at_2=_mean_bool(
                record.both_supporting_found_at_2 for record in valid_records
            ),
            both_supporting_found_rate_at_5=_mean_bool(
                record.both_supporting_found_at_5 for record in valid_records
            ),
            both_supporting_found_rate_at_10=_mean_bool(
                record.both_supporting_found_at_10 for record in valid_records
            ),
            both_supporting_found_rate_at_20=_mean_bool(
                record.both_supporting_found_at_20 for record in valid_records
            ),
            mean_supporting_doc_recall_at_2=mean(
                record.supporting_doc_recall_at_2 for record in valid_records
            ),
            mean_supporting_doc_recall_at_5=mean(
                record.supporting_doc_recall_at_5 for record in valid_records
            ),
            mean_supporting_doc_recall_at_10=mean(
                record.supporting_doc_recall_at_10 for record in valid_records
            ),
            mean_supporting_doc_recall_at_20=mean(
                record.supporting_doc_recall_at_20 for record in valid_records
            ),
            mean_first_support_mrr_at_10=mean(
                record.first_support_mrr_at_10 for record in valid_records
            ),
            mean_ndcg_at_10=mean(record.ndcg_at_10 for record in valid_records),
            mean_found_gold_rank=mean(found_gold_ranks) if found_gold_ranks else None,
            mean_missing_gold_document_rate_at_20=mean(
                record.missing_gold_document_rate_at_20 for record in valid_records
            ),
            average_latency_ms=mean(latencies) if latencies else None,
            average_retrieved_chunks=mean(
                record.retrieved_chunks_count for record in valid_records
            ),
            average_estimated_context_tokens_at_2=mean(
                record.estimated_context_tokens_at_2 for record in valid_records
            ),
            average_estimated_context_tokens_at_5=mean(
                record.estimated_context_tokens_at_5 for record in valid_records
            ),
            average_estimated_context_tokens_at_10=mean(
                record.estimated_context_tokens_at_10 for record in valid_records
            ),
            average_estimated_context_tokens_at_20=mean(
                record.estimated_context_tokens_at_20 for record in valid_records
            ),
        )

    def _skipped_record(
        self,
        *,
        question: RetrievalQuestion,
        reason: str,
        latency_ms: float | None,
    ) -> RetrievalEvaluationRecord:
        return RetrievalEvaluationRecord(
            question_id=question.question_id,
            question=question.question,
            gold_answer=question.gold_answer,
            gold_supporting_titles=list(question.gold_supporting_titles),
            gold_supporting_chunk_ids=list(question.gold_supporting_chunk_ids),
            retrieved_titles_top20=[],
            retrieved_chunk_ids_top20=[],
            retrieved_scores_top20=[],
            gold_title_ranks={},
            top1_supporting_hit=False,
            both_supporting_found_at_2=False,
            both_supporting_found_at_5=False,
            both_supporting_found_at_10=False,
            both_supporting_found_at_20=False,
            supporting_doc_recall_at_2=0.0,
            supporting_doc_recall_at_5=0.0,
            supporting_doc_recall_at_10=0.0,
            supporting_doc_recall_at_20=0.0,
            first_support_rank=None,
            first_support_mrr_at_10=0.0,
            ndcg_at_10=0.0,
            mean_found_gold_rank=None,
            missing_gold_document_rate_at_20=0.0,
            latency_ms=latency_ms,
            retrieved_chunks_count=0,
            estimated_context_tokens_at_2=0,
            estimated_context_tokens_at_5=0,
            estimated_context_tokens_at_10=0,
            estimated_context_tokens_at_20=0,
            skipped=True,
            skip_reason=reason,
        )


def build_retrieval_questions_from_chunks(
    chunks: Iterable[Mapping[str, Any]],
    *,
    strict_two_supporting_docs: bool = True,
) -> list[RetrievalQuestion]:
    """Build question-level gold labels from processed HotPotQA paragraph chunks.

    This uses each chunk's `is_supporting_doc` only to construct gold labels for
    that chunk's own source question. The evaluator does not use retrieved
    chunks' `is_supporting_doc` flags as relevance labels.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}

    for chunk in chunks:
        question_id = _get_mapping_field(
            chunk,
            ("question_id", "source_question_id", "qid", "_id"),
        )
        if not question_id:
            continue

        grouped.setdefault(str(question_id), []).append(chunk)

    questions: list[RetrievalQuestion] = []

    for question_id, question_chunks in grouped.items():
        first_chunk = question_chunks[0]

        question_text = str(
            _get_mapping_field(first_chunk, ("question", "query", "question_text")) or ""
        )

        gold_answer_value = _get_mapping_field(first_chunk, ("answer", "gold_answer"))
        gold_answer = str(gold_answer_value) if gold_answer_value is not None else None

        gold_titles: list[str] = []
        gold_chunk_ids: list[str] = []

        for chunk in question_chunks:
            if bool(chunk.get("is_supporting_doc", False)):
                title = _get_mapping_field(chunk, ("doc_title", "title"))
                chunk_id = _get_mapping_field(chunk, ("chunk_id", "id"))

                if title and str(title) not in gold_titles:
                    gold_titles.append(str(title))

                if chunk_id and str(chunk_id) not in gold_chunk_ids:
                    gold_chunk_ids.append(str(chunk_id))

        if strict_two_supporting_docs and len(set(gold_titles)) != 2:
            # Keep the question. The evaluator will mark it skipped with a clear reason.
            pass

        questions.append(
            RetrievalQuestion(
                question_id=question_id,
                question=question_text,
                gold_answer=gold_answer,
                gold_supporting_titles=gold_titles,
                gold_supporting_chunk_ids=gold_chunk_ids,
            )
        )

    return questions


def build_chunk_lookup(
    chunks: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}

    for chunk in chunks:
        chunk_id = _get_mapping_field(chunk, ("chunk_id", "id"))
        if chunk_id:
            lookup[str(chunk_id)] = chunk

    return lookup


def write_retrieval_eval_outputs(
    *,
    output_dir: Path,
    run_name: str,
    records: Sequence[RetrievalEvaluationRecord],
    summary: RetrievalEvaluationSummary,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_question_path = output_dir / f"{run_name}_per_question.jsonl"
    summary_json_path = output_dir / f"{run_name}_summary.json"
    summary_md_path = output_dir / f"{run_name}_summary.md"

    with per_question_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary.to_dict(), file, ensure_ascii=False, indent=2)

    summary_md_path.write_text(
        _build_markdown_summary(run_name=run_name, summary=summary),
        encoding="utf-8",
    )

    return per_question_path, summary_json_path, summary_md_path


def _build_markdown_summary(
    *,
    run_name: str,
    summary: RetrievalEvaluationSummary,
) -> str:
    summary_dict = summary.to_dict()

    lines = [
        f"# Retrieval Evaluation Summary: `{run_name}`",
        "",
        "## Scope",
        "",
        "This report evaluates whether the Hybrid Retriever retrieves the current "
        "HotPotQA question's gold supporting document titles from the global indexed corpus.",
        "",
        "No EPSA, LLM answer generation, answer EM/F1, dynamic top-k, or LLM judging is included.",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for key, value in summary_dict.items():
        if isinstance(value, float):
            rendered_value = f"{value:.4f}"
        else:
            rendered_value = str(value)

        lines.append(f"| `{key}` | {rendered_value} |")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `Both Supporting Documents Found@5/@10` are the primary retrieval success metrics.",
            "- `Supporting Document Recall@5/@10` shows partial evidence coverage.",
            "- `Both Supporting Documents Found@20` is diagnostic: it shows whether gold evidence exists lower in the candidate pool.",
            "- Relevance is title-level and question-specific. Retrieved chunks are not counted as relevant merely because their own `is_supporting_doc` flag is true.",
            "",
        ]
    )

    return "\n".join(lines)


def _estimated_tokens_at_k(texts: Sequence[str], k: int) -> int:
    return sum(approximate_token_count(text) for text in texts[:k])


def _mean_bool(values: Iterable[bool]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0

    return sum(1 for value in values_list if value) / len(values_list)


def _get_result_field(result: Any, names: Sequence[str]) -> Any:
    if isinstance(result, Mapping):
        return _get_mapping_field(result, names)

    for name in names:
        if hasattr(result, name):
            return getattr(result, name)

    return None


def _get_mapping_field(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]

    return None


def _get_optional_float_field(result: Any, names: Sequence[str]) -> float | None:
    value = _get_result_field(result, names)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None