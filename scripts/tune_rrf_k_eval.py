from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epsa_rag.config.retrieval_config import load_retrieval_settings
from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
    RetrievalQuestion,
)
from epsa_rag.retrieval.bm25_retriever import BM25Retriever
from epsa_rag.retrieval.dense_retriever import DenseRetriever
from epsa_rag.retrieval.embedding_backend import OpenAITextEmbedder
from epsa_rag.retrieval.fusion import reciprocal_rank_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune RRF k values for EPSA-RAG hybrid retrieval."
    )

    parser.add_argument(
        "--rrf-ks",
        type=str,
        default="10,20,30,60,80",
        help="Comma-separated RRF k values to evaluate. Default: 10,20,30,60,80.",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=20,
        help="Final fused retrieval depth. Default: 20.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of questions for smoke testing.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_eval" / "rrf_tuning",
        help="Output directory for RRF tuning results.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rrf_ks = parse_rrf_ks(args.rrf_ks)

    if args.max_k < 20:
        raise ValueError("--max-k must be at least 20.")

    settings = load_retrieval_settings()

    corpus_path = resolve_project_path(settings.paths.processed_corpus)
    dense_index_path = resolve_project_path(settings.paths.dense_index)
    dense_metadata_path = resolve_project_path(settings.paths.dense_metadata)

    print("Loading CorpusStore...")
    corpus_store = CorpusStore.from_jsonl(corpus_path)

    print("Building BM25Retriever...")
    bm25_retriever = BM25Retriever.from_corpus_store(corpus_store)

    print("Loading OpenAI query embedder...")
    embedder = OpenAITextEmbedder(
        model_name_or_path=settings.dense.model_name,
        batch_size=settings.dense.batch_size,
    )

    print("Loading DenseRetriever from existing FAISS index...")
    dense_retriever = DenseRetriever.load(
        corpus_store=corpus_store,
        index_path=dense_index_path,
        metadata_path=dense_metadata_path,
        embedder=embedder,
    )

    all_chunks = corpus_store.all_chunks()
    chunk_lookup = build_chunk_lookup(all_chunks)

    questions = build_retrieval_questions_from_corpus_chunks(
        all_chunks,
        strict_two_supporting_docs=True,
    )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1.")
        questions = questions[: args.limit]

    evaluator = RetrievalEvaluator(
        k_values=(1, 2, 5, 10, 20),
        strict_two_supporting_docs=True,
    )

    print()
    print("RRF tuning configuration")
    print("------------------------")
    print(f"Corpus path:       {corpus_path}")
    print(f"Dense index path:  {dense_index_path}")
    print(f"Dense metadata:    {dense_metadata_path}")
    print(f"Questions:         {len(questions)}")
    print(f"BM25 top-k:        {settings.retrieval.bm25_top_k}")
    print(f"Dense top-k:       {settings.retrieval.dense_top_k}")
    print(f"Final max-k:       {args.max_k}")
    print(f"RRF k candidates:  {rrf_ks}")
    print(f"Embedding model:   {embedder.model_name}")
    print()

    records_by_rrf_k: dict[int, list[Any]] = {rrf_k: [] for rrf_k in rrf_ks}
    retrieval_latencies_ms: list[float] = []

    for question_index, question in enumerate(questions, start=1):
        start_time = time.perf_counter()

        bm25_results = bm25_retriever.search(
            query=question.question,
            top_k=int(settings.retrieval.bm25_top_k),
        )

        dense_results = dense_retriever.search(
            query=question.question,
            top_k=int(settings.retrieval.dense_top_k),
        )

        retrieval_latency_ms = (time.perf_counter() - start_time) * 1000.0
        retrieval_latencies_ms.append(retrieval_latency_ms)

        print(
            f"[{question_index:>4}/{len(questions)}] "
            f"question_id={question.question_id} "
            f"source_retrieval_latency_ms={retrieval_latency_ms:.2f}"
        )

        for rrf_k in rrf_ks:
            fused_results = reciprocal_rank_fusion(
                bm25_results=bm25_results,
                dense_results=dense_results,
                final_top_k=args.max_k,
                rrf_k=rrf_k,
                retriever_name=f"hybrid_rrf_{rrf_k}",
            )

            record = evaluator.evaluate_question(
                question=question,
                retrieved_results=fused_results,
                chunk_lookup=chunk_lookup,
                latency_ms=retrieval_latency_ms,
                max_k=args.max_k,
            )

            records_by_rrf_k[rrf_k].append(record)

    summaries = []

    for rrf_k in rrf_ks:
        summary = evaluator.summarize(records_by_rrf_k[rrf_k])
        summary_dict = summary.to_dict()
        summary_dict["rrf_k"] = rrf_k
        summaries.append(summary_dict)

    ranked_summaries = sorted(
        summaries,
        key=lambda item: (
            -float(item["both_supporting_found_rate_at_10"]),
            -float(item["mean_supporting_doc_recall_at_10"]),
            -float(item["mean_ndcg_at_10"]),
            -float(item["mean_first_support_mrr_at_10"]),
            -float(item["both_supporting_found_rate_at_5"]),
            int(item["rrf_k"]),
        ),
    )

    best_summary = ranked_summaries[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rrf_tuning_{timestamp}"

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / f"{run_name}_summary.json"
    summary_csv_path = output_dir / f"{run_name}_summary.csv"
    per_question_jsonl_path = output_dir / f"{run_name}_per_question.jsonl"
    markdown_path = output_dir / f"{run_name}_summary.md"

    write_summary_json(summary_json_path, summaries)
    write_summary_csv(summary_csv_path, summaries)
    write_per_question_jsonl(per_question_jsonl_path, records_by_rrf_k)
    write_markdown_report(
        path=markdown_path,
        summaries=summaries,
        ranked_summaries=ranked_summaries,
        best_summary=best_summary,
        rrf_ks=rrf_ks,
        total_questions=len(questions),
        source_average_latency_ms=average(retrieval_latencies_ms),
    )

    print()
    print("RRF tuning complete.")
    print("--------------------")
    print(f"Summary JSON:       {summary_json_path}")
    print(f"Summary CSV:        {summary_csv_path}")
    print(f"Per-question JSONL: {per_question_jsonl_path}")
    print(f"Markdown report:    {markdown_path}")
    print()
    print("Best candidate by ranking rule")
    print("------------------------------")
    print(f"rrf_k:        {best_summary['rrf_k']}")
    print(f"Both Found@5: {float(best_summary['both_supporting_found_rate_at_5']):.4f}")
    print(f"Both Found@10:{float(best_summary['both_supporting_found_rate_at_10']):.4f}")
    print(f"Recall@5:     {float(best_summary['mean_supporting_doc_recall_at_5']):.4f}")
    print(f"Recall@10:    {float(best_summary['mean_supporting_doc_recall_at_10']):.4f}")
    print(f"nDCG@10:      {float(best_summary['mean_ndcg_at_10']):.4f}")
    print(f"MRR@10:       {float(best_summary['mean_first_support_mrr_at_10']):.4f}")


def parse_rrf_ks(raw_value: str) -> list[int]:
    values: list[int] = []

    for item in raw_value.split(","):
        stripped = item.strip()

        if not stripped:
            continue

        value = int(stripped)

        if value <= 0:
            raise ValueError("Every rrf_k value must be greater than 0.")

        values.append(value)

    if not values:
        raise ValueError("At least one rrf_k value is required.")

    return sorted(set(values))


def build_chunk_lookup(chunks: Iterable[Any]) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}

    for chunk in chunks:
        chunk_dict = chunk_to_dict(chunk)
        chunk_id = get_field(chunk_dict, "chunk_id")

        if chunk_id:
            lookup[str(chunk_id)] = chunk_dict

    return lookup


def build_retrieval_questions_from_corpus_chunks(
    chunks: Sequence[Any],
    *,
    strict_two_supporting_docs: bool = True,
) -> list[RetrievalQuestion]:
    grouped_chunks: dict[str, list[Any]] = defaultdict(list)

    for chunk in chunks:
        source_question_id = get_field(
            chunk,
            "source_question_id",
            "question_id",
            "qid",
            "_id",
        )

        if not source_question_id:
            continue

        grouped_chunks[str(source_question_id)].append(chunk)

    questions: list[RetrievalQuestion] = []

    for source_question_id, question_chunks in grouped_chunks.items():
        first_chunk = question_chunks[0]

        question_text = str(
            get_field(first_chunk, "question", "question_text", "query") or ""
        )

        answer_value = get_field(first_chunk, "answer", "gold_answer")
        gold_answer = str(answer_value) if answer_value is not None else None

        gold_supporting_titles: list[str] = []
        gold_supporting_chunk_ids: list[str] = []

        for chunk in question_chunks:
            is_supporting_doc = bool(get_field(chunk, "is_supporting_doc") or False)

            if not is_supporting_doc:
                continue

            doc_title = get_field(chunk, "doc_title", "title")
            chunk_id = get_field(chunk, "chunk_id", "id")

            if doc_title and str(doc_title) not in gold_supporting_titles:
                gold_supporting_titles.append(str(doc_title))

            if chunk_id and str(chunk_id) not in gold_supporting_chunk_ids:
                gold_supporting_chunk_ids.append(str(chunk_id))

        if strict_two_supporting_docs and len(set(gold_supporting_titles)) != 2:
            # Keep the question. RetrievalEvaluator will mark it skipped clearly.
            pass

        questions.append(
            RetrievalQuestion(
                question_id=str(source_question_id),
                question=question_text,
                gold_answer=gold_answer,
                gold_supporting_titles=gold_supporting_titles,
                gold_supporting_chunk_ids=gold_supporting_chunk_ids,
            )
        )

    questions.sort(key=lambda item: item.question_id)
    return questions


def chunk_to_dict(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return dict(chunk)

    if hasattr(chunk, "model_dump"):
        dumped = chunk.model_dump()
        if isinstance(dumped, dict):
            return dumped

    if hasattr(chunk, "dict"):
        dumped = chunk.dict()
        if isinstance(dumped, dict):
            return dumped

    if hasattr(chunk, "__dict__"):
        return dict(chunk.__dict__)

    raise TypeError(f"Cannot convert chunk object to dict: {type(chunk)!r}")


def get_field(obj: Any, *field_names: str) -> Any:
    if isinstance(obj, Mapping):
        for field_name in field_names:
            if field_name in obj:
                return obj[field_name]

        return None

    for field_name in field_names:
        if hasattr(obj, field_name):
            return getattr(obj, field_name)

    return None


def write_summary_json(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return

    fieldnames = [
        "rrf_k",
        "evaluated_questions",
        "skipped_questions",
        "top1_supporting_hit_rate",
        "both_supporting_found_rate_at_2",
        "both_supporting_found_rate_at_5",
        "both_supporting_found_rate_at_10",
        "both_supporting_found_rate_at_20",
        "mean_supporting_doc_recall_at_2",
        "mean_supporting_doc_recall_at_5",
        "mean_supporting_doc_recall_at_10",
        "mean_supporting_doc_recall_at_20",
        "mean_first_support_mrr_at_10",
        "mean_ndcg_at_10",
        "mean_found_gold_rank",
        "mean_missing_gold_document_rate_at_20",
        "average_latency_ms",
        "average_retrieved_chunks",
        "average_estimated_context_tokens_at_2",
        "average_estimated_context_tokens_at_5",
        "average_estimated_context_tokens_at_10",
        "average_estimated_context_tokens_at_20",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fieldnames})


def write_per_question_jsonl(
    path: Path,
    records_by_rrf_k: dict[int, list[Any]],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for rrf_k, records in records_by_rrf_k.items():
            for record in records:
                payload = record.to_dict()
                payload["rrf_k"] = rrf_k
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_markdown_report(
    *,
    path: Path,
    summaries: list[dict[str, Any]],
    ranked_summaries: list[dict[str, Any]],
    best_summary: dict[str, Any],
    rrf_ks: list[int],
    total_questions: int,
    source_average_latency_ms: float | None,
) -> None:
    lines: list[str] = [
        "# RRF k Tuning Evaluation",
        "",
        "## Scope",
        "",
        (
            "This report evaluates different Reciprocal Rank Fusion `rrf_k` values "
            "using the same BM25 and dense retrieval results per question."
        ),
        "",
        "This is a retrieval-ranking tuning experiment only. It does not include EPSA, "
        "LLM answer generation, dynamic top-k, or answer evaluation.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---:|",
        f"| Total questions | {total_questions} |",
        f"| RRF k candidates | `{rrf_ks}` |",
        f"| Average source retrieval latency ms | {format_optional_float(source_average_latency_ms)} |",
        "",
        "## Best Candidate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Best rrf_k | {best_summary['rrf_k']} |",
        f"| Both Found@5 | {float(best_summary['both_supporting_found_rate_at_5']):.4f} |",
        f"| Both Found@10 | {float(best_summary['both_supporting_found_rate_at_10']):.4f} |",
        f"| Recall@5 | {float(best_summary['mean_supporting_doc_recall_at_5']):.4f} |",
        f"| Recall@10 | {float(best_summary['mean_supporting_doc_recall_at_10']):.4f} |",
        f"| nDCG@10 | {float(best_summary['mean_ndcg_at_10']):.4f} |",
        f"| MRR@10 | {float(best_summary['mean_first_support_mrr_at_10']):.4f} |",
        "",
        "## Candidate Comparison",
        "",
        (
            "| rrf_k | Top1 Hit | Both@5 | Both@10 | Recall@5 | Recall@10 | "
            "Both@20 | Recall@20 | MRR@10 | nDCG@10 |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in sorted(summaries, key=lambda item: int(item["rrf_k"])):
        lines.append(
            f"| {summary['rrf_k']} "
            f"| {float(summary['top1_supporting_hit_rate']):.4f} "
            f"| {float(summary['both_supporting_found_rate_at_5']):.4f} "
            f"| {float(summary['both_supporting_found_rate_at_10']):.4f} "
            f"| {float(summary['mean_supporting_doc_recall_at_5']):.4f} "
            f"| {float(summary['mean_supporting_doc_recall_at_10']):.4f} "
            f"| {float(summary['both_supporting_found_rate_at_20']):.4f} "
            f"| {float(summary['mean_supporting_doc_recall_at_20']):.4f} "
            f"| {float(summary['mean_first_support_mrr_at_10']):.4f} "
            f"| {float(summary['mean_ndcg_at_10']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Ranking Rule",
            "",
            "The best candidate is selected by this priority order:",
            "",
            "```text",
            "1. Highest Both Found@10",
            "2. Highest Recall@10",
            "3. Highest nDCG@10",
            "4. Highest MRR@10",
            "5. Highest Both Found@5",
            "6. Smaller rrf_k as tie-breaker",
            "```",
            "",
            "## Ranked Candidates",
            "",
            "| Rank | rrf_k | Both@10 | Recall@10 | nDCG@10 | MRR@10 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for rank, summary in enumerate(ranked_summaries, start=1):
        lines.append(
            f"| {rank} "
            f"| {summary['rrf_k']} "
            f"| {float(summary['both_supporting_found_rate_at_10']):.4f} "
            f"| {float(summary['mean_supporting_doc_recall_at_10']):.4f} "
            f"| {float(summary['mean_ndcg_at_10']):.4f} "
            f"| {float(summary['mean_first_support_mrr_at_10']):.4f} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def average(values: Sequence[float]) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "-"

    return f"{value:.4f}"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    return PROJECT_ROOT / candidate


if __name__ == "__main__":
    main()