from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
    write_retrieval_eval_outputs,
)
from epsa_rag.retrieval.bm25_retriever import BM25Retriever
from epsa_rag.retrieval.dense_retriever import DenseRetriever
from epsa_rag.retrieval.embedding_backend import OpenAITextEmbedder
from epsa_rag.retrieval.hybrid_retriever import HybridRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the EPSA-RAG Hybrid Retriever on HotPotQA supporting "
            "document title retrieval."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of questions to evaluate for a smoke test.",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=20,
        help="Maximum retrieved results per question. Default: 20.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_eval",
        help="Directory where retrieval evaluation outputs are saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_k < 20:
        raise ValueError(
            "--max-k must be at least 20 because this evaluation computes @20 diagnostics."
        )

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

    print("Building HybridRetriever...")
    hybrid_retriever = HybridRetriever.from_settings(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        settings=settings,
    )

    all_chunks = corpus_store.all_chunks()
    chunk_lookup = build_chunk_lookup(all_chunks)

    questions = build_retrieval_questions_from_corpus_chunks(
        all_chunks,
        strict_two_supporting_docs=True,
    )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be greater than or equal to 1.")
        questions = questions[: args.limit]

    evaluator = RetrievalEvaluator(
        k_values=(1, 2, 5, 10, 20),
        strict_two_supporting_docs=True,
    )

    print()
    print("Retrieval evaluation configuration")
    print("----------------------------------")
    print(f"Corpus path:       {corpus_path}")
    print(f"Dense index path:  {dense_index_path}")
    print(f"Dense metadata:    {dense_metadata_path}")
    print(f"Questions:         {len(questions)}")
    print(f"Max retrieved k:   {args.max_k}")
    print(f"BM25 top-k:        {settings.retrieval.bm25_top_k}")
    print(f"Dense top-k:       {settings.retrieval.dense_top_k}")
    print(f"Fusion method:     {settings.retrieval.fusion_method}")
    print(f"RRF k:             {settings.retrieval.rrf_k}")
    print(f"Embedding model:   {embedder.model_name}")
    print()

    records = []

    for index, question in enumerate(questions, start=1):
        start_time = time.perf_counter()

        retrieved_results = hybrid_retriever.search(
            query=question.question,
            top_k=args.max_k,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        record = evaluator.evaluate_question(
            question=question,
            retrieved_results=retrieved_results,
            chunk_lookup=chunk_lookup,
            latency_ms=latency_ms,
            max_k=args.max_k,
        )

        records.append(record)

        print(
            f"[{index:>4}/{len(questions)}] "
            f"question_id={question.question_id} "
            f"top1={int(record.top1_supporting_hit)} "
            f"both@5={int(record.both_supporting_found_at_5)} "
            f"both@10={int(record.both_supporting_found_at_10)} "
            f"recall@10={record.supporting_doc_recall_at_10:.2f} "
            f"mrr@10={record.first_support_mrr_at_10:.3f} "
            f"latency_ms={latency_ms:.2f}"
        )

    summary = evaluator.summarize(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"retrieval_eval_{timestamp}"

    per_question_path, summary_json_path, summary_md_path = write_retrieval_eval_outputs(
        output_dir=args.output_dir,
        run_name=run_name,
        records=records,
        summary=summary,
    )

    print()
    print("Retrieval evaluation complete.")
    print("--------------------------------")
    print(f"Per-question JSONL: {per_question_path}")
    print(f"Summary JSON:       {summary_json_path}")
    print(f"Summary Markdown:   {summary_md_path}")
    print()
    print("Primary metrics")
    print("---------------")
    print(f"Top-1 Supporting Hit Rate: {summary.top1_supporting_hit_rate:.4f}")
    print(f"Both Found@5:              {summary.both_supporting_found_rate_at_5:.4f}")
    print(f"Both Found@10:             {summary.both_supporting_found_rate_at_10:.4f}")
    print(f"Recall@5:                  {summary.mean_supporting_doc_recall_at_5:.4f}")
    print(f"Recall@10:                 {summary.mean_supporting_doc_recall_at_10:.4f}")
    print()
    print("Diagnostic metrics")
    print("------------------")
    print(f"Both Found@20:             {summary.both_supporting_found_rate_at_20:.4f}")
    print(f"Recall@20:                 {summary.mean_supporting_doc_recall_at_20:.4f}")
    print(f"Mean MRR@10:               {summary.mean_first_support_mrr_at_10:.4f}")
    print(f"Mean nDCG@10:              {summary.mean_ndcg_at_10:.4f}")

    if summary.average_latency_ms is not None:
        print(f"Average latency ms:        {summary.average_latency_ms:.2f}")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


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
        source_question_id = get_field(chunk, "source_question_id", "question_id", "qid", "_id")

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
            # Keep the question object. RetrievalEvaluator will mark it skipped
            # with a clear skip reason. This makes skipped cases visible.
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


if __name__ == "__main__":
    main()