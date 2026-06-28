from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epsa_rag.rag.llm_client import OpenAIChatClient
from epsa_rag.rag.two_hop_baseline import TwoHopBaselineConfig, TwoHopHybridRAGBaseline


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    question: str
    gold_answer: str | None
    gold_supporting_titles: list[str]


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

    baseline = TwoHopHybridRAGBaseline(
        retriever=retriever,
        corpus_store=corpus_store,
        llm_client=llm_client,
        config=TwoHopBaselineConfig(
            hop1_top_k=args.hop1_top_k,
            hop2_top_k=args.hop2_top_k,
            temperature=args.temperature,
            hop2_query_max_tokens=args.hop2_query_max_tokens,
            final_answer_max_tokens=args.final_answer_max_tokens,
        ),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    records_path = output_dir / f"two_hop_baseline_records_{timestamp}.jsonl"
    summary_path = output_dir / f"two_hop_baseline_summary_{timestamp}.json"

    records: list[dict[str, Any]] = []

    with records_path.open("w", encoding="utf-8") as writer:
        for index, item in enumerate(questions, start=1):
            result = baseline.run(
                question_id=item.question_id,
                question=item.question,
                gold_answer=item.gold_answer,
                gold_supporting_titles=item.gold_supporting_titles,
            )

            record = result.to_dict()
            records.append(record)

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[{index}/{len(questions)}] "
                f"{item.question_id} | "
                f"EM={record['exact_match']:.0f} "
                f"F1={record['answer_f1']:.3f} "
                f"context_docs={record['num_context_documents']} "
                f"tokens={record['total_tokens']} "
                f"latency_ms={record['latency_ms']:.1f}"
            )

    summary = summarize_records(records)

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
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved baseline records:", records_path)
    print("Saved baseline summary:", summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed 2-hop Hybrid RAG baseline evaluation."
    )

    parser.add_argument("--questions-path", default="data/processed/hotpotqa_questions.jsonl")
    parser.add_argument("--corpus-path", default="data/processed/hotpotqa_chunks.jsonl")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")

    parser.add_argument("--dense-index-path", default="data/indexes/dense/faiss_index.bin")
    parser.add_argument("--dense-metadata-path", default="data/indexes/dense/dense_metadata.json")

    parser.add_argument("--output-dir", default="outputs/rag_baseline")

    parser.add_argument("--limit", type=int, default=100)

    parser.add_argument("--hop1-top-k", type=int, default=5)
    parser.add_argument("--hop2-top-k", type=int, default=5)

    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-batch-size", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--hop2-query-max-tokens", type=int, default=48)
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

    OpenAITextEmbedder = resolve_first_symbol(
        [
            "epsa_rag.embeddings.openai_text_embedder",
            "epsa_rag.embeddings.openai_embedder",
            "epsa_rag.retrieval.openai_text_embedder",
            "epsa_rag.retrieval.openai_embedder",
            "epsa_rag.retrieval.embedding_backend",
            "epsa_rag.retrieval.embedding_backends",
        ],
        "OpenAITextEmbedder",
    )

    try:
        settings = load_retrieval_settings(Path(args.retrieval_config))
    except TypeError:
        settings = load_retrieval_settings()

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


def resolve_first_symbol(module_names: list[str], symbol_name: str) -> Any:
    errors: list[str] = []

    for module_name in module_names:
        try:
            return resolve_symbol(module_name, symbol_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    raise ImportError(
        f"Could not resolve {symbol_name}. Tried:\n" + "\n".join(errors)
    )


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}. "
            f"Pass the correct path with the matching CLI argument."
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
    question_id = first_present(
        raw,
        "question_id",
        "id",
        "_id",
        default=f"line_{line_number}",
    )

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
    """
    Kept only for compatibility with existing HotPotQA processed records.

    These titles are passed through to the baseline runner, but the end-to-end
    RAG baseline intentionally does not log supporting-document metrics.
    """
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


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)

    if total == 0:
        return {"num_records": 0}

    return {
        "num_records": total,
        "exact_match": mean_field(records, "exact_match"),
        "partial_match": mean_field(records, "partial_match"),
        "answer_precision": mean_field(records, "answer_precision"),
        "answer_recall": mean_field(records, "answer_recall"),
        "answer_f1": mean_field(records, "answer_f1"),
        "average_context_docs_sent_to_final_llm": mean_field(records, "num_context_documents"),
        "average_estimated_context_tokens": mean_field(records, "estimated_context_tokens"),
        "average_total_llm_tokens": mean_field(records, "total_tokens"),
        "average_latency_ms": mean_field(records, "latency_ms"),
        "hop2_query_generation_failures": sum(
            1 for record in records if record.get("hop2_query_generation_error")
        ),
        "final_answer_generation_failures": sum(
            1 for record in records if record.get("final_answer_generation_error")
        ),
        "retrieval_failures": sum(
            1 for record in records if record.get("retrieval_error")
        ),
    }


def mean_field(records: list[dict[str, Any]], field_name: str) -> float:
    values = [float(record.get(field_name) or 0.0) for record in records]

    return round(statistics.fmean(values), 6) if values else 0.0


if __name__ == "__main__":
    main()