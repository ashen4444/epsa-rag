from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epsa_rag.config.retrieval_config import load_retrieval_settings
from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.bm25_retriever import BM25Retriever
from epsa_rag.retrieval.dense_retriever import DenseRetriever
from epsa_rag.retrieval.embedding_backend import OpenAITextEmbedder
from epsa_rag.retrieval.hybrid_retriever import HybridRetriever


@dataclass(frozen=True)
class TraceResultRecord:
    hybrid_rank: int
    chunk_id: str
    doc_title: str
    is_gold_title: bool

    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None
    fusion_score: float


@dataclass(frozen=True)
class FusionTraceQuestionReport:
    question_id: str
    question: str
    gold_answer: str | None
    gold_supporting_titles: list[str]
    failure_category: str

    original_gold_title_ranks: dict[str, int | None]
    traced_gold_title_ranks: dict[str, int | None]

    missing_gold_titles_at_failure_k: list[str]
    gold_titles_found_between_failure_and_diagnostic_k: list[str]

    top_results: list[TraceResultRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze BM25 rank, dense rank, and RRF fusion score for retrieval "
            "evaluation failure cases."
        )
    )

    parser.add_argument(
        "--per-question-path",
        type=Path,
        required=True,
        help="Path to retrieval_eval_<timestamp>_per_question.jsonl.",
    )

    parser.add_argument(
        "--failure-k",
        type=int,
        default=10,
        help="Main failure threshold. Default: 10.",
    )

    parser.add_argument(
        "--diagnostic-k",
        type=int,
        default=20,
        help="Trace depth. Default: 20.",
    )

    parser.add_argument(
        "--category",
        choices=[
            "all",
            "ranking",
            "partial",
        ],
        default="all",
        help=(
            "Which failure type to trace. "
            "'ranking' = recovered by diagnostic-k. "
            "'partial' = not recovered by diagnostic-k but some gold evidence found. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "retrieval_eval" / "fusion_trace_analysis",
        help="Output directory for fusion trace reports.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.failure_k < 1:
        raise ValueError("--failure-k must be >= 1.")

    if args.diagnostic_k <= args.failure_k:
        raise ValueError("--diagnostic-k must be greater than --failure-k.")

    per_question_path = resolve_project_path(args.per_question_path)
    output_dir = resolve_project_path(args.output_dir)

    records = load_jsonl(per_question_path)
    failure_records = select_failure_records(
        records=records,
        failure_k=args.failure_k,
        diagnostic_k=args.diagnostic_k,
        category=args.category,
    )

    print("Loading retrieval components...")
    settings = load_retrieval_settings()

    corpus_path = resolve_project_path(settings.paths.processed_corpus)
    dense_index_path = resolve_project_path(settings.paths.dense_index)
    dense_metadata_path = resolve_project_path(settings.paths.dense_metadata)

    corpus_store = CorpusStore.from_jsonl(corpus_path)

    bm25_retriever = BM25Retriever.from_corpus_store(corpus_store)

    embedder = OpenAITextEmbedder(
        model_name_or_path=settings.dense.model_name,
        batch_size=settings.dense.batch_size,
    )

    dense_retriever = DenseRetriever.load(
        corpus_store=corpus_store,
        index_path=dense_index_path,
        metadata_path=dense_metadata_path,
        embedder=embedder,
    )

    hybrid_retriever = HybridRetriever.from_settings(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        settings=settings,
    )

    print()
    print("Fusion trace configuration")
    print("--------------------------")
    print(f"Source per-question file: {per_question_path}")
    print(f"Failure k:                {args.failure_k}")
    print(f"Diagnostic k:             {args.diagnostic_k}")
    print(f"Category:                 {args.category}")
    print(f"Failure records selected: {len(failure_records)}")
    print(f"BM25 top-k:               {settings.retrieval.bm25_top_k}")
    print(f"Dense top-k:              {settings.retrieval.dense_top_k}")
    print(f"RRF k:                    {settings.retrieval.rrf_k}")
    print()

    reports: list[FusionTraceQuestionReport] = []

    for index, record in enumerate(failure_records, start=1):
        question_id = str(record.get("question_id", ""))
        question_text = str(record.get("question", ""))

        print(f"[{index}/{len(failure_records)}] tracing question_id={question_id}")

        traces = hybrid_retriever.search_with_trace(
            query=question_text,
            top_k=args.diagnostic_k,
        )

        report = build_question_trace_report(
            record=record,
            traces=traces,
            corpus_store=corpus_store,
            failure_k=args.failure_k,
            diagnostic_k=args.diagnostic_k,
        )

        reports.append(report)

    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = infer_run_name(per_question_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{run_name}_fusion_trace_{args.category}_{timestamp}"

    jsonl_path = output_dir / f"{output_prefix}.jsonl"
    md_path = output_dir / f"{output_prefix}.md"

    write_reports_jsonl(jsonl_path, reports)
    write_markdown_report(
        path=md_path,
        source_path=per_question_path,
        reports=reports,
        failure_k=args.failure_k,
        diagnostic_k=args.diagnostic_k,
        category=args.category,
    )

    print()
    print("Fusion trace analysis complete.")
    print("--------------------------------")
    print(f"JSONL output:    {jsonl_path}")
    print(f"Markdown report: {md_path}")


def select_failure_records(
    *,
    records: list[dict[str, Any]],
    failure_k: int,
    diagnostic_k: int,
    category: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for record in records:
        if bool(record.get("skipped", False)):
            continue

        if both_found_at_k(record, failure_k):
            continue

        recall_failure = recall_at_k(record, failure_k)
        recall_diagnostic = recall_at_k(record, diagnostic_k)
        both_diagnostic = both_found_at_k(record, diagnostic_k)

        if category == "ranking" and not both_diagnostic:
            continue

        if category == "partial":
            if both_diagnostic:
                continue
            if recall_diagnostic <= 0.0:
                continue

        selected.append(record)

    selected.sort(key=lambda item: str(item.get("question_id", "")))
    return selected


def build_question_trace_report(
    *,
    record: Mapping[str, Any],
    traces: Sequence[Any],
    corpus_store: CorpusStore,
    failure_k: int,
    diagnostic_k: int,
) -> FusionTraceQuestionReport:
    gold_titles = [str(title) for title in record.get("gold_supporting_titles", [])]
    gold_title_set = set(gold_titles)

    trace_results: list[TraceResultRecord] = []

    for hybrid_rank, trace in enumerate(traces, start=1):
        chunk_id = str(trace.chunk_id)
        chunk = corpus_store.get_chunk(chunk_id)
        doc_title = str(get_field(chunk, "doc_title") or "")

        trace_results.append(
            TraceResultRecord(
                hybrid_rank=hybrid_rank,
                chunk_id=chunk_id,
                doc_title=doc_title,
                is_gold_title=doc_title in gold_title_set,
                bm25_rank=trace.bm25_rank,
                dense_rank=trace.dense_rank,
                bm25_score=trace.bm25_score,
                dense_score=trace.dense_score,
                fusion_score=trace.fusion_score,
            )
        )

    traced_gold_ranks = get_gold_ranks_from_trace_results(
        trace_results=trace_results,
        gold_titles=gold_titles,
    )

    original_gold_ranks = normalize_gold_title_ranks(
        record.get("gold_title_ranks", {})
    )

    failure_category = classify_failure(
        both_at_diagnostic_k=both_found_at_k(record, diagnostic_k),
        recall_at_failure_k=recall_at_k(record, failure_k),
        recall_at_diagnostic_k=recall_at_k(record, diagnostic_k),
    )

    return FusionTraceQuestionReport(
        question_id=str(record.get("question_id", "")),
        question=str(record.get("question", "")),
        gold_answer=to_optional_str(record.get("gold_answer")),
        gold_supporting_titles=gold_titles,
        failure_category=failure_category,
        original_gold_title_ranks=original_gold_ranks,
        traced_gold_title_ranks=traced_gold_ranks,
        missing_gold_titles_at_failure_k=missing_titles_at_k(
            original_gold_ranks,
            k=failure_k,
        ),
        gold_titles_found_between_failure_and_diagnostic_k=titles_found_between(
            original_gold_ranks,
            lower_k=failure_k,
            upper_k=diagnostic_k,
        ),
        top_results=trace_results,
    )


def get_gold_ranks_from_trace_results(
    *,
    trace_results: list[TraceResultRecord],
    gold_titles: list[str],
) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {title: None for title in gold_titles}

    for result in trace_results:
        if result.doc_title in ranks and ranks[result.doc_title] is None:
            ranks[result.doc_title] = result.hybrid_rank

    return ranks


def classify_failure(
    *,
    both_at_diagnostic_k: bool,
    recall_at_failure_k: float,
    recall_at_diagnostic_k: float,
) -> str:
    if both_at_diagnostic_k:
        return "ranking_failure_recovered_by_diagnostic_k"

    if recall_at_diagnostic_k > recall_at_failure_k:
        return "ranking_plus_partial_coverage_failure"

    if recall_at_diagnostic_k > 0.0:
        return "partial_coverage_failure"

    return "coverage_failure_no_gold_found"


def both_found_at_k(record: Mapping[str, Any], k: int) -> bool:
    direct_field = f"both_supporting_found_at_{k}"

    if direct_field in record:
        return bool(record[direct_field])

    gold_ranks = normalize_gold_title_ranks(record.get("gold_title_ranks", {}))
    if len(gold_ranks) != 2:
        return False

    return all(rank is not None and rank <= k for rank in gold_ranks.values())


def recall_at_k(record: Mapping[str, Any], k: int) -> float:
    direct_field = f"supporting_doc_recall_at_{k}"

    if direct_field in record:
        return float(record[direct_field])

    gold_ranks = normalize_gold_title_ranks(record.get("gold_title_ranks", {}))

    if not gold_ranks:
        return 0.0

    found_count = sum(
        1 for rank in gold_ranks.values() if rank is not None and rank <= k
    )

    return found_count / len(gold_ranks)


def missing_titles_at_k(
    gold_title_ranks: Mapping[str, int | None],
    *,
    k: int,
) -> list[str]:
    return [
        title
        for title, rank in gold_title_ranks.items()
        if rank is None or rank > k
    ]


def titles_found_between(
    gold_title_ranks: Mapping[str, int | None],
    *,
    lower_k: int,
    upper_k: int,
) -> list[str]:
    return [
        title
        for title, rank in gold_title_ranks.items()
        if rank is not None and lower_k < rank <= upper_k
    ]


def normalize_gold_title_ranks(value: Any) -> dict[str, int | None]:
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, int | None] = {}

    for title, rank in value.items():
        if rank is None:
            normalized[str(title)] = None
        else:
            normalized[str(title)] = int(rank)

    return normalized


def write_reports_jsonl(
    path: Path,
    reports: list[FusionTraceQuestionReport],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for report in reports:
            file.write(json.dumps(asdict(report), ensure_ascii=False) + "\n")


def write_markdown_report(
    *,
    path: Path,
    source_path: Path,
    reports: list[FusionTraceQuestionReport],
    failure_k: int,
    diagnostic_k: int,
    category: str,
) -> None:
    category_counts = Counter(report.failure_category for report in reports)

    lines: list[str] = [
        "# Hybrid Fusion Trace Analysis",
        "",
        "## Scope",
        "",
        (
            "This report analyzes BM25 rank, dense rank, BM25 score, dense score, "
            "and final RRF fusion score for retrieval failure cases."
        ),
        "",
        "Source per-question file:",
        "",
        "```text",
        str(source_path),
        "```",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---:|",
        f"| Failure k | {failure_k} |",
        f"| Diagnostic k | {diagnostic_k} |",
        f"| Selected category | `{category}` |",
        f"| Questions analyzed | {len(reports)} |",
        "",
        "## Failure Category Counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]

    for failure_category, count in category_counts.most_common():
        lines.append(f"| `{failure_category}` | {count} |")

    lines.extend(
        [
            "",
            "## Question-Level Fusion Trace",
            "",
        ]
    )

    for report in reports:
        lines.extend(render_question_report(report, failure_k, diagnostic_k))

    path.write_text("\n".join(lines), encoding="utf-8")


def render_question_report(
    report: FusionTraceQuestionReport,
    failure_k: int,
    diagnostic_k: int,
) -> list[str]:
    lines: list[str] = [
        f"### {report.question_id}",
        "",
        f"**Failure category:** `{report.failure_category}`",
        "",
        f"**Question:** {report.question}",
        "",
        f"**Gold answer:** {report.gold_answer}",
        "",
        "**Gold supporting titles:**",
        "",
    ]

    for title in report.gold_supporting_titles:
        original_rank = report.original_gold_title_ranks.get(title)
        traced_rank = report.traced_gold_title_ranks.get(title)

        original_text = "missing" if original_rank is None else str(original_rank)
        traced_text = "missing" if traced_rank is None else str(traced_rank)

        lines.append(
            f"- `{title}` → original rank: {original_text}, traced rank: {traced_text}"
        )

    lines.extend(
        [
            "",
            f"**Missing at top-{failure_k}:** "
            + (", ".join(report.missing_gold_titles_at_failure_k) or "-"),
            "",
            (
                f"**Gold titles found between ranks {failure_k + 1}-{diagnostic_k}:** "
                + (
                    ", ".join(report.gold_titles_found_between_failure_and_diagnostic_k)
                    or "-"
                )
            ),
            "",
            "**Top traced hybrid results:**",
            "",
            (
                "| Hybrid rank | Title | Gold? | BM25 rank | Dense rank | "
                "BM25 score | Dense score | RRF score | Chunk ID |"
            ),
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for result in report.top_results:
        gold_text = "yes" if result.is_gold_title else ""

        lines.append(
            f"| {result.hybrid_rank} "
            f"| {escape_table_text(result.doc_title)} "
            f"| {gold_text} "
            f"| {format_optional_int(result.bm25_rank)} "
            f"| {format_optional_int(result.dense_rank)} "
            f"| {format_optional_float(result.bm25_score)} "
            f"| {format_optional_float(result.dense_score)} "
            f"| {result.fusion_score:.6f} "
            f"| `{result.chunk_id}` |"
        )

    lines.append("")
    return lines


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} in {path}"
                )

            records.append(value)

    return records


def infer_run_name(per_question_path: Path) -> str:
    stem = per_question_path.stem

    if stem.endswith("_per_question"):
        return stem[: -len("_per_question")]

    return stem


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


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


def to_optional_str(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)


def format_optional_int(value: int | None) -> str:
    if value is None:
        return "-"

    return str(value)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "-"

    return f"{value:.6f}"


def escape_table_text(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()