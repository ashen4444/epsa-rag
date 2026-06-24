from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FailureCase:
    question_id: str
    question: str
    gold_answer: str | None
    gold_supporting_titles: list[str]
    gold_title_ranks: dict[str, int | None]
    failure_category: str

    both_found_at_failure_k: bool
    both_found_at_diagnostic_k: bool
    recall_at_failure_k: float
    recall_at_diagnostic_k: float

    missing_titles_at_failure_k: list[str]
    missing_titles_at_diagnostic_k: list[str]
    titles_found_between_failure_and_diagnostic_k: list[str]

    retrieved_titles_top20: list[str]
    retrieved_chunk_ids_top20: list[str]

    top1_supporting_hit: bool
    first_support_rank: int | None
    first_support_mrr_at_10: float
    ndcg_at_10: float
    latency_ms: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze retrieval evaluation failure cases."
    )

    parser.add_argument(
        "--per-question-path",
        type=Path,
        required=True,
        help="Path to retrieval_eval_<timestamp>_per_question.jsonl.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "outputs/retrieval_eval/failure_analysis."
        ),
    )

    parser.add_argument(
        "--failure-k",
        type=int,
        default=10,
        help="Main top-k threshold to analyze failures. Default: 10.",
    )

    parser.add_argument(
        "--diagnostic-k",
        type=int,
        default=20,
        help="Larger diagnostic top-k threshold. Default: 20.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.failure_k < 1:
        raise ValueError("--failure-k must be >= 1.")

    if args.diagnostic_k <= args.failure_k:
        raise ValueError("--diagnostic-k must be greater than --failure-k.")

    per_question_path = resolve_project_path(args.per_question_path)

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "retrieval_eval" / "failure_analysis"
    else:
        output_dir = resolve_project_path(args.output_dir)

    records = load_jsonl(per_question_path)

    valid_records = [
        record for record in records if not bool(record.get("skipped", False))
    ]
    skipped_records = [
        record for record in records if bool(record.get("skipped", False))
    ]

    failure_cases = build_failure_cases(
        records=valid_records,
        failure_k=args.failure_k,
        diagnostic_k=args.diagnostic_k,
    )

    run_name = infer_run_name(per_question_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_jsonl_path = output_dir / f"{run_name}_failure_cases_at_{args.failure_k}.jsonl"
    failure_md_path = output_dir / f"{run_name}_failure_analysis_at_{args.failure_k}.md"

    write_failure_cases_jsonl(failure_jsonl_path, failure_cases)
    write_failure_analysis_markdown(
        path=failure_md_path,
        run_name=run_name,
        source_path=per_question_path,
        records=records,
        valid_records=valid_records,
        skipped_records=skipped_records,
        failure_cases=failure_cases,
        failure_k=args.failure_k,
        diagnostic_k=args.diagnostic_k,
    )

    print("Retrieval failure analysis complete.")
    print("------------------------------------")
    print(f"Source:             {per_question_path}")
    print(f"Failure cases JSONL: {failure_jsonl_path}")
    print(f"Markdown report:     {failure_md_path}")
    print()
    print("Summary")
    print("-------")
    print(f"Total records:       {len(records)}")
    print(f"Evaluated records:   {len(valid_records)}")
    print(f"Skipped records:     {len(skipped_records)}")
    print(f"Failures@{args.failure_k}:       {len(failure_cases)}")

    categories = Counter(case.failure_category for case in failure_cases)
    for category, count in categories.most_common():
        print(f"{category}: {count}")


def build_failure_cases(
    *,
    records: list[dict[str, Any]],
    failure_k: int,
    diagnostic_k: int,
) -> list[FailureCase]:
    failure_cases: list[FailureCase] = []

    for record in records:
        both_at_failure_k = both_found_at_k(record, failure_k)

        if both_at_failure_k:
            continue

        gold_title_ranks = normalize_gold_title_ranks(
            record.get("gold_title_ranks", {})
        )

        missing_at_failure_k = missing_titles_at_k(
            gold_title_ranks,
            k=failure_k,
        )
        missing_at_diagnostic_k = missing_titles_at_k(
            gold_title_ranks,
            k=diagnostic_k,
        )
        found_between = titles_found_between(
            gold_title_ranks,
            lower_k=failure_k,
            upper_k=diagnostic_k,
        )

        both_at_diagnostic_k = both_found_at_k(record, diagnostic_k)
        recall_failure_k = recall_at_k(record, failure_k)
        recall_diagnostic_k = recall_at_k(record, diagnostic_k)

        failure_category = classify_failure(
            both_at_diagnostic_k=both_at_diagnostic_k,
            recall_at_failure_k=recall_failure_k,
            recall_at_diagnostic_k=recall_diagnostic_k,
        )

        failure_cases.append(
            FailureCase(
                question_id=str(record.get("question_id", "")),
                question=str(record.get("question", "")),
                gold_answer=to_optional_str(record.get("gold_answer")),
                gold_supporting_titles=list(record.get("gold_supporting_titles", [])),
                gold_title_ranks=gold_title_ranks,
                failure_category=failure_category,
                both_found_at_failure_k=both_at_failure_k,
                both_found_at_diagnostic_k=both_at_diagnostic_k,
                recall_at_failure_k=recall_failure_k,
                recall_at_diagnostic_k=recall_diagnostic_k,
                missing_titles_at_failure_k=missing_at_failure_k,
                missing_titles_at_diagnostic_k=missing_at_diagnostic_k,
                titles_found_between_failure_and_diagnostic_k=found_between,
                retrieved_titles_top20=list(record.get("retrieved_titles_top20", [])),
                retrieved_chunk_ids_top20=list(record.get("retrieved_chunk_ids_top20", [])),
                top1_supporting_hit=bool(record.get("top1_supporting_hit", False)),
                first_support_rank=to_optional_int(record.get("first_support_rank")),
                first_support_mrr_at_10=float(record.get("first_support_mrr_at_10", 0.0)),
                ndcg_at_10=float(record.get("ndcg_at_10", 0.0)),
                latency_ms=to_optional_float(record.get("latency_ms")),
            )
        )

    failure_cases.sort(
        key=lambda case: (
            case.failure_category,
            -case.recall_at_diagnostic_k,
            case.question_id,
        )
    )

    return failure_cases


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


def write_failure_cases_jsonl(
    path: Path,
    failure_cases: list[FailureCase],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for failure_case in failure_cases:
            file.write(
                json.dumps(
                    asdict(failure_case),
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_failure_analysis_markdown(
    *,
    path: Path,
    run_name: str,
    source_path: Path,
    records: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    skipped_records: list[dict[str, Any]],
    failure_cases: list[FailureCase],
    failure_k: int,
    diagnostic_k: int,
) -> None:
    category_counts = Counter(case.failure_category for case in failure_cases)

    failure_rate = safe_rate(len(failure_cases), len(valid_records))
    success_count = len(valid_records) - len(failure_cases)
    success_rate = safe_rate(success_count, len(valid_records))

    recovered_cases = [
        case for case in failure_cases if case.both_found_at_diagnostic_k
    ]
    recovered_rate_among_failures = safe_rate(len(recovered_cases), len(failure_cases))
    recovered_rate_total = safe_rate(len(recovered_cases), len(valid_records))

    lines: list[str] = []

    lines.extend(
        [
            f"# Retrieval Failure Analysis: `{run_name}`",
            "",
            "## Scope",
            "",
            (
                f"This report analyzes questions where both gold supporting document "
                f"titles were **not** found within top-{failure_k}."
            ),
            "",
            f"Source file:",
            "",
            f"```text",
            str(source_path),
            f"```",
            "",
            "## High-Level Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total records | {len(records)} |",
            f"| Evaluated records | {len(valid_records)} |",
            f"| Skipped records | {len(skipped_records)} |",
            f"| Success@{failure_k} | {success_count} |",
            f"| Success@{failure_k} rate | {success_rate:.4f} |",
            f"| Failure@{failure_k} | {len(failure_cases)} |",
            f"| Failure@{failure_k} rate | {failure_rate:.4f} |",
            (
                f"| Failures recovered by top-{diagnostic_k} | "
                f"{len(recovered_cases)} |"
            ),
            (
                f"| Recovery rate among failures | "
                f"{recovered_rate_among_failures:.4f} |"
            ),
            (
                f"| Recovery rate across all evaluated questions | "
                f"{recovered_rate_total:.4f} |"
            ),
            "",
            "## Failure Categories",
            "",
            "| Failure category | Count | Meaning |",
            "|---|---:|---|",
        ]
    )

    category_meanings = {
        "ranking_failure_recovered_by_diagnostic_k": (
            f"Both gold documents were missing at top-{failure_k}, "
            f"but both appeared by top-{diagnostic_k}. This is mainly a ranking issue."
        ),
        "ranking_plus_partial_coverage_failure": (
            f"More gold evidence appeared after top-{failure_k}, but both were still "
            f"not found by top-{diagnostic_k}."
        ),
        "partial_coverage_failure": (
            f"At least one gold document was found, but the complete two-document "
            f"evidence set was not found by top-{diagnostic_k}."
        ),
        "coverage_failure_no_gold_found": (
            f"No gold supporting document was found by top-{diagnostic_k}."
        ),
    }

    for category, count in category_counts.most_common():
        meaning = category_meanings.get(category, "")
        lines.append(f"| `{category}` | {count} | {meaning} |")

    lines.extend(
        [
            "",
            "## Failure Case Overview",
            "",
            (
                f"| Question ID | Category | Recall@{failure_k} | "
                f"Recall@{diagnostic_k} | Missing@{failure_k} | "
                f"Found between {failure_k + 1}-{diagnostic_k} |"
            ),
            "|---|---|---:|---:|---|---|",
        ]
    )

    for case in failure_cases:
        missing_at_failure = "; ".join(case.missing_titles_at_failure_k) or "-"
        found_between_text = (
            "; ".join(case.titles_found_between_failure_and_diagnostic_k) or "-"
        )

        lines.append(
            f"| `{case.question_id}` "
            f"| `{case.failure_category}` "
            f"| {case.recall_at_failure_k:.2f} "
            f"| {case.recall_at_diagnostic_k:.2f} "
            f"| {escape_table_text(missing_at_failure)} "
            f"| {escape_table_text(found_between_text)} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Failure Cases",
            "",
        ]
    )

    for case in failure_cases:
        lines.extend(render_detailed_failure_case(case, failure_k, diagnostic_k))

    path.write_text("\n".join(lines), encoding="utf-8")


def render_detailed_failure_case(
    case: FailureCase,
    failure_k: int,
    diagnostic_k: int,
) -> list[str]:
    lines = [
        f"### {case.question_id}",
        "",
        f"**Category:** `{case.failure_category}`",
        "",
        f"**Question:** {case.question}",
        "",
        f"**Gold answer:** {case.gold_answer}",
        "",
        "**Gold supporting title ranks:**",
        "",
    ]

    for title, rank in case.gold_title_ranks.items():
        rank_text = "missing" if rank is None else str(rank)
        lines.append(f"- `{title}` → {rank_text}")

    lines.extend(
        [
            "",
            f"**Missing titles at top-{failure_k}:** "
            + (", ".join(case.missing_titles_at_failure_k) or "-"),
            "",
            f"**Missing titles at top-{diagnostic_k}:** "
            + (", ".join(case.missing_titles_at_diagnostic_k) or "-"),
            "",
            f"**Titles found between ranks {failure_k + 1}-{diagnostic_k}:** "
            + (
                ", ".join(case.titles_found_between_failure_and_diagnostic_k)
                or "-"
            ),
            "",
            "**Retrieved titles top-20:**",
            "",
        ]
    )

    for rank, title in enumerate(case.retrieved_titles_top20, start=1):
        chunk_id = ""
        if rank <= len(case.retrieved_chunk_ids_top20):
            chunk_id = case.retrieved_chunk_ids_top20[rank - 1]

        gold_marker = " ✅ GOLD" if title in case.gold_supporting_titles else ""
        lines.append(f"{rank}. `{title}` — `{chunk_id}`{gold_marker}")

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


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def to_optional_str(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)


def to_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    return int(value)


def to_optional_float(value: Any) -> float | None:
    if value is None:
        return None

    return float(value)


def escape_table_text(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()