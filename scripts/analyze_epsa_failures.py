from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epsa_rag.evaluation.epsa_failure_analysis import (
    analyze_epsa_failure_records,
    read_epsa_rag_csv,
    write_failure_analysis_json,
    write_failure_analysis_markdown,
)


def main() -> None:
    args = parse_args()

    records_path = Path(args.records_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_epsa_rag_csv(records_path)
    report = analyze_epsa_failure_records(records, max_examples=args.max_examples)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = records_path.stem
    json_path = output_dir / f"epsa_failure_analysis_{stem}_{timestamp}.json"
    md_path = output_dir / f"epsa_failure_analysis_{stem}_{timestamp}.md"

    write_failure_analysis_json(report, json_path)
    write_failure_analysis_markdown(report, md_path)

    print(f"Analyzed EPSA RAG records: {records_path}")
    print(f"Saved failure analysis JSON: {json_path}")
    print(f"Saved failure analysis Markdown: {md_path}")
    print("Core warning counts:")
    print(f"  potential_false_sufficient_count: {report['sufficiency']['potential_false_sufficient_count']}")
    print(f"  potential_false_sufficient_among_sufficient_rate: {report['sufficiency']['potential_false_sufficient_among_sufficient_rate']}")
    print(f"  insufficient_pruned_context_count: {report['context']['insufficient_pruned_context_count']}")
    print(f"  wrong_with_one_sentence: {report['failure_pattern_counts']['wrong_with_one_sentence']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze row-level EPSA RAG failures without retrieval or LLM calls."
    )
    parser.add_argument(
        "--records-path",
        required=True,
        help="Path to epsa_rag_results_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/epsa_failure_analysis",
        help="Directory for JSON/Markdown failure-analysis reports.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum example cases to include per failure bucket.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
