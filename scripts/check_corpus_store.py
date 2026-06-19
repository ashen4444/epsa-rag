"""Smoke-check the processed EPSA-RAG corpus using CorpusStore.

This script verifies that the real processed JSONL corpus can be loaded by
the CorpusStore and prints basic statistics/sample IDs.

It does not run retrieval, indexing, EPSA, or generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epsa_rag.corpus import CorpusStore


DEFAULT_CORPUS_PATH = Path("data/processed/hotpotqa_paragraph_chunks.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check processed corpus with CorpusStore.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to processed paragraph chunk JSONL corpus.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Number of sample chunks to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    store = CorpusStore.from_jsonl(args.path)

    print("CorpusStore smoke check passed.")
    print(json.dumps(store.stats(), indent=2))

    print("\nSample chunk IDs:")
    for chunk_id in store.all_chunk_ids()[: args.show]:
        print(f"- {chunk_id}")

    print("\nSample chunks:")
    for chunk in store.all_chunks()[: args.show]:
        print("-" * 80)
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"source_question_id: {chunk.source_question_id}")
        print(f"doc_title: {chunk.doc_title}")
        print(f"is_supporting_doc: {chunk.is_supporting_doc}")
        print(f"chunk_text_preview: {chunk.chunk_text[:300]}")


if __name__ == "__main__":
    main()