from __future__ import annotations

import argparse
from pathlib import Path

from epsa_rag.datasets.chunk_builder import build_paragraph_chunks
from epsa_rag.datasets.hotpotqa_loader import load_hotpotqa_examples
from epsa_rag.utils.jsonl import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare HotPotQA paragraph-level chunks as a JSONL corpus."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the raw HotPotQA JSON file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the output JSONL file.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of examples to process from the beginning of the file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples = load_hotpotqa_examples(
        input_path=args.input,
        sample_size=args.sample_size,
    )

    chunks = build_paragraph_chunks(examples)

    written_count = write_jsonl(
        records=(chunk.to_json_dict() for chunk in chunks),
        output_path=args.output,
    )

    print(f"Loaded examples: {len(examples)}")
    print(f"Processed examples: {len(examples)}")
    print(f"Written chunks: {written_count}")
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()