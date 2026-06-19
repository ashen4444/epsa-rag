"""Prepare HotPotQA distractor data into EPSA-RAG paragraph chunks.

Default rebuild command:

    python scripts/prepare_hotpotqa.py

This script loads the real HotPotQA dataset using Hugging Face datasets,
converts examples into the existing ParagraphChunk schema, and writes the
processed global paragraph corpus to JSONL.

It intentionally does not build BM25, dense embeddings, FAISS indexes,
retrieval results, EPSA outputs, or answer-generation outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from epsa_rag.datasets.schemas import ParagraphChunk, SentenceMetadata


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_NAME = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG_NAME = "distractor"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 100
DEFAULT_SHUFFLE = False
DEFAULT_SEED = 42
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "hotpotqa_paragraph_chunks.jsonl"


def _model_fields(model: type) -> dict[str, Any]:
    """Return Pydantic v2/v1 model fields without deprecation warnings."""

    model_fields = getattr(model, "model_fields", None)
    if model_fields is not None:
        return model_fields

    return getattr(model, "__fields__", {})


def _field_names(model: type) -> set[str]:
    return set(_model_fields(model).keys())


def _add_if_field(record: dict[str, Any], model: type, field_name: str, value: Any) -> None:
    if field_name in _field_names(model):
        record[field_name] = value


def _resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""

    if path.is_absolute():
        return path

    return BASE_DIR / path


def _sanitize_title_for_id(title: str) -> str:
    """Create a stable chunk-id-safe title string."""

    cleaned = title.strip().replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "", cleaned)
    return cleaned or "untitled"


def _parse_limit(value: str) -> int | None:
    """Parse --limit.

    Accepts:
        100
        500
        none
        all
        full
    """

    normalized = value.strip().lower()

    if normalized in {"none", "all", "full"}:
        return None

    limit = int(normalized)

    if limit <= 0:
        raise argparse.ArgumentTypeError("--limit must be positive, or use 'none'/'all'.")

    return limit


def _extract_context(example: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Extract title/sentences pairs from a HotPotQA example.

    Hugging Face HotPotQA usually stores context as:
        {"title": [...], "sentences": [[...], ...]}

    This function is defensive so the preparation script fails clearly if
    the structure changes.
    """

    context = example.get("context")

    if isinstance(context, dict):
        titles = context.get("title")
        sentence_groups = context.get("sentences")

        if not isinstance(titles, list) or not isinstance(sentence_groups, list):
            raise ValueError("HotPotQA context dict must contain list fields: title, sentences.")

        if len(titles) != len(sentence_groups):
            raise ValueError(
                "HotPotQA context has mismatched title and sentences lengths: "
                f"{len(titles)} titles vs {len(sentence_groups)} sentence groups."
            )

        pairs: list[tuple[str, list[str]]] = []
        for title, sentences in zip(titles, sentence_groups, strict=True):
            if not isinstance(sentences, list):
                raise ValueError(f"Context sentences for title {title!r} must be a list.")

            pairs.append((str(title), [str(sentence) for sentence in sentences]))

        return pairs

    if isinstance(context, list):
        pairs = []
        for item in context:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], list):
                pairs.append((str(item[0]), [str(sentence) for sentence in item[1]]))
            else:
                raise ValueError(f"Unsupported HotPotQA context item format: {item!r}")

        return pairs

    raise ValueError(f"Unsupported HotPotQA context format: {type(context).__name__}")


def _extract_supporting_facts(example: dict[str, Any]) -> dict[str, set[int]]:
    """Return mapping: document title -> supporting sentence ids."""

    supporting_facts = example.get("supporting_facts", {})
    mapping: dict[str, set[int]] = {}

    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", [])

        if len(titles) != len(sent_ids):
            raise ValueError(
                "HotPotQA supporting_facts has mismatched title and sent_id lengths: "
                f"{len(titles)} titles vs {len(sent_ids)} sentence ids."
            )

        for title, sent_id in zip(titles, sent_ids, strict=True):
            mapping.setdefault(str(title), set()).add(int(sent_id))

        return mapping

    if isinstance(supporting_facts, list):
        for item in supporting_facts:
            if isinstance(item, list) and len(item) == 2:
                title, sent_id = item
                mapping.setdefault(str(title), set()).add(int(sent_id))
            else:
                raise ValueError(f"Unsupported supporting fact item format: {item!r}")

        return mapping

    raise ValueError(
        f"Unsupported HotPotQA supporting_facts format: {type(supporting_facts).__name__}"
    )


def _build_sentence_metadata(
    sentence_texts: list[str],
    supporting_sentence_ids: set[int],
) -> list[dict[str, Any]]:
    """Build sentence metadata records compatible with the existing schema."""

    sentence_records: list[dict[str, Any]] = []
    cursor = 0

    for sentence_index, sentence_text in enumerate(sentence_texts):
        text = sentence_text.strip()
        start_char = cursor
        end_char = start_char + len(text)
        is_supporting = sentence_index in supporting_sentence_ids

        record: dict[str, Any] = {}

        _add_if_field(record, SentenceMetadata, "sentence_id", sentence_index)
        _add_if_field(record, SentenceMetadata, "sentence_index", sentence_index)
        _add_if_field(record, SentenceMetadata, "index", sentence_index)

        _add_if_field(record, SentenceMetadata, "text", text)
        _add_if_field(record, SentenceMetadata, "sentence_text", text)

        _add_if_field(record, SentenceMetadata, "start_char", start_char)
        _add_if_field(record, SentenceMetadata, "end_char", end_char)

        _add_if_field(record, SentenceMetadata, "is_supporting_fact", is_supporting)
        _add_if_field(record, SentenceMetadata, "is_supporting_sentence", is_supporting)
        _add_if_field(record, SentenceMetadata, "is_supporting", is_supporting)

        try:
            if hasattr(SentenceMetadata, "model_validate"):
                sentence = SentenceMetadata.model_validate(record)
                sentence_records.append(sentence.model_dump(mode="json"))
            else:
                sentence = SentenceMetadata.parse_obj(record)
                sentence_records.append(sentence.dict())
        except ValidationError as exc:
            raise ValueError(
                "Failed to validate SentenceMetadata record. "
                f"sentence_index={sentence_index}, record={record}"
            ) from exc

        cursor = end_char + 1

    return sentence_records


def _build_paragraph_chunk_record(
    *,
    example: dict[str, Any],
    split: str,
    example_index: int,
    doc_title: str,
    paragraph_index: int,
    sentence_texts: list[str],
    supporting_sentence_ids: set[int],
) -> dict[str, Any]:
    """Build and validate one ParagraphChunk-compatible record."""

    source_question_id = str(
        example.get("id")
        or example.get("_id")
        or f"hotpot_{split}_{example_index:06d}"
    )

    safe_title = _sanitize_title_for_id(doc_title)
    chunk_id = f"{source_question_id}::{safe_title}::p{paragraph_index}"

    paragraph_text = " ".join(sentence.strip() for sentence in sentence_texts if sentence.strip())
    chunk_text = f"Title: {doc_title}\nParagraph: {paragraph_text}"
    is_supporting_doc = bool(supporting_sentence_ids)

    sentence_records = _build_sentence_metadata(
        sentence_texts=sentence_texts,
        supporting_sentence_ids=supporting_sentence_ids,
    )

    record: dict[str, Any] = {}

    _add_if_field(record, ParagraphChunk, "chunk_id", chunk_id)

    _add_if_field(record, ParagraphChunk, "source_question_id", source_question_id)
    _add_if_field(record, ParagraphChunk, "question_id", source_question_id)

    _add_if_field(record, ParagraphChunk, "doc_title", doc_title)
    _add_if_field(record, ParagraphChunk, "title", doc_title)

    _add_if_field(record, ParagraphChunk, "paragraph_index", paragraph_index)
    _add_if_field(record, ParagraphChunk, "paragraph_text", paragraph_text)
    _add_if_field(record, ParagraphChunk, "chunk_text", chunk_text)
    _add_if_field(record, ParagraphChunk, "sentences", sentence_records)

    _add_if_field(record, ParagraphChunk, "is_supporting_doc", is_supporting_doc)
    _add_if_field(record, ParagraphChunk, "is_supporting_document", is_supporting_doc)

    sorted_supporting_sentence_ids = sorted(supporting_sentence_ids)

    _add_if_field(
        record,
        ParagraphChunk,
        "supporting_sentence_ids",
        sorted_supporting_sentence_ids,
    )
    _add_if_field(
        record,
        ParagraphChunk,
        "supporting_sentence_indices",
        sorted_supporting_sentence_ids,
    )

    _add_if_field(record, ParagraphChunk, "question", str(example.get("question", "")))
    _add_if_field(record, ParagraphChunk, "answer", str(example.get("answer", "")))
    _add_if_field(record, ParagraphChunk, "question_type", str(example.get("type", "")))
    _add_if_field(record, ParagraphChunk, "type", str(example.get("type", "")))
    _add_if_field(record, ParagraphChunk, "level", str(example.get("level", "")))
    _add_if_field(record, ParagraphChunk, "split", split)

    try:
        if hasattr(ParagraphChunk, "model_validate"):
            chunk = ParagraphChunk.model_validate(record)
            return chunk.model_dump(mode="json")

        chunk = ParagraphChunk.parse_obj(record)
        return chunk.dict()

    except ValidationError as exc:
        raise ValueError(
            "Failed to validate ParagraphChunk record. "
            f"source_question_id={source_question_id}, "
            f"doc_title={doc_title!r}, paragraph_index={paragraph_index}, "
            f"record={record}"
        ) from exc


def build_hotpotqa_chunks(
    *,
    dataset_name: str,
    config_name: str,
    split: str,
    limit: int | None,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """Load real HotPotQA examples and convert them into paragraph chunks."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Install it with `pip install datasets` "
            "or add `datasets` to requirements.txt and run `pip install -r requirements.txt`."
        ) from exc

    dataset_dict = load_dataset(dataset_name, config_name)

    if split not in dataset_dict:
        available = ", ".join(dataset_dict.keys())
        raise ValueError(f"Split {split!r} is not available. Available splits: {available}")

    dataset = dataset_dict[split]

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    chunks: list[dict[str, Any]] = []

    for example_index, example in enumerate(dataset):
        example_dict = dict(example)

        context_pairs = _extract_context(example_dict)
        supporting_fact_map = _extract_supporting_facts(example_dict)

        for paragraph_index, (doc_title, sentence_texts) in enumerate(context_pairs):
            supporting_sentence_ids = supporting_fact_map.get(doc_title, set())

            chunk_record = _build_paragraph_chunk_record(
                example=example_dict,
                split=split,
                example_index=example_index,
                doc_title=doc_title,
                paragraph_index=paragraph_index,
                sentence_texts=sentence_texts,
                supporting_sentence_ids=supporting_sentence_ids,
            )
            chunks.append(chunk_record)

    return chunks


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to JSONL deterministically."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare real HotPotQA data into EPSA-RAG paragraph chunks. "
            "Run without arguments to build the default research corpus."
        )
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset name. Default: {DEFAULT_DATASET_NAME}",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_NAME,
        help=f"HotPotQA dataset configuration. Default: {DEFAULT_CONFIG_NAME}",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to prepare. Default: {DEFAULT_SPLIT}",
    )
    parser.add_argument(
        "--limit",
        type=_parse_limit,
        default=DEFAULT_LIMIT,
        help=(
            f"Number of examples to process. Default: {DEFAULT_LIMIT}. "
            "Use 'none', 'all', or 'full' to process the full split."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=DEFAULT_SHUFFLE,
        help=f"Shuffle before selecting the limit. Default: {DEFAULT_SHUFFLE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used only when --shuffle is enabled. Default: {DEFAULT_SEED}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _resolve_project_path(args.output)

    chunks = build_hotpotqa_chunks(
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if not chunks:
        raise RuntimeError("No paragraph chunks were generated.")

    write_jsonl(output_path, chunks)

    supporting_count = sum(1 for chunk in chunks if chunk.get("is_supporting_doc") is True)
    unique_questions = {chunk.get("source_question_id") for chunk in chunks}
    unique_titles = {chunk.get("doc_title") for chunk in chunks}

    print("HotPotQA corpus preparation completed.")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit if args.limit is not None else 'full'}")
    print(f"Output path: {output_path}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Unique questions: {len(unique_questions)}")
    print(f"Unique document titles: {len(unique_titles)}")
    print(f"Supporting chunks: {supporting_count}")
    print(f"Non-supporting chunks: {len(chunks) - supporting_count}")


if __name__ == "__main__":
    main()