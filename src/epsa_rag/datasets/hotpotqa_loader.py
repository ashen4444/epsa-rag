from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from epsa_rag.datasets.schemas import HotPotQAExample


class HotPotQAFormatError(ValueError):
    """Raised when the raw HotPotQA file does not match the expected format."""


REQUIRED_FIELDS = {"_id", "question", "answer", "supporting_facts", "context"}


def load_hotpotqa_examples(
    input_path: str | Path,
    sample_size: int | None = None,
) -> list[HotPotQAExample]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"HotPotQA input file does not exist: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"HotPotQA input path is not a file: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list):
        raise HotPotQAFormatError("Expected HotPotQA file to contain a JSON list of examples.")

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer when provided.")
        raw_data = raw_data[:sample_size]

    examples: list[HotPotQAExample] = []

    for index, raw_example in enumerate(raw_data):
        examples.append(parse_hotpotqa_example(raw_example, example_index=index))

    return examples


def parse_hotpotqa_example(
    raw_example: dict[str, Any],
    example_index: int | None = None,
) -> HotPotQAExample:
    if not isinstance(raw_example, dict):
        location = _format_location(example_index)
        raise HotPotQAFormatError(f"{location} Expected each example to be a JSON object.")

    missing_fields = REQUIRED_FIELDS - set(raw_example.keys())
    if missing_fields:
        location = _format_location(example_index)
        missing = ", ".join(sorted(missing_fields))
        raise HotPotQAFormatError(f"{location} Missing required HotPotQA fields: {missing}")

    supporting_facts = _parse_supporting_facts(
        raw_example["supporting_facts"],
        example_index=example_index,
    )

    context = _parse_context(
        raw_example["context"],
        example_index=example_index,
    )

    try:
        return HotPotQAExample(
            source_question_id=raw_example["_id"],
            question=raw_example["question"],
            answer=raw_example["answer"],
            question_type=raw_example.get("type"),
            level=raw_example.get("level"),
            supporting_facts=supporting_facts,
            context=context,
        )
    except ValueError as exc:
        location = _format_location(example_index)
        raise HotPotQAFormatError(f"{location} Invalid HotPotQA example: {exc}") from exc


def _parse_supporting_facts(
    raw_supporting_facts: Any,
    example_index: int | None = None,
) -> list[tuple[str, int]]:
    if not isinstance(raw_supporting_facts, list):
        location = _format_location(example_index)
        raise HotPotQAFormatError(f"{location} supporting_facts must be a list.")

    parsed: list[tuple[str, int]] = []

    for fact_index, raw_fact in enumerate(raw_supporting_facts):
        if (
            not isinstance(raw_fact, list | tuple)
            or len(raw_fact) != 2
            or not isinstance(raw_fact[0], str)
            or not isinstance(raw_fact[1], int)
        ):
            location = _format_location(example_index)
            raise HotPotQAFormatError(
                f"{location} supporting_facts[{fact_index}] must be [title, sentence_index]."
            )

        title = raw_fact[0].strip()
        sentence_index = raw_fact[1]

        if not title:
            location = _format_location(example_index)
            raise HotPotQAFormatError(
                f"{location} supporting_facts[{fact_index}] has an empty title."
            )

        if sentence_index < 0:
            location = _format_location(example_index)
            raise HotPotQAFormatError(
                f"{location} supporting_facts[{fact_index}] has a negative sentence index."
            )

        parsed.append((title, sentence_index))

    return parsed


def _parse_context(
    raw_context: Any,
    example_index: int | None = None,
) -> list[tuple[str, list[str]]]:
    if not isinstance(raw_context, list):
        location = _format_location(example_index)
        raise HotPotQAFormatError(f"{location} context must be a list.")

    parsed: list[tuple[str, list[str]]] = []

    for doc_index, raw_doc in enumerate(raw_context):
        if not isinstance(raw_doc, list | tuple) or len(raw_doc) != 2:
            location = _format_location(example_index)
            raise HotPotQAFormatError(
                f"{location} context[{doc_index}] must be [title, sentences]."
            )

        title, sentences = raw_doc

        if not isinstance(title, str) or not title.strip():
            location = _format_location(example_index)
            raise HotPotQAFormatError(
                f"{location} context[{doc_index}] title must be a non-empty string."
            )

        if not isinstance(sentences, list):
            location = _format_location(example_index)
            raise HotPotQAFormatError(
                f"{location} context[{doc_index}] sentences must be a list."
            )

        if not all(isinstance(sentence, str) for sentence in sentences):
            location = _format_location(example_index)
            raise HotPotQAFormatError(
                f"{location} context[{doc_index}] contains a non-string sentence."
            )

        parsed.append((title.strip(), sentences))

    if not parsed:
        location = _format_location(example_index)
        raise HotPotQAFormatError(f"{location} context must contain at least one document.")

    return parsed


def _format_location(example_index: int | None) -> str:
    if example_index is None:
        return "HotPotQA example:"
    return f"HotPotQA example at index {example_index}:"