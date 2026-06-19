from __future__ import annotations

import re

from epsa_rag.datasets.schemas import (
    HotPotQAExample,
    ParagraphChunk,
    SentenceMetadata,
)


def build_paragraph_chunks(
    examples: list[HotPotQAExample],
) -> list[ParagraphChunk]:
    chunks: list[ParagraphChunk] = []

    for example in examples:
        chunks.extend(build_chunks_for_example(example))

    return chunks


def build_chunks_for_example(
    example: HotPotQAExample,
) -> list[ParagraphChunk]:
    chunks: list[ParagraphChunk] = []

    supporting_fact_map = _build_supporting_fact_map(example.supporting_facts)

    for paragraph_index, (doc_title, raw_sentences) in enumerate(example.context):
        paragraph_text, sentence_metadata = build_sentence_metadata(raw_sentences)
        supporting_sentence_ids = supporting_fact_map.get(doc_title, [])

        chunk = ParagraphChunk(
            chunk_id=build_chunk_id(
                source_question_id=example.source_question_id,
                doc_title=doc_title,
                paragraph_index=paragraph_index,
            ),
            source_question_id=example.source_question_id,
            question=example.question,
            answer=example.answer,
            question_type=example.question_type,
            level=example.level,
            doc_title=doc_title,
            paragraph_index=paragraph_index,
            chunk_text=f"Title: {doc_title}\nParagraph: {paragraph_text}",
            paragraph_text=paragraph_text,
            sentences=sentence_metadata,
            supporting_sentence_ids=supporting_sentence_ids,
            is_supporting_doc=len(supporting_sentence_ids) > 0,
        )
        chunks.append(chunk)

    return chunks


def build_sentence_metadata(
    sentences: list[str],
) -> tuple[str, list[SentenceMetadata]]:
    normalized_sentences = [sentence.strip() for sentence in sentences]
    paragraph_text = " ".join(normalized_sentences)

    metadata: list[SentenceMetadata] = []
    cursor = 0

    for sentence_id, sentence in enumerate(normalized_sentences):
        start_char = cursor
        end_char = start_char + len(sentence)

        metadata.append(
            SentenceMetadata(
                sentence_id=sentence_id,
                text=sentence,
                start_char=start_char,
                end_char=end_char,
            )
        )

        cursor = end_char + 1

    return paragraph_text, metadata


def build_chunk_id(
    source_question_id: str,
    doc_title: str,
    paragraph_index: int,
) -> str:
    safe_title = make_safe_doc_title(doc_title)
    return f"{source_question_id}::{safe_title}::p{paragraph_index}"


def make_safe_doc_title(doc_title: str) -> str:
    safe = doc_title.strip()
    safe = safe.replace(" ", "_")
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "", safe)
    safe = re.sub(r"_+", "_", safe)
    return safe or "untitled"


def _build_supporting_fact_map(
    supporting_facts: list[tuple[str, int]],
) -> dict[str, list[int]]:
    fact_map: dict[str, list[int]] = {}

    for doc_title, sentence_id in supporting_facts:
        fact_map.setdefault(doc_title, []).append(sentence_id)

    for doc_title in fact_map:
        fact_map[doc_title] = sorted(set(fact_map[doc_title]))

    return fact_map