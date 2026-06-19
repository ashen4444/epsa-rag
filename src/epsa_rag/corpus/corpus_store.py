"""Corpus Store for processed HotPotQA paragraph chunks.

The CorpusStore is the central read/access layer over the processed JSONL
paragraph corpus. It intentionally does not implement retrieval, ranking,
indexing, BM25, dense search, FAISS, EPSA, or answer generation.

Its main responsibilities are:

1. Load processed ParagraphChunk records from JSONL.
2. Validate every record against the existing ParagraphChunk schema.
3. Preserve JSONL order for future index alignment.
4. Provide deterministic lookup/filtering APIs used by retrievers and EPSA.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from epsa_rag.datasets.schemas import ParagraphChunk


class CorpusStoreError(ValueError):
    """Base error for CorpusStore failures."""


class DuplicateChunkIdError(CorpusStoreError):
    """Raised when the corpus contains duplicate chunk IDs."""


class ChunkNotFoundError(CorpusStoreError):
    """Raised when a requested chunk ID does not exist in the corpus."""


class EmptyCorpusError(CorpusStoreError):
    """Raised when a JSONL corpus exists but contains no records."""


class CorpusStore:
    """In-memory access layer for processed paragraph chunks.

    The store preserves the original JSONL order. This is important because
    future retrieval indexes must align their numeric index positions with:

        all_chunks()[i]
        all_chunk_ids()[i]
        all_chunk_texts()[i]

    Parameters
    ----------
    chunks:
        ParagraphChunk objects in the same order as the source JSONL file.
    """

    def __init__(self, chunks: list[ParagraphChunk]) -> None:
        if not chunks:
            raise EmptyCorpusError("CorpusStore cannot be initialized with an empty chunk list.")

        self._chunks: list[ParagraphChunk] = list(chunks)
        self._chunks_by_id: dict[str, ParagraphChunk] = {}
        self._chunks_by_question_id: dict[str, list[ParagraphChunk]] = defaultdict(list)
        self._chunks_by_doc_title: dict[str, list[ParagraphChunk]] = defaultdict(list)

        for chunk in self._chunks:
            if chunk.chunk_id in self._chunks_by_id:
                raise DuplicateChunkIdError(
                    f"Duplicate chunk_id found in corpus: {chunk.chunk_id!r}"
                )

            self._chunks_by_id[chunk.chunk_id] = chunk
            self._chunks_by_question_id[chunk.source_question_id].append(chunk)
            self._chunks_by_doc_title[chunk.doc_title].append(chunk)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CorpusStore":
        """Load a CorpusStore from a processed JSONL corpus file.

        Each line must contain one JSON object matching the existing
        ParagraphChunk schema.

        Parameters
        ----------
        path:
            Path to a JSONL file containing processed paragraph chunks.

        Returns
        -------
        CorpusStore
            Loaded and validated corpus store.

        Raises
        ------
        CorpusStoreError
            If the path does not exist, the file is empty, a line is invalid
            JSON, a record does not match the ParagraphChunk schema, or a
            duplicate chunk_id is found.
        """

        corpus_path = Path(path)

        if not corpus_path.exists():
            raise CorpusStoreError(f"Corpus JSONL path does not exist: {corpus_path}")

        if not corpus_path.is_file():
            raise CorpusStoreError(f"Corpus JSONL path is not a file: {corpus_path}")

        chunks: list[ParagraphChunk] = []

        with corpus_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    record = json.loads(stripped_line)
                except json.JSONDecodeError as exc:
                    raise CorpusStoreError(
                        f"Invalid JSON on line {line_number} in {corpus_path}: {exc}"
                    ) from exc

                chunks.append(cls._record_to_chunk(record, line_number, corpus_path))

        if not chunks:
            raise EmptyCorpusError(f"Corpus JSONL file is empty: {corpus_path}")

        return cls(chunks)

    @staticmethod
    def _record_to_chunk(
        record: dict[str, Any],
        line_number: int,
        path: Path,
    ) -> ParagraphChunk:
        """Validate and convert one JSON record into a ParagraphChunk."""

        if not isinstance(record, dict):
            raise CorpusStoreError(
                f"Invalid record on line {line_number} in {path}: "
                "expected a JSON object."
            )

        try:
            if hasattr(ParagraphChunk, "model_validate"):
                return ParagraphChunk.model_validate(record)

            return ParagraphChunk.parse_obj(record)

        except ValidationError as exc:
            raise CorpusStoreError(
                f"Record on line {line_number} in {path} does not match "
                f"ParagraphChunk schema: {exc}"
            ) from exc

    def get_chunk(self, chunk_id: str) -> ParagraphChunk:
        """Return one chunk by chunk_id."""

        try:
            return self._chunks_by_id[chunk_id]
        except KeyError as exc:
            raise ChunkNotFoundError(f"Chunk not found: {chunk_id!r}") from exc

    def get_chunks(self, chunk_ids: list[str]) -> list[ParagraphChunk]:
        """Return chunks in the same order as the requested chunk_ids."""

        return [self.get_chunk(chunk_id) for chunk_id in chunk_ids]

    def get_by_question_id(self, source_question_id: str) -> list[ParagraphChunk]:
        """Return all chunks belonging to one HotPotQA source question."""

        return list(self._chunks_by_question_id.get(source_question_id, []))

    def get_by_doc_title(self, doc_title: str) -> list[ParagraphChunk]:
        """Return all chunks with an exact document title match."""

        return list(self._chunks_by_doc_title.get(doc_title, []))

    def get_supporting_chunks(self) -> list[ParagraphChunk]:
        """Return chunks marked as supporting documents."""

        return [chunk for chunk in self._chunks if chunk.is_supporting_doc]

    def get_non_supporting_chunks(self) -> list[ParagraphChunk]:
        """Return chunks not marked as supporting documents."""

        return [chunk for chunk in self._chunks if not chunk.is_supporting_doc]

    def all_chunks(self) -> list[ParagraphChunk]:
        """Return all chunks in original JSONL order."""

        return list(self._chunks)

    def all_chunk_ids(self) -> list[str]:
        """Return all chunk IDs in original JSONL order."""

        return [chunk.chunk_id for chunk in self._chunks]

    def all_chunk_texts(self) -> list[str]:
        """Return all chunk_text values in original JSONL order."""

        return [chunk.chunk_text for chunk in self._chunks]

    def stats(self) -> dict[str, int]:
        """Return deterministic corpus-level statistics."""

        supporting_chunks = len(self.get_supporting_chunks())

        return {
            "total_chunks": len(self._chunks),
            "unique_questions": len(self._chunks_by_question_id),
            "unique_doc_titles": len(self._chunks_by_doc_title),
            "supporting_chunks": supporting_chunks,
            "non_supporting_chunks": len(self._chunks) - supporting_chunks,
        }

    def __len__(self) -> int:
        """Return total number of chunks."""

        return len(self._chunks)