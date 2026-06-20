from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class DenseSearchHit:
    rank: int
    chunk_id: str
    score: float


@dataclass
class FaissDenseIndex:
    """
    FAISS-backed dense vector index.

    For cosine similarity:
        1. document embeddings are L2-normalized before indexing
        2. query embeddings are L2-normalized before search
        3. FAISS IndexFlatIP returns inner product scores
    """

    chunk_ids: tuple[str, ...]
    index: Any
    dimension: int
    normalize_embeddings: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_embeddings(
        cls,
        chunk_ids: Sequence[str],
        embeddings: np.ndarray,
        normalize_embeddings: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> FaissDenseIndex:
        clean_chunk_ids = _validate_chunk_ids(chunk_ids)
        vectors = _validate_embeddings(
            embeddings=embeddings,
            expected_count=len(clean_chunk_ids),
        )

        if normalize_embeddings:
            vectors = _l2_normalize_matrix(vectors)

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for dense retrieval. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        dimension = int(vectors.shape[1])

        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        return cls(
            chunk_ids=tuple(clean_chunk_ids),
            index=index,
            dimension=dimension,
            normalize_embeddings=normalize_embeddings,
            metadata=dict(metadata or {}),
        )

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[DenseSearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        if self.size < 1:
            raise ValueError("Dense index is empty.")

        query_vector = _validate_query_embedding(
            query_embedding=query_embedding,
            expected_dimension=self.dimension,
        )

        if self.normalize_embeddings:
            query_vector = _l2_normalize_vector(query_vector)

        query_matrix = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))

        search_k = self.size
        scores, indices = self.index.search(query_matrix, search_k)

        ranked_pairs: list[tuple[str, float]] = []

        for raw_score, raw_index in zip(scores[0], indices[0], strict=False):
            index_position = int(raw_index)

            if index_position < 0:
                continue

            ranked_pairs.append(
                (
                    self.chunk_ids[index_position],
                    float(raw_score),
                )
            )

        ranked_pairs.sort(key=lambda item: (-item[1], item[0]))

        limited_pairs = ranked_pairs[: min(top_k, len(ranked_pairs))]

        return [
            DenseSearchHit(rank=rank, chunk_id=chunk_id, score=score)
            for rank, (chunk_id, score) in enumerate(limited_pairs, start=1)
        ]

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)

        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required to save dense indexes. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        faiss.write_index(self.index, str(index_file))

        metadata_payload = {
            "index_type": "faiss.IndexFlatIP",
            "similarity": "cosine" if self.normalize_embeddings else "inner_product",
            "normalize_embeddings": self.normalize_embeddings,
            "dimension": self.dimension,
            "size": self.size,
            "chunk_ids": list(self.chunk_ids),
            "metadata": self.metadata,
        }

        with metadata_file.open("w", encoding="utf-8") as file:
            json.dump(metadata_payload, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> FaissDenseIndex:
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS dense index file not found: {index_file}")

        if not metadata_file.exists():
            raise FileNotFoundError(f"Dense metadata file not found: {metadata_file}")

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required to load dense indexes. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        with metadata_file.open("r", encoding="utf-8") as file:
            metadata_payload = json.load(file)

        chunk_ids = _validate_chunk_ids(metadata_payload["chunk_ids"])
        dimension = int(metadata_payload["dimension"])
        normalize_embeddings = bool(metadata_payload["normalize_embeddings"])

        index = faiss.read_index(str(index_file))

        if index.ntotal != len(chunk_ids):
            raise ValueError(
                "FAISS index vector count does not match metadata chunk ID count. "
                f"Index has {index.ntotal}, metadata has {len(chunk_ids)}."
            )

        if index.d != dimension:
            raise ValueError(
                "FAISS index dimension does not match metadata dimension. "
                f"Index has {index.d}, metadata has {dimension}."
            )

        metadata = metadata_payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Dense metadata field must be a dictionary.")

        return cls(
            chunk_ids=tuple(chunk_ids),
            index=index,
            dimension=dimension,
            normalize_embeddings=normalize_embeddings,
            metadata=metadata,
        )


def _validate_chunk_ids(chunk_ids: Sequence[str]) -> list[str]:
    if not chunk_ids:
        raise ValueError("chunk_ids must contain at least one item.")

    cleaned_chunk_ids: list[str] = []

    for chunk_id in chunk_ids:
        if not isinstance(chunk_id, str):
            raise TypeError("Every chunk_id must be a string.")

        cleaned = chunk_id.strip()
        if not cleaned:
            raise ValueError("chunk_id must not be empty.")

        cleaned_chunk_ids.append(cleaned)

    if len(set(cleaned_chunk_ids)) != len(cleaned_chunk_ids):
        raise ValueError("chunk_ids must be unique.")

    return cleaned_chunk_ids


def _validate_embeddings(
    embeddings: np.ndarray,
    expected_count: int,
) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")

    if vectors.shape[0] != expected_count:
        raise ValueError(
            "embeddings row count must match chunk_ids length. "
            f"Expected {expected_count}, got {vectors.shape[0]}."
        )

    if vectors.shape[1] < 1:
        raise ValueError("embeddings must have at least one dimension.")

    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must not contain NaN or infinite values.")

    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("embeddings must not contain zero vectors.")

    return vectors


def _validate_query_embedding(
    query_embedding: np.ndarray,
    expected_dimension: int,
) -> np.ndarray:
    vector = np.asarray(query_embedding, dtype=np.float32)

    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]

    if vector.ndim != 1:
        raise ValueError("query_embedding must be a 1D vector.")

    if vector.shape[0] != expected_dimension:
        raise ValueError(
            "query_embedding dimension mismatch. "
            f"Expected {expected_dimension}, got {vector.shape[0]}."
        )

    if not np.isfinite(vector).all():
        raise ValueError("query_embedding must not contain NaN or infinite values.")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("query_embedding must not be a zero vector.")

    return vector


def _l2_normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms == 0.0):
        raise ValueError("Cannot normalize embeddings because at least one vector is zero.")

    return vectors / norms[:, None]


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))

    if norm == 0.0:
        raise ValueError("Cannot normalize query embedding because it is a zero vector.")

    return vector / norm