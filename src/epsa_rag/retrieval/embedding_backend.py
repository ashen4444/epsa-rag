from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from dotenv import load_dotenv


class TextEmbedder(Protocol):
    """Production interface for text embedding backends."""

    @property
    def model_name(self) -> str:
        """Embedding model name."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of corpus texts."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one retrieval query."""
        ...


@dataclass(frozen=True)
class OpenAITextEmbedder:
    """
    Production OpenAI embedding backend for dense semantic retrieval.

    Default model:
        text-embedding-3-small

    Future upgrade:
        text-embedding-3-large

    API key is loaded from .env using OPENAI_API_KEY.
    """

    model_name_or_path: str | None = None
    batch_size: int | None = None
    dimensions: int | None = None

    def __post_init__(self) -> None:
        load_dotenv()

        resolved_model = self.model_name_or_path or os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

        resolved_batch_size = self.batch_size or int(
            os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "64")
        )

        if not resolved_model.strip():
            raise ValueError("OpenAI embedding model name must not be empty.")

        if resolved_batch_size < 1:
            raise ValueError("OpenAI embedding batch size must be >= 1.")

        object.__setattr__(self, "model_name_or_path", resolved_model)
        object.__setattr__(self, "batch_size", resolved_batch_size)

    @property
    def model_name(self) -> str:
        return str(self.model_name_or_path)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        cleaned_texts = [self._validate_text(text, "text") for text in texts]

        if not cleaned_texts:
            raise ValueError("texts must contain at least one item.")

        embeddings: list[list[float]] = []

        for start_index in range(0, len(cleaned_texts), int(self.batch_size)):
            batch = cleaned_texts[start_index : start_index + int(self.batch_size)]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return self._to_numpy_embeddings(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        cleaned_query = self._validate_text(query, "query")
        embedding = self._embed_batch([cleaned_query])[0]
        return np.asarray(embedding, dtype=np.float32)

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAITextEmbedder. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to the project-root .env file."
            )

        client = OpenAI()

        request_kwargs: dict[str, object] = {
            "model": self.model_name,
            "input": list(texts),
        }

        if self.dimensions is not None:
            request_kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(**request_kwargs)

        sorted_items = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in sorted_items]

    @staticmethod
    def _validate_text(value: str, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")

        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty.")

        return cleaned

    @staticmethod
    def _to_numpy_embeddings(embeddings: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(embeddings, dtype=np.float32)

        if array.ndim != 2:
            raise ValueError("OpenAI embeddings response must be a 2D array.")

        if array.shape[0] < 1:
            raise ValueError("OpenAI embeddings response is empty.")

        if array.shape[1] < 1:
            raise ValueError("OpenAI embeddings must have at least one dimension.")

        if not np.isfinite(array).all():
            raise ValueError("OpenAI embeddings contain NaN or infinite values.")

        return array