from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RETRIEVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    bm25_top_k: int
    dense_top_k: int
    fusion_method: str
    rrf_k: int


@dataclass(frozen=True)
class DenseConfig:
    provider: str
    model_name: str
    normalize_embeddings: bool
    batch_size: int


@dataclass(frozen=True)
class PathsConfig:
    processed_corpus: Path
    bm25_index: Path
    dense_index: Path
    dense_metadata: Path


@dataclass(frozen=True)
class RetrievalSettings:
    retrieval: RetrievalConfig
    dense: DenseConfig
    paths: PathsConfig


def load_retrieval_settings(
    config_path: str | Path = DEFAULT_RETRIEVAL_CONFIG_PATH,
) -> RetrievalSettings:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Retrieval config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError("retrieval.yaml must contain a YAML object at the top level.")

    retrieval_section = _require_section(raw_config, "retrieval")
    dense_section = _require_section(raw_config, "dense")
    paths_section = _require_section(raw_config, "paths")

    retrieval = RetrievalConfig(
        top_k=_positive_int(retrieval_section, "top_k"),
        bm25_top_k=_positive_int(retrieval_section, "bm25_top_k"),
        dense_top_k=_positive_int(retrieval_section, "dense_top_k"),
        fusion_method=_non_empty_str(retrieval_section, "fusion_method"),
        rrf_k=_positive_int(retrieval_section, "rrf_k"),
    )

    dense = DenseConfig(
        provider=_non_empty_str(dense_section, "provider"),
        model_name=_non_empty_str(dense_section, "model_name"),
        normalize_embeddings=_bool_value(dense_section, "normalize_embeddings"),
        batch_size=_positive_int(dense_section, "batch_size"),
    )

    if dense.provider.lower() != "openai":
        raise ValueError(
            "Only dense.provider='openai' is supported in the current production dense retriever."
        )

    paths = PathsConfig(
        processed_corpus=_project_path(_non_empty_str(paths_section, "processed_corpus")),
        bm25_index=_project_path(_non_empty_str(paths_section, "bm25_index")),
        dense_index=_project_path(_non_empty_str(paths_section, "dense_index")),
        dense_metadata=_project_path(_non_empty_str(paths_section, "dense_metadata")),
    )

    return RetrievalSettings(
        retrieval=retrieval,
        dense=dense,
        paths=paths,
    )


def _require_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(f"retrieval.yaml must contain a '{section_name}' section.")

    return section


def _non_empty_str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)

    if not isinstance(value, str):
        raise TypeError(f"Config value '{key}' must be a string.")

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Config value '{key}' must not be empty.")

    return cleaned


def _positive_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)

    if not isinstance(value, int):
        raise TypeError(f"Config value '{key}' must be an integer.")

    if value < 1:
        raise ValueError(f"Config value '{key}' must be >= 1.")

    return value


def _bool_value(section: dict[str, Any], key: str) -> bool:
    value = section.get(key)

    if not isinstance(value, bool):
        raise TypeError(f"Config value '{key}' must be a boolean.")

    return value


def _project_path(path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path