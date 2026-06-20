from __future__ import annotations

from epsa_rag.config.retrieval_config import load_retrieval_settings
from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.dense_retriever import DenseRetriever
from epsa_rag.retrieval.embedding_backend import OpenAITextEmbedder


def main() -> None:
    settings = load_retrieval_settings()

    corpus_store = CorpusStore.from_jsonl(settings.paths.processed_corpus)

    embedder = OpenAITextEmbedder(
        model_name_or_path=settings.dense.model_name,
        batch_size=settings.dense.batch_size,
    )

    retriever = DenseRetriever.from_corpus_store(
        corpus_store=corpus_store,
        embedder=embedder,
        normalize_embeddings=settings.dense.normalize_embeddings,
    )

    retriever.save_index(
        index_path=settings.paths.dense_index,
        metadata_path=settings.paths.dense_metadata,
    )

    print("Dense FAISS index built successfully.")
    print(f"Corpus path: {settings.paths.processed_corpus}")
    print(f"Index path: {settings.paths.dense_index}")
    print(f"Metadata path: {settings.paths.dense_metadata}")
    print(f"Embedding provider: {settings.dense.provider}")
    print(f"Embedding model: {embedder.model_name}")
    print(f"Index size: {retriever.index_size}")
    print(f"Embedding dimension: {retriever.embedding_dimension}")
    print(f"Normalize embeddings: {settings.dense.normalize_embeddings}")


if __name__ == "__main__":
    main()