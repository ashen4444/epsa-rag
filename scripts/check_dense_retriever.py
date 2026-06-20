from __future__ import annotations

from epsa_rag.config.retrieval_config import load_retrieval_settings
from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.dense_retriever import DenseRetriever
from epsa_rag.retrieval.embedding_backend import OpenAITextEmbedder


SAMPLE_QUERIES = [
    "Who discovered radium?",
    "Where was the director of Inception born?",
    "What is the capital of England?",
]


def main() -> None:
    settings = load_retrieval_settings()

    corpus_store = CorpusStore.from_jsonl(settings.paths.processed_corpus)

    embedder = OpenAITextEmbedder(
        model_name_or_path=settings.dense.model_name,
        batch_size=settings.dense.batch_size,
    )

    retriever = DenseRetriever.load(
        corpus_store=corpus_store,
        index_path=settings.paths.dense_index,
        metadata_path=settings.paths.dense_metadata,
        embedder=embedder,
    )

    print("Dense retriever loaded successfully.")
    print(f"Corpus path: {settings.paths.processed_corpus}")
    print(f"Index path: {settings.paths.dense_index}")
    print(f"Metadata path: {settings.paths.dense_metadata}")
    print(f"Embedding model: {embedder.model_name}")
    print(f"Index size: {retriever.index_size}")
    print(f"Embedding dimension: {retriever.embedding_dimension}")

    top_k = min(5, settings.retrieval.dense_top_k)

    for query in SAMPLE_QUERIES:
        print("\n" + "=" * 88)
        print(f"Query: {query}")

        results = retriever.search(query=query, top_k=top_k)

        for result in results:
            chunk = corpus_store.get_chunk(result.chunk_id)
            preview = chunk.chunk_text.replace("\n", " ")[:180]

            print(
                f"rank={result.rank} "
                f"score={result.score:.4f} "
                f"chunk_id={result.chunk_id} "
                f"title={chunk.doc_title}"
            )
            print(f"preview={preview}")


if __name__ == "__main__":
    main()