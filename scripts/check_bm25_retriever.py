from pathlib import Path
from typing import Any

from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.bm25_retriever import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "hotpotqa_paragraph_chunks.jsonl"


def _field(obj: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field_name, default)

    return getattr(obj, field_name, default)


def _preview(text: str, max_chars: int = 180) -> str:
    clean_text = " ".join(text.split())

    if len(clean_text) <= max_chars:
        return clean_text

    return clean_text[:max_chars].rstrip() + "..."


def main() -> None:
    if not DEFAULT_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Processed corpus not found: {DEFAULT_CORPUS_PATH}\n"
            "Run scripts/prepare_hotpotqa.py first."
        )

    corpus_store = CorpusStore.from_jsonl(DEFAULT_CORPUS_PATH)
    retriever = BM25Retriever.from_corpus_store(corpus_store)

    queries = [
        "radium discovery",
        "film director birthplace",
        "capital city population",
    ]

    print("BM25 Retriever Smoke Check")
    print("=" * 80)
    print(f"Corpus path: {DEFAULT_CORPUS_PATH}")
    print(f"Corpus size: {retriever.corpus_size}")
    print()

    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)

        results = retriever.search(query=query, top_k=5)

        for result in results:
            chunk = corpus_store.get_chunk(result.chunk_id)

            title = _field(chunk, "doc_title", "<unknown title>")
            chunk_text = _field(chunk, "chunk_text", "")

            print(
                f"rank={result.rank} | "
                f"score={result.score:.4f} | "
                f"chunk_id={result.chunk_id} | "
                f"title={title}"
            )
            print(f"preview={_preview(chunk_text)}")
            print()

        print("=" * 80)
        print()


if __name__ == "__main__":
    main()