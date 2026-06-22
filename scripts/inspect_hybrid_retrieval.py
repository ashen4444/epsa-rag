from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import shutil
import textwrap
import yaml
from datetime import datetime

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


from epsa_rag.corpus.corpus_store import CorpusStore
from epsa_rag.retrieval.bm25_retriever import BM25Retriever
from epsa_rag.retrieval.dense_index import FaissDenseIndex
from epsa_rag.retrieval.dense_retriever import DenseRetriever
from epsa_rag.retrieval.embedding_backend import OpenAITextEmbedder
from epsa_rag.retrieval.hybrid_retriever import HybridRetriever
from epsa_rag.retrieval.retrieval_result import RetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval.yaml"

NUM_QUESTIONS_TO_INSPECT = 5
MAX_RETRIEVED_RESULTS_TO_PRINT = 10
MAX_PARAGRAPH_CHARS: int | None = None

OUTPUT_WIDTH = 110
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inspections"

def _print_wrapped_text(
    label: str,
    text: str,
    *,
    label_indent: str = "",
    text_indent: str = "  ",
    width: int | None = None,
) -> None:
    """
    Print long text in a readable wrapped format without changing the content.

    This only changes terminal layout. It does not add/remove retrieved content,
    supporting facts, questions, answers, or paragraph text.
    """

    terminal_width = shutil.get_terminal_size((OUTPUT_WIDTH, 20)).columns
    effective_width = width or min(max(terminal_width, 80), 140)

    print(f"{label_indent}{label}:")

    if not text:
        print(f"{text_indent}")
        return

    wrapped_text = textwrap.fill(
        text,
        width=effective_width,
        initial_indent=text_indent,
        subsequent_indent=text_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )

    print(wrapped_text)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return config


def _project_path(relative_path: str) -> Path:
    path = Path(relative_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean_line = line.strip()

            if not clean_line:
                continue

            try:
                record = json.loads(clean_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} in {path}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def _get_first_available(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)

        if value is not None:
            return value

    return None


def _as_clean_string(value: Any, default: str = "") -> str:
    if value is None:
        return default

    return str(value).strip()


def _get_question_id(chunk: dict[str, Any]) -> str:
    question_id = _get_first_available(
        chunk,
        [
            "question_id",
            "source_question_id",
            "hotpotqa_id",
            "_id",
            "id",
        ],
    )

    question_id_text = _as_clean_string(question_id)

    if not question_id_text:
        raise ValueError(
            "Could not find question id in chunk. Expected one of: "
            "question_id, source_question_id, hotpotqa_id, _id, id."
        )

    return question_id_text


def _get_chunk_id(chunk: dict[str, Any]) -> str:
    chunk_id = _as_clean_string(chunk.get("chunk_id"))

    if not chunk_id:
        raise ValueError("Chunk record is missing non-empty chunk_id.")

    return chunk_id


def _get_doc_title(chunk: dict[str, Any]) -> str:
    title = _get_first_available(
        chunk,
        [
            "doc_title",
            "title",
            "document_title",
        ],
    )

    return _as_clean_string(title, default="UNKNOWN_TITLE")


def _get_question_text(chunks: list[dict[str, Any]]) -> str:
    for chunk in chunks:
        question = _get_first_available(
            chunk,
            [
                "question",
                "query",
                "source_question",
                "question_text",
            ],
        )

        question_text = _as_clean_string(question)

        if question_text:
            return question_text

    return "QUESTION_TEXT_NOT_FOUND_IN_PROCESSED_CORPUS"


def _get_gold_answer(chunks: list[dict[str, Any]]) -> str:
    for chunk in chunks:
        answer = _get_first_available(
            chunk,
            [
                "answer",
                "gold_answer",
                "final_answer",
            ],
        )

        answer_text = _as_clean_string(answer)

        if answer_text:
            return answer_text

    return "ANSWER_NOT_FOUND_IN_PROCESSED_CORPUS"


def _is_supporting_doc(chunk: dict[str, Any], gold_supporting_titles: set[str]) -> bool:
    explicit_flag = chunk.get("is_supporting_doc")

    if isinstance(explicit_flag, bool):
        return explicit_flag

    title = _get_doc_title(chunk)

    return title in gold_supporting_titles


def _get_supporting_sentence_ids(chunk: dict[str, Any]) -> list[int]:
    value = _get_first_available(
        chunk,
        [
            "supporting_sentence_ids",
            "supporting_sentences",
            "gold_supporting_sentence_ids",
        ],
    )

    if value is None:
        return []

    if not isinstance(value, list):
        return []

    sentence_ids: list[int] = []

    for item in value:
        try:
            sentence_ids.append(int(item))
        except (TypeError, ValueError):
            continue

    return sentence_ids


def _get_sentences(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = chunk.get("sentences")

    if not isinstance(sentences, list):
        return []

    normalized: list[dict[str, Any]] = []

    for index, sentence in enumerate(sentences):
        if isinstance(sentence, dict):
            normalized.append(sentence)
        elif isinstance(sentence, str):
            normalized.append(
                {
                    "sentence_id": index,
                    "text": sentence,
                }
            )

    return normalized


def _get_sentence_text_by_id(chunk: dict[str, Any], sentence_id: int) -> str:
    for sentence in _get_sentences(chunk):
        current_sentence_id = sentence.get("sentence_id")

        try:
            current_sentence_id_int = int(current_sentence_id)
        except (TypeError, ValueError):
            continue

        if current_sentence_id_int == sentence_id:
            return _as_clean_string(sentence.get("text"))

    return ""


def _get_paragraph_text(chunk: dict[str, Any]) -> str:
    text = _get_first_available(
        chunk,
        [
            "paragraph_text",
            "chunk_text",
            "text",
            "content",
        ],
    )

    clean_text = " ".join(_as_clean_string(text).split())

    if MAX_PARAGRAPH_CHARS is not None and len(clean_text) > MAX_PARAGRAPH_CHARS:
        return clean_text[: MAX_PARAGRAPH_CHARS - 3] + "..."

    return clean_text


def _load_corpus_store(corpus_path: Path) -> CorpusStore:
    if hasattr(CorpusStore, "from_jsonl"):
        return CorpusStore.from_jsonl(corpus_path)

    if hasattr(CorpusStore, "from_path"):
        return CorpusStore.from_path(corpus_path)

    return CorpusStore(corpus_path)


def _build_bm25_retriever(
    corpus_store: CorpusStore,
    bm25_config: dict[str, Any],
) -> BM25Retriever:
    lowercase = bool(bm25_config.get("lowercase", True))
    remove_punctuation = bool(bm25_config.get("remove_punctuation", True))

    constructor_attempts = [
        lambda: BM25Retriever(
            corpus_store=corpus_store,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
        ),
        lambda: BM25Retriever(
            corpus_store=corpus_store,
        ),
        lambda: BM25Retriever(
            corpus_store,
        ),
        lambda: BM25Retriever(
            chunk_ids=corpus_store.all_chunk_ids(),
            documents=corpus_store.all_chunk_texts(),
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
        ),
        lambda: BM25Retriever(
            chunk_ids=corpus_store.all_chunk_ids(),
            documents=corpus_store.all_chunk_texts(),
        ),
        lambda: BM25Retriever(
            corpus_store.all_chunk_ids(),
            corpus_store.all_chunk_texts(),
        ),
        lambda: BM25Retriever.from_corpus_store(
            corpus_store=corpus_store,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
        ),
        lambda: BM25Retriever.from_corpus_store(corpus_store),
    ]

    last_error: Exception | None = None

    for attempt in constructor_attempts:
        try:
            return attempt()
        except (AttributeError, TypeError) as error:
            last_error = error

    raise TypeError(
        "Could not construct BM25Retriever with known constructor patterns."
    ) from last_error


def _load_dense_index(
    dense_index_path: Path,
    dense_metadata_path: Path,
) -> FaissDenseIndex:
    if hasattr(FaissDenseIndex, "load"):
        try:
            return FaissDenseIndex.load(
                index_path=dense_index_path,
                metadata_path=dense_metadata_path,
            )
        except TypeError:
            return FaissDenseIndex.load(
                dense_index_path,
                dense_metadata_path,
            )

    return FaissDenseIndex(
        index_path=dense_index_path,
        metadata_path=dense_metadata_path,
    )


def _build_embedder(dense_config: dict[str, Any]) -> OpenAITextEmbedder:
    model_name = str(dense_config["model_name"])
    batch_size = int(dense_config.get("batch_size", 32))

    constructor_attempts = [
        lambda: OpenAITextEmbedder(
            model_name=model_name,
            batch_size=batch_size,
        ),
        lambda: OpenAITextEmbedder(
            model_name=model_name,
        ),
        lambda: OpenAITextEmbedder(
            model_name,
        ),
    ]

    last_error: Exception | None = None

    for attempt in constructor_attempts:
        try:
            return attempt()
        except TypeError as error:
            last_error = error

    raise TypeError(
        "Could not construct OpenAITextEmbedder with known constructor patterns."
    ) from last_error


def _build_dense_retriever(
    dense_index: FaissDenseIndex,
    embedder: OpenAITextEmbedder,
    corpus_store: CorpusStore,
) -> DenseRetriever:
    constructor_attempts = [
        lambda: DenseRetriever(
            dense_index=dense_index,
            embedder=embedder,
            corpus_store=corpus_store,
        ),
        lambda: DenseRetriever(
            index=dense_index,
            embedder=embedder,
            corpus_store=corpus_store,
        ),
        lambda: DenseRetriever(
            dense_index=dense_index,
            embedder=embedder,
        ),
        lambda: DenseRetriever(
            index=dense_index,
            embedder=embedder,
        ),
        lambda: DenseRetriever(
            corpus_store,
            dense_index,
            embedder,
        ),
        lambda: DenseRetriever(
            dense_index,
            embedder,
        ),
    ]

    last_error: Exception | None = None

    for attempt in constructor_attempts:
        try:
            return attempt()
        except TypeError as error:
            last_error = error

    raise TypeError(
        "Could not construct DenseRetriever with known constructor patterns."
    ) from last_error


def _build_hybrid_retriever(
    corpus_store: CorpusStore,
    config: dict[str, Any],
) -> HybridRetriever:
    paths_config = config["paths"]
    retrieval_config = config["retrieval"]
    bm25_config = config["bm25"]
    dense_config = config["dense"]

    fusion_method = str(retrieval_config.get("fusion_method", "rrf")).lower()

    if fusion_method != "rrf":
        raise ValueError(
            "Hybrid inspection currently supports only fusion_method='rrf'."
        )

    dense_index_path = _project_path(paths_config["dense_index"])
    dense_metadata_path = _project_path(paths_config["dense_metadata"])

    bm25_retriever = _build_bm25_retriever(
        corpus_store=corpus_store,
        bm25_config=bm25_config,
    )

    dense_index = _load_dense_index(
        dense_index_path=dense_index_path,
        dense_metadata_path=dense_metadata_path,
    )

    embedder = _build_embedder(dense_config)

    dense_retriever = _build_dense_retriever(
        dense_index=dense_index,
        embedder=embedder,
        corpus_store=corpus_store,
    )

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        bm25_top_k=int(retrieval_config["bm25_top_k"]),
        dense_top_k=int(retrieval_config["dense_top_k"]),
        final_top_k=int(retrieval_config["top_k"]),
        rrf_k=int(retrieval_config["rrf_k"]),
    )


def _build_question_groups(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        question_id = _get_question_id(chunk)
        groups[question_id].append(chunk)

    return dict(groups)


def _get_gold_supporting_chunks(
    question_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    supporting_chunks = [
        chunk
        for chunk in question_chunks
        if chunk.get("is_supporting_doc") is True
        or len(_get_supporting_sentence_ids(chunk)) > 0
    ]

    return sorted(
        supporting_chunks,
        key=lambda chunk: (
            _get_doc_title(chunk),
            int(chunk.get("paragraph_index", 0) or 0),
        ),
    )


def _get_gold_supporting_titles(
    supporting_chunks: list[dict[str, Any]],
) -> set[str]:
    return {
        _get_doc_title(chunk)
        for chunk in supporting_chunks
        if _get_doc_title(chunk) != "UNKNOWN_TITLE"
    }


def _select_questions_for_inspection(
    question_groups: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    candidates: list[tuple[str, list[dict[str, Any]]]] = []

    for question_id, question_chunks in question_groups.items():
        supporting_chunks = _get_gold_supporting_chunks(question_chunks)

        if not supporting_chunks:
            continue

        candidates.append((question_id, question_chunks))

    candidates.sort(key=lambda item: item[0])

    return candidates[:NUM_QUESTIONS_TO_INSPECT]


def _is_gold_supporting_for_current_question(
    chunk: dict[str, Any],
    current_question_id: str,
    gold_supporting_titles: set[str],
) -> bool:
    """
    Return True only when the retrieved chunk is a gold supporting document
    for the question currently being inspected.

    Important:
    A chunk may have is_supporting_doc=True for its own original HotPotQA
    question, but that does not mean it supports the current inspected question.
    """

    chunk_question_id = _get_question_id(chunk)
    chunk_title = _get_doc_title(chunk)

    return (
        chunk_question_id == current_question_id
        and chunk_title in gold_supporting_titles
    )


def _print_gold_supporting_context(
    supporting_chunks: list[dict[str, Any]],
) -> None:
    print("Gold supporting documents / facts:")

    if not supporting_chunks:
        print("  SUPPORTING_CONTEXT_NOT_FOUND_IN_PROCESSED_CORPUS")
        return

    for index, chunk in enumerate(supporting_chunks, start=1):
        title = _get_doc_title(chunk)
        chunk_id = _get_chunk_id(chunk)
        sentence_ids = _get_supporting_sentence_ids(chunk)

        print()
        print(f"  {index}. Title: {title}")
        print(f"     Chunk ID: {chunk_id}")

        if sentence_ids:
            print(f"     Supporting sentence IDs: {sentence_ids}")

            for sentence_id in sentence_ids:
                sentence_text = _get_sentence_text_by_id(chunk, sentence_id)

                if sentence_text:
                    _print_wrapped_text(
                        label=f"s{sentence_id}",
                        text=sentence_text,
                        label_indent="     ",
                        text_indent="       ",
                    )
                else:
                    print(f"     s{sentence_id}: SENTENCE_TEXT_NOT_FOUND")
        else:
            print("     Supporting sentence IDs: NOT_FOUND")

        _print_wrapped_text(
            label="Paragraph",
            text=_get_paragraph_text(chunk),
            label_indent="     ",
            text_indent="       ",
        )


def _print_retrieved_results(
    results: list[RetrievalResult],
    chunk_by_id: dict[str, dict[str, Any]],
    question_id: str,
    gold_supporting_titles: set[str],
) -> tuple[int, int]:
    found_supporting_titles: set[str] = set()

    print("Hybrid retrieved chunks:")

    for result in results[:MAX_RETRIEVED_RESULTS_TO_PRINT]:
        chunk = chunk_by_id.get(result.chunk_id)

        if chunk is None:
            print(
                f"  Rank {result.rank:>2} | "
                f"score={result.score:.6f} | "
                f"chunk_id={result.chunk_id} | "
                "CHUNK_METADATA_NOT_FOUND"
            )
            continue

        title = _get_doc_title(chunk)
        is_current_gold_support = _is_gold_supporting_for_current_question(
            chunk=chunk,
            current_question_id=question_id,
            gold_supporting_titles=gold_supporting_titles,
        )

        support_label = (
            "SUPPORTING_DOC"
            if is_current_gold_support
            else "non-supporting"
        )

        if is_current_gold_support:
            found_supporting_titles.add(title)

        sentence_ids = (
            _get_supporting_sentence_ids(chunk)
            if is_current_gold_support
            else []
        )

        print(
            f"  Rank {result.rank:>2} | "
            f"score={result.score:.6f} | "
            f"{support_label} | "
            f"title={title}"
        )
        print(f"     Chunk ID: {result.chunk_id}")

        if sentence_ids:
            print(f"     Gold supporting sentence IDs in this chunk: {sentence_ids}")

        _print_wrapped_text(
            label="Paragraph",
            text=_get_paragraph_text(chunk),
            label_indent="     ",
            text_indent="       ",
        )
        print()

    return len(found_supporting_titles), len(gold_supporting_titles)

def _print_question_summary(
    found_supporting_count: int,
    total_supporting_count: int,
    top_k: int,
) -> None:
    if total_supporting_count == 0:
        print("Supporting-doc retrieval summary: no gold supporting docs available")
        return

    all_found = found_supporting_count == total_supporting_count

    print(
        "Supporting-doc retrieval summary: "
        f"{found_supporting_count}/{total_supporting_count} "
        f"gold supporting document titles found in top-{top_k}"
    )
    print(f"All gold supporting documents found: {all_found}")


def _safe_filename(value: str) -> str:
    safe_chars = []

    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    return "".join(safe_chars).strip("_")


def _write_hybrid_retrieval_markdown_report(
    question_id: str,
    question_chunks: list[dict[str, Any]],
    results: list[RetrievalResult],
    chunk_by_id: dict[str, dict[str, Any]],
    gold_supporting_titles: set[str],
    output_path: Path,
) -> None:
    question = _get_question_text(question_chunks)
    answer = _get_gold_answer(question_chunks)
    supporting_chunks = _get_gold_supporting_chunks(question_chunks)

    found_supporting_titles: set[str] = set()

    with output_path.open("w", encoding="utf-8") as file:
        file.write("# HotPotQA Hybrid Retrieval Inspection\n\n")

        file.write("## Question Metadata\n\n")
        file.write(f"- **Question ID:** `{question_id}`\n")
        file.write(f"- **Question:** {question}\n")
        file.write(f"- **Gold answer:** {answer}\n")
        file.write(
            f"- **Gold supporting document titles:** {sorted(gold_supporting_titles)}\n"
        )
        file.write(f"- **Generated at:** {datetime.now().isoformat(timespec='seconds')}\n\n")

        file.write("## Gold Supporting Documents / Facts\n\n")

        if not supporting_chunks:
            file.write("Supporting context not found in processed corpus.\n\n")
        else:
            for index, chunk in enumerate(supporting_chunks, start=1):
                title = _get_doc_title(chunk)
                chunk_id = _get_chunk_id(chunk)
                sentence_ids = _get_supporting_sentence_ids(chunk)

                file.write(f"### Gold Supporting Document {index}: {title}\n\n")
                file.write(f"- **Chunk ID:** `{chunk_id}`\n")
                file.write(f"- **Supporting sentence IDs:** {sentence_ids}\n\n")

                if sentence_ids:
                    file.write("#### Supporting Sentences\n\n")

                    for sentence_id in sentence_ids:
                        sentence_text = _get_sentence_text_by_id(chunk, sentence_id)

                        if sentence_text:
                            file.write(f"- **s{sentence_id}:** {sentence_text}\n")
                        else:
                            file.write(f"- **s{sentence_id}:** SENTENCE_TEXT_NOT_FOUND\n")

                    file.write("\n")

                file.write("#### Full Paragraph\n\n")
                file.write(f"{_get_paragraph_text(chunk)}\n\n")
                file.write("---\n\n")

        file.write("## Hybrid Retrieved Chunks\n\n")

        for result in results[:MAX_RETRIEVED_RESULTS_TO_PRINT]:
            chunk = chunk_by_id.get(result.chunk_id)

            if chunk is None:
                file.write(f"### Rank {result.rank}: Metadata Not Found\n\n")
                file.write(f"- **Fusion score:** {result.score:.6f}\n")
                file.write(f"- **Chunk ID:** `{result.chunk_id}`\n\n")
                file.write("---\n\n")
                continue

            title = _get_doc_title(chunk)
            is_supporting = _is_gold_supporting_for_current_question(
                chunk=chunk,
                current_question_id=question_id,
                gold_supporting_titles=gold_supporting_titles,
            )
            support_label = "SUPPORTING_DOC" if is_supporting else "non-supporting"

            if is_supporting:
                found_supporting_titles.add(title)

            sentence_ids = _get_supporting_sentence_ids(chunk) if is_supporting else []

            file.write(f"### Rank {result.rank}: {title}\n\n")
            file.write(f"- **Fusion score:** {result.score:.6f}\n")
            file.write(f"- **Chunk ID:** `{result.chunk_id}`\n")
            file.write(f"- **Retrieval label:** {support_label}\n")

            if sentence_ids:
                file.write(
                    f"- **Gold supporting sentence IDs in this chunk:** {sentence_ids}\n"
                )

            file.write("\n#### Paragraph\n\n")
            file.write(f"{_get_paragraph_text(chunk)}\n\n")
            file.write("---\n\n")

        file.write("## Supporting-Document Retrieval Summary\n\n")

        total_supporting_count = len(gold_supporting_titles)
        found_gold_supporting_titles = found_supporting_titles & gold_supporting_titles
        found_supporting_count = len(found_gold_supporting_titles)

        if total_supporting_count == 0:
            file.write("No gold supporting documents available.\n")
        else:
            missing_supporting_titles = gold_supporting_titles - found_gold_supporting_titles
            all_found = len(missing_supporting_titles) == 0

            file.write(
                f"- **Gold supporting document titles found:** "
                f"{found_supporting_count}/{total_supporting_count}\n"
            )
            file.write(
                f"- **Top-k inspected:** "
                f"{min(MAX_RETRIEVED_RESULTS_TO_PRINT, len(results))}\n"
            )
            file.write(f"- **All gold supporting documents found:** {all_found}\n")
            file.write(f"- **Found supporting titles:** {sorted(found_gold_supporting_titles)}\n")
            file.write(f"- **Missing supporting titles:** {sorted(missing_supporting_titles)}\n")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    config = _load_yaml_config(CONFIG_PATH)
    corpus_path = _project_path(config["paths"]["processed_corpus"])

    raw_chunks = _load_jsonl(corpus_path)
    chunk_by_id = {
        _get_chunk_id(chunk): chunk
        for chunk in raw_chunks
    }

    question_groups = _build_question_groups(raw_chunks)
    selected_questions = _select_questions_for_inspection(question_groups)

    if not selected_questions:
        first_record_keys = sorted(raw_chunks[0].keys())
        raise ValueError(
            "Could not find any questions with supporting documents in the "
            "processed corpus. Check whether prepare_hotpotqa.py stores "
            "is_supporting_doc or supporting_sentence_ids. "
            f"First record keys: {first_record_keys}"
        )

    corpus_store = _load_corpus_store(corpus_path)
    hybrid_retriever = _build_hybrid_retriever(
        corpus_store=corpus_store,
        config=config,
    )

    print("HotPotQA Hybrid Retrieval Inspection")
    print("=" * 100)
    print(f"Corpus path: {corpus_path}")
    print(f"Questions available in processed corpus: {len(question_groups)}")
    print(f"Questions inspected: {len(selected_questions)}")
    print(f"Hybrid final top-k: {hybrid_retriever.final_top_k}")
    print(f"Printed retrieved results per question: {MAX_RETRIEVED_RESULTS_TO_PRINT}")
    print("=" * 100)

    for question_number, (question_id, question_chunks) in enumerate(
        selected_questions,
        start=1,
    ):
        question = _get_question_text(question_chunks)
        answer = _get_gold_answer(question_chunks)
        supporting_chunks = _get_gold_supporting_chunks(question_chunks)
        gold_supporting_titles = _get_gold_supporting_titles(supporting_chunks)

        print()
        print("=" * 100)
        print(f"Question {question_number}")
        print("=" * 100)
        print(f"Question ID: {question_id}")
        _print_wrapped_text(
            label="Question",
            text=question,
            label_indent="",
            text_indent="  ",
        )
        print(f"Gold answer: {answer}")
        print(f"Gold supporting document titles: {sorted(gold_supporting_titles)}")
        print()

        _print_gold_supporting_context(supporting_chunks)

        if question == "QUESTION_TEXT_NOT_FOUND_IN_PROCESSED_CORPUS":
            print(
                "Skipping retrieval for this question because the processed "
                "corpus does not contain the original HotPotQA question text."
            )
            continue

        results = hybrid_retriever.search(question)

        found_supporting_count, total_supporting_count = _print_retrieved_results(
            results=results,
            chunk_by_id=chunk_by_id,
            question_id=question_id,
            gold_supporting_titles=gold_supporting_titles,
        )

        _print_question_summary(
            found_supporting_count=found_supporting_count,
            total_supporting_count=total_supporting_count,
            top_k=min(hybrid_retriever.final_top_k, MAX_RETRIEVED_RESULTS_TO_PRINT),
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = (
            OUTPUT_DIR
            / f"hybrid_retrieval_{_safe_filename(question_id)}.md"
        )

        _write_hybrid_retrieval_markdown_report(
            question_id=question_id,
            question_chunks=question_chunks,
            results=results,
            chunk_by_id=chunk_by_id,
            gold_supporting_titles=gold_supporting_titles,
            output_path=output_path,
        )

        print(f"Markdown report saved to: {output_path}")


if __name__ == "__main__":
    main()