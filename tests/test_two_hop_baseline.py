from __future__ import annotations

from dataclasses import dataclass

from epsa_rag.rag.llm_client import ChatMessage, LLMResponse
from epsa_rag.rag.two_hop_baseline import (
    RAGDocument,
    TwoHopBaselineConfig,
    TwoHopHybridRAGBaseline,
    format_documents_for_prompt,
    merge_unique_documents,
    sanitize_generated_query,
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_title: str
    chunk_text: str
    paragraph_index: int


@dataclass(frozen=True)
class RetrievalItem:
    chunk_id: str
    rank: int
    fusion_score: float


class InMemoryCorpusStore:
    def __init__(self) -> None:
        self.chunks = {
            "c1": Chunk(
                chunk_id="c1",
                doc_title="Inception",
                chunk_text="Title: Inception\nParagraph: Inception was directed by Christopher Nolan.",
                paragraph_index=0,
            ),
            "c2": Chunk(
                chunk_id="c2",
                doc_title="Christopher Nolan",
                chunk_text="Title: Christopher Nolan\nParagraph: Christopher Nolan was born in London.",
                paragraph_index=0,
            ),
            "d1": Chunk(
                chunk_id="d1",
                doc_title="London Film Festival",
                chunk_text="Title: London Film Festival\nParagraph: A film festival held in London.",
                paragraph_index=0,
            ),
        }

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[chunk_id]


class InMemoryRetriever:
    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        if "birthplace" in query.casefold():
            return [
                RetrievalItem(chunk_id="c2", rank=1, fusion_score=0.9),
                RetrievalItem(chunk_id="c1", rank=2, fusion_score=0.7),
            ][:top_k]

        return [
            RetrievalItem(chunk_id="c1", rank=1, fusion_score=0.95),
            RetrievalItem(chunk_id="d1", rank=2, fusion_score=0.4),
        ][:top_k]


class SequencedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.messages: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.messages.append(messages)
        return self.responses.pop(0)


def test_two_hop_baseline_runs_fixed_two_hop_and_merges_unique_context() -> None:
    llm = SequencedLLM(
        [
            LLMResponse(
                content="Christopher Nolan birthplace",
                prompt_tokens=20,
                completion_tokens=4,
                total_tokens=24,
                model_name="gpt-4o-mini",
            ),
            LLMResponse(
                content="London",
                prompt_tokens=50,
                completion_tokens=1,
                total_tokens=51,
                model_name="gpt-4o-mini",
            ),
        ]
    )

    baseline = TwoHopHybridRAGBaseline(
        retriever=InMemoryRetriever(),
        corpus_store=InMemoryCorpusStore(),
        llm_client=llm,
        config=TwoHopBaselineConfig(hop1_top_k=2, hop2_top_k=2),
    )

    result = baseline.run(
        question_id="q1",
        question="Where was the director of Inception born?",
        gold_answer="London",
        gold_supporting_titles=["Inception", "Christopher Nolan"],
    )

    assert result.generated_hop2_query == "Christopher Nolan birthplace"

    assert result.hop1_retrieved_chunk_ids == ["c1", "d1"]
    assert result.hop2_retrieved_chunk_ids == ["c2", "c1"]

    assert result.merged_context_chunk_ids == ["c1", "d1", "c2"]
    assert result.num_context_documents == 3

    assert result.final_answer == "London"

    assert result.exact_match == 1.0
    assert result.partial_match == 1.0
    assert result.answer_precision == 1.0
    assert result.answer_recall == 1.0
    assert result.answer_f1 == 1.0

    assert result.prompt_tokens == 70
    assert result.completion_tokens == 5
    assert result.total_tokens == 75

    assert result.hop2_query_generation_error is None
    assert result.final_answer_generation_error is None
    assert result.retrieval_error is None


def test_two_hop_baseline_result_does_not_log_supporting_doc_metrics() -> None:
    llm = SequencedLLM(
        [
            LLMResponse(content="Christopher Nolan birthplace"),
            LLMResponse(content="London"),
        ]
    )

    baseline = TwoHopHybridRAGBaseline(
        retriever=InMemoryRetriever(),
        corpus_store=InMemoryCorpusStore(),
        llm_client=llm,
        config=TwoHopBaselineConfig(hop1_top_k=2, hop2_top_k=2),
    )

    result = baseline.run(
        question_id="q1",
        question="Where was the director of Inception born?",
        gold_answer="London",
        gold_supporting_titles=["Inception", "Christopher Nolan"],
    )

    result_dict = result.to_dict()

    assert "supporting_docs_found_after_hops" not in result_dict
    assert "supporting_doc_recall_after_hops" not in result_dict
    assert "both_supporting_docs_found_after_hops" not in result_dict


def test_merge_unique_documents_preserves_first_seen_order() -> None:
    first = [RAGDocument(chunk_id="a", title="A", text="A text")]

    second = [
        RAGDocument(chunk_id="b", title="B", text="B text"),
        RAGDocument(chunk_id="a", title="A", text="A text duplicate"),
    ]

    merged = merge_unique_documents(first, second)

    assert [doc.chunk_id for doc in merged] == ["a", "b"]
    assert merged[0].text == "A text"


def test_sanitize_generated_query_removes_wrappers() -> None:
    assert (
        sanitize_generated_query('```text\n"Christopher Nolan birthplace"\n```')
        == "Christopher Nolan birthplace"
    )


def test_format_documents_for_prompt_contains_provenance() -> None:
    prompt_context = format_documents_for_prompt(
        [
            RAGDocument(
                chunk_id="c1",
                title="Inception",
                text="Title: Inception\nParagraph: Example.",
            )
        ]
    )

    assert "[Document 1]" in prompt_context
    assert "Chunk ID: c1" in prompt_context
    assert "Title: Inception" in prompt_context