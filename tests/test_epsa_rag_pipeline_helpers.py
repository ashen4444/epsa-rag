from __future__ import annotations

from types import SimpleNamespace

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_epsa_rag import (
    RetrievedCandidate,
    RAGDocument,
    choose_context_for_final_answer,
    chunk_to_epsa_input,
    count_selected_context_docs,
    count_selected_context_sentences,
    is_potential_false_insufficient,
    is_potential_false_sufficient,
    merge_retrieved_candidates,
    object_to_dict,
    selected_chunk_ids_from_epsa,
    serialize_list_for_csv,
)


class ChunkObject:
    def __init__(self) -> None:
        self.chunk_id = "c1"
        self.source_question_id = "q1"
        self.doc_title = "Inception"
        self.chunk_text = "Title: Inception\nParagraph: Inception was directed by Christopher Nolan."
        self.paragraph_index = 0
        self.sentences = [{"sentence_id": 0, "text": "Inception was directed by Christopher Nolan."}]


class RetrievalItem:
    chunk_id = "c1"
    rank = 3
    fusion_score = 0.75


def _candidate(chunk_id: str, title: str = "Title") -> RetrievedCandidate:
    return RetrievedCandidate(
        document=RAGDocument(chunk_id=chunk_id, title=title, text=f"Text for {chunk_id}"),
        epsa_chunk={"chunk_id": chunk_id, "doc_title": title, "chunk_text": f"Text for {chunk_id}"},
    )


def test_object_to_dict_supports_plain_objects() -> None:
    payload = object_to_dict(ChunkObject())

    assert payload["chunk_id"] == "c1"
    assert payload["doc_title"] == "Inception"


def test_chunk_to_epsa_input_adds_retrieval_rank_and_score() -> None:
    payload = chunk_to_epsa_input(ChunkObject(), RetrievalItem(), fallback_rank=1)

    assert payload["chunk_id"] == "c1"
    assert payload["question_id"] == "q1"
    assert payload["rank"] == 3
    assert payload["retrieval_rank"] == 3
    assert payload["score"] == 0.75
    assert payload["retrieval_score"] == 0.75
    assert payload["sentences"] == [{"sentence_id": 0, "text": "Inception was directed by Christopher Nolan."}]


def test_merge_retrieved_candidates_deduplicates_by_chunk_id_preserving_order() -> None:
    merged = merge_retrieved_candidates(
        [_candidate("c1"), _candidate("c2")],
        [_candidate("c2", title="Duplicate"), _candidate("c3")],
    )

    assert [candidate.document.chunk_id for candidate in merged] == ["c1", "c2", "c3"]
    assert merged[1].document.title == "Title"


def test_choose_context_for_final_answer_prefers_epsa_pruned_context() -> None:
    epsa_result = SimpleNamespace(
        pruned_context=SimpleNamespace(selected_context_text="[Title: Inception]\nEvidence sentence.")
    )

    context, source = choose_context_for_final_answer(
        epsa_result=epsa_result,
        fallback_documents=[RAGDocument(chunk_id="fallback", title="Fallback", text="Fallback text")],
    )

    assert source == "epsa_pruned_context"
    assert context == "[Title: Inception]\nEvidence sentence."
    assert "Fallback text" not in context


def test_choose_context_for_final_answer_falls_back_when_pruned_context_is_empty() -> None:
    epsa_result = SimpleNamespace(pruned_context=SimpleNamespace(selected_context_text=""))

    context, source = choose_context_for_final_answer(
        epsa_result=epsa_result,
        fallback_documents=[RAGDocument(chunk_id="c1", title="Fallback", text="Fallback text")],
    )

    assert source == "fallback_documents"
    assert "Chunk ID: c1" in context
    assert "Fallback text" in context


def test_context_counts_match_context_source() -> None:
    epsa_result = SimpleNamespace(
        selected_chunk_ids=["c1", "c2"],
        pruned_context=SimpleNamespace(selected_sentences=["s1", "s2"]),
    )

    assert count_selected_context_docs(epsa_result, [], "epsa_pruned_context") == 2
    assert count_selected_context_sentences(epsa_result, "epsa_pruned_context") == 2
    assert count_selected_context_docs(
        epsa_result,
        [RAGDocument(chunk_id="f1", title="F", text="F"), RAGDocument(chunk_id="f2", title="F", text="F")],
        "fallback_documents",
    ) == 2
    assert count_selected_context_sentences(epsa_result, "fallback_documents") == 0


def test_selected_chunk_ids_from_epsa_handles_missing_result() -> None:
    assert selected_chunk_ids_from_epsa(None) == []


def test_serialize_list_for_csv_is_stable_json() -> None:
    assert serialize_list_for_csv(["c1", "c2"]) == '["c1", "c2"]'


def test_potential_false_sufficient_flag_requires_sufficient_and_wrong_answer() -> None:
    assert is_potential_false_sufficient(
        epsa_sufficient=True,
        exact_match=0.0,
        partial_match=0.0,
    ) is True
    assert is_potential_false_sufficient(
        epsa_sufficient=True,
        exact_match=1.0,
        partial_match=1.0,
    ) is False
    assert is_potential_false_sufficient(
        epsa_sufficient=False,
        exact_match=0.0,
        partial_match=0.0,
    ) is False


def test_potential_false_insufficient_flag_when_hop2_used_but_final_selection_only_uses_hop1() -> None:
    assert is_potential_false_insufficient(
        hop1_sufficient=False,
        adaptive_stop_after_hop="2",
        final_sufficient=True,
        selected_chunk_ids=["h1_a", "h1_b"],
        hop1_chunk_ids=["h1_a", "h1_b", "h1_c"],
    ) is True

    assert is_potential_false_insufficient(
        hop1_sufficient=False,
        adaptive_stop_after_hop="2",
        final_sufficient=True,
        selected_chunk_ids=["h2_a"],
        hop1_chunk_ids=["h1_a"],
    ) is False
