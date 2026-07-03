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
    best_gold_title_rank,
    bound_fallback_documents_for_context,
    build_gold_title_diagnostics,
    choose_context_for_final_answer,
    chunk_to_epsa_input,
    count_selected_context_docs,
    count_selected_context_sentences,
    documents_for_selected_chunk_ids,
    gold_title_coverage_status,
    is_potential_false_insufficient,
    is_potential_false_sufficient,
    matching_gold_titles,
    merge_retrieved_candidates,
    normalize_title_for_matching,
    object_to_dict,
    resolve_insufficient_fallback_doc_limit,
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


def test_bound_fallback_documents_limits_only_when_epsa_is_insufficient() -> None:
    docs = [
        RAGDocument(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text {i}")
        for i in range(1, 13)
    ]

    insufficient_result = SimpleNamespace(sufficient=False)
    bounded = bound_fallback_documents_for_context(
        epsa_result=insufficient_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=8,
    )

    assert [doc.chunk_id for doc in bounded] == [f"c{i}" for i in range(1, 9)]

    sufficient_result = SimpleNamespace(sufficient=True)
    unbounded = bound_fallback_documents_for_context(
        epsa_result=sufficient_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=8,
    )

    assert [doc.chunk_id for doc in unbounded] == [f"c{i}" for i in range(1, 13)]


def test_bound_fallback_documents_can_be_disabled_with_zero_limit() -> None:
    docs = [
        RAGDocument(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text {i}")
        for i in range(1, 13)
    ]

    insufficient_result = SimpleNamespace(sufficient=False)
    unbounded = bound_fallback_documents_for_context(
        epsa_result=insufficient_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=0,
    )

    assert [doc.chunk_id for doc in unbounded] == [f"c{i}" for i in range(1, 13)]


def test_choose_context_uses_prebounded_insufficient_fallback_documents() -> None:
    epsa_result = SimpleNamespace(sufficient=False)
    docs = [
        RAGDocument(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text {i}")
        for i in range(1, 6)
    ]

    bounded_docs = bound_fallback_documents_for_context(
        epsa_result=epsa_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=2,
    )
    context, source = choose_context_for_final_answer(
        epsa_result=epsa_result,
        fallback_documents=bounded_docs,
    )

    assert source == "epsa_insufficient_fallback_documents"
    assert "Chunk ID: c1" in context
    assert "Chunk ID: c2" in context
    assert "Chunk ID: c3" not in context


def test_fixed_fallback_strategy_preserves_existing_limit_behavior() -> None:
    docs = [
        RAGDocument(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text {i}")
        for i in range(1, 13)
    ]
    epsa_result = SimpleNamespace(
        sufficient=False,
        sufficiency_decision=SimpleNamespace(sufficient=False, confidence=0.49),
    )

    bounded = bound_fallback_documents_for_context(
        epsa_result=epsa_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=8,
        insufficient_fallback_strategy="fixed",
    )

    assert [doc.chunk_id for doc in bounded] == [f"c{i}" for i in range(1, 9)]
    assert (
        resolve_insufficient_fallback_doc_limit(
            epsa_result=epsa_result,
            insufficient_fallback_doc_limit=8,
            insufficient_fallback_strategy="fixed",
        )
        == 8
    )


def test_adaptive_fallback_uses_8_docs_for_near_sufficient_insufficient_case() -> None:
    docs = [
        RAGDocument(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text {i}")
        for i in range(1, 13)
    ]
    epsa_result = SimpleNamespace(
        sufficient=False,
        sufficiency_decision=SimpleNamespace(sufficient=False, confidence=0.48),
    )

    bounded = bound_fallback_documents_for_context(
        epsa_result=epsa_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=8,
        insufficient_fallback_strategy="adaptive",
    )

    assert [doc.chunk_id for doc in bounded] == [f"c{i}" for i in range(1, 9)]
    assert (
        resolve_insufficient_fallback_doc_limit(
            epsa_result=epsa_result,
            insufficient_fallback_strategy="adaptive",
        )
        == 8
    )


def test_adaptive_fallback_uses_10_docs_for_medium_confidence_insufficient_case() -> None:
    docs = [
        RAGDocument(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text {i}")
        for i in range(1, 13)
    ]
    epsa_result = SimpleNamespace(
        sufficient=False,
        sufficiency_decision=SimpleNamespace(sufficient=False, confidence=0.42),
    )

    bounded = bound_fallback_documents_for_context(
        epsa_result=epsa_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=8,
        insufficient_fallback_strategy="adaptive",
    )

    assert [doc.chunk_id for doc in bounded] == [f"c{i}" for i in range(1, 11)]
    assert (
        resolve_insufficient_fallback_doc_limit(
            epsa_result=epsa_result,
            insufficient_fallback_strategy="adaptive",
        )
        == 10
    )


def test_adaptive_fallback_uses_12_docs_for_low_confidence_insufficient_case() -> None:
    docs = [
        RAGDocument(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text {i}")
        for i in range(1, 15)
    ]
    epsa_result = SimpleNamespace(
        sufficient=False,
        sufficiency_decision=SimpleNamespace(sufficient=False, confidence=0.30),
    )

    bounded = bound_fallback_documents_for_context(
        epsa_result=epsa_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=8,
        insufficient_fallback_strategy="adaptive",
    )

    assert [doc.chunk_id for doc in bounded] == [f"c{i}" for i in range(1, 13)]
    assert (
        resolve_insufficient_fallback_doc_limit(
            epsa_result=epsa_result,
            insufficient_fallback_strategy="adaptive",
        )
        == 12
    )


def test_adaptive_fallback_does_not_bound_epsa_sufficient_cases() -> None:
    docs = [
        RAGDocument(chunk_id=f"c{i}", title=f"Title {i}", text=f"Text {i}")
        for i in range(1, 13)
    ]
    epsa_result = SimpleNamespace(
        sufficient=True,
        sufficiency_decision=SimpleNamespace(sufficient=True, confidence=0.90),
    )

    unbounded = bound_fallback_documents_for_context(
        epsa_result=epsa_result,
        fallback_documents=docs,
        insufficient_fallback_doc_limit=8,
        insufficient_fallback_strategy="adaptive",
    )

    assert [doc.chunk_id for doc in unbounded] == [f"c{i}" for i in range(1, 13)]
    assert (
        resolve_insufficient_fallback_doc_limit(
            epsa_result=epsa_result,
            insufficient_fallback_strategy="adaptive",
        )
        is None
    )



def test_normalize_title_for_matching_is_case_and_whitespace_insensitive() -> None:
    assert normalize_title_for_matching("  The   Matrix  ") == normalize_title_for_matching("the matrix")


def test_matching_gold_titles_detects_hop1_documents() -> None:
    hop1_docs = [_candidate("h1", title="  Marie   Curie ").document]

    assert matching_gold_titles(hop1_docs, ["marie curie", "Radium"]) == ["marie curie"]


def test_gold_title_diagnostics_detects_hop2_documents() -> None:
    diagnostics = build_gold_title_diagnostics(
        gold_supporting_titles=["Marie Curie", "Radium"],
        hop1_candidates=[_candidate("h1", title="Marie Curie")],
        hop2_candidates=[_candidate("h2", title="Radium")],
        merged_candidates=[_candidate("h1", title="Marie Curie"), _candidate("h2", title="Radium")],
        selected_chunk_ids=[],
        final_context_documents=[],
        context_source="epsa_pruned_context",
    )

    assert diagnostics["gold_titles_in_hop2_count"] == 1
    assert diagnostics["gold_titles_in_hop2"] == '["Radium"]'


def test_gold_title_diagnostics_detects_merged_documents() -> None:
    diagnostics = build_gold_title_diagnostics(
        gold_supporting_titles=["Marie Curie", "Radium"],
        hop1_candidates=[_candidate("h1", title="Marie Curie")],
        hop2_candidates=[_candidate("h2", title="Radium")],
        merged_candidates=[_candidate("h1", title="Marie Curie"), _candidate("h2", title="Radium")],
        selected_chunk_ids=[],
        final_context_documents=[],
        context_source="epsa_pruned_context",
    )

    assert diagnostics["gold_titles_in_merged_count"] == 2
    assert diagnostics["gold_titles_missing_from_merged_count"] == 0


def test_selected_epsa_chunk_ids_map_back_to_selected_document_titles() -> None:
    merged_candidates = [
        _candidate("c1", title="Distractor"),
        _candidate("c2", title="Christopher Nolan"),
    ]

    selected_documents = documents_for_selected_chunk_ids(merged_candidates, ["c2"])
    diagnostics = build_gold_title_diagnostics(
        gold_supporting_titles=["Christopher Nolan"],
        hop1_candidates=merged_candidates,
        hop2_candidates=[],
        merged_candidates=merged_candidates,
        selected_chunk_ids=["c2"],
        final_context_documents=selected_documents,
        context_source="epsa_pruned_context",
    )

    assert [document.title for document in selected_documents] == ["Christopher Nolan"]
    assert diagnostics["gold_titles_selected_by_epsa_count"] == 1
    assert diagnostics["gold_titles_selected_by_epsa"] == '["Christopher Nolan"]'


def test_best_gold_title_rank_is_computed_from_merged_candidate_order() -> None:
    candidates = [
        _candidate("c1", title="Distractor"),
        _candidate("c2", title="Radium"),
        _candidate("c3", title="Marie Curie"),
    ]

    assert best_gold_title_rank(candidates, ["Marie Curie", "Radium"]) == 2


def test_gold_title_coverage_status_returns_gold_not_retrieved() -> None:
    assert (
        gold_title_coverage_status(
            gold_supporting_title_count=2,
            gold_titles_in_merged_count=0,
            gold_titles_selected_by_epsa_count=0,
            gold_titles_in_final_context_count=0,
            context_source="epsa_pruned_context",
        )
        == "gold_not_retrieved"
    )


def test_gold_title_coverage_status_returns_partial_gold_retrieved() -> None:
    assert (
        gold_title_coverage_status(
            gold_supporting_title_count=2,
            gold_titles_in_merged_count=1,
            gold_titles_selected_by_epsa_count=0,
            gold_titles_in_final_context_count=0,
            context_source="epsa_pruned_context",
        )
        == "partial_gold_retrieved"
    )


def test_gold_title_coverage_status_returns_all_gold_retrieved_not_selected() -> None:
    assert (
        gold_title_coverage_status(
            gold_supporting_title_count=2,
            gold_titles_in_merged_count=2,
            gold_titles_selected_by_epsa_count=0,
            gold_titles_in_final_context_count=0,
            context_source="epsa_pruned_context",
        )
        == "all_gold_retrieved_not_selected"
    )


def test_gold_title_coverage_status_returns_partial_gold_selected() -> None:
    assert (
        gold_title_coverage_status(
            gold_supporting_title_count=2,
            gold_titles_in_merged_count=2,
            gold_titles_selected_by_epsa_count=1,
            gold_titles_in_final_context_count=1,
            context_source="epsa_pruned_context",
        )
        == "partial_gold_selected"
    )


def test_gold_title_coverage_status_returns_all_gold_selected() -> None:
    assert (
        gold_title_coverage_status(
            gold_supporting_title_count=2,
            gold_titles_in_merged_count=2,
            gold_titles_selected_by_epsa_count=2,
            gold_titles_in_final_context_count=2,
            context_source="epsa_pruned_context",
        )
        == "all_gold_selected"
    )


def test_gold_title_coverage_status_returns_fallback_context_contains_gold() -> None:
    assert (
        gold_title_coverage_status(
            gold_supporting_title_count=2,
            gold_titles_in_merged_count=2,
            gold_titles_selected_by_epsa_count=0,
            gold_titles_in_final_context_count=2,
            context_source="epsa_insufficient_fallback_documents",
        )
        == "fallback_context_contains_gold"
    )
