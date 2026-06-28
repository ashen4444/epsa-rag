from __future__ import annotations

from epsa_rag.epsa.epsa_controller import EPSAController


def _chunk(
    *,
    chunk_id,
    doc_title,
    sentence_text,
    sentence_id=0,
    rank=1,
    score=0.9,
    supporting=True,
):
    return {
        "chunk_id": chunk_id,
        "question_id": "q1",
        "doc_title": doc_title,
        "paragraph_index": 0,
        "chunk_text": f"Title: {doc_title}\nParagraph: {sentence_text}",
        "paragraph_text": sentence_text,
        "sentences": [
            {
                "sentence_id": sentence_id,
                "text": sentence_text,
                "is_supporting_sentence": supporting,
            }
        ],
        "supporting_sentence_ids": [sentence_id] if supporting else [],
        "rank": rank,
        "score": score,
    }


def _complete_bridge_chunks():
    return [
        _chunk(
            chunk_id="chunk_inception",
            doc_title="Inception",
            sentence_text="Inception was directed by Christopher Nolan.",
            rank=1,
            score=0.95,
        ),
        _chunk(
            chunk_id="chunk_nolan",
            doc_title="Christopher Nolan",
            sentence_text="Christopher Nolan was born in London.",
            rank=2,
            score=0.89,
        ),
    ]


def test_controller_runs_full_epsa_chain_and_returns_intermediate_outputs():
    result = EPSAController().run(
        "Where was the director of Inception born?",
        _complete_bridge_chunks(),
    )

    assert result.question == "Where was the director of Inception born?"
    assert result.question_analysis.raw_question == "Where was the director of Inception born?"
    assert len(result.candidate_chunk_evidence) == 2
    assert len(result.evidence_units) == 2
    assert len(result.scored_evidence_units) == 2
    assert result.evidence_graph.nodes
    assert result.evidence_paths
    assert result.sufficiency_decision is not None
    assert result.pruned_context is not None
    assert result.metadata["num_retrieved_chunks"] == 2


def test_controller_returns_sufficient_true_for_complete_bridge_evidence():
    result = EPSAController().run(
        "Where was the director of Inception born?",
        _complete_bridge_chunks(),
    )

    assert result.sufficient is True
    assert result.sufficiency_decision.sufficient is True
    assert result.sufficiency_decision.answer_candidate == "London"
    assert result.next_hop_query is None
    assert result.selected_chunk_ids == ["chunk_inception", "chunk_nolan"]
    assert result.selected_evidence_unit_ids == ["chunk_inception::s0", "chunk_nolan::s0"]
    assert "Inception was directed by Christopher Nolan." in result.pruned_context.selected_context_text
    assert "Christopher Nolan was born in London." in result.pruned_context.selected_context_text


def test_controller_returns_insufficient_false_and_next_query_for_incomplete_bridge_evidence():
    result = EPSAController().run(
        "Where was the director of Inception born?",
        [_complete_bridge_chunks()[0]],
    )

    assert result.sufficient is False
    assert result.sufficiency_decision.sufficient is False
    assert result.selected_evidence_unit_ids == ["chunk_inception::s0"]
    assert result.next_hop_query is not None
    assert result.next_hop_query.query == "Christopher Nolan born birthplace"
    assert result.next_hop_query.query_type == "bridge_completion"


def test_controller_returns_pruned_context_from_selected_evidence():
    noisy = _chunk(
        chunk_id="chunk_noise",
        doc_title="Noise",
        sentence_text="A distracting sentence about another film.",
        rank=3,
        score=0.1,
        supporting=False,
    )
    result = EPSAController().run(
        "Where was the director of Inception born?",
        [*_complete_bridge_chunks(), noisy],
    )

    assert result.sufficient is True
    assert "distracting sentence" not in result.pruned_context.selected_context_text.lower()
    assert "chunk_noise::s0" in result.pruned_context.removed_evidence_unit_ids


def test_controller_handles_empty_retrieved_chunks_safely():
    result = EPSAController().run(
        "Where was the director of Inception born?",
        [],
    )

    assert result.sufficient is False
    assert result.candidate_chunk_evidence == []
    assert result.evidence_units == []
    assert result.scored_evidence_units == []
    assert result.evidence_paths == []
    assert result.pruned_context.selected_context_text == ""
    assert result.selected_chunk_ids == []
    assert result.selected_evidence_unit_ids == []
    assert result.next_hop_query is not None
    assert result.next_hop_query.query == "Inception directed director location"


def test_controller_preserves_selected_ids_from_pruned_context():
    result = EPSAController().run(
        "Where was the director of Inception born?",
        _complete_bridge_chunks(),
    )

    assert result.selected_chunk_ids == result.pruned_context.selected_chunk_ids
    assert result.selected_evidence_unit_ids == result.pruned_context.selected_evidence_unit_ids


def test_controller_handles_factoid_evidence():
    result = EPSAController().run(
        "Who directed Inception?",
        [
            _chunk(
                chunk_id="chunk_inception",
                doc_title="Inception",
                sentence_text="Inception was directed by Christopher Nolan.",
                rank=1,
                score=0.9,
            )
        ],
    )

    assert result.sufficient is True
    assert result.sufficiency_decision.question_type == "factoid"
    assert result.sufficiency_decision.answer_candidate == "Christopher Nolan"
    assert result.next_hop_query is None


def test_controller_handles_comparison_conservatively():
    result = EPSAController().run(
        "Which of River A and River B is longer?",
        [
            _chunk(
                chunk_id="river_a",
                doc_title="River A",
                sentence_text="River A has a length of 100 km.",
                rank=1,
                score=0.8,
            ),
            _chunk(
                chunk_id="river_b",
                doc_title="River B",
                sentence_text="River B has a length of 200 km.",
                rank=2,
                score=0.75,
            ),
        ],
    )

    assert result.sufficient is False
    assert result.sufficiency_decision.question_type == "comparison"
    assert result.next_hop_query is not None
    assert result.next_hop_query.query_type == "comparison_target_completion"


def test_controller_handles_yes_no_evidence_conservatively():
    result = EPSAController().run(
        "Was Inception directed by Christopher Nolan?",
        [
            _chunk(
                chunk_id="inception_yesno",
                doc_title="Inception",
                sentence_text="Inception was directed by Christopher Nolan.",
                rank=1,
                score=0.9,
            )
        ],
    )

    assert result.sufficient is True
    assert result.sufficiency_decision.question_type == "yes_no"
    assert result.sufficiency_decision.answer_type == "BOOLEAN"
    assert result.next_hop_query is None


def test_controller_does_not_own_retriever_or_llm_or_final_generation():
    controller = EPSAController()
    result = controller.run(
        "Where was the director of Inception born?",
        _complete_bridge_chunks(),
    )

    assert not hasattr(controller, "retriever")
    assert not hasattr(controller, "llm")
    assert not hasattr(controller, "model")
    assert not hasattr(controller, "answer_generator")
    assert result.metadata["calls_llm"] is False
    assert result.metadata["retrieves_documents"] is False
    assert result.metadata["modifies_retriever"] is False
    assert result.metadata["generates_final_answer"] is False
    assert result.metadata["runs_evaluation_pipeline"] is False
