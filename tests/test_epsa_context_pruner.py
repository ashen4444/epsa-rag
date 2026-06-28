from __future__ import annotations

from epsa_rag.epsa.context_pruner import ContextPruner
from epsa_rag.epsa.schemas import EvidencePath, EvidenceUnit, ScoredEvidenceUnit, SufficiencyDecision


def _unit(
    *,
    evidence_unit_id,
    chunk_id,
    doc_title,
    sentence_id,
    sentence_text,
    rank,
):
    return EvidenceUnit(
        evidence_unit_id=evidence_unit_id,
        chunk_id=chunk_id,
        doc_title=doc_title,
        paragraph_index=0,
        sentence_id=sentence_id,
        sentence_text=sentence_text,
        resolved_text=sentence_text,
        entities=[],
        relation_hints=[],
        answer_type_candidates=[],
        question_entity_overlap=[],
        question_token_overlap=0.0,
        is_supporting_sentence=True,
        retrieval_rank=rank,
        retrieval_score=1.0 / rank,
    )


def _scored(unit, score=0.8):
    return ScoredEvidenceUnit(evidence_unit=unit, final_score=score, score_breakdown={})


def _scored_units():
    return [
        _scored(
            _unit(
                evidence_unit_id="chunk_noise::s0",
                chunk_id="chunk_noise",
                doc_title="Noise",
                sentence_id=0,
                sentence_text="A distracting sentence.",
                rank=3,
            ),
            0.1,
        ),
        _scored(
            _unit(
                evidence_unit_id="chunk_inception::s0",
                chunk_id="chunk_inception",
                doc_title="Inception",
                sentence_id=0,
                sentence_text="Inception was directed by Christopher Nolan.",
                rank=1,
            ),
            0.94,
        ),
        _scored(
            _unit(
                evidence_unit_id="chunk_nolan::s0",
                chunk_id="chunk_nolan",
                doc_title="Christopher Nolan",
                sentence_id=0,
                sentence_text="Christopher Nolan was born in London.",
                rank=2,
            ),
            0.91,
        ),
    ]


def _decision(**overrides):
    data = {
        "sufficient": True,
        "confidence": 0.9,
        "question_type": "bridge",
        "best_path": EvidencePath(
            path_id="bridge::1",
            question_type="bridge",
            node_ids=[],
            edge_ids=[],
            evidence_unit_ids=["chunk_inception::s0", "chunk_nolan::s0"],
            entity_chain=["Inception", "Christopher Nolan", "London"],
            relation_chain=["directed", "born"],
            answer_candidate="London",
            answer_type="LOCATION",
            score=0.95,
        ),
        "selected_evidence_unit_ids": ["chunk_nolan::s0", "chunk_inception::s0", "chunk_nolan::s0"],
        "selected_chunk_ids": ["chunk_inception", "chunk_nolan"],
        "answer_candidate": "London",
        "answer_type": "LOCATION",
        "missing_evidence": None,
        "decision_reason": "Complete bridge evidence path found.",
        "rule_trace": [],
        "metadata": {},
    }
    data.update(overrides)
    return SufficiencyDecision(**data)


def test_keeps_only_evidence_units_selected_by_decision():
    pruned = ContextPruner().prune(_decision(), _scored_units())

    assert pruned.selected_evidence_unit_ids == ["chunk_inception::s0", "chunk_nolan::s0"]
    assert pruned.selected_chunk_ids == ["chunk_inception", "chunk_nolan"]
    assert "distracting" not in pruned.selected_context_text.lower()
    assert "Inception was directed by Christopher Nolan." in pruned.selected_context_text
    assert "Christopher Nolan was born in London." in pruned.selected_context_text


def test_deduplicates_evidence_unit_ids_and_preserves_deterministic_ordering():
    pruned = ContextPruner().prune(_decision(), _scored_units())

    assert pruned.selected_evidence_unit_ids == ["chunk_inception::s0", "chunk_nolan::s0"]
    assert pruned.selected_context_text.index("Inception was directed") < pruned.selected_context_text.index("born in London")


def test_formats_selected_sentence_context_with_provenance_metadata():
    pruned = ContextPruner().prune(_decision(), _scored_units())

    assert "[Title: Inception | Chunk: chunk_inception | Sentence: 0]" in pruned.selected_context_text
    assert "[Title: Christopher Nolan | Chunk: chunk_nolan | Sentence: 0]" in pruned.selected_context_text
    assert pruned.selected_sentences == [
        "Inception was directed by Christopher Nolan.",
        "Christopher Nolan was born in London.",
    ]


def test_estimates_context_tokens_deterministically():
    pruned_one = ContextPruner().prune(_decision(), _scored_units())
    pruned_two = ContextPruner().prune(_decision(), _scored_units())

    assert pruned_one.estimated_context_tokens > 0
    assert pruned_one.estimated_context_tokens == pruned_two.estimated_context_tokens


def test_lists_removed_evidence_units():
    pruned = ContextPruner().prune(_decision(), _scored_units())

    assert pruned.removed_evidence_unit_ids == ["chunk_noise::s0"]


def test_handles_insufficient_decision_with_partial_selected_evidence():
    decision = _decision(
        sufficient=False,
        selected_evidence_unit_ids=["chunk_inception::s0"],
        missing_evidence="Bridge path is incomplete after bridge entity Christopher Nolan.",
        decision_reason="No candidate bridge path satisfied all deterministic completeness rules.",
    )

    pruned = ContextPruner().prune(decision, _scored_units())

    assert pruned.pruning_strategy == "partial_evidence_sentence_pruning"
    assert pruned.selected_evidence_unit_ids == ["chunk_inception::s0"]
    assert pruned.metadata["sufficient"] is False
    assert pruned.metadata["missing_evidence"] == "Bridge path is incomplete after bridge entity Christopher Nolan."


def test_handles_empty_selected_evidence_safely():
    decision = _decision(
        sufficient=False,
        selected_evidence_unit_ids=[],
        selected_chunk_ids=[],
        answer_candidate=None,
        missing_evidence="No candidate evidence path found.",
    )

    pruned = ContextPruner().prune(decision, _scored_units())

    assert pruned.pruning_strategy == "empty_evidence_pruning"
    assert pruned.selected_evidence_unit_ids == []
    assert pruned.selected_chunk_ids == []
    assert pruned.selected_sentences == []
    assert pruned.selected_context_text == ""
    assert pruned.estimated_context_tokens == 0
    assert set(pruned.removed_evidence_unit_ids) == {"chunk_noise::s0", "chunk_inception::s0", "chunk_nolan::s0"}


def test_records_missing_requested_evidence_ids_without_inventing_evidence():
    decision = _decision(selected_evidence_unit_ids=["missing::s0", "chunk_inception::s0"])

    pruned = ContextPruner().prune(decision, _scored_units())

    assert pruned.selected_evidence_unit_ids == ["chunk_inception::s0"]
    assert pruned.metadata["missing_requested_evidence_unit_ids"] == ["missing::s0"]
    assert "missing::s0" not in pruned.selected_context_text


def test_pruner_does_not_make_sufficiency_decisions_or_retrieve_or_call_llm():
    pruner = ContextPruner()
    pruned = pruner.prune(_decision(), _scored_units())

    assert not hasattr(pruner, "decide")
    assert not hasattr(pruner, "retrieve")
    assert not hasattr(pruner, "llm")
    assert pruned.metadata["makes_sufficiency_decision"] is False
    assert pruned.metadata["retrieves_documents"] is False
    assert pruned.metadata["calls_llm"] is False
