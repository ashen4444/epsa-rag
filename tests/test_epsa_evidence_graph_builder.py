from epsa_rag.epsa.evidence_graph_builder import EvidenceGraphBuilder, stable_node_id, stable_sentence_node_id
from epsa_rag.epsa.schemas import EvidenceUnit, QuestionAnalysis, ScoredEvidenceUnit


def _question_analysis(**overrides):
    data = {
        "raw_question": "Where was the director of Inception born?",
        "question_type": "bridge",
        "seed_entities": ["Inception"],
        "expected_answer_type": "LOCATION",
        "required_relation_hints": ["directed", "born"],
        "comparison_targets": [],
        "normalized_question": "where was the director of inception born",
    }
    data.update(overrides)
    return QuestionAnalysis(**data)


def _evidence_unit(
    *,
    evidence_unit_id="chunk_1::s0",
    chunk_id="chunk_1",
    doc_title="Inception",
    paragraph_index=0,
    sentence_id=0,
    sentence_text="Inception was directed by Christopher Nolan.",
    entities=None,
    relation_hints=None,
    answer_type_candidates=None,
    question_entity_overlap=None,
    retrieval_rank=1,
    retrieval_score=0.95,
):
    return EvidenceUnit(
        evidence_unit_id=evidence_unit_id,
        chunk_id=chunk_id,
        doc_title=doc_title,
        paragraph_index=paragraph_index,
        sentence_id=sentence_id,
        sentence_text=sentence_text,
        resolved_text=sentence_text,
        entities=entities if entities is not None else ["Inception", "Christopher Nolan"],
        relation_hints=relation_hints if relation_hints is not None else ["directed"],
        answer_type_candidates=answer_type_candidates if answer_type_candidates is not None else ["PERSON"],
        question_entity_overlap=question_entity_overlap if question_entity_overlap is not None else ["Inception"],
        question_token_overlap=0.6,
        is_supporting_sentence=True,
        retrieval_rank=retrieval_rank,
        retrieval_score=retrieval_score,
    )


def _scored(evidence_unit, score=0.9):
    return ScoredEvidenceUnit(
        evidence_unit=evidence_unit,
        final_score=score,
        score_breakdown={"entity_match_score": score},
    )


def test_graph_builder_creates_core_nodes_and_edges():
    graph = EvidenceGraphBuilder().build(
        _question_analysis(),
        [
            _scored(_evidence_unit(), 0.92),
            _scored(
                _evidence_unit(
                    evidence_unit_id="chunk_2::s0",
                    chunk_id="chunk_2",
                    doc_title="Christopher Nolan",
                    paragraph_index=0,
                    sentence_id=0,
                    sentence_text="Christopher Nolan was born in London.",
                    entities=["Christopher Nolan", "London"],
                    relation_hints=["born"],
                    answer_type_candidates=["LOCATION"],
                    question_entity_overlap=[],
                    retrieval_rank=2,
                    retrieval_score=0.88,
                ),
                0.86,
            ),
        ],
    )

    assert stable_node_id("entity", "Inception") in graph.nodes
    assert stable_node_id("entity", "Christopher Nolan") in graph.nodes
    assert stable_node_id("entity", "London") in graph.nodes
    assert stable_node_id("chunk", "chunk_1") in graph.nodes
    assert stable_node_id("title", "Inception") in graph.nodes
    assert stable_sentence_node_id("chunk_1::s0") in graph.nodes
    assert stable_node_id("answer_type", "LOCATION") in graph.nodes
    assert stable_node_id("relation", "born") in graph.nodes

    edge_types = {edge.edge_type for edge in graph.edges}
    assert "chunk_to_sentence" in edge_types
    assert "title_to_sentence" in edge_types
    assert "sentence_mentions_entity" in edge_types
    assert "entity_cooccurs_with_entity" in edge_types
    assert "sentence_has_relation" in edge_types
    assert "sentence_has_answer_type" in edge_types
    assert "seed_entity_to_sentence" in edge_types
    assert "possible_bridge" in edge_types
    assert "possible_answer_candidate" in edge_types

    assert graph.metadata["makes_sufficiency_decision"] is False
    assert not hasattr(graph, "sufficient")


def test_graph_builder_preserves_evidence_unit_provenance_and_scores():
    graph = EvidenceGraphBuilder().build(_question_analysis(), [_scored(_evidence_unit(), 0.77)])
    sentence_node_id = stable_sentence_node_id("chunk_1::s0")

    sentence_node = graph.nodes[sentence_node_id]
    assert sentence_node.metadata["evidence_unit_id"] == "chunk_1::s0"
    assert sentence_node.metadata["chunk_id"] == "chunk_1"
    assert sentence_node.metadata["doc_title"] == "Inception"
    assert sentence_node.metadata["final_score"] == 0.77

    provenance_edges = [edge for edge in graph.edges if edge.evidence_unit_id == "chunk_1::s0"]
    assert provenance_edges
    assert all(edge.weight >= 0.0 for edge in provenance_edges)


def test_graph_builder_uses_stable_ids_and_avoids_duplicates():
    unit = _evidence_unit(entities=["Inception", "Christopher Nolan", "Christopher Nolan"])
    graph = EvidenceGraphBuilder().build(_question_analysis(), [_scored(unit, 0.8), _scored(unit, 0.8)])

    assert stable_node_id("entity", "Christopher Nolan") == "entity::christopher_nolan"
    assert stable_sentence_node_id("chunk_1::s0") == "sentence::chunk_1::s0"
    assert len(graph.nodes) == len(set(graph.nodes))

    edge_ids = [edge.edge_id for edge in graph.edges]
    assert len(edge_ids) == len(set(edge_ids))
