from __future__ import annotations

from dataclasses import replace

from epsa_rag.epsa.evidence_graph_builder import EvidenceGraphBuilder
from epsa_rag.epsa.evidence_path_searcher import EvidencePathSearcher
from epsa_rag.epsa.schemas import EvidencePath, EvidenceUnit, QuestionAnalysis, ScoredEvidenceUnit
from epsa_rag.epsa.sufficiency_decision_engine import SufficiencyDecisionEngine


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


def _unit(
    *,
    evidence_unit_id,
    chunk_id,
    doc_title,
    sentence_id,
    sentence_text,
    entities,
    relation_hints,
    answer_type_candidates,
    question_entity_overlap=None,
    rank=1,
):
    return EvidenceUnit(
        evidence_unit_id=evidence_unit_id,
        chunk_id=chunk_id,
        doc_title=doc_title,
        paragraph_index=0,
        sentence_id=sentence_id,
        sentence_text=sentence_text,
        resolved_text=sentence_text,
        entities=entities,
        relation_hints=relation_hints,
        answer_type_candidates=answer_type_candidates,
        question_entity_overlap=question_entity_overlap or [],
        question_token_overlap=0.5,
        is_supporting_sentence=True,
        retrieval_rank=rank,
        retrieval_score=1.0 / rank,
    )


def _scored(unit, score):
    return ScoredEvidenceUnit(
        evidence_unit=unit,
        final_score=score,
        score_breakdown={"test_score": score},
    )


def _complete_bridge_units():
    return [
        _scored(
            _unit(
                evidence_unit_id="chunk_inception::s0",
                chunk_id="chunk_inception",
                doc_title="Inception",
                sentence_id=0,
                sentence_text="Inception was directed by Christopher Nolan.",
                entities=["Inception", "Christopher Nolan"],
                relation_hints=["directed"],
                answer_type_candidates=["PERSON"],
                question_entity_overlap=["Inception"],
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
                entities=["Christopher Nolan", "London"],
                relation_hints=["born"],
                answer_type_candidates=["LOCATION"],
                rank=2,
            ),
            0.91,
        ),
    ]


def _build_graph_and_paths(question=None, units=None):
    question = question or _question_analysis()
    units = units if units is not None else _complete_bridge_units()
    graph = EvidenceGraphBuilder().build(question, units)
    paths = EvidencePathSearcher().search_paths(graph, question, max_paths=10)
    return graph, paths


def test_returns_insufficient_when_no_paths_are_provided():
    question = _question_analysis()
    graph = EvidenceGraphBuilder().build(question, [])

    decision = SufficiencyDecisionEngine().decide(question, graph, [])

    assert decision.sufficient is False
    assert decision.confidence == 0.0
    assert decision.best_path is None
    assert decision.selected_evidence_unit_ids == []
    assert decision.selected_chunk_ids == []
    assert decision.missing_evidence == "No candidate evidence path found."
    assert "does_not_generate_next_query=true" in decision.rule_trace
    assert "next_query" not in decision.metadata


def test_returns_sufficient_for_complete_bridge_path():
    question = _question_analysis()
    graph, paths = _build_graph_and_paths(question)

    decision = SufficiencyDecisionEngine().decide(question, graph, paths)

    assert decision.sufficient is True
    assert decision.question_type == "bridge"
    assert decision.best_path == paths[0]
    assert decision.answer_candidate == "London"
    assert decision.answer_type == "LOCATION"
    assert decision.missing_evidence is None
    assert decision.selected_evidence_unit_ids == ["chunk_inception::s0", "chunk_nolan::s0"]
    assert decision.selected_chunk_ids == ["chunk_inception", "chunk_nolan"]
    assert decision.metadata["bridge_entity"] == "Christopher Nolan"
    assert decision.metadata["makes_next_query"] is False


def test_returns_insufficient_for_incomplete_bridge_path():
    question = _question_analysis()
    first_unit = _complete_bridge_units()[0]
    graph = EvidenceGraphBuilder().build(question, [first_unit])
    partial_path = EvidencePath(
        path_id="bridge::partial",
        question_type="bridge",
        node_ids=list(graph.nodes.keys())[:3],
        edge_ids=[],
        evidence_unit_ids=["chunk_inception::s0"],
        entity_chain=["Inception", "Christopher Nolan"],
        relation_chain=["directed"],
        answer_candidate=None,
        answer_type="PERSON",
        score=0.8,
        metadata={"bridge_entity": "Christopher Nolan"},
    )

    decision = SufficiencyDecisionEngine().decide(question, graph, [partial_path])

    assert decision.sufficient is False
    assert decision.selected_evidence_unit_ids == ["chunk_inception::s0"]
    assert decision.missing_evidence == "Bridge path is incomplete after bridge entity Christopher Nolan."
    assert decision.metadata["makes_next_query"] is False


def test_bridge_requires_answer_candidate():
    question = _question_analysis()
    graph, paths = _build_graph_and_paths(question)
    broken = replace(paths[0], answer_candidate=None)

    decision = SufficiencyDecisionEngine().decide(question, graph, [broken])

    assert decision.sufficient is False
    assert decision.missing_evidence == "Bridge path is incomplete after bridge entity Christopher Nolan."


def test_checks_expected_answer_type_compatibility():
    question = _question_analysis(expected_answer_type="LOCATION")
    units = _complete_bridge_units()
    wrong_type_second_unit = replace(
        units[1].evidence_unit,
        answer_type_candidates=["PERSON"],
    )
    wrong_units = [units[0], _scored(wrong_type_second_unit, 0.91)]
    graph, paths = _build_graph_and_paths(question, wrong_units)
    assert paths

    decision = SufficiencyDecisionEngine().decide(question, graph, paths)

    assert decision.sufficient is False
    assert decision.missing_evidence == "No answer candidate connected to expected answer type LOCATION."


def test_selects_best_path_deterministically():
    question = _question_analysis()
    graph, paths = _build_graph_and_paths(question)
    better = replace(paths[0], path_id="bridge::better", score=0.95)
    worse = replace(paths[0], path_id="bridge::worse", score=0.25, answer_candidate="Paris")

    decision = SufficiencyDecisionEngine().decide(question, graph, [worse, better])

    assert decision.sufficient is True
    assert decision.best_path.path_id == "bridge::better"
    assert decision.answer_candidate == "London"


def test_handles_factoid_sufficiency():
    question = _question_analysis(
        raw_question="What is the capital of France?",
        question_type="factoid",
        seed_entities=["France"],
        expected_answer_type="LOCATION",
        required_relation_hints=["capital"],
        normalized_question="what is the capital of france",
    )
    units = [
        _scored(
            _unit(
                evidence_unit_id="chunk_france::s0",
                chunk_id="chunk_france",
                doc_title="France",
                sentence_id=0,
                sentence_text="Paris is the capital and largest city of France.",
                entities=["Paris", "France"],
                relation_hints=["capital"],
                answer_type_candidates=["LOCATION"],
                question_entity_overlap=["France"],
            ),
            0.88,
        )
    ]
    graph, paths = _build_graph_and_paths(question, units)

    decision = SufficiencyDecisionEngine().decide(question, graph, paths)

    assert decision.sufficient is True
    assert decision.question_type == "factoid"
    assert decision.answer_candidate == "Paris"
    assert decision.selected_evidence_unit_ids == ["chunk_france::s0"]


def test_handles_comparison_conservatively_without_hallucinating_answer():
    question = _question_analysis(
        question_type="comparison",
        seed_entities=["River A", "River B"],
        expected_answer_type="NUMBER",
        required_relation_hints=["length"],
        comparison_targets=["River A", "River B"],
        normalized_question="which river is longer river a or river b",
    )
    units = [
        _scored(
            _unit(
                evidence_unit_id="river_a::s0",
                chunk_id="river_a",
                doc_title="River A",
                sentence_id=0,
                sentence_text="River A has a length of 100 km.",
                entities=["River A", "100 km"],
                relation_hints=["length"],
                answer_type_candidates=["NUMBER"],
                question_entity_overlap=["River A"],
            ),
            0.75,
        ),
        _scored(
            _unit(
                evidence_unit_id="river_b::s0",
                chunk_id="river_b",
                doc_title="River B",
                sentence_id=0,
                sentence_text="River B has a length of 200 km.",
                entities=["River B", "200 km"],
                relation_hints=["length"],
                answer_type_candidates=["NUMBER"],
                question_entity_overlap=["River B"],
            ),
            0.78,
        ),
    ]
    graph, paths = _build_graph_and_paths(question, units)

    decision = SufficiencyDecisionEngine().decide(question, graph, paths)

    assert decision.sufficient is False
    assert decision.answer_candidate is not None  # partial evidence is preserved
    assert "specialized comparison resolution" in decision.decision_reason
    assert decision.metadata["requires_later_comparison_resolution"] is True
    assert decision.metadata["makes_next_query"] is False


def test_handles_yes_no_sufficiency_conservatively():
    question = _question_analysis(
        question_type="yes_no",
        seed_entities=["Inception", "Christopher Nolan"],
        expected_answer_type="BOOLEAN",
        required_relation_hints=["directed"],
        normalized_question="was inception directed by christopher nolan",
    )
    units = [
        _scored(
            _unit(
                evidence_unit_id="inception_yesno::s0",
                chunk_id="inception_yesno",
                doc_title="Inception",
                sentence_id=0,
                sentence_text="Inception was directed by Christopher Nolan.",
                entities=["Inception", "Christopher Nolan"],
                relation_hints=["directed"],
                answer_type_candidates=["BOOLEAN"],
                question_entity_overlap=["Inception", "Christopher Nolan"],
            ),
            0.9,
        )
    ]
    graph, paths = _build_graph_and_paths(question, units)

    decision = SufficiencyDecisionEngine().decide(question, graph, paths)

    assert decision.sufficient is True
    assert decision.question_type == "yes_no"
    assert decision.answer_candidate is None
    assert decision.answer_type == "BOOLEAN"
    assert decision.metadata["does_not_generate_yes_no_answer"] is True


def test_sufficiency_engine_does_not_expose_llm_or_next_query_behavior():
    question = _question_analysis()
    graph, paths = _build_graph_and_paths(question)

    engine = SufficiencyDecisionEngine()
    decision = engine.decide(question, graph, paths)

    assert not hasattr(engine, "llm")
    assert not hasattr(engine, "model")
    assert not hasattr(engine, "generate_next_query")
    assert "next_query" not in decision.metadata
    assert decision.metadata["makes_next_query"] is False
