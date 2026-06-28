from __future__ import annotations

from epsa_rag.epsa.next_query_generator import NextHopQueryGenerator
from epsa_rag.epsa.schemas import EvidencePath, QuestionAnalysis, SufficiencyDecision


def _question(**overrides):
    data = {
        "raw_question": "Where was the director of Inception born?",
        "normalized_question": "where was the director of inception born?",
        "question_type": "bridge",
        "expected_answer_type": "LOCATION",
        "seed_entities": ["Inception"],
        "required_relation_hints": ["directed", "born"],
        "comparison_targets": [],
        "answer_type_candidates": [],
        "metadata": {},
    }
    data.update(overrides)
    return QuestionAnalysis(**data)


def _path(**overrides):
    data = {
        "path_id": "bridge::partial",
        "question_type": "bridge",
        "node_ids": [],
        "edge_ids": [],
        "evidence_unit_ids": ["chunk_inception::s0"],
        "entity_chain": ["Inception", "Christopher Nolan"],
        "relation_chain": ["directed"],
        "answer_candidate": None,
        "answer_type": "PERSON",
        "score": 0.82,
        "metadata": {"bridge_entity": "Christopher Nolan"},
    }
    data.update(overrides)
    return EvidencePath(**data)


def _decision(**overrides):
    data = {
        "sufficient": False,
        "confidence": 0.45,
        "question_type": "bridge",
        "best_path": _path(),
        "selected_evidence_unit_ids": ["chunk_inception::s0"],
        "selected_chunk_ids": ["chunk_inception"],
        "answer_candidate": None,
        "answer_type": "PERSON",
        "missing_evidence": "Bridge path is incomplete after bridge entity Christopher Nolan.",
        "decision_reason": "No candidate bridge path satisfied all deterministic completeness rules.",
        "rule_trace": [],
        "metadata": {},
    }
    data.update(overrides)
    return SufficiencyDecision(**data)


def test_returns_no_query_when_evidence_is_already_sufficient():
    decision = _decision(
        sufficient=True,
        missing_evidence=None,
        answer_candidate="London",
        answer_type="LOCATION",
    )

    result = NextHopQueryGenerator().generate(_question(), decision)

    assert result.query is None
    assert result.query_type == "no_query"
    assert result.metadata["calls_llm"] is False
    assert result.metadata["retrieves_documents"] is False
    assert result.metadata["makes_sufficiency_decision"] is False


def test_generates_bridge_completion_query_from_incomplete_bridge_path():
    result = NextHopQueryGenerator().generate(_question(), _decision())

    assert result.query == "Christopher Nolan born birthplace"
    assert result.query_type == "bridge_completion"
    assert result.target_entity == "Christopher Nolan"
    assert result.missing_relation == "born"
    assert result.expected_answer_type == "LOCATION"
    assert result.confidence > 0.0


def test_uses_missing_evidence_relation_when_available():
    decision = _decision(missing_evidence="No evidence unit supports the required relation directed.")

    result = NextHopQueryGenerator().generate(_question(), decision)

    assert result.query == "Christopher Nolan directed director location"
    assert result.missing_relation == "directed"


def test_uses_best_path_entity_chain_when_bridge_metadata_is_absent():
    path = _path(metadata={}, entity_chain=["Inception", "Christopher Nolan"])

    result = NextHopQueryGenerator().generate(_question(), _decision(best_path=path))

    assert result.target_entity == "Christopher Nolan"
    assert result.query.startswith("Christopher Nolan")


def test_falls_back_to_seed_entity_and_relation_hints_when_no_path_exists():
    decision = _decision(
        best_path=None,
        selected_evidence_unit_ids=[],
        selected_chunk_ids=[],
        missing_evidence="No candidate evidence path found.",
        answer_candidate=None,
        answer_type="LOCATION",
    )

    result = NextHopQueryGenerator().generate(_question(), decision)

    assert result.query == "Inception directed director location"
    assert result.target_entity == "Inception"
    assert result.missing_relation == "directed"


def test_includes_expected_answer_type_keyword_when_useful():
    question = _question(
        raw_question="What is the capital of France?",
        normalized_question="what is the capital of france?",
        question_type="factoid",
        expected_answer_type="LOCATION",
        seed_entities=["France"],
        required_relation_hints=["capital"],
    )
    decision = _decision(
        question_type="factoid",
        best_path=None,
        missing_evidence="No answer candidate connected to expected answer type LOCATION.",
        answer_type="LOCATION",
        selected_evidence_unit_ids=[],
        selected_chunk_ids=[],
    )

    result = NextHopQueryGenerator().generate(question, decision)

    assert result.query == "France capital location"
    assert result.query_type in {"relation_completion", "factoid_completion"}


def test_handles_factoid_insufficiency():
    question = _question(
        raw_question="Who directed Inception?",
        normalized_question="who directed inception?",
        question_type="factoid",
        expected_answer_type="PERSON",
        seed_entities=["Inception"],
        required_relation_hints=["directed"],
    )
    decision = _decision(
        question_type="factoid",
        best_path=None,
        missing_evidence="No candidate evidence path found.",
        answer_type="PERSON",
        selected_evidence_unit_ids=[],
        selected_chunk_ids=[],
    )

    result = NextHopQueryGenerator().generate(question, decision)

    assert result.query == "Inception directed director person"
    assert result.target_entity == "Inception"


def test_handles_comparison_insufficiency_conservatively():
    question = _question(
        raw_question="Which river is longer, River A or River B?",
        normalized_question="which river is longer river a or river b?",
        question_type="comparison",
        expected_answer_type="NUMBER",
        seed_entities=["River A", "River B"],
        comparison_targets=["River A", "River B"],
        required_relation_hints=["length"],
    )
    decision = _decision(
        question_type="comparison",
        best_path=None,
        missing_evidence="Comparison target evidence is incomplete.",
        answer_type="NUMBER",
        selected_evidence_unit_ids=[],
        selected_chunk_ids=[],
    )

    result = NextHopQueryGenerator().generate(question, decision)

    assert result.query == "River A River B length number"
    assert result.query_type == "comparison_target_completion"


def test_handles_yes_no_insufficiency_conservatively():
    question = _question(
        raw_question="Was Inception directed by Christopher Nolan?",
        normalized_question="was inception directed by christopher nolan?",
        question_type="yes_no",
        expected_answer_type="BOOLEAN",
        seed_entities=["Inception", "Christopher Nolan"],
        required_relation_hints=["directed"],
    )
    decision = _decision(
        question_type="yes_no",
        best_path=None,
        missing_evidence="No evidence unit supports the required relation directed.",
        answer_type="BOOLEAN",
        selected_evidence_unit_ids=[],
        selected_chunk_ids=[],
    )

    result = NextHopQueryGenerator().generate(question, decision)

    assert result.query == "Inception Christopher Nolan directed director"
    assert result.query_type == "yes_no_relation_check"


def test_returns_query_none_safely_when_no_useful_signal_exists():
    question = _question(
        raw_question="What is it?",
        normalized_question="what is it?",
        question_type="factoid",
        expected_answer_type="UNKNOWN",
        seed_entities=[],
        required_relation_hints=[],
    )
    decision = _decision(
        question_type="factoid",
        best_path=None,
        missing_evidence="No candidate evidence path found.",
        answer_type="UNKNOWN",
        selected_evidence_unit_ids=[],
        selected_chunk_ids=[],
    )

    result = NextHopQueryGenerator().generate(question, decision)

    assert result.query is None
    assert result.query_type == "no_query"


def test_generator_does_not_expose_llm_retrieval_or_sufficiency_methods():
    generator = NextHopQueryGenerator()
    result = generator.generate(_question(), _decision())

    assert not hasattr(generator, "llm")
    assert not hasattr(generator, "retrieve")
    assert not hasattr(generator, "decide")
    assert result.metadata["calls_llm"] is False
    assert result.metadata["retrieves_documents"] is False
    assert result.metadata["makes_sufficiency_decision"] is False
