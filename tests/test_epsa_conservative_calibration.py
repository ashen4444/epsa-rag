from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epsa_rag.epsa.question_analyzer import QuestionAnalyzer, QuestionType
from epsa_rag.rag.two_hop_baseline import RAGDocument
from scripts.run_epsa_rag import choose_context_for_final_answer


def test_final_answer_context_falls_back_when_epsa_is_explicitly_insufficient() -> None:
    epsa_result = SimpleNamespace(
        sufficient=False,
        pruned_context=SimpleNamespace(
            selected_context_text="[Title: Partial]\nPartial evidence only."
        ),
    )

    context, source = choose_context_for_final_answer(
        epsa_result=epsa_result,
        fallback_documents=[
            RAGDocument(
                chunk_id="c1",
                title="Fallback",
                text="Full fallback evidence.",
            )
        ],
    )

    assert source == "epsa_insufficient_fallback_documents"
    assert "Full fallback evidence." in context
    assert "Partial evidence only." not in context


def test_question_analyzer_treats_or_first_questions_as_comparison() -> None:
    analysis = QuestionAnalyzer().analyze(
        "Which magazine was started first Arthur's Magazine or First for Women?"
    )

    assert analysis.question_type == QuestionType.COMPARISON


def test_question_analyzer_treats_or_choice_person_question_as_comparison() -> None:
    analysis = QuestionAnalyzer().analyze(
        "Who was inducted into the Rock and Roll Hall of Fame, David Lee Roth or Cia Berg?"
    )

    assert analysis.question_type == QuestionType.COMPARISON



from epsa_rag.epsa.schemas import (
    EntityMention,
    EvidenceGraph,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    EvidencePath,
    QuestionAnalysis,
    RelationHint,
)
from epsa_rag.epsa.sufficiency_decision_engine import (
    SufficiencyDecisionEngine,
    compute_selected_evidence_coverage,
    has_minimum_evidence_coverage,
)
from epsa_rag.epsa.question_analyzer import AnswerType


def _chat17_question_analysis(
    raw_question: str,
    *,
    question_type: QuestionType = QuestionType.BRIDGE,
    answer_type: AnswerType = AnswerType.PERSON,
) -> QuestionAnalysis:
    return QuestionAnalysis(
        raw_question=raw_question,
        normalized_question=raw_question.lower(),
        question_type=question_type,
        expected_answer_type=answer_type,
        seed_entities=[
            EntityMention(text="Seed", normalized="seed", source="test"),
        ],
        required_relation_hints=[
            RelationHint(
                relation="written",
                matched_text="written",
                source="test",
            ),
        ],
    )


def _chat17_graph(answer_type: str = "PERSON") -> EvidenceGraph:
    return EvidenceGraph(
        nodes={
            "entity::seed": EvidenceGraphNode(
                node_id="entity::seed",
                node_type="entity",
                label="Seed",
            ),
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="Seed to bridge evidence.",
                metadata={"evidence_unit_id": "u1", "chunk_id": "c1"},
            ),
            "sentence::u2": EvidenceGraphNode(
                node_id="sentence::u2",
                node_type="sentence",
                label="Bridge to answer evidence.",
                metadata={"evidence_unit_id": "u2", "chunk_id": "c2"},
            ),
        },
        edges=[
            EvidenceGraphEdge(
                edge_id="edge::u2_answer_type",
                source_id="sentence::u2",
                target_id="answer_type::person",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": answer_type},
            ),
            EvidenceGraphEdge(
                edge_id="edge::u1_answer_type",
                source_id="sentence::u1",
                target_id="answer_type::person",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": answer_type},
            ),
        ],
        question_type="bridge",
        seed_entity_node_ids=["entity::seed"],
        metadata={
            "expected_answer_type": answer_type,
            "required_relation_hints": ["written"],
        },
    )


def _chat17_bridge_path(answer_candidate: str) -> EvidencePath:
    return EvidencePath(
        path_id="path::bridge",
        question_type="bridge",
        node_ids=[
            "entity::seed",
            "sentence::u1",
            "entity::bridge",
            "sentence::u2",
            "entity::answer",
        ],
        edge_ids=[],
        evidence_unit_ids=["u1", "u2"],
        entity_chain=["Seed", "Bridge", answer_candidate],
        relation_chain=["written"],
        answer_candidate=answer_candidate,
        answer_type="PERSON",
        score=0.9,
        metadata={"bridge_entity": "Bridge"},
    )


def test_question_analyzer_treats_yes_no_or_choice_as_comparison() -> None:
    analysis = QuestionAnalyzer().analyze(
        "Is Children's National Medical Center or MedStar Washington Hospital Center "
        "the largest private hospital in Washington, D.C.?"
    )

    assert analysis.question_type == QuestionType.COMPARISON


def test_question_analyzer_prioritizes_length_over_nested_where() -> None:
    analysis = QuestionAnalyzer().analyze(
        "What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?"
    )

    assert analysis.expected_answer_type == AnswerType.NUMBER
    assert analysis.question_type == QuestionType.BRIDGE


def test_question_analyzer_detects_nested_film_question_as_bridge_title_answer() -> None:
    analysis = QuestionAnalyzer().analyze(
        "Which Oscar-nominated film was written by the screenwriter who wrote a "
        "1991 romantic drama based upon a screenplay by Sooni Taraporevala?"
    )

    assert analysis.expected_answer_type == AnswerType.TITLE_OR_WORK
    assert analysis.question_type == QuestionType.BRIDGE


def test_bridge_person_candidate_must_look_like_person_name() -> None:
    decision = SufficiencyDecisionEngine().decide(
        _chat17_question_analysis("Who wrote the film?"),
        _chat17_graph("PERSON"),
        [_chat17_bridge_path("Broadway")],
    )

    assert decision.sufficient is False
    assert "does not look compatible" in str(decision.missing_evidence)


def test_complete_bridge_with_specific_person_remains_sufficient() -> None:
    decision = SufficiencyDecisionEngine().decide(
        _chat17_question_analysis("Who developed the prototype pacemaker?"),
        _chat17_graph("PERSON"),
        [_chat17_bridge_path("R Adams Cowley")],
    )

    assert decision.sufficient is True


def test_complex_factoid_single_evidence_unit_is_insufficient() -> None:
    analysis = QuestionAnalysis(
        raw_question=(
            "Which Oscar-nominated film was written by the screenwriter who wrote "
            "a 1991 romantic drama?"
        ),
        normalized_question=(
            "which oscar-nominated film was written by the screenwriter who wrote "
            "a 1991 romantic drama?"
        ),
        question_type=QuestionType.FACTOID,
        expected_answer_type=AnswerType.TITLE_OR_WORK,
        seed_entities=[
            EntityMention(text="Seed", normalized="seed", source="test"),
        ],
        required_relation_hints=[],
    )
    path = EvidencePath(
        path_id="path::factoid",
        question_type="factoid",
        node_ids=["entity::seed", "sentence::u1", "entity::answer"],
        edge_ids=[],
        evidence_unit_ids=["u1"],
        entity_chain=["Seed", "Film Title"],
        relation_chain=[],
        answer_candidate="Film Title",
        answer_type="TITLE_OR_WORK",
        score=0.9,
        metadata={},
    )

    decision = SufficiencyDecisionEngine().decide(
        analysis,
        _chat17_graph("TITLE_OR_WORK"),
        [path],
    )

    assert decision.sufficient is False


def test_question_analyzer_extracts_head_office_relation() -> None:
    analysis = QuestionAnalyzer().analyze(
        "The Oberoi family is part of a hotel company that has a head office in what city?"
    )

    relations = {hint.relation for hint in analysis.required_relation_hints}

    assert "headquarters" in relations


def test_question_analyzer_extracts_named_after_relation() -> None:
    analysis = QuestionAnalyzer().analyze(
        'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" '
        "character Milhouse, who Matt Groening named after who?"
    )

    relations = {hint.relation for hint in analysis.required_relation_hints}

    assert "named" in relations


def test_question_analyzer_extracts_nationality_relation() -> None:
    analysis = QuestionAnalyzer().analyze(
        "What nationality was James Henry Miller's wife?"
    )

    relations = {hint.relation for hint in analysis.required_relation_hints}

    assert "nationality" in relations
    assert "spouse" in relations

def test_bridge_path_with_generic_company_bridge_is_insufficient() -> None:
    analysis = QuestionAnalysis(
        raw_question="The Oberoi family is part of a hotel company that has a head office in what city?",
        normalized_question="the oberoi family is part of a hotel company that has a head office in what city?",
        question_type=QuestionType.BRIDGE,
        expected_answer_type=AnswerType.LOCATION,
        seed_entities=[
            EntityMention(text="Oberoi family", normalized="oberoi family", source="test"),
        ],
        required_relation_hints=[
            RelationHint(relation="headquarters", matched_text="head office", source="test"),
        ],
    )
    graph = EvidenceGraph(
        nodes={
            "entity::oberoi_family": EvidenceGraphNode(
                node_id="entity::oberoi_family",
                node_type="entity",
                label="Oberoi family",
            ),
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="The Oberoi family is associated with a hotel company.",
                metadata={"evidence_unit_id": "u1", "chunk_id": "c1"},
            ),
            "sentence::u2": EvidenceGraphNode(
                node_id="sentence::u2",
                node_type="sentence",
                label="An unrelated company has its head office in Australia.",
                metadata={"evidence_unit_id": "u2", "chunk_id": "c2"},
            ),
        },
        edges=[
            EvidenceGraphEdge(
                edge_id="edge::u2_answer_type",
                source_id="sentence::u2",
                target_id="answer_type::location",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": "LOCATION"},
            ),
        ],
        question_type="bridge",
        seed_entity_node_ids=["entity::oberoi_family"],
        metadata={
            "expected_answer_type": "LOCATION",
            "required_relation_hints": ["headquarters"],
        },
    )
    path = EvidencePath(
        path_id="path::generic_bridge",
        question_type="bridge",
        node_ids=[
            "entity::oberoi_family",
            "sentence::u1",
            "entity::company",
            "sentence::u2",
            "entity::australia",
        ],
        edge_ids=[],
        evidence_unit_ids=["u1", "u2"],
        entity_chain=["Oberoi family", "company", "Australia"],
        relation_chain=["headquarters"],
        answer_candidate="Australia",
        answer_type="LOCATION",
        score=0.95,
        metadata={"bridge_entity": "company"},
    )

    decision = SufficiencyDecisionEngine().decide(analysis, graph, [path])

    assert decision.sufficient is False
    assert "too generic" in str(decision.missing_evidence)


def test_bridge_path_requires_bridge_entity_grounded_in_answer_side_chunk() -> None:
    analysis = QuestionAnalysis(
        raw_question="The Oberoi family is part of a hotel company that has a head office in what city?",
        normalized_question="the oberoi family is part of a hotel company that has a head office in what city?",
        question_type=QuestionType.BRIDGE,
        expected_answer_type=AnswerType.LOCATION,
        seed_entities=[
            EntityMention(text="Oberoi family", normalized="oberoi family", source="test"),
        ],
        required_relation_hints=[
            RelationHint(relation="headquarters", matched_text="head office", source="test"),
        ],
    )
    graph = EvidenceGraph(
        nodes={
            "entity::oberoi_family": EvidenceGraphNode(
                node_id="entity::oberoi_family",
                node_type="entity",
                label="Oberoi family",
            ),
            "entity::oberoi_group": EvidenceGraphNode(
                node_id="entity::oberoi_group",
                node_type="entity",
                label="Oberoi Group",
            ),
            "entity::australia": EvidenceGraphNode(
                node_id="entity::australia",
                node_type="entity",
                label="Australia",
            ),
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="The Oberoi family is associated with the Oberoi Group.",
                metadata={
                    "evidence_unit_id": "u1",
                    "chunk_id": "c1",
                    "doc_title": "Oberoi family",
                    "sentence_text": "The Oberoi family is associated with the Oberoi Group.",
                    "resolved_text": "The Oberoi family is associated with the Oberoi Group.",
                },
            ),
            "sentence::u2": EvidenceGraphNode(
                node_id="sentence::u2",
                node_type="sentence",
                label="Future Fibre Technologies has its head office in Australia.",
                metadata={
                    "evidence_unit_id": "u2",
                    "chunk_id": "c2",
                    "doc_title": "Future Fibre Technologies",
                    "sentence_text": "Future Fibre Technologies has its head office in Australia.",
                    "resolved_text": "Future Fibre Technologies has its head office in Australia.",
                },
            ),
        },
        edges=[
            EvidenceGraphEdge(
                edge_id="edge::u2_answer_type",
                source_id="sentence::u2",
                target_id="answer_type::location",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": "LOCATION"},
            ),
            EvidenceGraphEdge(
                edge_id="edge::u2_australia",
                source_id="sentence::u2",
                target_id="entity::australia",
                edge_type="sentence_mentions_entity",
                metadata={"entity": "Australia"},
            ),
        ],
        question_type="bridge",
        seed_entity_node_ids=["entity::oberoi_family"],
        metadata={
            "expected_answer_type": "LOCATION",
            "required_relation_hints": ["headquarters"],
        },
    )
    path = EvidencePath(
        path_id="path::ungrounded_bridge",
        question_type="bridge",
        node_ids=[
            "entity::oberoi_family",
            "sentence::u1",
            "entity::oberoi_group",
            "sentence::u2",
            "entity::australia",
        ],
        edge_ids=[],
        evidence_unit_ids=["u1", "u2"],
        entity_chain=["Oberoi family", "Oberoi Group", "Australia"],
        relation_chain=["headquarters"],
        answer_candidate="Australia",
        answer_type="LOCATION",
        score=0.95,
        metadata={"bridge_entity": "Oberoi Group"},
    )

    decision = SufficiencyDecisionEngine().decide(analysis, graph, [path])

    assert decision.sufficient is False
    assert "not strongly grounded" in str(decision.missing_evidence)


def test_bridge_path_rejects_adjectival_nationality_bridge_entity() -> None:
    analysis = QuestionAnalysis(
        raw_question="The Oberoi family is part of a hotel company that has a head office in what city?",
        normalized_question="the oberoi family is part of a hotel company that has a head office in what city?",
        question_type=QuestionType.BRIDGE,
        expected_answer_type=AnswerType.LOCATION,
        seed_entities=[
            EntityMention(text="Oberoi family", normalized="oberoi family", source="test"),
        ],
        required_relation_hints=[
            RelationHint(relation="headquarters", matched_text="head office", source="test"),
        ],
    )
    graph = EvidenceGraph(
        nodes={
            "entity::oberoi_family": EvidenceGraphNode(
                node_id="entity::oberoi_family",
                node_type="entity",
                label="Oberoi family",
            ),
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="The Oberoi family is an Indian family involved in hotels.",
                metadata={"evidence_unit_id": "u1", "chunk_id": "c1"},
            ),
            "sentence::u2": EvidenceGraphNode(
                node_id="sentence::u2",
                node_type="sentence",
                label="Future Fibre Technologies has an Indian head office in New Delhi.",
                metadata={"evidence_unit_id": "u2", "chunk_id": "c2"},
            ),
        },
        edges=[
            EvidenceGraphEdge(
                edge_id="edge::u2_answer_type",
                source_id="sentence::u2",
                target_id="answer_type::location",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": "LOCATION"},
            ),
        ],
        question_type="bridge",
        seed_entity_node_ids=["entity::oberoi_family"],
        metadata={
            "expected_answer_type": "LOCATION",
            "required_relation_hints": ["headquarters"],
        },
    )
    path = EvidencePath(
        path_id="path::adjectival_bridge",
        question_type="bridge",
        node_ids=[
            "entity::oberoi_family",
            "sentence::u1",
            "entity::indian",
            "sentence::u2",
            "entity::new_delhi",
        ],
        edge_ids=[],
        evidence_unit_ids=["u1", "u2"],
        entity_chain=["Oberoi family", "Indian", "New Delhi"],
        relation_chain=["headquarters"],
        answer_candidate="New Delhi",
        answer_type="LOCATION",
        score=0.95,
        metadata={"bridge_entity": "Indian"},
    )

    decision = SufficiencyDecisionEngine().decide(analysis, graph, [path])

    assert decision.sufficient is False
    assert "too generic" in str(decision.missing_evidence)

def test_number_answer_candidate_must_look_numeric() -> None:
    analysis = QuestionAnalysis(
        raw_question="At what age did the expert mentor win the championship?",
        normalized_question="at what age did the expert mentor win the championship?",
        question_type=QuestionType.BRIDGE,
        expected_answer_type=AnswerType.NUMBER,
        seed_entities=[
            EntityMention(text="Splash", normalized="splash", source="test"),
        ],
        required_relation_hints=[
            RelationHint(relation="won", matched_text="won", source="test"),
        ],
    )
    graph = _chat17_graph("NUMBER")
    path = EvidencePath(
        path_id="path::bad_number",
        question_type="bridge",
        node_ids=[
            "entity::seed",
            "sentence::u1",
            "entity::bridge",
            "sentence::u2",
            "entity::itv",
        ],
        edge_ids=[],
        evidence_unit_ids=["u1", "u2"],
        entity_chain=["Splash", "Bridge", "ITV"],
        relation_chain=["won"],
        answer_candidate="ITV",
        answer_type="NUMBER",
        score=0.9,
        metadata={"bridge_entity": "Bridge"},
    )

    decision = SufficiencyDecisionEngine().decide(analysis, graph, [path])

    assert decision.sufficient is False
    assert "does not look compatible" in str(decision.missing_evidence)


def test_location_answer_candidate_rejects_nationality_adjective() -> None:
    analysis = QuestionAnalysis(
        raw_question="The manufacturer was based in which city?",
        normalized_question="the manufacturer was based in which city?",
        question_type=QuestionType.FACTOID,
        expected_answer_type=AnswerType.LOCATION,
        seed_entities=[
            EntityMention(text="Pirna 014", normalized="pirna 014", source="test"),
        ],
        required_relation_hints=[
            RelationHint(relation="located", matched_text="based", source="test"),
        ],
    )
    graph = _chat17_graph("LOCATION")
    path = EvidencePath(
        path_id="path::bad_location",
        question_type="factoid",
        node_ids=["entity::seed", "sentence::u1", "entity::american"],
        edge_ids=[],
        evidence_unit_ids=["u1"],
        entity_chain=["Pirna 014", "American"],
        relation_chain=["located"],
        answer_candidate="American",
        answer_type="LOCATION",
        score=0.9,
        metadata={},
    )

    decision = SufficiencyDecisionEngine().decide(analysis, graph, [path])

    assert decision.sufficient is False
    assert "does not look compatible" in str(decision.missing_evidence)


def test_person_answer_candidate_rejects_organization_phrase() -> None:
    decision = SufficiencyDecisionEngine().decide(
        _chat17_question_analysis("Who was the Welsh member?"),
        _chat17_graph("PERSON"),
        [_chat17_bridge_path("Alpha Phi Alpha")],
    )

    assert decision.sufficient is False
    assert "does not look compatible" in str(decision.missing_evidence)


def test_person_answer_candidate_rejects_work_type_phrase() -> None:
    decision = SufficiencyDecisionEngine().decide(
        _chat17_question_analysis("What is her name?"),
        _chat17_graph("PERSON"),
        [_chat17_bridge_path("Comedy Series")],
    )

    assert decision.sufficient is False
    assert "does not look compatible" in str(decision.missing_evidence)


def test_incomplete_answer_candidate_with_dangling_word_is_insufficient() -> None:
    decision = SufficiencyDecisionEngine().decide(
        _chat17_question_analysis("Who funds the bowling team?"),
        _chat17_graph("PERSON"),
        [_chat17_bridge_path("Otto the")],
    )

    assert decision.sufficient is False
    assert "incomplete after bridge entity" in str(decision.missing_evidence)

def test_complete_bridge_records_passing_evidence_coverage() -> None:
    decision = SufficiencyDecisionEngine().decide(
        _chat17_question_analysis("Who developed the prototype pacemaker?"),
        _chat17_graph("PERSON"),
        [_chat17_bridge_path("R Adams Cowley")],
    )

    assert decision.sufficient is True
    assert decision.metadata["evidence_coverage"]["covered_evidence_unit_count"] == 2
    assert decision.metadata["evidence_coverage"]["evidence_source_count"] == 2
    assert "coverage_guard=true" in decision.rule_trace


def test_bridge_coverage_guard_rejects_single_document_bridge_path() -> None:
    analysis = _chat17_question_analysis("Who developed the prototype pacemaker?")
    graph = EvidenceGraph(
        nodes={
            "entity::seed": EvidenceGraphNode(
                node_id="entity::seed",
                node_type="entity",
                label="Seed",
            ),
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="Seed to Bridge evidence.",
                metadata={
                    "evidence_unit_id": "u1",
                    "chunk_id": "c1",
                    "doc_title": "Single Evidence Document",
                    "sentence_text": "Seed is linked to Bridge.",
                    "resolved_text": "Seed is linked to Bridge.",
                },
            ),
            "sentence::u2": EvidenceGraphNode(
                node_id="sentence::u2",
                node_type="sentence",
                label="Bridge to answer evidence.",
                metadata={
                    "evidence_unit_id": "u2",
                    "chunk_id": "c2",
                    "doc_title": "Single Evidence Document",
                    "sentence_text": "Bridge is linked to R Adams Cowley.",
                    "resolved_text": "Bridge is linked to R Adams Cowley.",
                },
            ),
        },
        edges=[
            EvidenceGraphEdge(
                edge_id="edge::u2_answer_type",
                source_id="sentence::u2",
                target_id="answer_type::person",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": "PERSON"},
            ),
        ],
        question_type="bridge",
        seed_entity_node_ids=["entity::seed"],
        metadata={
            "expected_answer_type": "PERSON",
            "required_relation_hints": ["written"],
        },
    )

    decision = SufficiencyDecisionEngine().decide(
        analysis,
        graph,
        [_chat17_bridge_path("R Adams Cowley")],
    )

    assert decision.sufficient is False
    assert "coverage is too narrow" in str(decision.missing_evidence)
    assert "coverage_guard=false" in decision.rule_trace


def test_coverage_guard_uses_doc_titles_before_chunk_count() -> None:
    graph = EvidenceGraph(
        nodes={
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="First evidence sentence.",
                metadata={
                    "evidence_unit_id": "u1",
                    "chunk_id": "c1",
                    "doc_title": "Same Document",
                    "sentence_text": "First evidence sentence.",
                },
            ),
            "sentence::u2": EvidenceGraphNode(
                node_id="sentence::u2",
                node_type="sentence",
                label="Second evidence sentence.",
                metadata={
                    "evidence_unit_id": "u2",
                    "chunk_id": "c2",
                    "doc_title": "Same Document",
                    "sentence_text": "Second evidence sentence.",
                },
            ),
        },
        edges=[],
        question_type="bridge",
    )

    coverage = compute_selected_evidence_coverage(graph, ["u1", "u2"])

    assert coverage["distinct_chunk_count"] == 2
    assert coverage["distinct_doc_title_count"] == 1
    assert coverage["evidence_source_count"] == 1
    assert has_minimum_evidence_coverage(
        question_type="bridge",
        coverage=coverage,
    ) is False


def test_number_answer_candidate_rejects_title_like_numeric_entity() -> None:
    analysis = _chat17_question_analysis(
        "What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?",
        answer_type=AnswerType.NUMBER,
    )
    path = _chat17_bridge_path("2013 Liqui Moly Bathurst 12 Hour")

    decision = SufficiencyDecisionEngine().decide(
        analysis,
        _chat17_graph("NUMBER"),
        [path],
    )

    assert decision.sufficient is False
    assert "does not look compatible" in str(decision.missing_evidence)


def test_role_aware_bridge_guard_rejects_weak_seed_side_grounding() -> None:
    analysis = QuestionAnalysis(
        raw_question="The Oberoi family is part of a hotel company that has a head office in what city?",
        normalized_question="the oberoi family is part of a hotel company that has a head office in what city?",
        question_type=QuestionType.BRIDGE,
        expected_answer_type=AnswerType.LOCATION,
        seed_entities=[EntityMention(text="Oberoi family", normalized="oberoi family", source="test")],
        required_relation_hints=[RelationHint(relation="headquarters", matched_text="head office", source="test")],
    )
    graph = EvidenceGraph(
        nodes={
            "entity::oberoi_family": EvidenceGraphNode(
                node_id="entity::oberoi_family",
                node_type="entity",
                label="Oberoi family",
            ),
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="A hotel company is associated with the Oberoi Group.",
                metadata={
                    "evidence_unit_id": "u1",
                    "chunk_id": "c1",
                    "doc_title": "Generic hotel company",
                    "sentence_text": "A hotel company is associated with the Oberoi Group.",
                },
            ),
            "sentence::u2": EvidenceGraphNode(
                node_id="sentence::u2",
                node_type="sentence",
                label="The Oberoi Group has its head office in Delhi.",
                metadata={
                    "evidence_unit_id": "u2",
                    "chunk_id": "c2",
                    "doc_title": "Oberoi Group",
                    "sentence_text": "The Oberoi Group has its head office in Delhi.",
                },
            ),
        },
        edges=[
            EvidenceGraphEdge(
                edge_id="edge::u2_answer_type",
                source_id="sentence::u2",
                target_id="answer_type::location",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": "LOCATION"},
            ),
        ],
        question_type="bridge",
        seed_entity_node_ids=["entity::oberoi_family"],
        metadata={"expected_answer_type": "LOCATION", "required_relation_hints": ["headquarters"]},
    )
    path = EvidencePath(
        path_id="path::weak_seed_role",
        question_type="bridge",
        node_ids=[
            "entity::oberoi_family",
            "sentence::u1",
            "entity::oberoi_group",
            "sentence::u2",
            "entity::delhi",
        ],
        edge_ids=[],
        evidence_unit_ids=["u1", "u2"],
        entity_chain=["Oberoi family", "Oberoi Group", "Delhi"],
        relation_chain=["headquarters"],
        answer_candidate="Delhi",
        answer_type="LOCATION",
        score=0.95,
        metadata={"bridge_entity": "Oberoi Group"},
    )

    decision = SufficiencyDecisionEngine().decide(analysis, graph, [path])

    assert decision.sufficient is False
    assert "seed-side evidence is not clearly connected" in str(decision.missing_evidence)
    assert "role_relevance_guard=false" in decision.rule_trace


def test_role_aware_factoid_guard_rejects_distractor_seedless_evidence() -> None:
    analysis = QuestionAnalysis(
        raw_question="The Thoen Stone is on display at a museum in what county?",
        normalized_question="the thoen stone is on display at a museum in what county?",
        question_type=QuestionType.FACTOID,
        expected_answer_type=AnswerType.LOCATION,
        seed_entities=[EntityMention(text="Thoen Stone", normalized="thoen stone", source="test")],
        required_relation_hints=[],
    )
    graph = EvidenceGraph(
        nodes={
            "entity::thoen_stone": EvidenceGraphNode(
                node_id="entity::thoen_stone",
                node_type="entity",
                label="Thoen Stone",
            ),
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="Clay County Historical Museum is located in Clay County.",
                metadata={
                    "evidence_unit_id": "u1",
                    "chunk_id": "c1",
                    "doc_title": "Clay County Historical Museum",
                    "sentence_text": "Clay County Historical Museum is located in Clay County.",
                },
            ),
            "entity::clay_county": EvidenceGraphNode(
                node_id="entity::clay_county",
                node_type="entity",
                label="Clay County",
            ),
        },
        edges=[
            EvidenceGraphEdge(
                edge_id="edge::u1_answer_type",
                source_id="sentence::u1",
                target_id="answer_type::location",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": "LOCATION"},
            ),
            EvidenceGraphEdge(
                edge_id="edge::u1_clay",
                source_id="sentence::u1",
                target_id="entity::clay_county",
                edge_type="sentence_mentions_entity",
            ),
        ],
        question_type="factoid",
        seed_entity_node_ids=["entity::thoen_stone"],
        metadata={"expected_answer_type": "LOCATION"},
    )
    path = EvidencePath(
        path_id="path::seedless_factoid",
        question_type="factoid",
        node_ids=["entity::thoen_stone", "sentence::u1", "entity::clay_county"],
        edge_ids=[],
        evidence_unit_ids=["u1"],
        entity_chain=["Thoen Stone", "Clay County"],
        relation_chain=[],
        answer_candidate="Clay County",
        answer_type="LOCATION",
        score=0.9,
        metadata={},
    )

    decision = SufficiencyDecisionEngine().decide(analysis, graph, [path])

    assert decision.sufficient is False
    assert "Factoid evidence is not clearly connected" in str(decision.missing_evidence)
    assert "role_relevance_guard=false" in decision.rule_trace


def test_role_aware_factoid_guard_keeps_direct_seed_answer_evidence_sufficient() -> None:
    analysis = QuestionAnalysis(
        raw_question="The Thoen Stone is on display at a museum in what county?",
        normalized_question="the thoen stone is on display at a museum in what county?",
        question_type=QuestionType.FACTOID,
        expected_answer_type=AnswerType.LOCATION,
        seed_entities=[EntityMention(text="Thoen Stone", normalized="thoen stone", source="test")],
        required_relation_hints=[],
    )
    graph = EvidenceGraph(
        nodes={
            "entity::thoen_stone": EvidenceGraphNode(
                node_id="entity::thoen_stone",
                node_type="entity",
                label="Thoen Stone",
            ),
            "sentence::u1": EvidenceGraphNode(
                node_id="sentence::u1",
                node_type="sentence",
                label="The Thoen Stone is displayed in a museum in Lawrence County.",
                metadata={
                    "evidence_unit_id": "u1",
                    "chunk_id": "c1",
                    "doc_title": "Thoen Stone",
                    "sentence_text": "The Thoen Stone is displayed in a museum in Lawrence County.",
                },
            ),
            "entity::lawrence_county": EvidenceGraphNode(
                node_id="entity::lawrence_county",
                node_type="entity",
                label="Lawrence County",
            ),
        },
        edges=[
            EvidenceGraphEdge(
                edge_id="edge::u1_answer_type",
                source_id="sentence::u1",
                target_id="answer_type::location",
                edge_type="sentence_has_answer_type",
                metadata={"answer_type": "LOCATION"},
            ),
            EvidenceGraphEdge(
                edge_id="edge::u1_lawrence",
                source_id="sentence::u1",
                target_id="entity::lawrence_county",
                edge_type="sentence_mentions_entity",
            ),
        ],
        question_type="factoid",
        seed_entity_node_ids=["entity::thoen_stone"],
        metadata={"expected_answer_type": "LOCATION"},
    )
    path = EvidencePath(
        path_id="path::direct_factoid",
        question_type="factoid",
        node_ids=["entity::thoen_stone", "sentence::u1", "entity::lawrence_county"],
        edge_ids=[],
        evidence_unit_ids=["u1"],
        entity_chain=["Thoen Stone", "Lawrence County"],
        relation_chain=[],
        answer_candidate="Lawrence County",
        answer_type="LOCATION",
        score=0.9,
        metadata={},
    )

    decision = SufficiencyDecisionEngine().decide(analysis, graph, [path])

    assert decision.sufficient is True
    assert "role_relevance_guard=true" in decision.rule_trace
