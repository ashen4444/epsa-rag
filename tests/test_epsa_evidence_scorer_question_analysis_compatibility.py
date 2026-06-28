from epsa_rag.epsa.evidence_scorer import EvidenceScorer
from epsa_rag.epsa.question_analyzer import QuestionAnalyzer
from epsa_rag.epsa.schemas import EvidenceUnit


def test_scorer_handles_actual_question_analysis_objects() -> None:
    question = QuestionAnalyzer().analyze("Who directed Inception?")
    unit = EvidenceUnit(
        evidence_unit_id="chunk_inception::s0",
        chunk_id="chunk_inception",
        doc_title="Inception",
        paragraph_index=0,
        sentence_id=0,
        sentence_text="Inception was directed by Christopher Nolan.",
        resolved_text="Inception was directed by Christopher Nolan.",
        entities=["Inception", "Christopher Nolan"],
        relation_hints=["directed"],
        answer_type_candidates=["PERSON", "ENTITY"],
        question_entity_overlap=["Inception"],
        question_token_overlap=0.5,
        is_supporting_sentence=True,
        retrieval_rank=1,
        retrieval_score=0.95,
    )

    scored = EvidenceScorer().score(unit, question)

    assert scored.score_breakdown["entity_match_score"] > 0.0
    assert scored.score_breakdown["relation_match_score"] > 0.0
    assert scored.score_breakdown["answer_type_match_score"] > 0.0
    assert scored.final_score > 0.0
