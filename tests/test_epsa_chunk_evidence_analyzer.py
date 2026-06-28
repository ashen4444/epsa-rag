from dataclasses import dataclass

from epsa_rag.epsa.chunk_evidence_analyzer import CandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.question_analyzer import QuestionAnalyzer
from epsa_rag.epsa.schemas import AnswerType


@dataclass
class ObjectChunk:
    chunk_id: str
    question_id: str
    doc_title: str
    paragraph_index: int
    chunk_text: str
    paragraph_text: str
    sentences: list
    is_supporting_doc: bool = False
    supporting_sentence_ids: list | None = None


def _sample_question_analysis():
    return QuestionAnalyzer().analyze("Where was the director of Inception born?")


def _sample_chunk_dict():
    paragraph = (
        "Inception is a 2010 science fiction film written and directed by Christopher Nolan. "
        "Nolan was born in London, England."
    )
    return {
        "chunk_id": "hotpot_train_0001::Inception::p0",
        "question_id": "hotpot_train_0001",
        "doc_title": "Inception",
        "paragraph_index": 0,
        "chunk_text": f"Title: Inception\nParagraph: {paragraph}",
        "paragraph_text": paragraph,
        "sentences": [
            {"sentence_id": 0, "text": "Inception is a 2010 science fiction film written and directed by Christopher Nolan."},
            {"sentence_id": 1, "text": "Nolan was born in London, England."},
        ],
        "is_supporting_doc": True,
        "supporting_sentence_ids": [0, 1],
    }


def test_extracts_doc_title_as_entity_and_preserves_metadata():
    evidence = CandidateChunkEvidenceAnalyzer().analyze(
        _sample_chunk_dict(),
        question_analysis=_sample_question_analysis(),
        retrieval_rank=1,
        retrieval_score=0.42,
    )

    assert evidence.chunk_id == "hotpot_train_0001::Inception::p0"
    assert evidence.doc_title == "Inception"
    assert evidence.paragraph_index == 0
    assert evidence.retrieval_rank == 1
    assert evidence.retrieval_score == 0.42
    assert evidence.source_question_id == "hotpot_train_0001"
    assert evidence.sentences
    assert "Inception" in {entity.text for entity in evidence.entities}


def test_extracts_capitalized_entities_and_relation_hints():
    evidence = CandidateChunkEvidenceAnalyzer().analyze(
        _sample_chunk_dict(),
        question_analysis=_sample_question_analysis(),
    )

    entity_texts = {entity.text for entity in evidence.entities}
    relation_labels = {hint.relation for hint in evidence.relation_hints}

    assert "Christopher Nolan" in entity_texts
    assert "London" in entity_texts
    assert "directed" in relation_labels
    assert "born" in relation_labels


def test_extracts_answer_type_candidates():
    evidence = CandidateChunkEvidenceAnalyzer().analyze(
        _sample_chunk_dict(),
        question_analysis=_sample_question_analysis(),
    )

    candidate_types = {candidate.answer_type for candidate in evidence.answer_type_candidates}
    candidate_texts = {candidate.text for candidate in evidence.answer_type_candidates}

    assert AnswerType.DATE in candidate_types
    assert AnswerType.LOCATION in candidate_types
    assert "2010" in candidate_texts
    assert "London" in candidate_texts


def test_detects_question_overlap_and_title_match():
    evidence = CandidateChunkEvidenceAnalyzer().analyze(
        _sample_chunk_dict(),
        question_analysis=_sample_question_analysis(),
    )

    assert "Inception" in evidence.question_entity_overlap
    assert "director" in evidence.question_token_overlap or "directed" in evidence.question_token_overlap
    assert evidence.question_token_overlap_score > 0
    assert evidence.is_title_match is True


def test_identifies_potential_bridge_entities():
    evidence = CandidateChunkEvidenceAnalyzer().analyze(
        _sample_chunk_dict(),
        question_analysis=_sample_question_analysis(),
    )

    bridge_texts = {entity.text for entity in evidence.potential_bridge_entities}
    assert "Christopher Nolan" in bridge_texts


def test_handles_object_like_chunks():
    chunk_dict = _sample_chunk_dict()
    object_chunk = ObjectChunk(**{key: chunk_dict[key] for key in ObjectChunk.__dataclass_fields__ if key in chunk_dict})

    evidence = CandidateChunkEvidenceAnalyzer().analyze(
        object_chunk,
        question_analysis=_sample_question_analysis(),
        retrieval_rank=2,
        retrieval_score=0.31,
    )

    assert evidence.chunk_id == chunk_dict["chunk_id"]
    assert evidence.doc_title == "Inception"
    assert evidence.retrieval_rank == 2
    assert evidence.retrieval_score == 0.31
