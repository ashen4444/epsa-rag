from __future__ import annotations

from types import SimpleNamespace

from epsa_rag.epsa.evidence_unit_extractor import EvidenceUnitExtractor


def _question_analysis() -> SimpleNamespace:
    return SimpleNamespace(
        question_type="bridge",
        seed_entities=["Marie Curie"],
        expected_answer_type="ENTITY",
        required_relation_hints=["discovered"],
        comparison_targets=[],
        normalized_question="What did Marie Curie discover?",
    )


def _candidate_evidence() -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id="hotpot_1::Marie_Curie::p0",
        doc_title="Marie Curie",
        paragraph_index=0,
        retrieval_rank=1,
        retrieval_score=0.92,
    )


def test_extracts_one_evidence_unit_per_sentence_from_dict_metadata() -> None:
    chunk = {
        "chunk_id": "hotpot_1::Marie_Curie::p0",
        "doc_title": "Marie Curie",
        "paragraph_index": 0,
        "chunk_text": "Marie Curie was a physicist. She discovered polonium and radium.",
        "sentences": [
            {
                "sentence_id": 0,
                "text": "Marie Curie was a physicist.",
                "start_char": 0,
                "end_char": 29,
            },
            {
                "sentence_id": 1,
                "text": "She discovered polonium and radium.",
                "start_char": 30,
                "end_char": 65,
            },
        ],
        "supporting_sentence_ids": [1],
    }

    extractor = EvidenceUnitExtractor()
    units = extractor.extract_from_chunk(
        candidate_evidence=_candidate_evidence(),
        chunk=chunk,
        question_analysis=_question_analysis(),
    )

    assert len(units) == 2

    assert units[0].evidence_unit_id == "hotpot_1::Marie_Curie::p0::s0"
    assert units[0].chunk_id == "hotpot_1::Marie_Curie::p0"
    assert units[0].doc_title == "Marie Curie"
    assert units[0].paragraph_index == 0
    assert units[0].sentence_id == 0
    assert units[0].sentence_text == "Marie Curie was a physicist."

    assert units[1].evidence_unit_id == "hotpot_1::Marie_Curie::p0::s1"
    assert units[1].sentence_text == "She discovered polonium and radium."
    assert units[1].resolved_text == "Marie Curie discovered polonium and radium."
    assert units[1].is_supporting_sentence is True
    assert units[1].retrieval_rank == 1
    assert units[1].retrieval_score == 0.92


def test_handles_plain_string_sentences() -> None:
    chunk = {
        "chunk_id": "chunk_plain",
        "doc_title": "Christopher Nolan",
        "paragraph_index": 2,
        "sentences": [
            "Christopher Nolan is a British-American filmmaker.",
            "He directed Inception.",
        ],
    }

    candidate = SimpleNamespace(
        chunk_id="chunk_plain",
        doc_title="Christopher Nolan",
        paragraph_index=2,
        retrieval_rank=3,
        retrieval_score=0.55,
    )

    question = SimpleNamespace(
        question_type="bridge",
        seed_entities=["Inception"],
        expected_answer_type="PERSON",
        required_relation_hints=["directed"],
        comparison_targets=[],
        normalized_question="Who directed Inception?",
    )

    extractor = EvidenceUnitExtractor()
    units = extractor.extract_from_chunk(
        candidate_evidence=candidate,
        chunk=chunk,
        question_analysis=question,
    )

    assert len(units) == 2
    assert units[0].sentence_id == 0
    assert units[1].sentence_id == 1
    assert units[1].resolved_text == "Christopher Nolan directed Inception."
    assert "directed" in units[1].relation_hints
    assert "Inception" in units[1].entities


def test_falls_back_to_chunk_text_when_sentences_are_missing() -> None:
    chunk = {
        "chunk_id": "fallback_chunk",
        "doc_title": "London",
        "paragraph_index": 4,
        "chunk_text": "London is the capital and largest city of England.",
    }

    candidate = SimpleNamespace(
        chunk_id="fallback_chunk",
        doc_title="London",
        paragraph_index=4,
        retrieval_rank=5,
        retrieval_score=0.44,
    )

    question = SimpleNamespace(
        question_type="factoid",
        seed_entities=["England"],
        expected_answer_type="LOCATION",
        required_relation_hints=["capital"],
        comparison_targets=[],
        normalized_question="What is the capital of England?",
    )

    extractor = EvidenceUnitExtractor()
    units = extractor.extract_from_chunk(
        candidate_evidence=candidate,
        chunk=chunk,
        question_analysis=question,
    )

    assert len(units) == 1
    assert units[0].evidence_unit_id == "fallback_chunk::s0"
    assert units[0].sentence_id == 0
    assert units[0].sentence_text == "London is the capital and largest city of England."
    assert "capital" in units[0].relation_hints
    assert "LOCATION" in units[0].answer_type_candidates


def test_handles_object_like_chunk() -> None:
    chunk = SimpleNamespace(
        chunk_id="object_chunk",
        doc_title="Inception",
        paragraph_index=1,
        chunk_text="",
        sentences=[
            {
                "sentence_id": 7,
                "text": "Inception was directed by Christopher Nolan.",
            }
        ],
        supporting_sentence_ids=[7],
    )

    candidate = SimpleNamespace(
        chunk_id="object_chunk",
        doc_title="Inception",
        paragraph_index=1,
        retrieval_rank=2,
        retrieval_score=0.77,
    )

    question = SimpleNamespace(
        question_type="bridge",
        seed_entities=["Inception"],
        expected_answer_type="PERSON",
        required_relation_hints=["directed"],
        comparison_targets=[],
        normalized_question="Who directed Inception?",
    )

    extractor = EvidenceUnitExtractor()
    units = extractor.extract_from_chunk(
        candidate_evidence=candidate,
        chunk=chunk,
        question_analysis=question,
    )

    assert len(units) == 1
    assert units[0].evidence_unit_id == "object_chunk::s7"
    assert units[0].is_supporting_sentence is True
    assert "directed" in units[0].relation_hints
    assert "Christopher Nolan" in units[0].entities


def test_extracts_entities_relation_hints_answer_types_and_question_overlap() -> None:
    chunk = {
        "chunk_id": "curie_chunk",
        "doc_title": "Marie Curie",
        "paragraph_index": 0,
        "sentences": [
            {
                "sentence_id": 0,
                "text": "Marie Curie discovered polonium and radium with Pierre Curie in 1898.",
            }
        ],
    }

    extractor = EvidenceUnitExtractor()
    units = extractor.extract_from_chunk(
        candidate_evidence=_candidate_evidence(),
        chunk=chunk,
        question_analysis=_question_analysis(),
    )

    unit = units[0]

    assert "Marie Curie" in unit.entities
    assert "Pierre Curie" in unit.entities
    assert "discovered" in unit.relation_hints
    assert "DATE" in unit.answer_type_candidates
    assert "NUMBER" in unit.answer_type_candidates
    assert "ENTITY" in unit.answer_type_candidates
    assert "Marie Curie" in unit.question_entity_overlap
    assert unit.question_token_overlap > 0.0


def test_extract_many_combines_units_from_multiple_chunks() -> None:
    candidate_1 = SimpleNamespace(
        chunk_id="chunk_1",
        doc_title="Marie Curie",
        paragraph_index=0,
        retrieval_rank=1,
        retrieval_score=0.9,
    )
    chunk_1 = {
        "chunk_id": "chunk_1",
        "doc_title": "Marie Curie",
        "paragraph_index": 0,
        "sentences": ["Marie Curie discovered radium."],
    }

    candidate_2 = SimpleNamespace(
        chunk_id="chunk_2",
        doc_title="Radium",
        paragraph_index=0,
        retrieval_rank=2,
        retrieval_score=0.8,
    )
    chunk_2 = {
        "chunk_id": "chunk_2",
        "doc_title": "Radium",
        "paragraph_index": 0,
        "sentences": ["Radium is a chemical element."],
    }

    extractor = EvidenceUnitExtractor()
    units = extractor.extract_many(
        candidate_chunk_pairs=[(candidate_1, chunk_1), (candidate_2, chunk_2)],
        question_analysis=_question_analysis(),
    )

    assert len(units) == 2
    assert units[0].chunk_id == "chunk_1"
    assert units[1].chunk_id == "chunk_2"