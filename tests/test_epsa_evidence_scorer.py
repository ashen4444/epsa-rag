from __future__ import annotations

from types import SimpleNamespace

from epsa_rag.epsa.evidence_scorer import EvidenceScorer
from epsa_rag.epsa.schemas import EvidenceUnit


def _question_analysis() -> SimpleNamespace:
    return SimpleNamespace(
        question_type="bridge",
        seed_entities=["Inception"],
        expected_answer_type="PERSON",
        required_relation_hints=["directed"],
        comparison_targets=[],
        normalized_question="Who directed Inception?",
    )


def _evidence_unit(
    *,
    evidence_unit_id: str = "chunk_1::s0",
    chunk_id: str = "chunk_1",
    doc_title: str = "Inception",
    sentence_id: int = 0,
    sentence_text: str = "Inception was directed by Christopher Nolan.",
    resolved_text: str = "Inception was directed by Christopher Nolan.",
    entities: list[str] | None = None,
    relation_hints: list[str] | None = None,
    answer_type_candidates: list[str] | None = None,
    retrieval_rank: int | None = 1,
    retrieval_score: float | None = 0.95,
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_unit_id=evidence_unit_id,
        chunk_id=chunk_id,
        doc_title=doc_title,
        paragraph_index=0,
        sentence_id=sentence_id,
        sentence_text=sentence_text,
        resolved_text=resolved_text,
        entities=entities if entities is not None else ["Inception", "Christopher Nolan"],
        relation_hints=relation_hints if relation_hints is not None else ["directed"],
        answer_type_candidates=answer_type_candidates if answer_type_candidates is not None else ["PERSON", "ENTITY"],
        question_entity_overlap=["Inception"],
        question_token_overlap=0.5,
        is_supporting_sentence=None,
        retrieval_rank=retrieval_rank,
        retrieval_score=retrieval_score,
    )


def test_score_returns_final_score_and_breakdown() -> None:
    scorer = EvidenceScorer()
    scored = scorer.score(
        evidence_unit=_evidence_unit(),
        question_analysis=_question_analysis(),
    )

    assert scored.final_score > 0.0
    assert isinstance(scored.score_breakdown, dict)

    expected_keys = {
        "entity_match_score",
        "relation_match_score",
        "answer_type_match_score",
        "token_overlap_score",
        "title_match_score",
        "retrieval_score_component",
        "bridge_entity_score",
        "noise_penalty",
    }

    assert expected_keys.issubset(scored.score_breakdown.keys())


def test_entity_match_increases_score() -> None:
    scorer = EvidenceScorer()

    matching = _evidence_unit(
        entities=["Inception", "Christopher Nolan"],
        resolved_text="Inception was directed by Christopher Nolan.",
    )

    non_matching = _evidence_unit(
        evidence_unit_id="chunk_2::s0",
        chunk_id="chunk_2",
        doc_title="Random Article",
        sentence_text="The album was released in 1995.",
        resolved_text="The album was released in 1995.",
        entities=["Random Article"],
        relation_hints=["released"],
        answer_type_candidates=["DATE", "ENTITY"],
        retrieval_rank=6,
        retrieval_score=0.2,
    )

    matching_score = scorer.score(matching, _question_analysis())
    non_matching_score = scorer.score(non_matching, _question_analysis())

    assert matching_score.score_breakdown["entity_match_score"] > non_matching_score.score_breakdown["entity_match_score"]
    assert matching_score.final_score > non_matching_score.final_score


def test_relation_match_increases_score() -> None:
    scorer = EvidenceScorer()

    with_relation = _evidence_unit(relation_hints=["directed"])
    without_relation = _evidence_unit(
        evidence_unit_id="chunk_2::s0",
        relation_hints=["released"],
    )

    with_relation_score = scorer.score(with_relation, _question_analysis())
    without_relation_score = scorer.score(without_relation, _question_analysis())

    assert with_relation_score.score_breakdown["relation_match_score"] == 1.0
    assert without_relation_score.score_breakdown["relation_match_score"] == 0.0
    assert with_relation_score.final_score > without_relation_score.final_score


def test_answer_type_match_increases_score() -> None:
    scorer = EvidenceScorer()

    matching_answer_type = _evidence_unit(answer_type_candidates=["PERSON", "ENTITY"])
    non_matching_answer_type = _evidence_unit(
        evidence_unit_id="chunk_2::s0",
        answer_type_candidates=["DATE", "NUMBER"],
    )

    matching_score = scorer.score(matching_answer_type, _question_analysis())
    non_matching_score = scorer.score(non_matching_answer_type, _question_analysis())

    assert matching_score.score_breakdown["answer_type_match_score"] == 1.0
    assert non_matching_score.score_breakdown["answer_type_match_score"] == 0.0
    assert matching_score.final_score > non_matching_score.final_score


def test_bridge_entity_contribution_is_given_for_non_question_entities() -> None:
    scorer = EvidenceScorer()

    bridge_unit = _evidence_unit(
        entities=["Inception", "Christopher Nolan"],
        relation_hints=["directed"],
    )

    no_bridge_unit = _evidence_unit(
        evidence_unit_id="chunk_2::s0",
        entities=["Inception"],
        relation_hints=["directed"],
    )

    bridge_score = scorer.score(bridge_unit, _question_analysis())
    no_bridge_score = scorer.score(no_bridge_unit, _question_analysis())

    assert bridge_score.score_breakdown["bridge_entity_score"] > no_bridge_score.score_breakdown["bridge_entity_score"]


def test_noise_penalty_applies_to_low_signal_sentence() -> None:
    scorer = EvidenceScorer()

    low_signal = _evidence_unit(
        evidence_unit_id="noise::s0",
        chunk_id="noise",
        doc_title="Noise",
        sentence_id=0,
        sentence_text="Yes.",
        resolved_text="Yes.",
        entities=[],
        relation_hints=[],
        answer_type_candidates=[],
        retrieval_rank=None,
        retrieval_score=None,
    )

    scored = scorer.score(low_signal, _question_analysis())

    assert scored.score_breakdown["noise_penalty"] > 0.0


def test_useful_evidence_ranks_above_irrelevant_evidence() -> None:
    scorer = EvidenceScorer()

    useful = _evidence_unit(
        evidence_unit_id="useful::s0",
        chunk_id="useful",
        doc_title="Inception",
        sentence_text="Inception was directed by Christopher Nolan.",
        resolved_text="Inception was directed by Christopher Nolan.",
        entities=["Inception", "Christopher Nolan"],
        relation_hints=["directed"],
        answer_type_candidates=["PERSON", "ENTITY"],
        retrieval_rank=1,
        retrieval_score=0.9,
    )

    irrelevant = _evidence_unit(
        evidence_unit_id="irrelevant::s0",
        chunk_id="irrelevant",
        doc_title="Random Film",
        sentence_text="The film was released in 1995.",
        resolved_text="The film was released in 1995.",
        entities=["Random Film"],
        relation_hints=["released"],
        answer_type_candidates=["DATE", "ENTITY"],
        retrieval_rank=8,
        retrieval_score=0.1,
    )

    ranked = scorer.rank(
        evidence_units=[irrelevant, useful],
        question_analysis=_question_analysis(),
    )

    assert ranked[0].evidence_unit.evidence_unit_id == "useful::s0"
    assert ranked[0].final_score > ranked[1].final_score


def test_scorer_does_not_make_sufficiency_decision() -> None:
    scorer = EvidenceScorer()
    scored = scorer.score(
        evidence_unit=_evidence_unit(),
        question_analysis=_question_analysis(),
    )

    assert not hasattr(scored, "sufficient")
    assert not hasattr(scored, "is_sufficient")
    assert not hasattr(scored.evidence_unit, "sufficient")
    assert not hasattr(scored.evidence_unit, "is_sufficient")


def test_score_many_returns_scored_units_for_all_inputs() -> None:
    scorer = EvidenceScorer()

    units = [
        _evidence_unit(evidence_unit_id="chunk_1::s0"),
        _evidence_unit(evidence_unit_id="chunk_1::s1", sentence_id=1),
    ]

    scored_units = scorer.score_many(
        evidence_units=units,
        question_analysis=_question_analysis(),
    )

    assert len(scored_units) == 2
    assert scored_units[0].evidence_unit.evidence_unit_id == "chunk_1::s0"
    assert scored_units[1].evidence_unit.evidence_unit_id == "chunk_1::s1"