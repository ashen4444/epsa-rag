from epsa_rag.epsa.evidence_graph_builder import EvidenceGraphBuilder
from epsa_rag.epsa.evidence_path_searcher import EvidencePathSearcher
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


def _build_bridge_graph(include_irrelevant=True):
    question = _question_analysis()
    units = [
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
    if include_irrelevant:
        units.extend(
            [
                _scored(
                    _unit(
                        evidence_unit_id="chunk_noise_seed::s0",
                        chunk_id="chunk_noise_seed",
                        doc_title="Inception",
                        sentence_id=0,
                        sentence_text="Inception stars Leonardo DiCaprio.",
                        entities=["Inception", "Leonardo DiCaprio"],
                        relation_hints=["stars"],
                        answer_type_candidates=["PERSON"],
                        question_entity_overlap=["Inception"],
                        rank=3,
                    ),
                    0.35,
                ),
                _scored(
                    _unit(
                        evidence_unit_id="chunk_noise_answer::s0",
                        chunk_id="chunk_noise_answer",
                        doc_title="Leonardo DiCaprio",
                        sentence_id=0,
                        sentence_text="Leonardo DiCaprio was born in Los Angeles.",
                        entities=["Leonardo DiCaprio", "Los Angeles"],
                        relation_hints=["born"],
                        answer_type_candidates=["LOCATION"],
                        rank=4,
                    ),
                    0.30,
                ),
            ]
        )
    return question, EvidenceGraphBuilder().build(question, units)


def test_path_searcher_returns_ranked_bridge_candidate_path():
    question, graph = _build_bridge_graph()
    paths = EvidencePathSearcher().search_paths(graph, question, max_paths=5)

    assert paths
    top_path = paths[0]
    assert top_path.question_type == "bridge"
    assert top_path.answer_candidate == "London"
    assert top_path.entity_chain == ["Inception", "Christopher Nolan", "London"]
    assert set(top_path.evidence_unit_ids) == {"chunk_inception::s0", "chunk_nolan::s0"}
    assert "directed" in top_path.relation_chain
    assert "born" in top_path.relation_chain
    assert top_path.score > 0
    assert top_path.metadata["makes_sufficiency_decision"] is False
    assert not hasattr(top_path, "sufficient")


def test_path_searcher_ranks_useful_bridge_path_above_irrelevant_path():
    question, graph = _build_bridge_graph(include_irrelevant=True)
    paths = EvidencePathSearcher().search_paths(graph, question, max_paths=10)

    candidates = [path.answer_candidate for path in paths]
    assert "London" in candidates
    assert "Los Angeles" in candidates
    assert candidates.index("London") < candidates.index("Los Angeles")


def test_path_searcher_returns_factoid_candidate_path():
    question = _question_analysis(
        question_type="factoid",
        seed_entities=["France"],
        expected_answer_type="LOCATION",
        required_relation_hints=["capital"],
        normalized_question="what is the capital of france",
    )
    graph = EvidenceGraphBuilder().build(
        question,
        [
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
        ],
    )

    paths = EvidencePathSearcher().search_paths(graph, question)
    assert paths
    assert paths[0].question_type == "factoid"
    assert paths[0].answer_candidate == "Paris"
    assert paths[0].entity_chain == ["France", "Paris"]
    assert paths[0].evidence_unit_ids == ["chunk_france::s0"]


def test_path_searcher_handles_no_path_cases_safely():
    question = _question_analysis(seed_entities=["Unknown Film"])
    graph = EvidenceGraphBuilder().build(question, [])

    assert EvidencePathSearcher().search_paths(graph, question) == []


def test_path_searcher_returns_comparison_target_partial_paths_without_deciding():
    question = _question_analysis(
        question_type="comparison",
        seed_entities=["River A", "River B"],
        expected_answer_type="NUMBER",
        required_relation_hints=["length"],
        comparison_targets=["River A", "River B"],
        normalized_question="which river is longer river a or river b",
    )
    graph = EvidenceGraphBuilder().build(
        question,
        [
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
        ],
    )

    paths = EvidencePathSearcher().search_paths(graph, question, max_paths=5)
    assert len(paths) >= 2
    assert {path.metadata["comparison_target"] for path in paths} >= {"River A", "River B"}
    assert all(path.metadata["does_not_compare_values_yet"] is True for path in paths)
    assert all(path.metadata["makes_sufficiency_decision"] is False for path in paths)


def test_path_searcher_returns_yes_no_evidence_paths_without_yes_no_decision():
    question = _question_analysis(
        question_type="yes_no",
        seed_entities=["Inception", "Christopher Nolan"],
        expected_answer_type="BOOLEAN",
        required_relation_hints=["directed"],
        normalized_question="was inception directed by christopher nolan",
    )
    graph = EvidenceGraphBuilder().build(
        question,
        [
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
        ],
    )

    paths = EvidencePathSearcher().search_paths(graph, question)
    assert paths
    assert paths[0].question_type == "yes_no"
    assert paths[0].answer_candidate is None
    assert paths[0].metadata["does_not_decide_yes_no"] is True
