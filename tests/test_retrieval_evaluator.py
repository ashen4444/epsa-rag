from epsa_rag.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
    RetrievalQuestion,
    build_chunk_lookup,
    build_retrieval_questions_from_chunks,
)


def test_evaluator_computes_per_question_metrics_from_controlled_results() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "doc_title": "Marie Curie",
            "chunk_text": "Title: Marie Curie\nParagraph: Marie Curie discovered radium.",
        },
        {
            "chunk_id": "c2",
            "doc_title": "Noise",
            "chunk_text": "Title: Noise\nParagraph: Irrelevant paragraph.",
        },
        {
            "chunk_id": "c3",
            "doc_title": "Pierre Curie",
            "chunk_text": "Title: Pierre Curie\nParagraph: Pierre Curie was a physicist.",
        },
    ]
    chunk_lookup = build_chunk_lookup(chunks)

    question = RetrievalQuestion(
        question_id="q1",
        question="Who worked with Marie Curie?",
        gold_answer="Pierre Curie",
        gold_supporting_titles=["Marie Curie", "Pierre Curie"],
        gold_supporting_chunk_ids=["c1", "c3"],
    )

    retrieved_results = [
        {"chunk_id": "c2", "score": 0.9},
        {"chunk_id": "c1", "score": 0.8},
        {"chunk_id": "c3", "score": 0.7},
    ]

    evaluator = RetrievalEvaluator()
    record = evaluator.evaluate_question(
        question=question,
        retrieved_results=retrieved_results,
        chunk_lookup=chunk_lookup,
        latency_ms=12.5,
    )

    assert record.top1_supporting_hit is False
    assert record.both_supporting_found_at_2 is False
    assert record.both_supporting_found_at_5 is True
    assert record.supporting_doc_recall_at_2 == 0.5
    assert record.supporting_doc_recall_at_5 == 1.0
    assert record.first_support_rank == 2
    assert record.first_support_mrr_at_10 == 0.5
    assert record.latency_ms == 12.5
    assert record.skipped is False


def test_retrieved_is_supporting_doc_flag_is_not_used_as_relevance_label() -> None:
    chunks = [
        {
            "chunk_id": "joe_heck",
            "doc_title": "Joe Heck",
            "chunk_text": "Title: Joe Heck\nParagraph: Distractor paragraph.",
            "is_supporting_doc": True,
        },
        {
            "chunk_id": "eisenhower",
            "doc_title": "Dwight D. Eisenhower",
            "chunk_text": "Title: Dwight D. Eisenhower\nParagraph: Gold paragraph.",
            "is_supporting_doc": False,
        },
    ]
    chunk_lookup = build_chunk_lookup(chunks)

    question = RetrievalQuestion(
        question_id="q2",
        question="Controlled evaluation question",
        gold_answer="answer",
        gold_supporting_titles=["Dwight D. Eisenhower", "R Adams Cowley"],
        gold_supporting_chunk_ids=["eisenhower"],
    )

    retrieved_results = [
        {"chunk_id": "joe_heck", "score": 1.0},
        {"chunk_id": "eisenhower", "score": 0.9},
    ]

    evaluator = RetrievalEvaluator()
    record = evaluator.evaluate_question(
        question=question,
        retrieved_results=retrieved_results,
        chunk_lookup=chunk_lookup,
    )

    assert record.top1_supporting_hit is False
    assert record.supporting_doc_recall_at_2 == 0.5
    assert record.both_supporting_found_at_2 is False


def test_evaluator_handles_missing_supporting_titles_as_skipped() -> None:
    evaluator = RetrievalEvaluator()

    question = RetrievalQuestion(
        question_id="q3",
        question="Question text",
        gold_answer="answer",
        gold_supporting_titles=[],
    )

    record = evaluator.evaluate_question(
        question=question,
        retrieved_results=[],
        chunk_lookup={},
    )

    assert record.skipped is True
    assert record.skip_reason == "missing_gold_supporting_titles"


def test_summary_aggregates_valid_records() -> None:
    evaluator = RetrievalEvaluator()

    question_1 = RetrievalQuestion(
        question_id="q1",
        question="Question 1",
        gold_answer="answer",
        gold_supporting_titles=["A", "B"],
    )
    question_2 = RetrievalQuestion(
        question_id="q2",
        question="Question 2",
        gold_answer="answer",
        gold_supporting_titles=["A", "B"],
    )

    chunk_lookup = {
        "a": {"chunk_id": "a", "doc_title": "A", "chunk_text": "A text"},
        "b": {"chunk_id": "b", "doc_title": "B", "chunk_text": "B text"},
        "x": {"chunk_id": "x", "doc_title": "X", "chunk_text": "X text"},
    }

    record_1 = evaluator.evaluate_question(
        question=question_1,
        retrieved_results=[{"chunk_id": "a"}, {"chunk_id": "b"}],
        chunk_lookup=chunk_lookup,
    )
    record_2 = evaluator.evaluate_question(
        question=question_2,
        retrieved_results=[{"chunk_id": "x"}, {"chunk_id": "a"}],
        chunk_lookup=chunk_lookup,
    )

    summary = evaluator.summarize([record_1, record_2])

    assert summary.evaluated_questions == 2
    assert summary.top1_supporting_hit_rate == 0.5
    assert summary.both_supporting_found_rate_at_2 == 0.5
    assert summary.mean_supporting_doc_recall_at_2 == 0.75


def test_build_retrieval_questions_from_chunks_groups_gold_titles_by_question() -> None:
    chunks = [
        {
            "chunk_id": "q1_a",
            "question_id": "q1",
            "question": "Question 1",
            "answer": "Answer 1",
            "doc_title": "A",
            "is_supporting_doc": True,
        },
        {
            "chunk_id": "q1_b",
            "question_id": "q1",
            "question": "Question 1",
            "answer": "Answer 1",
            "doc_title": "B",
            "is_supporting_doc": True,
        },
        {
            "chunk_id": "q1_c",
            "question_id": "q1",
            "question": "Question 1",
            "answer": "Answer 1",
            "doc_title": "C",
            "is_supporting_doc": False,
        },
    ]

    questions = build_retrieval_questions_from_chunks(chunks)

    assert len(questions) == 1
    assert questions[0].question_id == "q1"
    assert questions[0].gold_supporting_titles == ["A", "B"]
    assert questions[0].gold_supporting_chunk_ids == ["q1_a", "q1_b"]