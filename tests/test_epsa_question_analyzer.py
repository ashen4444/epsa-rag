from epsa_rag.epsa.question_analyzer import QuestionAnalyzer
from epsa_rag.epsa.schemas import AnswerType, QuestionType


def test_detects_yes_no_question():
    analysis = QuestionAnalyzer().analyze("Is Paris the capital of France?")

    assert analysis.question_type == QuestionType.YES_NO
    assert analysis.expected_answer_type == AnswerType.BOOLEAN


def test_detects_comparison_question_and_targets():
    analysis = QuestionAnalyzer().analyze("Which of The Hunger Games and Divergent was released earlier?")

    assert analysis.question_type == QuestionType.COMPARISON
    assert analysis.expected_answer_type in {AnswerType.ENTITY, AnswerType.DATE}
    target_texts = {entity.text for entity in analysis.comparison_targets}
    assert "The Hunger Games" in target_texts
    assert "Divergent" in target_texts


def test_detects_bridge_question():
    analysis = QuestionAnalyzer().analyze("Where was the director of Inception born?")

    assert analysis.question_type == QuestionType.BRIDGE
    assert analysis.expected_answer_type == AnswerType.LOCATION
    assert "Inception" in {entity.text for entity in analysis.seed_entities}
    assert {hint.relation for hint in analysis.required_relation_hints} >= {"directed", "born"}


def test_infers_location_date_number_person_answer_types():
    analyzer = QuestionAnalyzer()

    assert analyzer.analyze("Where was Christopher Nolan born?").expected_answer_type == AnswerType.LOCATION
    assert analyzer.analyze("When was Inception released?").expected_answer_type == AnswerType.DATE
    assert analyzer.analyze("How many people live in Paris?").expected_answer_type == AnswerType.NUMBER
    assert analyzer.analyze("Who directed Inception?").expected_answer_type == AnswerType.PERSON


def test_extracts_quoted_and_capitalized_seed_entities():
    analysis = QuestionAnalyzer().analyze('Who wrote "Pride and Prejudice" after working with Jane Austen?')

    entity_texts = {entity.text for entity in analysis.seed_entities}
    assert "Pride and Prejudice" in entity_texts
    assert "Jane Austen" in entity_texts


def test_returns_stable_normalized_question():
    analysis = QuestionAnalyzer().analyze("  Where   was   Inception   released?  ")

    assert analysis.normalized_question == "where was inception released?"
