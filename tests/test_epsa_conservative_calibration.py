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
