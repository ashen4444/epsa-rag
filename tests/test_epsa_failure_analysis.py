from __future__ import annotations

from epsa_rag.evaluation.epsa_failure_analysis import (
    analyze_epsa_failure_records,
    build_failure_analysis_markdown,
    infer_decision_family,
    is_false_sufficient_candidate,
    is_wrong_answer,
)


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "question_id": "q1",
        "question": "Question?",
        "gold_answer": "gold",
        "predicted_answer": "gold",
        "exact_match": 1.0,
        "partial_match": 1.0,
        "answer_f1": 1.0,
        "adaptive_stop_after_hop": "1",
        "epsa_hop1_sufficient": True,
        "epsa_final_sufficient": True,
        "selected_context_docs": 1,
        "selected_context_sentences": 1,
        "estimated_context_tokens": 50,
        "context_source": "epsa_pruned_context",
        "sufficiency_confidence": 0.9,
        "decision_reason": "Factoid path connects a seed entity to a typed answer candidate with supporting evidence.",
        "potential_false_sufficient_candidate": False,
        "potential_false_insufficient_candidate": False,
    }
    base.update(overrides)
    return base


def test_wrong_answer_requires_no_exact_or_partial_match() -> None:
    assert is_wrong_answer(_record(exact_match=0.0, partial_match=0.0)) is True
    assert is_wrong_answer(_record(exact_match=0.0, partial_match=1.0)) is False


def test_false_sufficient_candidate_uses_logged_flag_or_derived_condition() -> None:
    assert is_false_sufficient_candidate(
        _record(
            exact_match=0.0,
            partial_match=0.0,
            epsa_final_sufficient=True,
            potential_false_sufficient_candidate=False,
        )
    ) is True
    assert is_false_sufficient_candidate(
        _record(
            exact_match=0.0,
            partial_match=0.0,
            epsa_final_sufficient=False,
            potential_false_sufficient_candidate=False,
        )
    ) is False


def test_infer_decision_family_classifies_core_reasons() -> None:
    assert infer_decision_family(_record()) == "factoid_sufficient"
    assert infer_decision_family(
        _record(decision_reason="Complete bridge evidence path connects a seed entity through a bridge entity to an answer candidate.")
    ) == "bridge_sufficient"
    assert infer_decision_family(
        _record(decision_reason="Comparison requires later specialized comparison resolution; value comparison is not resolved in Chat 13.")
    ) == "comparison_insufficient"


def test_analyze_epsa_failure_records_computes_key_failure_buckets() -> None:
    records = [
        _record(question_id="correct_small"),
        _record(
            question_id="false_sufficient_hop1",
            gold_answer="Arthur's Magazine",
            predicted_answer="Insufficient evidence.",
            exact_match=0.0,
            partial_match=0.0,
            adaptive_stop_after_hop="1",
            epsa_final_sufficient=True,
            potential_false_sufficient_candidate=True,
            selected_context_docs=1,
            selected_context_sentences=1,
        ),
        _record(
            question_id="wrong_insufficient_hop2",
            gold_answer="Delhi",
            predicted_answer="Insufficient evidence.",
            exact_match=0.0,
            partial_match=0.0,
            adaptive_stop_after_hop="2",
            epsa_hop1_sufficient=False,
            epsa_final_sufficient=False,
            selected_context_docs=1,
            selected_context_sentences=1,
            context_source="epsa_pruned_context",
            decision_reason="No candidate factoid path satisfied all deterministic completeness rules.",
            potential_false_sufficient_candidate=False,
        ),
    ]

    report = analyze_epsa_failure_records(records, max_examples=5)

    assert report["num_records"] == 3
    assert report["sufficiency"]["epsa_final_sufficient_count"] == 2
    assert report["sufficiency"]["potential_false_sufficient_count"] == 1
    assert report["sufficiency"]["potential_false_sufficient_among_sufficient_rate"] == 0.5
    assert report["context"]["insufficient_pruned_context_count"] == 1
    assert report["failure_pattern_counts"]["hop1_stop_wrong_cases"] == 1
    assert report["failure_pattern_counts"]["hop2_used_wrong_cases"] == 1
    assert report["failure_pattern_counts"]["wrong_with_one_sentence"] == 2
    assert report["failure_pattern_counts"]["correct_with_small_context"] == 1
    assert report["grouped_counts"]["cases_by_adaptive_stop"]["1"]["count"] == 2
    assert report["examples"]["false_sufficient_cases"][0]["question_id"] == "false_sufficient_hop1"


def test_markdown_report_contains_core_sections() -> None:
    report = analyze_epsa_failure_records([_record()], max_examples=1)
    markdown = build_failure_analysis_markdown(report)

    assert "# EPSA Failure Analysis" in markdown
    assert "## Failure Pattern Counts" in markdown
    assert "## Recommended Next Checks" in markdown
