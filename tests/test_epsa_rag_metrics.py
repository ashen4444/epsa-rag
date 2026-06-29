from __future__ import annotations

from epsa_rag.evaluation.epsa_rag_metrics import (
    percentage_reduction,
    safe_mean,
    safe_rate,
    summarize_epsa_rag_records,
)


def test_safe_mean_ignores_missing_and_non_numeric_values() -> None:
    assert safe_mean([1, "2", None, "", "bad"]) == 1.5


def test_safe_rate_handles_zero_total() -> None:
    assert safe_rate(3, 0) == 0.0
    assert safe_rate(2, 4) == 0.5


def test_percentage_reduction_reports_positive_reduction() -> None:
    assert percentage_reduction(100, 75) == 25.0


def test_percentage_reduction_reports_negative_when_proposed_is_larger() -> None:
    assert percentage_reduction(100, 125) == -25.0


def test_summarize_epsa_rag_records_computes_core_averages_and_rates() -> None:
    records = [
        {
            "exact_match": 1.0,
            "partial_match": 1.0,
            "answer_precision": 1.0,
            "answer_recall": 1.0,
            "answer_f1": 1.0,
            "selected_context_docs": 2,
            "selected_context_sentences": 2,
            "estimated_context_tokens": 200,
            "total_llm_tokens": 300,
            "latency_ms": 1000,
            "adaptive_stop_after_hop": "1",
            "epsa_final_sufficient": True,
            "potential_false_sufficient_candidate": False,
            "potential_false_insufficient_candidate": False,
            "retrieval_failed": False,
            "final_answer_generation_failed": False,
            "epsa_failed": False,
        },
        {
            "exact_match": 0.0,
            "partial_match": 0.0,
            "answer_precision": 0.0,
            "answer_recall": 0.0,
            "answer_f1": 0.0,
            "selected_context_docs": 4,
            "selected_context_sentences": 3,
            "estimated_context_tokens": 400,
            "total_llm_tokens": 500,
            "latency_ms": 1500,
            "adaptive_stop_after_hop": "2",
            "epsa_final_sufficient": True,
            "potential_false_sufficient_candidate": True,
            "potential_false_insufficient_candidate": True,
            "retrieval_failed": False,
            "final_answer_generation_failed": False,
            "epsa_failed": False,
        },
    ]

    summary = summarize_epsa_rag_records(
        records,
        baseline_reference={
            "average_context_docs": 6.0,
            "average_estimated_context_tokens": 600.0,
            "average_total_llm_tokens": 800.0,
            "average_latency_ms": 2000.0,
            "exact_match": 0.7,
            "answer_f1": 0.8,
        },
    )

    assert summary["num_records"] == 2
    assert summary["exact_match"] == 0.5
    assert summary["answer_f1"] == 0.5
    assert summary["average_selected_context_docs"] == 3.0
    assert summary["average_selected_context_sentences"] == 2.5
    assert summary["average_estimated_context_tokens"] == 300.0
    assert summary["hop1_stop_count"] == 1
    assert summary["hop1_stop_rate"] == 0.5
    assert summary["hop2_used_count"] == 1
    assert summary["hop2_used_rate"] == 0.5
    assert summary["epsa_final_sufficient_count"] == 2
    assert summary["potential_false_sufficient_count"] == 1
    assert summary["potential_false_insufficient_count"] == 1
    assert summary["context_doc_reduction_percentage"] == 50.0
    assert summary["token_reduction_percentage"] == 50.0
    assert summary["total_llm_token_reduction_percentage"] == 50.0
    assert summary["latency_reduction_percentage"] == 37.5
    assert summary["exact_match_delta_vs_baseline"] == -0.2
    assert summary["answer_f1_delta_vs_baseline"] == -0.3


def test_summarize_epsa_rag_records_handles_empty_input() -> None:
    summary = summarize_epsa_rag_records([])

    assert summary["num_records"] == 0
    assert "baseline_reference" in summary
