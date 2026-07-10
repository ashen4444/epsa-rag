from __future__ import annotations

from epsa_rag.evaluation.answer_metrics import (
    answer_overlap_metrics,
    exact_match_score,
    partial_match_score,
    relaxed_answer_match,
)

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
            "relaxed_answer_correct": True,
            "relaxed_match_type": "exact",
            "strict_vs_relaxed_disagreement": False,
            "selected_context_docs": 2,
            "selected_context_sentences": 2,
            "estimated_context_tokens": 200,
            "total_llm_tokens": 300,
            "latency_ms": 1000,
            "adaptive_stop_after_hop": "1",
            "epsa_final_sufficient": True,
            "potential_false_sufficient_candidate": False,
            "potential_false_sufficient_relaxed": False,
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
            "relaxed_answer_correct": False,
            "relaxed_match_type": "no_match",
            "strict_vs_relaxed_disagreement": False,
            "selected_context_docs": 4,
            "selected_context_sentences": 3,
            "estimated_context_tokens": 400,
            "total_llm_tokens": 500,
            "latency_ms": 1500,
            "adaptive_stop_after_hop": "2",
            "epsa_final_sufficient": True,
            "potential_false_sufficient_candidate": True,
            "potential_false_sufficient_relaxed": True,
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
    assert summary["relaxed_answer_correct_count"] == 1
    assert summary["relaxed_answer_correct_rate"] == 0.5
    assert summary["strict_vs_relaxed_disagreement_count"] == 0
    assert summary["potential_false_sufficient_relaxed_count"] == 1
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


def test_relaxed_match_accepts_safe_profession_modifier_without_changing_exact_match() -> None:
    result = relaxed_answer_match("film director", "director")

    assert exact_match_score("film director", "director") == 0.0
    assert partial_match_score("film director", "director") == 1.0
    assert result.correct is True
    assert result.match_type == "profession_modifier"


def test_relaxed_match_accepts_party_suffix_and_rejects_honorific_only_mismatch() -> None:
    party_result = relaxed_answer_match("Conservative Party", "Conservative")

    assert party_result.correct is True
    assert party_result.match_type == "party_modifier"
    assert relaxed_answer_match("Dr", "Mr").correct is False


def test_relaxed_match_rejects_role_changing_modifier() -> None:
    result = relaxed_answer_match("assistant director", "director")

    assert result.correct is False
    assert result.match_type == "no_match"


def test_relaxed_match_accepts_multi_token_answer_phrase_at_response_end() -> None:
    result = relaxed_answer_match(
        "Peter O'Meara portrayed Norman Dike in \"Band of Brothers\".",
        "Band of Brothers",
    )

    assert result.correct is True
    assert result.match_type == "answer_phrase"


def test_relaxed_match_rejects_unrelated_answer_and_ambiguous_substring() -> None:
    assert relaxed_answer_match("California", "Hawaii").correct is False
    assert relaxed_answer_match("New York City", "York").correct is False


def test_existing_overlap_metrics_remain_unchanged() -> None:
    overlap = answer_overlap_metrics("film director", "director")

    assert exact_match_score("film director", "director") == 0.0
    assert partial_match_score("film director", "director") == 1.0
    assert overlap.precision == 0.5
    assert overlap.recall == 1.0
    assert overlap.f1 == 2 / 3


def test_summary_aggregates_relaxed_disagreements_and_relaxed_false_sufficient() -> None:
    records = [
        {
            "gold_answer": "director",
            "predicted_answer": "film director",
            "exact_match": 0.0,
            "partial_match": 1.0,
            "answer_precision": 0.5,
            "answer_recall": 1.0,
            "answer_f1": 2 / 3,
            "relaxed_answer_correct": True,
            "relaxed_match_type": "profession_modifier",
            "strict_vs_relaxed_disagreement": True,
            "epsa_final_sufficient": True,
            "potential_false_sufficient_candidate": False,
            "potential_false_sufficient_relaxed": False,
        },
        {
            "gold_answer": "Hawaii",
            "predicted_answer": "California",
            "exact_match": 0.0,
            "partial_match": 0.0,
            "answer_precision": 0.0,
            "answer_recall": 0.0,
            "answer_f1": 0.0,
            "relaxed_answer_correct": False,
            "relaxed_match_type": "no_match",
            "strict_vs_relaxed_disagreement": False,
            "epsa_final_sufficient": True,
            "potential_false_sufficient_candidate": True,
            "potential_false_sufficient_relaxed": True,
        },
    ]

    summary = summarize_epsa_rag_records(records)

    assert summary["exact_match"] == 0.0
    assert summary["relaxed_answer_correct_count"] == 1
    assert summary["relaxed_answer_correct_rate"] == 0.5
    assert summary["strict_vs_relaxed_disagreement_count"] == 1
    assert summary["strict_vs_relaxed_disagreement_rate"] == 0.5
    assert summary["potential_false_sufficient_count"] == 1
    assert summary["potential_false_sufficient_relaxed_count"] == 1
