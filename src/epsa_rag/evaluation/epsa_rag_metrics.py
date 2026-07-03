from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


BASELINE_REFERENCE: dict[str, float] = {
    "exact_match": 0.70,
    "partial_match": 0.79,
    "answer_precision": 0.7925,
    "answer_recall": 0.795,
    "answer_f1": 0.787,
    "average_context_docs": 13.54,
    "average_estimated_context_tokens": 2481.65,
    "average_total_llm_tokens": 4663.37,
    "average_latency_ms": 4403.7611,
}


def safe_mean(values: Sequence[Any]) -> float:
    """Return a stable numeric mean, treating missing/non-numeric values as absent."""

    numeric_values: list[float] = []

    for value in values:
        if value is None or value == "":
            continue

        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not numeric_values:
        return 0.0

    return round(sum(numeric_values) / len(numeric_values), 6)


def safe_rate(count: int, total: int) -> float:
    """Return count / total with zero-safe behavior."""

    if total <= 0:
        return 0.0

    return round(float(count) / float(total), 6)


def percentage_reduction(baseline_value: float | int | None, proposed_value: float | int | None) -> float:
    """Compute percentage reduction from baseline to proposed value.

    Positive means EPSA used less than the baseline. Negative means EPSA used more.
    """

    try:
        baseline = float(baseline_value or 0.0)
        proposed = float(proposed_value or 0.0)
    except (TypeError, ValueError):
        return 0.0

    if baseline <= 0.0:
        return 0.0

    return round(((baseline - proposed) / baseline) * 100.0, 6)


def summarize_epsa_rag_records(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_reference: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build aggregate EPSA RAG metrics from row-level result records."""

    baseline = dict(BASELINE_REFERENCE)
    baseline.update(dict(baseline_reference or {}))

    total = len(records)
    if total == 0:
        return {
            "num_records": 0,
            "baseline_reference": baseline,
        }

    average_context_docs = safe_mean([record.get("selected_context_docs") for record in records])
    average_estimated_context_tokens = safe_mean(
        [record.get("estimated_context_tokens") for record in records]
    )
    average_total_llm_tokens = safe_mean([record.get("total_llm_tokens") for record in records])
    average_latency_ms = safe_mean([record.get("latency_ms") for record in records])

    hop1_stop_count = _count_value(records, "adaptive_stop_after_hop", "1")
    hop2_used_count = _count_value(records, "adaptive_stop_after_hop", "2")
    no_query_count = _count_value(records, "adaptive_stop_after_hop", "1_no_query")
    final_sufficient_count = _count_truthy(records, "epsa_final_sufficient")
    potential_false_sufficient_count = _count_truthy(records, "potential_false_sufficient_candidate")
    potential_false_insufficient_count = _count_truthy(records, "potential_false_insufficient_candidate")

    gold_titles_available_count = _count_records(
        records,
        lambda record: _record_int(record, "gold_supporting_title_count") > 0,
    )
    gold_titles_all_retrieved_count = _count_records(
        records,
        lambda record: _has_all_gold_titles(record, "gold_titles_in_merged_count"),
    )
    gold_titles_partial_retrieved_count = _count_records(
        records,
        lambda record: _has_partial_gold_titles(record, "gold_titles_in_merged_count"),
    )
    gold_titles_not_retrieved_count = _count_records(
        records,
        lambda record: _has_no_gold_titles(record, "gold_titles_in_merged_count"),
    )
    gold_titles_all_selected_count = _count_records(
        records,
        lambda record: _has_all_gold_titles(record, "gold_titles_selected_by_epsa_count"),
    )
    gold_titles_partial_selected_count = _count_records(
        records,
        lambda record: _has_partial_gold_titles(record, "gold_titles_selected_by_epsa_count"),
    )
    gold_titles_not_selected_count = _count_records(
        records,
        lambda record: _has_no_gold_titles(record, "gold_titles_selected_by_epsa_count"),
    )
    gold_titles_all_in_final_context_count = _count_records(
        records,
        lambda record: _has_all_gold_titles(record, "gold_titles_in_final_context_count"),
    )

    summary: dict[str, Any] = {
        "num_records": total,
        "exact_match": safe_mean([record.get("exact_match") for record in records]),
        "partial_match": safe_mean([record.get("partial_match") for record in records]),
        "answer_precision": safe_mean([record.get("answer_precision") for record in records]),
        "answer_recall": safe_mean([record.get("answer_recall") for record in records]),
        "answer_f1": safe_mean([record.get("answer_f1") for record in records]),
        "average_selected_context_docs": average_context_docs,
        "average_selected_context_sentences": safe_mean(
            [record.get("selected_context_sentences") for record in records]
        ),
        "average_estimated_context_tokens": average_estimated_context_tokens,
        "average_total_llm_tokens": average_total_llm_tokens,
        "average_latency_ms": average_latency_ms,
        "hop1_stop_count": hop1_stop_count,
        "hop1_stop_rate": safe_rate(hop1_stop_count, total),
        "hop2_used_count": hop2_used_count,
        "hop2_used_rate": safe_rate(hop2_used_count, total),
        "no_query_count": no_query_count,
        "no_query_rate": safe_rate(no_query_count, total),
        "epsa_final_sufficient_count": final_sufficient_count,
        "epsa_final_sufficient_rate": safe_rate(final_sufficient_count, total),
        "potential_false_sufficient_count": potential_false_sufficient_count,
        "potential_false_sufficient_rate": safe_rate(potential_false_sufficient_count, total),
        "potential_false_insufficient_count": potential_false_insufficient_count,
        "potential_false_insufficient_rate": safe_rate(potential_false_insufficient_count, total),
        "gold_titles_available_count": gold_titles_available_count,
        "gold_titles_available_rate": safe_rate(gold_titles_available_count, total),
        "gold_titles_all_retrieved_count": gold_titles_all_retrieved_count,
        "gold_titles_all_retrieved_rate": safe_rate(gold_titles_all_retrieved_count, total),
        "gold_titles_partial_retrieved_count": gold_titles_partial_retrieved_count,
        "gold_titles_partial_retrieved_rate": safe_rate(gold_titles_partial_retrieved_count, total),
        "gold_titles_not_retrieved_count": gold_titles_not_retrieved_count,
        "gold_titles_not_retrieved_rate": safe_rate(gold_titles_not_retrieved_count, total),
        "gold_titles_all_selected_count": gold_titles_all_selected_count,
        "gold_titles_all_selected_rate": safe_rate(gold_titles_all_selected_count, total),
        "gold_titles_partial_selected_count": gold_titles_partial_selected_count,
        "gold_titles_partial_selected_rate": safe_rate(gold_titles_partial_selected_count, total),
        "gold_titles_not_selected_count": gold_titles_not_selected_count,
        "gold_titles_not_selected_rate": safe_rate(gold_titles_not_selected_count, total),
        "gold_titles_all_in_final_context_count": gold_titles_all_in_final_context_count,
        "gold_titles_all_in_final_context_rate": safe_rate(
            gold_titles_all_in_final_context_count,
            total,
        ),
        "coverage_status_gold_not_retrieved_count": _count_coverage_status(
            records,
            "gold_not_retrieved",
        ),
        "coverage_status_partial_gold_retrieved_count": _count_coverage_status(
            records,
            "partial_gold_retrieved",
        ),
        "coverage_status_all_gold_retrieved_not_selected_count": _count_coverage_status(
            records,
            "all_gold_retrieved_not_selected",
        ),
        "coverage_status_partial_gold_selected_count": _count_coverage_status(
            records,
            "partial_gold_selected",
        ),
        "coverage_status_all_gold_selected_count": _count_coverage_status(
            records,
            "all_gold_selected",
        ),
        "coverage_status_fallback_context_contains_gold_count": _count_coverage_status(
            records,
            "fallback_context_contains_gold",
        ),
        "false_sufficient_gold_titles_all_retrieved_count": _count_records(
            records,
            lambda record: _as_bool(record.get("potential_false_sufficient_candidate"))
            and _has_all_gold_titles(record, "gold_titles_in_merged_count"),
        ),
        "false_sufficient_gold_titles_not_fully_selected_count": _count_records(
            records,
            lambda record: _as_bool(record.get("potential_false_sufficient_candidate"))
            and _record_int(record, "gold_titles_selected_by_epsa_count")
            < _record_int(record, "gold_supporting_title_count"),
        ),
        "false_sufficient_gold_not_retrieved_count": _count_coverage_status(
            records,
            "gold_not_retrieved",
            only_potential_false_sufficient=True,
        ),
        "false_sufficient_partial_gold_retrieved_count": _count_coverage_status(
            records,
            "partial_gold_retrieved",
            only_potential_false_sufficient=True,
        ),
        "false_sufficient_all_gold_retrieved_not_selected_count": _count_coverage_status(
            records,
            "all_gold_retrieved_not_selected",
            only_potential_false_sufficient=True,
        ),
        "false_sufficient_partial_gold_selected_count": _count_coverage_status(
            records,
            "partial_gold_selected",
            only_potential_false_sufficient=True,
        ),
        "false_sufficient_all_gold_selected_count": _count_coverage_status(
            records,
            "all_gold_selected",
            only_potential_false_sufficient=True,
        ),
        "false_sufficient_fallback_context_contains_gold_count": _count_coverage_status(
            records,
            "fallback_context_contains_gold",
            only_potential_false_sufficient=True,
        ),
        "retrieval_failures": _count_truthy(records, "retrieval_failed"),
        "final_answer_generation_failures": _count_truthy(
            records,
            "final_answer_generation_failed",
        ),
        "epsa_failures": _count_truthy(records, "epsa_failed"),
        "baseline_reference": baseline,
        "context_doc_reduction_percentage": percentage_reduction(
            baseline.get("average_context_docs"),
            average_context_docs,
        ),
        "token_reduction_percentage": percentage_reduction(
            baseline.get("average_estimated_context_tokens"),
            average_estimated_context_tokens,
        ),
        "total_llm_token_reduction_percentage": percentage_reduction(
            baseline.get("average_total_llm_tokens"),
            average_total_llm_tokens,
        ),
        "latency_reduction_percentage": percentage_reduction(
            baseline.get("average_latency_ms"),
            average_latency_ms,
        ),
    }

    summary["exact_match_delta_vs_baseline"] = round(
        float(summary["exact_match"]) - float(baseline.get("exact_match", 0.0)),
        6,
    )
    summary["answer_f1_delta_vs_baseline"] = round(
        float(summary["answer_f1"]) - float(baseline.get("answer_f1", 0.0)),
        6,
    )

    return summary


def _count_truthy(records: Sequence[Mapping[str, Any]], field_name: str) -> int:
    return sum(1 for record in records if _as_bool(record.get(field_name)))


def _count_value(records: Sequence[Mapping[str, Any]], field_name: str, expected_value: str) -> int:
    return sum(1 for record in records if str(record.get(field_name) or "") == expected_value)


def _count_records(records: Sequence[Mapping[str, Any]], predicate: Any) -> int:
    return sum(1 for record in records if predicate(record))


def _count_coverage_status(
    records: Sequence[Mapping[str, Any]],
    expected_status: str,
    *,
    only_potential_false_sufficient: bool = False,
) -> int:
    return _count_records(
        records,
        lambda record: str(record.get("gold_title_coverage_status") or "") == expected_status
        and (
            not only_potential_false_sufficient
            or _as_bool(record.get("potential_false_sufficient_candidate"))
        ),
    )


def _has_all_gold_titles(record: Mapping[str, Any], field_name: str) -> bool:
    gold_count = _record_int(record, "gold_supporting_title_count")
    return gold_count > 0 and _record_int(record, field_name) >= gold_count


def _has_partial_gold_titles(record: Mapping[str, Any], field_name: str) -> bool:
    gold_count = _record_int(record, "gold_supporting_title_count")
    value = _record_int(record, field_name)
    return gold_count > 0 and 0 < value < gold_count


def _has_no_gold_titles(record: Mapping[str, Any], field_name: str) -> bool:
    gold_count = _record_int(record, "gold_supporting_title_count")
    return gold_count > 0 and _record_int(record, field_name) <= 0


def _record_int(record: Mapping[str, Any], field_name: str) -> int:
    value = record.get(field_name)

    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}

    return bool(value)
