from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_EXAMPLE_FIELDS: tuple[str, ...] = (
    "question_id",
    "question",
    "gold_answer",
    "predicted_answer",
    "exact_match",
    "partial_match",
    "answer_f1",
    "adaptive_stop_after_hop",
    "epsa_hop1_sufficient",
    "epsa_final_sufficient",
    "selected_context_docs",
    "selected_context_sentences",
    "estimated_context_tokens",
    "context_source",
    "sufficiency_confidence",
    "decision_reason",
    "missing_evidence",
    "answer_candidate",
    "answer_type",
    "next_hop_query",
    "next_hop_query_type",
    "selected_chunk_ids",
    "selected_evidence_unit_ids",
)


def read_epsa_rag_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read an EPSA RAG result CSV into plain dictionaries."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"EPSA RAG CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"EPSA RAG CSV path is not a file: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def analyze_epsa_failure_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Create deterministic failure-analysis aggregates from EPSA row records.

    The analysis is intentionally independent from OpenAI, FAISS, BM25, or live
    retrieval. It only uses fields already logged by ``scripts/run_epsa_rag.py``.
    """

    rows = [dict(record) for record in records]
    total = len(rows)

    sufficient_rows = [row for row in rows if _as_bool(row.get("epsa_final_sufficient"))]
    insufficient_rows = [row for row in rows if not _as_bool(row.get("epsa_final_sufficient"))]
    false_sufficient_rows = [row for row in rows if is_false_sufficient_candidate(row)]
    false_insufficient_rows = [row for row in rows if _as_bool(row.get("potential_false_insufficient_candidate"))]
    wrong_rows = [row for row in rows if is_wrong_answer(row)]
    correct_rows = [row for row in rows if not is_wrong_answer(row)]
    insufficient_pruned_rows = [
        row
        for row in insufficient_rows
        if str(row.get("context_source") or "") == "epsa_pruned_context"
        and _as_int(row.get("selected_context_sentences")) > 0
    ]

    analysis: dict[str, Any] = {
        "num_records": total,
        "answer_quality": {
            "exact_match": _mean(row.get("exact_match") for row in rows),
            "partial_match": _mean(row.get("partial_match") for row in rows),
            "answer_f1": _mean(row.get("answer_f1") for row in rows),
            "correct_or_partial_count": len(correct_rows),
            "wrong_count": len(wrong_rows),
            "wrong_rate": _rate(len(wrong_rows), total),
        },
        "sufficiency": {
            "epsa_final_sufficient_count": len(sufficient_rows),
            "epsa_final_sufficient_rate": _rate(len(sufficient_rows), total),
            "epsa_final_insufficient_count": len(insufficient_rows),
            "epsa_final_insufficient_rate": _rate(len(insufficient_rows), total),
            "potential_false_sufficient_count": len(false_sufficient_rows),
            "potential_false_sufficient_rate": _rate(len(false_sufficient_rows), total),
            "potential_false_sufficient_among_sufficient_rate": _rate(
                len(false_sufficient_rows),
                len(sufficient_rows),
            ),
            "potential_false_insufficient_count": len(false_insufficient_rows),
            "potential_false_insufficient_rate": _rate(len(false_insufficient_rows), total),
        },
        "context": {
            "average_selected_context_docs": _mean(row.get("selected_context_docs") for row in rows),
            "average_selected_context_sentences": _mean(
                row.get("selected_context_sentences") for row in rows
            ),
            "average_estimated_context_tokens": _mean(row.get("estimated_context_tokens") for row in rows),
            "one_doc_cases": _count_where(rows, lambda row: _as_int(row.get("selected_context_docs")) == 1),
            "one_sentence_cases": _count_where(
                rows,
                lambda row: _as_int(row.get("selected_context_sentences")) == 1,
            ),
            "one_sentence_false_sufficient_count": _count_where(
                false_sufficient_rows,
                lambda row: _as_int(row.get("selected_context_sentences")) == 1,
            ),
            "insufficient_pruned_context_count": len(insufficient_pruned_rows),
            "insufficient_pruned_context_rate": _rate(len(insufficient_pruned_rows), total),
            "insufficient_pruned_context_among_insufficient_rate": _rate(
                len(insufficient_pruned_rows),
                len(insufficient_rows),
            ),
        },
        "grouped_counts": {
            "cases_by_adaptive_stop": _group_summary(rows, "adaptive_stop_after_hop"),
            "cases_by_final_sufficiency": _group_summary(rows, "epsa_final_sufficient"),
            "cases_by_context_source": _group_summary(rows, "context_source"),
            "cases_by_selected_context_docs": _group_summary(rows, "selected_context_docs"),
            "cases_by_selected_context_sentences": _group_summary(
                rows,
                "selected_context_sentences",
            ),
            "cases_by_decision_family": _group_by_decision_family(rows),
        },
        "failure_pattern_counts": {
            "hop1_stop_wrong_cases": _count_where(
                rows,
                lambda row: _stop(row) == "1" and is_wrong_answer(row),
            ),
            "hop2_used_wrong_cases": _count_where(
                rows,
                lambda row: _stop(row) == "2" and is_wrong_answer(row),
            ),
            "wrong_with_one_sentence": _count_where(
                rows,
                lambda row: is_wrong_answer(row)
                and _as_int(row.get("selected_context_sentences")) == 1,
            ),
            "wrong_but_epsa_final_sufficient": len(false_sufficient_rows),
            "wrong_and_epsa_insufficient": _count_where(
                rows,
                lambda row: is_wrong_answer(row)
                and not _as_bool(row.get("epsa_final_sufficient")),
            ),
            "correct_with_small_context": _count_where(
                rows,
                lambda row: not is_wrong_answer(row)
                and _as_int(row.get("selected_context_docs")) <= 2
                and _as_int(row.get("selected_context_sentences")) <= 2,
            ),
            "factoid_false_sufficient_cases": _count_where(
                false_sufficient_rows,
                lambda row: infer_decision_family(row) == "factoid_sufficient",
            ),
            "bridge_false_sufficient_cases": _count_where(
                false_sufficient_rows,
                lambda row: infer_decision_family(row) == "bridge_sufficient",
            ),
        },
        "examples": {
            "false_sufficient_cases": _examples(false_sufficient_rows, max_examples=max_examples),
            "hop1_stop_wrong_cases": _examples(
                [row for row in rows if _stop(row) == "1" and is_wrong_answer(row)],
                max_examples=max_examples,
            ),
            "hop2_used_wrong_cases": _examples(
                [row for row in rows if _stop(row) == "2" and is_wrong_answer(row)],
                max_examples=max_examples,
            ),
            "correct_with_small_context": _examples(
                [
                    row
                    for row in rows
                    if not is_wrong_answer(row)
                    and _as_int(row.get("selected_context_docs")) <= 2
                    and _as_int(row.get("selected_context_sentences")) <= 2
                ],
                max_examples=max_examples,
            ),
            "wrong_with_one_sentence": _examples(
                [
                    row
                    for row in rows
                    if is_wrong_answer(row)
                    and _as_int(row.get("selected_context_sentences")) == 1
                ],
                max_examples=max_examples,
            ),
            "wrong_but_epsa_final_sufficient": _examples(
                false_sufficient_rows,
                max_examples=max_examples,
            ),
            "wrong_and_epsa_insufficient": _examples(
                [
                    row
                    for row in rows
                    if is_wrong_answer(row)
                    and not _as_bool(row.get("epsa_final_sufficient"))
                ],
                max_examples=max_examples,
            ),
            "insufficient_pruned_context_cases": _examples(
                insufficient_pruned_rows,
                max_examples=max_examples,
            ),
        },
    }

    analysis["recommended_next_checks"] = _recommended_next_checks(analysis)
    return analysis


def write_failure_analysis_json(report: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def build_failure_analysis_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown report for manual EPSA debugging."""

    lines: list[str] = [
        "# EPSA Failure Analysis",
        "",
        "## Core Counts",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for section_name in ("answer_quality", "sufficiency", "context"):
        section = report.get(section_name, {})
        if isinstance(section, Mapping):
            for key, value in section.items():
                lines.append(f"| `{section_name}.{key}` | {_format_value(value)} |")

    lines.extend(["", "## Failure Pattern Counts", "", "| Pattern | Count |", "|---|---:|"])
    pattern_counts = report.get("failure_pattern_counts", {})
    if isinstance(pattern_counts, Mapping):
        for key, value in pattern_counts.items():
            lines.append(f"| `{key}` | {_format_value(value)} |")

    lines.extend(["", "## Grouped Counts"])
    grouped_counts = report.get("grouped_counts", {})
    if isinstance(grouped_counts, Mapping):
        for group_name, group_data in grouped_counts.items():
            lines.extend(["", f"### `{group_name}`", "", "| Group | Count | Wrong | False sufficient | Avg docs | Avg sentences |", "|---|---:|---:|---:|---:|---:|"])
            if isinstance(group_data, Mapping):
                for key, stats in group_data.items():
                    if not isinstance(stats, Mapping):
                        continue
                    lines.append(
                        "| "
                        f"`{key}` | "
                        f"{stats.get('count', 0)} | "
                        f"{stats.get('wrong_count', 0)} | "
                        f"{stats.get('false_sufficient_count', 0)} | "
                        f"{_format_value(stats.get('average_selected_context_docs', 0.0))} | "
                        f"{_format_value(stats.get('average_selected_context_sentences', 0.0))} |"
                    )

    lines.extend(["", "## Recommended Next Checks", ""])
    for item in report.get("recommended_next_checks", []) or []:
        lines.append(f"- {item}")

    examples = report.get("examples", {})
    if isinstance(examples, Mapping):
        lines.extend(["", "## Example Cases", ""])
        for example_name, example_rows in examples.items():
            lines.extend([f"### `{example_name}`", ""])
            if not example_rows:
                lines.extend(["No examples.", ""])
                continue
            for row in example_rows:
                if not isinstance(row, Mapping):
                    continue
                question = str(row.get("question") or "").replace("\n", " ")
                lines.extend(
                    [
                        f"- `{row.get('question_id', '')}`",
                        f"  - Question: {question}",
                        f"  - Gold: `{row.get('gold_answer', '')}`",
                        f"  - Predicted: `{row.get('predicted_answer', '')}`",
                        f"  - Stop: `{row.get('adaptive_stop_after_hop', '')}`, sufficient: `{row.get('epsa_final_sufficient', '')}`, docs/sentences: `{row.get('selected_context_docs', '')}/{row.get('selected_context_sentences', '')}`",
                        f"  - Reason: {row.get('decision_reason', '')}",
                    ]
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_failure_analysis_markdown(report: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_failure_analysis_markdown(report), encoding="utf-8")


def is_false_sufficient_candidate(record: Mapping[str, Any]) -> bool:
    if _as_bool(record.get("potential_false_sufficient_candidate")):
        return True
    return _as_bool(record.get("epsa_final_sufficient")) and is_wrong_answer(record)


def is_wrong_answer(record: Mapping[str, Any]) -> bool:
    return _as_float(record.get("exact_match")) == 0.0 and _as_float(record.get("partial_match")) == 0.0


def infer_decision_family(record: Mapping[str, Any]) -> str:
    reason = str(record.get("decision_reason") or "").strip().casefold()
    if reason.startswith("factoid path connects"):
        return "factoid_sufficient"
    if reason.startswith("complete bridge evidence path"):
        return "bridge_sufficient"
    if reason.startswith("connected yes/no evidence path"):
        return "yes_no_sufficient"
    if reason.startswith("comparison requires"):
        return "comparison_insufficient"
    if reason.startswith("no candidate bridge"):
        return "bridge_insufficient"
    if reason.startswith("no candidate factoid"):
        return "factoid_insufficient"
    if reason.startswith("no yes/no"):
        return "yes_no_insufficient"
    if not reason:
        return "missing_decision_reason"
    return "other"


def _recommended_next_checks(report: Mapping[str, Any]) -> list[str]:
    recommendations: list[str] = []
    context = report.get("context", {}) if isinstance(report.get("context"), Mapping) else {}
    sufficiency = report.get("sufficiency", {}) if isinstance(report.get("sufficiency"), Mapping) else {}
    patterns = report.get("failure_pattern_counts", {}) if isinstance(report.get("failure_pattern_counts"), Mapping) else {}

    false_among_sufficient = _as_float(
        sufficiency.get("potential_false_sufficient_among_sufficient_rate")
    )
    insufficient_pruned_rate = _as_float(
        context.get("insufficient_pruned_context_among_insufficient_rate")
    )
    one_sentence_false = _as_int(context.get("one_sentence_false_sufficient_count"))
    factoid_false = _as_int(patterns.get("factoid_false_sufficient_cases"))

    if false_among_sufficient >= 0.25:
        recommendations.append(
            "Calibrate SufficiencyDecisionEngine conservatively; many sufficient decisions are wrong."
        )
    if insufficient_pruned_rate >= 0.5:
        recommendations.append(
            "Change the EPSA RAG runner fallback policy so final-insufficient EPSA results do not send only partial pruned evidence to the final LLM."
        )
    if one_sentence_false > 0:
        recommendations.append(
            "Inspect one-sentence false-sufficient cases; over-pruning is likely removing required supporting evidence."
        )
    if factoid_false > 0:
        recommendations.append(
            "Inspect factoid false-sufficient cases; generic entity/UNKNOWN answer compatibility is likely too permissive."
        )
    if not recommendations:
        recommendations.append("No dominant deterministic failure pattern was found in the logged fields.")
    return recommendations


def _group_summary(records: Sequence[Mapping[str, Any]], field_name: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get(field_name) if record.get(field_name) not in (None, "") else "<missing>")
        groups[key].append(record)

    return {
        key: _basic_group_stats(group_rows)
        for key, group_rows in sorted(groups.items(), key=lambda item: _natural_group_key(item[0]))
    }


def _group_by_decision_family(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[infer_decision_family(record)].append(record)
    return {key: _basic_group_stats(value) for key, value in sorted(groups.items())}


def _basic_group_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "wrong_count": _count_where(records, is_wrong_answer),
        "false_sufficient_count": _count_where(records, is_false_sufficient_candidate),
        "exact_match": _mean(row.get("exact_match") for row in records),
        "answer_f1": _mean(row.get("answer_f1") for row in records),
        "average_selected_context_docs": _mean(row.get("selected_context_docs") for row in records),
        "average_selected_context_sentences": _mean(row.get("selected_context_sentences") for row in records),
        "average_sufficiency_confidence": _mean(row.get("sufficiency_confidence") for row in records),
    }


def _examples(
    records: Sequence[Mapping[str, Any]],
    *,
    max_examples: int,
    fields: Sequence[str] = DEFAULT_EXAMPLE_FIELDS,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records[: max(0, max_examples)]:
        selected.append({field: _clean_scalar(record.get(field)) for field in fields if field in record})
    return selected


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    if text == "":
        return ""
    lowered = text.strip().casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." not in text:
            return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _mean(values: Iterable[Any]) -> float:
    numeric_values = [_as_float(value) for value in values if value not in (None, "")]
    if not numeric_values:
        return 0.0
    return round(sum(numeric_values) / len(numeric_values), 6)


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(float(count) / float(total), 6)


def _count_where(records: Sequence[Mapping[str, Any]], predicate: Any) -> int:
    return sum(1 for record in records if predicate(record))


def _stop(record: Mapping[str, Any]) -> str:
    value = record.get("adaptive_stop_after_hop")
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return False


def _as_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _natural_group_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


__all__ = [
    "analyze_epsa_failure_records",
    "build_failure_analysis_markdown",
    "infer_decision_family",
    "is_false_sufficient_candidate",
    "is_wrong_answer",
    "read_epsa_rag_csv",
    "write_failure_analysis_json",
    "write_failure_analysis_markdown",
]
