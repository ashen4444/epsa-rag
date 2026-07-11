from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_epsa_oracle_context_diagnostics import (
    auto_select_target_rows,
    build_oracle_contexts,
    build_prepared_result,
    build_question_interpretations,
    reconstruct_pruned_context,
    scenario_summary,
)


def _row() -> dict[str, object]:
    return {
        "question_id": "q1",
        "question": "Where are Building A and Building B located?",
        "gold_answer": "New York City",
        "predicted_answer": "Insufficient evidence.",
        "exact_match": 0.0,
        "answer_f1": 0.0,
        "epsa_final_sufficient": True,
        "context_source": "epsa_pruned_context",
        "selected_context_docs": 1,
        "selected_chunk_ids": json.dumps(["noise"]),
        "selected_evidence_unit_ids": json.dumps(["noise::s0"]),
        "merged_retrieved_chunk_ids": json.dumps(["gold_a", "gold_b", "noise"]),
        "gold_supporting_title_count": 2,
        "gold_titles_in_merged_count": 2,
        "gold_titles_selected_by_epsa_count": 0,
        "gold_titles_in_merged": json.dumps(["Building A", "Building B"]),
    }


def _corpus() -> dict[str, dict[str, object]]:
    return {
        "gold_a": {
            "chunk_id": "gold_a",
            "doc_title": "Building A",
            "chunk_text": "Title: Building A\nParagraph: Building A is in New York City.",
            "sentences": [
                {
                    "sentence_id": 0,
                    "text": "Building A is in New York City.",
                }
            ],
        },
        "gold_b": {
            "chunk_id": "gold_b",
            "doc_title": "Building B",
            "chunk_text": "Title: Building B\nParagraph: Building B is in New York City.",
            "sentences": [
                {
                    "sentence_id": 0,
                    "text": "Building B is in New York City.",
                }
            ],
        },
        "noise": {
            "chunk_id": "noise",
            "doc_title": "Noise",
            "chunk_text": "Title: Noise\nParagraph: This is unrelated.",
            "sentences": [
                {
                    "sentence_id": 0,
                    "text": "This is unrelated.",
                }
            ],
        },
    }


def test_auto_select_target_rows_uses_verified_false_sufficient_pattern() -> None:
    positive = _row()
    negative = dict(_row())
    negative["question_id"] = "q2"
    negative["gold_titles_selected_by_epsa_count"] = 1

    selected = auto_select_target_rows(pd.DataFrame([positive, negative]))

    assert selected["question_id"].tolist() == ["q1"]


def test_reconstruct_pruned_context_matches_pruner_provenance_format() -> None:
    context, chunk_ids = reconstruct_pruned_context(["noise::s0"], _corpus())

    assert chunk_ids == ["noise"]
    assert (
        context
        == "[Title: Noise | Chunk: noise | Sentence: 0]\nThis is unrelated."
    )


def test_build_oracle_contexts_creates_all_four_bounded_scenarios() -> None:
    contexts = build_oracle_contexts(_row(), _corpus())

    assert set(contexts) == {
        "A_current_epsa_context",
        "B_full_merged_context",
        "C_current_plus_omitted_gold",
        "D_gold_documents_only",
    }

    assert "This is unrelated." in contexts["A_current_epsa_context"].context_text
    assert "Building A is in New York City." in contexts[
        "B_full_merged_context"
    ].context_text
    assert "Oracle-added omitted gold supporting documents" in contexts[
        "C_current_plus_omitted_gold"
    ].context_text
    assert "This is unrelated." not in contexts[
        "D_gold_documents_only"
    ].context_text
    assert contexts["D_gold_documents_only"].chunk_ids == ["gold_a", "gold_b"]


def test_prepare_only_summary_is_not_misclassified_as_failed_or_unrecoverable() -> None:
    row = _row()
    contexts = build_oracle_contexts(row, _corpus())
    results = [
        build_prepared_result(
            row=row,
            context=contexts[scenario],
            repeat_index=1,
        )
        for scenario in contexts
    ]

    summaries = scenario_summary(results)
    assert all(item["failed_runs"] == 0 for item in summaries)
    assert all(item["prepared_runs"] == 1 for item in summaries)

    interpretations = build_question_interpretations(summaries)
    assert interpretations[0]["primary_interpretation"] == "oracle_generation_not_run"
    assert interpretations[0]["recoverable"] is False
