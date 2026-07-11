from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SCENARIO_ORDER = (
    "A_current_epsa_context",
    "B_full_merged_context",
    "C_current_plus_omitted_gold",
    "D_gold_documents_only",
)


@dataclass(frozen=True)
class OracleContext:
    scenario: str
    description: str
    context_text: str
    chunk_ids: list[str]
    gold_chunk_ids: list[str]
    note: str = ""


@dataclass(frozen=True)
class RuntimeComponents:
    llm_client_factory: Callable[..., Any]
    build_final_answer_messages: Callable[[str, str], Sequence[Any]]
    exact_match_score: Callable[[str | None, str | None], float]
    partial_match_score: Callable[[str | None, str | None], float]
    answer_overlap_metrics: Callable[[str | None, str | None], Any]
    relaxed_answer_match: Callable[[str | None, str | None], Any]


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, list):
        return [str(item) for item in value if item is not None]

    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected a JSON list but received: {text[:200]!r}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list but received {type(parsed).__name__}.")

    return [str(item) for item in parsed if item is not None]


def normalize_title(value: Any) -> str:
    return " ".join(str(value or "").casefold().strip().split())


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().casefold() in {"true", "1", "yes"}


def safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, *, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def load_epsa_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"EPSA results CSV not found: {path}")

    frame = pd.read_csv(path)
    required = {
        "question_id",
        "question",
        "gold_answer",
        "predicted_answer",
        "exact_match",
        "answer_f1",
        "epsa_final_sufficient",
        "context_source",
        "selected_context_docs",
        "selected_chunk_ids",
        "selected_evidence_unit_ids",
        "merged_retrieved_chunk_ids",
        "gold_supporting_title_count",
        "gold_titles_in_merged_count",
        "gold_titles_selected_by_epsa_count",
        "gold_titles_in_merged",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "EPSA results CSV is missing required columns: " + ", ".join(missing)
        )

    if frame["question_id"].astype(str).duplicated().any():
        duplicates = (
            frame.loc[frame["question_id"].astype(str).duplicated(), "question_id"]
            .astype(str)
            .tolist()
        )
        raise ValueError(f"Duplicate question_id values in EPSA CSV: {duplicates}")

    return frame


def load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus JSONL not found: {path}")

    corpus: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue

            raw = json.loads(line)
            chunk_id = str(raw.get("chunk_id") or raw.get("id") or "").strip()
            if not chunk_id:
                raise ValueError(f"Corpus line {line_number} has no chunk_id.")
            if chunk_id in corpus:
                raise ValueError(f"Duplicate corpus chunk_id: {chunk_id}")
            corpus[chunk_id] = raw

    if not corpus:
        raise ValueError(f"No chunks loaded from corpus: {path}")

    return corpus


def auto_select_target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the verified false-sufficient cluster without baseline labels.

    The rule uses only diagnostics already recorded in the EPSA result CSV:
    EPSA was sufficient, the strict answer was completely wrong, all gold
    supporting titles were retrieved, and EPSA selected none of those titles.
    """

    sufficient = frame["epsa_final_sufficient"].map(truthy)
    exact_wrong = pd.to_numeric(frame["exact_match"], errors="coerce").fillna(0.0) == 0.0
    f1_wrong = pd.to_numeric(frame["answer_f1"], errors="coerce").fillna(0.0) == 0.0
    gold_count = pd.to_numeric(
        frame["gold_supporting_title_count"], errors="coerce"
    ).fillna(0)
    merged_gold_count = pd.to_numeric(
        frame["gold_titles_in_merged_count"], errors="coerce"
    ).fillna(0)
    selected_gold_count = pd.to_numeric(
        frame["gold_titles_selected_by_epsa_count"], errors="coerce"
    ).fillna(0)

    mask = (
        sufficient
        & exact_wrong
        & f1_wrong
        & (gold_count > 0)
        & (merged_gold_count >= gold_count)
        & (selected_gold_count == 0)
    )

    return frame.loc[mask].copy()


def select_rows(
    frame: pd.DataFrame,
    *,
    question_ids: Sequence[str] | None,
) -> pd.DataFrame:
    if not question_ids:
        selected = auto_select_target_rows(frame)
        if selected.empty:
            raise ValueError(
                "Automatic target selection found no rows. "
                "Pass one or more --question-id values explicitly."
            )
        return selected

    requested = [str(value) for value in question_ids]
    indexed = frame.assign(_qid=frame["question_id"].astype(str)).set_index("_qid")
    missing = [question_id for question_id in requested if question_id not in indexed.index]
    if missing:
        raise ValueError(f"Question IDs not found in EPSA CSV: {missing}")

    return indexed.loc[requested].reset_index(drop=True)


def chunk_title(chunk: Mapping[str, Any]) -> str:
    return str(chunk.get("doc_title") or chunk.get("title") or "")


def chunk_text(chunk: Mapping[str, Any]) -> str:
    value = chunk.get("chunk_text") or chunk.get("text")
    if value:
        return str(value)

    paragraph = chunk.get("paragraph_text") or chunk.get("paragraph") or ""
    title = chunk_title(chunk)
    if title:
        return f"Title: {title}\nParagraph: {paragraph}"
    return str(paragraph)


def format_documents_for_prompt(
    chunk_ids: Sequence[str],
    corpus: Mapping[str, Mapping[str, Any]],
) -> str:
    """Match the current shared baseline document formatting."""

    blocks: list[str] = []

    for index, chunk_id in enumerate(chunk_ids, start=1):
        if chunk_id not in corpus:
            raise KeyError(f"Chunk ID from result CSV is missing from corpus: {chunk_id}")

        chunk = corpus[chunk_id]
        title = chunk_title(chunk)
        title_line = f"Title: {title}" if title else "Title: "
        text = chunk_text(chunk).strip()

        blocks.append(
            f"[Document {index}]\n"
            f"Chunk ID: {chunk_id}\n"
            f"{title_line}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(blocks)


def sentence_text_for_unit(
    evidence_unit_id: str,
    corpus: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, int, str]:
    marker = "::s"
    if marker not in evidence_unit_id:
        raise ValueError(f"Unrecognized evidence unit ID: {evidence_unit_id}")

    chunk_id, sentence_part = evidence_unit_id.rsplit(marker, 1)
    try:
        sentence_id = int(sentence_part)
    except ValueError as exc:
        raise ValueError(f"Invalid sentence ID in {evidence_unit_id}") from exc

    if chunk_id not in corpus:
        raise KeyError(f"Evidence unit chunk is missing from corpus: {chunk_id}")

    chunk = corpus[chunk_id]
    sentences = chunk.get("sentences") or []

    for sentence in sentences:
        if safe_int(sentence.get("sentence_id"), default=-1) == sentence_id:
            text = str(sentence.get("text") or "").strip()
            return chunk_id, chunk_title(chunk), sentence_id, text

    raise KeyError(
        f"Sentence {sentence_id} from evidence unit {evidence_unit_id} "
        f"was not found in the corpus."
    )


def reconstruct_pruned_context(
    evidence_unit_ids: Sequence[str],
    corpus: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str]]:
    """Reconstruct the current pruner's provenance-rich sentence format.

    The result CSV stores evidence IDs rather than the full selected context.
    The reconstruction therefore uses the original corpus sentence text. For
    the verified Chat 24 cluster, the selected rows do not depend on pronoun
    rewriting, so this reproduces the relevant content faithfully.
    """

    blocks: list[str] = []
    chunk_ids: list[str] = []

    for evidence_unit_id in evidence_unit_ids:
        chunk_id, title, sentence_id, sentence_text = sentence_text_for_unit(
            evidence_unit_id,
            corpus,
        )
        blocks.append(
            f"[Title: {title} | Chunk: {chunk_id} | Sentence: {sentence_id}]\n"
            f"{sentence_text}"
        )
        if chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)

    return "\n\n".join(blocks), chunk_ids


def current_epsa_context(
    row: Mapping[str, Any],
    corpus: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[str], str]:
    context_source = str(row.get("context_source") or "")

    if context_source == "epsa_pruned_context":
        evidence_ids = parse_json_list(row.get("selected_evidence_unit_ids"))
        context, chunk_ids = reconstruct_pruned_context(evidence_ids, corpus)
        return context, chunk_ids, "Reconstructed from selected evidence-unit IDs."

    merged_ids = parse_json_list(row.get("merged_retrieved_chunk_ids"))
    selected_doc_count = safe_int(row.get("selected_context_docs"), default=0)
    if selected_doc_count <= 0:
        selected_doc_count = len(merged_ids)

    selected_ids = merged_ids[:selected_doc_count]
    return (
        format_documents_for_prompt(selected_ids, corpus),
        selected_ids,
        "Reconstructed from the bounded prefix of merged retrieved documents.",
    )


def gold_chunk_ids_from_row(
    row: Mapping[str, Any],
    corpus: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    merged_ids = parse_json_list(row.get("merged_retrieved_chunk_ids"))
    gold_titles = parse_json_list(row.get("gold_titles_in_merged"))
    normalized_gold = {normalize_title(title) for title in gold_titles}
    normalized_gold.discard("")

    gold_chunk_ids: list[str] = []
    for chunk_id in merged_ids:
        if chunk_id not in corpus:
            raise KeyError(f"Merged chunk is missing from corpus: {chunk_id}")

        if normalize_title(chunk_title(corpus[chunk_id])) in normalized_gold:
            gold_chunk_ids.append(chunk_id)

    expected = safe_int(row.get("gold_titles_in_merged_count"), default=0)
    matched_titles = {
        normalize_title(chunk_title(corpus[chunk_id]))
        for chunk_id in gold_chunk_ids
    }
    if expected > 0 and len(matched_titles) < expected:
        raise ValueError(
            f"Question {row.get('question_id')} expected {expected} merged gold titles "
            f"but only matched {len(matched_titles)} from the corpus."
        )

    return gold_chunk_ids


def build_oracle_contexts(
    row: Mapping[str, Any],
    corpus: Mapping[str, Mapping[str, Any]],
) -> dict[str, OracleContext]:
    merged_ids = parse_json_list(row.get("merged_retrieved_chunk_ids"))
    current_text, current_chunk_ids, current_note = current_epsa_context(row, corpus)
    gold_chunk_ids = gold_chunk_ids_from_row(row, corpus)

    omitted_gold_ids = [
        chunk_id for chunk_id in gold_chunk_ids if chunk_id not in set(current_chunk_ids)
    ]

    if omitted_gold_ids:
        added_gold_context = format_documents_for_prompt(omitted_gold_ids, corpus)
        current_plus_gold = (
            f"{current_text}\n\n"
            "[Oracle-added omitted gold supporting documents]\n\n"
            f"{added_gold_context}"
        ).strip()
    else:
        current_plus_gold = current_text

    contexts = {
        "A_current_epsa_context": OracleContext(
            scenario="A_current_epsa_context",
            description="Current EPSA final context reconstructed from logged provenance.",
            context_text=current_text,
            chunk_ids=current_chunk_ids,
            gold_chunk_ids=gold_chunk_ids,
            note=current_note,
        ),
        "B_full_merged_context": OracleContext(
            scenario="B_full_merged_context",
            description="All unique Hop-1 and Hop-2 candidates in logged merged order.",
            context_text=format_documents_for_prompt(merged_ids, corpus),
            chunk_ids=merged_ids,
            gold_chunk_ids=gold_chunk_ids,
        ),
        "C_current_plus_omitted_gold": OracleContext(
            scenario="C_current_plus_omitted_gold",
            description="Current EPSA context plus full omitted gold supporting documents.",
            context_text=current_plus_gold,
            chunk_ids=current_chunk_ids + omitted_gold_ids,
            gold_chunk_ids=gold_chunk_ids,
            note=(
                f"Added {len(omitted_gold_ids)} omitted gold document(s)."
                if omitted_gold_ids
                else "No omitted gold documents were available to add."
            ),
        ),
        "D_gold_documents_only": OracleContext(
            scenario="D_gold_documents_only",
            description="Only the gold supporting documents already present in merged retrieval.",
            context_text=format_documents_for_prompt(gold_chunk_ids, corpus),
            chunk_ids=gold_chunk_ids,
            gold_chunk_ids=gold_chunk_ids,
        ),
    }

    for scenario in SCENARIO_ORDER:
        context = contexts[scenario]
        if not context.context_text.strip():
            raise ValueError(
                f"Question {row.get('question_id')} produced an empty {scenario} context."
            )

    return contexts


def load_runtime_components() -> RuntimeComponents:
    try:
        from epsa_rag.evaluation.answer_metrics import (
            answer_overlap_metrics,
            exact_match_score,
            partial_match_score,
            relaxed_answer_match,
        )
        from epsa_rag.rag.llm_client import OpenAIChatClient
        from epsa_rag.rag.prompt_templates import build_final_answer_messages
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the current EPSA-RAG runtime modules. "
            "Run this script from the repository root with the project virtual "
            "environment activated."
        ) from exc

    return RuntimeComponents(
        llm_client_factory=OpenAIChatClient,
        build_final_answer_messages=build_final_answer_messages,
        exact_match_score=exact_match_score,
        partial_match_score=partial_match_score,
        answer_overlap_metrics=answer_overlap_metrics,
        relaxed_answer_match=relaxed_answer_match,
    )


def build_prepared_result(
    *,
    row: Mapping[str, Any],
    context: OracleContext,
    repeat_index: int,
) -> dict[str, Any]:
    return {
        "question_id": str(row.get("question_id")),
        "question": str(row.get("question")),
        "gold_answer": str(row.get("gold_answer")),
        "historical_epsa_prediction": str(row.get("predicted_answer")),
        "historical_epsa_exact_match": safe_float(row.get("exact_match")),
        "historical_epsa_answer_f1": safe_float(row.get("answer_f1")),
        "scenario": context.scenario,
        "scenario_description": context.description,
        "repeat_index": repeat_index,
        "prediction": "",
        "exact_match": "",
        "partial_match": "",
        "answer_precision": "",
        "answer_recall": "",
        "answer_f1": "",
        "relaxed_answer_correct": "",
        "relaxed_match_type": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model_name": "",
        "generation_status": "prepared_only",
        "generation_error": "",
        "context_document_count": len(context.chunk_ids),
        "estimated_context_tokens": estimate_token_count(context.context_text),
        "context_sha256": hashlib.sha256(
            context.context_text.encode("utf-8")
        ).hexdigest(),
        "context_chunk_ids": json.dumps(context.chunk_ids, ensure_ascii=False),
        "gold_chunk_ids": json.dumps(context.gold_chunk_ids, ensure_ascii=False),
        "context_note": context.note,
    }


def run_generation(
    *,
    row: Mapping[str, Any],
    context: OracleContext,
    repeat_index: int,
    runtime: RuntimeComponents,
    llm_client: Any,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    result = build_prepared_result(
        row=row,
        context=context,
        repeat_index=repeat_index,
    )

    question = str(row.get("question") or "")
    gold_answer = str(row.get("gold_answer") or "")

    try:
        response = llm_client.complete(
            runtime.build_final_answer_messages(question, context.context_text),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prediction = str(response.content or "").strip()
        overlap = runtime.answer_overlap_metrics(prediction, gold_answer)
        relaxed = runtime.relaxed_answer_match(prediction, gold_answer)

        result.update(
            {
                "prediction": prediction,
                "exact_match": runtime.exact_match_score(prediction, gold_answer),
                "partial_match": runtime.partial_match_score(prediction, gold_answer),
                "answer_precision": overlap.precision,
                "answer_recall": overlap.recall,
                "answer_f1": overlap.f1,
                "relaxed_answer_correct": bool(relaxed.correct),
                "relaxed_match_type": relaxed.match_type,
                "prompt_tokens": int(response.prompt_tokens),
                "completion_tokens": int(response.completion_tokens),
                "total_tokens": int(response.total_tokens),
                "model_name": str(response.model_name or ""),
                "generation_status": "completed",
            }
        )
    except Exception as exc:
        result.update(
            {
                "generation_status": "failed",
                "generation_error": str(exc),
            }
        )

    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_context_manifest(
    path: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in manifest_rows:
            writer.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def scenario_summary(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}

    for row in results:
        key = (str(row["question_id"]), str(row["scenario"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []

    for (question_id, scenario), rows in grouped.items():
        completed = [row for row in rows if row["generation_status"] == "completed"]
        failed = [row for row in rows if row["generation_status"] == "failed"]
        prepared = [row for row in rows if row["generation_status"] == "prepared_only"]
        predictions = [str(row["prediction"]) for row in completed]
        unique_predictions = list(dict.fromkeys(predictions))

        exact_values = [safe_float(row["exact_match"]) for row in completed]
        f1_values = [safe_float(row["answer_f1"]) for row in completed]
        relaxed_values = [
            1.0 if truthy(row["relaxed_answer_correct"]) else 0.0 for row in completed
        ]

        summaries.append(
            {
                "question_id": question_id,
                "question": str(rows[0]["question"]),
                "gold_answer": str(rows[0]["gold_answer"]),
                "scenario": scenario,
                "runs_requested": len(rows),
                "runs_completed": len(completed),
                "failed_runs": len(failed),
                "prepared_runs": len(prepared),
                "strict_correct_count": int(sum(exact_values)),
                "strict_correct_rate": (
                    round(sum(exact_values) / len(exact_values), 6)
                    if exact_values
                    else None
                ),
                "mean_answer_f1": (
                    round(statistics.mean(f1_values), 6) if f1_values else None
                ),
                "relaxed_correct_count": int(sum(relaxed_values)),
                "relaxed_correct_rate": (
                    round(sum(relaxed_values) / len(relaxed_values), 6)
                    if relaxed_values
                    else None
                ),
                "unique_predictions": unique_predictions,
                "prediction_stable": len(unique_predictions) <= 1,
                "context_document_count": safe_int(
                    rows[0]["context_document_count"]
                ),
                "estimated_context_tokens": safe_int(
                    rows[0]["estimated_context_tokens"]
                ),
                "context_sha256": str(rows[0]["context_sha256"]),
            }
        )

    scenario_rank = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    summaries.sort(
        key=lambda item: (
            item["question_id"],
            scenario_rank.get(item["scenario"], 999),
        )
    )
    return summaries


def majority_correct(summary_row: Mapping[str, Any]) -> bool:
    rate = summary_row.get("strict_correct_rate")
    if rate is None:
        return False
    return float(rate) >= 0.5


def build_question_interpretations(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, Mapping[str, Any]]] = {}

    for row in summaries:
        by_question.setdefault(str(row["question_id"]), {})[
            str(row["scenario"])
        ] = row

    interpretations: list[dict[str, Any]] = []

    for question_id, scenario_map in by_question.items():
        missing = [name for name in SCENARIO_ORDER if name not in scenario_map]
        if missing:
            interpretations.append(
                {
                    "question_id": question_id,
                    "recoverable": False,
                    "primary_interpretation": "incomplete_oracle_results",
                    "details": f"Missing scenarios: {missing}",
                    "generation_instability": False,
                }
            )
            continue

        a = scenario_map["A_current_epsa_context"]
        b = scenario_map["B_full_merged_context"]
        c = scenario_map["C_current_plus_omitted_gold"]
        d = scenario_map["D_gold_documents_only"]

        if not any(safe_int(scenario_map[name].get("runs_completed")) > 0 for name in SCENARIO_ORDER):
            interpretations.append(
                {
                    "question_id": question_id,
                    "question": str(a["question"]),
                    "gold_answer": str(a["gold_answer"]),
                    "recoverable": False,
                    "primary_interpretation": "oracle_generation_not_run",
                    "details": (
                        "Contexts were prepared successfully, but no LLM generation "
                        "was run, so recoverability has not yet been measured."
                    ),
                    "generation_instability": False,
                }
            )
            continue

        a_correct = majority_correct(a)
        b_correct = majority_correct(b)
        c_correct = majority_correct(c)
        d_correct = majority_correct(d)

        unstable = any(
            not truthy(scenario_map[name]["prediction_stable"])
            for name in SCENARIO_ORDER
        )

        if not a_correct and c_correct:
            interpretation = "specific_omitted_gold_evidence_recovers_answer"
            details = (
                "Current EPSA context failed, while adding omitted gold supporting "
                "documents recovered the answer."
            )
            recoverable = True
        elif not a_correct and b_correct:
            interpretation = "full_merged_context_recovers_answer"
            details = (
                "Current EPSA context failed, while the full merged candidate "
                "context recovered the answer."
            )
            recoverable = True
        elif not d_correct:
            interpretation = "gold_context_did_not_reliably_recover_answer"
            details = (
                "Gold supporting documents alone did not reliably produce the "
                "correct answer, indicating generation, prompt, or ambiguity risk."
            )
            recoverable = False
        elif a_correct:
            interpretation = "current_context_rerun_succeeded"
            details = (
                "The reconstructed current EPSA context succeeded during rerun; "
                "treat the historical error as possible generation instability."
            )
            recoverable = False
        else:
            interpretation = "no_clear_context_recovery"
            details = "No oracle context produced a reliable strict recovery."
            recoverable = False

        interpretations.append(
            {
                "question_id": question_id,
                "question": str(a["question"]),
                "gold_answer": str(a["gold_answer"]),
                "recoverable": recoverable,
                "primary_interpretation": interpretation,
                "details": details,
                "generation_instability": unstable,
                "A_current_correct": a_correct,
                "B_full_merged_correct": b_correct,
                "C_plus_gold_correct": c_correct,
                "D_gold_only_correct": d_correct,
            }
        )

    interpretations.sort(key=lambda item: item["question_id"])
    return interpretations


def build_summary(
    *,
    args: argparse.Namespace,
    selected_rows: pd.DataFrame,
    results: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    interpretations: Sequence[Mapping[str, Any]],
    output_paths: Mapping[str, Path],
) -> dict[str, Any]:
    completed = sum(1 for row in results if row["generation_status"] == "completed")
    failed = sum(1 for row in results if row["generation_status"] == "failed")
    prepared = sum(1 for row in results if row["generation_status"] == "prepared_only")
    recoverable_count = sum(
        1 for item in interpretations if truthy(item.get("recoverable"))
    )
    instability_count = sum(
        1 for item in interpretations if truthy(item.get("generation_instability"))
    )

    return {
        "epsa_results_path": str(args.epsa_results),
        "corpus_path": str(args.corpus_path),
        "target_question_count": int(len(selected_rows)),
        "target_question_ids": selected_rows["question_id"].astype(str).tolist(),
        "scenario_count_per_question": len(SCENARIO_ORDER),
        "repeats": args.repeats,
        "prepare_only": args.prepare_only,
        "generation_runs_completed": completed,
        "generation_runs_failed": failed,
        "prepared_context_runs": prepared,
        "recoverable_question_count": recoverable_count,
        "generation_instability_question_count": instability_count,
        "cluster_meets_minimum_recoverable_count": recoverable_count >= 2,
        "method_note": (
            "Gold supporting titles are used only for bounded diagnostic oracle "
            "contexts and must never enter runtime EPSA behavior."
        ),
        "scenario_summaries": list(summaries),
        "question_interpretations": list(interpretations),
        "output_files": {key: str(value) for key, value in output_paths.items()},
    }


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# EPSA Oracle-Context Diagnostic Report",
        "",
        "## Run Summary",
        "",
        f"- Target questions: **{summary['target_question_count']}**",
        f"- Repeats per scenario: **{summary['repeats']}**",
        f"- Prepare-only: **{summary['prepare_only']}**",
        f"- Completed LLM calls: **{summary['generation_runs_completed']}**",
        f"- Failed LLM calls: **{summary['generation_runs_failed']}**",
        f"- Recoverable questions: **{summary['recoverable_question_count']}**",
        (
            "- Minimum two-question recovery condition: "
            f"**{summary['cluster_meets_minimum_recoverable_count']}**"
        ),
        "",
        "## Scenario Results",
        "",
        "| Question ID | Scenario | Correct rate | Mean F1 | Stable | Predictions |",
        "|---|---|---:|---:|---:|---|",
    ]

    for row in summary["scenario_summaries"]:
        correct_rate = row["strict_correct_rate"]
        f1 = row["mean_answer_f1"]
        rendered_rate = "not run" if correct_rate is None else f"{correct_rate:.3f}"
        rendered_f1 = "not run" if f1 is None else f"{f1:.3f}"
        predictions = "; ".join(row["unique_predictions"]) or "not generated"
        predictions = predictions.replace("|", "\\|")
        lines.append(
            f"| `{row['question_id']}` | `{row['scenario']}` | "
            f"{rendered_rate} | {rendered_f1} | "
            f"{row['prediction_stable']} | {predictions} |"
        )

    lines.extend(
        [
            "",
            "## Per-Question Interpretation",
            "",
        ]
    )

    for item in summary["question_interpretations"]:
        lines.extend(
            [
                f"### `{item['question_id']}`",
                "",
                f"- Interpretation: `{item['primary_interpretation']}`",
                f"- Recoverable through context: **{item['recoverable']}**",
                f"- Generation instability detected: **{item['generation_instability']}**",
                f"- Details: {item['details']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Research Boundary",
            "",
            (
                "Gold supporting titles were used only to construct diagnostic "
                "upper-bound contexts. They must not be used by the runtime EPSA "
                "algorithm."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded A-D oracle-context diagnostics over EPSA result rows "
            "without modifying EPSA runtime behavior."
        )
    )
    parser.add_argument(
        "--epsa-results",
        default="outputs/epsa_rag/epsa_rag_results_20260710T080712Z.csv",
    )
    parser.add_argument(
        "--corpus-path",
        default="data/processed/hotpotqa_paragraph_chunks.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/epsa_rag/oracle_context_diagnostics",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        dest="question_ids",
        help=(
            "Question ID to diagnose. Repeat this argument for multiple IDs. "
            "When omitted, the verified all-gold-retrieved-but-none-selected "
            "false-sufficient cluster is selected automatically."
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and export all oracle contexts without calling the LLM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than zero.")

    epsa_results_path = Path(args.epsa_results)
    corpus_path = Path(args.corpus_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_epsa_results(epsa_results_path)
    corpus = load_corpus(corpus_path)
    selected_rows = select_rows(frame, question_ids=args.question_ids)

    runtime: RuntimeComponents | None = None
    llm_client: Any | None = None

    if not args.prepare_only:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in the current PowerShell "
                "session or run with --prepare-only."
            )
        runtime = load_runtime_components()
        llm_client = runtime.llm_client_factory(
            model_name=args.llm_model,
            timeout=args.llm_timeout,
        )

    result_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for _, pandas_row in selected_rows.iterrows():
        row = pandas_row.to_dict()
        contexts = build_oracle_contexts(row, corpus)

        for scenario in SCENARIO_ORDER:
            context = contexts[scenario]
            manifest_rows.append(
                {
                    "question_id": str(row["question_id"]),
                    "question": str(row["question"]),
                    "gold_answer": str(row["gold_answer"]),
                    "historical_epsa_prediction": str(row["predicted_answer"]),
                    "scenario": scenario,
                    "description": context.description,
                    "context_chunk_ids": context.chunk_ids,
                    "gold_chunk_ids": context.gold_chunk_ids,
                    "estimated_context_tokens": estimate_token_count(
                        context.context_text
                    ),
                    "context_sha256": hashlib.sha256(
                        context.context_text.encode("utf-8")
                    ).hexdigest(),
                    "note": context.note,
                    "context_text": context.context_text,
                }
            )

            for repeat_index in range(1, args.repeats + 1):
                if args.prepare_only:
                    result = build_prepared_result(
                        row=row,
                        context=context,
                        repeat_index=repeat_index,
                    )
                else:
                    assert runtime is not None
                    assert llm_client is not None
                    result = run_generation(
                        row=row,
                        context=context,
                        repeat_index=repeat_index,
                        runtime=runtime,
                        llm_client=llm_client,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )

                result_rows.append(result)
                print(
                    f"{row['question_id']} | {scenario} | "
                    f"run={repeat_index}/{args.repeats} | "
                    f"status={result['generation_status']} | "
                    f"prediction={result['prediction']!r}"
                )

    results_path = out_dir / "oracle_context_results.csv"
    contexts_path = out_dir / "oracle_contexts.jsonl"
    scenario_summary_path = out_dir / "oracle_scenario_summary.csv"
    summary_json_path = out_dir / "oracle_context_summary.json"
    summary_md_path = out_dir / "oracle_context_report.md"

    write_csv(results_path, result_rows)
    write_context_manifest(contexts_path, manifest_rows)

    summaries = scenario_summary(result_rows)
    write_csv(scenario_summary_path, summaries)

    interpretations = build_question_interpretations(summaries)
    output_paths = {
        "results_csv": results_path,
        "contexts_jsonl": contexts_path,
        "scenario_summary_csv": scenario_summary_path,
        "summary_json": summary_json_path,
        "report_markdown": summary_md_path,
    }
    summary = build_summary(
        args=args,
        selected_rows=selected_rows,
        results=result_rows,
        summaries=summaries,
        interpretations=interpretations,
        output_paths=output_paths,
    )

    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_md_path.write_text(markdown_report(summary), encoding="utf-8")

    print("\nSaved oracle results:", results_path)
    print("Saved oracle contexts:", contexts_path)
    print("Saved scenario summary:", scenario_summary_path)
    print("Saved JSON summary:", summary_json_path)
    print("Saved Markdown report:", summary_md_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
