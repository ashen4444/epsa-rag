"""Context pruning for EPSA.

The pruner formats only the evidence selected by the Sufficiency Decision Engine.

Important boundary:
    This module does not decide sufficiency, retrieve documents, generate
    next-hop queries, or call an LLM.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from epsa_rag.epsa.schemas import PrunedContext, ScoredEvidenceUnit, SufficiencyDecision


class ContextPruner:
    """Select and format sentence-level context from a sufficiency decision."""

    def prune(
        self,
        sufficiency_decision: SufficiencyDecision,
        scored_evidence_units: list[ScoredEvidenceUnit],
    ) -> PrunedContext:
        """Return deterministic pruned context for final answer generation.

        Args:
            sufficiency_decision: Decision produced by SufficiencyDecisionEngine.
            scored_evidence_units: Candidate scored evidence units available to EPSA.

        Returns:
            PrunedContext containing selected provenance, formatted context text,
            token estimate, and removed evidence IDs.
        """

        all_units_by_id = {
            item.evidence_unit.evidence_unit_id: item
            for item in scored_evidence_units
        }
        all_ids = [item.evidence_unit.evidence_unit_id for item in scored_evidence_units]
        requested_ids = _dedupe_preserve_order(sufficiency_decision.selected_evidence_unit_ids)
        selected_units = [
            all_units_by_id[evidence_id]
            for evidence_id in requested_ids
            if evidence_id in all_units_by_id
        ]
        selected_units = _expand_bridge_context_if_needed(
            sufficiency_decision=sufficiency_decision,
            selected_units=selected_units,
            scored_evidence_units=scored_evidence_units,
        )
        selected_units = sorted(selected_units, key=_unit_sort_key)

        selected_ids = [item.evidence_unit.evidence_unit_id for item in selected_units]
        selected_chunk_ids = _dedupe_preserve_order(
            item.evidence_unit.chunk_id for item in selected_units if item.evidence_unit.chunk_id
        )
        selected_sentences = [
            item.evidence_unit.resolved_text or item.evidence_unit.sentence_text
            for item in selected_units
        ]
        selected_context_text = "\n\n".join(
            _format_evidence_unit(item) for item in selected_units
        )
        removed_ids = [evidence_id for evidence_id in all_ids if evidence_id not in set(selected_ids)]
        missing_requested_ids = [
            evidence_id for evidence_id in requested_ids if evidence_id not in all_units_by_id
        ]

        strategy = (
            "sufficient_bridge_neighbor_sentence_pruning"
            if sufficiency_decision.sufficient
            and sufficiency_decision.question_type == "bridge"
            and len(selected_ids) > len(requested_ids)
            else "sufficient_path_sentence_pruning"
            if sufficiency_decision.sufficient
            else "partial_evidence_sentence_pruning"
        )
        if not selected_units:
            strategy = "empty_evidence_pruning"

        return PrunedContext(
            selected_chunk_ids=selected_chunk_ids,
            selected_evidence_unit_ids=selected_ids,
            selected_sentences=selected_sentences,
            selected_context_text=selected_context_text,
            estimated_context_tokens=_estimate_tokens(selected_context_text),
            pruning_strategy=strategy,
            removed_evidence_unit_ids=removed_ids,
            metadata={
                "sufficient": sufficiency_decision.sufficient,
                "answer_candidate": sufficiency_decision.answer_candidate,
                "answer_type": sufficiency_decision.answer_type,
                "missing_evidence": sufficiency_decision.missing_evidence,
                "decision_reason": sufficiency_decision.decision_reason,
                "requested_evidence_unit_ids": requested_ids,
                "missing_requested_evidence_unit_ids": missing_requested_ids,
                "makes_sufficiency_decision": False,
                "retrieves_documents": False,
                "calls_llm": False,
            },
        )


def _expand_bridge_context_if_needed(
    *,
    sufficiency_decision: SufficiencyDecision,
    selected_units: list[ScoredEvidenceUnit],
    scored_evidence_units: list[ScoredEvidenceUnit],
) -> list[ScoredEvidenceUnit]:
    """Add bounded neighboring sentence context for sufficient bridge paths.

    Bridge answers often need the bridge sentence and the answer-side sentence.
    The selected evidence IDs are still preserved, but immediate neighbors from
    the same selected chunks give the final LLM enough local paragraph context
    without falling back to all retrieved documents.
    """

    if not sufficiency_decision.sufficient or sufficiency_decision.question_type != "bridge":
        return selected_units

    if not selected_units:
        return selected_units

    selected_by_id = {
        item.evidence_unit.evidence_unit_id: item
        for item in selected_units
    }
    selected_chunk_ids = {
        item.evidence_unit.chunk_id
        for item in selected_units
        if item.evidence_unit.chunk_id
    }
    selected_sentence_keys = {
        (item.evidence_unit.chunk_id, int(item.evidence_unit.sentence_id))
        for item in selected_units
    }

    expanded = dict(selected_by_id)
    for candidate in scored_evidence_units:
        unit = candidate.evidence_unit
        if unit.chunk_id not in selected_chunk_ids:
            continue

        candidate_key = (unit.chunk_id, int(unit.sentence_id))
        if any(
            candidate_key[0] == selected_key[0]
            and abs(candidate_key[1] - selected_key[1]) <= 1
            for selected_key in selected_sentence_keys
        ):
            expanded[unit.evidence_unit_id] = candidate

        if len(expanded) >= 6:
            break

    return list(expanded.values())


def _format_evidence_unit(scored_unit: ScoredEvidenceUnit) -> str:
    unit = scored_unit.evidence_unit
    sentence_text = unit.resolved_text or unit.sentence_text
    return (
        f"[Title: {unit.doc_title} | Chunk: {unit.chunk_id} | Sentence: {unit.sentence_id}]\n"
        f"{sentence_text}"
    )


def _unit_sort_key(scored_unit: ScoredEvidenceUnit) -> tuple[int, str, int, str]:
    unit = scored_unit.evidence_unit
    rank = unit.retrieval_rank if unit.retrieval_rank is not None else 10**9
    return (int(rank), unit.chunk_id, int(unit.sentence_id), unit.evidence_unit_id)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / 4))


def _dedupe_preserve_order(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


__all__ = ["ContextPruner"]
