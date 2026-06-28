"""Deterministic EPSA controller integration.

The controller runs the existing EPSA modules over an already-retrieved
candidate paragraph chunk pool.

Important boundary:
    This module does not own a retriever, modify retrieval behavior, call an LLM,
    generate final answers, or run the baseline-vs-EPSA evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

from epsa_rag.epsa.chunk_evidence_analyzer import CandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.context_pruner import ContextPruner
from epsa_rag.epsa.evidence_graph_builder import EvidenceGraphBuilder
from epsa_rag.epsa.evidence_path_searcher import EvidencePathSearcher
from epsa_rag.epsa.evidence_scorer import EvidenceScorer
from epsa_rag.epsa.evidence_unit_extractor import EvidenceUnitExtractor
from epsa_rag.epsa.next_query_generator import NextHopQueryGenerator
from epsa_rag.epsa.question_analyzer import QuestionAnalyzer
from epsa_rag.epsa.schemas import (
    CandidateChunkEvidence,
    EPSAControllerResult,
    EvidencePath,
    EvidenceUnit,
    NextHopQuery,
    QuestionAnalysis,
    ScoredEvidenceUnit,
)
from epsa_rag.epsa.sufficiency_decision_engine import SufficiencyDecisionEngine


class EPSAController:
    """Orchestrate EPSA modules for an already-retrieved candidate chunk pool."""

    def __init__(
        self,
        *,
        question_analyzer: QuestionAnalyzer | None = None,
        chunk_evidence_analyzer: CandidateChunkEvidenceAnalyzer | None = None,
        evidence_unit_extractor: EvidenceUnitExtractor | None = None,
        evidence_scorer: EvidenceScorer | None = None,
        evidence_graph_builder: EvidenceGraphBuilder | None = None,
        evidence_path_searcher: EvidencePathSearcher | None = None,
        sufficiency_decision_engine: SufficiencyDecisionEngine | None = None,
        context_pruner: ContextPruner | None = None,
        next_query_generator: NextHopQueryGenerator | None = None,
    ) -> None:
        self.question_analyzer = question_analyzer or QuestionAnalyzer()
        self.chunk_evidence_analyzer = chunk_evidence_analyzer or CandidateChunkEvidenceAnalyzer()
        self.evidence_unit_extractor = evidence_unit_extractor or EvidenceUnitExtractor()
        self.evidence_scorer = evidence_scorer or EvidenceScorer()
        self.evidence_graph_builder = evidence_graph_builder or EvidenceGraphBuilder()
        self.evidence_path_searcher = evidence_path_searcher or EvidencePathSearcher()
        self.sufficiency_decision_engine = sufficiency_decision_engine or SufficiencyDecisionEngine()
        self.context_pruner = context_pruner or ContextPruner()
        self.next_query_generator = next_query_generator or NextHopQueryGenerator()

    def run(
        self,
        question: str,
        retrieved_chunks: list[Any],
        *,
        max_paths: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> EPSAControllerResult:
        """Run deterministic EPSA over already-retrieved paragraph chunks.

        Args:
            question: Natural-language user/research question.
            retrieved_chunks: Candidate paragraph chunks returned by the fixed retriever.
            max_paths: Maximum candidate evidence paths to preserve.
            metadata: Optional caller metadata for later logging/failure analysis.

        Returns:
            EPSAControllerResult preserving all major intermediate EPSA outputs.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        chunks = list(retrieved_chunks or [])
        question_analysis = self.question_analyzer.analyze(question)

        candidate_chunk_evidence: list[CandidateChunkEvidence] = []
        evidence_units: list[EvidenceUnit] = []

        for index, chunk in enumerate(chunks):
            retrieval_rank = _safe_int(_first_present(chunk, "retrieval_rank", "rank"))
            retrieval_score = _safe_float_or_none(
                _first_present(chunk, "retrieval_score", "score", "fusion_score")
            )
            if retrieval_rank is None:
                retrieval_rank = index + 1

            candidate_evidence = self.chunk_evidence_analyzer.analyze(
                chunk=chunk,
                question_analysis=question_analysis,
                retrieval_rank=retrieval_rank,
                retrieval_score=retrieval_score,
            )
            candidate_chunk_evidence.append(candidate_evidence)
            evidence_units.extend(
                self.evidence_unit_extractor.extract_from_chunk(
                    candidate_evidence=candidate_evidence,
                    chunk=chunk,
                    question_analysis=question_analysis,
                )
            )

        scored_evidence_units: list[ScoredEvidenceUnit] = self.evidence_scorer.score_many(
            evidence_units=evidence_units,
            question_analysis=question_analysis,
        )
        evidence_graph = self.evidence_graph_builder.build(
            question_analysis=question_analysis,
            scored_evidence_units=scored_evidence_units,
        )
        evidence_paths = self.evidence_path_searcher.search_paths(
            evidence_graph=evidence_graph,
            question_analysis=question_analysis,
            max_paths=max_paths,
        )
        if not evidence_paths and scored_evidence_units:
            evidence_paths = _build_partial_evidence_paths(
                question_analysis=question_analysis,
                scored_evidence_units=scored_evidence_units,
                max_paths=max_paths,
            )
        sufficiency_decision = self.sufficiency_decision_engine.decide(
            question_analysis=question_analysis,
            evidence_graph=evidence_graph,
            evidence_paths=evidence_paths,
        )
        pruned_context = self.context_pruner.prune(
            sufficiency_decision=sufficiency_decision,
            scored_evidence_units=scored_evidence_units,
        )
        generated_next_query = self.next_query_generator.generate(
            question_analysis=question_analysis,
            sufficiency_decision=sufficiency_decision,
            evidence_graph=evidence_graph,
            evidence_paths=evidence_paths,
        )
        next_hop_query: NextHopQuery | None = (
            generated_next_query if generated_next_query.query else None
        )

        return EPSAControllerResult(
            question=question.strip(),
            question_analysis=question_analysis,
            candidate_chunk_evidence=candidate_chunk_evidence,
            evidence_units=evidence_units,
            scored_evidence_units=scored_evidence_units,
            evidence_graph=evidence_graph,
            evidence_paths=evidence_paths,
            sufficiency_decision=sufficiency_decision,
            pruned_context=pruned_context,
            next_hop_query=next_hop_query,
            selected_chunk_ids=list(pruned_context.selected_chunk_ids),
            selected_evidence_unit_ids=list(pruned_context.selected_evidence_unit_ids),
            sufficient=sufficiency_decision.sufficient,
            metadata={
                "controller": self.__class__.__name__,
                "version": "rule_based_v1",
                "num_retrieved_chunks": len(chunks),
                "num_candidate_chunk_evidence": len(candidate_chunk_evidence),
                "num_evidence_units": len(evidence_units),
                "num_scored_evidence_units": len(scored_evidence_units),
                "num_graph_nodes": len(evidence_graph.nodes),
                "num_graph_edges": len(evidence_graph.edges),
                "num_evidence_paths": len(evidence_paths),
                "generated_next_query_type": generated_next_query.query_type,
                "generated_next_query_reason": generated_next_query.reason,
                "calls_llm": False,
                "retrieves_documents": False,
                "modifies_retriever": False,
                "generates_final_answer": False,
                "runs_evaluation_pipeline": False,
                **(metadata or {}),
            },
        )


def _build_partial_evidence_paths(
    *,
    question_analysis: QuestionAnalysis,
    scored_evidence_units: list[ScoredEvidenceUnit],
    max_paths: int,
) -> list[EvidencePath]:
    if max_paths <= 0 or not scored_evidence_units:
        return []

    question_type = _question_type(question_analysis)
    ranked_units = sorted(
        scored_evidence_units,
        key=lambda item: (
            -float(item.final_score),
            item.evidence_unit.retrieval_rank if item.evidence_unit.retrieval_rank is not None else 10**9,
            item.evidence_unit.evidence_unit_id,
        ),
    )

    selected_units = ranked_units[:1]
    evidence_unit_ids = [item.evidence_unit.evidence_unit_id for item in selected_units]
    entities = _dedupe_preserve_order(
        entity for item in selected_units for entity in item.evidence_unit.entities
    )
    relations = _dedupe_preserve_order(
        relation for item in selected_units for relation in item.evidence_unit.relation_hints
    )
    seed_entities = _dedupe_preserve_order(_as_text_list(getattr(question_analysis, "seed_entities", [])))
    seed_norms = {_norm_text(seed) for seed in seed_entities}
    non_seed_entities = [entity for entity in entities if _norm_text(entity) not in seed_norms]
    entity_chain = _dedupe_preserve_order([*seed_entities[:1], *non_seed_entities[:2]])
    bridge_entity = non_seed_entities[0] if non_seed_entities else None

    answer_candidate = None
    if question_type == "factoid" and non_seed_entities:
        answer_candidate = non_seed_entities[0]

    average_score = sum(float(item.final_score) for item in selected_units) / len(selected_units)
    return [
        EvidencePath(
            path_id=f"controller_partial::{':'.join(evidence_unit_ids)}",
            question_type=question_type,
            node_ids=[],
            edge_ids=[],
            evidence_unit_ids=evidence_unit_ids,
            entity_chain=entity_chain,
            relation_chain=relations,
            answer_candidate=answer_candidate,
            answer_type=_expected_answer_type(question_analysis),
            score=round(average_score, 6),
            metadata={
                "path_kind": "controller_partial_evidence_fallback",
                "bridge_entity": bridge_entity,
                "makes_sufficiency_decision": False,
                "generated_by_controller_fallback": True,
            },
        )
    ]


def _question_type(question_analysis: QuestionAnalysis) -> str:
    value = getattr(question_analysis, "question_type", "factoid")
    normalized = _norm_text(value).replace("-", "_")
    return "yes_no" if normalized in {"yes no", "yes_no", "boolean"} else normalized or "factoid"


def _expected_answer_type(question_analysis: QuestionAnalysis) -> str:
    value = getattr(question_analysis, "expected_answer_type", "UNKNOWN")
    return _as_text(value).upper().replace(" ", "_").replace("-", "_") or "UNKNOWN"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    if hasattr(value, "relation"):
        return str(getattr(value, "relation") or "")
    if hasattr(value, "text"):
        return str(getattr(value, "text") or "")
    if hasattr(value, "answer_type"):
        return _as_text(getattr(value, "answer_type"))
    return str(value)


def _as_text_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    try:
        return [_as_text(value) for value in values if _as_text(value)]
    except TypeError:
        return [_as_text(values)]


def _norm_text(value: Any) -> str:
    return " ".join(_as_text(value).lower().replace("_", " ").replace("-", " ").split())


def _dedupe_preserve_order(values: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _as_text(value).strip()
        key = _norm_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def _first_present(obj: Any, *names: str) -> Any:
    for nested_name in ("result", "chunk", "document", "metadata"):
        nested = _get_value(obj, nested_name)
        if nested is not None and nested is not obj:
            nested_value = _first_present(nested, *names)
            if nested_value is not None:
                return nested_value

    for name in names:
        value = _get_value(obj, name)
        if value is not None:
            return value
    return None


def _get_value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    if is_dataclass(obj):
        return getattr(obj, name, None)
    return getattr(obj, name, None)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["EPSAController"]
