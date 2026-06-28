"""Evidence graph construction for EPSA.

The graph builder is a deterministic post-retrieval EPSA module. It converts
scored sentence-level evidence units into a lightweight, serializable evidence
network that can be searched later for candidate reasoning paths.

Important boundary:
    This module does not decide evidence sufficiency, does not prune context,
    and does not generate next-hop queries.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from epsa_rag.epsa.schemas import (
    EvidenceGraph,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    QuestionAnalysis,
    ScoredEvidenceUnit,
)


class EvidenceGraphBuilder:
    """Build deterministic evidence graphs from scored evidence units."""

    def build(
        self,
        question_analysis: QuestionAnalysis,
        scored_evidence_units: list[ScoredEvidenceUnit],
    ) -> EvidenceGraph:
        """Build an EvidenceGraph.

        Args:
            question_analysis: Deterministic question analysis from EPSA Chat 10.
            scored_evidence_units: Scored sentence evidence units from EPSA Chat 11.

        Returns:
            EvidenceGraph containing stable nodes and provenance-preserving edges.
        """

        node_map: dict[str, EvidenceGraphNode] = {}
        edge_map: dict[str, EvidenceGraphEdge] = {}

        question_type = _as_text(getattr(question_analysis, "question_type", "UNKNOWN"))
        expected_answer_type = _as_text(
            getattr(question_analysis, "expected_answer_type", "UNKNOWN")
        )
        required_relation_hints = _as_text_list(
            getattr(question_analysis, "required_relation_hints", [])
        )
        seed_entities = _dedupe_preserve_order(
            _as_text_list(getattr(question_analysis, "seed_entities", []))
        )

        seed_entity_node_ids: list[str] = []
        evidence_unit_node_ids: list[str] = []
        entity_node_ids: list[str] = []

        seed_norms = {_norm_key(seed) for seed in seed_entities if seed}

        for seed_entity in seed_entities:
            node_id = stable_node_id("entity", seed_entity)
            seed_entity_node_ids.append(node_id)
            _add_or_update_node(
                node_map,
                EvidenceGraphNode(
                    node_id=node_id,
                    node_type="entity",
                    label=seed_entity,
                    metadata={
                        "is_question_seed": True,
                        "original_label": seed_entity,
                    },
                ),
            )

        for scored_unit in scored_evidence_units:
            evidence_unit = scored_unit.evidence_unit
            evidence_score = _safe_float(getattr(scored_unit, "final_score", 0.0))
            evidence_unit_id = _as_text(getattr(evidence_unit, "evidence_unit_id", ""))
            chunk_id = _as_text(getattr(evidence_unit, "chunk_id", ""))
            doc_title = _as_text(getattr(evidence_unit, "doc_title", ""))
            paragraph_index = _safe_int(getattr(evidence_unit, "paragraph_index", -1))
            sentence_id = _safe_int(getattr(evidence_unit, "sentence_id", -1))
            sentence_text = _as_text(getattr(evidence_unit, "sentence_text", ""))
            resolved_text = _as_text(getattr(evidence_unit, "resolved_text", sentence_text))
            retrieval_rank = getattr(evidence_unit, "retrieval_rank", None)
            retrieval_score = getattr(evidence_unit, "retrieval_score", None)

            entities = _dedupe_preserve_order(
                _as_text_list(getattr(evidence_unit, "entities", []))
            )
            relation_hints = _dedupe_preserve_order(
                _as_text_list(getattr(evidence_unit, "relation_hints", []))
            )
            answer_type_candidates = _dedupe_preserve_order(
                _as_text_list(getattr(evidence_unit, "answer_type_candidates", []))
            )
            question_entity_overlap = _dedupe_preserve_order(
                _as_text_list(getattr(evidence_unit, "question_entity_overlap", []))
            )

            chunk_node_id = stable_node_id("chunk", chunk_id)
            title_node_id = stable_node_id("title", doc_title)
            sentence_node_id = stable_sentence_node_id(evidence_unit_id)

            if sentence_node_id not in evidence_unit_node_ids:
                evidence_unit_node_ids.append(sentence_node_id)

            _add_or_update_node(
                node_map,
                EvidenceGraphNode(
                    node_id=chunk_node_id,
                    node_type="chunk",
                    label=chunk_id,
                    metadata={
                        "chunk_id": chunk_id,
                        "doc_title": doc_title,
                        "paragraph_index": paragraph_index,
                    },
                ),
            )
            _add_or_update_node(
                node_map,
                EvidenceGraphNode(
                    node_id=title_node_id,
                    node_type="title",
                    label=doc_title,
                    metadata={
                        "doc_title": doc_title,
                        "normalized_title": _norm_key(doc_title),
                    },
                ),
            )
            _add_or_update_node(
                node_map,
                EvidenceGraphNode(
                    node_id=sentence_node_id,
                    node_type="sentence",
                    label=sentence_text,
                    metadata={
                        "evidence_unit_id": evidence_unit_id,
                        "chunk_id": chunk_id,
                        "doc_title": doc_title,
                        "paragraph_index": paragraph_index,
                        "sentence_id": sentence_id,
                        "sentence_text": sentence_text,
                        "resolved_text": resolved_text,
                        "final_score": evidence_score,
                        "score_breakdown": dict(
                            getattr(scored_unit, "score_breakdown", {}) or {}
                        ),
                        "retrieval_rank": retrieval_rank,
                        "retrieval_score": retrieval_score,
                        "is_supporting_sentence": getattr(
                            evidence_unit, "is_supporting_sentence", None
                        ),
                    },
                ),
            )

            _add_edge(
                edge_map,
                source_id=chunk_node_id,
                target_id=sentence_node_id,
                edge_type="chunk_to_sentence",
                weight=evidence_score,
                evidence_unit_id=evidence_unit_id,
                metadata={"chunk_id": chunk_id},
            )
            _add_edge(
                edge_map,
                source_id=title_node_id,
                target_id=sentence_node_id,
                edge_type="title_to_sentence",
                weight=evidence_score,
                evidence_unit_id=evidence_unit_id,
                metadata={"doc_title": doc_title},
            )

            title_entity_node_id = stable_node_id("entity", doc_title)
            if doc_title:
                _add_or_update_node(
                    node_map,
                    EvidenceGraphNode(
                        node_id=title_entity_node_id,
                        node_type="entity",
                        label=doc_title,
                        metadata={
                            "original_label": doc_title,
                            "from_doc_title": True,
                            "is_question_seed": _norm_key(doc_title) in seed_norms,
                        },
                    ),
                )
                if title_entity_node_id not in entity_node_ids:
                    entity_node_ids.append(title_entity_node_id)
                _add_edge(
                    edge_map,
                    source_id=title_node_id,
                    target_id=title_entity_node_id,
                    edge_type="title_to_entity",
                    weight=max(evidence_score, 0.1),
                    evidence_unit_id=evidence_unit_id,
                    metadata={"doc_title": doc_title},
                )
                _add_edge(
                    edge_map,
                    source_id=sentence_node_id,
                    target_id=title_entity_node_id,
                    edge_type="sentence_in_document_about_entity",
                    weight=max(evidence_score, 0.1),
                    evidence_unit_id=evidence_unit_id,
                    metadata={"doc_title": doc_title},
                )

            entity_node_ids_in_sentence: list[str] = []
            for entity in entities:
                entity_node_id = stable_node_id("entity", entity)
                entity_node_ids_in_sentence.append(entity_node_id)
                if entity_node_id not in entity_node_ids:
                    entity_node_ids.append(entity_node_id)

                _add_or_update_node(
                    node_map,
                    EvidenceGraphNode(
                        node_id=entity_node_id,
                        node_type="entity",
                        label=entity,
                        metadata={
                            "original_label": entity,
                            "is_question_seed": _norm_key(entity) in seed_norms,
                        },
                    ),
                )
                _add_edge(
                    edge_map,
                    source_id=sentence_node_id,
                    target_id=entity_node_id,
                    edge_type="sentence_mentions_entity",
                    weight=evidence_score,
                    evidence_unit_id=evidence_unit_id,
                    metadata={"entity": entity},
                )

            for relation_hint in relation_hints:
                relation_node_id = stable_node_id("relation", relation_hint)
                _add_or_update_node(
                    node_map,
                    EvidenceGraphNode(
                        node_id=relation_node_id,
                        node_type="relation",
                        label=relation_hint,
                        metadata={
                            "relation": relation_hint,
                            "matches_required_relation": _matches_any(
                                relation_hint, required_relation_hints
                            ),
                        },
                    ),
                )
                _add_edge(
                    edge_map,
                    source_id=sentence_node_id,
                    target_id=relation_node_id,
                    edge_type="sentence_has_relation",
                    weight=evidence_score,
                    evidence_unit_id=evidence_unit_id,
                    relation=relation_hint,
                    metadata={
                        "relation": relation_hint,
                        "matches_required_relation": _matches_any(
                            relation_hint, required_relation_hints
                        ),
                    },
                )

            for answer_type in answer_type_candidates:
                answer_type_node_id = stable_node_id("answer_type", answer_type)
                _add_or_update_node(
                    node_map,
                    EvidenceGraphNode(
                        node_id=answer_type_node_id,
                        node_type="answer_type",
                        label=answer_type,
                        metadata={
                            "answer_type": answer_type,
                            "matches_expected_answer_type": _same_label(
                                answer_type, expected_answer_type
                            ),
                        },
                    ),
                )
                _add_edge(
                    edge_map,
                    source_id=sentence_node_id,
                    target_id=answer_type_node_id,
                    edge_type="sentence_has_answer_type",
                    weight=evidence_score,
                    evidence_unit_id=evidence_unit_id,
                    metadata={
                        "answer_type": answer_type,
                        "matches_expected_answer_type": _same_label(
                            answer_type, expected_answer_type
                        ),
                    },
                )

            for left_id, right_id in _unique_pairs(entity_node_ids_in_sentence):
                _add_edge(
                    edge_map,
                    source_id=left_id,
                    target_id=right_id,
                    edge_type="entity_cooccurs_with_entity",
                    weight=evidence_score,
                    evidence_unit_id=evidence_unit_id,
                    metadata={"within_sentence_node_id": sentence_node_id},
                )

            matched_seed_node_ids = self._matched_seed_node_ids(
                seed_entities=seed_entities,
                sentence_entities=entities,
                question_entity_overlap=question_entity_overlap,
                doc_title=doc_title,
                sentence_text=resolved_text or sentence_text,
            )
            for seed_node_id in matched_seed_node_ids:
                _add_edge(
                    edge_map,
                    source_id=seed_node_id,
                    target_id=sentence_node_id,
                    edge_type="seed_entity_to_sentence",
                    weight=max(evidence_score, 0.1),
                    evidence_unit_id=evidence_unit_id,
                    metadata={"reason": "seed_entity_match"},
                )

                for entity_node_id in entity_node_ids_in_sentence:
                    if entity_node_id != seed_node_id:
                        _add_edge(
                            edge_map,
                            source_id=seed_node_id,
                            target_id=entity_node_id,
                            edge_type="possible_bridge",
                            weight=evidence_score,
                            evidence_unit_id=evidence_unit_id,
                            metadata={
                                "via_sentence_node_id": sentence_node_id,
                                "reason": "non_seed_entity_in_seed_matched_sentence",
                            },
                        )

            sentence_has_expected_answer_type = any(
                _same_label(candidate_type, expected_answer_type)
                for candidate_type in answer_type_candidates
            )
            if sentence_has_expected_answer_type:
                for entity_node_id in entity_node_ids_in_sentence:
                    entity_label = node_map[entity_node_id].label
                    if _norm_key(entity_label) not in seed_norms:
                        _add_edge(
                            edge_map,
                            source_id=sentence_node_id,
                            target_id=entity_node_id,
                            edge_type="possible_answer_candidate",
                            weight=evidence_score,
                            evidence_unit_id=evidence_unit_id,
                            metadata={
                                "expected_answer_type": expected_answer_type,
                                "reason": "entity_in_expected_answer_type_sentence",
                            },
                        )

        graph = EvidenceGraph(
            nodes=dict(sorted(node_map.items(), key=lambda item: item[0])),
            edges=sorted(edge_map.values(), key=lambda edge: edge.edge_id),
            question_type=question_type,
            seed_entity_node_ids=_dedupe_preserve_order(seed_entity_node_ids),
            evidence_unit_node_ids=_dedupe_preserve_order(evidence_unit_node_ids),
            entity_node_ids=_dedupe_preserve_order(entity_node_ids),
            metadata={
                "expected_answer_type": expected_answer_type,
                "required_relation_hints": required_relation_hints,
                "num_scored_evidence_units": len(scored_evidence_units),
                "makes_sufficiency_decision": False,
            },
        )
        return graph

    def _matched_seed_node_ids(
        self,
        *,
        seed_entities: list[str],
        sentence_entities: list[str],
        question_entity_overlap: list[str],
        doc_title: str,
        sentence_text: str,
    ) -> list[str]:
        sentence_entity_norms = {_norm_key(entity) for entity in sentence_entities}
        overlap_norms = {_norm_key(entity) for entity in question_entity_overlap}
        title_norm = _norm_key(doc_title)
        sentence_norm = f" {_norm_key(sentence_text).replace('_', ' ')} "

        matched: list[str] = []
        for seed in seed_entities:
            seed_norm = _norm_key(seed)
            seed_phrase = seed_norm.replace("_", " ")
            if (
                seed_norm in sentence_entity_norms
                or seed_norm in overlap_norms
                or seed_norm == title_norm
                or f" {seed_phrase} " in sentence_norm
            ):
                matched.append(stable_node_id("entity", seed))
        return _dedupe_preserve_order(matched)


# Public helper functions are intentionally kept in this module because stable
# IDs are central to graph explainability and useful in tests/failure analysis.


def stable_node_id(node_type: str, label: str) -> str:
    return f"{node_type}::{_norm_key(label)}"


def stable_sentence_node_id(evidence_unit_id: str) -> str:
    return f"sentence::{evidence_unit_id}"


def stable_edge_id(
    source_id: str,
    target_id: str,
    edge_type: str,
    evidence_unit_id: str | None = None,
    relation: str | None = None,
) -> str:
    evidence_part = _norm_key(evidence_unit_id or "global")
    relation_part = _norm_key(relation or "none")
    return (
        f"edge::{edge_type}::{source_id}::{target_id}::"
        f"evidence::{evidence_part}::relation::{relation_part}"
    )


def _add_or_update_node(
    node_map: dict[str, EvidenceGraphNode],
    node: EvidenceGraphNode,
) -> None:
    existing = node_map.get(node.node_id)
    if existing is None:
        node_map[node.node_id] = node
        return

    merged_metadata = dict(existing.metadata)
    for key, value in node.metadata.items():
        if key not in merged_metadata:
            merged_metadata[key] = value
        elif isinstance(merged_metadata[key], bool) and isinstance(value, bool):
            merged_metadata[key] = merged_metadata[key] or value
        elif key == "original_label" and merged_metadata[key] != value:
            labels = merged_metadata.get("original_labels")
            if labels is None:
                labels = [merged_metadata[key]]
            if value not in labels:
                labels.append(value)
            merged_metadata["original_labels"] = labels

    node_map[node.node_id] = replace(existing, metadata=merged_metadata)


def _add_edge(
    edge_map: dict[str, EvidenceGraphEdge],
    *,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    evidence_unit_id: str | None = None,
    relation: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    clean_weight = max(0.0, _safe_float(weight))
    edge_id = stable_edge_id(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        evidence_unit_id=evidence_unit_id,
        relation=relation,
    )
    existing = edge_map.get(edge_id)
    if existing is None:
        edge_map[edge_id] = EvidenceGraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=clean_weight,
            evidence_unit_id=evidence_unit_id,
            relation=relation,
            metadata=metadata or {},
        )
        return

    if clean_weight > existing.weight:
        merged_metadata = dict(existing.metadata)
        merged_metadata.update(metadata or {})
        edge_map[edge_id] = replace(
            existing,
            weight=clean_weight,
            metadata=merged_metadata,
        )


def _norm_key(text: Any) -> str:
    value = _as_text(text)
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


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
        answer_type = getattr(value, "answer_type")
        return _as_text(answer_type)
    return str(value)


def _as_text_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    if not isinstance(values, Iterable):
        return [_as_text(values)]
    return [_as_text(value) for value in values if _as_text(value)]


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return -1
        return int(value)
    except (TypeError, ValueError):
        return -1


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = _norm_key(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _same_label(left: str, right: str) -> bool:
    return _norm_key(left) == _norm_key(right)


def _matches_any(value: str, candidates: list[str]) -> bool:
    value_norm = _norm_key(value).replace("_", " ")
    for candidate in candidates:
        candidate_norm = _norm_key(candidate).replace("_", " ")
        if value_norm == candidate_norm:
            return True
        if value_norm in candidate_norm or candidate_norm in value_norm:
            return True
    return False


def _unique_pairs(values: list[str]) -> list[tuple[str, str]]:
    unique = _dedupe_preserve_order(values)
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(unique):
        for right in unique[index + 1 :]:
            if left == right:
                continue
            pairs.append(tuple(sorted((left, right))))
    return _dedupe_preserve_order_tuple(pairs)


def _dedupe_preserve_order_tuple(
    values: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique
