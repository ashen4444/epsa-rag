"""Candidate evidence path search for EPSA.

This module ranks candidate reasoning paths in an EvidenceGraph. It is still not
a sufficiency module: returned paths are candidates that later EPSA components
can inspect, prune around, or use to decide whether more retrieval is needed.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from epsa_rag.epsa.schemas import EvidenceGraph, EvidenceGraphEdge, EvidencePath, QuestionAnalysis


@dataclass(frozen=True)
class _GraphIndex:
    edges_by_type: dict[str, list[EvidenceGraphEdge]]
    outgoing_by_type: dict[tuple[str, str], list[EvidenceGraphEdge]]
    sentence_to_entities: dict[str, list[str]]
    entity_to_sentences: dict[str, list[str]]
    sentence_to_relations: dict[str, list[str]]
    sentence_to_answer_types: dict[str, list[str]]
    seed_to_sentences: dict[str, list[EvidenceGraphEdge]]
    possible_answer_edges_by_sentence: dict[str, list[EvidenceGraphEdge]]
    evidence_unit_by_sentence: dict[str, str]
    evidence_score_by_sentence: dict[str, float]


class EvidencePathSearcher:
    """Search an EPSA evidence graph for ranked candidate reasoning paths."""

    def search_paths(
        self,
        evidence_graph: EvidenceGraph,
        question_analysis: QuestionAnalysis,
        max_paths: int = 10,
    ) -> list[EvidencePath]:
        """Return ranked candidate paths without making sufficiency decisions."""

        if max_paths <= 0:
            return []

        question_type = _as_text(
            getattr(question_analysis, "question_type", evidence_graph.question_type)
        ).lower()
        index = self._build_index(evidence_graph)

        if question_type == "bridge":
            paths = self._search_bridge_paths(evidence_graph, question_analysis, index)
        elif question_type == "comparison":
            paths = self._search_comparison_partial_paths(
                evidence_graph, question_analysis, index
            )
        elif question_type in {"yes_no", "yes-no", "boolean"}:
            paths = self._search_yes_no_evidence_paths(
                evidence_graph, question_analysis, index
            )
        else:
            paths = self._search_factoid_paths(evidence_graph, question_analysis, index)

        paths = self._dedupe_paths(paths)
        paths.sort(
            key=lambda path: (
                -path.score,
                len(path.evidence_unit_ids),
                path.answer_candidate or "",
                path.path_id,
            )
        )
        return paths[:max_paths]

    def _search_bridge_paths(
        self,
        graph: EvidenceGraph,
        question_analysis: QuestionAnalysis,
        index: _GraphIndex,
    ) -> list[EvidencePath]:
        expected_answer_type = _as_text(
            getattr(question_analysis, "expected_answer_type", "UNKNOWN")
        )
        required_relation_hints = _as_text_list(
            getattr(question_analysis, "required_relation_hints", [])
        )
        seed_node_ids = graph.seed_entity_node_ids
        seed_node_id_set = set(seed_node_ids)

        paths: list[EvidencePath] = []
        for seed_node_id in seed_node_ids:
            for seed_sentence_edge in index.seed_to_sentences.get(seed_node_id, []):
                first_sentence_id = seed_sentence_edge.target_id
                first_sentence_entities = index.sentence_to_entities.get(
                    first_sentence_id, []
                )

                for bridge_node_id in first_sentence_entities:
                    if bridge_node_id in seed_node_id_set:
                        continue
                    second_sentence_ids = index.entity_to_sentences.get(bridge_node_id, [])
                    for second_sentence_id in second_sentence_ids:
                        if second_sentence_id == first_sentence_id:
                            continue

                        second_sentence_entities = index.sentence_to_entities.get(
                            second_sentence_id, []
                        )
                        answer_node_ids = self._rank_answer_candidate_nodes(
                            graph=graph,
                            sentence_id=second_sentence_id,
                            candidate_entity_node_ids=second_sentence_entities,
                            excluded_node_ids=set(seed_node_ids) | {bridge_node_id},
                            expected_answer_type=expected_answer_type,
                            index=index,
                        )

                        for answer_node_id in answer_node_ids:
                            edge_ids = self._collect_bridge_edge_ids(
                                index=index,
                                seed_sentence_edge=seed_sentence_edge,
                                first_sentence_id=first_sentence_id,
                                bridge_node_id=bridge_node_id,
                                second_sentence_id=second_sentence_id,
                                answer_node_id=answer_node_id,
                            )
                            evidence_unit_ids = self._evidence_ids_for_sentences(
                                index,
                                [first_sentence_id, second_sentence_id],
                            )
                            relation_chain = self._relations_for_sentences(
                                index,
                                [first_sentence_id, second_sentence_id],
                            )
                            node_ids = [
                                seed_node_id,
                                first_sentence_id,
                                bridge_node_id,
                                second_sentence_id,
                                answer_node_id,
                            ]
                            entity_chain = self._labels(graph, [seed_node_id, bridge_node_id, answer_node_id])
                            answer_candidate = graph.nodes[answer_node_id].label
                            score, score_breakdown = self._score_path(
                                graph=graph,
                                sentence_ids=[first_sentence_id, second_sentence_id],
                                relation_chain=relation_chain,
                                expected_answer_type=expected_answer_type,
                                answer_sentence_id=second_sentence_id,
                                required_relation_hints=required_relation_hints,
                                is_bridge=True,
                                bridge_node_id=bridge_node_id,
                                answer_node_id=answer_node_id,
                                index=index,
                            )
                            paths.append(
                                EvidencePath(
                                    path_id=self._path_id("bridge", node_ids, edge_ids),
                                    question_type="bridge",
                                    node_ids=node_ids,
                                    edge_ids=edge_ids,
                                    evidence_unit_ids=evidence_unit_ids,
                                    entity_chain=entity_chain,
                                    relation_chain=relation_chain,
                                    answer_candidate=answer_candidate,
                                    answer_type=expected_answer_type,
                                    score=score,
                                    metadata={
                                        "path_kind": "bridge_candidate",
                                        "bridge_entity": graph.nodes[bridge_node_id].label,
                                        "score_breakdown": score_breakdown,
                                        "makes_sufficiency_decision": False,
                                    },
                                )
                            )
        return paths

    def _search_factoid_paths(
        self,
        graph: EvidenceGraph,
        question_analysis: QuestionAnalysis,
        index: _GraphIndex,
    ) -> list[EvidencePath]:
        expected_answer_type = _as_text(
            getattr(question_analysis, "expected_answer_type", "UNKNOWN")
        )
        required_relation_hints = _as_text_list(
            getattr(question_analysis, "required_relation_hints", [])
        )
        seed_node_ids = graph.seed_entity_node_ids
        seed_node_id_set = set(seed_node_ids)

        paths: list[EvidencePath] = []
        for seed_node_id in seed_node_ids:
            for seed_sentence_edge in index.seed_to_sentences.get(seed_node_id, []):
                sentence_id = seed_sentence_edge.target_id
                candidate_entity_ids = index.sentence_to_entities.get(sentence_id, [])
                answer_node_ids = self._rank_answer_candidate_nodes(
                    graph=graph,
                    sentence_id=sentence_id,
                    candidate_entity_node_ids=candidate_entity_ids,
                    excluded_node_ids=seed_node_id_set,
                    expected_answer_type=expected_answer_type,
                    index=index,
                )

                for answer_node_id in answer_node_ids:
                    mention_edge = self._first_edge(
                        index,
                        source_id=sentence_id,
                        target_id=answer_node_id,
                        edge_type="sentence_mentions_entity",
                    )
                    edge_ids = [seed_sentence_edge.edge_id]
                    if mention_edge is not None:
                        edge_ids.append(mention_edge.edge_id)
                    evidence_unit_ids = self._evidence_ids_for_sentences(index, [sentence_id])
                    relation_chain = self._relations_for_sentences(index, [sentence_id])
                    node_ids = [seed_node_id, sentence_id, answer_node_id]
                    score, score_breakdown = self._score_path(
                        graph=graph,
                        sentence_ids=[sentence_id],
                        relation_chain=relation_chain,
                        expected_answer_type=expected_answer_type,
                        answer_sentence_id=sentence_id,
                        required_relation_hints=required_relation_hints,
                        is_bridge=False,
                        bridge_node_id=None,
                        answer_node_id=answer_node_id,
                        index=index,
                    )
                    paths.append(
                        EvidencePath(
                            path_id=self._path_id("factoid", node_ids, edge_ids),
                            question_type="factoid",
                            node_ids=node_ids,
                            edge_ids=edge_ids,
                            evidence_unit_ids=evidence_unit_ids,
                            entity_chain=self._labels(graph, [seed_node_id, answer_node_id]),
                            relation_chain=relation_chain,
                            answer_candidate=graph.nodes[answer_node_id].label,
                            answer_type=expected_answer_type,
                            score=score,
                            metadata={
                                "path_kind": "factoid_candidate",
                                "score_breakdown": score_breakdown,
                                "makes_sufficiency_decision": False,
                            },
                        )
                    )
        return paths

    def _search_comparison_partial_paths(
        self,
        graph: EvidenceGraph,
        question_analysis: QuestionAnalysis,
        index: _GraphIndex,
    ) -> list[EvidencePath]:
        expected_answer_type = _as_text(
            getattr(question_analysis, "expected_answer_type", "UNKNOWN")
        )
        comparison_targets = _as_text_list(
            getattr(question_analysis, "comparison_targets", [])
        )
        target_node_ids = [
            node_id
            for target in comparison_targets
            for node_id, node in graph.nodes.items()
            if node.node_type == "entity" and _same_label(node.label, target)
        ]
        if not target_node_ids:
            target_node_ids = list(graph.seed_entity_node_ids)

        paths: list[EvidencePath] = []
        for target_node_id in target_node_ids:
            sentence_ids = index.entity_to_sentences.get(target_node_id, [])
            for sentence_id in sentence_ids:
                candidate_entity_ids = index.sentence_to_entities.get(sentence_id, [])
                answer_node_ids = self._rank_answer_candidate_nodes(
                    graph=graph,
                    sentence_id=sentence_id,
                    candidate_entity_node_ids=candidate_entity_ids,
                    excluded_node_ids={target_node_id},
                    expected_answer_type=expected_answer_type,
                    index=index,
                )
                if not answer_node_ids:
                    continue
                answer_node_id = answer_node_ids[0]
                mention_edge = self._first_edge(
                    index,
                    source_id=sentence_id,
                    target_id=answer_node_id,
                    edge_type="sentence_mentions_entity",
                )
                target_mention_edge = self._first_edge(
                    index,
                    source_id=sentence_id,
                    target_id=target_node_id,
                    edge_type="sentence_mentions_entity",
                )
                edge_ids = [
                    edge.edge_id
                    for edge in [target_mention_edge, mention_edge]
                    if edge is not None
                ]
                evidence_unit_ids = self._evidence_ids_for_sentences(index, [sentence_id])
                relation_chain = self._relations_for_sentences(index, [sentence_id])
                node_ids = [target_node_id, sentence_id, answer_node_id]
                score, score_breakdown = self._score_path(
                    graph=graph,
                    sentence_ids=[sentence_id],
                    relation_chain=relation_chain,
                    expected_answer_type=expected_answer_type,
                    answer_sentence_id=sentence_id,
                    required_relation_hints=_as_text_list(
                        getattr(question_analysis, "required_relation_hints", [])
                    ),
                    is_bridge=False,
                    bridge_node_id=None,
                    answer_node_id=answer_node_id,
                    index=index,
                )
                paths.append(
                    EvidencePath(
                        path_id=self._path_id("comparison_partial", node_ids, edge_ids),
                        question_type="comparison",
                        node_ids=node_ids,
                        edge_ids=edge_ids,
                        evidence_unit_ids=evidence_unit_ids,
                        entity_chain=self._labels(graph, [target_node_id, answer_node_id]),
                        relation_chain=relation_chain,
                        answer_candidate=graph.nodes[answer_node_id].label,
                        answer_type=expected_answer_type,
                        score=score,
                        metadata={
                            "path_kind": "comparison_target_partial",
                            "comparison_target": graph.nodes[target_node_id].label,
                            "score_breakdown": score_breakdown,
                            "does_not_compare_values_yet": True,
                            "makes_sufficiency_decision": False,
                        },
                    )
                )
        return paths

    def _search_yes_no_evidence_paths(
        self,
        graph: EvidenceGraph,
        question_analysis: QuestionAnalysis,
        index: _GraphIndex,
    ) -> list[EvidencePath]:
        seed_node_ids = list(graph.seed_entity_node_ids)
        if len(seed_node_ids) < 2:
            return self._search_factoid_paths(graph, question_analysis, index)

        paths: list[EvidencePath] = []
        for left_index, left_seed_id in enumerate(seed_node_ids):
            for right_seed_id in seed_node_ids[left_index + 1 :]:
                shared_sentence_ids = sorted(
                    set(index.entity_to_sentences.get(left_seed_id, []))
                    & set(index.entity_to_sentences.get(right_seed_id, []))
                )
                for sentence_id in shared_sentence_ids:
                    left_edge = self._first_edge(
                        index,
                        source_id=sentence_id,
                        target_id=left_seed_id,
                        edge_type="sentence_mentions_entity",
                    )
                    right_edge = self._first_edge(
                        index,
                        source_id=sentence_id,
                        target_id=right_seed_id,
                        edge_type="sentence_mentions_entity",
                    )
                    edge_ids = [
                        edge.edge_id for edge in [left_edge, right_edge] if edge is not None
                    ]
                    evidence_unit_ids = self._evidence_ids_for_sentences(index, [sentence_id])
                    relation_chain = self._relations_for_sentences(index, [sentence_id])
                    node_ids = [left_seed_id, sentence_id, right_seed_id]
                    sentence_score = index.evidence_score_by_sentence.get(sentence_id, 0.0)
                    paths.append(
                        EvidencePath(
                            path_id=self._path_id("yes_no", node_ids, edge_ids),
                            question_type="yes_no",
                            node_ids=node_ids,
                            edge_ids=edge_ids,
                            evidence_unit_ids=evidence_unit_ids,
                            entity_chain=self._labels(graph, [left_seed_id, right_seed_id]),
                            relation_chain=relation_chain,
                            answer_candidate=None,
                            answer_type="BOOLEAN",
                            score=round(sentence_score + 0.15, 6),
                            metadata={
                                "path_kind": "yes_no_evidence_connection",
                                "does_not_decide_yes_no": True,
                                "makes_sufficiency_decision": False,
                            },
                        )
                    )
        return paths

    def _build_index(self, graph: EvidenceGraph) -> _GraphIndex:
        edges_by_type: dict[str, list[EvidenceGraphEdge]] = defaultdict(list)
        outgoing_by_type: dict[tuple[str, str], list[EvidenceGraphEdge]] = defaultdict(list)
        sentence_to_entities: dict[str, list[str]] = defaultdict(list)
        entity_to_sentences: dict[str, list[str]] = defaultdict(list)
        sentence_to_relations: dict[str, list[str]] = defaultdict(list)
        sentence_to_answer_types: dict[str, list[str]] = defaultdict(list)
        seed_to_sentences: dict[str, list[EvidenceGraphEdge]] = defaultdict(list)
        possible_answer_edges_by_sentence: dict[str, list[EvidenceGraphEdge]] = defaultdict(list)
        evidence_unit_by_sentence: dict[str, str] = {}
        evidence_score_by_sentence: dict[str, float] = {}

        for node_id, node in graph.nodes.items():
            if node.node_type == "sentence":
                evidence_unit_id = _as_text(node.metadata.get("evidence_unit_id", ""))
                if evidence_unit_id:
                    evidence_unit_by_sentence[node_id] = evidence_unit_id
                evidence_score_by_sentence[node_id] = _safe_float(
                    node.metadata.get("final_score", 0.0)
                )

        for edge in graph.edges:
            edges_by_type[edge.edge_type].append(edge)
            outgoing_by_type[(edge.source_id, edge.edge_type)].append(edge)

            if edge.edge_type == "sentence_mentions_entity":
                _append_unique(sentence_to_entities[edge.source_id], edge.target_id)
                _append_unique(entity_to_sentences[edge.target_id], edge.source_id)
            elif edge.edge_type == "sentence_in_document_about_entity":
                _append_unique(entity_to_sentences[edge.target_id], edge.source_id)
            elif edge.edge_type == "sentence_has_relation":
                relation = edge.relation or _as_text(edge.metadata.get("relation", ""))
                if relation:
                    _append_unique(sentence_to_relations[edge.source_id], relation)
            elif edge.edge_type == "sentence_has_answer_type":
                answer_type = _as_text(edge.metadata.get("answer_type", ""))
                if answer_type:
                    _append_unique(sentence_to_answer_types[edge.source_id], answer_type)
            elif edge.edge_type == "seed_entity_to_sentence":
                seed_to_sentences[edge.source_id].append(edge)
            elif edge.edge_type == "possible_answer_candidate":
                possible_answer_edges_by_sentence[edge.source_id].append(edge)

        for mapping in [sentence_to_entities, entity_to_sentences, sentence_to_relations, sentence_to_answer_types]:
            for key in mapping:
                mapping[key] = sorted(mapping[key])
        for key in seed_to_sentences:
            seed_to_sentences[key] = sorted(seed_to_sentences[key], key=lambda edge: edge.edge_id)
        for key in possible_answer_edges_by_sentence:
            possible_answer_edges_by_sentence[key] = sorted(
                possible_answer_edges_by_sentence[key], key=lambda edge: edge.edge_id
            )

        return _GraphIndex(
            edges_by_type=dict(edges_by_type),
            outgoing_by_type=dict(outgoing_by_type),
            sentence_to_entities=dict(sentence_to_entities),
            entity_to_sentences=dict(entity_to_sentences),
            sentence_to_relations=dict(sentence_to_relations),
            sentence_to_answer_types=dict(sentence_to_answer_types),
            seed_to_sentences=dict(seed_to_sentences),
            possible_answer_edges_by_sentence=dict(possible_answer_edges_by_sentence),
            evidence_unit_by_sentence=evidence_unit_by_sentence,
            evidence_score_by_sentence=evidence_score_by_sentence,
        )

    def _rank_answer_candidate_nodes(
        self,
        *,
        graph: EvidenceGraph,
        sentence_id: str,
        candidate_entity_node_ids: list[str],
        excluded_node_ids: set[str],
        expected_answer_type: str,
        index: _GraphIndex,
    ) -> list[str]:
        candidates = [
            node_id
            for node_id in candidate_entity_node_ids
            if node_id not in excluded_node_ids and node_id in graph.nodes
        ]
        if not candidates:
            return []

        possible_answer_targets = {
            edge.target_id for edge in index.possible_answer_edges_by_sentence.get(sentence_id, [])
        }
        answer_types = index.sentence_to_answer_types.get(sentence_id, [])
        expected_type_match = any(
            _same_label(answer_type, expected_answer_type) for answer_type in answer_types
        )

        def candidate_key(node_id: str) -> tuple[float, str]:
            node = graph.nodes[node_id]
            score = 0.0
            if node_id in possible_answer_targets:
                score += 0.4
            if expected_type_match:
                score += 0.25
            label = node.label
            if _looks_like_specific_answer(label):
                score += 0.1
            if _same_label(label, expected_answer_type):
                score -= 0.2
            return (-score, label.lower())

        return sorted(candidates, key=candidate_key)

    def _collect_bridge_edge_ids(
        self,
        *,
        index: _GraphIndex,
        seed_sentence_edge: EvidenceGraphEdge,
        first_sentence_id: str,
        bridge_node_id: str,
        second_sentence_id: str,
        answer_node_id: str,
    ) -> list[str]:
        edges: list[EvidenceGraphEdge | None] = [seed_sentence_edge]
        edges.append(
            self._first_edge(
                index,
                source_id=first_sentence_id,
                target_id=bridge_node_id,
                edge_type="sentence_mentions_entity",
            )
        )
        edges.append(
            self._first_edge(
                index,
                source_id=second_sentence_id,
                target_id=bridge_node_id,
                edge_type="sentence_mentions_entity",
            )
        )
        edges.append(
            self._first_edge(
                index,
                source_id=second_sentence_id,
                target_id=answer_node_id,
                edge_type="sentence_mentions_entity",
            )
        )
        return _dedupe_preserve_order([edge.edge_id for edge in edges if edge is not None])

    def _first_edge(
        self,
        index: _GraphIndex,
        *,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> EvidenceGraphEdge | None:
        for edge in index.outgoing_by_type.get((source_id, edge_type), []):
            if edge.target_id == target_id:
                return edge
        return None

    def _score_path(
        self,
        *,
        graph: EvidenceGraph,
        sentence_ids: list[str],
        relation_chain: list[str],
        expected_answer_type: str,
        answer_sentence_id: str,
        required_relation_hints: list[str],
        is_bridge: bool,
        bridge_node_id: str | None,
        answer_node_id: str,
        index: _GraphIndex,
    ) -> tuple[float, dict[str, float]]:
        evidence_scores = [
            index.evidence_score_by_sentence.get(sentence_id, 0.0)
            for sentence_id in sentence_ids
        ]
        average_evidence_score = (
            sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
        )

        answer_types = index.sentence_to_answer_types.get(answer_sentence_id, [])
        expected_answer_type_match = 1.0 if any(
            _same_label(answer_type, expected_answer_type) for answer_type in answer_types
        ) else 0.0

        matched_relation_count = 0
        for relation in relation_chain:
            if _matches_any(relation, required_relation_hints):
                matched_relation_count += 1
        relation_match_score = min(1.0, matched_relation_count / max(1, len(required_relation_hints)))

        path_length_score = 1.0
        if is_bridge and len(set(sentence_ids)) == 2:
            path_length_score = 1.0
        elif is_bridge:
            path_length_score = 0.4

        bridge_entity_quality = 0.0
        if bridge_node_id is not None and bridge_node_id in graph.nodes:
            bridge_label = graph.nodes[bridge_node_id].label
            bridge_entity_quality = 0.2 if _looks_like_specific_answer(bridge_label) else 0.05

        answer_specificity_score = 0.15 if _looks_like_specific_answer(
            graph.nodes[answer_node_id].label
        ) else 0.0

        retrieval_quality_score = 0.0
        ranks: list[float] = []
        for sentence_id in sentence_ids:
            node = graph.nodes.get(sentence_id)
            if node is None:
                continue
            rank = node.metadata.get("retrieval_rank")
            try:
                if rank is not None:
                    ranks.append(float(rank))
            except (TypeError, ValueError):
                pass
        if ranks:
            retrieval_quality_score = 1.0 / min(ranks)

        score_breakdown = {
            "average_evidence_score": round(average_evidence_score, 6),
            "expected_answer_type_match": round(expected_answer_type_match, 6),
            "relation_match_score": round(relation_match_score, 6),
            "path_length_score": round(path_length_score, 6),
            "bridge_entity_quality": round(bridge_entity_quality, 6),
            "answer_specificity_score": round(answer_specificity_score, 6),
            "retrieval_quality_score": round(retrieval_quality_score, 6),
        }
        score = (
            average_evidence_score
            + 0.25 * expected_answer_type_match
            + 0.20 * relation_match_score
            + 0.10 * path_length_score
            + bridge_entity_quality
            + answer_specificity_score
            + 0.05 * retrieval_quality_score
        )
        return round(score, 6), score_breakdown

    def _relations_for_sentences(
        self,
        index: _GraphIndex,
        sentence_ids: list[str],
    ) -> list[str]:
        relations: list[str] = []
        for sentence_id in sentence_ids:
            for relation in index.sentence_to_relations.get(sentence_id, []):
                _append_unique(relations, relation)
        return relations

    def _evidence_ids_for_sentences(
        self,
        index: _GraphIndex,
        sentence_ids: list[str],
    ) -> list[str]:
        evidence_ids: list[str] = []
        for sentence_id in sentence_ids:
            evidence_unit_id = index.evidence_unit_by_sentence.get(sentence_id)
            if evidence_unit_id:
                _append_unique(evidence_ids, evidence_unit_id)
        return evidence_ids

    def _labels(self, graph: EvidenceGraph, node_ids: list[str]) -> list[str]:
        return [graph.nodes[node_id].label for node_id in node_ids if node_id in graph.nodes]

    def _path_id(self, prefix: str, node_ids: list[str], edge_ids: list[str]) -> str:
        raw = "|".join([prefix, *node_ids, *edge_ids])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}::{digest}"

    def _dedupe_paths(self, paths: list[EvidencePath]) -> list[EvidencePath]:
        best_by_signature: dict[tuple[Any, ...], EvidencePath] = {}
        for path in paths:
            signature = (
                path.question_type,
                tuple(path.evidence_unit_ids),
                tuple(path.entity_chain),
                path.answer_candidate,
            )
            existing = best_by_signature.get(signature)
            if existing is None or path.score > existing.score:
                best_by_signature[signature] = path
        return list(best_by_signature.values())


# Small local helpers mirror graph-builder behavior without importing private
# functions, keeping this module easy to test independently.


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
    try:
        return [_as_text(value) for value in values if _as_text(value)]
    except TypeError:
        return [_as_text(values)]


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _same_label(left: str, right: str) -> bool:
    return _norm_label(left) == _norm_label(right)


def _norm_label(value: str) -> str:
    return " ".join(_as_text(value).lower().strip().replace("_", " ").split())


def _matches_any(value: str, candidates: list[str]) -> bool:
    value_norm = _norm_label(value)
    for candidate in candidates:
        candidate_norm = _norm_label(candidate)
        if value_norm == candidate_norm:
            return True
        if value_norm in candidate_norm or candidate_norm in value_norm:
            return True
    return False


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        _append_unique(unique, value)
    return unique


def _looks_like_specific_answer(label: str) -> bool:
    text = _as_text(label).strip()
    if not text:
        return False
    generic_labels = {
        "person",
        "location",
        "date",
        "number",
        "boolean",
        "entity",
        "organization",
        "title_or_work",
        "unknown",
    }
    if text.lower() in generic_labels:
        return False
    return any(char.isalpha() or char.isdigit() for char in text)
