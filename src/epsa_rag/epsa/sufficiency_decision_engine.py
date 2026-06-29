"""Rule-based sufficiency decision engine for EPSA.

This module inspects candidate evidence paths and decides whether the retrieved
candidate evidence is complete enough for final answer generation.

Important boundary:
    This module does not retrieve documents, prune context, generate next-hop
    queries, or call an LLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from epsa_rag.epsa.schemas import (
    EvidenceGraph,
    EvidencePath,
    QuestionAnalysis,
    SufficiencyDecision,
)


class SufficiencyDecisionEngine:
    """Make deterministic EPSA sufficiency decisions from candidate paths."""

    def decide(
        self,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        evidence_paths: list[EvidencePath],
    ) -> SufficiencyDecision:
        """Return a rule-based sufficiency decision.

        Args:
            question_analysis: Output of the deterministic Question Analyzer.
            evidence_graph: EvidenceGraph built from scored sentence evidence.
            evidence_paths: Ranked or unranked candidate paths from EvidencePathSearcher.

        Returns:
            SufficiencyDecision with selected evidence provenance and an explicit
            reason/missing-evidence description.
        """

        question_type = _question_type(question_analysis, evidence_graph)
        ranked_paths = self._rank_paths(evidence_paths)

        if not ranked_paths:
            return self._insufficient(
                question_analysis=question_analysis,
                evidence_graph=evidence_graph,
                question_type=question_type,
                best_path=None,
                selected_evidence_unit_ids=[],
                missing_evidence="No candidate evidence path found.",
                decision_reason="No candidate evidence path was available for sufficiency checking.",
                rule_trace=[
                    "received_paths=0",
                    "sufficient=false",
                    "does_not_generate_next_query=true",
                ],
            )

        if question_type == "bridge":
            return self._decide_bridge(question_analysis, evidence_graph, ranked_paths)
        if question_type == "factoid":
            return self._decide_factoid(question_analysis, evidence_graph, ranked_paths)
        if question_type == "comparison":
            return self._decide_comparison(question_analysis, evidence_graph, ranked_paths)
        if question_type in {"yes_no", "yes-no", "boolean"}:
            return self._decide_yes_no(question_analysis, evidence_graph, ranked_paths)

        return self._decide_factoid(question_analysis, evidence_graph, ranked_paths)

    def _decide_bridge(
        self,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        ranked_paths: list[EvidencePath],
    ) -> SufficiencyDecision:
        expected_answer_type = _expected_answer_type(question_analysis, evidence_graph)
        required_relations = _required_relation_hints(question_analysis, evidence_graph)
        seed_labels = _seed_labels(question_analysis, evidence_graph)

        best_partial_path = ranked_paths[0]
        best_partial_trace: list[str] = []
        best_partial_missing = "No complete bridge evidence path found."

        for path in ranked_paths:
            trace = [
                f"path_id={path.path_id}",
                "question_type=bridge",
                f"path_score={path.score}",
            ]
            selected_ids = _dedupe_preserve_order(path.evidence_unit_ids)

            if not self._path_connects_seed(path, evidence_graph, seed_labels):
                trace.append("seed_connection=false")
                best_partial_trace = trace
                best_partial_missing = "Bridge path does not connect to a question seed entity."
                continue
            trace.append("seed_connection=true")

            if len(selected_ids) < 2:
                trace.append("multi_hop_evidence_units=false")
                bridge_entity = _bridge_entity(path, seed_labels)
                if bridge_entity:
                    best_partial_missing = f"Bridge path is incomplete after bridge entity {bridge_entity}."
                else:
                    best_partial_missing = "Bridge path has fewer than two useful evidence units."
                best_partial_trace = trace
                continue
            trace.append("multi_hop_evidence_units=true")

            bridge_entity = _bridge_entity(path, seed_labels)
            if not bridge_entity:
                trace.append("non_seed_bridge_entity=false")
                best_partial_missing = "No non-seed bridge entity found in the candidate path."
                best_partial_trace = trace
                continue
            trace.append(f"bridge_entity={bridge_entity}")

            if not _specific_answer(path.answer_candidate):
                trace.append("answer_candidate=false")
                best_partial_missing = f"Bridge path is incomplete after bridge entity {bridge_entity}."
                best_partial_trace = trace
                continue
            trace.append(f"answer_candidate={path.answer_candidate}")

            if not self._answer_type_compatible(path, evidence_graph, expected_answer_type):
                trace.append("answer_type_compatible=false")
                best_partial_missing = (
                    f"No answer candidate connected to expected answer type {expected_answer_type}."
                )
                best_partial_trace = trace
                continue
            trace.append("answer_type_compatible=true")

            matched_relations = _matched_relation_count(path.relation_chain, required_relations)
            minimum_relation_matches = min(len(required_relations), 2)
            if required_relations and matched_relations < minimum_relation_matches:
                trace.append(
                    f"relation_matches={matched_relations}/{len(required_relations)}"
                )
                missing_relation = _first_missing_relation(path.relation_chain, required_relations)
                best_partial_missing = (
                    f"No evidence unit supports the required relation {missing_relation}."
                    if missing_relation
                    else "Required relation evidence is incomplete."
                )
                best_partial_trace = trace
                continue
            trace.append(f"relation_matches={matched_relations}/{len(required_relations)}")

            trace.extend(["sufficient=true", "does_not_generate_next_query=true"])
            return self._sufficient(
                question_analysis=question_analysis,
                evidence_graph=evidence_graph,
                question_type="bridge",
                best_path=path,
                confidence=self._confidence(path, base=0.70),
                decision_reason="Complete bridge evidence path connects a seed entity through a bridge entity to an answer candidate.",
                rule_trace=trace,
                metadata={
                    "bridge_entity": bridge_entity,
                    "matched_relation_count": matched_relations,
                    "required_relation_count": len(required_relations),
                    "makes_next_query": False,
                },
            )

        return self._insufficient(
            question_analysis=question_analysis,
            evidence_graph=evidence_graph,
            question_type="bridge",
            best_path=best_partial_path,
            selected_evidence_unit_ids=_dedupe_preserve_order(best_partial_path.evidence_unit_ids),
            missing_evidence=best_partial_missing,
            decision_reason="No candidate bridge path satisfied all deterministic completeness rules.",
            rule_trace=[*best_partial_trace, "sufficient=false", "does_not_generate_next_query=true"],
        )

    def _decide_factoid(
        self,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        ranked_paths: list[EvidencePath],
    ) -> SufficiencyDecision:
        expected_answer_type = _expected_answer_type(question_analysis, evidence_graph)
        required_relations = _required_relation_hints(question_analysis, evidence_graph)
        seed_labels = _seed_labels(question_analysis, evidence_graph)

        best_partial_path = ranked_paths[0]
        best_partial_trace: list[str] = []
        best_partial_missing = "No complete factoid evidence path found."

        for path in ranked_paths:
            trace = [
                f"path_id={path.path_id}",
                "question_type=factoid",
                f"path_score={path.score}",
            ]
            selected_ids = _dedupe_preserve_order(path.evidence_unit_ids)

            if _is_controller_partial_path(path):
                trace.append("controller_partial_evidence_fallback=true")
                best_partial_missing = (
                    "Only a controller partial-evidence fallback path was available; "
                    "factoid sufficiency requires a real searched evidence path."
                )
                best_partial_trace = trace
                continue

            if not selected_ids:
                trace.append("supporting_evidence_unit=false")
                best_partial_missing = "No evidence unit supports the answer candidate."
                best_partial_trace = trace
                continue
            trace.append("supporting_evidence_unit=true")

            if seed_labels and not self._path_connects_seed(path, evidence_graph, seed_labels):
                trace.append("seed_connection=false")
                best_partial_missing = "Candidate path does not connect a seed entity to an answer candidate."
                best_partial_trace = trace
                continue
            trace.append("seed_connection=true")

            if not _specific_answer(path.answer_candidate):
                trace.append("answer_candidate=false")
                best_partial_missing = "No answer candidate found in the candidate path."
                best_partial_trace = trace
                continue
            trace.append(f"answer_candidate={path.answer_candidate}")

            if not self._answer_type_compatible(path, evidence_graph, expected_answer_type):
                trace.append("answer_type_compatible=false")
                best_partial_missing = (
                    f"No answer candidate connected to expected answer type {expected_answer_type}."
                )
                best_partial_trace = trace
                continue
            trace.append("answer_type_compatible=true")

            matched_relations = _matched_relation_count(path.relation_chain, required_relations)
            if required_relations and matched_relations < len(required_relations):
                trace.append(
                    f"relation_matches={matched_relations}/{len(required_relations)}"
                )
                missing_relation = _first_missing_relation(path.relation_chain, required_relations)
                best_partial_missing = (
                    f"No evidence unit supports the required relation {missing_relation}."
                    if missing_relation
                    else "Required relation evidence is incomplete."
                )
                best_partial_trace = trace
                continue
            trace.append(f"relation_matches={matched_relations}/{len(required_relations)}")

            if _generic_answer_type(expected_answer_type) and not required_relations and len(selected_ids) < 2:
                trace.append("generic_answer_type_single_evidence_unit=true")
                best_partial_missing = (
                    "Generic factoid answer type has only one evidence unit and no explicit relation evidence."
                )
                best_partial_trace = trace
                continue

            trace.extend(["sufficient=true", "does_not_generate_next_query=true"])
            return self._sufficient(
                question_analysis=question_analysis,
                evidence_graph=evidence_graph,
                question_type="factoid",
                best_path=path,
                confidence=self._confidence(path, base=0.68),
                decision_reason="Factoid path connects a seed entity to a typed answer candidate with supporting evidence.",
                rule_trace=trace,
                metadata={
                    "matched_relation_count": matched_relations,
                    "required_relation_count": len(required_relations),
                    "makes_next_query": False,
                },
            )

        return self._insufficient(
            question_analysis=question_analysis,
            evidence_graph=evidence_graph,
            question_type="factoid",
            best_path=best_partial_path,
            selected_evidence_unit_ids=_dedupe_preserve_order(best_partial_path.evidence_unit_ids),
            missing_evidence=best_partial_missing,
            decision_reason="No candidate factoid path satisfied all deterministic completeness rules.",
            rule_trace=[*best_partial_trace, "sufficient=false", "does_not_generate_next_query=true"],
        )

    def _decide_comparison(
        self,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        ranked_paths: list[EvidencePath],
    ) -> SufficiencyDecision:
        selected_ids = _dedupe_preserve_order(
            evidence_id
            for path in ranked_paths
            for evidence_id in path.evidence_unit_ids
        )
        targets = _as_text_list(getattr(question_analysis, "comparison_targets", []))
        represented_targets = _dedupe_preserve_order(
            _as_text(path.metadata.get("comparison_target"))
            for path in ranked_paths
            if _as_text(path.metadata.get("comparison_target"))
        )
        return self._insufficient(
            question_analysis=question_analysis,
            evidence_graph=evidence_graph,
            question_type="comparison",
            best_path=ranked_paths[0],
            selected_evidence_unit_ids=selected_ids,
            missing_evidence="Comparison target evidence is incomplete or requires later specialized comparison resolution.",
            decision_reason="Comparison requires later specialized comparison resolution; value comparison is not resolved in Chat 13.",
            rule_trace=[
                "question_type=comparison",
                f"comparison_targets={len(targets)}",
                f"represented_targets={len(represented_targets)}",
                "specialized_comparison_resolution=false",
                "sufficient=false",
                "does_not_generate_next_query=true",
            ],
            metadata={
                "comparison_targets": targets,
                "represented_targets": represented_targets,
                "requires_later_comparison_resolution": True,
                "makes_next_query": False,
            },
        )

    def _decide_yes_no(
        self,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        ranked_paths: list[EvidencePath],
    ) -> SufficiencyDecision:
        required_relations = _required_relation_hints(question_analysis, evidence_graph)
        seed_labels = _seed_labels(question_analysis, evidence_graph)

        best_partial_path = ranked_paths[0]
        best_partial_trace: list[str] = []
        best_partial_missing = "No connected yes/no evidence path found."

        for path in ranked_paths:
            trace = [
                f"path_id={path.path_id}",
                "question_type=yes_no",
                f"path_score={path.score}",
            ]
            selected_ids = _dedupe_preserve_order(path.evidence_unit_ids)

            if not selected_ids:
                trace.append("supporting_evidence_unit=false")
                best_partial_missing = "No evidence unit supports the yes/no claim."
                best_partial_trace = trace
                continue
            trace.append("supporting_evidence_unit=true")

            if seed_labels and not self._path_connects_seed(path, evidence_graph, seed_labels):
                trace.append("seed_connection=false")
                best_partial_missing = "Evidence path does not connect the relevant question entities."
                best_partial_trace = trace
                continue
            trace.append("seed_connection=true")

            matched_relations = _matched_relation_count(path.relation_chain, required_relations)
            if required_relations and matched_relations < len(required_relations):
                trace.append(
                    f"relation_matches={matched_relations}/{len(required_relations)}"
                )
                missing_relation = _first_missing_relation(path.relation_chain, required_relations)
                best_partial_missing = (
                    f"No evidence unit supports the required relation {missing_relation}."
                    if missing_relation
                    else "Required yes/no relation evidence is incomplete."
                )
                best_partial_trace = trace
                continue
            trace.append(f"relation_matches={matched_relations}/{len(required_relations)}")
            trace.extend(["sufficient=true", "does_not_generate_next_query=true"])
            return SufficiencyDecision(
                sufficient=True,
                confidence=self._confidence(path, base=0.62),
                question_type="yes_no",
                best_path=path,
                selected_evidence_unit_ids=_dedupe_preserve_order(path.evidence_unit_ids),
                selected_chunk_ids=self._chunk_ids_for_evidence_units(evidence_graph, path.evidence_unit_ids),
                answer_candidate=None,
                answer_type="BOOLEAN",
                missing_evidence=None,
                decision_reason="Connected yes/no evidence path with matching relation evidence found; final polarity is not generated here.",
                rule_trace=trace,
                metadata={
                    "does_not_generate_yes_no_answer": True,
                    "matched_relation_count": matched_relations,
                    "required_relation_count": len(required_relations),
                    "makes_next_query": False,
                },
            )

        return self._insufficient(
            question_analysis=question_analysis,
            evidence_graph=evidence_graph,
            question_type="yes_no",
            best_path=best_partial_path,
            selected_evidence_unit_ids=_dedupe_preserve_order(best_partial_path.evidence_unit_ids),
            missing_evidence=best_partial_missing,
            decision_reason="No yes/no candidate path satisfied conservative deterministic rules.",
            rule_trace=[*best_partial_trace, "sufficient=false", "does_not_generate_next_query=true"],
        )

    def _sufficient(
        self,
        *,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        question_type: str,
        best_path: EvidencePath,
        confidence: float,
        decision_reason: str,
        rule_trace: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> SufficiencyDecision:
        selected_ids = _dedupe_preserve_order(best_path.evidence_unit_ids)
        expected_answer_type = _expected_answer_type(question_analysis, evidence_graph)
        answer_type = best_path.answer_type or expected_answer_type
        return SufficiencyDecision(
            sufficient=True,
            confidence=confidence,
            question_type=question_type,
            best_path=best_path,
            selected_evidence_unit_ids=selected_ids,
            selected_chunk_ids=self._chunk_ids_for_evidence_units(evidence_graph, selected_ids),
            answer_candidate=best_path.answer_candidate,
            answer_type=answer_type,
            missing_evidence=None,
            decision_reason=decision_reason,
            rule_trace=rule_trace,
            metadata={
                "expected_answer_type": expected_answer_type,
                "path_score": best_path.score,
                **(metadata or {}),
            },
        )

    def _insufficient(
        self,
        *,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        question_type: str,
        best_path: EvidencePath | None,
        selected_evidence_unit_ids: list[str],
        missing_evidence: str,
        decision_reason: str,
        rule_trace: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> SufficiencyDecision:
        expected_answer_type = _expected_answer_type(question_analysis, evidence_graph)
        selected_ids = _dedupe_preserve_order(selected_evidence_unit_ids)
        return SufficiencyDecision(
            sufficient=False,
            confidence=self._confidence(best_path, base=0.25) if best_path is not None else 0.0,
            question_type=question_type,
            best_path=best_path,
            selected_evidence_unit_ids=selected_ids,
            selected_chunk_ids=self._chunk_ids_for_evidence_units(evidence_graph, selected_ids),
            answer_candidate=best_path.answer_candidate if best_path is not None else None,
            answer_type=best_path.answer_type if best_path is not None else expected_answer_type,
            missing_evidence=missing_evidence,
            decision_reason=decision_reason,
            rule_trace=rule_trace,
            metadata={
                "expected_answer_type": expected_answer_type,
                "path_score": best_path.score if best_path is not None else None,
                "makes_next_query": False,
                **(metadata or {}),
            },
        )

    @staticmethod
    def _rank_paths(paths: list[EvidencePath]) -> list[EvidencePath]:
        return sorted(
            list(paths),
            key=lambda path: (
                -_safe_float(path.score),
                len(path.evidence_unit_ids),
                path.answer_candidate or "",
                path.path_id,
            ),
        )

    @staticmethod
    def _confidence(path: EvidencePath | None, base: float) -> float:
        if path is None:
            return 0.0
        score = _safe_float(path.score)
        # Keep confidence deterministic and bounded without pretending the path
        # score is calibrated probability.
        confidence = base + min(score, 1.0) * 0.25
        return round(max(0.0, min(1.0, confidence)), 6)

    @staticmethod
    def _path_connects_seed(
        path: EvidencePath,
        graph: EvidenceGraph,
        seed_labels: list[str],
    ) -> bool:
        if not seed_labels and not graph.seed_entity_node_ids:
            return True
        if set(path.node_ids).intersection(graph.seed_entity_node_ids):
            return True
        path_entities = {_norm_label(entity) for entity in path.entity_chain}
        return any(_norm_label(seed) in path_entities for seed in seed_labels)

    @staticmethod
    def _answer_type_compatible(
        path: EvidencePath,
        graph: EvidenceGraph,
        expected_answer_type: str,
    ) -> bool:
        expected = _norm_label(expected_answer_type)
        if expected in {"", "unknown", "entity"}:
            return True

        path_answer_type = _norm_label(path.answer_type)
        if path_answer_type and path_answer_type not in {"unknown", "entity"} and path_answer_type != expected:
            return False

        sentence_ids = _sentence_node_ids_for_path(graph, path)
        for edge in graph.edges:
            if edge.edge_type != "sentence_has_answer_type":
                continue
            if edge.source_id not in sentence_ids:
                continue
            answer_type = _as_text(edge.metadata.get("answer_type")) or edge.relation or ""
            if _norm_label(answer_type) == expected:
                return True
        return False

    @staticmethod
    def _chunk_ids_for_evidence_units(
        evidence_graph: EvidenceGraph,
        evidence_unit_ids: Iterable[str],
    ) -> list[str]:
        wanted = set(evidence_unit_ids)
        chunk_ids: list[str] = []
        for node in evidence_graph.nodes.values():
            if node.node_type != "sentence":
                continue
            if node.metadata.get("evidence_unit_id") in wanted:
                chunk_id = _as_text(node.metadata.get("chunk_id"))
                if chunk_id:
                    _append_unique(chunk_ids, chunk_id)
        return chunk_ids


def _question_type(question_analysis: QuestionAnalysis, evidence_graph: EvidenceGraph) -> str:
    value = getattr(question_analysis, "question_type", None) or evidence_graph.question_type
    normalized = _norm_label(value).replace("-", "_")
    return "yes_no" if normalized in {"yes no", "yes_no", "boolean"} else normalized


def _expected_answer_type(question_analysis: QuestionAnalysis, evidence_graph: EvidenceGraph) -> str:
    value = getattr(question_analysis, "expected_answer_type", None)
    if value is None:
        value = evidence_graph.metadata.get("expected_answer_type", "UNKNOWN")
    return _as_text(value) or "UNKNOWN"


def _required_relation_hints(question_analysis: QuestionAnalysis, evidence_graph: EvidenceGraph) -> list[str]:
    values = getattr(question_analysis, "required_relation_hints", None)
    if values is None:
        values = evidence_graph.metadata.get("required_relation_hints", [])
    return _dedupe_preserve_order(_as_text_list(values))


def _seed_labels(question_analysis: QuestionAnalysis, evidence_graph: EvidenceGraph) -> list[str]:
    values = getattr(question_analysis, "seed_entities", None) or []
    seed_labels = _dedupe_preserve_order(_as_text_list(values))
    if seed_labels:
        return seed_labels
    graph_labels: list[str] = []
    for node_id in evidence_graph.seed_entity_node_ids:
        node = evidence_graph.nodes.get(node_id)
        if node is not None:
            graph_labels.append(node.label)
    return _dedupe_preserve_order(graph_labels)


def _bridge_entity(path: EvidencePath, seed_labels: list[str]) -> str | None:
    metadata_bridge = _as_text(path.metadata.get("bridge_entity"))
    if metadata_bridge and _norm_label(metadata_bridge) not in {_norm_label(seed) for seed in seed_labels}:
        return metadata_bridge

    seed_norms = {_norm_label(seed) for seed in seed_labels}
    answer_norm = _norm_label(path.answer_candidate)
    for entity in path.entity_chain:
        entity_norm = _norm_label(entity)
        if not entity_norm or entity_norm in seed_norms or entity_norm == answer_norm:
            continue
        return entity
    return None


def _matched_relation_count(path_relations: list[str], required_relations: list[str]) -> int:
    matched = 0
    for required in required_relations:
        if _relation_matches_any(required, path_relations):
            matched += 1
    return matched


def _first_missing_relation(path_relations: list[str], required_relations: list[str]) -> str | None:
    for required in required_relations:
        if not _relation_matches_any(required, path_relations):
            return required
    return None


def _relation_matches_any(required: str, path_relations: list[str]) -> bool:
    required_norm = _norm_label(required)
    for relation in path_relations:
        relation_norm = _norm_label(relation)
        if not relation_norm:
            continue
        if required_norm == relation_norm:
            return True
        if required_norm in relation_norm or relation_norm in required_norm:
            return True
    return False


def _sentence_node_ids_for_path(graph: EvidenceGraph, path: EvidencePath) -> set[str]:
    ids = {node_id for node_id in path.node_ids if graph.nodes.get(node_id, None) and graph.nodes[node_id].node_type == "sentence"}
    wanted_evidence_ids = set(path.evidence_unit_ids)
    for node_id, node in graph.nodes.items():
        if node.node_type == "sentence" and node.metadata.get("evidence_unit_id") in wanted_evidence_ids:
            ids.add(node_id)
    return ids


def _is_controller_partial_path(path: EvidencePath) -> bool:
    metadata = dict(getattr(path, "metadata", {}) or {})
    path_kind = _norm_label(metadata.get("path_kind"))
    return bool(
        metadata.get("generated_by_controller_fallback")
        or path_kind == "controller partial evidence fallback"
    )


def _generic_answer_type(expected_answer_type: str) -> bool:
    return _norm_label(expected_answer_type) in {"", "unknown", "entity"}


def _specific_answer(value: str | None) -> bool:
    text = _as_text(value)
    if not text:
        return False
    return _norm_label(text) not in {
        "person",
        "location",
        "date",
        "number",
        "boolean",
        "entity",
        "organization",
        "title or work",
        "title_or_work",
        "unknown",
    }


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


def _norm_label(value: Any) -> str:
    return " ".join(_as_text(value).lower().replace("_", " ").replace("-", " ").split())


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _as_text(value)
        key = _norm_label(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


__all__ = ["SufficiencyDecisionEngine"]
